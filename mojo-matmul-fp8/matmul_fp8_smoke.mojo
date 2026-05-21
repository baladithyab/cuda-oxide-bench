# Wave 22.4 -- mojo-matmul-fp8: hand-rolled e4m3 FP8 m16n8k32 tiled matmul
#
# Strategy: hand-roll fragment loads from per-warp shared-memory tiles per
# PTX 9.7.13.4.7 m16n8k32 distribution (8-bit inputs), call raw `mma()`
# from `std.gpu.compute.mma` (Wave 22.4 probe verified the dispatcher
# supports e4m3/e4m3/f32/f32 m16n8k32 — emits QMMA.16832.F32.E4M3.E4M3
# on sm_120a), and hand-roll the f32 epilogue per PTX 9.7.13.4.8 m16n8.
#
# We do NOT use TensorCore.load_a / load_b here — those are the
# bf16 hybrid pattern (Wave 21). For FP8, the 8-bit distribution is
# different (4-elem groups across the K dim, with a +16 offset for the
# second half) so we explicitly index the smem tiles.
#
# Per-thread fragment shapes for m16n8k32 e4m3 (from `std.gpu.compute.arch.mma_nvidia`
# + skill ref + PTX ISA 9.7.13.4):
#   A: SIMD[float8_e4m3fn, 16]  (16 rows × 32 cols, 16 elems/lane)
#   B: SIMD[float8_e4m3fn,  8]  (32 rows ×  8 cols,  8 elems/lane)
#   C: SIMD[float32, 4]         (16 rows ×  8 cols,  4 elems/lane)
#   D: SIMD[float32, 4]
#
# Tile shape: small fixed M=N=K=32 single block, single warp. Goal is
# author + correctness; perf comes later (if at all). 8 MMAs per warp,
# 1 K-step.
#
# Acceptance signal: QMMA.16832.F32.E4M3.E4M3 > 0 in SASS dump
# (NOT HMMA — Blackwell consumer FP8 lowers to QMMA, verified Wave 22.4 probe).
# Tolerance: atol=1e-1 + rtol=5e-2 (FP8 e4m3 has ~3-bit mantissa).

from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu import (
    WARP_SIZE,
    barrier,
    block_idx,
    thread_idx,
    warp_id,
    lane_id,
)
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.compute.mma import mma
from layout.layout_tensor import Layout, LayoutTensor
from std.utils.index import Index


def matmul_fp8_kernel[
    layout_a: Layout,
    layout_b: Layout,
    layout_c: Layout,
    M: Int,
    N: Int,
    K: Int,
](
    A: LayoutTensor[DType.float8_e4m3fn, layout_a, MutAnyOrigin],
    B: LayoutTensor[DType.float8_e4m3fn, layout_b, MutAnyOrigin],
    C: LayoutTensor[DType.float32, layout_c, MutAnyOrigin],
):
    """C = A @ B with e4m3 inputs and f32 accumulator on Blackwell sm_120.

    Single-block, single-warp. Hand-rolled fragment loads from smem per
    PTX 9.7.13.4.7 m16n8k32 distribution; raw mma(); hand-rolled epilogue
    per PTX 9.7.13.4.8 m16n8 distribution.

    Tile geometry: M=N=K=32, MMA m16n8k32. One K-step. 8 MMAs (2 mma_m × 4 mma_n).
    """
    alias MMA_M = 16
    alias MMA_N = 8
    alias MMA_K = 32

    # Per-lane indices for both fragment loads and epilogue.
    var lane = Int(lane_id())
    var group_id = lane >> 2
    var tid_in_grp = lane & 3

    # Shared memory for A (M x K) and B (K x N).
    # M=N=K=32 here so the entire problem fits in one block's smem.
    var A_smem = LayoutTensor[
        DType.float8_e4m3fn,
        Layout.row_major(M, K),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var B_smem = LayoutTensor[
        DType.float8_e4m3fn,
        Layout.row_major(K, N),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # Cooperatively load A and B into smem. Single warp, 32 threads.
    # A has M*K = 1024 bytes; each thread loads 1024/32 = 32 elements.
    # Linear index for this thread: lane * 32 + n for n in 0..31.
    comptime for n in range(M * K // 32):
        var lin = lane * (M * K // 32) + n
        var r = lin // K
        var c = lin % K
        A_smem[r, c] = A[r, c][0]

    comptime for n in range(K * N // 32):
        var lin = lane * (K * N // 32) + n
        var r = lin // N
        var c = lin % N
        B_smem[r, c] = B[r, c][0]

    barrier()

    # f32 accumulator: per-warp WM/MMA_M × WN/MMA_N MMA tiles, each 4 f32/lane.
    # We have 2 × 4 = 8 MMAs. Store as flat array of SIMD[f32, 4].
    alias NUM_MMA_M = M // MMA_M  # 2
    alias NUM_MMA_N = N // MMA_N  # 4
    alias NUM_MMA = NUM_MMA_M * NUM_MMA_N  # 8

    # Use a LayoutTensor in LOCAL space for the c_reg accumulator.
    # Layout (NUM_MMA_M, NUM_MMA_N * 4): rows index mma_m, cols index (mma_n, lane_reg).
    var c_reg = (
        LayoutTensor[
            DType.float32,
            Layout.row_major(NUM_MMA_M, NUM_MMA_N * 4),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0.0)
    )

    # Single K-step (K=32 == MMA_K).
    # 8 MMAs.
    comptime for mma_m in range(NUM_MMA_M):
        comptime for mma_n in range(NUM_MMA_N):
            # Hand-rolled fragment loads from smem per PTX 9.7.13.4.7.
            # The (mma_m, mma_n) MMA position consumes:
            #   A_smem submatrix: rows [mma_m*16 .. mma_m*16+15], cols [0..31]
            #   B_smem submatrix: rows [0..31], cols [mma_n*8 .. mma_n*8+7]
            #
            # Per-lane A_frag (16 elems) for m16n8k32 8-bit:
            #   For sub in {0,1,2,3}, elem in {0,1,2,3}:
            #     i = sub*4 + elem
            #     row_off = (sub & 1) * 8        # 0,8,0,8 for sub=0..3
            #     col_off = (sub >> 1) * 16      # 0,0,16,16
            #     row = group_id + row_off
            #     col = tid_in_grp * 4 + col_off + elem
            var a_frag = SIMD[DType.float8_e4m3fn, 16](0)
            comptime for sub in range(4):
                comptime for elem in range(4):
                    var row_off = (sub & 1) * 8
                    var col_off = (sub >> 1) * 16
                    var row = mma_m * 16 + group_id + row_off
                    var col = tid_in_grp * 4 + col_off + elem
                    a_frag[sub * 4 + elem] = A_smem[row, col][0]

            # Per-lane B_frag (8 elems) for m16n8k32 8-bit:
            #   For sub in {0,1}, elem in {0,1,2,3}:
            #     i = sub*4 + elem
            #     row = sub*16 + tid_in_grp*4 + elem
            #     col = group_id
            var b_frag = SIMD[DType.float8_e4m3fn, 8](0)
            comptime for sub in range(2):
                comptime for elem in range(4):
                    var row = sub * 16 + tid_in_grp * 4 + elem
                    var col = mma_n * 8 + group_id
                    b_frag[sub * 4 + elem] = B_smem[row, col][0]

            # Pull current accumulator.
            var c_frag = SIMD[DType.float32, 4](0.0)
            c_frag[0] = c_reg[mma_m, mma_n * 4 + 0][0]
            c_frag[1] = c_reg[mma_m, mma_n * 4 + 1][0]
            c_frag[2] = c_reg[mma_m, mma_n * 4 + 2][0]
            c_frag[3] = c_reg[mma_m, mma_n * 4 + 3][0]
            var d_frag = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)

            # The MMA: d = a*b + c.
            mma(d_frag, a_frag, b_frag, c_frag)

            # Write back accumulator.
            c_reg[mma_m, mma_n * 4 + 0] = d_frag[0]
            c_reg[mma_m, mma_n * 4 + 1] = d_frag[1]
            c_reg[mma_m, mma_n * 4 + 2] = d_frag[2]
            c_reg[mma_m, mma_n * 4 + 3] = d_frag[3]

    # ----- Hand-rolled epilogue per PTX 9.7.13.4.8 m16n8 distribution. -----
    # row = group_id + (i >> 1) * 8     # i in {0,1} -> row=group_id; i in {2,3} -> +8
    # col = (tid_in_grp << 1) + (i & 1) # 2*tid_in_grp + i_parity
    comptime for mma_m in range(NUM_MMA_M):
        comptime for mma_n in range(NUM_MMA_N):
            comptime for i in range(4):
                var row = mma_m * 16 + group_id + (i >> 1) * 8
                var col = mma_n * 8 + (tid_in_grp << 1) + (i & 1)
                C[row, col] = c_reg[mma_m, mma_n * 4 + i][0]


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
        return

    with DeviceContext() as ctx:
        print("GPU:", ctx.name())

        # ----- Problem size: minimum where m16n8k32 fits cleanly -----
        comptime M = 32
        comptime N = 32
        comptime K = 32
        comptime layout_a = Layout.row_major(M, K)
        comptime layout_b = Layout.row_major(K, N)
        comptime layout_c = Layout.row_major(M, N)

        comptime a_type = DType.float8_e4m3fn
        comptime c_type = DType.float32

        # ----- Buffers -----
        var a_dev = ctx.enqueue_create_buffer[a_type](M * K)
        var b_dev = ctx.enqueue_create_buffer[a_type](K * N)
        var c_dev = ctx.enqueue_create_buffer[c_type](M * N)
        var a_host = ctx.enqueue_create_host_buffer[a_type](M * K)
        var b_host = ctx.enqueue_create_host_buffer[a_type](K * N)
        var c_host = ctx.enqueue_create_host_buffer[c_type](M * N)
        ctx.synchronize()

        # ----- Init A, B with deterministic small-magnitude e4m3.
        # e4m3 range: ~[-448, 448]; to hit good representability use small values.
        # Knuth golden-ratio mod 16 -> small ints, scaled by 0.0625 -> [0..15] * 0.0625
        # then cast to e4m3.
        for i in range(M * K):
            a_host[i] = (Float32(((i * 2654435761) % 16)) * 0.0625).cast[a_type]()
        for i in range(K * N):
            b_host[i] = (Float32((((i + 17) * 2654435761) % 16)) * 0.0625).cast[a_type]()
        ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
        ctx.enqueue_copy(dst_buf=b_dev, src_buf=b_host)
        ctx.synchronize()

        var A_lt = LayoutTensor[a_type, layout_a, MutAnyOrigin](a_dev.unsafe_ptr())
        var B_lt = LayoutTensor[a_type, layout_b, MutAnyOrigin](b_dev.unsafe_ptr())
        var C_lt = LayoutTensor[c_type, layout_c, MutAnyOrigin](c_dev.unsafe_ptr())

        comptime kernel = matmul_fp8_kernel[
            layout_a, layout_b, layout_c, M, N, K,
        ]

        # ----- Run with SASS dump -----
        ctx.enqueue_function[kernel, kernel, _dump_sass=True](
            A_lt, B_lt, C_lt,
            grid_dim=(1, 1),
            block_dim=(WARP_SIZE,),  # 32 threads, single warp
        )
        ctx.synchronize()

        # ----- Copy back -----
        ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
        ctx.synchronize()

        # ----- Numerical correctness: full M*N=1024 pairs (small enough). -----
        # CPU reference: A and B are quantized to e4m3 on the device side via cast.
        # Use the *cast* values from host buffers (they already round-trip through e4m3).
        var max_err: Float32 = 0.0
        var max_rel_err: Float32 = 0.0
        var fail_i = -1
        var fail_j = -1
        var fail_got: Float32 = 0.0
        var fail_ref: Float32 = 0.0
        # Tolerance per task spec: atol=1e-1 + rtol=5e-2 (FP8 e4m3 ~3-bit mantissa).
        var atol: Float32 = 1e-1
        var rtol: Float32 = 5e-2

        for i in range(M):
            for j in range(N):
                var expected: Float32 = 0.0
                for kk in range(K):
                    expected += a_host[i * K + kk].cast[DType.float32]() * b_host[kk * N + j].cast[DType.float32]()
                var got = c_host[i * N + j]
                var abs_err = abs(got - expected)
                var ref_abs = abs(expected)
                var rel_err: Float32 = 0.0
                if ref_abs > 0.0:
                    rel_err = abs_err / ref_abs
                if abs_err > max_err:
                    max_err = abs_err
                if rel_err > max_rel_err:
                    max_rel_err = rel_err
                if abs_err > atol + rtol * ref_abs and fail_i < 0:
                    fail_i = i
                    fail_j = j
                    fail_got = got
                    fail_ref = expected

        print("[mojo-matmul-fp8] M=N=K=", M,
              " a_type=e4m3 c_type=f32",
              " MMA=16x8x32",
              " max_abs_err=", max_err,
              " max_rel_err=", max_rel_err)
        if fail_i >= 0:
            print("[mojo-matmul-fp8] FAIL at (", fail_i, ",", fail_j,
                  "): got=", fail_got, " ref=", fail_ref,
                  " (atol=", atol, " rtol=", rtol, ")")
        else:
            print("[mojo-matmul-fp8] correctness PASSED at M=N=K=", M,
                  " (atol=", atol, " rtol=", rtol, ")")
        # Print a small diagonal sample for visual sanity.
        print("[mojo-matmul-fp8] C[0,0]=", c_host[0],
              " C[15,7]=", c_host[15 * N + 7],
              " C[16,8]=", c_host[16 * N + 8],
              " C[31,31]=", c_host[31 * N + 31])
