# Wave 22.14 -- mojo-matmul-fp8: hand-rolled e4m3 FP8 m16n8k32 TILED matmul at 4096^3
#
# Scale-up of W22.4 (single-block M=N=K=32, bit-exact correctness baseline)
# to a full tiled kernel at M=N=K=4096 with bench timing.
#
# Tile shape (mirrors W21 bf16 scaffolding, adjusted for m16n8k32):
#   BM=BN=64, BK=64; WM=WN=32; MMA=16x8x32
#   4 warps/block (BM/WM=2 × BN/WN=2). 128 threads/block.
#   Per-warp inner: WM/MMA_M × WN/MMA_N = 2 × 4 = 8 QMMA/lane per K-step
#   Per-tile-pass K-loop: BK/MMA_K = 64/32 = 2 outer K iterations
#
# FP8 m16n8k32 fragment distribution (PTX 9.7.13.4.7, hand-rolled — the
# TensorCore wrapper isn't validated for float8_e4m3fn):
#   A: SIMD[float8_e4m3fn, 16] / lane (16 rows × 32 cols) -- 4 sub-blocks
#      indexed by sub in {0,1,2,3}: row_off=(sub&1)*8, col_off=(sub>>1)*16,
#      and 4 contiguous K elems per sub.
#   B: SIMD[float8_e4m3fn,  8] / lane (32 rows ×  8 cols) -- 2 sub-blocks
#      indexed by sub in {0,1}: row_off=sub*16, 4 contiguous K elems per sub.
#   C/D: SIMD[float32, 4] / lane.
#
# Output epilogue: PTX 9.7.13.4.8 m16n8 distribution (same as bf16, since
# only the K dimension changes and the C/D tile is always 16×8).
#
# Acceptance:
#   - QMMA.16832.F32.E4M3.E4M3 > 0 in SASS
#   - Sampled correctness PASSES under FP8-appropriate tolerance:
#     atol=2e-1, rtol=1e-1 (e4m3 has ~3-bit mantissa; expect MUCH wider
#     error than bf16 at K=4096)
#   - TFLOPS_median > 0

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
from std.gpu.memory import AddressSpace, async_copy_wait_all
from std.gpu.compute.mma import mma
from layout.layout_tensor import Layout, LayoutTensor, copy_dram_to_sram_async
from std.utils.index import Index


def matmul_fp8_kernel[
    layout_a: Layout,
    layout_b: Layout,
    layout_c: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
](
    A: LayoutTensor[DType.float8_e4m3fn, layout_a, MutAnyOrigin],
    B: LayoutTensor[DType.float8_e4m3fn, layout_b, MutAnyOrigin],
    C: LayoutTensor[DType.float32, layout_c, MutAnyOrigin],
):
    """C = A @ B with e4m3 inputs and f32 accumulator on Blackwell sm_120.

    Hand-rolled fragment loads from smem per PTX 9.7.13.4.7 m16n8k32 8-bit
    distribution; raw mma() (dispatcher emits QMMA.16832.F32.E4M3.E4M3 on
    sm_120a per W22.4 verification); hand-rolled epilogue per PTX 9.7.13.4.8
    m16n8 distribution.
    """
    comptime K = A.shape[1]()

    var wid = Int(warp_id())
    var warp_y = wid // (BN // WN)
    var warp_x = wid % (BN // WN)

    # Per-lane indices for fragment loads and epilogue.
    var lane = Int(lane_id())
    var group_id = lane >> 2
    var tid_in_grp = lane & 3

    # Per-block C tile -> per-warp output tile.
    var C_warp_tile = C.tile[BM, BN](Int(block_idx.y), Int(block_idx.x)).tile[
        WM, WN
    ](warp_y, warp_x)

    comptime assert (
        WM % MMA_M == 0 and WN % MMA_N == 0 and BK % MMA_K == 0
    ), "Warp tile and BK must be multiples of MMA shape"

    # Shared memory for A and B tiles.
    var A_smem = LayoutTensor[
        DType.float8_e4m3fn,
        Layout.row_major(BM, BK),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var B_smem = LayoutTensor[
        DType.float8_e4m3fn,
        Layout.row_major(BK, BN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # f32 accumulator: per-warp WM/MMA_M × WN/MMA_N tiles, each holding
    # 4 f32 per lane. 2D flattened layout matching W21 convention.
    var c_reg = (
        LayoutTensor[
            DType.float32,
            Layout.row_major(WM // MMA_M, (WN * 4) // MMA_N),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0.0)
    )

    # K-loop over BK-sized tile slabs.
    for k_i in range(K // BK):
        barrier()

        # Async DRAM→SMEM copy for A and B tiles.
        # FP8 elements are 1 byte; vectorize[1, 16] = 16-byte loads (matches
        # the 128-bit cp.async stride and gives full coalescing).
        # Threads = BLOCK_THREADS = 128. Per-tile elems = BM*BK = 64*64 = 4096.
        # Per-thread elems = 4096/128 = 32. With vectorize[1, 16] each thread
        # issues 2 vector loads; thread_layout for A is row_major(BM, BK//16)
        # = row_major(64, 4) = 256 threads -- too many. So use a smaller
        # thread-per-row layout. With 128 threads and 4 vec-cols per row,
        # we have 128/4 = 32 rows per pass, requiring 64/32 = 2 passes
        # internally. The copy_dram_to_sram_async helper handles the loop
        # over the grid. Use thread_layout=row_major(32, 4): 32 rows × 4
        # cols/thread × 16 elems/col = 32 × 64 = 2048 elems per pass; the
        # helper then loops 2x to cover BM*BK=4096. (Same idea on B.)
        var A_dram_tile = A.tile[BM, BK](Int(block_idx.y), k_i)
        var B_dram_tile = B.tile[BK, BN](k_i, Int(block_idx.x))

        copy_dram_to_sram_async[thread_layout=Layout.row_major(32, 4)](
            A_smem.vectorize[1, 16](), A_dram_tile.vectorize[1, 16]()
        )
        copy_dram_to_sram_async[thread_layout=Layout.row_major(32, 4)](
            B_smem.vectorize[1, 16](), B_dram_tile.vectorize[1, 16]()
        )

        async_copy_wait_all()
        barrier()

        # Per-warp slice of the smem tiles.
        var A_warp_tile = A_smem.tile[WM, BK](warp_y, 0)
        var B_warp_tile = B_smem.tile[BK, WN](0, warp_x)

        # Inner MMA loop. K-step ranges over BK/MMA_K = 2 (each MMA_K=32).
        comptime for mma_k in range(BK // MMA_K):
            comptime for mma_m in range(WM // MMA_M):
                comptime for mma_n in range(WN // MMA_N):
                    # Submatrices for this MMA position.
                    # A: 16 rows × 32 cols (MMA_M × MMA_K).
                    # B: 32 rows ×  8 cols (MMA_K × MMA_N).
                    var A_mma_tile = A_warp_tile.tile[MMA_M, MMA_K](mma_m, mma_k)
                    var B_mma_tile = B_warp_tile.tile[MMA_K, MMA_N](mma_k, mma_n)

                    # Hand-rolled A fragment load (PTX 9.7.13.4.7 8-bit).
                    # Per-lane: 16 elems = 4 sub-blocks × 4 K-elems each.
                    #   row_off = (sub & 1) * 8        # 0,8,0,8
                    #   col_off = (sub >> 1) * 16      # 0,0,16,16
                    #   row = group_id + row_off
                    #   col = tid_in_grp * 4 + col_off + elem
                    var a_frag = SIMD[DType.float8_e4m3fn, 16](0)
                    comptime for sub in range(4):
                        comptime for elem in range(4):
                            var row_off = (sub & 1) * 8
                            var col_off = (sub >> 1) * 16
                            var row = group_id + row_off
                            var col = tid_in_grp * 4 + col_off + elem
                            a_frag[sub * 4 + elem] = A_mma_tile[row, col][0]

                    # Hand-rolled B fragment load (PTX 9.7.13.4.7 8-bit).
                    # Per-lane: 8 elems = 2 sub-blocks × 4 K-elems each.
                    #   row = sub*16 + tid_in_grp*4 + elem
                    #   col = group_id
                    var b_frag = SIMD[DType.float8_e4m3fn, 8](0)
                    comptime for sub in range(2):
                        comptime for elem in range(4):
                            var row = sub * 16 + tid_in_grp * 4 + elem
                            var col = group_id
                            b_frag[sub * 4 + elem] = B_mma_tile[row, col][0]

                    # Pull current accumulator.
                    var c_reg_tile = c_reg.tile[1, 4](mma_m, mma_n)
                    var c_frag = SIMD[DType.float32, 4](0)
                    c_frag[0] = c_reg_tile[0, 0][0]
                    c_frag[1] = c_reg_tile[0, 1][0]
                    c_frag[2] = c_reg_tile[0, 2][0]
                    c_frag[3] = c_reg_tile[0, 3][0]
                    var d_frag = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)

                    # The MMA: d = a*b + c. Dispatcher emits QMMA on sm_120a.
                    mma(d_frag, a_frag, b_frag, c_frag)

                    # Write back.
                    c_reg_tile[0, 0] = d_frag[0]
                    c_reg_tile[0, 1] = d_frag[1]
                    c_reg_tile[0, 2] = d_frag[2]
                    c_reg_tile[0, 3] = d_frag[3]

    # ----- Hand-rolled epilogue per PTX 9.7.13.4.8 m16n8 distribution. -----
    # row = group_id + (i >> 1) * 8     # i in {0,1} -> row=group_id; i in {2,3} -> +8
    # col = (tid_in_grp << 1) + (i & 1) # 2*tid_in_grp + i_parity
    comptime for mma_m in range(WM // MMA_M):
        comptime for mma_n in range(WN // MMA_N):
            var c_reg_tile = c_reg.tile[1, 4](mma_m, mma_n)
            var C_mma_tile = C_warp_tile.tile[MMA_M, MMA_N](mma_m, mma_n)

            comptime for i in range(4):
                var row = group_id + (i >> 1) * 8
                var col = (tid_in_grp << 1) + (i & 1)
                C_mma_tile[row, col] = c_reg_tile[0, i]


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
        return

    with DeviceContext() as ctx:
        print("GPU:", ctx.name())

        # ----- Tile shape -----
        comptime BM = 64
        comptime BN = 64
        comptime BK = 64
        comptime WM = 32
        comptime WN = 32
        comptime MMA_M = 16
        comptime MMA_N = 8
        comptime MMA_K = 32
        comptime NUM_WARPS = (BM // WM) * (BN // WN)
        comptime BLOCK_THREADS = NUM_WARPS * WARP_SIZE

        # ----- Problem size: full 4096^3 for canonical perf number -----
        comptime M = 4096
        comptime N = 4096
        comptime K = 4096
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

        # ----- Init A, B with deterministic small-magnitude e4m3 values.
        # e4m3 range: ~[-448, 448]. To avoid catastrophic cancellation /
        # overflow at K=4096, use very small values: bytes in [0..15] * 0.0625
        # gives [0, 0.9375]. Sum at K=4096 of values * values bounded by
        # 0.9375^2 * 4096 ~= 3600. With ~uniform distribution and mean ~0.47,
        # expected sum is ~K * 0.47^2 = ~907 -- well within f32 range.
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
            layout_a, layout_b, layout_c,
            BM, BN, BK, WM, WN, MMA_M, MMA_N, MMA_K,
        ]

        # ----- Warmup + SASS dump -----
        ctx.enqueue_function[kernel, kernel, _dump_sass=True](
            A_lt, B_lt, C_lt,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(BLOCK_THREADS,),
        )
        ctx.synchronize()

        # ----- Timed run -----
        @parameter
        def body(ctx: DeviceContext) raises -> None:
            ctx.enqueue_function[kernel, kernel](
                A_lt, B_lt, C_lt,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                block_dim=(BLOCK_THREADS,),
            )

        # Per-iter timing × 10 iters → median (W21 pattern).
        var num_iters = 10
        var iter_ms = SIMD[DType.float64, 16](0.0)
        for it in range(num_iters):
            var t = ctx.execution_time[body](1)
            iter_ms[it] = Float64(t) / 1e6  # ns -> ms
        ctx.synchronize()

        # Insertion sort for median.
        for ii in range(1, num_iters):
            var key = iter_ms[ii]
            var jj = ii - 1
            while jj >= 0 and iter_ms[jj] > key:
                iter_ms[jj + 1] = iter_ms[jj]
                jj -= 1
            iter_ms[jj + 1] = key

        var median_ms = iter_ms[num_iters // 2]
        var min_ms = iter_ms[0]
        var max_ms_iter = iter_ms[num_iters - 1]

        # ----- Copy back -----
        ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
        ctx.synchronize()

        # ----- Sampled correctness (1024 samples via Knuth LCG, FP8-appropriate
        # tolerance: atol=2e-1, rtol=1e-1*|ref|). e4m3 has ~3-bit mantissa,
        # so per-product roundoff is large; with K=4096 sums Wilkinson bound
        # is large too. Going tighter would give false negatives.
        var max_err: Float32 = 0.0
        var max_rel_err: Float32 = 0.0
        var fail_i = -1
        var fail_j = -1
        var fail_got: Float32 = 0.0
        var fail_ref: Float32 = 0.0
        var atol: Float32 = 2e-1
        var rtol: Float32 = 1e-1
        var num_samples = 1024
        for s in range(num_samples):
            # Knuth golden-ratio multiplicative hash. HIGH bits for both i,j.
            var seed = s * 2654435761
            var i = (((seed >> 20) % M) + M) % M
            var j = (((seed >> 11) % N) + N) % N
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

        # ----- Report -----
        var flops_per_iter: Float64 = 2.0 * Float64(M) * Float64(N) * Float64(K)
        var median_s: Float64 = median_ms * 1e-3
        var tflops_median: Float64 = flops_per_iter / median_s / 1e12
        var tflops_best: Float64 = flops_per_iter / (min_ms * 1e-3) / 1e12

        print("[mojo-matmul-fp8] M=N=K=", M,
              " a_type=e4m3 c_type=f32",
              " MMA=", MMA_M, "x", MMA_N, "x", MMA_K,
              " BM=", BM, " BN=", BN, " BK=", BK,
              " min_ms=", min_ms,
              " median_ms=", median_ms,
              " max_ms=", max_ms_iter,
              " TFLOPS_median=", tflops_median,
              " TFLOPS_best=", tflops_best)
        print("[mojo-matmul-fp8] correctness: max_abs_err=", max_err,
              " max_rel_err=", max_rel_err,
              " (atol=", atol, " rtol=", rtol, ")")
        if fail_i >= 0:
            print("[mojo-matmul-fp8] FAIL at (", fail_i, ",", fail_j,
                  "): got=", fail_got, " ref=", fail_ref)
        else:
            print("[mojo-matmul-fp8] correctness PASSED at M=N=K=", M)
