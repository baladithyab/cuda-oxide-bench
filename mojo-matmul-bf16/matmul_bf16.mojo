# Wave 21 Task 4-7 -- mojo-matmul-bf16: hand-rolled bf16-in/f32-acc tiled matmul
#
# Strategy: bypass TensorCore wrapper's same-dtype constraint by:
#   1. Using TensorCore[bf16, bf16, m16n8k16] for load_a/load_b ONLY (these
#      don't enforce A.dtype == C.dtype — that's only mma_op + store_d).
#   2. Call raw `mma(d, a, b, c)` from std.gpu.compute.mma directly with
#      bf16 inputs and f32 accumulator.
#   3. Hand-roll C-fragment store per PTX 9.7.13.4.8 m16n8 distribution.
#
# Tile shape: BM=BN=64, BK=32, WM=WN=32, MMA=16x8x16.
# 4 warps/block (BM/WM=2 × BN/WN=2). 128 threads/block.
# Per-warp inner: WM/MMA_M × WN/MMA_N = 2 × 4 = 8 MMA calls per K-tile-pass.
# Per-tile-pass K-loop: BK/MMA_K = 32/16 = 2 outer K iterations.

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
from layout.tensor_core import TensorCore
from std.utils.index import Index


def matmul_bf16_kernel[
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
    A: LayoutTensor[DType.bfloat16, layout_a, MutAnyOrigin],
    B: LayoutTensor[DType.bfloat16, layout_b, MutAnyOrigin],
    C: LayoutTensor[DType.float32, layout_c, MutAnyOrigin],
):
    """C = A * B with bf16 inputs and f32 accumulator on Blackwell sm_120.

    Hand-rolled hybrid: TensorCore.load_a/load_b for fragment loads (uniform
    bf16 dtype, no constraint hit), raw mma() for the MMA, hand-rolled
    epilogue per PTX 9.7.13.4.8 m16n8 distribution for f32 output store.
    """
    comptime K = A.shape[1]()

    var wid = Int(warp_id())
    var warp_y = wid // (BN // WN)
    var warp_x = wid % (BN // WN)

    # Load-A and load-B helpers as bf16/bf16 TensorCore (uniform dtype, OK).
    var loader = TensorCore[DType.bfloat16, DType.bfloat16, Index(MMA_M, MMA_N, MMA_K)]()

    # Per-block C tile -> per-warp output tile.
    var C_warp_tile = C.tile[BM, BN](Int(block_idx.y), Int(block_idx.x)).tile[
        WM, WN
    ](warp_y, warp_x)

    comptime assert (
        WM % MMA_M == 0 and WN % MMA_N == 0 and BK % MMA_K == 0
    ), "Warp tile and BK must be multiples of MMA shape"

    # Shared memory for A and B tiles.
    var A_smem = LayoutTensor[
        DType.bfloat16,
        Layout.row_major(BM, BK),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var B_smem = LayoutTensor[
        DType.bfloat16,
        Layout.row_major(BK, BN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # f32 accumulator: per-warp WM/MMA_M × WN/MMA_N tiles, each holding
    # 4 f32 per lane (the m16n8 output distribution: 4 regs/lane).
    # We store as a 2D LayoutTensor in LOCAL space (registers).
    # Layout: (WM/MMA_M) × (WN/MMA_N × 4) — same convention as Wave 19.
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
        var A_dram_tile = A.tile[BM, BK](Int(block_idx.y), k_i)
        var B_dram_tile = B.tile[BK, BN](k_i, Int(block_idx.x))

        copy_dram_to_sram_async[thread_layout=Layout.row_major(4, 8)](
            A_smem.vectorize[1, 4](), A_dram_tile.vectorize[1, 4]()
        )
        copy_dram_to_sram_async[thread_layout=Layout.row_major(4, 8)](
            B_smem.vectorize[1, 4](), B_dram_tile.vectorize[1, 4]()
        )

        async_copy_wait_all()
        barrier()

        # Per-warp slice of the smem tiles.
        var A_warp_tile = A_smem.tile[WM, BK](warp_y, 0)
        var B_warp_tile = B_smem.tile[BK, WN](0, warp_x)

        # Inner MMA loop.
        comptime for mma_k in range(BK // MMA_K):
            comptime for mma_m in range(WM // MMA_M):
                comptime for mma_n in range(WN // MMA_N):
                    # 16×16 A submatrix and 16×8 B submatrix at this MMA position.
                    var A_mma_tile = A_warp_tile.tile[MMA_M, MMA_K](mma_m, mma_k)
                    var B_mma_tile = B_warp_tile.tile[MMA_K, MMA_N](mma_k, mma_n)

                    # Load fragments via the TensorCore wrapper helpers.
                    # These are bf16-typed uniform; no same-dtype error.
                    var a_lt = loader.load_a(A_mma_tile)
                    var b_lt = loader.load_b(B_mma_tile)

                    # Convert LayoutTensor frag → SIMD for raw mma() call.
                    # Per Wave 20 probe + mma_nvidia source: m16n8k16 bf16
                    # uses A=SIMD[bf16,8], B=SIMD[bf16,4], C/D=SIMD[f32,4].
                    # LayoutTensor[t,k] returns SIMD[t, element_size]; for
                    # element_size==1 we grab [0] to get the scalar lane.
                    var a_frag = SIMD[DType.bfloat16, 8](0)
                    a_frag[0] = a_lt[0, 0][0]
                    a_frag[1] = a_lt[0, 1][0]
                    a_frag[2] = a_lt[0, 2][0]
                    a_frag[3] = a_lt[0, 3][0]
                    a_frag[4] = a_lt[0, 4][0]
                    a_frag[5] = a_lt[0, 5][0]
                    a_frag[6] = a_lt[0, 6][0]
                    a_frag[7] = a_lt[0, 7][0]
                    var b_frag = SIMD[DType.bfloat16, 4](0)
                    b_frag[0] = b_lt[0, 0][0]
                    b_frag[1] = b_lt[0, 1][0]
                    b_frag[2] = b_lt[0, 2][0]
                    b_frag[3] = b_lt[0, 3][0]

                    # Pull current accumulator (4 f32 from c_reg).
                    var c_reg_tile = c_reg.tile[1, 4](mma_m, mma_n)
                    var c_frag = SIMD[DType.float32, 4](0)
                    c_frag[0] = c_reg_tile[0, 0][0]
                    c_frag[1] = c_reg_tile[0, 1][0]
                    c_frag[2] = c_reg_tile[0, 2][0]
                    c_frag[3] = c_reg_tile[0, 3][0]
                    var d_frag = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)

                    # The MMA: d = a*b + c.
                    mma(d_frag, a_frag, b_frag, c_frag)

                    # Write back accumulator.
                    c_reg_tile[0, 0] = d_frag[0]
                    c_reg_tile[0, 1] = d_frag[1]
                    c_reg_tile[0, 2] = d_frag[2]
                    c_reg_tile[0, 3] = d_frag[3]

    # ----- Hand-rolled epilogue per PTX 9.7.13.4.8 m16n8 distribution. -----
    # Each lane holds 4 f32 in d_frag positions [0,1,2,3] mapping to:
    #   row = group_id + (i >> 1) * 8     # i in {0,1} -> row=group_id; i in {2,3} -> row=group_id+8
    #   col = (tid_in_grp << 1) + (i & 1) # 2*tid_in_grp + i_parity
    # where group_id = laneid >> 2, tid_in_grp = laneid & 3.

    var lane = Int(lane_id())
    var group_id = lane >> 2
    var tid_in_grp = lane & 3

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
        comptime BK = 32
        comptime WM = 32
        comptime WN = 32
        comptime MMA_M = 16
        comptime MMA_N = 8
        comptime MMA_K = 16
        comptime NUM_WARPS = (BM // WM) * (BN // WN)
        comptime BLOCK_THREADS = NUM_WARPS * WARP_SIZE

        # ----- Problem size: full 4096^3 for canonical perf number -----
        comptime M = 4096
        comptime N = 4096
        comptime K = 4096
        comptime layout_a = Layout.row_major(M, K)
        comptime layout_b = Layout.row_major(K, N)
        comptime layout_c = Layout.row_major(M, N)

        comptime a_type = DType.bfloat16
        comptime c_type = DType.float32

        # ----- Buffers -----
        var a_dev = ctx.enqueue_create_buffer[a_type](M * K)
        var b_dev = ctx.enqueue_create_buffer[a_type](K * N)
        var c_dev = ctx.enqueue_create_buffer[c_type](M * N)
        var a_host = ctx.enqueue_create_host_buffer[a_type](M * K)
        var b_host = ctx.enqueue_create_host_buffer[a_type](K * N)
        var c_host = ctx.enqueue_create_host_buffer[c_type](M * N)
        ctx.synchronize()

        # ----- Init A, B with deterministic small-magnitude bf16. -----
        for i in range(M * K):
            a_host[i] = (Float32(((i * 2654435761) % 256)) * 0.001).cast[a_type]()
        for i in range(K * N):
            b_host[i] = (Float32((((i + 17) * 2654435761) % 256)) * 0.001).cast[a_type]()
        ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
        ctx.enqueue_copy(dst_buf=b_dev, src_buf=b_host)
        ctx.synchronize()

        var A_lt = LayoutTensor[a_type, layout_a, MutAnyOrigin](a_dev.unsafe_ptr())
        var B_lt = LayoutTensor[a_type, layout_b, MutAnyOrigin](b_dev.unsafe_ptr())
        var C_lt = LayoutTensor[c_type, layout_c, MutAnyOrigin](c_dev.unsafe_ptr())

        comptime kernel = matmul_bf16_kernel[
            layout_a, layout_b, layout_c,
            BM, BN, BK, WM, WN, MMA_M, MMA_N, MMA_K,
        ]

        # ----- Warmup -----
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

        # ----- Timed run: collect per-iter times for median (R1 noted plan
        # specified median, original code reported mean). Loop calls
        # ctx.execution_time[body](1) for each iteration to get per-iter
        # timing, then sort and pick the median. -----
        var num_iters = 10
        var iter_ms = SIMD[DType.float64, 16](0.0)  # Up to 16 iters
        for it in range(num_iters):
            var t = ctx.execution_time[body](1)
            iter_ms[it] = Float64(t) / 1e6  # ns -> ms
        ctx.synchronize()

        # Simple insertion sort of iter_ms[0:num_iters] for median.
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

        # ----- Numerical correctness: sampled check (CPU ref over full
        # 4096^3 is ~138 GFLOPs, too slow). Reviewer-driven tightening:
        #   - 1024 samples (was 256; R1+R3 found 256 has 5-80% miss rate)
        #   - i and j both use HIGH bits of LCG output (R3 found low bits
        #     give near-½ Weyl stride => 2-stripe clustering on i-axis)
        #   - tighter tolerance atol=1e-2, rtol=1e-3 (R3 proved atol=10
        #     was 5500x looser than observed error and 700x looser than
        #     Wilkinson f32 worst-case for K=4096 sum to ~67)
        var max_err: Float32 = 0.0
        var max_rel_err: Float32 = 0.0
        var fail_i = -1
        var fail_j = -1
        var fail_got: Float32 = 0.0
        var fail_ref: Float32 = 0.0
        var num_samples = 1024
        for s in range(num_samples):
            # Knuth golden-ratio multiplicative hash. Use HIGH bits for both
            # i and j (different bit windows). >>20 takes bits 20-31 for i;
            # >>11 takes bits 11-22 for j. Both windows are dominated by the
            # multiplier's golden-ratio fraction giving good Weyl spread.
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
            # Tightened tolerance: atol=1e-2 + rtol=1e-3*|ref|. For |ref|=67,
            # this allows max abs error of 1e-2 + 6.7e-2 = 7.7e-2 -- still
            # comfortably above the f32-acc differential bound (~3e-4
            # Wilkinson) but tight enough to catch a 0.1% systematic error.
            if abs_err > 1e-2 + 1e-3 * ref_abs and fail_i < 0:
                fail_i = i
                fail_j = j
                fail_got = got
                fail_ref = expected

        # ----- Report -----
        var flops_per_iter: Float64 = 2.0 * Float64(M) * Float64(N) * Float64(K)
        var median_s: Float64 = median_ms * 1e-3
        var tflops_median: Float64 = flops_per_iter / median_s / 1e12
        var tflops_best: Float64 = flops_per_iter / (min_ms * 1e-3) / 1e12

        print("[mojo-matmul-bf16] M=N=K=", M,
              " a_type=bf16 c_type=f32",
              " MMA=", MMA_M, "x", MMA_N, "x", MMA_K,
              " BM=", BM, " BN=", BN, " BK=", BK,
              " min_ms=", min_ms,
              " median_ms=", median_ms,
              " max_ms=", max_ms_iter,
              " TFLOPS_median=", tflops_median,
              " TFLOPS_best=", tflops_best)
        print("[mojo-matmul-bf16] correctness: max_abs_err=", max_err,
              " max_rel_err=", max_rel_err)
        if fail_i >= 0:
            print("[mojo-matmul-bf16] FAIL at (", fail_i, ",", fail_j,
                  "): got=", fail_got, " ref=", fail_ref)
        else:
            print("[mojo-matmul-bf16] correctness PASSED at M=N=K=", M)
