# Wave 21 Task 3 -- mojo-matmul-bf16: harness shell with stub kernel
#
# Wires up the full DeviceContext / LayoutTensor / timing harness for a
# bf16-in/f32-acc tiled matmul, but uses a STUB kernel that just zeroes
# C. This validates the host-side plumbing (bf16 init, layout tensors,
# correctness-check skeleton, timing) before we touch fragments in
# Tasks 4-7.
#
# Compile target: M=N=K=64 by default (small mode for fast iteration);
# bump to 4096 in Task 7.

from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu import block_idx, thread_idx, block_dim, WARP_SIZE
from std.gpu.host import DeviceContext
from layout.layout_tensor import Layout, LayoutTensor


def matmul_bf16_kernel_stub[
    layout_a: Layout,
    layout_b: Layout,
    layout_c: Layout,
    BM: Int,
    BN: Int,
](
    A: LayoutTensor[DType.bfloat16, layout_a, MutAnyOrigin],
    B: LayoutTensor[DType.bfloat16, layout_b, MutAnyOrigin],
    C: LayoutTensor[DType.float32, layout_c, MutAnyOrigin],
):
    """Stub kernel: zero my output tile. Real kernel arrives in Task 4."""
    var C_block_tile = C.tile[BM, BN](Int(block_idx.y), Int(block_idx.x))
    var tid = Int(thread_idx.x)
    var threads_per_block = Int(block_dim.x)
    # Cooperatively zero the BM×BN tile.
    var i = tid
    while i < BM * BN:
        var row = i // BN
        var col = i % BN
        C_block_tile[row, col] = Float32(0.0)
        i += threads_per_block


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
        return

    with DeviceContext() as ctx:
        print("GPU:", ctx.name())

        # ----- Tile shape (Wave 21 plan) -----
        comptime BM = 64
        comptime BN = 64
        comptime BK = 32
        comptime WM = 32
        comptime WN = 32
        comptime MMA_M = 16
        comptime MMA_N = 8
        comptime MMA_K = 16  # bf16 m16n8k16 path
        comptime NUM_WARPS = (BM // WM) * (BN // WN)  # = 4
        comptime BLOCK_THREADS = NUM_WARPS * WARP_SIZE  # = 128

        # ----- Problem size (small for Task 3 stub) -----
        comptime M = 64
        comptime N = 64
        comptime K = 64
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

        # ----- Init A, B with deterministic small-magnitude bf16 values -----
        # Per skill pitfall #12: explicit .cast[a_type]() required.
        for i in range(M * K):
            a_host[i] = (Float32(((i * 2654435761) % 256)) * 0.001).cast[a_type]()
        for i in range(K * N):
            b_host[i] = (Float32((((i + 17) * 2654435761) % 256)) * 0.001).cast[a_type]()
        ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
        ctx.enqueue_copy(dst_buf=b_dev, src_buf=b_host)
        ctx.synchronize()

        # ----- LayoutTensor wrappers -----
        var A_lt = LayoutTensor[a_type, layout_a, MutAnyOrigin](a_dev.unsafe_ptr())
        var B_lt = LayoutTensor[a_type, layout_b, MutAnyOrigin](b_dev.unsafe_ptr())
        var C_lt = LayoutTensor[c_type, layout_c, MutAnyOrigin](c_dev.unsafe_ptr())

        comptime kernel = matmul_bf16_kernel_stub[
            layout_a, layout_b, layout_c, BM, BN
        ]

        # ----- Warmup -----
        ctx.enqueue_function[kernel, kernel](
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

        var num_iters = 10
        var elapsed_ns = ctx.execution_time[body](num_iters)
        ctx.synchronize()

        # ----- Copy back -----
        ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
        ctx.synchronize()

        # ----- Report -----
        var flops_per_iter: Float64 = 2.0 * Float64(M) * Float64(N) * Float64(K)
        var total_flops: Float64 = flops_per_iter * Float64(num_iters)
        var elapsed_s: Float64 = Float64(elapsed_ns) * 1e-9
        var avg_ms: Float64 = Float64(elapsed_ns) / Float64(num_iters) / 1e6
        var tflops: Float64 = total_flops / elapsed_s / 1e12

        print("[mojo-matmul-bf16] M=N=K=", M,
              " a_type=bf16 c_type=f32 (stub kernel — outputs zero)",
              " MMA=", MMA_M, "x", MMA_N, "x", MMA_K,
              " BM=", BM, " BN=", BN, " BK=", BK,
              " avg_ms/iter=", avg_ms,
              " TFLOPS=", tflops)

        # Sanity: stub should output all zeros.
        var nonzero_count = 0
        for i in range(M * N):
            if c_host[i] != 0.0:
                nonzero_count += 1
        print("[mojo-matmul-bf16] stub-check: nonzero output count =", nonzero_count, "(expected 0)")
