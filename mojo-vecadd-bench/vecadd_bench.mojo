# Wave 18 Phase B.1 -- mojo-vecadd-bench
#
# Benchmarks Mojo's vecadd on RTX 5090 sm_120 across N in {1M, 16M, 64M, 256M}.
# Uses ctx.execution_time for cudaEvent-based GPU timing (1 warmup + 10 iters,
# total wall-clock GB/s reported).
#
# Algorithm: c[i] = a[i] + b[i], 3 buffers x 4 bytes = 12 bytes/elem at fp32.
# Memory-bound regime (HBM peak ~1750 GB/s on RTX 5090).

from std.math import ceildiv
from std.sys import has_accelerator

from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

comptime float_dtype = DType.float32
comptime block_size: Int = 256


# Vec-add kernel using raw UnsafePointer with MutAnyOrigin (the wildcard
# mutable origin -- works fine for arrays we own and pass for the duration of
# a kernel launch). Output `c` must be mutable; inputs can be immutable but
# we use MutAnyOrigin uniformly for simplicity.
def vector_addition(
    a: UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    c: UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    n: Int,
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid < n:
        c[tid] = a[tid] + b[tid]


def run_one(ctx: DeviceContext, n: Int, label: String) raises:
    """Allocate, init, time num_iters launches, report GB/s."""

    var a_dev = ctx.enqueue_create_buffer[float_dtype](n)
    var b_dev = ctx.enqueue_create_buffer[float_dtype](n)
    var c_dev = ctx.enqueue_create_buffer[float_dtype](n)

    var a_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var b_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    ctx.synchronize()

    for i in range(n):
        a_host[i] = Float32(i)
        b_host[i] = Float32(2 * i)
    ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
    ctx.enqueue_copy(dst_buf=b_dev, src_buf=b_host)
    ctx.synchronize()

    var num_blocks = ceildiv(n, block_size)
    var num_iters = 10

    var a_ptr = a_dev.unsafe_ptr()
    var b_ptr = b_dev.unsafe_ptr()
    var c_ptr = c_dev.unsafe_ptr()

    # Warmup -- covers JIT + warm caches
    ctx.enqueue_function[vector_addition, vector_addition](
        a_ptr, b_ptr, c_ptr, n,
        grid_dim=num_blocks, block_dim=block_size,
    )
    ctx.synchronize()

    # @parameter capturing closure -- the canonical Mojo pattern for higher-
    # order functions like execution_time. Captures pointers + n by reference.
    @parameter
    def body(ctx: DeviceContext) raises -> None:
        ctx.enqueue_function[vector_addition, vector_addition](
            a_ptr, b_ptr, c_ptr, n,
            grid_dim=num_blocks, block_dim=block_size,
        )

    var elapsed_ns = ctx.execution_time[body](num_iters)
    ctx.synchronize()

    var bytes_per_iter: Float64 = Float64(n) * 12.0
    var total_bytes: Float64 = bytes_per_iter * Float64(num_iters)
    var elapsed_s: Float64 = Float64(elapsed_ns) * 1e-9
    var avg_us: Float64 = Float64(elapsed_ns) / Float64(num_iters) / 1000.0
    var gbps: Float64 = total_bytes / elapsed_s / 1e9

    print("[", label, "] N=", n,
          " iters=", num_iters,
          " avg_us/iter=", avg_us,
          " GB/s=", gbps)


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
        return

    with DeviceContext() as ctx:
        print("GPU: ", ctx.name())

        run_one(ctx,    1_048_576, "1M")
        run_one(ctx,   16_777_216, "16M")
        run_one(ctx,   67_108_864, "64M")
        run_one(ctx,  268_435_456, "256M")
