# Wave 18 Phase B.2 -- mojo-reduction
#
# 1 GB f32 sum-reduction on RTX 5090 sm_120 using Mojo's std.gpu.primitives.block.
# Mirrors oxide-reduction / cuda-reduction / cutile-reduction:
#   - sizes: 1M, 16M, 256M elements
#   - 1 warmup + 10 timed iters per N
#   - cudaEvent timing via ctx.execution_time
#   - 2-stage block reduction + atomicAdd into out[0]
#   - block=256 threads, grid-stride loop with grid=4096 fixed
#
# Mojo gives us block.sum() as a one-liner that internally does the
# warp-shuffle + smem two-stage pattern. Comparable to oxide's hand-written
# warp::shuffle_xor_f32 chain + smem-partials-then-warp0 reduction, and
# cuTile's `ct.sum`. The interesting question for SASS analysis is which
# memory-load primitive Mojo picks: warp-shuffle path or TMA path.

from std.math import ceildiv
from std.sys import has_accelerator

from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.primitives import block as gpu_block
from std.atomic import Atomic, Ordering

comptime float_dtype = DType.float32
comptime BLOCK: Int = 256
comptime GRID: Int = 4096


def reduce_sum_kernel(
    data: UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    result: UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    n: Int,
):
    var tid = thread_idx.x
    var bid = block_idx.x
    var bdim = block_dim.x
    var gdim = grid_dim.x

    # Grid-stride loop: each thread accumulates a local partial across
    # multiple elements of the input.
    var acc: Float32 = 0.0
    var stride = bdim * gdim
    var i = bid * bdim + tid
    while i < n:
        acc = acc + data[i]
        i = i + stride

    # Block-wide reduction. Mojo's block.sum internally uses warp-shuffle +
    # shared memory, so we don't have to write the two-stage pattern by hand
    # the way oxide-reduction does. We pass `broadcast=False` so only thread 0
    # holds the final partial -- it's the only one that will atomic-add.
    var block_partial = gpu_block.sum[block_size=BLOCK, broadcast=False](
        SIMD[float_dtype, 1](acc)
    )

    # Thread 0 of each block atomic-adds its partial into result[0].
    if tid == 0:
        _ = Atomic.fetch_add[ordering=Ordering.RELAXED](result, block_partial[0])


def run_one(ctx: DeviceContext, n: Int, label: String) raises:
    """Allocate, init, time num_iters launches, report GB/s."""

    var data_dev = ctx.enqueue_create_buffer[float_dtype](n)
    var out_dev = ctx.enqueue_create_buffer[float_dtype](1)

    var data_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var out_host = ctx.enqueue_create_host_buffer[float_dtype](1)
    ctx.synchronize()

    # Init: simple ramp -- we'll just check the answer is "reasonable" rather
    # than bit-exact, since reduction order varies.
    for i in range(n):
        data_host[i] = 1.0   # sum should be exactly N

    ctx.enqueue_copy(dst_buf=data_dev, src_buf=data_host)
    ctx.synchronize()

    var data_ptr = data_dev.unsafe_ptr()
    var out_ptr = out_dev.unsafe_ptr()
    var num_iters = 10

    # Warmup -- covers JIT + warm caches. Reset out[0] before each launch
    # because the kernel does atomicAdd. Also dump SASS on first launch for
    # offline analysis (Mojo's _dump_sass kwarg is undocumented but works).
    out_host[0] = 0.0
    ctx.enqueue_copy(dst_buf=out_dev, src_buf=out_host)
    ctx.enqueue_function[
        reduce_sum_kernel, reduce_sum_kernel,
        _dump_sass=True,
    ](
        data_ptr, out_ptr, n,
        grid_dim=GRID, block_dim=BLOCK,
    )
    ctx.synchronize()

    # Verify warmup result
    ctx.enqueue_copy(dst_buf=out_host, src_buf=out_dev)
    ctx.synchronize()
    var got = out_host[0]
    var expected = Float32(n)
    var rel_err = abs(got - expected) / expected
    print("[", label, "] warmup got=", got, " expected=", expected,
          " rel_err=", rel_err)

    # Timed iters. We reset out_dev each iter using enqueue_memset (fully
    # GPU-side, ~hundreds of ns), so the closure body is entirely on the GPU
    # stream. The memset cost is included in the timed window because there's
    # no clean way to interleave host code; cuTile/oxide handle this by
    # putting the memset OUTSIDE their event window. So our bench includes
    # memset overhead in the kernel time -- documented and accepted, since
    # the memset is microscopic vs the reduction.
    @parameter
    def body(ctx: DeviceContext) raises -> None:
        ctx.enqueue_memset(out_dev, Float32(0.0))
        ctx.enqueue_function[reduce_sum_kernel, reduce_sum_kernel](
            data_ptr, out_ptr, n,
            grid_dim=GRID, block_dim=BLOCK,
        )

    var elapsed_ns = ctx.execution_time[body](num_iters)
    ctx.synchronize()

    # Memory traffic = N * 4 bytes/elem (the 4 bytes for out[0] is negligible)
    var bytes_per_iter: Float64 = Float64(n) * 4.0
    var total_bytes: Float64 = bytes_per_iter * Float64(num_iters)
    var elapsed_s: Float64 = Float64(elapsed_ns) * 1e-9
    var avg_us: Float64 = Float64(elapsed_ns) / Float64(num_iters) / 1000.0
    var gbps: Float64 = total_bytes / elapsed_s / 1e9

    print("[", label, "] N=", n, " iters=", num_iters,
          " avg_us/iter=", avg_us, " GB/s=", gbps)


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
        return

    with DeviceContext() as ctx:
        print("GPU: ", ctx.name())

        run_one(ctx,    1_048_576, "1M")
        run_one(ctx,   16_777_216, "16M")
        run_one(ctx,  268_435_456, "256M")
