# DOC: mojo/docs/manual/gpu/intro-tutorial.mdx
# Adapted from https://raw.githubusercontent.com/modular/modular/mojo/v1.0.0b1/mojo/examples/gpu-intro/vector_addition.mojo
# Wave 18 Phase A — minimal smoke test.
# Goal: confirm Mojo compiles a GPU kernel and runs it on RTX 5090 sm_120.

from std.math import ceildiv
from std.sys import has_accelerator

from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from layout import TileTensor, row_major

# Vector data type and size
comptime float_dtype = DType.float32
comptime vector_size = 1024
comptime layout = row_major[vector_size]()

comptime block_size = 256
comptime num_blocks = ceildiv(vector_size, block_size)


def vector_addition(
    lhs_tensor: TileTensor[float_dtype, type_of(layout), MutAnyOrigin],
    rhs_tensor: TileTensor[float_dtype, type_of(layout), MutAnyOrigin],
    out_tensor: TileTensor[float_dtype, type_of(layout), MutAnyOrigin],
):
    """Element-wise sum of two vectors on the GPU."""
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid < vector_size:
        out_tensor[tid] = lhs_tensor[tid] + rhs_tensor[tid]


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
    else:
        ctx = DeviceContext()
        print("GPU detected: ", ctx.name())

        # Host-side init
        lhs_host = ctx.enqueue_create_host_buffer[float_dtype](vector_size)
        rhs_host = ctx.enqueue_create_host_buffer[float_dtype](vector_size)
        ctx.synchronize()

        for i in range(vector_size):
            lhs_host[i] = Float32(i)
            rhs_host[i] = Float32(Float64(i) * 0.5)

        # Device-side buffers
        lhs_dev = ctx.enqueue_create_buffer[float_dtype](vector_size)
        rhs_dev = ctx.enqueue_create_buffer[float_dtype](vector_size)
        out_dev = ctx.enqueue_create_buffer[float_dtype](vector_size)

        ctx.enqueue_copy(dst_buf=lhs_dev, src_buf=lhs_host)
        ctx.enqueue_copy(dst_buf=rhs_dev, src_buf=rhs_host)

        lhs_tensor = TileTensor(lhs_dev, layout)
        rhs_tensor = TileTensor(rhs_dev, layout)
        out_tensor = TileTensor(out_dev, layout)

        ctx.enqueue_function[vector_addition, vector_addition](
            lhs_tensor,
            rhs_tensor,
            out_tensor,
            grid_dim=num_blocks,
            block_dim=block_size,
        )

        out_host = ctx.enqueue_create_host_buffer[float_dtype](vector_size)
        ctx.enqueue_copy(dst_buf=out_host, src_buf=out_dev)
        ctx.synchronize()

        # Correctness verification (replaces just printing)
        var max_abs_err: Float32 = 0.0
        var first_err_idx: Int = -1
        for i in range(vector_size):
            var expected: Float32 = Float32(i) + Float32(Float64(i) * 0.5)
            var got: Float32 = out_host[i]
            var diff: Float32 = abs(got - expected)
            if diff > max_abs_err:
                max_abs_err = diff
                if first_err_idx == -1 and diff > 0.0:
                    first_err_idx = i

        print("N =", vector_size)
        print("max_abs_err =", max_abs_err)
        print("first nonzero err idx =", first_err_idx)
        print("out[0]   =", out_host[0], "  (expected 0.0)")
        print("out[1]   =", out_host[1], "  (expected 1.5)")
        print("out[100] =", out_host[100], "  (expected 150.0)")
        print("out[N-1] =", out_host[vector_size - 1], "  (expected", Float32(vector_size - 1) * 1.5, ")")

        if max_abs_err == 0.0:
            print("PASS: vecadd numerically exact")
        elif max_abs_err < 1e-5:
            print("PASS: vecadd within fp32 tolerance")
        else:
            print("FAIL: max_abs_err exceeds tolerance")
