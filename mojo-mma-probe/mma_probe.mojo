# Wave 20 W2 -- mojo-mma-probe: minimal sm_120 m16n8k16 bf16 MMA smoke test
#
# Single-warp probe: one m16n8k16 mma.sync, bf16 inputs, f32 accumulator.
# Goal: verify Mojo can emit `HMMA.16816.F32.BF16` on sm_120 via
# `from std.gpu.compute.mma import mma`. If this works, Wave 20 W2 (full
# tiled matmul) is unblocked. If it fails or falls back to scalar, we
# document the constraint and stop.
#
# Per-thread fragment shapes (per std.gpu.compute.arch.mma_nvidia source,
# the m16n8k16 BF16 dispatch lane):
#   A: SIMD[bfloat16, 8]  (each warp lane holds 8 bf16 -> 16x16 / 32 lanes = 8)
#   B: SIMD[bfloat16, 4]  (each warp lane holds 4 bf16 -> 16x8 / 32 lanes = 4)
#   C: SIMD[float32,  4]  (each warp lane holds 4 f32 -> 16x8 / 32 lanes = 4)
#   D: SIMD[float32,  4]  (output, same shape as C)
#
# We don't need ld_matrix here -- this is a pure "does mma() lower correctly
# on sm_120" probe. We hardcode trivial fragment values so we can read the
# SASS and confirm tensor-core engagement without worrying about per-lane
# data layout correctness.

from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.gpu import thread_idx
from std.gpu.compute.mma import mma


def mma_probe_kernel(
    out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
):
    # Build trivial fragments. All threads participate (warp-sync).
    # We use thread_idx.x as a deterministic per-lane seed so the compiler
    # can't constant-fold the whole thing away.
    var lane = Float32(Int(thread_idx.x))

    var a_frag = SIMD[DType.bfloat16, 8](
        (lane * 0.001).cast[DType.bfloat16](),
        (lane * 0.002).cast[DType.bfloat16](),
        (lane * 0.003).cast[DType.bfloat16](),
        (lane * 0.004).cast[DType.bfloat16](),
        (lane * 0.005).cast[DType.bfloat16](),
        (lane * 0.006).cast[DType.bfloat16](),
        (lane * 0.007).cast[DType.bfloat16](),
        (lane * 0.008).cast[DType.bfloat16](),
    )
    var b_frag = SIMD[DType.bfloat16, 4](
        (lane * 0.011).cast[DType.bfloat16](),
        (lane * 0.012).cast[DType.bfloat16](),
        (lane * 0.013).cast[DType.bfloat16](),
        (lane * 0.014).cast[DType.bfloat16](),
    )
    var c_frag = SIMD[DType.float32, 4](
        lane * 0.0001,
        lane * 0.0002,
        lane * 0.0003,
        lane * 0.0004,
    )
    var d_frag = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)

    # The MMA call itself. d = a*b + c.
    mma(d_frag, a_frag, b_frag, c_frag)

    # Write each lane's 4-element accumulator out so the compiler can't
    # delete the mma. 32 lanes * 4 floats = 128 stores per block.
    out_ptr[Int(thread_idx.x) * 4 + 0] = d_frag[0]
    out_ptr[Int(thread_idx.x) * 4 + 1] = d_frag[1]
    out_ptr[Int(thread_idx.x) * 4 + 2] = d_frag[2]
    out_ptr[Int(thread_idx.x) * 4 + 3] = d_frag[3]


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
        return

    with DeviceContext() as ctx:
        print("GPU: ", ctx.name())

        var out_dev = ctx.enqueue_create_buffer[DType.float32](128)
        var out_host = ctx.enqueue_create_host_buffer[DType.float32](128)
        ctx.synchronize()

        var out_ptr = out_dev.unsafe_ptr()

        # 32 threads = 1 warp.
        ctx.enqueue_function[mma_probe_kernel, mma_probe_kernel, _dump_sass=True](
            out_ptr,
            grid_dim=(1,),
            block_dim=(32,),
        )
        ctx.synchronize()

        ctx.enqueue_copy(dst_buf=out_host, src_buf=out_dev)
        ctx.synchronize()

        # Print first few outputs as a sanity check (not numerical correctness;
        # see ANALYSIS.md for why).
        print("[mma-probe] First 8 of 128 output floats:")
        for i in range(8):
            print("  out[", i, "] = ", out_host[i])
        print("[mma-probe] If you see HMMA.16816.F32 in stderr SASS dump,")
        print("           the bf16 MMA path engaged on sm_120. Done.")
