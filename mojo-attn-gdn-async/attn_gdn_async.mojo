# Wave C3.4 -- mojo-attn-gdn-async: Gated DeltaNet decode in Mojo with
# `copy_dram_to_sram_async` for the state-tile load.
#
# Lift hypothesis: cuda-attn-gdn-async (W22.9) regressed -25% vs W1c FFMA
# baseline because `cuda::pipeline<thread_scope_thread>` introduces a smem
# round-trip + LDGSTS coordination overhead at TPB=16 that overwhelms the
# latency-hiding benefit on this single-CTA-per-bh shape. Question: does
# Mojo's `copy_dram_to_sram_async` -- which issues a CTA-cooperative bulk
# `cp.async` then `async_copy_wait_all` (no per-thread pipeline scope, no
# producer/consumer split) -- avoid the regression?
#
# Compared to the C2.3 baseline (mojo-attn-gdn at 320.5 GB/s):
#   * Baseline pass-1: per-thread `s_in[row_off + i] * alpha` synchronous
#     LDG inline with FFMA accumulation (latency interleaved into the
#     compute chain via the warp scheduler).
#   * This cell: replace the synchronous LDG with a single up-front
#     `copy_dram_to_sram_async` of the whole (D_K, BLOCK_V) state tile,
#     then `async_copy_wait_all() + barrier()`, then read back from smem
#     and run the same FFMA chain. Identical numerics, only the load
#     ordering changes.
#
# Compared to cuda-attn-gdn-async (311.8 GB/s, -25%):
#   * NO per-thread pipeline scope (Mojo's primitive is CTA-cooperative).
#   * NO ring buffer / producer-consumer split (single bulk load + drain).
#   * Same `cp.async` SASS class lowering (LDGSTS.E.BYPASS.128) is expected.
#
# If this matches or beats 320.5 -> async-pipe primitive is neutral-to-helpful
# for GDN (Mojo's collective form sidesteps the W22.9 regression).
# If it regresses (closer to 311.8 or worse) -> the regression is intrinsic
# to the smem round-trip on this shape, not the producer/consumer split.

from std.math import sqrt
from std.sys import has_accelerator
from std.gpu import (
    barrier,
    block_idx,
    thread_idx,
)
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace, async_copy_wait_all
from layout.layout_tensor import Layout, LayoutTensor, copy_dram_to_sram_async


# Compile-time tile params (mirror C2.3 baseline exactly).
comptime D_K: Int     = 256
comptime D_V: Int     = 256
comptime BLOCK_V: Int = 64           # 64 KB smem state tile
comptime VLANES: Int  = BLOCK_V // 4  # = 16 threads/CTA
comptime TPB: Int     = VLANES       # one thread per 4-col stripe


# ============================================================================
# GDN-decode kernel with async-copy state-tile load.
# Inputs/grid/block all match the C2.3 baseline.
# ============================================================================
def gdn_decode_async_kernel(
    q:    UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    k:    UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    v:    UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    a:    UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    bp:   UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    s_in: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    s_out:UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    o:    UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
):
    var bh = Int(block_idx.x)
    var bv = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var col0 = bv * BLOCK_V + tid * 4

    # ---- Shared memory ----
    var sm_q = LayoutTensor[
        DType.float32,
        Layout.row_major(D_K),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var sm_k = LayoutTensor[
        DType.float32,
        Layout.row_major(D_K),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # Single S tile: cp.async-loaded raw S_in, then scaled in-place pass-1
    # so pass-2 reads alpha-scaled values from same buffer. This gives us
    # the same 64 KiB footprint as the sync baseline (vs 128 KiB if we
    # kept a separate raw + scaled buffer). Each thread owns 4 cols of
    # this tile (col0..col0+3), so the in-place rewrite is race-free.
    var sm_S = LayoutTensor[
        DType.float32,
        Layout.row_major(D_K, BLOCK_V),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var sm_ab = LayoutTensor[
        DType.float32,
        Layout.row_major(2),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # ---- Cooperative load q, k (f16 -> f32) into smem ----
    var kk: Int = tid
    while kk < D_K:
        sm_q[kk] = q[bh * D_K + kk].cast[DType.float32]()
        sm_k[kk] = k[bh * D_K + kk].cast[DType.float32]()
        kk += TPB

    # ---- Per-block scalars: alpha, beta ----
    if tid == 0:
        sm_ab[0] = a[bh].cast[DType.float32]()
    if tid == 1 % TPB:
        sm_ab[1] = bp[bh].cast[DType.float32]()

    # ---- Per-thread v_vec[4] ----
    var v0: Float32 = v[bh * D_V + col0 + 0].cast[DType.float32]()
    var v1: Float32 = v[bh * D_V + col0 + 1].cast[DType.float32]()
    var v2: Float32 = v[bh * D_V + col0 + 2].cast[DType.float32]()
    var v3: Float32 = v[bh * D_V + col0 + 3].cast[DType.float32]()

    # ============================================================================
    # ASYNC LOAD of full S_in tile [D_K rows, BLOCK_V cols] into sm_S_raw.
    #
    # Per-CTA tile: S_in[bh, :, bv*BLOCK_V:bv*BLOCK_V+BLOCK_V].
    # Construct a LayoutTensor view of S_in for this `bh` (shape D_K x D_V),
    # then `.tile[D_K, BLOCK_V](0, bv)` to slice out our (D_K, BLOCK_V)
    # sub-block. The .tile() method preserves the row stride D_V, so
    # adjacent rows of the sub-block are NOT contiguous in gmem -- the
    # cp.async lowering must handle this with one gmem-load per row segment
    # (16 B = 4 f32). Each thread copies (D_K/TPB)=16 rows x (BLOCK_V/4)=16
    # vec-elems = 256 cp.async loads.
    # ============================================================================
    var bh_state_off: Int = bh * D_K * D_V

    var s_dram_full = LayoutTensor[
        DType.float32,
        Layout.row_major(D_K, D_V),
        MutAnyOrigin,
    ](s_in + bh_state_off)
    var s_dram_tile = s_dram_full.tile[D_K, BLOCK_V](0, bv)

    # Cooperative async copy. `vectorize[1,4]` packs 4 consecutive f32 cols
    # into a 16 B vector. Thread layout row_major(TPB, 1) = 16 row-threads.
    copy_dram_to_sram_async[thread_layout=Layout.row_major(TPB, 1)](
        sm_S.vectorize[1, 4](), s_dram_tile.vectorize[1, 4]()
    )
    async_copy_wait_all()
    barrier()

    var alpha: Float32 = sm_ab[0][0]
    var beta:  Float32 = sm_ab[1][0]

    # ============================================================================
    # Pass 1: read raw S from sm_S, scale by alpha (and write back in-place
    # for pass-2), accumulate u = sum_k k[k] * (alpha * S_in[k, ...]).
    # Per-thread cols within sm_S: tid*4 .. tid*4+3.
    # ============================================================================
    var u0: Float32 = 0.0
    var u1: Float32 = 0.0
    var u2: Float32 = 0.0
    var u3: Float32 = 0.0

    var tcol = tid * 4

    var k_iter: Int = 0
    while k_iter < D_K:
        var s0 = sm_S[k_iter, tcol + 0][0] * alpha
        var s1 = sm_S[k_iter, tcol + 1][0] * alpha
        var s2 = sm_S[k_iter, tcol + 2][0] * alpha
        var s3 = sm_S[k_iter, tcol + 3][0] * alpha

        # Write back scaled value for pass-2.
        sm_S[k_iter, tcol + 0] = s0
        sm_S[k_iter, tcol + 1] = s1
        sm_S[k_iter, tcol + 2] = s2
        sm_S[k_iter, tcol + 3] = s3

        var kkv: Float32 = sm_k[k_iter][0]
        u0 += kkv * s0
        u1 += kkv * s1
        u2 += kkv * s2
        u3 += kkv * s3

        k_iter += 1

    # ---- Residual r = v - u ----
    var r0: Float32 = v0 - u0
    var r1: Float32 = v1 - u1
    var r2: Float32 = v2 - u2
    var r3: Float32 = v3 - u3

    # ============================================================================
    # Pass 2 (UNCHANGED from baseline): S_out = S_scaled + beta*k*r;
    # accumulate o = sum_k q[k] * S_out_row.
    # ============================================================================
    var o0: Float32 = 0.0
    var o1: Float32 = 0.0
    var o2: Float32 = 0.0
    var o3: Float32 = 0.0

    k_iter = 0
    while k_iter < D_K:
        var s0: Float32 = sm_S[k_iter, tcol + 0][0]
        var s1: Float32 = sm_S[k_iter, tcol + 1][0]
        var s2: Float32 = sm_S[k_iter, tcol + 2][0]
        var s3: Float32 = sm_S[k_iter, tcol + 3][0]

        var kkv: Float32 = sm_k[k_iter][0]
        var qkv: Float32 = sm_q[k_iter][0]
        var bk:  Float32 = beta * kkv

        s0 = s0 + bk * r0
        s1 = s1 + bk * r1
        s2 = s2 + bk * r2
        s3 = s3 + bk * r3

        var row_off = bh_state_off + k_iter * D_V + col0
        s_out[row_off + 0] = s0
        s_out[row_off + 1] = s1
        s_out[row_off + 2] = s2
        s_out[row_off + 3] = s3

        o0 += qkv * s0
        o1 += qkv * s1
        o2 += qkv * s2
        o3 += qkv * s3

        k_iter += 1

    # ---- Store o (f16) ----
    o[bh * D_V + col0 + 0] = o0.cast[DType.float16]()
    o[bh * D_V + col0 + 1] = o1.cast[DType.float16]()
    o[bh * D_V + col0 + 2] = o2.cast[DType.float16]()
    o[bh * D_V + col0 + 3] = o3.cast[DType.float16]()


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
        return

    with DeviceContext() as ctx:
        print("[mojo-attn-gdn-async] GPU:", ctx.name())

        comptime B  = 1
        comptime NH = 16
        comptime BH = B * NH
        comptime QKV_ELEMS  = BH * D_K
        comptime V_ELEMS    = BH * D_V
        comptime O_ELEMS    = BH * D_V
        comptime SCAL_ELEMS = BH
        comptime S_ELEMS    = BH * D_K * D_V

        var q_dev = ctx.enqueue_create_buffer[DType.float16](QKV_ELEMS)
        var k_dev = ctx.enqueue_create_buffer[DType.float16](QKV_ELEMS)
        var v_dev = ctx.enqueue_create_buffer[DType.float16](V_ELEMS)
        var a_dev = ctx.enqueue_create_buffer[DType.float16](SCAL_ELEMS)
        var b_dev = ctx.enqueue_create_buffer[DType.float16](SCAL_ELEMS)
        var sin_dev  = ctx.enqueue_create_buffer[DType.float32](S_ELEMS)
        var sout_dev = ctx.enqueue_create_buffer[DType.float32](S_ELEMS)
        var o_dev = ctx.enqueue_create_buffer[DType.float16](O_ELEMS)

        var q_host = ctx.enqueue_create_host_buffer[DType.float16](QKV_ELEMS)
        var k_host = ctx.enqueue_create_host_buffer[DType.float16](QKV_ELEMS)
        var v_host = ctx.enqueue_create_host_buffer[DType.float16](V_ELEMS)
        var a_host = ctx.enqueue_create_host_buffer[DType.float16](SCAL_ELEMS)
        var b_host = ctx.enqueue_create_host_buffer[DType.float16](SCAL_ELEMS)
        var sin_host = ctx.enqueue_create_host_buffer[DType.float32](S_ELEMS)
        var o_host = ctx.enqueue_create_host_buffer[DType.float16](O_ELEMS)
        var sout_host = ctx.enqueue_create_host_buffer[DType.float32](S_ELEMS)
        ctx.synchronize()

        for i in range(QKV_ELEMS):
            q_host[i] = (Float32(((i * 2654435761) % 64)) * 0.01 - 0.32).cast[DType.float16]()
        for i in range(QKV_ELEMS):
            k_host[i] = (Float32((((i + 17) * 2654435761) % 64)) * 0.01 - 0.32).cast[DType.float16]()
        for i in range(V_ELEMS):
            v_host[i] = (Float32((((i + 31) * 2654435761) % 64)) * 0.01 - 0.32).cast[DType.float16]()
        for i in range(SCAL_ELEMS):
            a_host[i] = (Float32(0.85) + Float32((i * 13) % 10) * 0.01).cast[DType.float16]()
            b_host[i] = (Float32(0.05) + Float32((i * 7) % 10) * 0.01).cast[DType.float16]()
        for i in range(S_ELEMS):
            sin_host[i] = Float32((((i + 53) * 2654435761) % 128)) * 0.001 - 0.064

        ctx.enqueue_copy(dst_buf=q_dev, src_buf=q_host)
        ctx.enqueue_copy(dst_buf=k_dev, src_buf=k_host)
        ctx.enqueue_copy(dst_buf=v_dev, src_buf=v_host)
        ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
        ctx.enqueue_copy(dst_buf=b_dev, src_buf=b_host)
        ctx.enqueue_copy(dst_buf=sin_dev, src_buf=sin_host)
        ctx.synchronize()

        comptime grid_x = BH
        comptime grid_y = D_V // BLOCK_V

        # ---- Warmup ----
        ctx.enqueue_function[gdn_decode_async_kernel, gdn_decode_async_kernel](
            q_dev.unsafe_ptr(), k_dev.unsafe_ptr(), v_dev.unsafe_ptr(),
            a_dev.unsafe_ptr(), b_dev.unsafe_ptr(),
            sin_dev.unsafe_ptr(), sout_dev.unsafe_ptr(), o_dev.unsafe_ptr(),
            grid_dim=(grid_x, grid_y), block_dim=(TPB,),
        )
        ctx.enqueue_function[gdn_decode_async_kernel, gdn_decode_async_kernel](
            q_dev.unsafe_ptr(), k_dev.unsafe_ptr(), v_dev.unsafe_ptr(),
            a_dev.unsafe_ptr(), b_dev.unsafe_ptr(),
            sin_dev.unsafe_ptr(), sout_dev.unsafe_ptr(), o_dev.unsafe_ptr(),
            grid_dim=(grid_x, grid_y), block_dim=(TPB,),
        )
        ctx.synchronize()

        @parameter
        def body(ctx: DeviceContext) raises -> None:
            ctx.enqueue_function[gdn_decode_async_kernel, gdn_decode_async_kernel](
                q_dev.unsafe_ptr(), k_dev.unsafe_ptr(), v_dev.unsafe_ptr(),
                a_dev.unsafe_ptr(), b_dev.unsafe_ptr(),
                sin_dev.unsafe_ptr(), sout_dev.unsafe_ptr(), o_dev.unsafe_ptr(),
                grid_dim=(grid_x, grid_y), block_dim=(TPB,),
            )

        var num_iters = 50
        var iter_ms = SIMD[DType.float64, 64](0.0)
        for it in range(num_iters):
            var t = ctx.execution_time[body](1)
            iter_ms[it] = Float64(t) / 1e6
        ctx.synchronize()

        for ii in range(1, num_iters):
            var key = iter_ms[ii]
            var jj = ii - 1
            while jj >= 0 and iter_ms[jj] > key:
                iter_ms[jj + 1] = iter_ms[jj]
                jj -= 1
            iter_ms[jj + 1] = key
        var min_ms    = iter_ms[0]
        var median_ms = iter_ms[num_iters // 2]
        var max_ms    = iter_ms[num_iters - 1]

        var state_bytes: Float64 = 2.0 * Float64(D_K) * Float64(D_V) * 4.0
        var io_bytes:    Float64 = (2.0 * Float64(D_K) + 2.0 * Float64(D_V) + 2.0) * 2.0
        var bytes_per_iter: Float64 = Float64(BH) * (state_bytes + io_bytes)

        var min_s    = min_ms    * 1e-3
        var median_s = median_ms * 1e-3
        var gbps_best:   Float64 = bytes_per_iter / min_s    / 1e9
        var gbps_median: Float64 = bytes_per_iter / median_s / 1e9

        ctx.enqueue_copy(dst_buf=o_host, src_buf=o_dev)
        ctx.enqueue_copy(dst_buf=sout_host, src_buf=sout_dev)
        ctx.synchronize()

        var max_abs_o: Float32 = 0.0
        var max_abs_s: Float32 = 0.0
        var fail_bh: Int = -1
        var fail_j:  Int = -1
        var fail_got: Float32 = 0.0
        var fail_ref: Float32 = 0.0

        var n_samples_o = 256
        var n_samples_s = 256

        for s_idx in range(n_samples_o):
            var seed = s_idx * 2654435761
            var bh = (((seed >> 24) % BH) + BH) % BH
            var j  = (((seed >> 11) % D_V) + D_V) % D_V

            var alpha_v: Float32 = a_host[bh].cast[DType.float32]()
            var beta_v:  Float32 = b_host[bh].cast[DType.float32]()
            var v_j:     Float32 = v_host[bh * D_V + j].cast[DType.float32]()

            var u_j: Float32 = 0.0
            for t in range(D_K):
                var k_t: Float32 = k_host[bh * D_K + t].cast[DType.float32]()
                var s_t: Float32 = sin_host[bh * D_K * D_V + t * D_V + j]
                u_j += k_t * (alpha_v * s_t)
            var r_j: Float32 = v_j - u_j

            var o_j: Float32 = 0.0
            for t in range(D_K):
                var q_t: Float32 = q_host[bh * D_K + t].cast[DType.float32]()
                var k_t: Float32 = k_host[bh * D_K + t].cast[DType.float32]()
                var s_t: Float32 = sin_host[bh * D_K * D_V + t * D_V + j]
                var s_new = alpha_v * s_t + beta_v * k_t * r_j
                o_j += q_t * s_new
            var o_j_f16: Float32 = o_j.cast[DType.float16]().cast[DType.float32]()

            var got: Float32 = o_host[bh * D_V + j].cast[DType.float32]()
            var err = abs(got - o_j_f16)
            if err > max_abs_o:
                max_abs_o = err
                if err > 1e-3 and fail_bh < 0:
                    fail_bh = bh
                    fail_j = j
                    fail_got = got
                    fail_ref = o_j_f16

        for s_idx in range(n_samples_s):
            var seed = (s_idx + 9999) * 2654435761
            var bh = (((seed >> 24) % BH) + BH) % BH
            var t  = (((seed >> 13) % D_K) + D_K) % D_K
            var j  = (((seed >> 5)  % D_V) + D_V) % D_V

            var alpha_v: Float32 = a_host[bh].cast[DType.float32]()
            var beta_v:  Float32 = b_host[bh].cast[DType.float32]()
            var v_j:     Float32 = v_host[bh * D_V + j].cast[DType.float32]()
            var k_t:     Float32 = k_host[bh * D_K + t].cast[DType.float32]()
            var s_tj:    Float32 = sin_host[bh * D_K * D_V + t * D_V + j]

            var u_j: Float32 = 0.0
            for tt in range(D_K):
                var k_tt: Float32 = k_host[bh * D_K + tt].cast[DType.float32]()
                var s_tt: Float32 = sin_host[bh * D_K * D_V + tt * D_V + j]
                u_j += k_tt * (alpha_v * s_tt)
            var r_j = v_j - u_j

            var s_new_ref = alpha_v * s_tj + beta_v * k_t * r_j
            var got = sout_host[bh * D_K * D_V + t * D_V + j]
            var err = abs(got - s_new_ref)
            if err > max_abs_s:
                max_abs_s = err

        var ok_o = max_abs_o <= 1e-3
        var ok_s = max_abs_s <= 5e-3
        print("[mojo-attn-gdn-async] shape: B=", B, " H=", NH, " D_K=", D_K, " D_V=", D_V)
        print("[mojo-attn-gdn-async] tile: BLOCK_V=", BLOCK_V, " VLANES=", VLANES, " TPB=", TPB)
        print("[mojo-attn-gdn-async] bytes/iter=", bytes_per_iter / 1024.0, " KiB")
        print("[mojo-attn-gdn-async] timing: min_ms=", min_ms, " median_ms=", median_ms,
              " max_ms=", max_ms)
        print("[mojo-attn-gdn-async] GB/s: best=", gbps_best, " median=", gbps_median)
        print("[mojo-attn-gdn-async] correctness:")
        print("[mojo-attn-gdn-async]   o    max_abs_err=", max_abs_o, " (atol=1e-3) ",
              "OK" if ok_o else "FAIL")
        print("[mojo-attn-gdn-async]   Sout max_abs_err=", max_abs_s, " (atol=5e-3) ",
              "OK" if ok_s else "FAIL")
        if fail_bh >= 0:
            print("[mojo-attn-gdn-async] first o-fail: bh=", fail_bh, " j=", fail_j,
                  " got=", fail_got, " ref=", fail_ref)

        print("[mojo-attn-gdn-async] cross-cell GDN comparison:")
        print("[mojo-attn-gdn-async]   mojo-attn-gdn (sync FFMA, C2.3)  =  320.5 GB/s")
        print("[mojo-attn-gdn-async]   cuda-attn-gdn-async (W22.9)      =  311.8 GB/s (regression -25% vs W1c)")
        print("[mojo-attn-gdn-async]   mojo-attn-gdn-async (this)       = ", gbps_best, " GB/s (best)")
        var ratio_sync = gbps_best / 320.5
        var ratio_cuasy = gbps_best / 311.8
        print("[mojo-attn-gdn-async] ratios: vs mojo-sync=", ratio_sync,
              " vs cuda-async=", ratio_cuasy)
