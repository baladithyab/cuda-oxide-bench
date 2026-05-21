# Wave C2.3 -- mojo-attn-gdn: Gated DeltaNet single-timestep decode in Mojo.
#
# 5th frontend port of GDN. FFMA-class baseline (no TMA — Mojo 1.0.0b1 lacks
# TMA primitives per W22.1 BLOCKED.md). Mirrors the W1c CUDA C++ FFMA-LDG.E.128
# pattern (cuda-attn-gdn/attn_gdn.cu), NOT the W22.10/13 TMA path.
#
# Algorithm (per (batch b, head h, bv-block) tile, single timestep S=1):
#     S_scaled  = α · S_in                         (D_K, BLOCK_V) f32
#     u         = k^T · S_scaled                   (BLOCK_V,)     f32
#     residual  = v - u                            (BLOCK_V,)     f32
#     S_out     = S_scaled + β · k ⊗ residual      (D_K, BLOCK_V) f32
#     o         = q^T · S_out                      (BLOCK_V,)     f32  (cast → f16)
#
# Per-block layout (matches W1c with smaller BLOCK_V to fit static smem):
#   gridDim  = (B*H, D_V / BLOCK_V)
#   blockDim = (BLOCK_V / 4)   -- each thread owns 4 contiguous d_v columns
#                              -- (vectorize-of-4 stripe).
#
# Smem (static, must fit Blackwell sm_120's 48 KB default static-smem cap):
#   q[D_K]   f32     -- q upcast once (1 KB at D_K=256)
#   k[D_K]   f32     -- k upcast once (1 KB at D_K=256)
#   S_tile[D_K, VLANES, 4]  f32  -- alpha-scaled state tile, kept between passes
#                                -- D_K=256, VLANES=BLOCK_V/4, 4 cols/lane.
#                                -- BLOCK_V=32 -> 256*8*4*4B = 32 KB; total 34 KB ✓.
#                                -- BLOCK_V=64 -> 64 KB; would need dynamic smem
#                                --     setattr (not exposed in Mojo 1.0.0b1).
#
# Per-thread regs:
#   v_vec[4]      f32  -- v columns
#   u_acc[4]      f32  -- partial u = k · S_scaled for this stripe
#   r[4]          f32  -- residual = v - u
#   o_acc[4]      f32  -- output accumulator
#
# Memory pattern:
#   Each thread loads 4 contiguous f32 cols of S_in per d_k row -> LDG.E.128.
#   Each thread stores 4 contiguous f32 cols of S_out per d_k row -> STG.E.128.
#   (FFMA-class; no HMMA, no UTMALDG.)
#
# Bench shape: qwen3_next_decode  B=1 H=16 D_K=D_V=256.
# Correctness shape: same (smaller would need a separate template instantiation;
# for cross-frontend bench parity we only run the qwen3 shape and verify against
# an inline CPU GDN reference at atol=1e-3).
#
# Bytes/iter (matches cuda-attn-gdn formula, same as W1c bench harness):
#   per (bh): state read+write = 2 * D_K * D_V * 4    = 524288
#             io  = (2*D_K + 2*D_V + 2) * 2           = 2052
#             total per bh = 526340 ≈ 514 KiB
#   * B*H = 16 -> ~8.04 MiB per iter.
#
# Algorithmic concern for Mojo: state-recurrence in-CTA-shared-mem WORKS in
# Mojo 1.0.0b1's idioms (verified by mojo-attn-bf16's softmax kernel which
# uses similar shared-mem reduce + multi-pass-over-row pattern). The only
# real constraint is the static-smem cap; we sidestep with BLOCK_V=32.

from std.math import sqrt
from std.sys import has_accelerator
from std.gpu import (
    barrier,
    block_idx,
    thread_idx,
)
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout.layout_tensor import Layout, LayoutTensor


# Compile-time tile params.
comptime D_K: Int     = 256
comptime D_V: Int     = 256
comptime BLOCK_V: Int = 64           # 64 KB smem state tile; Mojo dynsmem promo OK
comptime VLANES: Int  = BLOCK_V // 4  # = 8 threads/CTA
comptime TPB: Int     = VLANES       # one thread per 4-col stripe


# ============================================================================
# GDN-decode kernel.
#
# Inputs (all UnsafePointer + plain indexing -- no LayoutTensor for the dram
# tensors; we want the simplest possible code path so the codegen produces
# straight LDG.E.128 / FFMA, similar to mojo-matmul.mojo).
#
#   Q     (B*H, D_K)         f16
#   K     (B*H, D_K)         f16
#   V     (B*H, D_V)         f16
#   A     (B*H,)             f16   alpha
#   Be    (B*H,)             f16   beta
#   S_in  (B*H, D_K, D_V)    f32
#   S_out (B*H, D_K, D_V)    f32
#   O     (B*H, D_V)         f16
#
# Grid:  (B*H, D_V / BLOCK_V) = (16, 8) at qwen3 shape  -> 128 CTAs.
# Block: (TPB,) = (8,)  -- 1/4 warp utilization (perf concession to fit smem).
# ============================================================================
def gdn_decode_kernel(
    q:    UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    k:    UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    v:    UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    a:    UnsafePointer[Scalar[DType.float16], MutAnyOrigin],   # alpha
    bp:   UnsafePointer[Scalar[DType.float16], MutAnyOrigin],   # beta
    s_in: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    s_out:UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    o:    UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
):
    var bh = Int(block_idx.x)            # 0..B*H-1
    var bv = Int(block_idx.y)            # 0..D_V/BLOCK_V-1
    var tid = Int(thread_idx.x)          # 0..TPB-1
    var col0 = bv * BLOCK_V + tid * 4    # first d_v column for this thread

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
    # State tile: [D_K rows, VLANES lanes, 4 cols-per-lane], all f32.
    var sm_S = LayoutTensor[
        DType.float32,
        Layout.row_major(D_K, VLANES, 4),
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

    # ---- Per-thread v_vec[4]: load 4 f16 v cols, upcast to f32 ----
    var v0: Float32 = v[bh * D_V + col0 + 0].cast[DType.float32]()
    var v1: Float32 = v[bh * D_V + col0 + 1].cast[DType.float32]()
    var v2: Float32 = v[bh * D_V + col0 + 2].cast[DType.float32]()
    var v3: Float32 = v[bh * D_V + col0 + 3].cast[DType.float32]()

    barrier()
    var alpha: Float32 = sm_ab[0][0]
    var beta:  Float32 = sm_ab[1][0]

    # ============================================================================
    # Pass 1: load (D_K, BLOCK_V) state tile, scale by alpha, cache to smem,
    #         accumulate u_acc = sum_k k[k] * (alpha * S_in[k, col0:col0+4])
    #         per thread (4-wide).
    # ============================================================================
    var u0: Float32 = 0.0
    var u1: Float32 = 0.0
    var u2: Float32 = 0.0
    var u3: Float32 = 0.0

    var bh_state_off: Int = bh * D_K * D_V

    var k_iter: Int = 0
    while k_iter < D_K:
        var row_off = bh_state_off + k_iter * D_V + col0
        var s0 = s_in[row_off + 0] * alpha
        var s1 = s_in[row_off + 1] * alpha
        var s2 = s_in[row_off + 2] * alpha
        var s3 = s_in[row_off + 3] * alpha

        # Cache S_scaled into smem for pass 2.
        sm_S[k_iter, tid, 0] = s0
        sm_S[k_iter, tid, 1] = s1
        sm_S[k_iter, tid, 2] = s2
        sm_S[k_iter, tid, 3] = s3

        # u_acc += k_k * s_scaled
        var kkv: Float32 = sm_k[k_iter][0]
        u0 += kkv * s0
        u1 += kkv * s1
        u2 += kkv * s2
        u3 += kkv * s3

        k_iter += 1

    # ---- Residual r = v - u  (per-thread, no cross-thread reduction needed) ----
    var r0: Float32 = v0 - u0
    var r1: Float32 = v1 - u1
    var r2: Float32 = v2 - u2
    var r3: Float32 = v3 - u3

    # ============================================================================
    # Pass 2: S_out = S_scaled + beta · k ⊗ r,  then  o += q[k] * S_out_row
    # ============================================================================
    var o0: Float32 = 0.0
    var o1: Float32 = 0.0
    var o2: Float32 = 0.0
    var o3: Float32 = 0.0

    k_iter = 0
    while k_iter < D_K:
        var s0: Float32 = sm_S[k_iter, tid, 0][0]
        var s1: Float32 = sm_S[k_iter, tid, 1][0]
        var s2: Float32 = sm_S[k_iter, tid, 2][0]
        var s3: Float32 = sm_S[k_iter, tid, 3][0]

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


# ============================================================================
# Main: deterministic init -> warmup -> 50-iter timed bench -> CPU GDN ref.
# ============================================================================
def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
        return

    with DeviceContext() as ctx:
        print("[mojo-attn-gdn] GPU:", ctx.name())

        # ---- Shape: qwen3_next_decode ----
        comptime B  = 1
        comptime NH = 16
        comptime BH = B * NH
        comptime QKV_ELEMS  = BH * D_K   # Q, K
        comptime V_ELEMS    = BH * D_V   # V
        comptime O_ELEMS    = BH * D_V
        comptime SCAL_ELEMS = BH         # alpha, beta
        comptime S_ELEMS    = BH * D_K * D_V

        # ---- Buffers ----
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

        # ---- Deterministic init (Knuth golden-ratio hash, small magnitudes) ----
        # Q, K small so q·k stays bounded; V small; alpha ~0.9 (decay-ish);
        # beta ~0.1 (small update); S_in small.
        for i in range(QKV_ELEMS):
            q_host[i] = (Float32(((i * 2654435761) % 64)) * 0.01 - 0.32).cast[DType.float16]()
        for i in range(QKV_ELEMS):
            k_host[i] = (Float32((((i + 17) * 2654435761) % 64)) * 0.01 - 0.32).cast[DType.float16]()
        for i in range(V_ELEMS):
            v_host[i] = (Float32((((i + 31) * 2654435761) % 64)) * 0.01 - 0.32).cast[DType.float16]()
        for i in range(SCAL_ELEMS):
            # alpha in [0.85, 0.95]
            a_host[i] = (Float32(0.85) + Float32((i * 13) % 10) * 0.01).cast[DType.float16]()
            # beta in [0.05, 0.15]
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

        # ---- Launch params ----
        comptime grid_x = BH
        comptime grid_y = D_V // BLOCK_V

        # ---- Warmup ----
        ctx.enqueue_function[gdn_decode_kernel, gdn_decode_kernel](
            q_dev.unsafe_ptr(), k_dev.unsafe_ptr(), v_dev.unsafe_ptr(),
            a_dev.unsafe_ptr(), b_dev.unsafe_ptr(),
            sin_dev.unsafe_ptr(), sout_dev.unsafe_ptr(), o_dev.unsafe_ptr(),
            grid_dim=(grid_x, grid_y), block_dim=(TPB,),
        )
        ctx.enqueue_function[gdn_decode_kernel, gdn_decode_kernel](
            q_dev.unsafe_ptr(), k_dev.unsafe_ptr(), v_dev.unsafe_ptr(),
            a_dev.unsafe_ptr(), b_dev.unsafe_ptr(),
            sin_dev.unsafe_ptr(), sout_dev.unsafe_ptr(), o_dev.unsafe_ptr(),
            grid_dim=(grid_x, grid_y), block_dim=(TPB,),
        )
        ctx.synchronize()

        # ---- Timed bench: 50 iters, ctx.execution_time per iter ----
        @parameter
        def body(ctx: DeviceContext) raises -> None:
            ctx.enqueue_function[gdn_decode_kernel, gdn_decode_kernel](
                q_dev.unsafe_ptr(), k_dev.unsafe_ptr(), v_dev.unsafe_ptr(),
                a_dev.unsafe_ptr(), b_dev.unsafe_ptr(),
                sin_dev.unsafe_ptr(), sout_dev.unsafe_ptr(), o_dev.unsafe_ptr(),
                grid_dim=(grid_x, grid_y), block_dim=(TPB,),
            )

        var num_iters = 50
        # Use a fixed-size SIMD-backed array; 64 lanes covers our 50 iters.
        var iter_ms = SIMD[DType.float64, 64](0.0)
        for it in range(num_iters):
            var t = ctx.execution_time[body](1)
            iter_ms[it] = Float64(t) / 1e6  # ns -> ms
        ctx.synchronize()

        # ---- Sort first num_iters entries (insertion sort) for median/best ----
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

        # ---- Bytes/iter (matches cuda-attn-gdn bench formula) ----
        var state_bytes: Float64 = 2.0 * Float64(D_K) * Float64(D_V) * 4.0
        var io_bytes:    Float64 = (2.0 * Float64(D_K) + 2.0 * Float64(D_V) + 2.0) * 2.0
        var bytes_per_iter: Float64 = Float64(BH) * (state_bytes + io_bytes)

        var min_s    = min_ms    * 1e-3
        var median_s = median_ms * 1e-3
        var gbps_best:   Float64 = bytes_per_iter / min_s    / 1e9
        var gbps_median: Float64 = bytes_per_iter / median_s / 1e9

        # ---- Copy back O + S_out for correctness check ----
        ctx.enqueue_copy(dst_buf=o_host, src_buf=o_dev)
        ctx.enqueue_copy(dst_buf=sout_host, src_buf=sout_dev)
        ctx.synchronize()

        # =====================================================================
        # CPU GDN reference (full O(B*H*d_k*d_v) at single timestep -- 1M ops,
        # under 1 s in interpreted Mojo).
        #
        # For each (bh, j in [0, D_V)):
        #   u_j     = sum_t k[bh, t] * alpha * S_in[bh, t, j]
        #   r_j     = v[bh, j] - u_j
        #   o_j     = sum_t q[bh, t] * (alpha * S_in[bh, t, j] + beta * k[bh, t] * r_j)
        #   S_out[bh, t, j] = alpha * S_in[bh, t, j] + beta * k[bh, t] * r_j
        # =====================================================================
        var max_abs_o: Float32 = 0.0
        var max_abs_s: Float32 = 0.0
        var fail_bh: Int = -1
        var fail_j:  Int = -1
        var fail_got: Float32 = 0.0
        var fail_ref: Float32 = 0.0

        # We sample 256 (bh, j) pairs for output check (each requires D_K mults
        # = 256, total 65k ops -> instant). For S_out, we sample 256 (bh, t, j)
        # triples.
        var n_samples_o = 256
        var n_samples_s = 256

        for s_idx in range(n_samples_o):
            var seed = s_idx * 2654435761
            var bh = (((seed >> 24) % BH) + BH) % BH
            var j  = (((seed >> 11) % D_V) + D_V) % D_V

            var alpha_v: Float32 = a_host[bh].cast[DType.float32]()
            var beta_v:  Float32 = b_host[bh].cast[DType.float32]()
            var v_j:     Float32 = v_host[bh * D_V + j].cast[DType.float32]()

            # u_j = sum_t k_t * alpha * S_in[bh, t, j]
            var u_j: Float32 = 0.0
            for t in range(D_K):
                var k_t: Float32 = k_host[bh * D_K + t].cast[DType.float32]()
                var s_t: Float32 = sin_host[bh * D_K * D_V + t * D_V + j]
                u_j += k_t * (alpha_v * s_t)
            var r_j: Float32 = v_j - u_j

            # o_j = sum_t q_t * (alpha * S_in[bh,t,j] + beta * k_t * r_j)
            var o_j: Float32 = 0.0
            for t in range(D_K):
                var q_t: Float32 = q_host[bh * D_K + t].cast[DType.float32]()
                var k_t: Float32 = k_host[bh * D_K + t].cast[DType.float32]()
                var s_t: Float32 = sin_host[bh * D_K * D_V + t * D_V + j]
                var s_new = alpha_v * s_t + beta_v * k_t * r_j
                o_j += q_t * s_new
            # Cast through f16 to match kernel's output dtype.
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

        # S_out check (sample 256 triples)
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

            # u_j (full reduce over D_K)
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

        # ---- Report ----
        var ok_o = max_abs_o <= 1e-3
        var ok_s = max_abs_s <= 5e-3   # f32 path; tighter than W1c qwen3 1e-2
        print("[mojo-attn-gdn] shape: B=", B, " H=", NH, " D_K=", D_K, " D_V=", D_V)
        print("[mojo-attn-gdn] tile: BLOCK_V=", BLOCK_V, " VLANES=", VLANES, " TPB=", TPB)
        print("[mojo-attn-gdn] bytes/iter=", bytes_per_iter / 1024.0, " KiB")
        print("[mojo-attn-gdn] timing: min_ms=", min_ms, " median_ms=", median_ms,
              " max_ms=", max_ms)
        print("[mojo-attn-gdn] GB/s: best=", gbps_best, " median=", gbps_median)
        print("[mojo-attn-gdn] correctness:")
        print("[mojo-attn-gdn]   o    max_abs_err=", max_abs_o, " (atol=1e-3) ",
              "OK" if ok_o else "FAIL")
        print("[mojo-attn-gdn]   Sout max_abs_err=", max_abs_s, " (atol=5e-3) ",
              "OK" if ok_s else "FAIL")
        if fail_bh >= 0:
            print("[mojo-attn-gdn] first o-fail: bh=", fail_bh, " j=", fail_j,
                  " got=", fail_got, " ref=", fail_ref)

        # ---- Cross-frontend GDN comparison ----
        print("[mojo-attn-gdn] cross-frontend GDN at qwen3_next_decode shape:")
        print("[mojo-attn-gdn]   cuda-attn-gdn-tma         = 1032 GB/s (W22.10, TMA)")
        print("[mojo-attn-gdn]   cutile-attn-gdn           =  610 GB/s (saturation)")
        print("[mojo-attn-gdn]   cuda-attn-gdn (W1c FFMA)  =  417 GB/s (best)")
        print("[mojo-attn-gdn]   oxide-attn-gdn (FFMA)     =  276 GB/s")
        print("[mojo-attn-gdn]   mojo-attn-gdn (this)      = ", gbps_best, " GB/s (best, FFMA)")
        var ratio_w1c = gbps_best / 417.0
        var ratio_oxi = gbps_best / 276.0
        var ratio_cut = gbps_best / 610.0
        print("[mojo-attn-gdn] ratios: vs W1c=", ratio_w1c,
              " vs oxide=", ratio_oxi, " vs cutile=", ratio_cut)
