// Wave 22.11 — cuda-attn-gdn-async-tpb128 — single-timestep GDN decode using
// EXPLICIT WARP SPECIALIZATION (1 producer warp + 3 consumer warps) on top
// of cuda::pipeline<thread_scope_block> + cuda::memcpy_async.
//
// Background. Wave 22.9 (cuda-attn-gdn-async, TPB=16, cuda::pipeline at
// thread scope) measured 311.8 GB/s — REGRESSION vs W1c's 417.7 GB/s and
// far below cuTile's 610 GB/s. That falsified the "cuda::pipeline alone
// closes the gap" hypothesis from W22.8.
//
// W22.11 hypothesis (W22.8's #2-ranked candidate). cuTile uses
// TPB=128+ with explicit producer/consumer warp roles. The remaining
// untested variable from W22.8's hypothesis ranking is **launch geometry +
// warp specialization**, not async loads per se. Widening TPB to 128 and
// dedicating one warp to issuing cp.async while three warps drive the FFMA
// chain mimics cuTile's structural pattern. If this gets us to ~520 GB/s
// (the W22.8 prediction's middle-of-CI), then the gap is structural
// (warp roles → SM scheduler instruction-level parallelism), not API-level.
//
// Warp split chosen: **1 producer + 3 consumers** (vs 2P+2C). Reasoning:
//   * The producer side is bound by LSU issue rate (cp.async throughput),
//     not by warp count. One warp can saturate the LSU on Blackwell
//     because LDGSTS issues at 1/cycle/SM.
//   * The consumer side is bound by FFMA throughput AND smem-read
//     throughput. Tripling consumer warps gives the SM scheduler 3 warps
//     to round-robin through, hiding FFMA + LDS latency.
//   * Asymmetric (1+3) ≠ symmetric (2+2): cuTile profiles show higher
//     consumer-side warp count, consistent with this asymmetry.
//
// Math layout (consumer side). VLANES = BLOCK_V/4 = 16 float4 stripes per
// state-row tile (BLOCK_V=64 cols, vectorised as float4). The 96 consumer
// threads only have 16 owned stripes, so 16 of the 96 do the math:
// **consumer warp 1 lanes 0..15 = "math lanes"** (tids 32..47). The
// remaining 80 consumer threads are warp-specialization "padding" — their
// presence increases warps-per-block from 1 (W22.9) to 4, which:
//   (a) Lets the SM scheduler dual-issue producer-warp cp.async alongside
//       consumer-warp FFMA from a different warp.
//   (b) Pushes occupancy upward (warps-per-SM minimum is 4 for full
//       SM scheduler engagement on Blackwell).
//   This matches cuTile's pattern: "few math lanes, many warps for
//   scheduler-level concurrency."
//
// Pipeline scope. We use `cuda::pipeline<thread_scope_block>` so producer
// and consumer warps share the same pipeline object and barriers. This
// is the *required* scope for cross-warp producer/consumer separation —
// thread_scope_thread (W22.9) cannot coordinate across warps.
//
// Per-block shared memory. Same layout as W22.9 (the only change is which
// threads do which work, not what data lives where).
//   smem_q[D_K] (f32)
//   smem_k[D_K] (f32)
//   smem_S[D_K * VLANES] (float4) — alpha-scaled state cache, populated by
//                                    consumer warps in Pass 1, read in Pass 2
//   smem_ring[N_STAGES * VLANES] (float4) — async ring buffer, written
//                                            by producer warp, read by
//                                            consumer warps
// At D_K=256, BLOCK_V=64, N_STAGES=4: 1 KiB + 1 KiB + 64 KiB + 1 KiB =
// 67 KiB. Within sm_120's 100 KiB dynamic-smem budget after opt-in.
//
// Build:
//   make attn_gdn_async_tpb128
//
// Acceptance (per W22.11 task):
//   * compiles
//   * correctness vs PyTorch GDN-naive ≤ 1e-3 (output) / 5e-3 (state)
//   * SASS shows BSSY/BSYNC.RECONVERGENT or barrier instructions matching
//     cuTile's pattern AND LDGSTS > 0 (cp.async preserved).
//
// DO NOT run timed benches in this cell — orchestrator runs ./bench
// separately on idle GPU per cuda-exploration session convention.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/pipeline>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA err %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while(0)

// ============================================================================
// .npy loader (NPY1.0 / NPY2.0; little-endian only). Cloned from cuda-attn-gdn.
// ============================================================================
struct Npy {
    std::vector<int64_t> shape;
    std::string dtype;   // "<f2" or "<f4"
    std::vector<uint8_t> data;
    size_t elem_size = 0;
};

static bool read_npy(const std::string& path, Npy& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    uint8_t magic[6];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "%s: not an NPY file\n", path.c_str()); fclose(f); return false;
    }
    uint8_t ver[2];
    if (fread(ver, 1, 2, f) != 2) { fclose(f); return false; }
    uint32_t header_len;
    if (ver[0] == 1) {
        uint16_t hl; if (fread(&hl, 1, 2, f) != 2) { fclose(f); return false; }
        header_len = hl;
    } else {
        if (fread(&header_len, 1, 4, f) != 4) { fclose(f); return false; }
    }
    std::string hdr(header_len, ' ');
    if (fread(&hdr[0], 1, header_len, f) != header_len) { fclose(f); return false; }

    auto find_after = [&](const std::string& key) -> size_t {
        size_t p = hdr.find(key);
        return p == std::string::npos ? std::string::npos : p + key.size();
    };
    size_t p = find_after("'descr':");
    if (p == std::string::npos) { fprintf(stderr, "no descr\n"); fclose(f); return false; }
    while (p < hdr.size() && hdr[p] != '\'') ++p; ++p;
    size_t q = p;
    while (q < hdr.size() && hdr[q] != '\'') ++q;
    out.dtype = hdr.substr(p, q - p);
    if (out.dtype == "<f2") out.elem_size = 2;
    else if (out.dtype == "<f4") out.elem_size = 4;
    else { fprintf(stderr, "unsupported dtype %s\n", out.dtype.c_str()); fclose(f); return false; }

    p = find_after("'shape':");
    if (p == std::string::npos) { fprintf(stderr, "no shape\n"); fclose(f); return false; }
    while (p < hdr.size() && hdr[p] != '(') ++p; ++p;
    q = p;
    while (q < hdr.size() && hdr[q] != ')') ++q;
    std::string shape_str = hdr.substr(p, q - p);
    size_t pos = 0;
    while (pos < shape_str.size()) {
        while (pos < shape_str.size() && !isdigit(shape_str[pos])) ++pos;
        if (pos >= shape_str.size()) break;
        int64_t v = 0;
        while (pos < shape_str.size() && isdigit(shape_str[pos])) { v = v * 10 + (shape_str[pos] - '0'); ++pos; }
        out.shape.push_back(v);
    }

    size_t nelem = 1;
    for (auto d : out.shape) nelem *= (size_t)d;
    out.data.resize(nelem * out.elem_size);
    if (fread(out.data.data(), 1, out.data.size(), f) != out.data.size()) {
        fprintf(stderr, "%s: short read\n", path.c_str()); fclose(f); return false;
    }
    fclose(f);
    return true;
}

// ============================================================================
// GDN-decode kernel — TPB=128, 1 producer warp + 3 consumer warps,
// cuda::pipeline<thread_scope_block>.
//
// Template params:
//   D_K      — key dim (state rows). Compile-time so loops fully unroll.
//   BLOCK_V  — d_v-tile width per block (multiple of 4).
//   N_STAGES — number of in-flight cuda::memcpy_async slots.
//   TPB      — total threads per block (must be 128 for this variant).
//
// Runtime parallelism (4× wider than W22.9):
//   gridDim  = (B * H, D_V / BLOCK_V)
//   blockDim = TPB  (128 = 4 warps)
//
// Warp roles (decided at runtime via threadIdx.x / 32):
//   warp 0 (tids 0..31)    : producer  (lanes 0..15 active for cp.async)
//   warps 1..3 (tids 32..127): consumer (warp 1 lanes 0..15 = math lanes,
//                              i.e. tids 32..47; tids 48..127 are
//                              warp-specialization padding)
// ============================================================================
template <int D_K, int BLOCK_V, int N_STAGES, int TPB>
__global__ void gdn_decode_async_tpb128_kernel(
    const __half* __restrict__ Q,       // (B*H, D_K)
    const __half* __restrict__ K,       // (B*H, D_K)
    const __half* __restrict__ V,       // (B*H, D_V)
    const __half* __restrict__ Alpha,   // (B*H,)
    const __half* __restrict__ Beta,    // (B*H,)
    const float*  __restrict__ S_in,    // (B*H, D_K, D_V)
    float*        __restrict__ S_out,   // (B*H, D_K, D_V)
    __half*       __restrict__ O,       // (B*H, D_V)
    int B_H, int D_V)
{
    static_assert(BLOCK_V % 4 == 0, "BLOCK_V must be multiple of 4 for float4 lanes");
    static_assert(N_STAGES >= 2,    "N_STAGES must be >= 2 to overlap load with consume");
    static_assert(TPB == 128,       "this variant is fixed at TPB=128 (4 warps)");
    constexpr int VLANES = BLOCK_V / 4;     // # float4 stripes per row = 16 for BV=64
    static_assert(VLANES <= 32,     "VLANES must fit in one warp's lanes for the 1P+3C split");

    const int bh  = blockIdx.x;             // 0..B*H-1
    const int bv  = blockIdx.y;             // 0..D_V/BLOCK_V-1
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;           // 0..3
    const int lane_id = tid & 31;           // 0..31

    // Warp-role classification.
    const bool is_producer       = (warp_id == 0);
    const bool is_consumer       = (warp_id >= 1);
    // Within consumer warps, only warp 1 lanes 0..VLANES-1 are "math lanes"
    // — they own the float4 stripes and do the FFMA recurrence.
    const bool is_math_lane      = (warp_id == 1) && (lane_id < VLANES);
    // Producer-active lanes: lanes 0..VLANES-1 of warp 0 issue the cp.async.
    const bool is_producer_lane  = (warp_id == 0) && (lane_id < VLANES);

    // Each math lane (and each producer-active lane) owns 4 contiguous d_v cols.
    // We derive col0 from lane_id (NOT tid) so producer & math lanes index
    // the same 4-column stripe consistently.
    const int my_lane_col0 = bv * BLOCK_V + lane_id * 4;

    // ── Shared-memory layout (dynamic). Same as W22.9. ──
    extern __shared__ unsigned char smem_raw[];
    float*  smem_q    = reinterpret_cast<float*>(smem_raw);                              // [D_K]
    float*  smem_k    = smem_q + D_K;                                                    // [D_K]
    float4* smem_S    = reinterpret_cast<float4*>(smem_k + D_K);                         // [D_K * VLANES]
    float4* smem_ring = smem_S + (size_t)D_K * VLANES;                                   // [N_STAGES * VLANES]

    // ── Cooperative load of q, k (f16 → f32) into smem. All 128 threads help. ──
    const __half* Qbh = Q + (size_t)bh * D_K;
    const __half* Kbh = K + (size_t)bh * D_K;
    #pragma unroll 4
    for (int kk = tid; kk < D_K; kk += TPB) {
        smem_q[kk] = __half2float(Qbh[kk]);
        smem_k[kk] = __half2float(Kbh[kk]);
    }

    __shared__ float s_alpha;
    __shared__ float s_beta;
    if (tid == 0) {
        s_alpha = __half2float(Alpha[bh]);
        s_beta  = __half2float(Beta[bh]);
    }

    // Load v stripe (4 halves → float4) — only math lanes need v.
    float4 v_vec = make_float4(0.f, 0.f, 0.f, 0.f);
    if (is_math_lane) {
        const __half* Vbh = V + (size_t)bh * D_V;
        const float2* Vptr = reinterpret_cast<const float2*>(Vbh + my_lane_col0);
        float2 v_packed = *Vptr;
        const __half* hv = reinterpret_cast<const __half*>(&v_packed);
        v_vec.x = __half2float(hv[0]);
        v_vec.y = __half2float(hv[1]);
        v_vec.z = __half2float(hv[2]);
        v_vec.w = __half2float(hv[3]);
    }

    __syncthreads();
    const float alpha = s_alpha;
    const float beta  = s_beta;

    // ── Pass 1: 1P+3C software pipeline over k=[0..D_K). ──
    //
    // Producer (warp 0) issues cp.async for ring slot s = k % N_STAGES.
    // Consumer (warp 1 math lanes) reads ring slot, alpha-scales, FFMA into
    // u_acc, writes alpha-scaled tile to smem_S for Pass 2.
    //
    // We use cuda::pipeline at BLOCK scope so the producer-commit on warp 0
    // releases the consumer-wait on warp 1. This is the cuTile-style
    // cross-warp barrier.
    const size_t bh_state_off = (size_t)bh * D_K * D_V;
    const float* Sbh = S_in + bh_state_off;

    // Block-scope pipeline state. Each role has its own role enum; ALL
    // threads must construct the pipeline (collective op) but only the
    // active producer/consumer threads call acquire/wait. cuda::pipeline
    // requires a pipeline_shared_state in shared memory.
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, N_STAGES> pipe_state;

    // Each thread joins the pipeline as either a producer or a consumer.
    // Convention: 32 producer threads (warp 0), 96 consumer threads (warps 1-3).
    auto pipe = cuda::make_pipeline(
        cg::this_thread_block(),
        &pipe_state,
        is_producer ? cuda::pipeline_role::producer
                    : cuda::pipeline_role::consumer);

    constexpr int K_LIMIT = D_K;

    // ── Prologue: producer issues first N_STAGES async copies. ──
    if (is_producer) {
        #pragma unroll
        for (int s = 0; s < N_STAGES; ++s) {
            if (s < K_LIMIT) {
                pipe.producer_acquire();
                if (is_producer_lane) {
                    const float4* gptr =
                        reinterpret_cast<const float4*>(Sbh + (size_t)s * D_V + my_lane_col0);
                    float4* sptr = smem_ring + (size_t)s * VLANES + lane_id;
                    cuda::memcpy_async(sptr, gptr,
                                       cuda::aligned_size_t<16>(sizeof(float4)),
                                       pipe);
                }
                pipe.producer_commit();
            }
        }
    }

    // ── Steady state: consumers drive the FFMA recurrence; producer
    //    refills slots in lock-step with consumer_release.
    // Both roles must call wait/release in the same logical order so the
    // pipeline barriers stay paired. We use a single integer k loop and
    // each role does its own portion.
    float4 u_acc = make_float4(0.f, 0.f, 0.f, 0.f);

    // We iterate the FULL k-range from both roles; producer gets to k+N_STAGES
    // *before* the consumer needs k. Each role's branch performs the same
    // number of acquire/commit OR wait/release calls so the block-scope
    // pipeline state stays balanced.
    #pragma unroll 4
    for (int k = 0; k < K_LIMIT; ++k) {
        if (is_consumer) {
            // Wait for stage k to be ready (producer has committed it).
            pipe.consumer_wait();

            // Only math lanes do the FFMA; other consumer threads idle here.
            if (is_math_lane) {
                const int slot = k % N_STAGES;
                float4 s = smem_ring[(size_t)slot * VLANES + lane_id];
                s.x *= alpha;
                s.y *= alpha;
                s.z *= alpha;
                s.w *= alpha;

                // Cache the alpha-scaled tile for Pass 2.
                smem_S[(size_t)k * VLANES + lane_id] = s;

                // u_acc += k_k * S_scaled[k]
                const float kk_v = smem_k[k];
                u_acc.x += kk_v * s.x;
                u_acc.y += kk_v * s.y;
                u_acc.z += kk_v * s.z;
                u_acc.w += kk_v * s.w;
            }

            pipe.consumer_release();
        } else {
            // Producer: refill slot k+N_STAGES if there is one.
            const int next_k = k + N_STAGES;
            if (next_k < K_LIMIT) {
                pipe.producer_acquire();
                if (is_producer_lane) {
                    const int slot = next_k % N_STAGES;
                    const float4* gptr =
                        reinterpret_cast<const float4*>(Sbh + (size_t)next_k * D_V + my_lane_col0);
                    float4* sptr = smem_ring + (size_t)slot * VLANES + lane_id;
                    cuda::memcpy_async(sptr, gptr,
                                       cuda::aligned_size_t<16>(sizeof(float4)),
                                       pipe);
                }
                pipe.producer_commit();
            }
        }
    }

    // Drain producer side: producer issued (N_STAGES + (K_LIMIT - N_STAGES)) =
    // K_LIMIT total acquires/commits in prologue+steady; consumer issued
    // K_LIMIT waits/releases. Pipeline is balanced; no extra drain needed.

    // ── Residual r = v - u (math lanes only). ──
    float4 r = make_float4(0.f, 0.f, 0.f, 0.f);
    if (is_math_lane) {
        r.x = v_vec.x - u_acc.x;
        r.y = v_vec.y - u_acc.y;
        r.z = v_vec.z - u_acc.z;
        r.w = v_vec.w - u_acc.w;
    }

    // smem_S written by consumer math lanes; Pass 2 reads it from the same
    // lanes. Block-wide sync to make sure all stores are visible (and to
    // sync producer/consumer roles before Pass 2 runs uniformly across the
    // block).
    __syncthreads();

    // ── Pass 2: S_out = S_scaled + beta * k ⊗ r,  o += q[k] * S_out_row.
    //    Only math lanes participate. Pass 2 has no async loads — all
    //    state is already in smem from Pass 1.
    if (is_math_lane) {
        float* Sout_bh = S_out + bh_state_off;
        float4 o_acc = make_float4(0.f, 0.f, 0.f, 0.f);

        #pragma unroll 8
        for (int k = 0; k < D_K; ++k) {
            float4 s = smem_S[(size_t)k * VLANES + lane_id];   // S_scaled[k, my_4_cols]
            const float kk_v = smem_k[k];
            const float qk = smem_q[k];
            const float bk = beta * kk_v;

            s.x = s.x + bk * r.x;
            s.y = s.y + bk * r.y;
            s.z = s.z + bk * r.z;
            s.w = s.w + bk * r.w;

            // Vectorized 128-bit store of S_out row.
            float4* sout_ptr =
                reinterpret_cast<float4*>(Sout_bh + (size_t)k * D_V + my_lane_col0);
            *sout_ptr = s;

            o_acc.x += qk * s.x;
            o_acc.y += qk * s.y;
            o_acc.z += qk * s.z;
            o_acc.w += qk * s.w;
        }

        // Store o (f16). 4 halves contiguously → 8-byte store.
        __half* Obh = O + (size_t)bh * D_V;
        __half halves[4];
        halves[0] = __float2half(o_acc.x);
        halves[1] = __float2half(o_acc.y);
        halves[2] = __float2half(o_acc.z);
        halves[3] = __float2half(o_acc.w);
        float2 packed;
        memcpy(&packed, halves, sizeof(packed));
        *reinterpret_cast<float2*>(Obh + my_lane_col0) = packed;
    }
}

// ============================================================================
// Host launcher
// ============================================================================
struct GDNShape {
    const char* name;
    int batch;
    int n_heads;
    int d_k;
    int d_v;
};

template <int D_K, int BLOCK_V, int N_STAGES>
static void launch_gdn_async_tpb128(const GDNShape& sh,
                             const __half* dQ, const __half* dK, const __half* dV,
                             const __half* dA, const __half* dB,
                             const float*  dS_in, float* dS_out, __half* dO,
                             cudaStream_t stream)
{
    const int B_H = sh.batch * sh.n_heads;
    constexpr int TPB = 128;
    constexpr int VLANES = BLOCK_V / 4;
    static_assert(VLANES <= 32, "VLANES must fit in one warp for 1P+3C split");
    dim3 grid(B_H, sh.d_v / BLOCK_V);
    dim3 block(TPB);

    // smem: q, k (f32, D_K each) + S_scaled tile (D_K * VLANES * 16 B) +
    //       ring buffer (N_STAGES * VLANES * 16 B).
    size_t smem_bytes =
        2 * D_K * sizeof(float) +
        (size_t)D_K * VLANES * sizeof(float4) +
        (size_t)N_STAGES * VLANES * sizeof(float4);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            (const void*)gdn_decode_async_tpb128_kernel<D_K, BLOCK_V, N_STAGES, TPB>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes);
    }

    gdn_decode_async_tpb128_kernel<D_K, BLOCK_V, N_STAGES, TPB>
        <<<grid, block, smem_bytes, stream>>>(
            dQ, dK, dV, dA, dB, dS_in, dS_out, dO, B_H, sh.d_v);
}

// Dispatch (D_K, BLOCK_V) at runtime to the right template instantiation.
static void launch_gdn_dispatch(const GDNShape& sh,
                                const __half* dQ, const __half* dK, const __half* dV,
                                const __half* dA, const __half* dB,
                                const float*  dS_in, float* dS_out, __half* dO,
                                cudaStream_t stream)
{
    if (sh.d_k == 64 && sh.d_v == 64) {
        launch_gdn_async_tpb128<64, 64, 4>(sh, dQ, dK, dV, dA, dB, dS_in, dS_out, dO, stream);
    } else if (sh.d_k == 256 && sh.d_v == 256) {
        launch_gdn_async_tpb128<256, 64, 4>(sh, dQ, dK, dV, dA, dB, dS_in, dS_out, dO, stream);
    } else {
        fprintf(stderr, "unsupported (d_k=%d, d_v=%d) — only correctness (64/64) and qwen3_next_decode (256/256) wired\n",
                sh.d_k, sh.d_v);
        exit(2);
    }
}

// ============================================================================
// Correctness driver — same shape as cuda-attn-gdn (W1c). Tolerance ≤ 1e-3.
// ============================================================================
static int run_correctness(const std::string& inputs_dir) {
    GDNShape sh{"correctness", /*B*/2, /*H*/4, /*d_k*/64, /*d_v*/64};
    printf("[gdn-async-tpb128] === correctness run (B=%d H=%d d_k=%d d_v=%d) ===\n",
           sh.batch, sh.n_heads, sh.d_k, sh.d_v);

    Npy q, k, v, a, b, sin_, oexp, sexp;
    if (!read_npy(inputs_dir + "/gdn_correctness_q_f16.npy",            q))    return 1;
    if (!read_npy(inputs_dir + "/gdn_correctness_k_f16.npy",            k))    return 1;
    if (!read_npy(inputs_dir + "/gdn_correctness_v_f16.npy",            v))    return 1;
    if (!read_npy(inputs_dir + "/gdn_correctness_alpha_f16.npy",        a))    return 1;
    if (!read_npy(inputs_dir + "/gdn_correctness_beta_f16.npy",         b))    return 1;
    if (!read_npy(inputs_dir + "/gdn_correctness_S_in_f32.npy",         sin_)) return 1;
    if (!read_npy(inputs_dir + "/gdn_correctness_o_expected_f16.npy",   oexp)) return 1;
    if (!read_npy(inputs_dir + "/gdn_correctness_S_out_expected_f32.npy", sexp)) return 1;

    const int B_H = sh.batch * sh.n_heads;
    size_t qkv_elems = (size_t)B_H * sh.d_k;
    size_t v_elems   = (size_t)B_H * sh.d_v;
    size_t s_elems   = (size_t)B_H * sh.d_k * sh.d_v;
    size_t o_elems   = (size_t)B_H * sh.d_v;
    size_t scal      = (size_t)B_H;

    if (q.data.size() != qkv_elems * 2 ||
        k.data.size() != qkv_elems * 2 ||
        v.data.size() != v_elems * 2 ||
        a.data.size() != scal * 2 ||
        b.data.size() != scal * 2 ||
        sin_.data.size() != s_elems * 4 ||
        oexp.data.size() != o_elems * 2 ||
        sexp.data.size() != s_elems * 4) {
        fprintf(stderr, "[gdn-async-tpb128] correctness: NPY shape/size mismatch\n");
        return 1;
    }

    __half *dQ, *dK, *dV, *dA, *dB, *dO;
    float  *dSin, *dSout;
    CK(cudaMalloc(&dQ,    qkv_elems * sizeof(__half)));
    CK(cudaMalloc(&dK,    qkv_elems * sizeof(__half)));
    CK(cudaMalloc(&dV,    v_elems   * sizeof(__half)));
    CK(cudaMalloc(&dA,    scal      * sizeof(__half)));
    CK(cudaMalloc(&dB,    scal      * sizeof(__half)));
    CK(cudaMalloc(&dSin,  s_elems   * sizeof(float)));
    CK(cudaMalloc(&dSout, s_elems   * sizeof(float)));
    CK(cudaMalloc(&dO,    o_elems   * sizeof(__half)));

    CK(cudaMemcpy(dQ,    q.data.data(),    q.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK,    k.data.data(),    k.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV,    v.data.data(),    v.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dA,    a.data.data(),    a.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,    b.data.data(),    b.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSin,  sin_.data.data(), sin_.data.size(), cudaMemcpyHostToDevice));
    CK(cudaMemset(dSout, 0,                s_elems * sizeof(float)));
    CK(cudaMemset(dO,    0,                o_elems * sizeof(__half)));

    launch_gdn_dispatch(sh, dQ, dK, dV, dA, dB, dSin, dSout, dO, 0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<__half> ho(o_elems);
    std::vector<float>  hs(s_elems);
    CK(cudaMemcpy(ho.data(), dO,    o_elems * sizeof(__half), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hs.data(), dSout, s_elems * sizeof(float),  cudaMemcpyDeviceToHost));

    const __half* o_exp_h = reinterpret_cast<const __half*>(oexp.data.data());
    const float*  s_exp_f = reinterpret_cast<const float*> (sexp.data.data());

    double max_abs_o = 0.0, max_rel_o = 0.0, exp_o_mag = 0.0;
    for (size_t i = 0; i < o_elems; ++i) {
        float got = __half2float(ho[i]);
        float want = __half2float(o_exp_h[i]);
        double aw = fabs((double)want);
        if (aw > exp_o_mag) exp_o_mag = aw;
        double a = fabs((double)got - (double)want);
        if (a > max_abs_o) max_abs_o = a;
        double denom = aw < 1e-6 ? 1e-6 : aw;
        double r = a / denom;
        if (r > max_rel_o) max_rel_o = r;
    }
    double max_abs_s = 0.0, max_rel_s = 0.0, exp_s_mag = 0.0;
    for (size_t i = 0; i < s_elems; ++i) {
        float got = hs[i];
        float want = s_exp_f[i];
        double aw = fabs((double)want);
        if (aw > exp_s_mag) exp_s_mag = aw;
        double a = fabs((double)got - (double)want);
        if (a > max_abs_s) max_abs_s = a;
        double denom = aw < 1e-6 ? 1e-6 : aw;
        double r = a / denom;
        if (r > max_rel_s) max_rel_s = r;
    }

    const double ATOL_O = 1e-3;
    const double ATOL_S = 5e-3;
    bool ok_o = max_abs_o <= ATOL_O;
    bool ok_s = max_abs_s <= ATOL_S;

    printf("[gdn-async-tpb128] o    max_abs=%.3e max_rel=%.3e   |want|max=%.3e   %s\n",
           max_abs_o, max_rel_o, exp_o_mag, ok_o ? "OK" : "FAIL");
    printf("[gdn-async-tpb128] Sout max_abs=%.3e max_rel=%.3e   |want|max=%.3e   %s\n",
           max_abs_s, max_rel_s, exp_s_mag, ok_s ? "OK" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dA); cudaFree(dB);
    cudaFree(dSin); cudaFree(dSout); cudaFree(dO);
    return (ok_o && ok_s) ? 0 : 2;
}

// ============================================================================
// Bench-shape smoke (NO timed iters — orchestrator runs ./bench separately).
// ============================================================================
static int run_bench_shape_smoke(const std::string& inputs_dir) {
    GDNShape sh{"qwen3_next_decode", 1, 16, 256, 256};
    printf("\n[gdn-async-tpb128] === bench-shape smoke (B=%d H=%d d_k=%d d_v=%d) ===\n",
           sh.batch, sh.n_heads, sh.d_k, sh.d_v);

    Npy q, k, v, a, b, sin_, oexp, sexp;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_q_f16.npy",            q))    return 1;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_k_f16.npy",            k))    return 1;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_v_f16.npy",            v))    return 1;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_alpha_f16.npy",        a))    return 1;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_beta_f16.npy",         b))    return 1;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_S_in_f32.npy",         sin_)) return 1;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_o_expected_f16.npy",   oexp)) return 1;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_S_out_expected_f32.npy", sexp)) return 1;

    const int B_H = sh.batch * sh.n_heads;
    size_t qkv_elems = (size_t)B_H * sh.d_k;
    size_t v_elems   = (size_t)B_H * sh.d_v;
    size_t s_elems   = (size_t)B_H * sh.d_k * sh.d_v;
    size_t o_elems   = (size_t)B_H * sh.d_v;
    size_t scal      = (size_t)B_H;

    __half *dQ, *dK, *dV, *dA, *dB, *dO;
    float  *dSin, *dSout;
    CK(cudaMalloc(&dQ,    qkv_elems * sizeof(__half)));
    CK(cudaMalloc(&dK,    qkv_elems * sizeof(__half)));
    CK(cudaMalloc(&dV,    v_elems   * sizeof(__half)));
    CK(cudaMalloc(&dA,    scal      * sizeof(__half)));
    CK(cudaMalloc(&dB,    scal      * sizeof(__half)));
    CK(cudaMalloc(&dSin,  s_elems   * sizeof(float)));
    CK(cudaMalloc(&dSout, s_elems   * sizeof(float)));
    CK(cudaMalloc(&dO,    o_elems   * sizeof(__half)));

    CK(cudaMemcpy(dQ,    q.data.data(),    q.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK,    k.data.data(),    k.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV,    v.data.data(),    v.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dA,    a.data.data(),    a.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,    b.data.data(),    b.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSin,  sin_.data.data(), sin_.data.size(), cudaMemcpyHostToDevice));
    CK(cudaMemset(dSout, 0,                s_elems * sizeof(float)));
    CK(cudaMemset(dO,    0,                o_elems * sizeof(__half)));

    launch_gdn_dispatch(sh, dQ, dK, dV, dA, dB, dSin, dSout, dO, 0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<__half> ho(o_elems);
    std::vector<float>  hs(s_elems);
    CK(cudaMemcpy(ho.data(), dO,    o_elems * sizeof(__half), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hs.data(), dSout, s_elems * sizeof(float),  cudaMemcpyDeviceToHost));

    const __half* o_exp_h = reinterpret_cast<const __half*>(oexp.data.data());
    const float*  s_exp_f = reinterpret_cast<const float*> (sexp.data.data());

    double max_abs_o = 0.0, max_abs_s = 0.0;
    for (size_t i = 0; i < o_elems; ++i) {
        float got = __half2float(ho[i]);
        float want = __half2float(o_exp_h[i]);
        double a = fabs((double)got - (double)want);
        if (a > max_abs_o) max_abs_o = a;
    }
    for (size_t i = 0; i < s_elems; ++i) {
        double a = fabs((double)hs[i] - (double)s_exp_f[i]);
        if (a > max_abs_s) max_abs_s = a;
    }
    bool ok_o = max_abs_o <= 1e-2;
    bool ok_s = max_abs_s <= 1e-2;
    printf("[gdn-async-tpb128] (qwen3) o    max_abs=%.3e   %s\n", max_abs_o, ok_o ? "OK" : "FAIL");
    printf("[gdn-async-tpb128] (qwen3) Sout max_abs=%.3e   %s\n", max_abs_s, ok_s ? "OK" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dA); cudaFree(dB);
    cudaFree(dSin); cudaFree(dSout); cudaFree(dO);
    return (ok_o && ok_s) ? 0 : 3;
}

int main(int argc, char** argv) {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("[gdn-async-tpb128] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    std::string inputs_dir =
        "/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs";
    if (argc > 1) inputs_dir = argv[1];
    printf("[gdn-async-tpb128] inputs dir: %s\n", inputs_dir.c_str());

    int rc = run_correctness(inputs_dir);
    if (rc != 0) {
        fprintf(stderr, "[gdn-async-tpb128] correctness FAILED at correctness shape — stopping\n");
        return rc;
    }
    rc = run_bench_shape_smoke(inputs_dir);
    return rc;
}
