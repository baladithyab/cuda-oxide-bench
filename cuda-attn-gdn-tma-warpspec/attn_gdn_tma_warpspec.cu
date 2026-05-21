// Wave 22.13 — cuda-attn-gdn-tma-warpspec:
// Combine W22.10's TMA path (cp.async.bulk.tensor.2d → SASS UTMALDG) with
// W22.12's warp-specialisation prescription (TPB widened, 3 named barriers,
// math density via consumer-warp D_K split, >49 KB dyn smem opt-in).
//
// Reference baselines on RTX 5090 / sm_120:
//   W1c   cuda-attn-gdn               417.7 GB/s  LDG.E.128, TPB=16
//   cuTile cutile-attn-gdn            610.6 GB/s  LDG.E×128, 8-warp warp-spec
//   W22.9 cuda-attn-gdn-async         245.3 GB/s  cuda::pipeline (regress)
//   W22.10 cuda-attn-gdn-tma         1032   GB/s  TMA + TPB=16 (best so far)
//   W22.11 cuda-attn-gdn-async-tpb128 ~250 GB/s   1P+3C cuda::pipeline (no TMA)
//
// W22.13 hypothesis. Stack W22.10's TMA load with W22.12's warp-spec
// prescription:
//   1. Keep W22.10's single-TMA-load-of-full-(D_K,BLOCK_V)-tile structure.
//      That is the load-side win (one bulk-tensor issue, mbarrier-gated,
//      hardware completes tx-count, UTMALDG SASS line). Multi-stage ring
//      buffering ON TOP would split this into 4 small TMAs and lose the
//      batched-load efficiency. **TMA path = unchanged from W22.10.**
//   2. Widen blockDim from 16 → 128 (1 producer warp + 3 consumer warps).
//      This is the W22.12 launch-geometry recommendation #1.
//   3. Restructure math (W22.12 recommendation #2): split the FFMA recurrence
//      across the 3 consumer warps along the D_K axis. Each consumer warp
//      `c` ∈ {1,2,3} owns rows [c_lo, c_hi) = [bal_split(c, D_K), bal_split(c+1, D_K)).
//      All 16 active math lanes per warp (lanes 0..VLANES-1=15) are doing FFMA
//      simultaneously → 48 active math lanes vs W22.10's 16 (3× density).
//   4. After the per-warp partial u_acc is computed, reduce across consumer
//      warps via shared memory before computing residual r = v - u.
//   5. Pass 2 is split the same way (each consumer warp owns its D_K-row
//      range for both the S_out store and the partial o_acc). The o_acc
//      is reduced across consumer warps in smem at the very end.
//   6. Three named-barrier slots used (W22.12 EIATTR_NUM_BARRIERS=3 target):
//        - mbarrier `bar` (TMA completion)         — covered by mbarrier.try_wait
//        - bar.sync ID 1 (consumer-warp partial reduction)
//        - bar.sync ID 2 (Pass-1 → Pass-2 fence + Pass-2 → output fence)
//      Plus the implicit __syncthreads() (BAR.SYNC ID 0) used pre-TMA.
//   7. Dynamic smem ≈ 17 KB at correctness shape, 75 KB at qwen3 shape;
//      both under the 99 KB carveout. Opt in via cudaFuncSetAttribute
//      MaxDynamicSharedMemorySize=99000 host-side per task.
//
// Why TPB=128, NOT 256:
//   * VLANES = BLOCK_V/4 = 16 stripes per row. Even with 256 threads, only
//     16 are math-active per row. Extra threads buy SM-scheduler concurrency
//     (W22.11 lesson) but no math density unless we ALSO split D_K across
//     warps.
//   * With 4 warps (1 producer + 3 consumers) and D_K-row split across
//     consumer warps, we get 3× math density (48 vs 16 active math lanes).
//   * 8 warps would give 7 consumer warps and 7× density, but the grid is
//     only 64 blocks (qwen3 shape) on 170 SMs → already 1-CTA/SM. Going
//     wider per CTA without expanding the grid leaves SMs idle. cuTile uses
//     8 warps because its math (FMUL=900, FADD=771 per CTA) is heavy enough
//     to amortise; nvcc's per-CTA math at 168 effective FMAs is too thin.
//     We stop at 128 threads = 4 warps. (W22.12 doc explicitly notes
//     ">256 threads or >100 KB smem won't help".)
//
// The correctness shape is small (D_K=64, BH=8, grid=8 blocks at BV=64,
// D_V=64 → grid (8,1)). With 1 consumer warp this would still work, but
// matching the W22.12 prescription requires real warp-spec which only kicks
// in at 4 warps. The kernel is templated so the same code path runs both
// shapes.
//
// Build:
//   /usr/local/cuda/bin/nvcc -ccbin clang-14 -O3 -arch=sm_120 -lcuda \
//       -o attn_gdn_tma_warpspec attn_gdn_tma_warpspec.cu
//
// SM_90+ requirement (sm_120 satisfies):
//   * cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes
//     — PTX ISA 8.6, available __CUDA_ARCH__ >= 900
//   * mbarrier.* shared.b64 ops — sm_80+
//   * cuTensorMapEncodeTiled (driver API) — CUDA 12.0+, requires linking -lcuda
//   * bar.sync.aligned with named ID — sm_70+ (we use IDs 0,1,2)

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
#include <cuda.h>          // for CUtensorMap, cuTensorMapEncodeTiled

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA err %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while(0)

#define CKD(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
    const char* msg = nullptr; cuGetErrorString(r, &msg); \
    fprintf(stderr, "CU driver err %s @ %s:%d\n", msg ? msg : "(unknown)", __FILE__, __LINE__); exit(1); } } while(0)

// ============================================================================
// .npy loader (NPY1.0 / NPY2.0; little-endian only).
// Cloned verbatim from W22.10.
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
        uint16_t hl;
        if (fread(&hl, 1, 2, f) != 2) { fclose(f); return false; }
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
    if (p == std::string::npos) { fclose(f); return false; }
    while (p < hdr.size() && hdr[p] != '\'') ++p; ++p;
    size_t q = p;
    while (q < hdr.size() && hdr[q] != '\'') ++q;
    out.dtype = hdr.substr(p, q - p);
    if (out.dtype == "<f2") out.elem_size = 2;
    else if (out.dtype == "<f4") out.elem_size = 4;
    else { fclose(f); return false; }

    p = find_after("'shape':");
    if (p == std::string::npos) { fclose(f); return false; }
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
        fclose(f); return false;
    }
    fclose(f);
    return true;
}

// ============================================================================
// Device-side TMA / mbarrier / named-bar helpers (inline PTX).
// ============================================================================
__device__ __forceinline__ uint32_t smem_to_u32(const void* smem_ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
}

__device__ __forceinline__ void mbarrier_init(uint64_t* bar_smem, uint32_t arrive_count) {
#if __CUDA_ARCH__ >= 800
    uint32_t bar_addr = smem_to_u32(bar_smem);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
                 :: "r"(bar_addr), "r"(arrive_count) : "memory");
#endif
}

__device__ __forceinline__ void mbarrier_inval(uint64_t* bar_smem) {
#if __CUDA_ARCH__ >= 800
    uint32_t bar_addr = smem_to_u32(bar_smem);
    asm volatile("mbarrier.inval.shared.b64 [%0];\n" :: "r"(bar_addr) : "memory");
#endif
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar_smem, uint32_t tx_count) {
#if __CUDA_ARCH__ >= 900
    uint32_t bar_addr = smem_to_u32(bar_smem);
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(bar_addr), "r"(tx_count) : "memory");
#endif
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* bar_smem, uint32_t parity) {
#if __CUDA_ARCH__ >= 900
    uint32_t bar_addr = smem_to_u32(bar_smem);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "WAIT_LOOP:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
        "@p bra DONE;\n\t"
        "bra WAIT_LOOP;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(bar_addr), "r"(parity) : "memory");
#endif
}

__device__ __forceinline__ void cp_async_bulk_tensor_2d_g2s(
    void* dst_smem,
    const CUtensorMap* tensorMap,
    int32_t coord_x, int32_t coord_y,
    uint64_t* bar_smem)
{
#if __CUDA_ARCH__ >= 900
    uint32_t dst_addr = smem_to_u32(dst_smem);
    uint32_t bar_addr = smem_to_u32(bar_smem);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :
        : "r"(dst_addr),
          "l"(reinterpret_cast<uint64_t>(tensorMap)),
          "r"(coord_x), "r"(coord_y),
          "r"(bar_addr)
        : "memory");
#endif
}

// Named-barrier `bar.sync.aligned` with explicit ID + thread-count. Allows
// per-warp-group rendezvous (the W22.12 EIATTR_NUM_BARRIERS=3 pattern).
//   - bar_id ∈ [0, 15] (sm_70+)
//   - thread_count must be multiple of 32; here we use 96 (3 consumer warps).
// We leave bar.sync ID 0 to ptxas/__syncthreads(). Use IDs 1 and 2 for
// consumer-only synchronisation.
__device__ __forceinline__ void named_barrier_sync(int bar_id, int thread_count) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(bar_id), "r"(thread_count) : "memory");
}

// Helper: compute consumer-warp's D_K row range. Rounds-down split.
//   3 consumer warps (warp_id 1, 2, 3) → consumer index c = warp_id - 1, c ∈ {0, 1, 2}.
//   D_K split into [c * D_K / 3, (c+1) * D_K / 3) — last warp absorbs remainder.
__device__ __forceinline__ int dk_split_lo(int c, int D_K) {
    return (c * D_K) / 3;
}
__device__ __forceinline__ int dk_split_hi(int c, int D_K) {
    return ((c + 1) * D_K) / 3;
}

// ============================================================================
// GDN-decode kernel — TMA + warp-spec (1 producer + 3 consumers).
//
// Template params:
//   D_K     — key dim (state rows). Compile-time so loops fully unroll.
//   BLOCK_V — d_v-tile width per block. VLANES = BLOCK_V/4 must be ≤ 32
//             so it fits in one warp (we use lanes 0..VLANES-1 as math lanes).
//
// Layout:
//   gridDim   = (B*H, D_V / BLOCK_V)
//   blockDim  = TPB = 128 (4 warps)
//   warps:
//     0          : producer (issues TMA, idle in math)
//     1, 2, 3    : consumers; each owns D_K/3 rows of state.
//
// ============================================================================
template <int D_K, int BLOCK_V>
__global__ void __launch_bounds__(128) gdn_decode_tma_warpspec_kernel(
    const __half* __restrict__ Q,                        // (B*H, D_K)
    const __half* __restrict__ K,                        // (B*H, D_K)
    const __half* __restrict__ V,                        // (B*H, D_V)
    const __half* __restrict__ Alpha,                    // (B*H,)
    const __half* __restrict__ Beta,                     // (B*H,)
    float*        __restrict__ S_out,                    // (B*H, D_K, D_V)
    __half*       __restrict__ O,                        // (B*H, D_V)
    const __grid_constant__ CUtensorMap S_in_tmap,       // descriptor for S_in
    int B_H, int D_V)
{
    static_assert(BLOCK_V % 4 == 0, "BLOCK_V must be multiple of 4 for float4 lanes");
    constexpr int VLANES = BLOCK_V / 4;
    static_assert(VLANES <= 32, "VLANES must fit in one warp (=lane indices 0..31)");
    constexpr int TPB    = 128;
    constexpr int N_CONSUMERS = 3;
    constexpr int CONSUMER_THREADS = N_CONSUMERS * 32;   // = 96

    const int bh   = blockIdx.x;
    const int bv   = blockIdx.y;
    const int tid  = threadIdx.x;
    const int wid  = tid >> 5;            // warp_id ∈ {0,1,2,3}
    const int lane = tid & 31;            // lane_id
    const bool is_producer = (wid == 0);
    const bool is_consumer = (wid >= 1);
    const int  cidx = wid - 1;            // consumer index ∈ {0,1,2} (only valid if is_consumer)
    const bool is_math_lane = is_consumer && (lane < VLANES);

    // Each math lane owns 4 contiguous d_v cols, derived from `lane` (NOT tid).
    const int my_lane_col0 = bv * BLOCK_V + lane * 4;

    // ── Shared memory layout (dynamic). ──
    // smem_S_raw      : (D_K, BLOCK_V) f32 — TMA destination, 128B-aligned.
    //                   After Pass 1, holds alpha-scaled tile (in-place).
    //                   At BLOCK_V=64, D_K=64 → 16 KB; D_K=256 → 64 KB.
    // smem_q, smem_k  : D_K f32 each.
    // smem_partial_u  : N_CONSUMERS * VLANES * 4 floats (3 consumer warps each
    //                   contribute a (VLANES-stripe) partial u_acc f32 slab).
    //                   Used to reduce u across consumer warps.
    // smem_partial_o  : same — for o_acc reduction at the end.
    // bar             : 8B mbarrier guarding TMA completion.
    // s_alpha, s_beta : 4B each.
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float*    smem_S_raw  = reinterpret_cast<float*>(smem_raw);                            // [D_K * BLOCK_V]
    float*    smem_q      = smem_S_raw + (size_t)D_K * BLOCK_V;                            // [D_K]
    float*    smem_k      = smem_q + D_K;                                                  // [D_K]
    // 4 floats per math lane × VLANES lanes × N_CONSUMERS warps.
    float4*   smem_part_u = reinterpret_cast<float4*>(smem_k + D_K);                       // [N_CONSUMERS * VLANES]
    float4*   smem_part_o = smem_part_u + (size_t)N_CONSUMERS * VLANES;                    // [N_CONSUMERS * VLANES]
    uint64_t* bar         = reinterpret_cast<uint64_t*>(smem_part_o + (size_t)N_CONSUMERS * VLANES);  // [1]
    float*    s_scalars   = reinterpret_cast<float*>(bar + 1);                             // [2] — alpha, beta

    // ── Initialise mbarrier (one thread). arrive_count=1: only the producer
    //    thread will arrive on this barrier. All threads wait via parity. ──
    if (tid == 0) {
        mbarrier_init(bar, /*arrive_count=*/1);
    }

    // ── Cooperative load q, k (f16 → f32) into smem; alpha/beta scalars. ──
    const __half* Qbh = Q + (size_t)bh * D_K;
    const __half* Kbh = K + (size_t)bh * D_K;
    #pragma unroll 4
    for (int k = tid; k < D_K; k += TPB) {
        smem_q[k] = __half2float(Qbh[k]);
        smem_k[k] = __half2float(Kbh[k]);
    }
    if (tid == 0) {
        s_scalars[0] = __half2float(Alpha[bh]);
        s_scalars[1] = __half2float(Beta[bh]);
    }

    // ── Per-math-lane: load v stripe (4 halves → float4) from gmem. ──
    float4 v_vec = make_float4(0.f, 0.f, 0.f, 0.f);
    if (is_math_lane && cidx == 0) {
        // Only consumer-warp-1 math lanes need v (one per d_v-stripe).
        // It will be added to r and broadcast via smem if needed by other
        // warps — but r is only used in Pass 2 by the same lanes that own
        // each stripe. Other consumer warps own DIFFERENT D_K rows of the
        // SAME d_v stripes; they need r for their rows of S_out too.
        // Solution: load v in ALL consumer warps (each one's lane `l` reads
        // the same v[bv*BV + l*4]).
    }
    // Load v in every consumer warp — small redundant load to avoid an
    // extra cross-warp broadcast. Each consumer warp's lanes 0..VLANES-1
    // read their own 4-half stripe.
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

    // ── Producer issues TMA bulk-tensor load: pull the FULL (D_K, BLOCK_V)
    //    S_in tile for this (bh, bv) directly into smem_S_raw via UTMALDG.
    //    This is the W22.10 path — UNCHANGED. The whole tile lands in one
    //    TMA issue; the mbarrier with expect_tx=TILE_BYTES gates completion.
    //
    // Tile coords: bv*BLOCK_V (innermost = d_v offset), bh*D_K (outer).
    constexpr uint32_t TILE_BYTES = (uint32_t)D_K * (uint32_t)BLOCK_V * sizeof(float);

    // BAR.SYNC ID 0 (default __syncthreads): make sure mbarrier_init from
    // tid==0 is visible to the producer thread before it arrives.
    __syncthreads();

    if (is_producer && lane == 0) {
        mbarrier_arrive_expect_tx(bar, TILE_BYTES);
        cp_async_bulk_tensor_2d_g2s(
            smem_S_raw, &S_in_tmap,
            /*coord_x=*/bv * BLOCK_V,
            /*coord_y=*/bh * D_K,
            bar);
    }
    // ALL threads wait for TMA. mbarrier.try_wait.parity is the warp-spec
    // primitive — producer thread arrives, consumer warps spin on parity.
    mbarrier_wait_parity(bar, /*parity=*/0);

    // BAR.SYNC ID 0 again — guarantees q/k stores from earlier coop loop
    // are visible to all consumer warps (they all read from smem_q/smem_k
    // in the FFMA loop).
    __syncthreads();

    const float alpha = s_scalars[0];
    const float beta  = s_scalars[1];

    // ── Pass 1 — split D_K rows across the 3 consumer warps. Each warp's
    //    16 math lanes process its assigned D_K-row range over all VLANES
    //    stripes (lane `l` owns stripe `l`, i.e. cols [l*4, l*4+4)). ──
    //
    // Each consumer warp accumulates a partial u_acc covering its row range.
    // After the loop, partials are summed across warps via smem_part_u.
    float4 u_part = make_float4(0.f, 0.f, 0.f, 0.f);

    if (is_consumer) {
        const int k_lo = dk_split_lo(cidx, D_K);
        const int k_hi = dk_split_hi(cidx, D_K);

        if (is_math_lane) {
            #pragma unroll 4
            for (int k = k_lo; k < k_hi; ++k) {
                float4* sptr = reinterpret_cast<float4*>(&smem_S_raw[(size_t)k * BLOCK_V + lane * 4]);
                float4 s = *sptr;          // LDS.128
                s.x *= alpha;
                s.y *= alpha;
                s.z *= alpha;
                s.w *= alpha;
                *sptr = s;                 // STS.128 — overwrite with scaled value for Pass 2

                float kk = smem_k[k];
                u_part.x += kk * s.x;
                u_part.y += kk * s.y;
                u_part.z += kk * s.z;
                u_part.w += kk * s.w;
            }
        }

        // Each math lane writes its partial u_acc to smem_part_u
        // (cidx, lane) → smem_part_u[cidx * VLANES + lane].
        if (is_math_lane) {
            smem_part_u[(size_t)cidx * VLANES + lane] = u_part;
        }
    }

    // BAR.SYNC ID 1 — consumer-only barrier, 96 threads. This is the
    // "consumer-warp partial reduction" sync (W22.12 EIATTR_NUM_BARRIERS=3
    // slot 2). All 3 consumer warps must finish writing their partials
    // before any consumer reduces.
    if (is_consumer) {
        named_barrier_sync(/*bar_id=*/1, /*thread_count=*/CONSUMER_THREADS);
    }

    // ── Reduce partials across consumer warps → final u_acc.
    //    Each math lane reads the 3 partial slabs and sums them. ──
    float4 u_acc = make_float4(0.f, 0.f, 0.f, 0.f);
    if (is_math_lane) {
        #pragma unroll
        for (int c = 0; c < N_CONSUMERS; ++c) {
            float4 p = smem_part_u[(size_t)c * VLANES + lane];
            u_acc.x += p.x;
            u_acc.y += p.y;
            u_acc.z += p.z;
            u_acc.w += p.w;
        }
    }

    // ── Residual r = v - u (per math lane). ──
    float4 r;
    r.x = v_vec.x - u_acc.x;
    r.y = v_vec.y - u_acc.y;
    r.z = v_vec.z - u_acc.z;
    r.w = v_vec.w - u_acc.w;

    // BAR.SYNC ID 0 — fence Pass 1 → Pass 2: ensure smem_S_raw STS.128 stores
    // (alpha-scaled tile) from all consumer warps are visible block-wide
    // before Pass 2 reads them. Producer warp also participates so it
    // doesn't race ahead at exit.
    __syncthreads();

    // ── Pass 2 — same D_K-row split. Each consumer warp computes:
    //      S_out[k, my_cols] = S_scaled[k, my_cols] + beta * k[k] * r
    //      o_part           += q[k] * S_out[k, my_cols]
    //    for k in [k_lo, k_hi). Producer warp idle.
    //
    //    Then partials of o_acc are reduced across consumer warps via
    //    smem_part_o, and consumer warp 1 writes the final o stripe to gmem. ──
    float4 o_part = make_float4(0.f, 0.f, 0.f, 0.f);

    if (is_consumer) {
        const int k_lo = dk_split_lo(cidx, D_K);
        const int k_hi = dk_split_hi(cidx, D_K);

        const size_t bh_state_off = (size_t)bh * D_K * D_V;
        float* Sout_bh = S_out + bh_state_off;

        if (is_math_lane) {
            #pragma unroll 4
            for (int k = k_lo; k < k_hi; ++k) {
                float4* sptr = reinterpret_cast<float4*>(&smem_S_raw[(size_t)k * BLOCK_V + lane * 4]);
                float4 s = *sptr;                  // alpha-scaled tile (LDS.128)
                float kk = smem_k[k];
                float qk = smem_q[k];
                float bk = beta * kk;

                s.x = s.x + bk * r.x;
                s.y = s.y + bk * r.y;
                s.z = s.z + bk * r.z;
                s.w = s.w + bk * r.w;

                // STG.E.128 store of S_out row.
                float4* sout_ptr =
                    reinterpret_cast<float4*>(Sout_bh + (size_t)k * D_V + my_lane_col0);
                *sout_ptr = s;

                o_part.x += qk * s.x;
                o_part.y += qk * s.y;
                o_part.z += qk * s.z;
                o_part.w += qk * s.w;
            }

            // Write per-warp partial o_acc to smem.
            smem_part_o[(size_t)cidx * VLANES + lane] = o_part;
        }
    }

    // BAR.SYNC ID 2 — consumer-only barrier, 96 threads. (W22.12
    // EIATTR_NUM_BARRIERS=3 slot 3.) Wait for all consumer warps to finish
    // their partial-o writes.
    if (is_consumer) {
        named_barrier_sync(/*bar_id=*/2, /*thread_count=*/CONSUMER_THREADS);
    }

    // ── Reduce o partials across consumer warps; consumer warp 1 writes
    //    the final o stripe to gmem. (Other consumer warps could also write
    //    other stripes, but with VLANES=BLOCK_V/4 stripes per block and only
    //    one BLOCK_V-wide block-y-tile per (bh, bv), warp 1 alone covers it.) ──
    if (is_consumer && cidx == 0 && is_math_lane) {
        float4 o_acc = make_float4(0.f, 0.f, 0.f, 0.f);
        #pragma unroll
        for (int c = 0; c < N_CONSUMERS; ++c) {
            float4 p = smem_part_o[(size_t)c * VLANES + lane];
            o_acc.x += p.x;
            o_acc.y += p.y;
            o_acc.z += p.z;
            o_acc.w += p.w;
        }

        // Store o (f16). 4 halves contiguously → 8B store (= float2).
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

    // Invalidate barrier so the next launch reuses cleanly.
    if (tid == 0) {
        mbarrier_inval(bar);
    }
}

// ============================================================================
// Host-side TMA descriptor builder.
// (Cloned verbatim from W22.10 — same descriptor encodes the SAME tile.)
// ============================================================================
struct TmaDesc {
    CUtensorMap map;
};

static void build_tma_descriptor_S_in(
    TmaDesc& out,
    const float* d_S_in_base,
    int B_H, int D_K, int D_V,
    int BLOCK_V)
{
    cuuint64_t globalDim[2]    = { (cuuint64_t)D_V, (cuuint64_t)B_H * (cuuint64_t)D_K };
    cuuint64_t globalStrides[1] = { (cuuint64_t)D_V * sizeof(float) };
    cuuint32_t boxDim[2]       = { (cuuint32_t)BLOCK_V, (cuuint32_t)D_K };
    cuuint32_t elemStrides[2]  = { 1u, 1u };

    CUresult r = cuTensorMapEncodeTiled(
        &out.map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        /*tensorRank=*/2,
        const_cast<void*>(reinterpret_cast<const void*>(d_S_in_base)),
        globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        const char* msg = nullptr; cuGetErrorString(r, &msg);
        fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", msg ? msg : "(?)");
        exit(1);
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

template <int D_K, int BLOCK_V>
static void launch_gdn_tma_warpspec(const GDNShape& sh,
                                    const __half* dQ, const __half* dK, const __half* dV,
                                    const __half* dA, const __half* dB,
                                    const float*  dS_in, float* dS_out, __half* dO,
                                    cudaStream_t stream)
{
    const int B_H = sh.batch * sh.n_heads;
    constexpr int TPB = 128;
    constexpr int N_CONSUMERS = 3;
    constexpr int VLANES = BLOCK_V / 4;
    dim3 grid(B_H, sh.d_v / BLOCK_V);
    dim3 block(TPB);

    // Build TMA descriptor (driver call — host-side, once per launch).
    TmaDesc desc{};
    build_tma_descriptor_S_in(desc, dS_in, B_H, D_K, sh.d_v, BLOCK_V);

    // Shared memory layout (must match kernel):
    //   smem_S_raw      : D_K * BLOCK_V * 4 bytes
    //   smem_q          : D_K * 4
    //   smem_k          : D_K * 4
    //   smem_part_u     : N_CONSUMERS * VLANES * 16  (= float4 per math lane)
    //   smem_part_o     : N_CONSUMERS * VLANES * 16
    //   bar             : 8 (uint64_t mbarrier)
    //   s_scalars       : 8 (alpha, beta)
    size_t smem_bytes =
        (size_t)D_K * BLOCK_V * sizeof(float) +
        2 * (size_t)D_K * sizeof(float) +
        2 * (size_t)N_CONSUMERS * VLANES * sizeof(float4) +
        sizeof(uint64_t) +
        2 * sizeof(float);
    smem_bytes = (smem_bytes + 15) & ~size_t(15);

    // Opt into >49 KB dyn smem path per W22.12 prescription. Even at the
    // correctness shape (smem_bytes ~17 KB) we pre-emptively raise the cap
    // so the same launch path is exercised on both shapes — and the host
    // attribute set is what's required for the qwen3 shape (~75 KB).
    cudaFuncSetAttribute(
        (const void*)gdn_decode_tma_warpspec_kernel<D_K, BLOCK_V>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        99 * 1024);

    gdn_decode_tma_warpspec_kernel<D_K, BLOCK_V><<<grid, block, smem_bytes, stream>>>(
        dQ, dK, dV, dA, dB, dS_out, dO, desc.map, B_H, sh.d_v);
}

static void launch_gdn_dispatch(const GDNShape& sh,
                                const __half* dQ, const __half* dK, const __half* dV,
                                const __half* dA, const __half* dB,
                                const float*  dS_in, float* dS_out, __half* dO,
                                cudaStream_t stream)
{
    if (sh.d_k == 64 && sh.d_v == 64) {
        launch_gdn_tma_warpspec<64, 64>(sh, dQ, dK, dV, dA, dB, dS_in, dS_out, dO, stream);
    } else if (sh.d_k == 256 && sh.d_v == 256) {
        launch_gdn_tma_warpspec<256, 64>(sh, dQ, dK, dV, dA, dB, dS_in, dS_out, dO, stream);
    } else {
        fprintf(stderr, "unsupported (d_k=%d, d_v=%d)\n", sh.d_k, sh.d_v);
        exit(2);
    }
}

#ifndef ATTN_GDN_TMA_WS_BENCH_HARNESS
// ============================================================================
// Correctness driver
// ============================================================================
static int run_correctness(const std::string& inputs_dir) {
    GDNShape sh{"correctness", 2, 4, 64, 64};
    printf("[gdn-tma-ws] === correctness run (B=%d H=%d d_k=%d d_v=%d) ===\n",
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

    double max_abs_o = 0.0, exp_o_mag = 0.0;
    for (size_t i = 0; i < o_elems; ++i) {
        float got = __half2float(ho[i]);
        float want = __half2float(o_exp_h[i]);
        double aw = fabs((double)want);
        if (aw > exp_o_mag) exp_o_mag = aw;
        double a = fabs((double)got - (double)want);
        if (a > max_abs_o) max_abs_o = a;
    }
    double max_abs_s = 0.0, exp_s_mag = 0.0;
    for (size_t i = 0; i < s_elems; ++i) {
        float got = hs[i];
        float want = s_exp_f[i];
        double aw = fabs((double)want);
        if (aw > exp_s_mag) exp_s_mag = aw;
        double a = fabs((double)got - (double)want);
        if (a > max_abs_s) max_abs_s = a;
    }

    const double ATOL_O = 1e-3;
    const double ATOL_S = 5e-3;
    bool ok_o = max_abs_o <= ATOL_O;
    bool ok_s = max_abs_s <= ATOL_S;

    printf("[gdn-tma-ws] o    max_abs=%.3e   |want|max=%.3e   %s\n",
           max_abs_o, exp_o_mag, ok_o ? "OK" : "FAIL");
    printf("[gdn-tma-ws] Sout max_abs=%.3e   |want|max=%.3e   %s\n",
           max_abs_s, exp_s_mag, ok_s ? "OK" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dA); cudaFree(dB);
    cudaFree(dSin); cudaFree(dSout); cudaFree(dO);
    return (ok_o && ok_s) ? 0 : 2;
}

static int run_bench_shape_smoke(const std::string& inputs_dir) {
    GDNShape sh{"qwen3_next_decode", 1, 16, 256, 256};
    printf("\n[gdn-tma-ws] === bench-shape smoke (B=%d H=%d d_k=%d d_v=%d) ===\n",
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
    printf("[gdn-tma-ws] (qwen3) o   max_abs=%.3e   %s\n", max_abs_o, ok_o ? "OK" : "FAIL");
    printf("[gdn-tma-ws] (qwen3) Sout max_abs=%.3e %s\n", max_abs_s, ok_s ? "OK" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dA); cudaFree(dB);
    cudaFree(dSin); cudaFree(dSout); cudaFree(dO);
    return (ok_o && ok_s) ? 0 : 3;
}

int main(int argc, char** argv) {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("[gdn-tma-ws] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
    if (p.major < 9) {
        fprintf(stderr, "[gdn-tma-ws] needs sm_90+ (TMA / cp.async.bulk.tensor)\n");
        return 1;
    }

    std::string inputs_dir =
        "/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs";
    if (argc > 1) inputs_dir = argv[1];
    printf("[gdn-tma-ws] inputs dir: %s\n", inputs_dir.c_str());

    int rc = run_correctness(inputs_dir);
    if (rc != 0) {
        fprintf(stderr, "[gdn-tma-ws] correctness FAILED at correctness shape — stopping\n");
        return rc;
    }
    rc = run_bench_shape_smoke(inputs_dir);
    return rc;
}
#endif // ATTN_GDN_TMA_WS_BENCH_HARNESS
