// Wave 22.10 — cuda-attn-gdn-tma:
// nvcc CUDA C++ Gated DeltaNet (GDN) single-timestep decode using TMA
// (cp.async.bulk.tensor) for the dominant S_in tile load.
//
// Reference baselines on the same RTX 5090 / sm_120 box:
//   W1c   cuda-attn-gdn         417.7 GB/s  uses LDG.E.128 × 16
//   cuTile cutile-attn-gdn      610.6 GB/s  uses LDG.E × 128 (NO TMA, smem pipeline)
//   W22.9 cuda-attn-gdn-async   245.3 GB/s  uses cp.async via cuda::pipeline
//
// W22.10 hypothesis: replace the per-thread float4 LDG.E.128 walk over S_in
// with a single bulk-tensor load that fetches the whole (D_K × BLOCK_V) tile
// into shared via UTMALDG. This is the ONE remaining axis cuTile didn't take
// (cuTile uses scalar smem-pipeline). The W22.8 SASS showed UTMALDG=0 in BOTH
// nvcc and cuTile, so this kernel is the first nvcc data point with UTMALDG > 0
// on this hardware/algorithm.
//
// Per-block layout matches W1c:
//   gridDim  = (B*H, D_V / BLOCK_V)
//   blockDim = (BLOCK_V / 4)         each thread owns 4 d_v columns
//
// Per-block the TMA tile we want is the (D_K, BLOCK_V) slab of S_in for a given
// (bh, bv). We encode S_in as a 2D TMA descriptor with global shape (D_V, B_H*D_K)
// in column-major (TMA convention: innermost dimension first). Per launch we
// supply tile coordinates {col_start, row_start} = {bv*BLOCK_V, bh*D_K} and box
// dims {BLOCK_V, D_K}. The hardware then performs UTMALDG into shared memory
// at smem_S_raw, gated by an mbarrier.
//
// Build:
//   /usr/local/cuda/bin/nvcc -ccbin clang-14 -O3 -arch=sm_120 -lcuda -o attn_gdn_tma attn_gdn_tma.cu
//
// SM_90+ requirement (sm_120 satisfies):
//   * cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes
//     — PTX ISA 8.6, available __CUDA_ARCH__ >= 900 (sm_120 = 1200, OK).
//     Per CCCL header `cccl/cuda/__ptx/instructions/generated/cp_async_bulk_tensor.h`:
//       #if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
//   * mbarrier.init.shared.b64               — sm_80+
//   * mbarrier.arrive.expect_tx.shared.b64   — sm_90+
//   * mbarrier.try_wait.parity.shared.b64    — sm_90+
//   * cuTensorMapEncodeTiled (driver API)    — CUDA 12.0+, requires linking -lcuda

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
    fread(ver, 1, 2, f);
    uint32_t header_len;
    if (ver[0] == 1) {
        uint16_t hl; fread(&hl, 1, 2, f); header_len = hl;
    } else {
        fread(&header_len, 1, 4, f);
    }
    std::string hdr(header_len, ' ');
    fread(&hdr[0], 1, header_len, f);

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
// Device-side TMA / mbarrier helpers (inline PTX).
//
// We write inline PTX directly rather than including <cuda/ptx> to keep the
// kernel self-contained, easy to read, and easy to grep at the SASS level.
// All instructions used here are gated on __CUDA_ARCH__ >= 900 in their CCCL
// counterparts, which is exactly the same gate we need for sm_120.
// ============================================================================
__device__ __forceinline__ uint32_t smem_to_u32(const void* smem_ptr) {
    // Convert generic shared-memory pointer to the 32-bit integer
    // address-space-tagged value that PTX shared::* instructions need.
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

// One-issuer "arrive + expect_tx" — used by the thread that issues TMA. The
// TMA hardware completes the tx-count when bytes-loaded == expected.
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar_smem, uint32_t tx_count) {
#if __CUDA_ARCH__ >= 900
    uint32_t bar_addr = smem_to_u32(bar_smem);
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(bar_addr), "r"(tx_count) : "memory");
#endif
}

// Wait until phase parity flips. We init phase=0, so first wait uses parity=0.
__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* bar_smem, uint32_t parity) {
#if __CUDA_ARCH__ >= 900
    uint32_t bar_addr = smem_to_u32(bar_smem);
    // Spin on try_wait until done. Standard pattern.
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

// cp.async.bulk.tensor.2d — load a (BLOCK_V × D_K) tile from gmem to shared.
// tensorMap   : pointer to a CUtensorMap stored in __grid_constant__ memory
//               or any addressable buffer (cmem/param). NVIDIA recommends
//               kernel-parameter-by-value with __grid_constant__.
// dst_smem    : shared-memory destination. Must be 128-byte aligned.
// coord_x     : tile origin, innermost (col-major) dim — i.e. d_v offset.
// coord_y     : tile origin, next dim — bh*D_K row index.
// bar_smem    : mbarrier guarding completion.
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

// ============================================================================
// GDN-decode kernel — TMA-loaded Pass 1.
//
// Template params:
//   D_K     — key dim (state rows). Compile-time so loops fully unroll.
//   BLOCK_V — d_v-tile width per block (must be multiple of 4 for float4 lanes).
//
// We changed Pass 1's structure relative to W1c:
//   W1c: each thread issues an LDG.E.128 per d_k row → 16 LDG.E.128 / thread.
//   W22.10: ONE TMA bulk-tensor load brings the whole (D_K, BLOCK_V) tile
//           into shared. Then each thread reads its 4 floats per d_k row from
//           shared via LDS, scales by alpha, accumulates u, and writes the
//           scaled tile back to shared for Pass 2.
//
// Pass 2 (S_out store + o accumulate) is unchanged from W1c — TMA is only
// applied to the gmem-load side of the dominant traffic.
// ============================================================================
template <int D_K, int BLOCK_V>
__global__ void __launch_bounds__(BLOCK_V/4) gdn_decode_tma_kernel(
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
    constexpr int TPB    = VLANES;

    const int bh  = blockIdx.x;
    const int bv  = blockIdx.y;
    const int tid = threadIdx.x;

    const int col0 = bv * BLOCK_V + tid * 4;     // first d_v column for this thread

    // ── Shared memory layout (static + 128B aligned for TMA destination) ──
    //
    // smem_S_raw  : (D_K, BLOCK_V) f32 — TMA destination. Layout matches the
    //               TMA tile box: innermost dim is d_v (BLOCK_V floats per row),
    //               outer dim is d_k (D_K rows). So `smem_S_raw[k * BLOCK_V + v]`
    //               is element (k, v) of the tile.
    // smem_q, smem_k : D_K f32 each (q, k upcast).
    // bar         : 8B mbarrier, also shared.
    // s_alpha, s_beta : 4B each.
    //
    // We also reuse smem_S_scaled for the alpha-scaled tile passed to Pass 2.
    // To save space we store the scaled values in-place over smem_S_raw.
    extern __shared__ __align__(16) unsigned char smem_raw[];
    // TMA dest must be 128-byte aligned; place it first.
    float*    smem_S_raw = reinterpret_cast<float*>(smem_raw);                            // [D_K * BLOCK_V]
    float*    smem_q     = smem_S_raw + (size_t)D_K * BLOCK_V;                            // [D_K]
    float*    smem_k     = smem_q + D_K;                                                  // [D_K]
    uint64_t* bar        = reinterpret_cast<uint64_t*>(smem_k + D_K);                     // [1] — 8B
    float*    s_scalars  = reinterpret_cast<float*>(bar + 1);                             // alpha, beta

    // ── Initialize mbarrier (one thread, arrive_count=1: only the producer
    //    thread will arrive; consumers wait on parity). ──
    if (tid == 0) {
        mbarrier_init(bar, /*arrive_count=*/1);
    }

    // ── Cooperatively load q, k (f16 → f32) into shared. ──
    const __half* Qbh = Q + (size_t)bh * D_K;
    const __half* Kbh = K + (size_t)bh * D_K;
    #pragma unroll
    for (int k = tid; k < D_K; k += TPB) {
        smem_q[k] = __half2float(Qbh[k]);
        smem_k[k] = __half2float(Kbh[k]);
    }

    // ── Load alpha, beta scalars. ──
    if (tid == 0) {
        s_scalars[0] = __half2float(Alpha[bh]);
        s_scalars[1] = __half2float(Beta[bh]);
    }

    // ── Per-thread: load v (4 f16) from gmem and convert. Same as W1c. ──
    float4 v_vec;
    {
        const __half* Vbh = V + (size_t)bh * D_V;
        const float2* Vptr = reinterpret_cast<const float2*>(Vbh + col0);
        float2 v_packed = *Vptr;
        const __half* hv = reinterpret_cast<const __half*>(&v_packed);
        v_vec.x = __half2float(hv[0]);
        v_vec.y = __half2float(hv[1]);
        v_vec.z = __half2float(hv[2]);
        v_vec.w = __half2float(hv[3]);
    }

    // ── Issue TMA bulk-tensor load: pull the (D_K rows, BLOCK_V cols) S_in
    //    slab for this (bh, bv) directly into smem_S_raw via UTMALDG. ──
    //
    // Tile coordinates: TMA is in elements (not bytes). The descriptor says
    // global shape is (D_V, B_H*D_K) col-major, box (BLOCK_V, D_K). So we
    // want the tile starting at (col=bv*BLOCK_V, row=bh*D_K).
    constexpr uint32_t TILE_BYTES = (uint32_t)D_K * (uint32_t)BLOCK_V * sizeof(float);
    if (tid == 0) {
        // expect_tx must include exactly the bytes the TMA will deposit.
        mbarrier_arrive_expect_tx(bar, TILE_BYTES);
        cp_async_bulk_tensor_2d_g2s(
            smem_S_raw, &S_in_tmap,
            /*coord_x=*/bv * BLOCK_V,
            /*coord_y=*/bh * D_K,
            bar);
    }
    // All threads must wait for the TMA to land before reading smem_S_raw.
    mbarrier_wait_parity(bar, /*parity=*/0);
    __syncthreads();   // ensures alpha/beta/q/k stores from above are visible too.

    const float alpha = s_scalars[0];
    const float beta  = s_scalars[1];

    // ── Pass 1: each thread reads its 4 f32 columns per d_k row from shared,
    //    scales by alpha (and writes the scaled value back to shared for Pass 2),
    //    and accumulates u_acc = sum_k k[k] * (alpha * S_in[k, col0:col0+4]). ──
    //
    // smem_S_raw[k * BLOCK_V + (tid*4 + j)]  for j in 0..3 is this thread's
    // 4 columns of d_k row k. We pun as float4 since that aligned read is
    // exactly the 16-byte LDS.128 the SASS scheduler likes.
    float4 u_acc = make_float4(0.f, 0.f, 0.f, 0.f);
    #pragma unroll 8
    for (int k = 0; k < D_K; ++k) {
        float4* sptr = reinterpret_cast<float4*>(&smem_S_raw[(size_t)k * BLOCK_V + tid * 4]);
        float4 s = *sptr;          // LDS.128 from TMA-deposited tile
        s.x *= alpha;
        s.y *= alpha;
        s.z *= alpha;
        s.w *= alpha;
        *sptr = s;                 // STS.128 — overwrite with scaled value for Pass 2

        float kk = smem_k[k];
        u_acc.x += kk * s.x;
        u_acc.y += kk * s.y;
        u_acc.z += kk * s.z;
        u_acc.w += kk * s.w;
    }

    // ── Residual r = v - u ──
    float4 r;
    r.x = v_vec.x - u_acc.x;
    r.y = v_vec.y - u_acc.y;
    r.z = v_vec.z - u_acc.z;
    r.w = v_vec.w - u_acc.w;

    // ── Pass 2: S_out = S_scaled + beta · k ⊗ r,  then  o += q[k] * S_out ──
    // Same as W1c — STG.E.128 store of the per-row state, accumulate o in regs.
    const size_t bh_state_off = (size_t)bh * D_K * D_V;
    float* Sout_bh = S_out + bh_state_off;
    float4 o_acc = make_float4(0.f, 0.f, 0.f, 0.f);

    #pragma unroll 8
    for (int k = 0; k < D_K; ++k) {
        float4* sptr = reinterpret_cast<float4*>(&smem_S_raw[(size_t)k * BLOCK_V + tid * 4]);
        float4 s = *sptr;          // alpha-scaled tile lives here now
        float kk = smem_k[k];
        float qk = smem_q[k];
        float bk = beta * kk;

        s.x = s.x + bk * r.x;
        s.y = s.y + bk * r.y;
        s.z = s.z + bk * r.z;
        s.w = s.w + bk * r.w;

        // Vectorised 128-bit STORE of S_out row.
        float4* sout_ptr =
            reinterpret_cast<float4*>(Sout_bh + (size_t)k * D_V + col0);
        *sout_ptr = s;             // STG.E.128

        o_acc.x += qk * s.x;
        o_acc.y += qk * s.y;
        o_acc.z += qk * s.z;
        o_acc.w += qk * s.w;
    }

    // ── Store o (f16). Each thread writes 4 halves contiguously → 8B store. ──
    __half* Obh = O + (size_t)bh * D_V;
    __half halves[4];
    halves[0] = __float2half(o_acc.x);
    halves[1] = __float2half(o_acc.y);
    halves[2] = __float2half(o_acc.z);
    halves[3] = __float2half(o_acc.w);
    float2 packed;
    memcpy(&packed, halves, sizeof(packed));
    *reinterpret_cast<float2*>(Obh + col0) = packed;

    // Invalidate barrier to let the next launch reuse this slot cleanly.
    if (tid == 0) {
        mbarrier_inval(bar);
    }
}

// ============================================================================
// Host-side TMA descriptor builder.
//
// We view S_in (B*H, D_K, D_V) row-major as a 2D tensor with:
//   inner dim (size D_V, stride 1 element)
//   outer dim (size B_H * D_K, stride D_V elements)
// TMA descriptor convention: dimensions are listed innermost-first and strides
// are in BYTES, with the innermost stride implicit (= 1 element). Box dims are
// also innermost-first.
//
// Per the cuTensorMapEncodeTiled docs:
//   globalDim[0]    = innermost size (D_V)
//   globalDim[1]    = outer size     (B_H * D_K)
//   globalStrides[] = strides of dims 1..rank-1 in BYTES (D_V * 4 here)
//   boxDim[0]       = BLOCK_V
//   boxDim[1]       = D_K
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
    // Driver API expects col-major-ordered arrays: index 0 is the innermost
    // (fastest-changing) dimension. For S_in row-major (B_H*D_K, D_V) the
    // innermost dim is D_V; outer dim is B_H*D_K.
    cuuint64_t globalDim[2]    = { (cuuint64_t)D_V, (cuuint64_t)B_H * (cuuint64_t)D_K };
    cuuint64_t globalStrides[1] = { (cuuint64_t)D_V * sizeof(float) }; // stride of dim 1 in BYTES
    cuuint32_t boxDim[2]       = { (cuuint32_t)BLOCK_V, (cuuint32_t)D_K };
    cuuint32_t elemStrides[2]  = { 1u, 1u };

    // Inner-dim box bytes = BLOCK_V * 4. Must be multiple of 16 for swizzle::none.
    // For BLOCK_V=64 that's 256B — fine. Smaller BLOCK_V (≥ 4) also passes the
    // 16B-multiple gate at f32.
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
static void launch_gdn_tma(const GDNShape& sh,
                           const __half* dQ, const __half* dK, const __half* dV,
                           const __half* dA, const __half* dB,
                           const float*  dS_in, float* dS_out, __half* dO,
                           cudaStream_t stream)
{
    const int B_H = sh.batch * sh.n_heads;
    constexpr int VLANES = BLOCK_V / 4;
    constexpr int TPB = VLANES;
    dim3 grid(B_H, sh.d_v / BLOCK_V);
    dim3 block(TPB);

    // Build TMA descriptor (driver call — host-side, once per launch).
    // We build it on the stack and pass into kernel by value as a
    // __grid_constant__ parameter.
    TmaDesc desc{};
    build_tma_descriptor_S_in(desc, dS_in, B_H, D_K, sh.d_v, BLOCK_V);

    // Shared memory layout (must match kernel):
    //   smem_S_raw   : D_K * BLOCK_V floats           = D_K * BLOCK_V * 4 bytes
    //   smem_q       : D_K floats                     = D_K * 4
    //   smem_k       : D_K floats                     = D_K * 4
    //   bar          : 1 uint64_t                     = 8
    //   s_scalars    : 2 floats (alpha, beta)         = 8
    // Round up to 16B alignment to keep TMA dest 128B-aligned at offset 0.
    size_t smem_bytes =
        (size_t)D_K * BLOCK_V * sizeof(float) +
        2 * (size_t)D_K * sizeof(float) +
        sizeof(uint64_t) +
        2 * sizeof(float);
    smem_bytes = (smem_bytes + 15) & ~size_t(15);

    // For sm_120 the default per-block dynamic-smem max is 48 KB; opt in to
    // larger if we exceed.
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            (const void*)gdn_decode_tma_kernel<D_K, BLOCK_V>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes);
    }

    gdn_decode_tma_kernel<D_K, BLOCK_V><<<grid, block, smem_bytes, stream>>>(
        dQ, dK, dV, dA, dB, dS_out, dO, desc.map, B_H, sh.d_v);
}

static void launch_gdn_dispatch(const GDNShape& sh,
                                const __half* dQ, const __half* dK, const __half* dV,
                                const __half* dA, const __half* dB,
                                const float*  dS_in, float* dS_out, __half* dO,
                                cudaStream_t stream)
{
    // Wave 22.10 supported only the qwen3_next_decode shape (d_k=d_v=256).
    // Wave 22.15 adds the sweep shapes: tiny (64/64), small (128/128),
    // large (256/256, same template as qwen3_next_decode).
    //
    // BLOCK_V picker matches the cuTile-side `pick_block_v` policy: keep
    // the (D_K × BLOCK_V) f32 state tile ≤ 64 KB so it fits comfortably
    // in dynamic shared memory after the q/k/bar overhead. For D_K ≤ 256
    // BLOCK_V=64 is fine.
    //
    // 'wide' (D_K=512) is intentionally NOT instantiated here. The single-
    // tile TMA descriptor (boxDim ≤ 256) cannot cover a 512-row slab — see
    // bench_sweep.cu for the explicit skip + comment. Splitting D_K=512
    // into two TMAs would require a kernel-body refactor that this cell
    // is explicitly forbidden from making.
    if (sh.d_k == 64 && sh.d_v == 64) {
        launch_gdn_tma<64, 64>(sh, dQ, dK, dV, dA, dB, dS_in, dS_out, dO, stream);
    } else if (sh.d_k == 128 && sh.d_v == 128) {
        launch_gdn_tma<128, 64>(sh, dQ, dK, dV, dA, dB, dS_in, dS_out, dO, stream);
    } else if (sh.d_k == 256 && sh.d_v == 256) {
        launch_gdn_tma<256, 64>(sh, dQ, dK, dV, dA, dB, dS_in, dS_out, dO, stream);
    } else {
        fprintf(stderr, "unsupported (d_k=%d, d_v=%d)\n", sh.d_k, sh.d_v);
        exit(2);
    }
}

#ifndef ATTN_GDN_TMA_BENCH_HARNESS
// ============================================================================
// Correctness driver
// ============================================================================
static int run_correctness(const std::string& inputs_dir) {
    GDNShape sh{"correctness", 2, 4, 64, 64};
    printf("[gdn-tma] === correctness run (B=%d H=%d d_k=%d d_v=%d) ===\n",
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

    const double ATOL_O = 1e-3;   // same as W1c
    const double ATOL_S = 5e-3;
    bool ok_o = max_abs_o <= ATOL_O;
    bool ok_s = max_abs_s <= ATOL_S;

    printf("[gdn-tma] o    max_abs=%.3e   |want|max=%.3e   %s\n",
           max_abs_o, exp_o_mag, ok_o ? "OK" : "FAIL");
    printf("[gdn-tma] Sout max_abs=%.3e   |want|max=%.3e   %s\n",
           max_abs_s, exp_s_mag, ok_s ? "OK" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dA); cudaFree(dB);
    cudaFree(dSin); cudaFree(dSout); cudaFree(dO);
    return (ok_o && ok_s) ? 0 : 2;
}

static int run_bench_shape_smoke(const std::string& inputs_dir) {
    GDNShape sh{"qwen3_next_decode", 1, 16, 256, 256};
    printf("\n[gdn-tma] === bench-shape smoke (B=%d H=%d d_k=%d d_v=%d) ===\n",
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
    printf("[gdn-tma] (qwen3) o   max_abs=%.3e   %s\n", max_abs_o, ok_o ? "OK" : "FAIL");
    printf("[gdn-tma] (qwen3) Sout max_abs=%.3e %s\n", max_abs_s, ok_s ? "OK" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dA); cudaFree(dB);
    cudaFree(dSin); cudaFree(dSout); cudaFree(dO);
    return (ok_o && ok_s) ? 0 : 3;
}

int main(int argc, char** argv) {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("[gdn-tma] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
    if (p.major < 9) {
        fprintf(stderr, "[gdn-tma] needs sm_90+ (TMA / cp.async.bulk.tensor)\n");
        return 1;
    }

    std::string inputs_dir =
        "/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs";
    if (argc > 1) inputs_dir = argv[1];
    printf("[gdn-tma] inputs dir: %s\n", inputs_dir.c_str());

    int rc = run_correctness(inputs_dir);
    if (rc != 0) {
        fprintf(stderr, "[gdn-tma] correctness FAILED at correctness shape — stopping\n");
        return rc;
    }
    rc = run_bench_shape_smoke(inputs_dir);
    return rc;
}
#endif // ATTN_GDN_TMA_BENCH_HARNESS
