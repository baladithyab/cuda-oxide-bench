// Wave 23.2 — cuda-attn-mla-tma:
// Apply the W22.10 TMA recipe (cp.async.bulk.tensor.2d via cuTensorMapEncodeTiled)
// to MLA's 3-kernel-decomposed attention.
//
// Mechanism difference vs W22.10 (GDN):
//   GDN was memory-bound, single-pass state recurrence; TMA replaced 16 LDG.E.128
//   per thread with one UTMALDG per block. That gave +69% (1032 GB/s vs 610 GB/s
//   cuTile baseline) — the win was on per-thread address computation in a
//   state-traffic loop.
//
//   MLA is decomposed:
//     1) Q @ K^T   (matmul, WMMA)
//     2) softmax    (memory-bound)
//     3) P @ V     (matmul, WMMA)
//   The matmul stages are arithmetic-bound when shapes are large; per-thread
//   gmem-address pressure is small. So TMA's value here is to act as a bulk
//   loader for tiles into shared memory before the WMMA fragments read them
//   via load_matrix_sync. We expect a mild correctness-equivalent codegen
//   change (UTMALDG > 0, LDG.E count drops on Q/K/V/P paths, HMMA unchanged
//   because the MMA shape is the same).
//
// Layout-compatibility risk (the Wave-23.2 stop condition):
//   wmma::load_matrix_sync expects f16 sources with a leading-dim and a
//   row-major / col-major flag. The TMA descriptor produces row-major tiles
//   (innermost dim contiguous, no swizzle). For Q (matrix_a, row_major) and
//   for V (matrix_b, row_major) and for P (matrix_a, row_major) this is a
//   perfect match. For K (matrix_b, col_major in the existing kernel) we
//   index K as `Kh + col0*QK + k` with ld=QK — that IS the row-major layout
//   of K viewed transposed; load_matrix_sync with col_major handles the
//   transpose internally. So a row-major TMA tile of K's rows works as the
//   col_major fragment_b source.
//
//   The 16-byte alignment requirement on smem destination, and the "innermost
//   bytes multiple of 16 with swizzle::none" requirement on the descriptor:
//     • Q-tile: 16 rows × QK_eff cols, ld=QK_eff. QK_eff in {96, 128, 192, 256}.
//       96*2 = 192 B (ok, mult of 16). 128*2 = 256 B (ok). All cases pass.
//     • V-tile: 16 rows × Dv cols. Dv in {64, 128}. 64*2=128 B (ok), 128*2=256 (ok).
//     • P-tile: same as Q with cols=S in {128, 2048}. 128*2=256 (ok), 2048*2=4096 (ok).
//   Smem destination is dynamic shared, placed first → 16-byte aligned by
//   `extern __shared__ __align__(16)` and the 128B alignment that the TMA
//   PTX requires is achieved by placing tiles at offsets that are multiples
//   of 128 bytes (we lay out smem deliberately).
//
// Acceptance (W23.2):
//   • kernel compiles
//   • correctness PASS (max_abs_err <= 1e-2 on the W17 correctness shape:
//     B=1, n_h=4, S=128, qk=96, d_v=64 — this is the available `correctness_mla`
//     tensor on disk; the task brief listed qk=64 but the on-disk tensors are
//     qk=96, so we use what's available; tolerance is the same 1e-2 W17 used).
//   • SASS shows UTMALDG > 0 AND HMMA > 0
//   • NO timed bench (W23.2 is author + correctness only)
//
// Build:
//   /usr/local/cuda/bin/nvcc -ccbin clang-14 -O3 -arch=sm_120 -std=c++17 \
//       -lineinfo -lstdc++ -lm -lcuda -o attn_mla_tma attn_mla_tma.cu

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
#include <cuda.h>            // CUtensorMap, cuTensorMapEncodeTiled
#include <mma.h>

using namespace nvcuda;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA err %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while(0)

#define CKD(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
    const char* msg = nullptr; cuGetErrorString(r, &msg); \
    fprintf(stderr, "CU driver err %s @ %s:%d\n", msg ? msg : "(?)", __FILE__, __LINE__); exit(1); } } while(0)

constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;

// ============================================================================
// .npy loader (NPY1.0/2.0, little-endian, C-contiguous f16/f32).
// Lifted line-for-line from cuda-attn-mla — same harness contract.
// ============================================================================
struct Npy {
    std::vector<int64_t> shape;
    std::string dtype;
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
    uint8_t ver[2]; fread(ver, 1, 2, f);
    uint32_t header_len;
    if (ver[0] == 1) { uint16_t hl; fread(&hl, 1, 2, f); header_len = hl; }
    else { fread(&header_len, 1, 4, f); }
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
// Device-side TMA / mbarrier helpers (inline PTX). Identical to W22.10.
// All instructions gated on __CUDA_ARCH__ >= 900 (sm_120 = 1200, OK).
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
        "WAIT_LOOP_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
        "@p bra DONE_%=;\n\t"
        "bra WAIT_LOOP_%=;\n\t"
        "DONE_%=:\n\t"
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

// ============================================================================
// Kernel 1 (TMA): QKt
//
// Per output tile (16, 16) in (i, j), one warp:
//   • TMA-load Q's 16-row × QK_eff slab into smem_Q  (16-byte-aligned, row-major)
//   • TMA-load K's 16-row × QK_eff slab into smem_K  (row-major; col_major for fragment_b)
//   • Walk K-loop k=0..QK_eff step 16, load_matrix_sync from smem
//   • Scale and store with mem_row_major
//
// Smem layout (one warp per block, 32 threads):
//   smem_Q : 16 * QK_max  __half, 128B-aligned at offset 0
//   smem_K : 16 * QK_max  __half, 128B-aligned
//   bar    : 8 B uint64_t, padded to 16
//
// TMA descriptor coords (Q):
//   Q gmem viewed as 2D (B*Nh*S, QK_eff) row-major
//   innermost dim = QK_eff (cols)
//   tile origin = (col=0, row=(bh*S + row0))
//   box = (QK_eff cols, 16 rows)
// Same for K.
// ============================================================================
__global__ void __launch_bounds__(32) mla_qkt_tma_kernel(
    float*        __restrict__ Sm,                    // [B*Nh*S, S] f32
    const __grid_constant__ CUtensorMap Q_tmap,
    const __grid_constant__ CUtensorMap K_tmap,
    int B, int Nh, int S, int QK_eff,
    float scale)
{
    int tile_i = blockIdx.x;
    int tile_j = blockIdx.y;
    int bh    = blockIdx.z;
    int row0  = tile_i * WM;
    int col0  = tile_j * WN;
    if (row0 >= S || col0 >= S) return;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    // Place TMA destinations first, both 128B-aligned.
    __half*   smem_Q = reinterpret_cast<__half*>(smem_raw);                 // 16 × QK_eff
    // Pad each tile to 128B boundary. 16 × QK_eff * 2 bytes; for QK_eff=96 that's 3072 B (mult of 128).
    // For QK_eff=128 it's 4096. Both are multiples of 128 — no padding needed at this offset.
    size_t Q_bytes = (size_t)16 * QK_eff * sizeof(__half);
    __half*   smem_K = reinterpret_cast<__half*>(smem_raw + Q_bytes);
    size_t K_bytes = (size_t)16 * QK_eff * sizeof(__half);
    // bar: pad to 16-byte boundary after K.
    size_t bar_off = ((Q_bytes + K_bytes) + 15) & ~size_t(15);
    uint64_t* bar = reinterpret_cast<uint64_t*>(smem_raw + bar_off);

    int tid = threadIdx.x;
    if (tid == 0) {
        mbarrier_init(bar, /*arrive_count=*/1);
    }
    __syncwarp();

    // Issue two TMA loads. We need both to be tracked by the same barrier
    // (different barriers would force two waits). The PTX expect_tx pattern
    // accumulates: arrive_expect_tx(N1) and the TMA injects N1 bytes; we
    // call arrive_expect_tx with the SUM of both tile byte counts and issue
    // both TMAs — they each contribute their own byte count.
    constexpr uint32_t /*placeholder*/ _unused_ = 0;
    uint32_t total_bytes = (uint32_t)Q_bytes + (uint32_t)K_bytes;
    if (tid == 0) {
        mbarrier_arrive_expect_tx(bar, total_bytes);
        // Q tile: rows row0..row0+15 of the (bh,*,*) Q matrix. Innermost dim
        // is QK_eff; outer flat row index = bh*S + row0. coord_x=0 (col), coord_y=outer.
        cp_async_bulk_tensor_2d_g2s(
            smem_Q, &Q_tmap,
            /*coord_x=*/0,
            /*coord_y=*/bh * S + row0,
            bar);
        // K tile: rows col0..col0+15 of the (bh,*,*) K matrix (we use K as B with col_major,
        // i.e. we want K's rows at col0..col0+16 viewed as fragment_b cols).
        cp_async_bulk_tensor_2d_g2s(
            smem_K, &K_tmap,
            /*coord_x=*/0,
            /*coord_y=*/bh * S + col0,
            bar);
    }
    mbarrier_wait_parity(bar, /*parity=*/0);
    __syncwarp();

    // Now do the WMMA over smem.
    wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> cf;
    wmma::fill_fragment(cf, 0.0f);

    // K-loop: for each k in [0, QK_eff) step 16, load (16,16) slabs from smem.
    // smem_Q is 16×QK_eff row-major, ld=QK_eff. The slab at column-offset k
    // is &smem_Q[0*QK_eff + k] with the same ld.
    // smem_K is 16×QK_eff row-major. With col_major fragment_b semantics,
    // load_matrix_sync interprets the source as (k_dim × n_dim) col-major.
    // The SAME memory pattern that the original kernel used (Bptr = Kh + col0*QK + k
    // with ld=QK and col_major) is preserved: smem_K row r corresponds to
    // K-row col0+r, and the K-loop reads all 16 K-rows × WK cols at offset k.
    for (int k = 0; k < QK_eff; k += WK) {
        const __half* Aptr = smem_Q + k;            // (16 × WK) at column offset k, ld=QK_eff
        const __half* Bptr = smem_K + k;            // (16 × WK) at column offset k, ld=QK_eff
        wmma::load_matrix_sync(af, Aptr, QK_eff);
        wmma::load_matrix_sync(bf, Bptr, QK_eff);
        wmma::mma_sync(cf, af, bf, cf);
    }

    #pragma unroll
    for (int t = 0; t < cf.num_elements; ++t) cf.x[t] *= scale;

    float* Sh = Sm + (size_t)bh * S * S;
    wmma::store_matrix_sync(Sh + (size_t)row0 * S + col0, cf, S, wmma::mem_row_major);

    if (tid == 0) {
        mbarrier_inval(bar);
    }
}

// ============================================================================
// Kernel 2: softmax — UNCHANGED from cuda-attn-mla. TMA doesn't help here:
// it's a per-row reduction with one block per row, no large gmem tile to
// hoist. Keep it identical so any correctness regression isolates to the
// TMA-touched matmul kernels.
// ============================================================================
constexpr int SOFTMAX_TPB = 128;

__global__ void softmax_kernel(
    const float* __restrict__ Sm,
    __half*      __restrict__ P,
    int /*B*/, int /*Nh*/, int S)
{
    int row = blockIdx.x;
    int bh  = blockIdx.y;
    int tid = threadIdx.x;

    const float* Srow = Sm + (size_t)bh * S * S + (size_t)row * S;
    __half*      Prow = P  + (size_t)bh * S * S + (size_t)row * S;

    __shared__ float sbuf[SOFTMAX_TPB];

    float lmax = -INFINITY;
    for (int j = tid; j < S; j += SOFTMAX_TPB) {
        float v = Srow[j];
        if (v > lmax) lmax = v;
    }
    sbuf[tid] = lmax;
    __syncthreads();
    for (int stride = SOFTMAX_TPB / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = sbuf[tid], b = sbuf[tid + stride];
            sbuf[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    float rmax = sbuf[0];

    float lsum = 0.0f;
    for (int j = tid; j < S; j += SOFTMAX_TPB) {
        lsum += expf(Srow[j] - rmax);
    }
    sbuf[tid] = lsum;
    __syncthreads();
    for (int stride = SOFTMAX_TPB / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sbuf[tid] += sbuf[tid + stride];
        __syncthreads();
    }
    float rsum = sbuf[0];
    float inv_sum = 1.0f / rsum;

    for (int j = tid; j < S; j += SOFTMAX_TPB) {
        float p = expf(Srow[j] - rmax) * inv_sum;
        Prow[j] = __float2half(p);
    }
}

// ============================================================================
// Kernel 3 (TMA): PV
//
// Per output tile (16, 16) in (i, j) over (S, Dv), one warp:
//   • TMA-load P's 16-row × S slab into smem_P (row-major matrix_a)
//   • TMA-load V's S-row × 16-col slab is awkward because TMA inner dim must be the
//     fastest-changing global dim. V is row-major (S, Dv); innermost = Dv.
//     A box of (16-cols × S-rows) would have inner-box=16 cols; that works.
//     But coordinates in TMA tile load: (coord_x=col0, coord_y=0) loads box
//     of (16 cols, S rows). innermost-bytes = 16*2 = 32 (mult of 16). OK.
//   • Walk K-loop k=0..S step 16, load_matrix_sync from smem
//
// We use ONE V-descriptor per launch (whose box is (16 cols, S rows)), shared
// across all (i,j) tiles. All blocks with same tile_j load the same V slab —
// a gmem-side reuse opportunity that the L2 cache will exploit.
// ============================================================================
__global__ void __launch_bounds__(32) mla_pv_tma_kernel(
    __half*       __restrict__ O,                    // [B*Nh*S, Dv]
    const __grid_constant__ CUtensorMap P_tmap,
    const __grid_constant__ CUtensorMap V_tmap,
    int B, int Nh, int S, int Dv)
{
    int tile_i = blockIdx.x;
    int tile_j = blockIdx.y;
    int bh    = blockIdx.z;
    int row0  = tile_i * WM;
    int col0  = tile_j * WN;
    if (row0 >= S || col0 >= Dv) return;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* smem_P = reinterpret_cast<__half*>(smem_raw);                   // 16 × S
    size_t  P_bytes = (size_t)16 * S * sizeof(__half);
    __half* smem_V = reinterpret_cast<__half*>(smem_raw + P_bytes);         // S × 16
    size_t  V_bytes = (size_t)S * 16 * sizeof(__half);
    size_t  bar_off = ((P_bytes + V_bytes) + 15) & ~size_t(15);
    uint64_t* bar = reinterpret_cast<uint64_t*>(smem_raw + bar_off);

    int tid = threadIdx.x;
    if (tid == 0) {
        mbarrier_init(bar, 1);
    }
    __syncwarp();

    uint32_t total_bytes = (uint32_t)P_bytes + (uint32_t)V_bytes;
    if (tid == 0) {
        mbarrier_arrive_expect_tx(bar, total_bytes);
        // P tile: rows row0..row0+15 of the (bh, *, *) P matrix. innermost is S (cols);
        // outer flat row index = bh*S + row0.
        cp_async_bulk_tensor_2d_g2s(
            smem_P, &P_tmap,
            /*coord_x=*/0,
            /*coord_y=*/bh * S + row0,
            bar);
        // V tile: cols col0..col0+15 of the (bh, *, *) V matrix. innermost is Dv (cols).
        // box=(16 cols, S rows). coord_x=col0, coord_y=bh*S.
        cp_async_bulk_tensor_2d_g2s(
            smem_V, &V_tmap,
            /*coord_x=*/col0,
            /*coord_y=*/bh * S,
            bar);
    }
    mbarrier_wait_parity(bar, 0);
    __syncwarp();

    // smem_P is row-major (16, S), ld=S.
    // smem_V is row-major (S, 16), ld=16.
    wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> cf;
    wmma::fill_fragment(cf, 0.0f);

    for (int k = 0; k < S; k += WK) {
        const __half* Aptr = smem_P + k;             // (16 × WK), ld=S
        const __half* Bptr = smem_V + (size_t)k * 16; // (WK × 16), ld=16
        wmma::load_matrix_sync(af, Aptr, S);
        wmma::load_matrix_sync(bf, Bptr, 16);
        wmma::mma_sync(cf, af, bf, cf);
    }

    // Cast accumulator → f16 and write back. Use a small staging area in
    // shared memory to do the row-major store.
    __shared__ float stage[WM * WN];
    wmma::store_matrix_sync(stage, cf, WN, wmma::mem_row_major);
    int lane = threadIdx.x;
    __half* Oh = O + (size_t)bh * S * Dv;
    #pragma unroll
    for (int e = lane; e < WM * WN; e += 32) {
        int r = e / WN;
        int c = e % WN;
        Oh[(size_t)(row0 + r) * Dv + (col0 + c)] = __float2half(stage[e]);
    }

    if (tid == 0) {
        mbarrier_inval(bar);
    }
}

// ============================================================================
// Host-side TMA descriptor builders.
//
// All matrices are row-major in gmem. cuTensorMapEncodeTiled wants
// dimensions listed innermost-first.
//
// For Q (and K, P) viewed as 2D (B*Nh*S, QK_eff) row-major:
//   globalDim    = { QK_eff (innermost), B*Nh*S (outer) }
//   globalStrides[0] = QK_eff * 2 bytes   (stride of dim 1)
//   boxDim       = { QK_eff, 16 }
//
// For V viewed as 2D (B*Nh*S, Dv) row-major, we want a box of (16 cols, S rows)
// per (bh) — but TMA descriptor box is fixed at encode time. We can either
// build one descriptor per (bh) or use a single descriptor with a small box
// and let the kernel pick coord_y = bh*S as the per-bh row offset:
//   globalDim    = { Dv, B*Nh*S }
//   globalStrides[0] = Dv * 2 bytes
//   boxDim       = { 16, S }
// ============================================================================
static void build_tma_2d_f16(
    CUtensorMap* out,
    const __half* d_base,
    int outer_size,           // outer dim: B*Nh*S
    int inner_size,           // innermost dim
    int box_inner,            // box innermost size (cols)
    int box_outer)            // box outer size (rows)
{
    cuuint64_t globalDim[2]    = { (cuuint64_t)inner_size, (cuuint64_t)outer_size };
    cuuint64_t globalStrides[1] = { (cuuint64_t)inner_size * sizeof(__half) };
    cuuint32_t boxDim[2]       = { (cuuint32_t)box_inner, (cuuint32_t)box_outer };
    cuuint32_t elemStrides[2]  = { 1u, 1u };

    CUresult r = cuTensorMapEncodeTiled(
        out,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        /*tensorRank=*/2,
        const_cast<void*>(reinterpret_cast<const void*>(d_base)),
        globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        const char* msg = nullptr; cuGetErrorString(r, &msg);
        fprintf(stderr, "cuTensorMapEncodeTiled failed (inner=%d outer=%d boxI=%d boxO=%d): %s\n",
                inner_size, outer_size, box_inner, box_outer, msg ? msg : "?");
        exit(1);
    }
}

// ============================================================================
// Host driver
// ============================================================================
struct MLAShape {
    const char* name;
    int B, Nh, S, QK, Dv;
};

static int pad_qk(int qk, bool pad_to_pow2) {
    if (!pad_to_pow2) {
        if (qk % WK != 0) { fprintf(stderr, "qk=%d not multiple of %d\n", qk, WK); exit(1); }
        return qk;
    }
    int p = 1;
    while (p < qk) p *= 2;
    return p;
}

static void run_pipeline(const MLAShape& sh, int QK_eff,
                         const __half* dQ, const __half* dK, const __half* dV,
                         float* dS_scores, __half* dP, __half* dO)
{
    float scale = 1.0f / sqrtf((float)sh.QK);
    int B_H = sh.B * sh.Nh;

    // Build 4 descriptors: Q, K, P, V.
    // Q descriptor: outer=B*Nh*S rows, inner=QK_eff cols, box=(QK_eff, 16).
    CUtensorMap Q_tmap{}, K_tmap{}, P_tmap{}, V_tmap{};
    build_tma_2d_f16(&Q_tmap, dQ, B_H * sh.S, QK_eff, /*box_inner=*/QK_eff, /*box_outer=*/16);
    build_tma_2d_f16(&K_tmap, dK, B_H * sh.S, QK_eff, /*box_inner=*/QK_eff, /*box_outer=*/16);
    build_tma_2d_f16(&P_tmap, dP, B_H * sh.S, sh.S,    /*box_inner=*/sh.S,   /*box_outer=*/16);
    build_tma_2d_f16(&V_tmap, dV, B_H * sh.S, sh.Dv,   /*box_inner=*/16,     /*box_outer=*/sh.S);

    // ---- Kernel 1: QKt ----
    {
        dim3 grid(sh.S / WM, sh.S / WN, B_H);
        dim3 block(32);
        // smem: smem_Q (16*QK_eff*2) + smem_K (16*QK_eff*2) + bar pad-to-16
        size_t Q_bytes = (size_t)16 * QK_eff * sizeof(__half);
        size_t K_bytes = (size_t)16 * QK_eff * sizeof(__half);
        size_t smem_bytes = ((Q_bytes + K_bytes) + 15) & ~size_t(15);
        smem_bytes += 16; // bar
        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute((const void*)mla_qkt_tma_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        }
        mla_qkt_tma_kernel<<<grid, block, smem_bytes>>>(
            dS_scores, Q_tmap, K_tmap, sh.B, sh.Nh, sh.S, QK_eff, scale);
    }

    // ---- Kernel 2: softmax (unchanged) ----
    {
        dim3 grid(sh.S, B_H);
        dim3 block(SOFTMAX_TPB);
        softmax_kernel<<<grid, block>>>(dS_scores, dP, sh.B, sh.Nh, sh.S);
    }

    // ---- Kernel 3: PV ----
    {
        dim3 grid(sh.S / WM, sh.Dv / WN, B_H);
        dim3 block(32);
        size_t P_bytes = (size_t)16 * sh.S * sizeof(__half);
        size_t V_bytes = (size_t)sh.S * 16 * sizeof(__half);
        size_t smem_bytes = ((P_bytes + V_bytes) + 15) & ~size_t(15);
        smem_bytes += 16; // bar
        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute((const void*)mla_pv_tma_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        }
        mla_pv_tma_kernel<<<grid, block, smem_bytes>>>(
            dO, P_tmap, V_tmap, sh.B, sh.Nh, sh.S, sh.Dv);
    }
}

static __half* alloc_padded_qk(const __half* host, const MLAShape& sh, int QK_eff) {
    size_t nelem = (size_t)sh.B * sh.Nh * sh.S * QK_eff;
    __half* d;
    CK(cudaMalloc(&d, nelem * sizeof(__half)));
    if (QK_eff == sh.QK) {
        CK(cudaMemcpy(d, host, nelem * sizeof(__half), cudaMemcpyHostToDevice));
    } else {
        std::vector<__half> staged(nelem, __float2half(0.0f));
        for (size_t bh = 0; bh < (size_t)sh.B * sh.Nh; ++bh) {
            for (int s = 0; s < sh.S; ++s) {
                size_t src = (bh * sh.S + s) * sh.QK;
                size_t dst = (bh * sh.S + s) * QK_eff;
                memcpy(&staged[dst], &host[src], sh.QK * sizeof(__half));
            }
        }
        CK(cudaMemcpy(d, staged.data(), nelem * sizeof(__half), cudaMemcpyHostToDevice));
    }
    return d;
}

static double run_correctness_variant(const MLAShape& sh, bool padded,
                                      const std::string& inputs_dir,
                                      const char* shape_prefix)
{
    int QK_eff = pad_qk(sh.QK, padded);

    Npy q, k, v, exp_out;
    std::string base = inputs_dir + "/mla_" + shape_prefix;
    if (!read_npy(base + "_q_f16.npy", q)) exit(1);
    if (!read_npy(base + "_k_f16.npy", k)) exit(1);
    if (!read_npy(base + "_v_f16.npy", v)) exit(1);
    if (!read_npy(base + "_expected_f32.npy", exp_out)) exit(1);

    size_t qk_native = (size_t)sh.B * sh.Nh * sh.S * sh.QK;
    size_t v_elems   = (size_t)sh.B * sh.Nh * sh.S * sh.Dv;
    size_t s_elems   = (size_t)sh.B * sh.Nh * sh.S * sh.S;
    if (q.data.size() != qk_native * 2 || k.data.size() != qk_native * 2 ||
        v.data.size() != v_elems * 2) {
        fprintf(stderr, "shape mismatch on %s (qk_native=%zu got_q=%zu)\n",
                shape_prefix, qk_native * 2, q.data.size());
        exit(1);
    }

    const __half* hq = reinterpret_cast<const __half*>(q.data.data());
    const __half* hk = reinterpret_cast<const __half*>(k.data.data());

    __half* dQ = alloc_padded_qk(hq, sh, QK_eff);
    __half* dK = alloc_padded_qk(hk, sh, QK_eff);
    __half* dV;
    CK(cudaMalloc(&dV, v_elems * sizeof(__half)));
    CK(cudaMemcpy(dV, v.data.data(), v_elems * sizeof(__half), cudaMemcpyHostToDevice));

    float*  dS;
    __half *dP, *dO;
    CK(cudaMalloc(&dS, s_elems * sizeof(float)));
    CK(cudaMalloc(&dP, s_elems * sizeof(__half)));
    CK(cudaMalloc(&dO, v_elems * sizeof(__half)));
    CK(cudaMemset(dO, 0, v_elems * sizeof(__half)));

    run_pipeline(sh, QK_eff, dQ, dK, dV, dS, dP, dO);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<__half> ho(v_elems);
    CK(cudaMemcpy(ho.data(), dO, v_elems * sizeof(__half), cudaMemcpyDeviceToHost));

    const float* exp_ptr = reinterpret_cast<const float*>(exp_out.data.data());
    double max_abs = 0.0, max_rel = 0.0, exp_max_abs = 0.0;
    for (size_t i = 0; i < v_elems; ++i) {
        float got = __half2float(ho[i]);
        float want = exp_ptr[i];
        double a = fabs((double)got - (double)want);
        if (a > max_abs) max_abs = a;
        double aw = fabs((double)want);
        if (aw > exp_max_abs) exp_max_abs = aw;
        double denom = aw < 1e-6 ? 1e-6 : aw;
        double r = a / denom;
        if (r > max_rel) max_rel = r;
    }

    printf("[mla-tma] %-9s qk=%d (qk_eff=%d, %s) -> max_abs_err=%.3e max_rel=%.3e expected_max_abs=%.3e\n",
           shape_prefix, sh.QK, QK_eff, padded ? "padded" : "native",
           max_abs, max_rel, exp_max_abs);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV);
    cudaFree(dS); cudaFree(dP); cudaFree(dO);
    return max_abs;
}

static int run_correctness(const std::string& inputs_dir) {
    // Use the same correctness shape as cuda-attn-mla (W17 W1a). Task brief
    // requested qk=64, but the on-disk `correctness_mla` tensors are qk=96.
    // Tolerance 1e-2 matches W17 W1a acceptance row.
    MLAShape sh{"correctness_mla", 1, 4, 128, 96, 64};
    printf("[mla-tma] === correctness run (B=%d n_h=%d S=%d qk=%d d_v=%d) ===\n",
           sh.B, sh.Nh, sh.S, sh.QK, sh.Dv);

    const double TOL = 1e-2;
    double err_native = run_correctness_variant(sh, /*padded=*/false, inputs_dir, "correctness_mla");
    double err_padded = run_correctness_variant(sh, /*padded=*/true,  inputs_dir, "correctness_mla");

    bool ok_native = (err_native <= TOL);
    bool ok_padded = (err_padded <= TOL);
    printf("[mla-tma] correctness: native=%.3e (%s)  padded=%.3e (%s)  TOL=%.0e\n",
           err_native, ok_native ? "OK" : "FAIL",
           err_padded, ok_padded ? "OK" : "FAIL", TOL);
    return (ok_native && ok_padded) ? 0 : 2;
}

int main(int argc, char** argv) {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("[mla-tma] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
    if (p.major < 9) {
        fprintf(stderr, "[mla-tma] needs sm_90+ (TMA / cp.async.bulk.tensor)\n");
        return 1;
    }

    std::string inputs_dir =
        "/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs";
    if (argc > 1) inputs_dir = argv[1];
    printf("[mla-tma] inputs dir: %s\n", inputs_dir.c_str());

    int rc = run_correctness(inputs_dir);
    if (rc != 0) {
        fprintf(stderr, "[mla-tma] correctness failed\n");
        return rc;
    }
    printf("[mla-tma] author + correctness only (W23.2). No timed bench.\n");
    return 0;
}
