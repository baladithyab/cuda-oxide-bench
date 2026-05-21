// Wave C2.1 — cuda-matmul-tc-bf16: hand-WMMA bf16 matmul (CUDA C++).
//
// Closes the cuda-matmul TC column. Counterparts:
//   - cuda-matmul (naive FFMA)
//   - cuda-matmul-tiled (FFMA + register microtile)
//   - cublas-matmul / cublas-half-precision (cublas reference, ~219 TF bf16)
//   - mojo-matmul-bf16 (Mojo hand-MMA m16n8k16, ~79.85 TF)
//   - oxide-tcgen05-matmul (Rust + sm_120 tcgen05 PTX)
//
// Approach: nvcuda::wmma m16n16k16 BF16 → F32 fragments. CTA tile BM=BN=128,
// BK=32, 4 warps per CTA arranged 2x2 over a 128x128 output tile. Each warp
// owns a 64x64 sub-tile = 4x4 = 16 WMMA fragments. Per K-tile pass uses
// BK/16 = 2 wmma::mma_sync calls per fragment. cp.async stages A and B tiles
// from global -> shared so the first iteration can be issued concurrently
// with compute on the previous tile (single-buffered for simplicity here).
//
// FLOPS budget: 2 * 4096^3 ≈ 137.4 GFLOP per matmul. At RTX 5090 ~219 TF bf16
// (cublas), that's ~0.63 ms; expected hand-WMMA cell sits 50-150 TF.
//
// Build: /usr/local/cuda/bin/nvcc -ccbin clang-14 -O3 -arch=sm_120 ...
// ADR-0001: cudaEvent timing. ADR-0002: native sm_120, no PTX-JIT.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

// ---- compile-time tile geometry ----
constexpr int BM   = 128;
constexpr int BN   = 128;
constexpr int BK   = 32;
constexpr int WM   = 64;     // per-warp output tile rows
constexpr int WN   = 64;     // per-warp output tile cols
constexpr int MMA_M = 16;
constexpr int MMA_N = 16;
constexpr int MMA_K = 16;
constexpr int WARPS_M = BM / WM;          // 2
constexpr int WARPS_N = BN / WN;          // 2
constexpr int WARPS_PER_CTA = WARPS_M * WARPS_N; // 4
constexpr int THREADS_PER_CTA = WARPS_PER_CTA * 32; // 128
constexpr int FRAGS_M = WM / MMA_M;       // 4
constexpr int FRAGS_N = WN / MMA_N;       // 4
constexpr int FRAGS_K = BK / MMA_K;       // 2

#define WARMUPS 5
#define ITERS 50

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA err %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while(0)

// cp.async helpers (sm_80+ syntax, supported on sm_120 / Blackwell).
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem_int = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_int), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

// Each CTA computes a BM x BN tile of C. Shared layout:
//   As : [BM][BK]  bf16, row-major. Stride = BK.
//   Bs : [BK][BN]  bf16, row-major. Stride = BN.
// Loaders move 16-byte chunks (= 8 bf16 elements) per thread per pass.
__global__ __launch_bounds__(THREADS_PER_CTA, 2)
void matmul_tc_bf16(const __nv_bfloat16* __restrict__ A,
                    const __nv_bfloat16* __restrict__ B,
                    float* __restrict__ C, int N) {
    __shared__ __nv_bfloat16 As[BM][BK];
    __shared__ __nv_bfloat16 Bs[BK][BN];

    const int tid    = threadIdx.x;
    const int warp   = tid / 32;
    const int lane   = tid % 32;
    const int warp_y = warp / WARPS_N;        // 0..1
    const int warp_x = warp % WARPS_N;        // 0..1
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // Per-warp accumulator fragments.
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> cf[FRAGS_M][FRAGS_N];
    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i)
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j)
            wmma::fill_fragment(cf[i][j], 0.0f);

    // Per-tile-pass: A loads BM*BK = 128*32 = 4096 bf16 = 8192 B = 512 x 16B.
    // 128 threads * 4 chunks each = 512. Same math for B.
    // Each thread loads its 4 16-byte chunks for A and 4 for B.
    constexpr int CHUNK_BF16 = 8;        // 16B / 2B
    constexpr int A_CHUNKS_PER_TILE = (BM * BK) / CHUNK_BF16;     // 512
    constexpr int B_CHUNKS_PER_TILE = (BK * BN) / CHUNK_BF16;     // 512
    constexpr int A_CHUNKS_PER_THR  = A_CHUNKS_PER_TILE / THREADS_PER_CTA; // 4
    constexpr int B_CHUNKS_PER_THR  = B_CHUNKS_PER_TILE / THREADS_PER_CTA; // 4
    constexpr int A_CHUNKS_PER_ROW  = BK / CHUNK_BF16;            // 4
    constexpr int B_CHUNKS_PER_ROW  = BN / CHUNK_BF16;            // 16

    const int ntiles = N / BK;

    for (int t = 0; t < ntiles; ++t) {
        const int k0 = t * BK;

        // ---- stage A tile via cp.async ----
        #pragma unroll
        for (int c = 0; c < A_CHUNKS_PER_THR; ++c) {
            int chunk_idx = tid + c * THREADS_PER_CTA;          // 0..511
            int row = chunk_idx / A_CHUNKS_PER_ROW;             // 0..127
            int col_chunk = chunk_idx % A_CHUNKS_PER_ROW;       // 0..3
            int col = col_chunk * CHUNK_BF16;                   // 0..24 step 8
            const __nv_bfloat16* gptr = A + (size_t)(block_row + row) * N + (k0 + col);
            __nv_bfloat16* sptr = &As[row][col];
            cp_async_16(sptr, gptr);
        }
        // ---- stage B tile via cp.async ----
        #pragma unroll
        for (int c = 0; c < B_CHUNKS_PER_THR; ++c) {
            int chunk_idx = tid + c * THREADS_PER_CTA;          // 0..511
            int row = chunk_idx / B_CHUNKS_PER_ROW;             // 0..31
            int col_chunk = chunk_idx % B_CHUNKS_PER_ROW;       // 0..15
            int col = col_chunk * CHUNK_BF16;                   // 0..120 step 8
            const __nv_bfloat16* gptr = B + (size_t)(k0 + row) * N + (block_col + col);
            __nv_bfloat16* sptr = &Bs[row][col];
            cp_async_16(sptr, gptr);
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // ---- compute on staged tiles ----
        // Per-warp output tile starts at (warp_y*WM, warp_x*WN) within the CTA tile.
        wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, __nv_bfloat16, wmma::row_major> af[FRAGS_M][FRAGS_K];
        wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, __nv_bfloat16, wmma::row_major> bf[FRAGS_K][FRAGS_N];

        // Load A fragments (warp_y row band, all K).
        #pragma unroll
        for (int i = 0; i < FRAGS_M; ++i) {
            #pragma unroll
            for (int kk = 0; kk < FRAGS_K; ++kk) {
                const __nv_bfloat16* aptr =
                    &As[warp_y * WM + i * MMA_M][kk * MMA_K];
                wmma::load_matrix_sync(af[i][kk], aptr, BK);
            }
        }
        // Load B fragments (warp_x col band, all K).
        #pragma unroll
        for (int kk = 0; kk < FRAGS_K; ++kk) {
            #pragma unroll
            for (int j = 0; j < FRAGS_N; ++j) {
                const __nv_bfloat16* bptr =
                    &Bs[kk * MMA_K][warp_x * WN + j * MMA_N];
                wmma::load_matrix_sync(bf[kk][j], bptr, BN);
            }
        }
        // Accumulate.
        #pragma unroll
        for (int i = 0; i < FRAGS_M; ++i) {
            #pragma unroll
            for (int j = 0; j < FRAGS_N; ++j) {
                #pragma unroll
                for (int kk = 0; kk < FRAGS_K; ++kk) {
                    wmma::mma_sync(cf[i][j], af[i][kk], bf[kk][j], cf[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // ---- store per-warp 64x64 sub-tile ----
    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            float* cptr = C +
                (size_t)(block_row + warp_y * WM + i * MMA_M) * N +
                (block_col + warp_x * WN + j * MMA_N);
            wmma::store_matrix_sync(cptr, cf[i][j], N, wmma::mem_row_major);
        }
    }
}

// ----- bf16 helpers (host-side ref -> bf16 conversion) -----
static inline __nv_bfloat16 f2b(float f) {
    // round-to-nearest-even via the cuda host helper
    return __float2bfloat16(f);
}
static inline float b2f(__nv_bfloat16 b) {
    return __bfloat162float(b);
}

// CPU f32 reference at a single (row, col).
static double ref_elem(const float* hAf, const float* hBf, int n, int row, int col) {
    double acc = 0.0;
    for (int k = 0; k < n; ++k) {
        acc += (double)hAf[row * n + k] * (double)hBf[k * n + col];
    }
    return acc;
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("[cuda-tc-bf16] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    const int N = 4096;
    const size_t mat_elems = (size_t)N * N;
    const size_t bf16_bytes = mat_elems * sizeof(__nv_bfloat16);
    const size_t f32_bytes  = mat_elems * sizeof(float);
    double flops = 2.0 * (double)N * N * N;

    // Host buffers.
    float* hAf = (float*)malloc(f32_bytes);
    float* hBf = (float*)malloc(f32_bytes);
    float* hC  = (float*)malloc(f32_bytes);
    __nv_bfloat16* hAb = (__nv_bfloat16*)malloc(bf16_bytes);
    __nv_bfloat16* hBb = (__nv_bfloat16*)malloc(bf16_bytes);

    // Deterministic small inputs (matches cuda-matmul-tiled pattern but rescaled
    // so partial sums stay representable in bf16). Range ~[0, 0.06].
    for (size_t i = 0; i < mat_elems; ++i) {
        hAf[i] = ((i % 7) * 0.01f);
        hBf[i] = ((i % 11) * 0.01f);
        hAb[i] = f2b(hAf[i]);
        hBb[i] = f2b(hBf[i]);
    }

    __nv_bfloat16 *dA, *dB; float* dC;
    CK(cudaMalloc(&dA, bf16_bytes));
    CK(cudaMalloc(&dB, bf16_bytes));
    CK(cudaMalloc(&dC, f32_bytes));
    CK(cudaMemcpy(dA, hAb, bf16_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hBb, bf16_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemset(dC, 0, f32_bytes));

    cudaEvent_t evs, eve;
    CK(cudaEventCreate(&evs));
    CK(cudaEventCreate(&eve));

    dim3 block(THREADS_PER_CTA);
    dim3 grid(N / BN, N / BM);

    printf("[cuda-tc-bf16] N=%d BM=%d BN=%d BK=%d warps=%d threads=%d frags=%dx%dx%d grid=(%d,%d)\n",
           N, BM, BN, BK, WARPS_PER_CTA, THREADS_PER_CTA,
           FRAGS_M, FRAGS_N, FRAGS_K, grid.x, grid.y);

    // Warmup.
    for (int w = 0; w < WARMUPS; ++w) {
        matmul_tc_bf16<<<grid, block>>>(dA, dB, dC, N);
    }
    CK(cudaDeviceSynchronize());
    cudaError_t kerr = cudaGetLastError();
    if (kerr != cudaSuccess) {
        fprintf(stderr, "[cuda-tc-bf16] kernel launch error: %s\n", cudaGetErrorString(kerr));
        return 1;
    }

    // Bench.
    FILE* csv = fopen("results.csv", "w");
    if (csv) fprintf(csv, "impl,kernel,N,iter,gpu_ms,tflops\n");

    double times[ITERS];
    float ms;
    for (int i = 0; i < ITERS; ++i) {
        cudaEventRecord(evs);
        matmul_tc_bf16<<<grid, block>>>(dA, dB, dC, N);
        cudaEventRecord(eve);
        cudaEventSynchronize(eve);
        cudaEventElapsedTime(&ms, evs, eve);
        times[i] = ms;
        double tflops = (flops / 1e12) / (ms / 1000.0);
        printf("[cuda-tc-bf16] N=%d iter=%d gpu_ms=%.3f tflops=%.3f\n", N, i, ms, tflops);
        if (csv) fprintf(csv, "cuda-tc-bf16,matmul_tc_bf16,%d,%d,%.6f,%.6f\n", N, i, ms, tflops);
    }

    // Pull C back for correctness sampling.
    CK(cudaMemcpy(hC, dC, f32_bytes, cudaMemcpyDeviceToHost));

    // Correctness sweep — 1024 sampled (row,col) cells vs CPU f32 ref.
    // Stride sample positions evenly across the 4096x4096 matrix.
    const int SAMPLES = 1024;
    const float atol = 2e-1f;
    const float rtol = 5e-2f;
    int ok = 0, bad = 0;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int worst_r = 0, worst_c = 0;
    float worst_got = 0, worst_want = 0;
    // Pseudorandom-ish but reproducible sample coords.
    uint64_t seed = 0x9E3779B97F4A7C15ull;
    for (int s = 0; s < SAMPLES; ++s) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        int r = (int)(seed % (uint64_t)N);
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        int c = (int)(seed % (uint64_t)N);
        float got = hC[r * N + c];
        double want_d = ref_elem(hAf, hBf, N, r, c);
        float want = (float)want_d;
        float aerr = fabsf(got - want);
        float rerr = aerr / fmaxf(fabsf(want), 1e-6f);
        if (aerr > max_abs_err) {
            max_abs_err = aerr;
            max_rel_err = rerr;
            worst_r = r; worst_c = c;
            worst_got = got; worst_want = want;
        }
        bool pass = (aerr <= atol) || (rerr <= rtol);
        if (pass) ok++; else bad++;
    }
    printf("[cuda-tc-bf16] correctness: %d/%d OK (atol=%.2e rtol=%.2e)\n",
           ok, SAMPLES, atol, rtol);
    printf("[cuda-tc-bf16] worst cell (%d,%d): got=%.4f want=%.4f abs_err=%.4f rel_err=%.4e\n",
           worst_r, worst_c, worst_got, worst_want, max_abs_err, max_rel_err);

    // Stats.
    double sorted[ITERS];
    for (int i = 0; i < ITERS; ++i) sorted[i] = times[i];
    for (int i = 0; i < ITERS; ++i)
        for (int j = i + 1; j < ITERS; ++j)
            if (sorted[j] < sorted[i]) {
                double t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t;
            }
    double bst = sorted[0];
    double med = 0.5 * (sorted[ITERS/2 - 1] + sorted[ITERS/2]);
    double best_tf   = (flops / 1e12) / (bst / 1000.0);
    double median_tf = (flops / 1e12) / (med / 1000.0);

    printf("\n[cuda-tc-bf16] ===== SUMMARY =====\n");
    printf("[cuda-tc-bf16] %6s  %10s  %10s  %10s  %10s\n", "N", "best_ms", "median_ms", "best_TF", "median_TF");
    printf("[cuda-tc-bf16] %6d  %10.3f  %10.3f  %10.3f  %10.3f\n",
           N, bst, med, best_tf, median_tf);

    if (csv) fclose(csv);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hAf); free(hBf); free(hC); free(hAb); free(hBb);

    // Acceptance gate (informational; CI/orchestrator parses):
    bool tflops_ok  = (best_tf >= 50.0 && best_tf <= 220.0);
    bool correct_ok = (max_abs_err <= 2e-1f) || (max_rel_err <= 5e-2f);
    printf("[cuda-tc-bf16] gate: tflops_in_50_220=%d correctness_pass=%d\n",
           (int)tflops_ok, (int)correct_ok);

    return (tflops_ok && correct_ok) ? 0 : 2;
}
