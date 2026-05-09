// Wave 2 W2B: tiled CUDA C++ SGEMM with register tiling (sm_120 native).
// Each block computes a BM x BN (32x32) output tile; K is swept in tiles of BK (16).
// Each thread computes a TM x TN (4x4) micro-tile in registers, so block = 8x8 threads.
// A_s, B_s are staged in __shared__ with one __syncthreads after load and one after compute.
// Same input pattern / N sweep / timing protocol as cuda-matmul (W1B).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define MAXN 4096
#define BM 32
#define BN 32
#define BK 16
#define TM 4
#define TN 4
// Threads per block: (BM/TM) x (BN/TN) = 8 x 8 = 64.
#define WARMUPS 1
#define ITERS 10

__global__ void matmul_tiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C, int dim) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tx = threadIdx.x;         // 0..7 (column of micro-tile)
    const int ty = threadIdx.y;         // 0..7 (row of micro-tile)
    const int tid = ty * (BN / TN) + tx; // 0..63 flat id
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    // Each block loads BM*BK = 512 As elements and BK*BN = 512 Bs elements per K-tile.
    // 64 threads -> 8 loads each per array.
    const int ntiles = dim / BK;
    for (int t = 0; t < ntiles; ++t) {
        // Load As: 512 elements / 64 threads = 8 per thread.
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = tid + i * 64;           // 0..511
            int r = idx / BK;                 // 0..31 (row within As)
            int c = idx % BK;                 // 0..15 (col within As)
            As[r][c] = A[(block_row + r) * dim + (t * BK + c)];
        }
        // Load Bs: 512 elements / 64 threads = 8 per thread.
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = tid + i * 64;           // 0..511
            int r = idx / BN;                 // 0..15 (row within Bs)
            int c = idx % BN;                 // 0..31 (col within Bs)
            Bs[r][c] = B[(t * BK + r) * dim + (block_col + c)];
        }
        __syncthreads();

        // Compute: each thread does TM*TN = 16 FMAs per k.
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_reg[TM];
            float b_reg[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a_reg[i] = As[ty * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b_reg[j] = Bs[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads();
    }

    // Write back TM x TN micro-tile.
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            C[(block_row + ty * TM + i) * dim + (block_col + tx * TN + j)] = acc[i][j];
        }
    }
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); exit(1); } } while(0)

static float ref_elem(const float* hA, const float* hB, int n, int row, int col) {
    double acc = 0.0;
    for (int k = 0; k < n; ++k) {
        acc += (double)hA[row * n + k] * (double)hB[k * n + col];
    }
    return (float)acc;
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("[cuda-tiled] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    const int NS[3] = {1024, 2048, 4096};
    const size_t max_bytes = (size_t)MAXN * MAXN * sizeof(float);

    float *hA = (float*)malloc(max_bytes);
    float *hB = (float*)malloc(max_bytes);
    float *hC = (float*)malloc(max_bytes);

    float *dA, *dB, *dC;
    CK(cudaMalloc(&dA, max_bytes));
    CK(cudaMalloc(&dB, max_bytes));
    CK(cudaMalloc(&dC, max_bytes));

    cudaEvent_t evs, eve;
    CK(cudaEventCreate(&evs));
    CK(cudaEventCreate(&eve));

    FILE* csv = fopen("cuda-matmul-tiled/results.csv", "w");
    if (!csv) csv = fopen("results.csv", "w");
    if (!csv) { fprintf(stderr, "cannot open results.csv\n"); return 1; }
    fprintf(csv, "impl,kernel,N,iter,gpu_ms,tflops\n");

    double best_ms[3], median_ms[3], best_tf[3], median_tf[3];

    for (int ni = 0; ni < 3; ++ni) {
        int n = NS[ni];
        size_t nbytes = (size_t)n * n * sizeof(float);
        double flops = 2.0 * (double)n * n * n;

        for (int i = 0; i < n * n; ++i) {
            hA[i] = (i % 7) * 0.01f;
            hB[i] = (i % 11) * 0.01f;
        }
        CK(cudaMemcpy(dA, hA, nbytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, hB, nbytes, cudaMemcpyHostToDevice));
        CK(cudaMemset(dC, 0, nbytes));

        dim3 block(BN / TN, BM / TM);       // 8 x 8 = 64 threads
        dim3 grid(n / BN, n / BM);

        printf("[cuda-tiled] N=%d matmul f32, %.2f GFLOP/iter\n", n, flops / 1e9);

        for (int w = 0; w < WARMUPS; ++w) {
            matmul_tiled<<<grid, block>>>(dA, dB, dC, n);
        }
        CK(cudaDeviceSynchronize());

        double times[ITERS];
        float ms;
        for (int i = 0; i < ITERS; ++i) {
            cudaEventRecord(evs);
            matmul_tiled<<<grid, block>>>(dA, dB, dC, n);
            cudaEventRecord(eve);
            cudaEventSynchronize(eve);
            cudaEventElapsedTime(&ms, evs, eve);
            times[i] = ms;
            double tflops = (flops / 1e12) / (ms / 1000.0);
            printf("[cuda-tiled] N=%d iter=%d gpu_ms=%.3f tflops=%.3f\n", n, i, ms, tflops);
            fprintf(csv, "cuda-tiled,matmul_tiled,%d,%d,%.6f,%.6f\n", n, i, ms, tflops);
        }

        CK(cudaMemcpy(hC, dC, nbytes, cudaMemcpyDeviceToHost));
        int pts[3][2] = { {0, 0}, {n / 2, n / 2}, {n - 1, n - 1} };
        int ok = 0;
        for (int pi = 0; pi < 3; ++pi) {
            int r = pts[pi][0], c = pts[pi][1];
            float got = hC[r * n + c];
            float want = ref_elem(hA, hB, n, r, c);
            float rel = fabsf(got - want) / fmaxf(fabsf(want), 1e-6f);
            const char* tag = (rel < 1e-3f) ? "OK" : "FAIL";
            if (rel < 1e-3f) ok++;
            printf("[cuda-tiled] N=%d check (%d,%d): got=%.4f want=%.4f rel=%.3e %s\n",
                   n, r, c, got, want, rel, tag);
        }
        printf("[cuda-tiled] N=%d correctness: %d/3 OK\n", n, ok);

        double sorted[ITERS];
        for (int i = 0; i < ITERS; ++i) sorted[i] = times[i];
        for (int i = 0; i < ITERS; ++i)
            for (int j = i + 1; j < ITERS; ++j)
                if (sorted[j] < sorted[i]) {
                    double t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t;
                }
        double bst = sorted[0];
        double med = 0.5 * (sorted[ITERS/2 - 1] + sorted[ITERS/2]);
        best_ms[ni] = bst;
        median_ms[ni] = med;
        best_tf[ni] = (flops / 1e12) / (bst / 1000.0);
        median_tf[ni] = (flops / 1e12) / (med / 1000.0);
    }

    printf("\n[cuda-tiled] ===== SUMMARY =====\n");
    printf("[cuda-tiled] %6s  %10s  %10s  %10s  %10s\n", "N", "best_ms", "median_ms", "best_TF", "median_TF");
    for (int ni = 0; ni < 3; ++ni) {
        int n = NS[ni];
        printf("[cuda-tiled] %6d  %10.3f  %10.3f  %10.3f  %10.3f\n",
               n, best_ms[ni], median_ms[ni], best_tf[ni], median_tf[ni]);
    }

    fclose(csv);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
