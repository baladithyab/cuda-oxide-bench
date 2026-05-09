// Raw CUDA C++ reference: naive matmul, identical algorithm to wgpu/cuda-oxide
// 16x16 thread block, one output element per thread, NO shared memory tiling.
// Wave 1 W1B: native sm_120 build + size sweep N in {1024, 2048, 4096}
// Per ADR-0001 (cudaEvent timing), ADR-0002 (sm_120 native).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define MAXN 4096
#define BS 16
#define WARMUPS 1
#define ITERS 10

__global__ void matmul(const float* __restrict__ A, const float* __restrict__ B,
                       float* __restrict__ C, int dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dim || col >= dim) return;
    float acc = 0.0f;
    for (int k = 0; k < dim; ++k) {
        acc += A[row * dim + k] * B[k * dim + col];
    }
    C[row * dim + col] = acc;
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); exit(1); } } while(0)

// Reference on host for small spot-checks. For sub-matrix of size n from buffers
// that were filled as if they were full MAXN-sized (we use top-left n-by-n slice
// of the MAXN-sized host arrays, so reference must use stride n on those n x n
// re-copies).
static float ref_elem(const float* hA, const float* hB, int n, int row, int col) {
    double acc = 0.0;
    for (int k = 0; k < n; ++k) {
        acc += (double)hA[row * n + k] * (double)hB[k * n + col];
    }
    return (float)acc;
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("[cuda] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    const int NS[3] = {1024, 2048, 4096};
    const size_t max_bytes = (size_t)MAXN * MAXN * sizeof(float);

    // Allocate once for MAXN. For smaller N we reuse the top-left n x n sub-use,
    // but we repack host arrays for each N so the stride matches the device
    // kernel's stride (kernel uses stride = n).
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

    // CSV output
    FILE* csv = fopen("cuda-matmul/results.csv", "w");
    if (!csv) csv = fopen("results.csv", "w");
    if (!csv) { fprintf(stderr, "cannot open results.csv\n"); return 1; }
    fprintf(csv, "impl,kernel,N,iter,gpu_ms,tflops\n");

    // Per-N summary
    double best_ms[3], median_ms[3], best_tf[3], median_tf[3];

    for (int ni = 0; ni < 3; ++ni) {
        int n = NS[ni];
        size_t nbytes = (size_t)n * n * sizeof(float);
        double flops = 2.0 * (double)n * n * n;

        // Fill host arrays with stride == n (per-iter input pattern)
        for (int i = 0; i < n * n; ++i) {
            hA[i] = (i % 7) * 0.01f;
            hB[i] = (i % 11) * 0.01f;
        }
        CK(cudaMemcpy(dA, hA, nbytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, hB, nbytes, cudaMemcpyHostToDevice));
        CK(cudaMemset(dC, 0, nbytes));

        dim3 block(BS, BS);
        dim3 grid((n + BS - 1) / BS, (n + BS - 1) / BS);

        printf("[cuda] N=%d matmul f32, %.2f GFLOP/iter\n", n, flops / 1e9);

        // warmup
        for (int w = 0; w < WARMUPS; ++w) {
            matmul<<<grid, block>>>(dA, dB, dC, n);
        }
        CK(cudaDeviceSynchronize());

        double times[ITERS];
        float ms;
        for (int i = 0; i < ITERS; ++i) {
            cudaEventRecord(evs);
            matmul<<<grid, block>>>(dA, dB, dC, n);
            cudaEventRecord(eve);
            cudaEventSynchronize(eve);
            cudaEventElapsedTime(&ms, evs, eve);
            times[i] = ms;
            double tflops = (flops / 1e12) / (ms / 1000.0);
            printf("[cuda] N=%d iter=%d gpu_ms=%.3f tflops=%.3f\n", n, i, ms, tflops);
            fprintf(csv, "cuda-matmul,matmul,%d,%d,%.6f,%.6f\n", n, i, ms, tflops);
        }

        // Correctness spot-check at (0,0), (n/2, n/2), (n-1, n-1)
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
            printf("[cuda] N=%d check (%d,%d): got=%.4f want=%.4f rel=%.3e %s\n",
                   n, r, c, got, want, rel, tag);
        }
        printf("[cuda] N=%d correctness: %d/3 OK\n", n, ok);

        // sort times for median
        double sorted[ITERS];
        for (int i = 0; i < ITERS; ++i) sorted[i] = times[i];
        for (int i = 0; i < ITERS; ++i)
            for (int j = i + 1; j < ITERS; ++j)
                if (sorted[j] < sorted[i]) {
                    double t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t;
                }
        double bst = sorted[0];
        double med = 0.5 * (sorted[ITERS/2 - 1] + sorted[ITERS/2]); // ITERS=10 → avg of 5th & 6th
        best_ms[ni] = bst;
        median_ms[ni] = med;
        best_tf[ni] = (flops / 1e12) / (bst / 1000.0);
        median_tf[ni] = (flops / 1e12) / (med / 1000.0);
    }

    // Summary table
    printf("\n[cuda] ===== SUMMARY =====\n");
    printf("[cuda] %6s  %10s  %10s  %10s  %10s\n", "N", "best_ms", "median_ms", "best_TF", "median_TF");
    for (int ni = 0; ni < 3; ++ni) {
        int n = NS[ni];
        printf("[cuda] %6d  %10.3f  %10.3f  %10.3f  %10.3f\n",
               n, best_ms[ni], median_ms[ni], best_tf[ni], median_tf[ni]);
    }

    fclose(csv);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
