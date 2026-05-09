// Raw CUDA C++ reference: naive matmul, identical algorithm to wgpu/cuda-oxide
// 16x16 thread block, one output element per thread, NO shared memory tiling.
// This is the "what speed of light looks like for this exact algorithm" baseline.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 4096
#define BS 16

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

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("[cuda] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    size_t bytes = (size_t)N * N * sizeof(float);
    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes);
    for (int i = 0; i < N*N; ++i) { hA[i] = (i % 7) * 0.01f; hB[i] = (i % 11) * 0.01f; }

    float *dA, *dB, *dC;
    CK(cudaMalloc(&dA, bytes)); CK(cudaMalloc(&dB, bytes)); CK(cudaMalloc(&dC, bytes));
    CK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    dim3 block(BS, BS), grid((N + BS - 1) / BS, (N + BS - 1) / BS);

    cudaEvent_t evs, eve; CK(cudaEventCreate(&evs)); CK(cudaEventCreate(&eve));
    double total_flops = 2.0 * (double)N * N * N;
    printf("[cuda] matmul %dx%d f32, %.2f GFLOP/iter\n", N, N, total_flops/1e9);

    double best = 1e30, median;
    double times[5];
    // warmup
    matmul<<<grid, block>>>(dA, dB, dC, N); CK(cudaDeviceSynchronize());
    float ms;
    cudaEventRecord(evs); matmul<<<grid, block>>>(dA, dB, dC, N); cudaEventRecord(eve);
    cudaEventSynchronize(eve); cudaEventElapsedTime(&ms, evs, eve);
    printf("[cuda] warmup: %.2f ms (%.3f TFLOPS)\n", ms, (total_flops/1e12)/(ms/1000.0));
    for (int i = 0; i < 5; ++i) {
        cudaEventRecord(evs);
        matmul<<<grid, block>>>(dA, dB, dC, N);
        cudaEventRecord(eve); cudaEventSynchronize(eve);
        cudaEventElapsedTime(&ms, evs, eve);
        times[i] = ms;
        if (ms < best) best = ms;
        printf("[cuda] iter %d: %.2f ms (%.3f TFLOPS)\n", i, ms, (total_flops/1e12)/(ms/1000.0));
    }
    // sort for median
    for (int i = 0; i < 5; ++i) for (int j = i+1; j < 5; ++j)
        if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }
    median = times[2];
    printf("\n[cuda] BEST   %.2f ms  %.3f TFLOPS\n", best, (total_flops/1e12)/(best/1000.0));
    printf("[cuda] MEDIAN %.2f ms  %.3f TFLOPS\n", median, (total_flops/1e12)/(median/1000.0));

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB);
    return 0;
}
