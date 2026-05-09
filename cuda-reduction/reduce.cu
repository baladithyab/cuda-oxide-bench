// Wave 4 W4A: nvcc CUDA C++ parallel sum-reduction.
// 2-stage reduction: warp-shuffle within warp -> smem across warps -> atomicAdd.
// Block size = 256 threads = 8 warps.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

constexpr int BLOCK = 256;
constexpr int WARPS_PER_BLOCK = BLOCK / 32;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    // Butterfly / xor tree reduction. All lanes end up with the full sum.
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v +=  __shfl_xor_sync(0xffffffff, v, 8);
    v +=  __shfl_xor_sync(0xffffffff, v, 4);
    v +=  __shfl_xor_sync(0xffffffff, v, 2);
    v +=  __shfl_xor_sync(0xffffffff, v, 1);
    return v;
}

__global__ void reduce_sum_kernel(const float* __restrict__ data,
                                  float* __restrict__ out,
                                  uint64_t n) {
    __shared__ float partials[WARPS_PER_BLOCK];

    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride load + local accumulate.
    float acc = 0.0f;
    for (uint64_t i = gid; i < n; i += stride) {
        acc += data[i];
    }

    // Warp-level reduce via shuffle-xor.
    acc = warp_reduce_sum(acc);

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    if (lane == 0) partials[warp] = acc;
    __syncthreads();

    // First warp reduces the 8 per-warp partials.
    if (warp == 0) {
        float v = (lane < WARPS_PER_BLOCK) ? partials[lane] : 0.0f;
        // Reduce 8 lanes worth -- shuffle down 4,2,1 (higher widths safely 0).
        v += __shfl_xor_sync(0xffffffff, v, 4);
        v += __shfl_xor_sync(0xffffffff, v, 2);
        v += __shfl_xor_sync(0xffffffff, v, 1);
        if (lane == 0) atomicAdd(out, v);
    }
}

int main() {
    CUDA_CHECK(cudaFree(0)); // ctx init.

    const size_t SIZES[] = { (size_t)1024*1024, (size_t)16*1024*1024, (size_t)256*1024*1024 };
    const int WARMUP = 1;
    const int ITERS = 10;

    FILE* csv = std::fopen("cuda-reduction/results.csv", "w");
    if (!csv) { fprintf(stderr, "cannot open results.csv\n"); return 1; }
    std::fprintf(csv, "impl,kernel,N_elems,iter,gpu_ms,GB_per_s\n");

    // Summary accumulator.
    struct Sum { size_t n; double best_ms, med_ms, best_gbs, med_gbs; double cpu_ref, gpu_val, rel_err; };
    std::vector<Sum> summary;

    // Allocate a single device buffer at max size.
    const size_t N_MAX = SIZES[sizeof(SIZES)/sizeof(SIZES[0]) - 1];
    float* d_data = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N_MAX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    std::printf("[cuda-reduction] sum-reduction sweep, block=%d, 1 warmup + %d iters per N\n",
                BLOCK, ITERS);

    for (size_t n : SIZES) {
        // Host data using (i%7)*0.01 pattern.
        std::vector<float> h(n);
        for (size_t i = 0; i < n; ++i) h[i] = (float)((i % 7)) * 0.01f;

        // CPU oracle: Kahan sum in double.
        double cpu_sum = 0.0, c_err = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double y = (double)h[i] - c_err;
            double t = cpu_sum + y;
            c_err = (t - cpu_sum) - y;
            cpu_sum = t;
        }

        CUDA_CHECK(cudaMemcpy(d_data, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        // Launch config: cap grid to device-practical upper bound (~48K blocks),
        // letting grid-stride loop handle the rest.
        int grid = (int)std::min<size_t>((n + BLOCK - 1) / BLOCK, 4096);
        if (grid < 1) grid = 1;

        // Warmup.
        for (int w = 0; w < WARMUP; ++w) {
            CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float)));
            reduce_sum_kernel<<<grid, BLOCK>>>(d_data, d_out, (uint64_t)n);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> ms_list;
        float final_gpu = 0.0f;
        for (int it = 0; it < ITERS; ++it) {
            CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float)));
            CUDA_CHECK(cudaEventRecord(ev_start));
            reduce_sum_kernel<<<grid, BLOCK>>>(d_data, d_out, (uint64_t)n);
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            double bytes = (double)n * 4.0;
            double gbs = (bytes / 1.0e9) / (ms / 1000.0);
            std::fprintf(csv, "cuda,reduce_sum,%zu,%d,%.6f,%.6f\n", n, it, ms, gbs);
            ms_list.push_back(ms);

            CUDA_CHECK(cudaMemcpy(&final_gpu, d_out, sizeof(float), cudaMemcpyDeviceToHost));
            std::printf("[cuda-reduction] N=%zu iter=%d gpu_ms=%.3f GB/s=%.1f\n",
                        n, it, ms, gbs);
        }

        std::vector<double> sorted_ms = ms_list;
        std::sort(sorted_ms.begin(), sorted_ms.end());
        double best = sorted_ms.front();
        double med = sorted_ms[sorted_ms.size()/2];
        double best_gbs = ((double)n * 4.0 / 1.0e9) / (best / 1000.0);
        double med_gbs  = ((double)n * 4.0 / 1.0e9) / (med  / 1000.0);

        double rel_err = std::fabs((double)final_gpu - cpu_sum) / std::max(std::fabs(cpu_sum), 1e-12);
        std::printf("[cuda-reduction] N=%zu  cpu_sum=%.6f gpu_sum=%.6f rel_err=%.3e\n",
                    n, cpu_sum, (double)final_gpu, rel_err);
        summary.push_back({n, best, med, best_gbs, med_gbs, cpu_sum, (double)final_gpu, rel_err});
    }

    std::fclose(csv);

    std::printf("\n================== SUMMARY ==================\n");
    std::printf("%-14s %14s %14s %14s %14s %12s\n",
                "N_elems", "best_ms", "med_ms", "best_GB/s", "med_GB/s", "rel_err");
    for (int i = 0; i < 84; ++i) std::putchar('-');
    std::putchar('\n');
    for (const auto& s : summary) {
        std::printf("%-14zu %14.3f %14.3f %14.1f %14.1f %12.3e\n",
                    s.n, s.best_ms, s.med_ms, s.best_gbs, s.med_gbs, s.rel_err);
    }
    std::printf("=============================================\n");

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    return 0;
}
