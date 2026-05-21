// Wave 22.13 — bench harness for cuda-attn-gdn-tma-warpspec.
//
// Per task: this cell is COMPILE + CORRECTNESS only. No timed benches in
// authoring stage. The harness is wired so the orchestrator can run it
// serially on idle GPU later.
//
// Includes attn_gdn_tma_warpspec.cu directly so we share the kernel definition
// + NPY loader. We add cudaEvent timing scaffolding around a single launch
// at the bench shape and emit results.csv.
//
// Build:
//   /usr/local/cuda/bin/nvcc -O3 -arch=sm_120 -ccbin clang-14 -o bench bench.cu -lcuda

#define ATTN_GDN_TMA_WS_BENCH_HARNESS 1

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA err %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while(0)

// Bring in the kernel + helpers. attn_gdn_tma_warpspec.cu defines main()
// guarded by !ATTN_GDN_TMA_WS_BENCH_HARNESS, so we just include directly.
#include "attn_gdn_tma_warpspec.cu"

// -- bench --
static int run_bench(const std::string& inputs_dir, int warmup, int iters) {
    GDNShape sh{"qwen3_next_decode", 1, 16, 256, 256};
    printf("[gdn-tma-ws-bench] shape=%s B=%d H=%d d_k=%d d_v=%d  warmup=%d iters=%d\n",
           sh.name, sh.batch, sh.n_heads, sh.d_k, sh.d_v, warmup, iters);

    Npy q, k, v, a, b, sin_;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_q_f16.npy",     q))    return 1;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_k_f16.npy",     k))    return 1;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_v_f16.npy",     v))    return 1;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_alpha_f16.npy", a))    return 1;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_beta_f16.npy",  b))    return 1;
    if (!read_npy(inputs_dir + "/gdn_qwen3_next_decode_S_in_f32.npy",  sin_)) return 1;

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

    const double state_bytes = 2.0 * sh.d_k * sh.d_v * 4.0;
    const double io_bytes    = (2.0 * sh.d_k + 2.0 * sh.d_v + 2.0) * 2.0;
    const double bytes_per_iter = (double)B_H * (state_bytes + io_bytes);
    printf("[gdn-tma-ws-bench] bytes/iter = %.2f KB\n", bytes_per_iter / 1024.0);

    cudaEvent_t evs, eve;
    CK(cudaEventCreate(&evs));
    CK(cudaEventCreate(&eve));

    for (int w = 0; w < warmup; ++w) {
        launch_gdn_dispatch(sh, dQ, dK, dV, dA, dB, dSin, dSout, dO, 0);
    }
    CK(cudaDeviceSynchronize());

    FILE* csv = fopen("results.csv", "w");
    if (!csv) { fprintf(stderr, "cannot open results.csv\n"); return 1; }
    fprintf(csv, "impl,kernel,batch,n_heads,d_k,d_v,iter,gpu_ms,gbps\n");

    std::vector<double> times;
    times.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(evs);
        launch_gdn_dispatch(sh, dQ, dK, dV, dA, dB, dSin, dSout, dO, 0);
        cudaEventRecord(eve);
        cudaEventSynchronize(eve);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, evs, eve);
        times.push_back(ms);
        double gbps = bytes_per_iter / (ms * 1e-3) / 1e9;
        if (i < 3 || i == iters - 1) {
            printf("[gdn-tma-ws-bench] iter=%d gpu_ms=%.4f gbps=%.1f\n", i, ms, gbps);
        }
        fprintf(csv, "cuda-attn-gdn-tma-warpspec,gdn_decode_tma_ws,%d,%d,%d,%d,%d,%.6f,%.3f\n",
                sh.batch, sh.n_heads, sh.d_k, sh.d_v, i, ms, gbps);
    }
    fclose(csv);

    if (!times.empty()) {
        double best = times[0], sum = 0.0;
        for (double t : times) { if (t < best) best = t; sum += t; }
        double mean = sum / (double)times.size();
        double best_gbps = bytes_per_iter / (best * 1e-3) / 1e9;
        double mean_gbps = bytes_per_iter / (mean * 1e-3) / 1e9;
        printf("\n[gdn-tma-ws-bench] ===== SUMMARY =====\n");
        printf("[gdn-tma-ws-bench] best  gpu_us=%.2f gbps=%.1f\n", best * 1000.0, best_gbps);
        printf("[gdn-tma-ws-bench] mean  gpu_us=%.2f gbps=%.1f\n", mean * 1000.0, mean_gbps);
        printf("[gdn-tma-ws-bench] W1c=417.7  cuTile=610  W22.10=1032 GB/s   HBM peak=1792\n");
    }

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dA); cudaFree(dB);
    cudaFree(dSin); cudaFree(dSout); cudaFree(dO);
    return 0;
}

int main(int argc, char** argv) {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("[gdn-tma-ws-bench] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    std::string inputs_dir =
        "/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs";
    int warmup = 2, iters = 50;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--inputs" && i + 1 < argc) { inputs_dir = argv[++i]; }
        else if (a == "--warmup" && i + 1 < argc) { warmup = atoi(argv[++i]); }
        else if (a == "--iters"  && i + 1 < argc) { iters  = atoi(argv[++i]); }
        else { fprintf(stderr, "[gdn-tma-ws-bench] unknown arg %s\n", a.c_str()); return 64; }
    }
    return run_bench(inputs_dir, warmup, iters);
}
