// Wave C2.6 — bench harness for cuda-attn-kda.
//
// Shape: kimi_linear_decode (1, 32, 128, 128) is the canonical KDA decode shape
// from cutile-attn-kda's registry. Also benches `large` (4, 64, 256, 256) for
// saturation comparison vs cutile-attn-kda's W22.7 best (1170 GB/s).
//
// 50 timed iters + 2 warmup, cudaEvent timing per iter, results.csv per shape.
//
// Build: see Makefile (target `bench`).

#define ATTN_KDA_BENCH_HARNESS 1

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA err %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while(0)

// Bring in the kernel + helpers. attn_kda.cu defines main(); guard against it
// by renaming.
#define main attn_kda_unused_main_
#include "attn_kda.cu"
#undef main

// Bytes/iter for KDA — matches cutile-attn-kda/main.py::kda_decode_bytes:
//   state = 2 * d_k * d_v * 4 (f32 read + write)
//   io    = (3 * d_k + 2 * d_v + 1) * 2     (q, k, g f16 in;  v, o f16; beta scalar f16)
// Multiply by B*H.
static double kda_bytes_per_iter(const KDAShape& sh) {
    const double bh = (double)sh.batch * sh.n_heads;
    const double state = 2.0 * sh.d_k * sh.d_v * 4.0;
    const double io    = (3.0 * sh.d_k + 2.0 * sh.d_v + 1.0) * 2.0;
    return bh * (state + io);
}

static int run_bench_shape(const std::string& inputs_dir,
                           const KDAShape& sh, int warmup, int iters,
                           const std::string& csv_path) {
    printf("[kda-bench] shape=%s B=%d H=%d d_k=%d d_v=%d  warmup=%d iters=%d\n",
           sh.name, sh.batch, sh.n_heads, sh.d_k, sh.d_v, warmup, iters);

    std::string prefix = inputs_dir + "/kda_" + sh.name;
    Npy q, k, v, g, b, sin_;
    if (!read_npy(prefix + "_q_f16.npy",     q))    return 1;
    if (!read_npy(prefix + "_k_f16.npy",     k))    return 1;
    if (!read_npy(prefix + "_v_f16.npy",     v))    return 1;
    if (!read_npy(prefix + "_g_f16.npy",     g))    return 1;
    if (!read_npy(prefix + "_beta_f16.npy",  b))    return 1;
    if (!read_npy(prefix + "_S_in_f32.npy",  sin_)) return 1;

    const int B_H = sh.batch * sh.n_heads;
    size_t qkv_elems = (size_t)B_H * sh.d_k;
    size_t v_elems   = (size_t)B_H * sh.d_v;
    size_t s_elems   = (size_t)B_H * sh.d_k * sh.d_v;
    size_t o_elems   = (size_t)B_H * sh.d_v;
    size_t scal      = (size_t)B_H;

    __half *dQ, *dK, *dV, *dG, *dB, *dO;
    float  *dSin, *dSout;
    CK(cudaMalloc(&dQ,    qkv_elems * sizeof(__half)));
    CK(cudaMalloc(&dK,    qkv_elems * sizeof(__half)));
    CK(cudaMalloc(&dV,    v_elems   * sizeof(__half)));
    CK(cudaMalloc(&dG,    qkv_elems * sizeof(__half)));
    CK(cudaMalloc(&dB,    scal      * sizeof(__half)));
    CK(cudaMalloc(&dSin,  s_elems   * sizeof(float)));
    CK(cudaMalloc(&dSout, s_elems   * sizeof(float)));
    CK(cudaMalloc(&dO,    o_elems   * sizeof(__half)));

    CK(cudaMemcpy(dQ,    q.data.data(),    q.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK,    k.data.data(),    k.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV,    v.data.data(),    v.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dG,    g.data.data(),    g.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,    b.data.data(),    b.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSin,  sin_.data.data(), sin_.data.size(), cudaMemcpyHostToDevice));

    const double bytes_per_iter = kda_bytes_per_iter(sh);
    printf("[kda-bench] bytes/iter = %.2f KB\n", bytes_per_iter / 1024.0);
    int n_blocks = B_H * (sh.d_v / ((sh.d_k == 128) ? 128 : (sh.d_k == 64) ? 64 : 64));
    printf("[kda-bench] n_blocks   = %d\n", n_blocks);

    cudaEvent_t evs, eve;
    CK(cudaEventCreate(&evs));
    CK(cudaEventCreate(&eve));

    for (int w = 0; w < warmup; ++w) {
        launch_kda_dispatch(sh, dQ, dK, dV, dG, dB, dSin, dSout, dO, 0);
    }
    CK(cudaDeviceSynchronize());

    FILE* csv = fopen(csv_path.c_str(), "w");
    if (!csv) { fprintf(stderr, "cannot open %s\n", csv_path.c_str()); return 1; }
    fprintf(csv, "impl,kernel,batch,n_heads,d_k,d_v,iter,gpu_ms,gbps\n");

    std::vector<double> times;
    times.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(evs);
        launch_kda_dispatch(sh, dQ, dK, dV, dG, dB, dSin, dSout, dO, 0);
        cudaEventRecord(eve);
        cudaEventSynchronize(eve);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, evs, eve);
        times.push_back(ms);
        double gbps = bytes_per_iter / (ms * 1e-3) / 1e9;
        if (i < 3 || i == iters - 1) {
            printf("[kda-bench] iter=%d gpu_us=%.2f gbps=%.1f\n", i, ms*1000.0, gbps);
        }
        fprintf(csv, "cuda-attn-kda,kda_decode_fused,%d,%d,%d,%d,%d,%.6f,%.3f\n",
                sh.batch, sh.n_heads, sh.d_k, sh.d_v, i, ms, gbps);
    }
    fclose(csv);

    if (!times.empty()) {
        std::vector<double> sorted_t = times;
        std::sort(sorted_t.begin(), sorted_t.end());
        double best = sorted_t.front();
        double med  = sorted_t[sorted_t.size() / 2];
        double sum = 0; for (double t : times) sum += t;
        double mean = sum / times.size();
        double best_gbps = bytes_per_iter / (best * 1e-3) / 1e9;
        double med_gbps  = bytes_per_iter / (med  * 1e-3) / 1e9;
        double mean_gbps = bytes_per_iter / (mean * 1e-3) / 1e9;
        printf("\n[kda-bench] ===== SUMMARY (%s) =====\n", sh.name);
        printf("[kda-bench] best   gpu_us=%.2f gbps=%.1f\n", best * 1000.0, best_gbps);
        printf("[kda-bench] median gpu_us=%.2f gbps=%.1f\n", med  * 1000.0, med_gbps);
        printf("[kda-bench] mean   gpu_us=%.2f gbps=%.1f\n", mean * 1000.0, mean_gbps);
        printf("[kda-bench] cuTile reference (W22.7 saturation @ large): 1170 GB/s best\n");
        printf("[kda-bench] HBM peak (RTX 5090): 1792 GB/s\n");
    }

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dG); cudaFree(dB);
    cudaFree(dSin); cudaFree(dSout); cudaFree(dO);
    return 0;
}

int main(int argc, char** argv) {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("[kda-bench] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    std::string inputs_dir =
        "/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs";
    int warmup = 2, iters = 50;
    std::string only_shape = "";  // empty = run both kimi_linear_decode + large
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--inputs" && i + 1 < argc) { inputs_dir = argv[++i]; }
        else if (a == "--warmup" && i + 1 < argc) { warmup = atoi(argv[++i]); }
        else if (a == "--iters"  && i + 1 < argc) { iters  = atoi(argv[++i]); }
        else if (a == "--shape"  && i + 1 < argc) { only_shape = argv[++i]; }
        else { fprintf(stderr, "[kda-bench] unknown arg %s\n", a.c_str()); return 64; }
    }

    KDAShape kld{"kimi_linear_decode", 1, 32, 128, 128};
    KDAShape lrg{"large",              4, 64, 256, 256};

    int rc = 0;
    if (only_shape.empty() || only_shape == "kimi_linear_decode") {
        rc = run_bench_shape(inputs_dir, kld, warmup, iters, "results.csv");
        if (rc) return rc;
    }
    if (only_shape.empty() || only_shape == "large") {
        rc = run_bench_shape(inputs_dir, lrg, warmup, iters, "results_large.csv");
        if (rc) return rc;
    }
    return 0;
}
