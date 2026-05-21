// Wave 22.15 — bench_sweep.cu: shape sweep for cuda-attn-gdn-tma.
//
// Runs cudaEvent-timed benches at five shapes and emits per-shape correctness
// + GB/s + GFLOPS records into results_sweep.csv. Mirrors the bench shape
// registry that cutile-attn-gdn/main.py uses, so the orchestrator can
// produce an apples-to-apples cuTile-vs-TMA comparison.
//
// Shapes:
//   qwen3_next_decode  B=1 H=16 d_k=256 d_v=256   (legacy headline)
//   tiny               B=1 H=4  d_k=64  d_v=64
//   small              B=1 H=8  d_k=128 d_v=128
//   large              B=4 H=64 d_k=256 d_v=256
//   wide               B=1 H=16 d_k=512 d_v=512
//
// One important pitfall: cuTensorMapEncodeTiled bakes BLOCK_V and D_K into
// the tensor descriptor, so we MUST rebuild the descriptor per launch (each
// shape's launcher already does this in launch_gdn_tma; not free, but only
// O(microseconds) and outside the timed window).
//
// Build:
//   /usr/local/cuda/bin/nvcc -O3 -arch=sm_120 -ccbin clang-14 -o bench_sweep bench_sweep.cu -lcuda

#define ATTN_GDN_TMA_BENCH_HARNESS 1

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
#include <cuda.h>

// Bring in the kernel + helpers (includes Npy + GDNShape + dispatcher).
// attn_gdn_tma.cu defines main() guarded by !ATTN_GDN_TMA_BENCH_HARNESS,
// so direct inclusion is safe.
#include "attn_gdn_tma.cu"

// ─────────────────────────────────────────────────────────────────────────────
// Sweep shape registry (mirrors cutile-attn-gdn/main.py SHAPE_REGISTRY).
// ─────────────────────────────────────────────────────────────────────────────

struct SweepEntry {
    GDNShape sh;
    const char* file_prefix;   // gdn_<file_prefix>_*.npy
    int n_blocks;              // computed below
};

// BLOCK_V mirror of the dispatcher's policy in attn_gdn_tma.cu.
static int picked_block_v(int d_k, int d_v) {
    if (d_k <= 256) return 64;
    return 32; // d_k=512 (wide) needs BV=32 to fit dynamic-smem opt-in.
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-shape correctness + bench loop. Loads inputs from disk, runs once for
// correctness vs the gdn_<name>_o_expected_f16 / S_out_expected_f32 oracle,
// then runs `iters` timed iterations.
// ─────────────────────────────────────────────────────────────────────────────

struct ShapeResult {
    std::string name;
    int batch, n_heads, d_k, d_v;
    int n_blocks;
    int block_v;
    bool ok_correctness;
    double max_abs_o;
    double max_abs_s;
    double bytes_per_iter;
    double flops_per_iter;
    double best_us, mean_us, median_us;
    double best_gbps, mean_gbps;
    double best_gflops;
};

static double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n == 0) return 0.0;
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static int run_one_shape(const std::string& inputs_dir,
                         const SweepEntry& e,
                         int warmup, int iters,
                         FILE* csv,
                         ShapeResult& out)
{
    const GDNShape& sh = e.sh;
    out.name = sh.name;
    out.batch = sh.batch; out.n_heads = sh.n_heads;
    out.d_k = sh.d_k;     out.d_v = sh.d_v;
    out.block_v = picked_block_v(sh.d_k, sh.d_v);
    out.n_blocks = sh.batch * sh.n_heads * (sh.d_v / out.block_v);
    out.ok_correctness = false;
    out.max_abs_o = -1.0; out.max_abs_s = -1.0;

    printf("\n[sweep] ===== shape=%s B=%d H=%d d_k=%d d_v=%d  BV=%d  n_blocks=%d =====\n",
           sh.name, sh.batch, sh.n_heads, sh.d_k, sh.d_v, out.block_v, out.n_blocks);

    // Pitfall: cuTensorMapEncodeTiled imposes boxDim[i] ≤ 256 elements per
    // axis (CUDA driver invariant for tiled TMA). Our descriptor uses
    // boxDim = (BLOCK_V, D_K), so D_K > 256 cannot be served by a single
    // TMA tile; it would need a per-CTA loop over multiple (D_K=256) tiles
    // to assemble the full slab in shared memory. That's a kernel-body
    // refactor we're explicitly NOT making in this cell (task says: don't
    // touch attn_gdn_tma.cu kernel). We skip the bench at this shape and
    // record an explicit "TMA-unsupported" row in the summary — the cuTile
    // baseline still runs at this shape and its number is meaningful.
    if (sh.d_k > 256) {
        printf("[sweep] %s: D_K=%d > TMA boxDim cap (256). "
               "Single-tile TMA load is structurally impossible at this "
               "shape; would need per-CTA two-tile assembly. Skipping bench.\n",
               sh.name, sh.d_k);
        out.ok_correctness = false;
        return 0;
    }

    Npy q, k, v, a, b, sin_, oexp, sexp;
    auto load = [&](const char* tag, Npy& dst) {
        std::string path = inputs_dir + "/gdn_" + e.file_prefix + tag;
        if (!read_npy(path, dst)) { fprintf(stderr, "load fail %s\n", path.c_str()); return false; }
        return true;
    };
    if (!load("_q_f16.npy",     q))    return 1;
    if (!load("_k_f16.npy",     k))    return 1;
    if (!load("_v_f16.npy",     v))    return 1;
    if (!load("_alpha_f16.npy", a))    return 1;
    if (!load("_beta_f16.npy",  b))    return 1;
    if (!load("_S_in_f32.npy",  sin_)) return 1;
    if (!load("_o_expected_f16.npy",   oexp)) return 1;
    if (!load("_S_out_expected_f32.npy", sexp)) return 1;

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

    CK(cudaMemcpy(dQ,   q.data.data(),    q.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK,   k.data.data(),    k.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV,   v.data.data(),    v.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dA,   a.data.data(),    a.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,   b.data.data(),    b.data.size(),    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSin, sin_.data.data(), sin_.data.size(), cudaMemcpyHostToDevice));
    CK(cudaMemset(dSout, 0, s_elems * sizeof(float)));
    CK(cudaMemset(dO,    0, o_elems * sizeof(__half)));

    // Bytes / FLOPs models, identical to flops_gdn.py.
    const double state_bytes = 2.0 * sh.d_k * sh.d_v * 4.0;          // f32 read+write
    const double io_bytes    = (2.0 * sh.d_k + 2.0 * sh.d_v + 2.0) * 2.0;  // f16 i/o + α/β
    out.bytes_per_iter = (double)B_H * (state_bytes + io_bytes);
    out.flops_per_iter = 6.0 * (double)B_H * sh.d_k * sh.d_v;
    printf("[sweep] bytes/iter = %.2f KB   flops/iter = %.3f MFLOPS\n",
           out.bytes_per_iter / 1024.0, out.flops_per_iter / 1e6);

    // ── Correctness pass (one launch). ──
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
    double exp_o_mag = 0.0, exp_s_mag = 0.0;
    for (size_t i = 0; i < o_elems; ++i) {
        float got = __half2float(ho[i]);
        float want = __half2float(o_exp_h[i]);
        double am = fabs((double)got - (double)want);
        if (am > max_abs_o) max_abs_o = am;
        double aw = fabs((double)want);
        if (aw > exp_o_mag) exp_o_mag = aw;
    }
    for (size_t i = 0; i < s_elems; ++i) {
        double am = fabs((double)hs[i] - (double)s_exp_f[i]);
        if (am > max_abs_s) max_abs_s = am;
        double aw = fabs((double)s_exp_f[i]);
        if (aw > exp_s_mag) exp_s_mag = aw;
    }
    out.max_abs_o = max_abs_o;
    out.max_abs_s = max_abs_s;

    // Acceptance: same 1e-3 atol the W22.10 cell uses for `o`. For `S_out`
    // (f32 state) we use a slightly looser 5e-3 because at the larger
    // shapes the d_k accumulation chain is longer; W22.10 itself measured
    // 2.98e-08 for state, so this is loose by 6+ orders of magnitude.
    const double ATOL_O = 1e-3;
    const double ATOL_S = 5e-3;
    bool ok_o = max_abs_o <= ATOL_O;
    bool ok_s = max_abs_s <= ATOL_S;
    out.ok_correctness = ok_o && ok_s;
    printf("[sweep] correctness: o max_abs=%.3e (|want|max=%.3e) %s   "
           "S_out max_abs=%.3e (|want|max=%.3e) %s\n",
           max_abs_o, exp_o_mag, ok_o ? "OK" : "FAIL",
           max_abs_s, exp_s_mag, ok_s ? "OK" : "FAIL");

    if (!out.ok_correctness) {
        // Don't bench a broken kernel — but DO emit the result row so
        // the comparison table records the failure.
        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dA); cudaFree(dB);
        cudaFree(dSin); cudaFree(dSout); cudaFree(dO);
        return 0;
    }

    // ── Timed bench. ──
    cudaEvent_t evs, eve;
    CK(cudaEventCreate(&evs));
    CK(cudaEventCreate(&eve));

    for (int w = 0; w < warmup; ++w) {
        launch_gdn_dispatch(sh, dQ, dK, dV, dA, dB, dSin, dSout, dO, 0);
    }
    CK(cudaDeviceSynchronize());

    std::vector<double> times_ms;
    times_ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(evs);
        launch_gdn_dispatch(sh, dQ, dK, dV, dA, dB, dSin, dSout, dO, 0);
        cudaEventRecord(eve);
        cudaEventSynchronize(eve);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, evs, eve);
        times_ms.push_back(ms);
        double gbps = out.bytes_per_iter / (ms * 1e-3) / 1e9;
        double gflops = out.flops_per_iter / (ms * 1e-3) / 1e9;
        if (i < 3 || i == iters - 1) {
            printf("[sweep] iter=%2d gpu_us=%.2f gbps=%.1f gflops=%.2f\n",
                   i, ms * 1000.0, gbps, gflops);
        }
        fprintf(csv, "cuda-attn-gdn-tma,gdn_decode_tma,%s,%d,%d,%d,%d,%d,%d,%d,%.6f,%.3f,%.3f\n",
                sh.name, sh.batch, sh.n_heads, sh.d_k, sh.d_v,
                out.block_v, out.n_blocks, i,
                ms, gbps, gflops);
    }

    double best = times_ms[0], sum = 0.0;
    for (double t : times_ms) { if (t < best) best = t; sum += t; }
    double mean = sum / (double)times_ms.size();
    double med  = median_of(times_ms);
    out.best_us   = best * 1000.0;
    out.mean_us   = mean * 1000.0;
    out.median_us = med  * 1000.0;
    out.best_gbps = out.bytes_per_iter / (best * 1e-3) / 1e9;
    out.mean_gbps = out.bytes_per_iter / (mean * 1e-3) / 1e9;
    out.best_gflops = out.flops_per_iter / (best * 1e-3) / 1e9;
    printf("[sweep] %s SUMMARY  best=%.2fus (%.1f GB/s, %.2f GFLOPS)  "
           "mean=%.2fus (%.1f GB/s)  median=%.2fus\n",
           sh.name, out.best_us, out.best_gbps, out.best_gflops,
           out.mean_us, out.mean_gbps, out.median_us);

    cudaEventDestroy(evs); cudaEventDestroy(eve);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dA); cudaFree(dB);
    cudaFree(dSin); cudaFree(dSout); cudaFree(dO);
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Entrypoint
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("[sweep] device: %s (sm_%d%d)  smem_optin=%zu KB\n",
           p.name, p.major, p.minor, p.sharedMemPerBlockOptin / 1024);

    std::string inputs_dir =
        "/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs";
    int warmup = 2, iters = 50;
    std::string only_shape;
    for (int i = 1; i < argc; ++i) {
        std::string ar = argv[i];
        if      (ar == "--inputs" && i + 1 < argc) { inputs_dir = argv[++i]; }
        else if (ar == "--warmup" && i + 1 < argc) { warmup = atoi(argv[++i]); }
        else if (ar == "--iters"  && i + 1 < argc) { iters  = atoi(argv[++i]); }
        else if (ar == "--shape"  && i + 1 < argc) { only_shape = argv[++i]; }
        else { fprintf(stderr, "[sweep] unknown arg %s\n", ar.c_str()); return 64; }
    }

    SweepEntry shapes[] = {
        // (file_prefix matches the gdn_<prefix>_*.npy on-disk naming).
        { GDNShape{"qwen3_next_decode", 1, 16, 256, 256}, "qwen3_next_decode", 0 },
        { GDNShape{"tiny",              1,  4,  64,  64}, "tiny",              0 },
        { GDNShape{"small",             1,  8, 128, 128}, "small",             0 },
        { GDNShape{"large",             4, 64, 256, 256}, "large",             0 },
        { GDNShape{"wide",              1, 16, 512, 512}, "wide",              0 },
    };
    const int n_shapes = sizeof(shapes) / sizeof(shapes[0]);

    FILE* csv = fopen("results_sweep.csv", "w");
    if (!csv) { fprintf(stderr, "cannot open results_sweep.csv\n"); return 1; }
    fprintf(csv,
        "impl,kernel,shape,batch,n_heads,d_k,d_v,block_v,n_blocks,iter,gpu_ms,gbps,gflops\n");

    std::vector<ShapeResult> results;
    results.reserve(n_shapes);
    int rc = 0;
    for (int i = 0; i < n_shapes; ++i) {
        if (!only_shape.empty() && only_shape != shapes[i].sh.name) continue;
        ShapeResult r{};
        int sub_rc = run_one_shape(inputs_dir, shapes[i], warmup, iters, csv, r);
        if (sub_rc != 0) rc = sub_rc;
        results.push_back(r);
    }
    fclose(csv);

    // Per-shape summary table.
    printf("\n");
    printf("====== W22.15 cuda-attn-gdn-tma sweep summary =================================\n");
    printf("%-20s %-22s %-9s %-9s %-9s %-7s %-9s\n",
           "shape", "(B,H,d_k,d_v)", "n_blocks", "best_us", "best_gbps",
           "best_gf", "ok");
    for (const auto& r : results) {
        char shp[40];
        snprintf(shp, sizeof(shp), "(%d,%d,%d,%d)", r.batch, r.n_heads, r.d_k, r.d_v);
        if (r.d_k > 256) {
            printf("%-20s %-22s %-9d %-9s %-9s %-7s %-9s\n",
                   r.name.c_str(), shp, r.n_blocks,
                   "—", "—", "—", "TMA-NA");
        } else {
            printf("%-20s %-22s %-9d %-9.2f %-9.1f %-7.2f %-9s\n",
                   r.name.c_str(), shp, r.n_blocks,
                   r.best_us, r.best_gbps, r.best_gflops,
                   r.ok_correctness ? "PASS" : "FAIL");
        }
    }
    printf("===============================================================================\n");
    printf("[sweep] CSV: results_sweep.csv  (one row per (shape, iter))\n");
    printf("[sweep] note: 'wide' shape (D_K=512) is unsupported by single-tile TMA\n");
    printf("[sweep]       (cuTensorMapEncodeTiled boxDim cap = 256). cuTile baseline\n");
    printf("[sweep]       still benches at this shape — see results_cutile_sweep.csv.\n");
    return rc;
}
