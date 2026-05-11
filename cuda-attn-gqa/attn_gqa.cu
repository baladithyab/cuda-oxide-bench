// Wave 15.1 — nvcc CUDA C++ GQA (Grouped-Query Attention), 3-kernel naive.
//
// Pipeline (no fusion):
//   1) gqa_qkt_kernel:   S = Q @ K^T * scale            [B, Nq, S, S]  (f16×f16→f32)
//   2) softmax_kernel:   P = softmax(S, dim=-1)         [B, Nq, S, S]  f32→f16
//   3) gqa_pv_kernel:    O = P @ V                      [B, Nq, S, D]  (f16×f16→f32→f16)
//
// Tensor cores: WMMA API, tile m16n16k16 f16→f32. Verified via
//   cuobjdump --dump-sass attn_gqa | grep HMMA   (expect > 0 matches).
//
// GQA broadcast is implicit: for Q-head hq in [0, n_q), the KV head is
// hkv = hq / groups. We never expand K/V in memory.
//
// Shapes loaded from analysis/wave15-attention-architecture/inputs/ as
// little-endian NumPy .npy files. A small inline parser walks the NPY1.0
// header (~150 bytes ASCII) and memcpy's the raw data block.
//
// Correctness: SHAPE_CORRECTNESS (b=1,s=128,nq=4,nkv=2,d=64)
//              tolerance f16: atol=5e-3, rtol=5e-3 (from tolerances.py)
// Bench:       SHAPE_BENCH    (b=1,s=2048,nq=32,nkv=8,d=128)
//              FLOPS = 4 * b * nq * s^2 * d  (from flops.py)
//
// Build: /usr/local/cuda/bin/nvcc -ccbin clang-14 -O3 -arch=sm_120
//        -lstdc++ -o attn_gqa attn_gqa.cu

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
#include <mma.h>

using namespace nvcuda;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA err %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while(0)

constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;

// ---- .npy loader (NPY1.0 / NPY2.0, little-endian only; matches numpy.save output) ----
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
    } else { // 2.x, 3.x
        fread(&header_len, 1, 4, f);
    }
    std::string hdr(header_len, ' ');
    fread(&hdr[0], 1, header_len, f);

    // Super simple header parse: look for "'descr': '<f2'" or '<f4', and "'shape': (a, b, ...)"
    auto find_after = [&](const std::string& key) -> size_t {
        size_t p = hdr.find(key);
        return p == std::string::npos ? std::string::npos : p + key.size();
    };
    size_t p = find_after("'descr':");
    if (p == std::string::npos) { fprintf(stderr, "no descr\n"); fclose(f); return false; }
    while (p < hdr.size() && hdr[p] != '\'') ++p; ++p;
    size_t q = p;
    while (q < hdr.size() && hdr[q] != '\'') ++q;
    out.dtype = hdr.substr(p, q - p);
    if (out.dtype == "<f2") out.elem_size = 2;
    else if (out.dtype == "<f4") out.elem_size = 4;
    else { fprintf(stderr, "unsupported dtype %s\n", out.dtype.c_str()); fclose(f); return false; }

    p = find_after("'shape':");
    if (p == std::string::npos) { fprintf(stderr, "no shape\n"); fclose(f); return false; }
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
        fprintf(stderr, "%s: short read\n", path.c_str()); fclose(f); return false;
    }
    fclose(f);
    return true;
}

// ============================================================================
// Kernel 1: QKt — S[b, hq, i, j] = scale * sum_k Q[b, hq, i, k] * K[b, hkv, j, k]
// Each Q head hq picks up KV head hkv = hq / groups.
// Tile: 16x16 output, K-step 16. Use WMMA f16×f16→f32.
// Grid: (tiles_row, tiles_col, B*Nq).  Block: 32 threads (one warp).
// ============================================================================
__global__ void gqa_qkt_kernel(
    const __half* __restrict__ Q,  // [B, Nq, S, D]
    const __half* __restrict__ K,  // [B, Nkv, S, D]
    float*        __restrict__ Sm, // [B, Nq, S, S] (f32 output)
    int B, int Nq, int Nkv, int S, int D,
    float scale)
{
    int tile_i = blockIdx.x;  // row tile (over S)
    int tile_j = blockIdx.y;  // col tile (over S)
    int bh    = blockIdx.z;   // batch*nq index
    int b     = bh / Nq;
    int hq    = bh % Nq;
    int groups = Nq / Nkv;
    int hkv    = hq / groups;

    int row0 = tile_i * WM;
    int col0 = tile_j * WN;
    if (row0 >= S || col0 >= S) return;

    // Pointers to this head's Q and K sub-matrices.
    const __half* Qh = Q + ((size_t)b * Nq + hq) * S * D;
    const __half* Kh = K + ((size_t)b * Nkv + hkv) * S * D;
    float*        Sh = Sm + ((size_t)b * Nq + hq) * S * S;

    // Q tile A is row-major (S, D), stride = D.
    // We want S = Q * K^T. K is row-major (S, D) stride D; K^T is col-major (D, S).
    // Equivalently: load K as a "col_major" fragment of shape (k=D by n=S) with lda = D.
    wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> cf;
    wmma::fill_fragment(cf, 0.0f);

    // We assume S % WM == 0 and S % WN == 0 for both canonical shapes (128 and 2048).
    // D also a multiple of 16 (64, 128) → no K-tail to handle.
    for (int k = 0; k < D; k += WK) {
        const __half* Aptr = Qh + row0 * D + k;            // (WM x WK) row-major, ld=D
        const __half* Bptr = Kh + col0 * D + k;            // (WK x WN) col-major, ld=D
        // For col_major matrix_b of logical shape (K, N), load ptr points to element (0,0)
        // with leading dim = K steps between consecutive columns. Here columns of K^T are
        // rows of K, so each "column" is Kh[row=col0+n, :] — starting at Kh + col0*D, and
        // stepping between columns is D. So ldb = D. Load
        wmma::load_matrix_sync(af, Aptr, D);
        wmma::load_matrix_sync(bf, Bptr, D);
        wmma::mma_sync(cf, af, bf, cf);
    }

    // Apply scale and store to S[row0:row0+WM, col0:col0+WN].
    #pragma unroll
    for (int t = 0; t < cf.num_elements; ++t) cf.x[t] *= scale;

    wmma::store_matrix_sync(Sh + row0 * S + col0, cf, S, wmma::mem_row_major);
}

// ============================================================================
// Kernel 2: softmax — P[b, hq, i, :] = softmax(S[b, hq, i, :])
// Output is f16 so the next kernel can use tensor cores on P as matrix_a.
// One block per (b, hq, row). Block-wide reduction for max, then for sum.
// ============================================================================
constexpr int SOFTMAX_TPB = 128;

__global__ void softmax_kernel(
    const float* __restrict__ Sm,  // [B, Nq, S, S] f32
    __half*      __restrict__ P,   // [B, Nq, S, S] f16
    int B, int Nq, int S)
{
    int row = blockIdx.x;          // 0..S-1
    int bh  = blockIdx.y;          // 0..B*Nq-1
    int tid = threadIdx.x;

    const float* Srow = Sm + (size_t)bh * S * S + (size_t)row * S;
    __half*      Prow = P  + (size_t)bh * S * S + (size_t)row * S;

    __shared__ float sbuf[SOFTMAX_TPB];

    // Pass 1: max
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

    // Pass 2: sum of exp
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

    // Pass 3: normalize, write f16
    for (int j = tid; j < S; j += SOFTMAX_TPB) {
        float p = expf(Srow[j] - rmax) * inv_sum;
        Prow[j] = __float2half(p);
    }
}

// ============================================================================
// Kernel 3: PV — O[b, hq, i, d] = sum_j P[b, hq, i, j] * V[b, hkv, j, d]
// P is (S, S) row-major f16; V is (S, D) row-major f16.
// Use WMMA m16n16k16, accum f32, store f16.
// Grid: (tiles_row_s, tiles_col_d, B*Nq). Block: 32 threads (one warp).
// ============================================================================
__global__ void gqa_pv_kernel(
    const __half* __restrict__ P,   // [B, Nq, S, S]
    const __half* __restrict__ V,   // [B, Nkv, S, D]
    __half*       __restrict__ O,   // [B, Nq, S, D]
    int B, int Nq, int Nkv, int S, int D)
{
    int tile_i = blockIdx.x;   // row over S
    int tile_j = blockIdx.y;   // col over D
    int bh    = blockIdx.z;
    int b     = bh / Nq;
    int hq    = bh % Nq;
    int groups = Nq / Nkv;
    int hkv    = hq / groups;

    int row0 = tile_i * WM;
    int col0 = tile_j * WN;
    if (row0 >= S || col0 >= D) return;

    const __half* Ph = P + ((size_t)b * Nq + hq)  * S * S;
    const __half* Vh = V + ((size_t)b * Nkv + hkv) * S * D;
    __half*       Oh = O + ((size_t)b * Nq + hq)  * S * D;

    wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> cf;
    wmma::fill_fragment(cf, 0.0f);

    for (int k = 0; k < S; k += WK) {
        const __half* Aptr = Ph + row0 * S + k;       // (WM x WK), ld = S
        const __half* Bptr = Vh + k * D + col0;       // (WK x WN), ld = D
        wmma::load_matrix_sync(af, Aptr, S);
        wmma::load_matrix_sync(bf, Bptr, D);
        wmma::mma_sync(cf, af, bf, cf);
    }

    // Convert acc f32 → f16 for output. Use a tiny staging f32 → half loop.
    __shared__ float stage[WM * WN];
    wmma::store_matrix_sync(stage, cf, WN, wmma::mem_row_major);
    // 32 threads per block; each handles WM*WN/32 = 8 elements.
    int lane = threadIdx.x;
    #pragma unroll
    for (int e = lane; e < WM * WN; e += 32) {
        int r = e / WN;
        int c = e % WN;
        Oh[(row0 + r) * D + (col0 + c)] = __float2half(stage[e]);
    }
}

// ============================================================================
// Host driver
// ============================================================================
struct GQAShape {
    const char* name;
    int B, Nq, Nkv, S, D;
};

static void run_pipeline(const GQAShape& sh,
                         const __half* dQ, const __half* dK, const __half* dV,
                         float* dS_scores, __half* dP, __half* dO)
{
    float scale = 1.0f / sqrtf((float)sh.D);

    // QKt: tiles_row = S/16, tiles_col = S/16, heads = B*Nq.
    dim3 gQK(sh.S / WM, sh.S / WN, sh.B * sh.Nq);
    dim3 bQK(32);
    gqa_qkt_kernel<<<gQK, bQK>>>(dQ, dK, dS_scores, sh.B, sh.Nq, sh.Nkv, sh.S, sh.D, scale);

    // Softmax: one block per (B*Nq, row).
    dim3 gSM(sh.S, sh.B * sh.Nq);
    dim3 bSM(SOFTMAX_TPB);
    softmax_kernel<<<gSM, bSM>>>(dS_scores, dP, sh.B, sh.Nq, sh.S);

    // PV: tiles_row = S/16, tiles_col = D/16.
    dim3 gPV(sh.S / WM, sh.D / WN, sh.B * sh.Nq);
    dim3 bPV(32);
    gqa_pv_kernel<<<gPV, bPV>>>(dP, dV, dO, sh.B, sh.Nq, sh.Nkv, sh.S, sh.D);
}

static int run_correctness(const std::string& inputs_dir) {
    GQAShape sh{"correctness", 1, 4, 2, 128, 64};
    printf("[gqa] === correctness run (b=%d nq=%d nkv=%d s=%d d=%d) ===\n",
           sh.B, sh.Nq, sh.Nkv, sh.S, sh.D);

    Npy q, k, v, exp_out;
    if (!read_npy(inputs_dir + "/gqa_correctness_q_f16.npy", q)) return 1;
    if (!read_npy(inputs_dir + "/gqa_correctness_k_f16.npy", k)) return 1;
    if (!read_npy(inputs_dir + "/gqa_correctness_v_f16.npy", v)) return 1;
    if (!read_npy(inputs_dir + "/gqa_correctness_expected_f32.npy", exp_out)) return 1;

    size_t q_elems = (size_t)sh.B * sh.Nq  * sh.S * sh.D;
    size_t k_elems = (size_t)sh.B * sh.Nkv * sh.S * sh.D;
    size_t o_elems = q_elems;
    size_t s_elems = (size_t)sh.B * sh.Nq * sh.S * sh.S;
    if (q.data.size() != q_elems * 2 || k.data.size() != k_elems * 2) {
        fprintf(stderr, "shape mismatch\n"); return 1;
    }

    __half *dQ, *dK, *dV, *dP, *dO;
    float  *dS;
    CK(cudaMalloc(&dQ, q_elems * sizeof(__half)));
    CK(cudaMalloc(&dK, k_elems * sizeof(__half)));
    CK(cudaMalloc(&dV, k_elems * sizeof(__half)));
    CK(cudaMalloc(&dO, o_elems * sizeof(__half)));
    CK(cudaMalloc(&dS, s_elems * sizeof(float)));
    CK(cudaMalloc(&dP, s_elems * sizeof(__half)));

    CK(cudaMemcpy(dQ, q.data.data(), q.data.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, k.data.data(), k.data.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, v.data.data(), v.data.size(), cudaMemcpyHostToDevice));

    run_pipeline(sh, dQ, dK, dV, dS, dP, dO);
    CK(cudaDeviceSynchronize());

    std::vector<__half> ho(o_elems);
    CK(cudaMemcpy(ho.data(), dO, o_elems * sizeof(__half), cudaMemcpyDeviceToHost));

    // Compare f16 output (upcast to f32) against expected f32.
    const float* exp_ptr = reinterpret_cast<const float*>(exp_out.data.data());
    double max_abs = 0.0, max_rel = 0.0;
    double exp_max_abs = 0.0;
    for (size_t i = 0; i < o_elems; ++i) {
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

    // f16 tolerance from tolerances.py
    const double atol = 5e-3;
    const double rtol = 5e-3;
    // Typical PyTorch-style check: |got - want| <= atol + rtol * |want|
    bool ok = true;
    int bad = 0;
    for (size_t i = 0; i < o_elems && bad < 5; ++i) {
        float got = __half2float(ho[i]);
        float want = exp_ptr[i];
        double a = fabs((double)got - (double)want);
        if (a > atol + rtol * fabs((double)want)) {
            if (bad < 5) {
                fprintf(stderr, "  diff [%zu]: got=%g want=%g abs=%g\n", i, got, want, a);
            }
            bad++;
            ok = false;
        }
    }
    printf("[gqa] correctness: max_abs_err=%.3e max_rel_err=%.3e expected_max_abs=%.3e => %s\n",
           max_abs, max_rel, exp_max_abs, ok ? "OK" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dS); cudaFree(dP);
    return ok ? 0 : 2;
}

static int run_bench(const std::string& inputs_dir) {
    GQAShape sh{"llama3_8b", 1, 32, 8, 2048, 128};
    printf("\n[gqa] === bench run (b=%d nq=%d nkv=%d s=%d d=%d) ===\n",
           sh.B, sh.Nq, sh.Nkv, sh.S, sh.D);

    Npy q, k, v;
    if (!read_npy(inputs_dir + "/gqa_llama3_8b_q_f16.npy", q)) return 1;
    if (!read_npy(inputs_dir + "/gqa_llama3_8b_k_f16.npy", k)) return 1;
    if (!read_npy(inputs_dir + "/gqa_llama3_8b_v_f16.npy", v)) return 1;

    size_t q_elems = (size_t)sh.B * sh.Nq  * sh.S * sh.D;
    size_t k_elems = (size_t)sh.B * sh.Nkv * sh.S * sh.D;
    size_t o_elems = q_elems;
    size_t s_elems = (size_t)sh.B * sh.Nq * sh.S * sh.S;
    printf("[gqa] alloc: Q=%.1f MB K=%.1f MB V=%.1f MB S=%.1f MB P=%.1f MB O=%.1f MB\n",
           q_elems * 2.0 / 1e6, k_elems * 2.0 / 1e6, k_elems * 2.0 / 1e6,
           s_elems * 4.0 / 1e6, s_elems * 2.0 / 1e6, o_elems * 2.0 / 1e6);

    __half *dQ, *dK, *dV, *dP, *dO;
    float  *dS;
    CK(cudaMalloc(&dQ, q_elems * sizeof(__half)));
    CK(cudaMalloc(&dK, k_elems * sizeof(__half)));
    CK(cudaMalloc(&dV, k_elems * sizeof(__half)));
    CK(cudaMalloc(&dO, o_elems * sizeof(__half)));
    CK(cudaMalloc(&dS, s_elems * sizeof(float)));
    CK(cudaMalloc(&dP, s_elems * sizeof(__half)));

    CK(cudaMemcpy(dQ, q.data.data(), q.data.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, k.data.data(), k.data.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, v.data.data(), v.data.size(), cudaMemcpyHostToDevice));

    // flops: 4 * B * Nq * S^2 * D  (matches reference/flops.py)
    const double flops = 4.0 * sh.B * sh.Nq * (double)sh.S * sh.S * sh.D;
    printf("[gqa] flops/iter = %.3f GFLOPS\n", flops / 1e9);

    cudaEvent_t evs, eve;
    CK(cudaEventCreate(&evs));
    CK(cudaEventCreate(&eve));

    // warmup
    run_pipeline(sh, dQ, dK, dV, dS, dP, dO);
    CK(cudaDeviceSynchronize());

    FILE* csv = fopen("results.csv", "w");
    if (!csv) { fprintf(stderr, "cannot open results.csv\n"); return 1; }
    fprintf(csv, "impl,kernel,batch,seq,n_q,n_kv,d_head,iter,gpu_ms,tflops\n");

    constexpr int ITERS = 10;
    double times[ITERS];
    for (int i = 0; i < ITERS; ++i) {
        cudaEventRecord(evs);
        run_pipeline(sh, dQ, dK, dV, dS, dP, dO);
        cudaEventRecord(eve);
        cudaEventSynchronize(eve);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, evs, eve);
        times[i] = ms;
        double tflops = (flops / 1e12) / (ms * 1e-3);
        printf("[gqa] iter=%d gpu_ms=%.4f tflops=%.2f\n", i, ms, tflops);
        fprintf(csv, "cuda-attn-gqa,3kernel_wmma,%d,%d,%d,%d,%d,%d,%.6f,%.6f\n",
                sh.B, sh.S, sh.Nq, sh.Nkv, sh.D, i, ms, tflops);
    }
    fclose(csv);

    // Summary: best + median.
    double sorted[ITERS];
    for (int i = 0; i < ITERS; ++i) sorted[i] = times[i];
    for (int i = 0; i < ITERS; ++i)
        for (int j = i + 1; j < ITERS; ++j)
            if (sorted[j] < sorted[i]) { double t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }
    double best = sorted[0];
    double med  = 0.5 * (sorted[ITERS/2 - 1] + sorted[ITERS/2]);
    double best_tf = (flops / 1e12) / (best * 1e-3);
    double med_tf  = (flops / 1e12) / (med  * 1e-3);

    printf("\n[gqa] ===== SUMMARY (llama3_8b) =====\n");
    printf("[gqa] best   gpu_ms = %.4f  tflops = %.2f\n", best, best_tf);
    printf("[gqa] median gpu_ms = %.4f  tflops = %.2f\n", med,  med_tf);
    printf("[gqa] cuBLAS hgemm reference = 218.0 TF (from Wave 14.1)\n");
    printf("[gqa] fraction of hgemm peak = %.1f%%\n", 100.0 * best_tf / 218.0);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dS); cudaFree(dP);
    return 0;
}

int main(int argc, char** argv) {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("[gqa] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    std::string inputs_dir = "/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs";
    if (argc > 1) inputs_dir = argv[1];
    printf("[gqa] inputs dir: %s\n", inputs_dir.c_str());

    int rc = run_correctness(inputs_dir);
    if (rc != 0) {
        fprintf(stderr, "[gqa] correctness failed — skipping bench\n");
        return rc;
    }
    return run_bench(inputs_dir);
}
