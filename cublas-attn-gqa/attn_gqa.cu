// Wave 15.1 — cuBLAS-3-kernel GQA attention.
//
// Pipeline (per batch, per Q head h_q; KV head = h_q / groups):
//   Stage 1 (cublasGemmEx): scores[h_q] = Q[h_q] @ K[h_kv]^T   (S,d) x (d,S) -> (S,S) f32
//   Stage 2 (custom softmax): probs[h_q] = softmax(scores * scale, dim=-1)  -> (S,S) f16
//   Stage 3 (cublasGemmEx): out[h_q]    = probs[h_q] @ V[h_kv]   (S,S) x (S,d) -> (S,d) f16
//
// f16 matmul inputs/outputs, f32 compute accumulator, CUBLAS_GEMM_DEFAULT_TENSOR_OP.
//
// Per-stage cudaEvent timing: we record events around the entire n_q loop
// for each stage, so we can attribute bench time to QKᵀ vs softmax vs PV.
// This is Wave 15's central signal for the "3-kernel vs fused" tradeoff.
//
// Loads Q, K, V, and the expected output from
// analysis/wave15-attention-architecture/inputs/ as .npy (row-major,
// shape (B, n_heads, S, d), f16 for inputs, f32 for expected).
//
// ADR-0001 (cudaEvent timing). Matches the structure of
// cublas-half-precision/matmul.cu (canonical cuBLAS reference in this
// repo).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)
#define CB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS status %d at %s:%d\n", (int)s, __FILE__, __LINE__); exit(1); } } while(0)

// ---- Minimal .npy loader -----------------------------------------------
// Supports NPY v1.0 C-contiguous arrays of f16 / f32. Returns raw bytes
// + parsed shape. Throws (exits) on unsupported headers.

struct Npy {
    std::vector<int64_t> shape;
    std::string dtype;   // "<f2" or "<f4"
    std::vector<uint8_t> data;
    size_t elem_size() const { return dtype == "<f2" ? 2 : 4; }
    size_t num_elems() const {
        size_t n = 1; for (auto d : shape) n *= (size_t)d; return n;
    }
};

static Npy load_npy(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    uint8_t magic[6];
    if (fread(magic, 1, 6, f) != 6) { fprintf(stderr, "short read on %s\n", path); exit(1); }
    if (memcmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "%s: not a .npy file\n", path); exit(1);
    }
    uint8_t ver_major = 0, ver_minor = 0;
    if (fread(&ver_major, 1, 1, f) != 1) exit(1);
    if (fread(&ver_minor, 1, 1, f) != 1) exit(1);
    uint32_t header_len = 0;
    if (ver_major == 1) {
        uint16_t hl16 = 0;
        if (fread(&hl16, 2, 1, f) != 1) exit(1);
        header_len = hl16;
    } else {
        if (fread(&header_len, 4, 1, f) != 1) exit(1);
    }
    std::string header(header_len, ' ');
    if (fread(header.data(), 1, header_len, f) != header_len) exit(1);

    Npy out;

    // dtype
    auto dp = header.find("'descr':");
    if (dp == std::string::npos) dp = header.find("\"descr\":");
    auto sq1 = header.find("'", dp + 8);
    auto sq2 = header.find("'", sq1 + 1);
    out.dtype = header.substr(sq1 + 1, sq2 - sq1 - 1);

    // fortran_order
    if (header.find("'fortran_order': True") != std::string::npos) {
        fprintf(stderr, "%s: fortran_order not supported\n", path); exit(1);
    }

    // shape
    auto sp = header.find("'shape':");
    auto lp = header.find("(", sp);
    auto rp = header.find(")", lp);
    std::string shape_str = header.substr(lp + 1, rp - lp - 1);
    int64_t cur = 0; bool have = false;
    for (char c : shape_str) {
        if (c >= '0' && c <= '9') { cur = cur * 10 + (c - '0'); have = true; }
        else {
            if (have) { out.shape.push_back(cur); cur = 0; have = false; }
        }
    }
    if (have) out.shape.push_back(cur);

    size_t n_bytes = out.num_elems() * out.elem_size();
    out.data.resize(n_bytes);
    if (fread(out.data.data(), 1, n_bytes, f) != n_bytes) {
        fprintf(stderr, "%s: short read on data\n", path); exit(1);
    }
    fclose(f);
    return out;
}

// ---- softmax kernel launcher (defined in softmax.cu) -------------------

extern "C" void launch_row_softmax_scale(
    const float* d_scores,
    __half*      d_probs,
    int          num_rows,
    int          seq,
    float        scale,
    cudaStream_t stream);

// ---- Bench shape config ------------------------------------------------

struct Shape {
    const char* name;
    int batch;
    int seq;
    int n_q;
    int n_kv;
    int d_head;
    int groups() const { return n_q / n_kv; }
};

// ---- per-head stride helpers (row-major (B, H, S, D)) ------------------
static inline size_t q_head_offset(int b, int h, int S, int D, int nH) {
    return ((size_t)b * nH + h) * (size_t)S * D;
}

// ========================================================================
// Run one forward-pass. All buffers pre-allocated.
//
// Layout of intermediates:
//   d_scores : (B, n_q, S, S) f32
//   d_probs  : (B, n_q, S, S) f16
//   d_out    : (B, n_q, S, D) f16
//
// We loop B*n_q cublasGemmEx calls for stage 1 and stage 3 (option (a)
// from the task description; batched-with-broadcasting (option (b)) is a
// Wave 15.6 candidate).
//
// stage_ms[0..2] get filled with QKt / softmax / PV times (one full loop
// per stage, not per-head).
// ========================================================================

static void run_attention_once(
    cublasHandle_t cublas,
    cudaStream_t stream,
    const Shape& sh,
    const __half* d_Q,   // (B, n_q,  S, D) f16
    const __half* d_K,   // (B, n_kv, S, D) f16
    const __half* d_V,   // (B, n_kv, S, D) f16
    float*        d_scores,   // (B, n_q, S, S) f32
    __half*       d_probs,    // (B, n_q, S, S) f16
    __half*       d_out,      // (B, n_q, S, D) f16
    float stage_ms[3],
    cudaEvent_t ev_a, cudaEvent_t ev_b, cudaEvent_t ev_c, cudaEvent_t ev_d)
{
    const int B = sh.batch;
    const int S = sh.seq;
    const int Nq = sh.n_q;
    const int Nkv = sh.n_kv;
    const int D = sh.d_head;
    const int G = sh.groups();
    const float alpha = 1.0f, beta = 0.0f;

    // ---- Stage 1: Q @ K^T  -> scores (f32) --------------------------
    // For each (b, h_q): Q_head (S, D) in row-major; K_head (S, D) row-major.
    // We want (Q)(K^T) in row-major, i.e. C[i,j] = sum_k Q[i,k] * K[j,k].
    // Row-major Q is (S,D); row-major K is (S,D); row-major K^T is (D,S).
    // Using cuBLAS col-major with (B,A) swap trick:
    //   cuBLAS computes C_col = op(A_col) * op(B_col)
    //   We want row-major C(S,S) = Q(S,D) * K^T(D,S) -> C^T(S,S) = K(S,D) * Q^T(D,S)
    //   Treat C as col-major (S,S): pass op_A = N, A = K (leading dim D seen from col-major S)
    //     Hmm this requires more care. Let's redo cleanly:
    //
    // In cuBLAS column-major view:
    //   Row-major Q of shape (S, D) with stride D == column-major matrix
    //   of shape (D, S) with leading dim D.  Call this Q_col: (D, S).
    //   Similarly K_col: (D, S).
    //   We want row-major scores C(S, S) == column-major C_col(S, S).
    //   C[i,j] = <Q[i,:], K[j,:]> = sum_k Q[i,k] K[j,k] = sum_k Q_col[k,i] K_col[k,j]
    //   So  C_col[i,j] = sum_k (Q_col^T)[i,k] * K_col[k,j]
    //                  = (Q_col^T * K_col)[i,j]
    //   => op_A = T applied to Q_col (MxK with M=S, K=D), op_B = N applied to K_col (KxN with K=D, N=S)
    //   Leading dims: Q_col has lda = D; K_col has ldb = D; C_col has ldc = S.
    //
    // So: cublasGemmEx(T, N, M=S, N=S, K=D,
    //                  alpha, Q_head f16 CUDA_R_16F lda=D,
    //                  K_head f16 CUDA_R_16F ldb=D,
    //                  beta,  C_head f32 CUDA_R_32F ldc=S,
    //                  CUBLAS_COMPUTE_32F, DEFAULT_TENSOR_OP)

    CK(cudaEventRecord(ev_a, stream));
    for (int b = 0; b < B; ++b) {
        for (int hq = 0; hq < Nq; ++hq) {
            int hkv = hq / G;
            const __half* q_head = d_Q + q_head_offset(b, hq,  S, D, Nq);
            const __half* k_head = d_K + q_head_offset(b, hkv, S, D, Nkv);
            float*        s_head = d_scores + q_head_offset(b, hq, S, S, Nq);
            CB(cublasGemmEx(
                cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                S, S, D,
                &alpha,
                k_head, CUDA_R_16F, D,   // A = K_col (D,S)
                q_head, CUDA_R_16F, D,   // B = Q_col (D,S)
                &beta,
                s_head, CUDA_R_32F, S,   // C_col writes so row-major reader sees scores[i,j] = Q[i]·K[j]
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }
    CK(cudaEventRecord(ev_b, stream));

    // ---- Stage 2: softmax(scores * scale) -> probs (f16) -----------
    int num_rows = B * Nq * S;
    float scale = 1.0f / sqrtf((float)D);
    launch_row_softmax_scale(d_scores, d_probs, num_rows, S, scale, stream);
    CK(cudaEventRecord(ev_c, stream));

    // ---- Stage 3: probs @ V -> out (f16) ---------------------------
    // probs[b,h_q]: (S, S) row-major.  V_head[b,h_kv]: (S, D) row-major.
    // Row-major output (S, D) = probs(S,S) * V(S,D).
    // Column-major view: probs_col is (S, S) lda=S;  V_col is (D, S) lda=D.
    // row-major out O(S,D) == col-major O_col(D,S).
    //   O[i,j]_row = sum_k probs[i,k] V[k,j]
    //             = sum_k probs_col[k,i] V_col[j,k]
    //   O_col[j,i] = sum_k V_col[j,k] * probs_col[k,i]
    //              = (V_col * probs_col)[j,i]
    //   => gemm(N, N, M=D, N=S, K=S, A=V_col (DxS) lda=D, B=probs_col(SxS) ldb=S,
    //           C=O_col(DxS) ldc=D)
    // (ev_c already recorded above right after softmax launch — it marks
    //  the softmax-end / PV-start boundary.)
    for (int b = 0; b < B; ++b) {
        for (int hq = 0; hq < Nq; ++hq) {
            int hkv = hq / G;
            const __half* v_head = d_V + q_head_offset(b, hkv, S, D, Nkv);
            const __half* p_head = d_probs + q_head_offset(b, hq, S, S, Nq);
            __half*       o_head = d_out + q_head_offset(b, hq, S, D, Nq);
            CB(cublasGemmEx(
                cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                D, S, S,
                &alpha,
                v_head, CUDA_R_16F, D,
                p_head, CUDA_R_16F, S,
                &beta,
                o_head, CUDA_R_16F, D,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }
    CK(cudaEventRecord(ev_d, stream));

    CK(cudaEventSynchronize(ev_d));
    float ms_ab = 0.0f, ms_bc = 0.0f, ms_cd = 0.0f;
    CK(cudaEventElapsedTime(&ms_ab, ev_a, ev_b));
    CK(cudaEventElapsedTime(&ms_bc, ev_b, ev_c));
    CK(cudaEventElapsedTime(&ms_cd, ev_c, ev_d));
    stage_ms[0] = ms_ab;
    stage_ms[1] = ms_bc;
    stage_ms[2] = ms_cd;
}

// ========================================================================

static double gqa_flops(const Shape& s) {
    // 4 * B * n_q * seq^2 * d_head  (see reference/flops.py)
    return 4.0 * s.batch * s.n_q * (double)s.seq * s.seq * s.d_head;
}

static void run_cell(cublasHandle_t cublas, cudaStream_t stream, const Shape& sh,
                     const char* q_path, const char* k_path, const char* v_path,
                     const char* exp_path, int iters, FILE* csv,
                     bool check_correctness, float f16_atol, float f16_rtol)
{
    sh.n_q % sh.n_kv == 0 || (fprintf(stderr, "shape invalid\n"), exit(1), 0);

    printf("\n[cublas-attn-gqa] ===== shape=%s B=%d S=%d n_q=%d n_kv=%d d=%d =====\n",
           sh.name, sh.batch, sh.seq, sh.n_q, sh.n_kv, sh.d_head);

    // Load inputs
    printf("[cublas-attn-gqa] loading Q/K/V/expected ...\n");
    Npy q = load_npy(q_path);
    Npy k = load_npy(k_path);
    Npy v = load_npy(v_path);
    Npy e = load_npy(exp_path);
    printf("[cublas-attn-gqa] Q: shape=(%ld,%ld,%ld,%ld) dtype=%s\n",
           (long)q.shape[0], (long)q.shape[1], (long)q.shape[2], (long)q.shape[3], q.dtype.c_str());
    printf("[cublas-attn-gqa] K: shape=(%ld,%ld,%ld,%ld) dtype=%s\n",
           (long)k.shape[0], (long)k.shape[1], (long)k.shape[2], (long)k.shape[3], k.dtype.c_str());
    printf("[cublas-attn-gqa] V: shape=(%ld,%ld,%ld,%ld) dtype=%s\n",
           (long)v.shape[0], (long)v.shape[1], (long)v.shape[2], (long)v.shape[3], v.dtype.c_str());
    printf("[cublas-attn-gqa] Exp: shape=(%ld,%ld,%ld,%ld) dtype=%s\n",
           (long)e.shape[0], (long)e.shape[1], (long)e.shape[2], (long)e.shape[3], e.dtype.c_str());

    // Device buffers
    size_t q_bytes = sh.batch * sh.n_q  * sh.seq * sh.d_head * sizeof(__half);
    size_t k_bytes = sh.batch * sh.n_kv * sh.seq * sh.d_head * sizeof(__half);
    size_t v_bytes = k_bytes;
    size_t o_bytes = q_bytes;
    size_t s_bytes = (size_t)sh.batch * sh.n_q * sh.seq * sh.seq * sizeof(float);
    size_t p_bytes = (size_t)sh.batch * sh.n_q * sh.seq * sh.seq * sizeof(__half);
    printf("[cublas-attn-gqa] alloc: Q=%.1fMB K=%.1fMB V=%.1fMB out=%.1fMB scores=%.1fMB probs=%.1fMB\n",
           q_bytes/1e6, k_bytes/1e6, v_bytes/1e6, o_bytes/1e6, s_bytes/1e6, p_bytes/1e6);

    __half *dQ, *dK, *dV, *dO, *dP;
    float  *dS;
    CK(cudaMalloc(&dQ, q_bytes));
    CK(cudaMalloc(&dK, k_bytes));
    CK(cudaMalloc(&dV, v_bytes));
    CK(cudaMalloc(&dO, o_bytes));
    CK(cudaMalloc(&dS, s_bytes));
    CK(cudaMalloc(&dP, p_bytes));
    CK(cudaMemcpy(dQ, q.data.data(), q_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, k.data.data(), k_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, v.data.data(), v_bytes, cudaMemcpyHostToDevice));

    // Events for per-stage + overall timing
    cudaEvent_t ev_a, ev_b, ev_c, ev_d;
    CK(cudaEventCreate(&ev_a));
    CK(cudaEventCreate(&ev_b));
    CK(cudaEventCreate(&ev_c));
    CK(cudaEventCreate(&ev_d));

    // --- Warmup ---
    float ms_stage[3];
    run_attention_once(cublas, stream, sh, dQ, dK, dV, dS, dP, dO, ms_stage,
                       ev_a, ev_b, ev_c, ev_d);
    printf("[cublas-attn-gqa] warmup total=%.3fms (QKt=%.3f softmax=%.3f PV=%.3f)\n",
           ms_stage[0]+ms_stage[1]+ms_stage[2], ms_stage[0], ms_stage[1], ms_stage[2]);

    // --- Correctness (against f32 expected; tol from reference/tolerances.py) ---
    if (check_correctness) {
        std::vector<__half> hO_h(sh.batch * sh.n_q * sh.seq * sh.d_head);
        CK(cudaMemcpy(hO_h.data(), dO, o_bytes, cudaMemcpyDeviceToHost));
        const float* exp32 = (const float*)e.data.data();
        size_t n = hO_h.size();
        double max_abs = 0.0, max_rel = 0.0;
        size_t n_bad = 0;
        // numpy-style allclose: |got - want| <= atol + rtol * |want|
        for (size_t i = 0; i < n; ++i) {
            float got  = __half2float(hO_h[i]);
            float want = exp32[i];
            float abs_e = fabsf(got - want);
            float rel_e = abs_e / fmaxf(fabsf(want), 1e-6f);
            float thresh = f16_atol + f16_rtol * fabsf(want);
            if (abs_e > thresh) ++n_bad;
            if (abs_e > max_abs) max_abs = abs_e;
            if (rel_e > max_rel) max_rel = rel_e;
        }
        printf("[cublas-attn-gqa] correctness: max_abs=%.3e max_rel=%.3e bad=%zu/%zu (atol=%.0e rtol=%.0e)\n",
               max_abs, max_rel, n_bad, n, f16_atol, f16_rtol);
        bool ok = (n_bad == 0);
        printf("[cublas-attn-gqa] correctness: %s\n", ok ? "OK" : "FAIL");
        if (!ok && sh.seq <= 256) {
            // On the small correctness shape, bail out if we fail.
            fprintf(stderr, "correctness failed; refusing to bench\n");
            cudaFree(dQ); cudaFree(dK); cudaFree(dV);
            cudaFree(dO); cudaFree(dS); cudaFree(dP);
            return;
        }
    }

    // --- Timed iters ---
    double flops = gqa_flops(sh);
    double best_total_ms = 1e30;
    double sum_stage_ms[3] = {0,0,0};
    for (int i = 0; i < iters; ++i) {
        run_attention_once(cublas, stream, sh, dQ, dK, dV, dS, dP, dO, ms_stage,
                           ev_a, ev_b, ev_c, ev_d);
        double total = ms_stage[0] + ms_stage[1] + ms_stage[2];
        double tflops = (flops / 1e12) / (total / 1000.0);
        printf("[cublas-attn-gqa] shape=%s iter=%d total=%.3fms QKt=%.3f softmax=%.3f PV=%.3f tflops=%.3f\n",
               sh.name, i, total, ms_stage[0], ms_stage[1], ms_stage[2], tflops);
        // CSV per-stage + total
        fprintf(csv, "cublas-attn-gqa,QKt,%d,%d,%d,%d,%d,%d,%.6f,%.6f\n",
                sh.batch, sh.seq, sh.n_q, sh.n_kv, sh.d_head, i, ms_stage[0], 0.0);
        fprintf(csv, "cublas-attn-gqa,softmax,%d,%d,%d,%d,%d,%d,%.6f,%.6f\n",
                sh.batch, sh.seq, sh.n_q, sh.n_kv, sh.d_head, i, ms_stage[1], 0.0);
        fprintf(csv, "cublas-attn-gqa,PV,%d,%d,%d,%d,%d,%d,%.6f,%.6f\n",
                sh.batch, sh.seq, sh.n_q, sh.n_kv, sh.d_head, i, ms_stage[2], 0.0);
        fprintf(csv, "cublas-attn-gqa,total,%d,%d,%d,%d,%d,%d,%.6f,%.6f\n",
                sh.batch, sh.seq, sh.n_q, sh.n_kv, sh.d_head, i, total, tflops);
        if (total < best_total_ms) best_total_ms = total;
        for (int s = 0; s < 3; ++s) sum_stage_ms[s] += ms_stage[s];
    }
    double avg_stage[3] = { sum_stage_ms[0]/iters, sum_stage_ms[1]/iters, sum_stage_ms[2]/iters };
    double avg_total = avg_stage[0] + avg_stage[1] + avg_stage[2];
    double best_tf  = (flops / 1e12) / (best_total_ms / 1000.0);
    double avg_tf   = (flops / 1e12) / (avg_total     / 1000.0);
    printf("[cublas-attn-gqa] shape=%s SUMMARY best_total=%.3fms avg_total=%.3fms best_TF=%.3f avg_TF=%.3f\n",
           sh.name, best_total_ms, avg_total, best_tf, avg_tf);
    printf("[cublas-attn-gqa] shape=%s STAGES  avg_QKt=%.3fms(%.1f%%) avg_softmax=%.3fms(%.1f%%) avg_PV=%.3fms(%.1f%%)\n",
           sh.name,
           avg_stage[0], 100.0*avg_stage[0]/avg_total,
           avg_stage[1], 100.0*avg_stage[1]/avg_total,
           avg_stage[2], 100.0*avg_stage[2]/avg_total);

    cudaEventDestroy(ev_a); cudaEventDestroy(ev_b);
    cudaEventDestroy(ev_c); cudaEventDestroy(ev_d);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV);
    cudaFree(dO); cudaFree(dS); cudaFree(dP);
}

// ========================================================================
int main(int argc, char** argv) {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("[cublas-attn-gqa] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    cublasHandle_t cublas; CB(cublasCreate(&cublas));
    int ver = 0; CB(cublasGetVersion(cublas, &ver));
    printf("[cublas-attn-gqa] cuBLAS version: %d.%d.%d\n",
           ver / 10000, (ver / 100) % 100, ver % 100);
    CB(cublasSetMathMode(cublas, CUBLAS_DEFAULT_MATH));

    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));
    CB(cublasSetStream(cublas, stream));

    FILE* csv = fopen("results.csv", "w");
    if (!csv) { fprintf(stderr, "cannot open results.csv\n"); return 1; }
    fprintf(csv, "impl,kernel,batch,seq,n_q,n_kv,d_head,iter,gpu_ms,tflops\n");

    const char* INPUTS = "/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs";
    char q_path[512], k_path[512], v_path[512], e_path[512];

    // --- Correctness shape ---
    Shape correctness = {"correctness", 1, 128, 4, 2, 64};
    snprintf(q_path, sizeof(q_path), "%s/gqa_correctness_q_f16.npy", INPUTS);
    snprintf(k_path, sizeof(k_path), "%s/gqa_correctness_k_f16.npy", INPUTS);
    snprintf(v_path, sizeof(v_path), "%s/gqa_correctness_v_f16.npy", INPUTS);
    snprintf(e_path, sizeof(e_path), "%s/gqa_correctness_expected_f32.npy", INPUTS);
    run_cell(cublas, stream, correctness, q_path, k_path, v_path, e_path,
             /*iters=*/3, csv, /*check=*/true, 5e-3f, 5e-3f);

    // --- Llama-3 8B bench shape ---
    Shape llama3 = {"llama3_8b", 1, 2048, 32, 8, 128};
    snprintf(q_path, sizeof(q_path), "%s/gqa_llama3_8b_q_f16.npy", INPUTS);
    snprintf(k_path, sizeof(k_path), "%s/gqa_llama3_8b_k_f16.npy", INPUTS);
    snprintf(v_path, sizeof(v_path), "%s/gqa_llama3_8b_v_f16.npy", INPUTS);
    snprintf(e_path, sizeof(e_path), "%s/gqa_llama3_8b_expected_f32.npy", INPUTS);
    run_cell(cublas, stream, llama3, q_path, k_path, v_path, e_path,
             /*iters=*/10, csv, /*check=*/true, 5e-3f, 5e-3f);

    fclose(csv);
    cudaStreamDestroy(stream);
    cublasDestroy(cublas);
    return 0;
}
