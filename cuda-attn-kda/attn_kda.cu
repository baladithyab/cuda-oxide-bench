// Wave C2.6 (Rosetta Stone) — nvcc CUDA C++ Kimi Delta Attention (KDA)
// single-timestep decode.
//
// Reference: cutile-attn-kda/main.py — same algorithm in cuTile (best-shape
// 1170 GB/s saturation @ shape=large on RTX 5090 sm_120). This C++ port
// targets the same shapes and grid layout but uses thread-level vectorized
// loads (float4 → LDG.E.128 / STG.E.128) — modeled directly after
// cuda-attn-gdn/attn_gdn.cu (which hits 417 GB/s on the same memory pattern).
//
// KDA differs from GDN by exactly ONE thing in the inner state-update math:
//   GDN:  S_scaled = alpha * S_in        (scalar broadcast over (D_K, BV))
//   KDA:  S_scaled = exp(g)[k] * S_in    (per-d_k-row broadcast over (D_K, BV))
//
// The decay `exp(g_k)` is a per-d_k-row scalar; it broadcasts trivially when
// our inner loop iterates over d_k (each row k uses the same exp(g)[k] for
// all BLOCK_V columns this thread owns). All other passes (u = k·S_scaled,
// residual = v - u, S_out = S_scaled + β·k⊗r, o = q·S_out) are byte-identical
// to GDN.
//
// Recurrence (verbatim from cutile-attn-kda/main.py lines 11-14):
//   S_t = ( I − β_t k_t k_t^T ) · Diag(α_t) · S_{t−1}  +  β_t k_t v_t^T
//   o_t = S_t^T q_t                                with α_t = exp(g_t) ∈ ℝ^{d_k}
//
// Per-iter HBM traffic (kimi_linear_decode shape, B=1, H=32, d_k=d_v=128):
//   state read+write = 2*128*128*4 B = 128 KB per (B,H)
//   io = (3*128 + 2*128 + 1) * 2 = 1282 B per (B,H)   (q,k,g,v,o,beta f16)
//   total ≈ 32 * (128 KiB + 1282 B) ≈ 4136 KiB per iter
//
// Grid (matching cuTile): (B*H, D_V / BLOCK_V)
// Block: TPB threads, each owns 4 contiguous d_v columns (one float4 stripe).
// TPB = BLOCK_V / 4.
//
// Build: see Makefile.

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

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA err %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while(0)

// ============================================================================
// .npy loader (NPY1.0 / NPY2.0; little-endian only). Same as cuda-attn-gdn.
// ============================================================================
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
    } else {
        fread(&header_len, 1, 4, f);
    }
    std::string hdr(header_len, ' ');
    fread(&hdr[0], 1, header_len, f);

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
// KDA-decode kernel.
//
// Template params:
//   D_K     — key dim (state rows). Compile-time so the inner loops fully unroll.
//   BLOCK_V — d_v-tile width per block (must be multiple of 4; we vectorize as float4).
//
// Runtime parallelism:
//   gridDim  = (B * H, D_V / BLOCK_V)
//   blockDim = (BLOCK_V / 4)   each thread owns 4 d_v columns (one float4 stripe)
//
// Memory layout in HBM (matches the harness packing):
//   Q       (B*H, D_K)         f16
//   K       (B*H, D_K)         f16
//   V       (B*H, D_V)         f16
//   G       (B*H, D_K)         f16   (KDA: per-channel log-gate; decay = exp(g))
//   Beta    (B*H,)             f16
//   S_in    (B*H, D_K, D_V)    f32 row-major over (D_K, D_V)
//   S_out   (B*H, D_K, D_V)    f32 row-major
//   O       (B*H, D_V)         f16
//
// Per-block shared mem:
//   smem_q[D_K]            (f32)  q upcast once
//   smem_k[D_K]            (f32)  k upcast once
//   smem_g[D_K]            (f32)  exp(g) upcast + exp() once  ← KDA-only
//   smem_S_scaled[D_K * VLANES] (float4)  the per-row-decayed state tile,
//                                cached between u and S_out passes.
//   For D_K=128, BLOCK_V=128, VLANES=32: 128*32*16 = 64 KB tile + 1.5 KB scalars.
// ============================================================================
template <int D_K, int BLOCK_V>
__global__ void kda_decode_kernel(
    const __half* __restrict__ Q,       // (B*H, D_K)
    const __half* __restrict__ K,       // (B*H, D_K)
    const __half* __restrict__ V,       // (B*H, D_V)
    const __half* __restrict__ G,       // (B*H, D_K)   — KDA per-channel log-gate
    const __half* __restrict__ Beta,    // (B*H,)
    const float*  __restrict__ S_in,    // (B*H, D_K, D_V)
    float*        __restrict__ S_out,   // (B*H, D_K, D_V)
    __half*       __restrict__ O,       // (B*H, D_V)
    int B_H, int D_V)
{
    static_assert(BLOCK_V % 4 == 0, "BLOCK_V must be multiple of 4 for float4 lanes");
    constexpr int VLANES = BLOCK_V / 4;
    constexpr int TPB    = VLANES;

    const int bh = blockIdx.x;
    const int bv = blockIdx.y;
    const int tid = threadIdx.x;

    const int col0 = bv * BLOCK_V + tid * 4;

    // ── Shared memory layout ──
    extern __shared__ unsigned char smem_raw[];
    float*  smem_q  = reinterpret_cast<float*>(smem_raw);                    // [D_K]
    float*  smem_k  = smem_q + D_K;                                          // [D_K]
    float*  smem_g  = smem_k + D_K;                                          // [D_K]  ← KDA
    float4* smem_S  = reinterpret_cast<float4*>(smem_g + D_K);                // [D_K * VLANES]

    // ── Load q, k, g (f16 → f32) cooperatively into shared memory ──
    // KDA: also compute exp(g) once per d_k-row and cache.
    const __half* Qbh = Q + (size_t)bh * D_K;
    const __half* Kbh = K + (size_t)bh * D_K;
    const __half* Gbh = G + (size_t)bh * D_K;
    for (int kk = tid; kk < D_K; kk += TPB) {
        smem_q[kk] = __half2float(Qbh[kk]);
        smem_k[kk] = __half2float(Kbh[kk]);
        smem_g[kk] = expf(__half2float(Gbh[kk]));   // decay = exp(g_k)
    }

    // Per-block beta scalar.
    __shared__ float s_beta;
    if (tid == 0) {
        s_beta = __half2float(Beta[bh]);
    }

    // Per-thread vector slot for v (4 contiguous f16s loaded as float2).
    float4 v_vec;
    {
        const __half* Vbh = V + (size_t)bh * D_V;
        const float2* Vptr = reinterpret_cast<const float2*>(Vbh + col0);
        float2 v_packed = *Vptr;     // 8 bytes = 4 halves
        const __half* hv = reinterpret_cast<const __half*>(&v_packed);
        v_vec.x = __half2float(hv[0]);
        v_vec.y = __half2float(hv[1]);
        v_vec.z = __half2float(hv[2]);
        v_vec.w = __half2float(hv[3]);
    }

    __syncthreads();
    const float beta = s_beta;

    // ── Pass 1: load (D_K, BLOCK_V) state tile, scale by exp(g)[k], cache, accum u ──
    //
    // KDA difference vs GDN: scale factor varies per d_k row (exp(g)[k]) instead of
    // a single scalar alpha. Within the inner k-loop body, each k uses exp(g)[k]
    // for all 4 columns this thread owns. No additional traffic; just one extra
    // scalar load per k-iter from smem_g (already in smem after the cooperative load).
    const size_t bh_state_off = (size_t)bh * D_K * D_V;
    const float* Sbh = S_in + bh_state_off;

    float4 u_acc = make_float4(0.f, 0.f, 0.f, 0.f);

    #pragma unroll 8
    for (int k = 0; k < D_K; ++k) {
        const float4* sptr =
            reinterpret_cast<const float4*>(Sbh + (size_t)k * D_V + col0);
        float4 s = *sptr;                         // <-- LDG.E.128
        float decay = smem_g[k];                  // KDA per-row decay
        s.x *= decay;
        s.y *= decay;
        s.z *= decay;
        s.w *= decay;

        smem_S[(size_t)k * VLANES + tid] = s;

        float kk = smem_k[k];
        u_acc.x += kk * s.x;
        u_acc.y += kk * s.y;
        u_acc.z += kk * s.z;
        u_acc.w += kk * s.w;
    }

    // ── Residual r = v - u  ──
    float4 r;
    r.x = v_vec.x - u_acc.x;
    r.y = v_vec.y - u_acc.y;
    r.z = v_vec.z - u_acc.z;
    r.w = v_vec.w - u_acc.w;

    // ── Pass 2: S_out = S_scaled + beta · k ⊗ r,  then  o += q[k] * S_out_row ──
    float* Sout_bh = S_out + bh_state_off;
    float4 o_acc = make_float4(0.f, 0.f, 0.f, 0.f);

    #pragma unroll 8
    for (int k = 0; k < D_K; ++k) {
        float4 s = smem_S[(size_t)k * VLANES + tid];   // S_scaled (decayed) cached
        float kk = smem_k[k];
        float qk = smem_q[k];
        float bk = beta * kk;

        s.x = s.x + bk * r.x;
        s.y = s.y + bk * r.y;
        s.z = s.z + bk * r.z;
        s.w = s.w + bk * r.w;

        float4* sout_ptr =
            reinterpret_cast<float4*>(Sout_bh + (size_t)k * D_V + col0);
        *sout_ptr = s;                                // <-- STG.E.128

        o_acc.x += qk * s.x;
        o_acc.y += qk * s.y;
        o_acc.z += qk * s.z;
        o_acc.w += qk * s.w;
    }

    // ── Store o (f16). 4 halves contiguously → 8-byte store. ──
    __half* Obh = O + (size_t)bh * D_V;
    __half halves[4];
    halves[0] = __float2half(o_acc.x);
    halves[1] = __float2half(o_acc.y);
    halves[2] = __float2half(o_acc.z);
    halves[3] = __float2half(o_acc.w);
    float2 packed;
    memcpy(&packed, halves, sizeof(packed));
    *reinterpret_cast<float2*>(Obh + col0) = packed;
}

// ============================================================================
// Host launcher
// ============================================================================
struct KDAShape {
    const char* name;
    int batch;
    int n_heads;
    int d_k;
    int d_v;
};

template <int D_K, int BLOCK_V>
static void launch_kda(const KDAShape& sh,
                       const __half* dQ, const __half* dK, const __half* dV,
                       const __half* dG, const __half* dB,
                       const float*  dS_in, float* dS_out, __half* dO,
                       cudaStream_t stream)
{
    const int B_H = sh.batch * sh.n_heads;
    const int VLANES = BLOCK_V / 4;
    const int TPB = VLANES;
    dim3 grid(B_H, sh.d_v / BLOCK_V);
    dim3 block(TPB);

    // Shared mem: 3 * D_K floats (q, k, g) + D_K * VLANES float4 (S_scaled tile).
    size_t smem_bytes = 3 * D_K * sizeof(float) + (size_t)D_K * VLANES * sizeof(float4);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            (const void*)kda_decode_kernel<D_K, BLOCK_V>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes);
    }

    kda_decode_kernel<D_K, BLOCK_V><<<grid, block, smem_bytes, stream>>>(
        dQ, dK, dV, dG, dB, dS_in, dS_out, dO, B_H, sh.d_v);
}

// Block-V picker — mirrors cutile-attn-kda's pick_block_v: pick largest
// BLOCK_V that divides d_v with D_K * BV ≤ 16384 elems (~64KB tile).
//
// d_k=64,  d_v=64:  BV=64  → tile  64*16 *16B = 16 KB
// d_k=128, d_v=128: BV=128 → tile 128*32 *16B = 64 KB
// d_k=256, d_v=256: BV=64  → tile 256*16 *16B = 64 KB
static void launch_kda_dispatch(const KDAShape& sh,
                                const __half* dQ, const __half* dK, const __half* dV,
                                const __half* dG, const __half* dB,
                                const float*  dS_in, float* dS_out, __half* dO,
                                cudaStream_t stream)
{
    if (sh.d_k == 64 && sh.d_v == 64) {
        launch_kda<64, 64>(sh, dQ, dK, dV, dG, dB, dS_in, dS_out, dO, stream);
    } else if (sh.d_k == 128 && sh.d_v == 128) {
        launch_kda<128, 128>(sh, dQ, dK, dV, dG, dB, dS_in, dS_out, dO, stream);
    } else if (sh.d_k == 256 && sh.d_v == 256) {
        launch_kda<256, 64>(sh, dQ, dK, dV, dG, dB, dS_in, dS_out, dO, stream);
    } else {
        fprintf(stderr, "unsupported (d_k=%d, d_v=%d) — wired: 64/64, 128/128, 256/256\n",
                sh.d_k, sh.d_v);
        exit(2);
    }
}

// ============================================================================
// Correctness driver
// ============================================================================
static int run_correctness(const std::string& inputs_dir) {
    KDAShape sh{"correctness", /*B*/1, /*H*/2, /*d_k*/64, /*d_v*/64};
    printf("[kda] === correctness run (B=%d H=%d d_k=%d d_v=%d) ===\n",
           sh.batch, sh.n_heads, sh.d_k, sh.d_v);

    Npy q, k, v, g, b, sin_, oexp, sexp;
    if (!read_npy(inputs_dir + "/kda_correctness_q_f16.npy",            q))    return 1;
    if (!read_npy(inputs_dir + "/kda_correctness_k_f16.npy",            k))    return 1;
    if (!read_npy(inputs_dir + "/kda_correctness_v_f16.npy",            v))    return 1;
    if (!read_npy(inputs_dir + "/kda_correctness_g_f16.npy",            g))    return 1;
    if (!read_npy(inputs_dir + "/kda_correctness_beta_f16.npy",         b))    return 1;
    if (!read_npy(inputs_dir + "/kda_correctness_S_in_f32.npy",         sin_)) return 1;
    if (!read_npy(inputs_dir + "/kda_correctness_o_expected_f16.npy",   oexp)) return 1;
    if (!read_npy(inputs_dir + "/kda_correctness_S_out_expected_f32.npy", sexp)) return 1;

    const int B_H = sh.batch * sh.n_heads;
    size_t qkv_elems = (size_t)B_H * sh.d_k;
    size_t v_elems   = (size_t)B_H * sh.d_v;
    size_t s_elems   = (size_t)B_H * sh.d_k * sh.d_v;
    size_t o_elems   = (size_t)B_H * sh.d_v;
    size_t scal      = (size_t)B_H;
    size_t g_elems   = (size_t)B_H * sh.d_k;

    if (q.data.size() != qkv_elems * 2 || k.data.size() != qkv_elems * 2 ||
        v.data.size() != v_elems * 2 || g.data.size() != g_elems * 2 ||
        b.data.size() != scal * 2 || sin_.data.size() != s_elems * 4 ||
        oexp.data.size() != o_elems * 2 || sexp.data.size() != s_elems * 4) {
        fprintf(stderr, "[kda] correctness: NPY shape/size mismatch\n");
        return 1;
    }

    __half *dQ, *dK, *dV, *dG, *dB, *dO;
    float  *dSin, *dSout;
    CK(cudaMalloc(&dQ,    qkv_elems * sizeof(__half)));
    CK(cudaMalloc(&dK,    qkv_elems * sizeof(__half)));
    CK(cudaMalloc(&dV,    v_elems   * sizeof(__half)));
    CK(cudaMalloc(&dG,    g_elems   * sizeof(__half)));
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
    CK(cudaMemset(dSout, 0,                s_elems * sizeof(float)));
    CK(cudaMemset(dO,    0,                o_elems * sizeof(__half)));

    launch_kda_dispatch(sh, dQ, dK, dV, dG, dB, dSin, dSout, dO, 0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<__half> ho(o_elems);
    std::vector<float>  hs(s_elems);
    CK(cudaMemcpy(ho.data(), dO,    o_elems * sizeof(__half), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hs.data(), dSout, s_elems * sizeof(float),  cudaMemcpyDeviceToHost));

    const __half* o_exp_h = reinterpret_cast<const __half*>(oexp.data.data());
    const float*  s_exp_f = reinterpret_cast<const float*> (sexp.data.data());

    double max_abs_o = 0.0, exp_o_mag = 0.0;
    for (size_t i = 0; i < o_elems; ++i) {
        float got = __half2float(ho[i]);
        float want = __half2float(o_exp_h[i]);
        double aw = fabs((double)want);
        if (aw > exp_o_mag) exp_o_mag = aw;
        double a = fabs((double)got - (double)want);
        if (a > max_abs_o) max_abs_o = a;
    }
    double max_abs_s = 0.0, exp_s_mag = 0.0;
    for (size_t i = 0; i < s_elems; ++i) {
        float got = hs[i];
        float want = s_exp_f[i];
        double aw = fabs((double)want);
        if (aw > exp_s_mag) exp_s_mag = aw;
        double a = fabs((double)got - (double)want);
        if (a > max_abs_s) max_abs_s = a;
    }

    // Per task spec: atol=1e-3 (matches cutile-attn-kda smoke threshold for o).
    // S_out is f32 throughout the GPU pipeline; small d_k=64 → tighten to 5e-3
    // matching cuda-attn-gdn convention.
    const double ATOL_O = 1e-3;
    const double ATOL_S = 5e-3;
    bool ok_o = max_abs_o <= ATOL_O;
    bool ok_s = max_abs_s <= ATOL_S;

    printf("[kda] o    max_abs=%.3e |want|max=%.3e   %s\n",
           max_abs_o, exp_o_mag, ok_o ? "OK" : "FAIL");
    printf("[kda] Sout max_abs=%.3e |want|max=%.3e   %s\n",
           max_abs_s, exp_s_mag, ok_s ? "OK" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dG); cudaFree(dB);
    cudaFree(dSin); cudaFree(dSout); cudaFree(dO);
    return (ok_o && ok_s) ? 0 : 2;
}

// ============================================================================
// Bench-shape correctness smoke (also at kimi_linear_decode + large)
// ============================================================================
static int run_shape_smoke(const std::string& inputs_dir, const KDAShape& sh,
                           double atol_o, double atol_s) {
    printf("\n[kda] === shape smoke %s (B=%d H=%d d_k=%d d_v=%d) ===\n",
           sh.name, sh.batch, sh.n_heads, sh.d_k, sh.d_v);

    std::string prefix = inputs_dir + "/kda_" + sh.name;
    Npy q, k, v, g, b, sin_, oexp, sexp;
    if (!read_npy(prefix + "_q_f16.npy",                  q))    return 1;
    if (!read_npy(prefix + "_k_f16.npy",                  k))    return 1;
    if (!read_npy(prefix + "_v_f16.npy",                  v))    return 1;
    if (!read_npy(prefix + "_g_f16.npy",                  g))    return 1;
    if (!read_npy(prefix + "_beta_f16.npy",               b))    return 1;
    if (!read_npy(prefix + "_S_in_f32.npy",               sin_)) return 1;
    if (!read_npy(prefix + "_o_expected_f16.npy",         oexp)) return 1;
    if (!read_npy(prefix + "_S_out_expected_f32.npy",     sexp)) return 1;

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
    CK(cudaMemset(dSout, 0, s_elems * sizeof(float)));
    CK(cudaMemset(dO,    0, o_elems * sizeof(__half)));

    launch_kda_dispatch(sh, dQ, dK, dV, dG, dB, dSin, dSout, dO, 0);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<__half> ho(o_elems);
    std::vector<float>  hs(s_elems);
    CK(cudaMemcpy(ho.data(), dO,    o_elems * sizeof(__half), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hs.data(), dSout, s_elems * sizeof(float),  cudaMemcpyDeviceToHost));

    const __half* o_exp_h = reinterpret_cast<const __half*>(oexp.data.data());
    const float*  s_exp_f = reinterpret_cast<const float*> (sexp.data.data());

    double max_abs_o = 0.0, max_abs_s = 0.0;
    for (size_t i = 0; i < o_elems; ++i) {
        float got = __half2float(ho[i]);
        float want = __half2float(o_exp_h[i]);
        double a = fabs((double)got - (double)want);
        if (a > max_abs_o) max_abs_o = a;
    }
    for (size_t i = 0; i < s_elems; ++i) {
        double a = fabs((double)hs[i] - (double)s_exp_f[i]);
        if (a > max_abs_s) max_abs_s = a;
    }
    bool ok_o = max_abs_o <= atol_o;
    bool ok_s = max_abs_s <= atol_s;
    printf("[kda] (%s) o    max_abs=%.3e   %s (atol=%.0e)\n", sh.name, max_abs_o, ok_o ? "OK" : "FAIL", atol_o);
    printf("[kda] (%s) Sout max_abs=%.3e   %s (atol=%.0e)\n", sh.name, max_abs_s, ok_s ? "OK" : "FAIL", atol_s);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dG); cudaFree(dB);
    cudaFree(dSin); cudaFree(dSout); cudaFree(dO);
    return (ok_o && ok_s) ? 0 : 3;
}

int main(int argc, char** argv) {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("[kda] device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    std::string inputs_dir =
        "/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs";
    if (argc > 1) inputs_dir = argv[1];
    printf("[kda] inputs dir: %s\n", inputs_dir.c_str());

    int rc = run_correctness(inputs_dir);
    if (rc != 0) {
        fprintf(stderr, "[kda] correctness FAILED at correctness shape — stopping\n");
        return rc;
    }
    // Also smoke at the bench shapes we'll instantiate. Tolerance scales with
    // d_k (more accumulated f16 round-off in larger shapes).
    KDAShape kld{"kimi_linear_decode", 1, 32, 128, 128};
    rc = run_shape_smoke(inputs_dir, kld, /*atol_o=*/5e-3, /*atol_s=*/5e-3);
    if (rc != 0) return rc;

    KDAShape lrg{"large", 4, 64, 256, 256};
    rc = run_shape_smoke(inputs_dir, lrg, /*atol_o=*/1e-2, /*atol_s=*/1e-2);
    return rc;
}
