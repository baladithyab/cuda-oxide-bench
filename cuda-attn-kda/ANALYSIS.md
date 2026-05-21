# cuda-attn-kda — Wave C2.6 (Rosetta Stone)

CUDA-C++ port of Kimi Delta Attention (KDA) single-timestep decode. Closes
the KDA column of the Rosetta Stone matrix (previously cutile-only).

## Headlines (RTX 5090, sm_120, 50 timed iters + 2 warmup)

| shape | (B,H,d_k,d_v) | n_blocks | best GB/s | median GB/s | best gpu_us |
|---|---|---:|---:|---:|---:|
| kimi_linear_decode | (1,32,128,128) | 32   | **360.6** | 340.2 | 11.74 |
| large (saturation) | (4,64,256,256) | 1024 | **568.4** | 557.4 | 237.28 |

Bytes/iter formula matches cutile-attn-kda::kda_decode_bytes (2·d_k·d_v·4 state R+W
plus io = (3·d_k + 2·d_v + 1)·2 for q/k/g/v/o f16 + beta scalar).

## vs cuTile reference

cutile-attn-kda W22.7 best @ large = **1170 GB/s** (saturation, source of headline).

Our cuda-attn-kda best @ large = 568.4 GB/s → **48.6% of cuTile saturation.**

This is at the upper end of (or above) the 25-40% expected range from the task
prompt. The KDA per-channel decay adds essentially zero overhead vs GDN (one
extra `smem_g[k]` read per inner-k iteration; the multiply is a free FFMA chain).

## Comparison vs cuda-attn-gdn (the closest C++ algorithm reference)

cuda-attn-gdn @ qwen3_next_decode (1,16,256,256, 64 blocks) = 417 GB/s best.
cuda-attn-kda @ large (4,64,256,256, 1024 blocks)          = 568.4 GB/s best.

KDA's `large` shape has 16× more blocks than GDN's qwen3, so it's better
saturated. At the apples-to-apples qwen3_next_gdn_parity shape (1,16,256,256),
launch geometry would be the same; we did not bench that explicitly here.

## SASS instruction mix

```
HMMA      : 0      (memory-bound, no TC use — as expected)
FFMA      : 300
LDG.E.128 : 24     (vectorized f32 state-tile reads)
STG.E.128 : 24     (vectorized f32 state-tile writes)
MUFU      : 3      (exp() in setup phase — per-channel decay)
```

The 3 MUFU.EX2 instructions are the only structural difference vs
cuda-attn-gdn's SASS (which has 0 MUFU). The exp() runs once per d_k row
during the cooperative load phase; cost is fully amortized across the inner
k-loop.

## Algorithm — what changed vs GDN

KDA = "GDN with per-channel decay vector" instead of "GDN with scalar decay":

```c++
// GDN (cuda-attn-gdn): scalar alpha
const float alpha = s_alpha;
// inner loop:
s.x *= alpha; s.y *= alpha; s.z *= alpha; s.w *= alpha;

// KDA (this file): per-d_k-row decay = exp(g[k])
// One-time setup:
for (int kk = tid; kk < D_K; kk += TPB) {
    smem_g[kk] = expf(__half2float(Gbh[kk]));
}
// inner loop:
float decay = smem_g[k];   // varies per k
s.x *= decay; s.y *= decay; s.z *= decay; s.w *= decay;
```

All other passes (u = k·S_scaled, residual, S_out = S_scaled + β·k⊗r,
o = q·S_out) are byte-identical to the GDN kernel.

## Correctness

All three shapes PASS vs the float64 NumPy oracle (naive_recurrent_kda_step
ported from cutile-attn-kda/main.py and saved to disk via gen_kda_inputs.py):

- correctness (B=1,H=2,d_k=d_v=64):     o max_abs = 0.0,    Sout max_abs = 6e-8  (atol=1e-3, 5e-3)
- kimi_linear_decode (1,32,128,128):    o max_abs = 1.2e-7, Sout max_abs = 6e-8  (atol=5e-3 for both)
- large (4,64,256,256):                 o max_abs = 1.2e-4, Sout max_abs = 6e-8  (atol=1e-2)

The o-error scaling with d_k (more f16 round-off accumulates in the q·S_out
reduction) matches GDN's pattern; well within the spec's atol=1e-3 floor at
the canonical correctness shape.

## Files

- `attn_kda.cu`         — kernel + correctness driver
- `bench.cu`            — 50-iter cudaEvent timing harness
- `Makefile`            — nvcc 13.2, `-arch=sm_120`, clang-14 host compiler
- `run.sh`              — build + correctness + bench + SASS dump
- `attn_kda.sass`       — disassembled SASS
- `results.csv`         — kimi_linear_decode per-iter
- `results_large.csv`   — large per-iter
- `build.log` / `run.log` / `bench.log`

Reference inputs (new this wave, written once to shared inputs/ tree):

- `analysis/wave15-attention-architecture/reference/pytorch_reference_kda.py`
- `analysis/wave15-attention-architecture/reference/gen_kda_inputs.py`
- `analysis/wave15-attention-architecture/inputs/kda_*_*.npy` (4 shapes × 13 files)

## Pitfalls

1. **Block-V picker differs from cuda-attn-gdn**: at d_k=128 the cutile picker
   uses BV=128 (since 128·64 = 8192 elems ≤ 16384), but cuda-attn-gdn only
   wires (64,64) and (256,64). For KDA we explicitly added the (128,128)
   instantiation; this gives a 64 KB shared tile (128·32·16 B) which exceeds
   the 48 KB default and requires `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`
   at launch time. The launcher already gated on `> 48 KB`.

2. **The g tensor adds qkv-elems-sized HBM traffic** (B·H·d_k f16 = 4 KB at
   kimi_linear_decode), but it's negligible vs the state R+W (128 KB). The
   bytes/iter formula already includes this in the io term (`3*d_k` covers
   q+k+g instead of GDN's `2*d_k`).

3. **No reference inputs existed on disk before this wave** — task prompt
   anticipated this and instructed to write the generator first. Did so;
   inputs are now seeded from `make_inputs(shape, seed=0xCAFE+hash(name))`
   for shape-disjoint reproducibility, mirroring gen_w22_15_inputs.py's
   `0xBAD_F00D + hash(name)` convention.

4. **Saturation at `large` (568 GB/s) is well below cuTile's 1170 GB/s** at
   the same shape. The gap is the same one diagnosed in W22.8 for cuda-attn-gdn:
   no producer/consumer warp-specialized async pipeline, no Blackwell async
   barriers — direct LDG.E.128 from gmem with per-warp load latency exposure.
   Closing this gap would require a cuda::pipeline (W22.9 attempted for GDN
   and got -25% regression) or explicit cuTensorMapEncodeTiled TMA path. Out
   of scope for this cell.

## Build

```sh
make            # builds attn_kda + bench
./attn_kda      # correctness on 3 shapes (correctness, kimi_linear_decode, large)
./bench         # 50-iter timed bench on kimi_linear_decode + large
make sass       # SASS dump + instruction histogram
```
