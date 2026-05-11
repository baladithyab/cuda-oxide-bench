# Wave 15.1 — GQA across 3 frontends: the FlashAttention fusion win

**Date:** 2026-05-11. Continuation of Wave 15 (research + architecture + tooling).
Three parallel implementation subagents built one Grouped-Query Attention cell each;
orchestrator independently re-ran each on idle GPU in same session and verified
SASS-level tensor-core engagement before commit.

## Headline @ Llama-3-8B canonical shape (B=1, S=2048, n_q=32, n_kv=8, d=128)

| impl | best ms | best TFLOPS | vs cuBLAS hgemm 218 TF | HMMA count |
|---|---:|---:|---:|---:|
| `cuda-attn-gqa` (nvcc 3-kernel WMMA) | 2.928 | **23.47** | 10.8% | 20 |
| `cublas-attn-gqa` (cuBLAS-3-kernel) | 1.482 | **46.38** | 21.3% | (cuBLAS) |
| **`cutile-attn-gqa` (cuTile fused)** | **0.416** | **165.14** | **75.7%** | **256** |

**The headline is the fusion ratio, not absolute TFLOPS.** cuTile's fused single-kernel
implementation is **3.6× faster than the same algorithm split across 3 cuBLAS-tuned
matmul + custom softmax + cuBLAS-tuned matmul kernels.** This is the FlashAttention
win on Blackwell consumer hardware via a Python tile DSL.

## Why fusion wins by 3.6× (the SASS / memory story)

For Llama-3-8B GQA at S=2048 the (seq × seq) attention matrix is **537 MB** at f32
(scores) plus **268 MB** at f16 (probs after softmax) — the full attention matrix
must transit HBM twice in a 3-kernel pipeline (write scores, read scores in softmax,
write probs, read probs in PV). Fused FlashAttention-2 keeps it in registers.

Per-stage breakdown of the cuBLAS-3-kernel cell (idle-GPU rerun):

| stage | avg ms | % of total |
|---|---:|---:|
| QKᵀ (cublasGemmEx) | 0.512 | 31.3% |
| softmax (custom kernel, 1.5 TB/s ≈ 70% HBM peak) | 0.505 | 30.9% |
| PV (cublasGemmEx) | 0.620 | 37.9% |
| **total** | 1.637 | 100% |

The softmax kernel alone costs as much wall-clock as 32 cuBLAS matmuls; it's
purely memory-bandwidth-bound, sitting on the critical path. The fused kernel
eliminates this stage's HBM round-trip entirely. cuTile reaches **165 TFLOPS**
which is 96% of cuTile's own single-matmul peak (172.5 TF from Wave 13.1) —
the fused-attention overhead vs pure GEMM is only ~4%.

## Independent verification (orchestrator skepticism check)

Each subagent's headline number re-checked by orchestrator on idle GPU in this
same session, after all 3 subagents completed:

```
$ ./cuda-attn-gqa/attn_gqa
[gqa] best gpu_ms = 2.9284  tflops = 23.47   ✓

$ ./cublas-attn-gqa/attn_gqa
[cublas-attn-gqa] best_total=1.482ms best_TF=46.38  ✓

$ python cutile-attn-gqa/main.py --bench
[bench] best : 0.416 ms   165.14 TFLOPS  ✓
```

SASS-level TC engagement verified independently:

```
$ /usr/local/cuda/bin/cuobjdump --dump-sass cuda-attn-gqa/attn_gqa | grep -c HMMA   # → 20  ✓
$ grep -c HMMA cutile-attn-gqa/gqa_fwd_fused.sass                                    # → 256 ✓
```

(cuBLAS doesn't expose its kernel SASS the same way; trust cuBLAS engineers
re: TC engagement, but confirmed via the 46 TF result being far above any
CUDA-core ceiling.)

## Correctness (PyTorch SDPA reference @ correctness shape, b=1 s=128 n_q=4 n_kv=2 d=64)

All three cells:

```
[gqa]            correctness: max_abs_err=1.559e-04 (atol=5e-3) → OK (30× margin)
[cublas-attn-gqa] correctness: max_abs=1.559e-04 (atol=5e-3) → OK
[cutile-attn-gqa] correctness: max_abs=1.3e-4    (atol=5e-3) → OK
```

Identical max_abs_err on the first two suggests they converge to the same f16
arithmetic; cuTile's slight difference is from the online-softmax rescaling
running in f32 registers (numerically more stable, slightly different rounding).

## What this tells us about the 4 frontends

1. **cuTile excels at fused kernel patterns.** The Python tile DSL was designed
   for exactly this workload (fused attention with online softmax). 76% of cuBLAS
   hgemm peak from a single decorator is competitive — and beats hand-rolled
   3-kernel pipelines using the same hardware-tuned cuBLAS matmul calls.

2. **Hand-written WMMA in nvcc CUDA C++ at 23 TF** is not impressive on its own,
   but the gap (10.8% of cuBLAS hgemm) is interesting: the cell uses tiny 16×16
   WMMA tiles with no shared-memory tiling. It's a correctness oracle, not a perf
   oracle. To match cuBLAS would require ~300+ more LOC for proper shared-mem
   tiling + register microtile + double-buffered loads.

3. **cuBLAS-3-kernel (46 TF, 21% of hgemm)** is the right ceiling for unfused
   attention from off-the-shelf GEMM. The 3.6× headroom over this ceiling is
   the FlashAttention argument distilled to one number.

4. **cuda-oxide deliberately deferred to Wave 15.5.** Wave 14.4 found cuda-oxide
   has no usable TC API on consumer Blackwell; the expected ceiling is 5-10 TF
   for f32-only GQA. That's a "no-TC ceiling" data point, not a comparable-with-
   the-others measurement, so it gets its own wave.

## Files added

```
analysis/wave15-attention-architecture/
  reference/
    shapes.py, flops.py, tolerances.py
    pytorch_reference.py + reference_run.log
    README.md
  PLAN.md
  inputs/.gitignore  (.npy tensors are ~105MB, regenerable via pytorch_reference.py)

analysis/wave15-attention-research/MECHANISMS.md  (Wave 15 research)
analysis/wave15-nsight-tooling/REPORT.md         (nsys works, ncu blocked)
analysis/wave15-nsight-tooling/probe-artifacts/   (3 .nsys-rep files)

cuda-attn-gqa/
  attn_gqa.cu (~410 LOC, 3-kernel WMMA naive)
  Makefile, .gitignore
  results.csv, run.log
  ANALYSIS.md

cublas-attn-gqa/
  attn_gqa.cu (~440 LOC, cuBLAS pipeline)
  softmax.cu (~95 LOC, row-wise online softmax)
  Makefile, .gitignore
  results.csv, run.log
  ANALYSIS.md

cutile-attn-gqa/
  main.py (~460 LOC, fused FlashAttention-2 in @ct.kernel)
  run.sh, .gitignore
  results.csv, run.log
  gqa_fwd_fused.sass (1.5 MB, 12.5k lines, evidence for HMMA count)
  ANALYSIS.md
```

## Pitfalls captured (from the 3 subagents)

- **`/usr/bin/cuobjdump` is CUDA 12 stale** and silently produces empty SASS on
  sm_120 cubins. Always use `/usr/local/cuda/bin/cuobjdump`. Same trap as the
  `/usr/bin/nvcc` shim issue from Wave 1; carried over into other tooling.

- **cuTile `ct.store(tiled_view, ...)`** doesn't work — tiled_view's store method is
  used as `tiled_view.store((row, 0), tile)`, not via the free-function `ct.store(...)`
  which expects a raw `Array`. One-line pitfall, not in any docs.

- **cuTile BLOCK_M=128 falls off a register cliff** at d_head=128 (46 TF vs 165 TF
  at 64×64). cuTile 1.3.0 has no user-facing register-budget hints; 64×64 is the
  safe spot for fused attention with d=128.

- **cuBLAS column-major bug**: first cuBLAS-3-kernel attempt passed Q as A and K as B
  with op=(T,N), which silently computes the *transpose* of the desired scores. The
  correctness-shape's tighter check caught it; the bench shape's tolerance would
  have masked it. Lesson: always run correctness BEFORE bench, with tight tolerances.

- **GPU contention during parallel subagent execution** caused 2-3× slow iters
  in mid-run. Re-run on idle GPU (Wave 15 standard pattern) used as the canonical
  numbers. Both contention-affected and clean numbers shown in run.log files.

## What's next (Wave 15.5 candidates, all in /home/codeseys/cuda-exploration/)

- **W15.2 cuda-attn-gqa-oxide**: GQA via cuda-oxide — the no-TC ceiling data point.
  Expected ~5-10 TF (f32 only, no MMA). Tests whether cuda-oxide's register-microtile
  pattern from the matmul work generalizes to attention.

- **W15.3 cutile-attn-gqa-fused-bigger-tile**: try BLOCK_M=BLOCK_N=128 with explicit
  register-budget hints if cuTile adds them; or 64×128 / 128×64 asymmetric tiles.
  Closing the 24% gap to cuBLAS hgemm ceiling.

- **W15.4 cutile-attn-mla**: MLA implementation in cuTile (next mechanism per
  research recommendation). DeepSeek-V3 shape: n_h=128, d_h=128, decoupled RoPE
  d_rope=64, latent dim d_c=512.

- **W15.5 cutile-attn-gdn-decode**: GDN decode kernel — fundamentally different
  regime (recurrent state, no softmax). Tests whether cuTile expresses stateful
  recurrence well.

- **W15.6 ncu profiling once unblocked**: if the user can flip the perf-counter
  permission in Windows NVIDIA Control Panel (the only path on WSL2 — see
  wave15-nsight-tooling/REPORT.md), ncu can give occupancy / memory throughput /
  stall metrics for the 3 cells. Without it we have only kernel timeline + total
  TFLOPS.
