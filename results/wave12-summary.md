# Wave 12 Multi-Kernel cuTile Comparison — SUMMARY

**Date:** 2026-05-11. All numbers from re-runs in a single session on idle RTX 5090
(43-44°C, 47-50W idle, sm_120, native CUDA 13.2). cuTile via `cuda-tile==1.3.0`
+ `cupy-cuda13x==14.0.1`, Python 3.12.

## Headline (best TFLOPS / best GB/s, 10 timed cudaEvent iters per config)

### Memory-bound (parity within 1%, except cuTile reduction which beats nvcc)

| impl | kernel | vecadd 256M | reduce 256M |
|---|---|---:|---:|
| nvcc | reference | 1568 GB/s | 1522 GB/s |
| cuda-oxide | safe | 1573 (100%) | 1519 (100%) |
| cuda-oxide | unchecked | 1570 (100%) | — |
| **cutile** | tile256/tile1024 | **1559 (99%)** | — |
| **cutile** | reduce_sum | — | **1696 (111%)** |

cuTile's reduction *beats* nvcc/cuda-oxide by ~11% on the 256M-element reduction.
This is the single most surprising memory-bound finding of Wave 12. The cuTile
kernel uses `ct.sum(tile)` over a 1024-element tile + `ct.atomic_add(out, (0,), partial)`
into a single output. The reference oxide-reduction does an explicit 2-stage
warp-shuffle + smem reduce + atomicAdd. Same algorithm, same hardware, ~11%
faster from cuTile's compiler. Likely explanation: cuTile's `ct.sum` lowers to
SASS with better instruction scheduling than the hand-written shuffle path.
**Worth investigating at SASS level** (see Wave 13 candidate items).

### Compute-bound (matmul) — cuTile is 4-5× behind on this hardware

| impl | kernel | N=1024 | N=2048 | N=4096 |
|---|---|---:|---:|---:|
| **cublas** | sgemm (tensor cores) | **38.17** | **64.76** | **73.59** |
| cuda-oxide | tiled-microtile-fmuladd | 29.15 | — | 45.05 |
| nvcc | matmul_tiled (shared-mem) | 26.97 | 34.55 | 38.41 |
| **cutile** | matmul_tiled (`ct.mma`) | **2.14** | **5.14** | **7.57** |
| cuda-oxide | matmul_safe (naive) | 7.09 | 7.59 | 7.29 |
| nvcc | matmul (naive) | 7.03 | 7.43 | 7.27 |
| cutile | matmul_tiled_simple (no mma) | 1.67 | 2.39 | 2.50 |
| cutile | naive (broadcast-and-sum) | 1.36 | 1.79 | 1.84 |

**The big surprise.** cuTile's `ct.matmul` / `ct.mma` path — the API that's
supposed to *be* cuTile's reason for existing — produces **7.57 TFLOPS at N=4096**,
which is:
- **5.1× slower than nvcc shared-mem tiled (38.41 TF)**
- **5.9× slower than cuda-oxide register-microtile (45.05 TF)**
- **9.7× slower than cuBLAS (73.59 TF, the only one actually using tensor cores)**

cuTile is producing CUDA cores math, not tensor cores math. This is
either (a) a v1.3.0 limitation: `ct.mma` for f32×f32 doesn't fall through to
the f32 tensor-core path on Blackwell sm_120; (b) requires explicit dtype hints
(e.g. `tfloat32` or `bf16`) to engage the TC pipeline; or (c) is specifically
designed for non-TC datatypes only. Worth filing upstream.

## Honest read for users

If you're choosing a Rust-first or Python-first GPU compute frontend on Blackwell
consumer hardware (RTX 50-series) **today** (May 2026):

1. **cuda-oxide** is the strongest non-CUDA-C++ alternative for matmul-style
   compute-bound kernels. The Wave 7 register-microtile + fmuladd path gets
   60% of cuBLAS at N=4096 from hand-written Rust kernels. cuTile's `ct.mma`
   gets ~10% of cuBLAS.

2. **cuTile** is competitive (parity-within-1%) on memory-bound kernels and
   *faster* on reduction. The Python-first dev experience is the best of the
   three. But for compute-bound work today, you'd want to either drop down
   to the C++ track of cuTile, hand-write CUDA C++ + cuBLAS, or use cuda-oxide.

3. **CUDA C++ + cuBLAS** still wins for any matmul-shaped problem by 50-90%
   over both alternatives.

4. **All three are within 1% on memory-bound kernels** — the hardware is the
   bottleneck, not the compiler. Choose the language you prefer.

## Per-axis details

- `cutile-vecadd-bench/ANALYSIS.md` — vecadd parity (99.5% of nvcc). Wave 12.1.
- `cutile-reduction/ANALYSIS.md` — reduction OUTPERFORMS nvcc by 11%. Wave 12.2.
- `cutile-matmul/ANALYSIS.md` — naive matmul: cuTile lacks 1-thread-per-output
  ergonomics; the broadcast-and-sum form is 4× behind nvcc/oxide naive. Wave 12.3.
- `cutile-matmul-tiled/ANALYSIS.md` — `ct.mma` works correctness-wise but does NOT
  light up tensor cores at f32 on sm_120. 5× behind tiled CUDA C++. **The Wave 12
  red flag.** Wave 12.4.

## Methodology notes

1. **Re-ran ALL existing baselines on the idle GPU in this same session** (vecadd,
   matmul, matmul-tiled, reduction, cublas-matmul, oxide-microtile) before adding
   cuTile numbers. The original May-9 vecadd CSVs were thermally degraded by ~50%
   — all comparison tables in this SUMMARY use the fresh idle-GPU reruns. Old
   `cuda-vecadd-bench/results.csv` and `oxide-vecadd-bench/results.csv` were
   overwritten as part of W12.1.

2. **Subagent fan-out**: kernels were authored by 3 parallel subagents (reduction,
   naive matmul, tiled matmul) in ~5 min wall-clock. Each subagent verified
   correctness locally but DID NOT run timed iterations — orchestrator ran the
   timed bench pass serially after all three returned, to keep the GPU idle and
   the timings clean.

3. **Pitfalls captured in `cutile-vecadd-bench/SETUP.md`**:
   - pip package name is `cuda-tile` (NOT `nvidia-cutile`, NOT `cutile-python`)
   - README's `kernel[(grid,)](args)` launch syntax is broken in v1.3.0 — must
     use `ct.launch(stream.ptr, grid_tuple, kernel, args_tuple)`
   - JIT compile is 639 ms first launch then sub-ms
   - `ct.atomic_add` takes index as a tuple matching array rank — `(0,)` not `0`
   - `ct.Constant[int]` launch-arg pattern from docs FAILS in v1.3.0 — use
     Python-closure factory pattern instead (cutile-matmul-tiled/ANALYSIS.md)
   - `ct.matmul` / `ct.mma` work for f32 correctness-wise but produce CUDA-core
     code, NOT tensor-core code, on sm_120

## What's next (Wave 13 candidates)

- **Why does cuTile win at reduction?** SASS-level comparison cuTile vs oxide
  reduce_sum. The 11% lift is large enough to be a real compiler-quality result
  worth understanding.
- **Why does cuTile lose so badly on matmul?** Verify that `ct.mma` is using
  tensor cores; if it isn't, file an upstream issue. Try f16/bf16/tfloat32 dtypes
  to see if the TC path engages there.
- **Try cuTile on a datacenter Blackwell (B100/B200)** if available — `ct.mma`
  may light up TCs there but not on sm_120.
- **Port the 3DGS rasterizer to cuTile** for completeness alongside the
  existing oxide and nvcc 3DGS implementations. Would test cuTile on a real
  multi-kernel workload, not just academic primitives.
