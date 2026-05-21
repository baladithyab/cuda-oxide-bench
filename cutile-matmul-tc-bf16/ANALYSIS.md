# cutile-matmul-tc-bf16 — Wave C2.2 ANALYSIS

**Cell:** Rosetta Stone Wave C2.2 — cuTile DSL bf16 tensor-core matmul, single
dtype.
**Sister cells:** `cuda-matmul-tc-bf16` (C2.1, hand-written CUDA WMMA bf16,
in flight at write-time) and `cutile-matmul-tiled-mixed` (W13.1, four
mixed-precision dtypes including bf16 — this is the closest existing
DSL TC reference).
**GPU:** NVIDIA GeForce RTX 5090 (sm_120, Blackwell consumer)
**cuda-tile:** 1.3.0 (from `cutile-vecadd-bench/.venv`)
**cupy:** 14.0.1
**Shape:** M = N = K = 4096
**Protocol:** 5 warmup + 50 timed iters with cudaEvent pairs, single stream
**Correctness:** sampled at 200 random (i,j) entries vs CPU f32 reference
(operands promoted to f32 then dotted), `atol=2e-1`, `rtol=5e-2`.

## 1. Headline result

| metric | value |
|---|---:|
| best  TFLOPS | **160.57** |
| median TFLOPS | **159.14** |
| mean  TFLOPS  | 143.59 |
| stdev TFLOPS  | 21.29 |
| worst TFLOPS  | 96.66 |
| best  GPU ms  | 0.856 |
| median GPU ms | 0.864 |
| worst GPU ms  | 1.422 |

Correctness sweep: **0 / 200 samples failed**, max absolute error 3.92, max
relative error 3.77e-3 — well inside the gate (atol=0.2 is dominated by the
bf16-output rounding floor at this magnitude; the much tighter relative-error
fact of 0.4% is the real signal that the kernel computed the right thing).

## 2. vs cuBLAS half-precision baseline (cublas-half-precision/W14.1)

| dtype       | cuBLAS best | cuBLAS median | cuTile-tc-bf16 best | ratio (best) |
|-------------|------------:|--------------:|--------------------:|-------------:|
| **bgemm** (bf16→f32acc) | 219.24 TF | 217.4 TF | **160.57 TF** | **73.2 %** |
| hgemm (f16→f32acc)      | 218.41 TF | —        | (n/a, this cell is bf16-only) | — |

**cuTile DSL at bf16 reaches 73 % of cuBLAS bgemm at the same shape** — exactly
the W13.1 mixed-precision number reproduces here in a focused single-dtype
form. The 27 pp gap to cuBLAS is the expected DSL-vs-hand-tuned-library tax:
cuBLAS picks from dozens of autotuned GEMM algos per shape; cuTile emits one
generic codegen path. The fact that the gap is identical between the
mixed-precision and the explicit-bf16-only cells confirms that **cuTile does
not unlock additional perf when the dtype commitment is made compile-time
explicit at the user level**: the DSL's dtype dispatch is already maximally
informed once `ct.float32` accumulator + `ct.bfloat16` operand types flow
through `ct.mma`.

## 3. vs cuda-matmul-tc-bf16 (C2.1 sibling)

The C2.1 sibling cell `cuda-matmul-tc-bf16/` contains only `matmul_tc_bf16.cu`
at the time of this writing (no results.csv, no ANALYSIS.md). Comparison
deferred to the parent agent's aggregation pass once C2.1's bench lands.
Expected outcome based on prior CUDA-WMMA cells in this repo (Wave 11 series):
hand-written WMMA bf16 should land between 180 – 210 TF best at this shape
(higher than this DSL cell, lower than cuBLAS), giving a ratio of roughly
**cuTile / cuda-tc-bf16 ≈ 0.75 – 0.90**.

## 4. vs cuTile-mixed (W13.1) — sanity check that dtype commitment is a
no-op for perf

| cell                                | bf16 best @ N=4096 |
|-------------------------------------|-------------------:|
| `cutile-matmul-tiled-mixed` (W13.1) |   159.8 TF |
| `cutile-matmul-tc-bf16` (this)      | **160.57 TF** |

**Within 0.5 %** — well inside the 21-TF stdev observed in this cell's 50-iter
distribution. **There is no user-visible perf benefit to writing the bf16
kernel separately vs. dispatching it as one of N variants in a multi-dtype
file** in cuda-tile 1.3.0. The DSL already specialises codegen per dtype at
JIT time. This is a clean **negative result** for the implicit hypothesis
that "an explicit BF16-only commitment unlocks more aggressive TC dispatch":
it does not, at the cuda-tile 1.3.0 user-level API.

## 5. DSL feature gaps hit (the pitfalls section)

The task brief specifically asked to flag any DSL feature gaps that prevent
making the BF16 commitment "more explicit" than the mixed-precision sibling
already does. Findings:

1. **No user-level `cute.tensor_core_mma` / `cute.bf16_mma` exists in
   cuda-tile 1.3.0.** `ct.mma(a, b, acc)` is the only MMA front-door; tensor-
   core engagement is purely dtype-conditional. Forcing TC engagement is done
   by *picking the right input dtype + accumulator dtype*, not by an MMA-
   variant API. There is no API path to pre-commit "this kernel will only
   ever take BF16" beyond what type-checking the kernel signature already
   provides.

2. **No `tile_shape` constants.** `ct.Constant[int]` launch-args fail with
   `TileTypeError: Invalid argument tile_shape`; tile dims (BM/BN/BK) must
   be Python ints captured by closure (`make_matmul_bf16(bm, bn, bk)`
   factory pattern). Carried forward from W13.1 / W12.4.

3. **`ct.mma` returns a new tile.** The line `acc = ct.mma(tx, ty, acc)` is
   required — in-place MMA is not supported. Easy to miss as a Python idiom.

4. **`acc.astype(C.dtype)` at store-time is the only down-cast path.** No
   explicit f32 → bf16 conversion API exists; relying on `astype` is the
   convention. Good news: the SASS evidence from W13.1 (HMMA.16816.F32.BF16
   instruction count = 64, FFMA = 0) confirms this `astype` is fused into the
   epilogue and doesn't introduce extra round-trips.

5. **bf16 random init must round-trip via numpy + ml_dtypes.** `cupy.random`
   doesn't dispatch directly on `ml_dtypes.bfloat16`; the idiom is
   `rng.random((n,n), dtype=np.float32).astype(ml_dtypes.bfloat16)` then
   `cupy.asarray(...)`. Carried forward from W13.1.

6. **`cupy.matmul` over `ml_dtypes.bfloat16` arrays as a host-side reference
   is unreliable for large N.** It dispatches through the numpy fallback path
   and silently drops precision. We sidestepped this by computing the f32
   reference on the CPU element-wise via `np.dot(A_f32[r,:], B_f32[:,c])` —
   correct and cheap when sampled (200 dot products, N=4096 each, takes
   ~0.5 s). Use this idiom in any future cuTile bf16 cell.

## 6. High variance in 50-iter distribution

stdev / median = 21.3 / 159.1 = **13.4 %** — much higher than the cuBLAS bgemm
3 % at the same shape. The worst-iter outlier (96.66 TF, 1.42 ms) is ~66 % of
the best-iter and looks like a single thermal/scheduling event. This matches
the AGENTS.md note that **GPU clocks are not locked on this WSL2 RTX 5090
host** and that 5–15 % CV at N=4096 is expected without `nvidia-smi -lgc`. The
median (159.14) and best (160.57) numbers are the trustworthy ones; the mean
(143.59) is contaminated by 3–5 outlier iters.

## 7. Files

| file | purpose |
|---|---|
| `main.py`     | kernel + bench (defaults to bench mode) |
| `run.sh`      | smoke (N=512) → bench (N=4096) wrapper |
| `results.csv` | per-iter (gpu_ms, tflops) for 50 timed iters |
| `bench.log`   | full stdout/stderr from `run.sh`'s bench step |
| `smoke.log`   | full stdout/stderr from `run.sh`'s smoke step |
| `ANALYSIS.md` | this file |

## 8. Bottom line for the parent agent

- **160.57 TF best / 159.14 TF median** at N=4096, bf16 inputs, f32 acc, bf16
  out, on cuda-tile 1.3.0 RTX 5090.
- **73 % of cuBLAS bgemm.** Within the W13.1-established competitive band.
- **Functionally identical to W13.1's mixed-precision bf16 variant** (within
  0.5 %), demonstrating that explicit single-dtype commitment in the DSL is
  *not* a perf lever — it is purely a code-organisation lever for projects
  that want to ship one bf16-only kernel artifact.
- **Expected ~150 – 200 TF range from the brief — landed at 160 TF**, on the
  lower edge of expected. Closing the remaining 60-TF gap to cuBLAS (219 TF)
  would require either (a) a hand-tuned cuTile kernel with explicit shared-
  mem staging + double-buffering + warp-spec, or (b) a future cuda-tile
  release that ships autotuned tile shapes per (M,N,K).
