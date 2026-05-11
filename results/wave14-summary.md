# Wave 14 — fair half-precision baseline + upstream issues + cuda-oxide TC verdict

**Date:** 2026-05-11. Continuation of Wave 13. Three parallel subagents fanned
out for cuBLAS hgemm/bgemm baseline (W14.1), upstream-issue drafts (W14.2),
and cuda-oxide tensor-core API investigation (W14.4). Each subagent's
critical claims independently verified by orchestrator before commit.

## Headline — cuTile half-precision IS competitive with cuBLAS

The Wave 13.1 finding was "cuTile reaches 172.5 TFLOPS f16 at N=4096" with a
caveat that it wasn't apples-to-apples vs cuBLAS sgemm (different dtype). W14.1
removes the caveat by adding cuBLAS hgemm + bgemm + sgemm-tf32 baselines on
the same hardware in the same session.

### Final apples-to-apples table @ N=4096 (best TFLOPS, RTX 5090 sm_120, idle GPU)

| dtype class | cuTile | cuBLAS | cuTile / cuBLAS |
|---|---:|---:|---:|
| f16 → f32 acc | 172.5 | **218.4** | **79.0%** |
| bf16 → f32 acc | 159.8 | **219.2** | **72.9%** |
| tf32 internal | 84.0 | **104.2** | **80.6%** |
| f32 (no TC) | 8.7 | 73.6 (sgemm) | 11.8% |

**cuTile f16 at 79% of cuBLAS hgemm is a genuinely competitive number for a
Python-first DSL on Blackwell.** That's tensor-core engagement working
correctly, not a fallback path. The Wave 12.4 framing of "cuTile is 5×
behind on matmul" is now decisively overturned: it was a dtype mismatch
in test design, not a cuTile bug.

cuBLAS hgemm at 218 TF on RTX 5090 sm_120 is ~66% of the hardware's marketed
f16 TC peak (~330 TF). cuTile at 172 TF is ~52% of peak. Both well above any
CUDA-core ceiling.

### Surprise side-finding: existing cublas-matmul/sgemm baseline was pedantic mode

The repo's existing `cublas-matmul/` reports 73.6 TFLOPS for sgemm on N=4096.
W14.1's `cublas-half-precision/sgemm_tf32` at the same shape (with explicit
`CUBLAS_TF32_TENSOR_OP_MATH`) gets **104.2 TFLOPS**, a 42% lift. The existing
73.6 TF baseline is therefore **pedantic IEEE 754 sgemm without TF32**, not
the default-fast path. For any future apples-to-apples cuTile-vs-cuBLAS f32
comparison, the **104.2 TF tf32-mode sgemm is the right reference** if the
user can tolerate TF32 accuracy loss; the 73.6 TF pedantic number is the
right reference if they need bit-exact f32.

This is documented in `cublas-half-precision/ANALYSIS.md` so future readers
don't accidentally compare against the wrong baseline.

## cuda-oxide tensor-core verdict — no usable TC API on RTX 5090 today

W14.4 source-read all of NVlabs/cuda-oxide v0.1.0 looking for a TC API on
consumer Blackwell. **There isn't one.** Five-line summary:

1. cuda-oxide v0.1.0 has **no usable TC API on RTX 5090 sm_120**.
2. The two TC-adjacent modules:
   - `cuda_device::wgmma` — Hopper sm_90 only; the MMA-issue path is **literally a placeholder comment** (verified at `mir-lower/src/convert/intrinsics/wgmma.rs:141`: `"// wgmma.mma placeholder"`).
   - `cuda_device::tcgen05` — datacenter Blackwell sm_100a only; PTX hard-codes `.target sm_100a`, driver rejects on consumer sm_120 with `CUDA_ERROR_INVALID_PTX`.
3. The classic `mma.sync.aligned.m16n8k{8,16}` family that sm_120 needs is **completely absent** — zero matches across the whole tree.
4. `asm!` is not supported in `#[kernel]` functions, so we cannot work around the gap in user code.
5. The gap is a **wrapper-surface gap, not a hardware/codegen gap** — libNVVM, the rustc-codegen-cuda emitter, and the driver can all produce sm_120 HMMA; it just needs upstream to add a `cuda_device::mma` module mirroring `tcgen05.rs`.

So the cuTile-vs-cuda-oxide comparison should be framed as **tensor-core vs
CUDA-core** rather than apples-to-apples. cuda-oxide's `oxide-matmul-tiled-microtile`
at 45 TFLOPS f32 IS competitive with nvcc's 38 TFLOPS f32 (the meaningful
non-TC-aware comparison) and beats cuBLAS pedantic-sgemm 73.6 TF only because
that's also non-TC. None of these can compete with cuBLAS hgemm 218 TF until
cuda-oxide ships a tensor-core API.

Independently verified by orchestrator:
```
$ grep -rn 'mma\.sync' /home/codeseys/.cargo/git/checkouts/cuda-oxide-*/*/crates/ --include='*.rs'
(zero results)
$ grep -rn 'placeholder' .../crates/mir-lower/src/convert/intrinsics/wgmma.rs
.../wgmma.rs:141:        "// wgmma.mma placeholder",  ✓
```

## Upstream issues drafted (W14.2)

Two issue MDs ready to submit at https://github.com/nvidia/cutile-python/issues:

1. `docs/upstream-issues/01-ctmma-f32-no-ffma-fusion.md` — `ct.mma(f32, f32, f32)`
   on RTX 5090 sm_120 produces 0 FFMA, 2051 FMUL, 2176 FADD (no FMA contraction
   at all). Same kernel at f16/bf16/tf32 emits 64-128 HMMAs correctly. nvcc on
   the same algorithm at f32 emits 256 FFMAs. End-to-end perf gap: 8.7 TF cuTile
   vs 38.4 TF nvcc, 4.4× slower than the same algorithm in C++. Suggested root
   cause: the f32 path of `ct.mma` emits NVVM IR without the `contract` /
   `FastmathFlags::CONTRACT` flag, so libNVVM's FFMA pattern-matcher declines
   to fuse.

2. `docs/upstream-issues/02-constant-int-launch-arg-tiled-view.md` — `ct.Constant[int]`
   launch-arg pattern from the docs fails with `TileTypeError: Invalid argument
   "tile_shape" of _m_array_tiled_view(): Expected a constant integer tuple,
   but given value is not constant`. The Python-closure factory pattern works
   as a workaround, but the docs example itself doesn't run.

Both issues include full environment table, copyable Python reproducer,
SASS instruction tables, independently-verifiable grep commands, and links
back to https://github.com/baladithyab/cuda-exploration. Tone is cooperative,
narrow, evidence-first.

## Independent verification by orchestrator (skepticism check on each subagent)

Every Wave 14 claim that drives a headline number was re-verified before commit:

```bash
# W14.1 cuBLAS hgemm 218 TF
$ tail -30 cublas-half-precision/run.log | grep "best_TF\|N=4096"
... best_TF=218.407 (iter=7) ✓
$ grep -c " OK$" cublas-half-precision/run.log     # → 30 (27+3) ✓

# W14.4 oxide TC verdict
$ grep -rn 'mma\.sync\|mma\.m16n8' .../cuda-oxide-*/  --include='*.rs' | wc -l   # → 0 ✓
$ ls .../cuda-device/src/{mma,tensor,hmma}.rs                                   # → none ✓
$ grep 'placeholder' .../wgmma.rs                                                # → :141 ✓

# W14.2 upstream issue cited SASS counts
$ grep -c HMMA cutile-matmul-tiled-mixed/mma_f32xf32_f32acc.sass  # → 0 ✓
$ grep -c FFMA analysis/wave13-sass/cuda_matmul_tiled.sass        # → 256 ✓
```

All three subagents' headline claims hold up. No retractions needed.

## Updated user-facing read (replaces Wave 13 take)

If you're choosing a Python-first or Rust-first GPU compute frontend on
Blackwell consumer hardware **today** (May 2026):

1. **For matmul-shaped compute-bound work at half-precision**: cuTile reaches
   79% of cuBLAS hgemm with one Python decorator. That's competitive. **Use
   cuTile**.
2. **For matmul at f32**: cuBLAS sgemm with TF32 mode gets 104 TFLOPS; cuBLAS
   pedantic-sgemm gets 73.6 TF; cuda-oxide tiled-microtile gets 45 TF; nvcc
   shared-tiled gets 38 TF; cuTile f32 gets 8.7 TF. Order of preference for
   raw perf: cuBLAS-tf32 > cuBLAS-sgemm > oxide > nvcc > cuTile.
3. **For memory-bound reduction-pattern work**: cuTile beats nvcc/oxide by 11%
   via TMA bulk loads. **Use cuTile**.
4. **For memory-bound vec-add**: any of the three (parity within 1%).
5. **For naive (no-TC) matmul**: cuda-oxide is the strongest non-CUDA-C++
   alternative. cuTile's broadcast-and-sum form is 4× slower than oxide's
   slice-indexing inner loop.
6. **cuda-oxide does not expose a TC API on consumer Blackwell today.** The
   45-TFLOPS f32 microtile result is the cuda-oxide ceiling on RTX 5090 with
   v0.1.0 APIs. Wait for upstream to add `cuda_device::mma` (or wait for
   `asm!` support in `#[kernel]` to roll your own).

## Files added in Wave 14

- `cublas-half-precision/` — hgemm + bgemm + sgemm-tf32 baseline
  - `matmul.cu`, `results.csv`, `run.log`, `ANALYSIS.md`, `.gitignore`
- `analysis/wave14-oxide-tc-investigation/`
  - `REPORT.md` — 238 lines, the cuda-oxide TC verdict + roadmap
- `docs/upstream-issues/`
  - `01-ctmma-f32-no-ffma-fusion.md` — issue draft (~170 lines)
  - `02-constant-int-launch-arg-tiled-view.md` — issue draft (~140 lines)
  - `README.md` — index + filing context
- `results/wave14-summary.md` — this document

## Wave 15 candidates

- **W15.1: file the upstream issues at github.com/nvidia/cutile-python/issues** — drafts ready, just needs a human submitter
- **W15.2: cuBLAS hgemm baseline at smaller N** (1024, 2048) shows the
  non-headline regime — does cuTile/cuBLAS ratio hold or improve at smaller N?
  Bench data already in cublas-half-precision/results.csv; just needs a
  ratio-table writeup vs cuTile per-N.
- **W15.3: 3DGS rasterizer port to cuTile** — completeness alongside oxide and nvcc.
- **W15.4: revisit cuda-oxide ceiling** — once upstream ships `cuda_device::mma`
  or `asm!` support, redo the matmul comparison. Track NVlabs/cuda-oxide releases.
