# Wave 20 — Mojo bf16-in/f32-acc TC reach probe

**Status:** ✅ W1+W3 SHIPPED. W2 (full tiled bf16 matmul) deferred to Wave 21 — the `TensorCore` wrapper rejects mixed-precision in BOTH `mma_op` and `store_d`, so a full bf16 matmul requires hand-rolled `ld_matrix` + `mma()` + manual epilogue (out of scope for this wave).
**Date:** 2026-05-20 (single-session continuation of Wave 19)
**GPU:** RTX 5090, sm_120, driver 596.21
**Mojo:** 1.0.0b1 (a9591de6)

## Headline finding

**Mojo CAN reach the bf16-in/f32-acc tensor-core path on consumer
Blackwell sm_120, via `from std.gpu.compute.mma import mma`.** SASS
evidence (`mojo-mma-probe/mma_probe.sass`):

```
.target sm_120a
HMMA.16816.F32.BF16 R4, R4, R12, R8 ;   ← × 1
```

The Wave 19 same-dtype constraint is a property of the **`TensorCore`
wrapper**, NOT the underlying hardware path or the `mma()` primitive.
The wrapper simply hasn't been generalized to mixed-precision yet.

This reframes the user-facing Mojo TC story:

- **High-level API (`from layout.tensor_core import TensorCore`)**:
  same-dtype only. TF32 m16n8k4 reachable (Wave 19, **55.5 TF**).
  bf16/f16 mixed-precision blocked by both `mma_op` and `store_d`
  having `A.dtype == C.dtype` constraints.
- **Low-level API (`from std.gpu.compute.mma import mma`)**: the full
  m16n8k16 bf16-in/f32-acc path is reachable. The bottleneck is
  scaffolding (per-thread fragment layout, ld_matrix, epilogue write
  per PTX 9.7.13.4.8), not the hardware or compiler.

## Phases

### W1: `mojo-mma-probe/` — minimal m16n8k16 bf16 mma() smoke test ⚡

Single-warp (32 threads) kernel. Constructs trivial bf16 fragments
seeded by `thread_idx.x` (so the compiler can't constant-fold them
away), calls `mma(d, a, b, c)` with the m16n8k16 BF16 dispatch shape:

```mojo
var a_frag = SIMD[DType.bfloat16, 8](...)  # per-lane
var b_frag = SIMD[DType.bfloat16, 4](...)
var c_frag = SIMD[DType.float32,  4](...)
var d_frag = SIMD[DType.float32,  4](0,0,0,0)
mma(d_frag, a_frag, b_frag, c_frag)
```

**Result**: compiles, runs, and `_dump_sass=True` reveals
`HMMA.16816.F32.BF16` × 1 on `.target sm_120a`. The path is real.

The SIMD widths come from the `_mma_nvidia` source for the m16n8k16
BF16 dispatch lane: `_has_shape[(8, 4, 4, 4)]`. So per-warp:
- A: 8 bf16 elements/lane × 32 lanes = 256 elements (16×16 = 256 ✓)
- B: 4 bf16 elements/lane × 32 lanes = 128 elements (16×8 = 128 ✓)
- C/D: 4 f32 elements/lane × 32 lanes = 128 elements (16×8 = 128 ✓)

### W2: `mojo-matmul-mixed/` — TensorCore wrapper retest (DELETED)

We attempted the easy path: re-use the Wave 19 kernel structure, just
swap `a_type=bf16, c_type=f32`. **Failed at compile time** in TWO
places (not just the Wave 19-known `store_d`):

```
tensor_core.mojo:781:9: constraint failed: destination tensor must
                         have the same type
                         ↑ store_d -- known from Wave 19

tensor_core.mojo:842:10: error: rebind input type '!pop.simd<4, f32>'
                         does not match result type '!pop.simd<4, bf16>'
                         ↑ mma_op -- NEW finding, the wrapper's
                           internal rebind also assumes uniform dtype
```

So even if you replace just `store_d` with a hand-rolled epilogue, the
`mma_op` itself rejects the mixed-precision call. **The whole
`TensorCore` struct is fundamentally same-dtype.** Cell deleted from
the tree; the negative finding is captured in
`mojo-mma-probe/ANALYSIS.md` ("Why no full matmul yet" section).

### W4 (numerical correctness): cancelled

Required a working full-tile bf16 matmul that we don't have. Deferred
to Wave 21 alongside the actual harness.

## Cross-frontend status (no perf changes from Wave 19)

| frontend | bf16 TC reach? | f32→TF32 TC reach? |
|---|---|---|
| cuTile | **YES** (159 TF) | **YES** (84 TF) |
| Mojo via `TensorCore` | **NO** (same-dtype constraint) | **YES** (Wave 19, 55.5 TF) |
| **Mojo via raw `mma()`** | **YES (path proven, harness TBD)** | yes (would also work) |
| cuda-oxide | NO (no TC API) | NO |
| nvcc CUDA C++ | yes (manually with mma.sync inline asm) | yes |

The headline matmul TFLOPS table doesn't change in Wave 20 — we
proved a path exists but didn't ship a perf number for it. **The
cuTile-vs-Mojo gap on bf16 (159 vs N/A through wrapper) is now
**definitively** an API-coverage gap, not a hardware gap.**

## Pitfalls discovered

The W2 attempt confirmed that Wave 19's listing of "constraint #11
(store_d same-dtype)" was incomplete. Updated to:

- **`TensorCore` wrapper requires `A.dtype == C.dtype` in BOTH
  `mma_op` AND `store_d`.** Replacing just the epilogue is not
  sufficient. To reach mixed-precision through Mojo, drop to
  `from std.gpu.compute.mma import mma` and hand-roll the entire
  fragment/tile/epilogue path.

This nuance is added to the rust-gpu-compute skill (item 11
expanded).

## Why we're stopping here, not pushing into Wave 21

A full hand-rolled bf16-in/f32-acc tiled matmul needs:

1. `ld_matrix[bf16, simd_width]` calls for A and B fragment loads
2. Manual smem layout + per-thread fragment math (the `TensorCore`
   wrapper hides this complexity)
3. Direct `mma()` calls in the warp inner loop
4. Hand-rolled epilogue write per PTX 9.7.13.4.8 distribution
   (groupID/tid_in_grp, 4 outputs/lane, row+col arithmetic)
5. Numerical correctness check via `vendor_blas.matmul`

That's roughly the same engineering complexity as the Wave 19 TC
kernel BUT with explicit fragment management throughout. Reasonable
size for a dedicated wave; not a quick add-on. Wave 21 is the natural
home for it.

The Wave 20 finding stands on its own: the **path exists**, the
**hardware engages**, and the **API maturity gap** is the only reason
Mojo can't yet match cuTile's bf16 lane.

## Files

**Created in `/home/codeseys/cuda-exploration/`:**

- `mojo-mma-probe/mma_probe.mojo` (~3.9KB) — the kernel
- `mojo-mma-probe/mma_probe.sass` (82 lines, ~3KB) — captured SASS,
  **`HMMA.16816.F32.BF16` × 1** on `.target sm_120a`
- `mojo-mma-probe/ANALYSIS.md` — full writeup with PTX layout reference
- `mojo-mma-probe/run.sh`, `run.log`, `.gitignore`
- `results/wave20-summary.md` (this file)

**Modified:**

- `~/.hermes/skills/mlops/rust-gpu-compute/SKILL.md` — pitfall #11
  expanded to note `mma_op` also blocks mixed-precision (not just
  `store_d`); Wave 20 finding section added

## Wave 21 candidate

`mojo-matmul-bf16/`: full hand-rolled bf16-in/f32-acc tiled matmul
using `from std.gpu.compute.mma import mma` directly. Expected
performance target: 100-130 TFLOPS at 4096³ (60-80% of cuTile's 159
TF), gap explained by:

- Mojo path uses `cp.async` not TMA (same gap as Wave 19 TF32 path —
  cuTile uses UTMALDG, Mojo doesn't auto-emit it through the high
  level)
- Manual fragment layout less optimized than cuTile's compiler-
  generated lowering

Plus optional W21+:
- TMA loads in the bf16 path via `std.gpu.host.nvidia.tma`
- Numerical correctness vs `vendor_blas.matmul`
- f16 lane in addition to bf16 (m16n8k16 f16 path also exists, same
  wrapper)
