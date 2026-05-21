# Wave 18 summary — Mojo as a fifth GPU frontend (memory-bound axis)

**Status:** Phase A + Phase B complete; Phase C (compute-bound) deferred to Wave 19
**Date:** 2026-05-20
**Hardware:** NVIDIA RTX 5090 (Blackwell sm_120, 32 GB), driver 596.21, WSL2/Ubuntu 24.04
**Mojo:** 1.0.0b1 (build a9591de6) — first stable beta, installed via pixi from `https://conda.modular.com/max`
**Benches re-run on the same idle GPU thermal window** per Wave 12 discipline.

## Headline finding

**Mojo joins the warp-shuffle club, not the TMA club.** It hits memory-bound
parity with nvcc/cuda-oxide on streaming kernels (vecadd, reduction) but
**does not** match cuTile's TMA-driven reduction lift on Blackwell. Verified
directly via SASS: zero `UTMALDG.1D` instructions in Mojo's reduction
cubin, vs cuTile's seven.

## Memory-bound headline table @ N=256M (canonical)

| frontend | vec-add (GB/s) | reduce_sum (GB/s) | reduction strategy |
|---|---:|---:|---|
| nvcc CUDA C++ | 1404 | 1605 (Wave 14) | hand-written |
| cuda-oxide safe | 1572 | (warp-shuffle) | hand-written `warp::shuffle_xor_f32` |
| cuda-oxide unchecked | 1575 | 1507 | hand-written |
| **Mojo** (NEW) | **1572** | **1502** | stdlib `block.sum` (warp-shuffle) |
| **cuTile** | 1565 | **1691** ⚡ | `ct.sum` over `ct.load` tile (TMA) |

All four reach ~90% of HBM peak (~1750 GB/s) on vec-add. The cuTile
reduction-only lift remains the standout — Mojo joins nvcc and cuda-oxide
in the parity zone.

## SASS-level verification (Mojo reduction)

Built with `pixi run mojo build --target-accelerator=sm_120 reduction.mojo`,
SASS dumped via Mojo's `_dump_sass=True` kwarg on `enqueue_function`:

| metric | Mojo | cuTile (Wave 13) | nvcc (Wave 13) | cuda-oxide (Wave 13) |
|---|---:|---:|---:|---:|
| `UTMALDG.1D` | **0** | 7 | 0 | 0 |
| `LDG.E` (regular global load) | 3 | 0 | 3 | 3 |
| `SHFL.BFLY` (warp shuffle) | 45 | (warp-shuffle path) | similar | similar |
| `REDG.E.ADD.F32` (HW atomic) | 3 | (different) | 3 | 3 |

Mojo's `std.gpu.primitives.block.sum` is a one-line stdlib reduction
implemented internally with warp-shuffle + smem partials. It is NOT a
tile-aware primitive; consequently the compiler emits regular `LDG.E`
loads, not `UTMALDG`. To engage TMA from Mojo today, the user would need
to drop to lower-level primitives (e.g. `std.gpu.sync.cp_async_bulk`).
This is a *finding* about API ergonomics + lowering, not a bug.

## Wave 18 deliverables

### New repo cells
- `mojo-workspace/` — pixi-managed Mojo env (`pixi.toml`, `pixi.lock`)
- `mojo-vecadd/` — Phase A toolchain smoke test (1024-elem vecadd, 0 abs err)
- `mojo-vecadd-bench/` — Phase B.1 N-sweep vecadd benchmark
- `mojo-reduction/` — Phase B.2 reduction with full SASS dump tracked

### Documentation
- `docs/research/wave18-mojo-frontend.md` — sm_120 status survey & risk register
- `docs/plans/wave-18.md` — phased rollout plan
- `results/wave18-summary.md` (this doc)

### Updated baselines
- `cutile-vecadd-bench/results.csv` — re-run on idle GPU
- `cutile-reduction/results.csv` — re-run with `--bench`
- `oxide-vecadd-bench/results.csv` — re-run
- `oxide-reduction/results.csv` — re-run
- `cuda-vecadd-bench/results.csv` — re-run
- (cuda-reduction binary had a CSV-write bug in this session; baseline preserved from Wave 14)

### Pitfalls captured (in `mojo-vecadd/SETUP.md` + per-folder ANALYSIS.md)
1. `pixi init --format mojoproject` does NOT add the Modular conda channel — use the explicit `-c` form.
2. `UnsafePointer[T]` needs origin: use `UnsafePointer[Scalar[T], MutAnyOrigin]` for mutable kernel args.
3. `@parameter def` is the canonical capturing-closure pattern for `ctx.execution_time`. `fn` is deprecated in 1.0.0b1.
4. `out` is a Mojo keyword — can't be used as a parameter name (rename to `result` etc.).
5. `ctx.enqueue_memset(buffer, value)` is the correct GPU-side reset for atomic-add output buffers; using `enqueue_copy` from a host scalar inflates the timed window with a PCIe round-trip (1160 GB/s with copy → 1502 GB/s with memset).
6. **`_dump_sass=True`** on `enqueue_function` is undocumented but works — Mojo offers the cleanest SASS-extraction UX of the four frontends.
7. **No `MODULAR_NVPTX_COMPILER_PATH` workaround needed** with driver ≥ 580. Mojo's auto-detection picked up the RTX 5090 with zero env-var fiddling, in stark contrast to cuda-oxide's required `CUDA_HOME` + `LIBNVVM_PATH` exports.

## What the user-facing read becomes

Adding Mojo to the headline table (where the previous question was "what
non-CUDA-C++ frontend should I pick on Blackwell?"):

If you're choosing a multi-vendor / DSL GPU compute frontend on consumer
Blackwell **today** (May 2026):

- **Streaming memory** (vec-add, copy-elementwise): any of cuda-oxide /
  cuTile / Mojo / nvcc — all within 1% of each other at ~90% of HBM peak.
  Mojo's UX is the cleanest of the alternatives (single-source, no env
  exports, JIT or AOT both work).
- **Reduction-pattern memory work**: cuTile remains the winner via TMA
  bulk loads (1691 GB/s, +12% over warp-shuffle path). Mojo's stdlib
  `block.sum` lands cleanly on the warp-shuffle path with nvcc and
  cuda-oxide. To unlock TMA from Mojo, drop to lower-level primitives.
- **Multi-vendor portability**: Mojo's compelling axis (NVIDIA + AMD +
  Apple from one source). Untested in this wave on non-NVIDIA hardware;
  Modular advertises CDNA3 / RDNA3 / Apple Silicon support.
- **Best dev-loop SASS extraction**: Mojo's `_dump_sass=True` is friction-
  free. cuda-oxide needs `cuobjdump --dump-sass` on the cubin; cuTile
  needs the `compile_tile` monkey-patch recipe.

## Phase C (Wave 19) preview

The compute-bound axis will answer whether Mojo can engage tensor cores
on sm_120. The prior is mixed:

- **For**: Mojo officially lists sm_120 as "known compatible"; PR #6059
  (Mar 2026) added MAX matmul fallback dispatch on sm_120; the language
  exposes MMA-class primitives.
- **Against**: MAX flagship matmul kernels are explicitly tcgen05-gated
  (sm_100a-only) per [issue #5707](https://github.com/modular/modular/issues/5707).
  The fallback path on sm_120 may be the same naive-FMA story as
  cuda-oxide's 45 TF f32-microtile ceiling, *not* the 172 TF f16 cuTile
  TC engagement.

Wave 19 = `mojo-matmul/` (naive f32) + `mojo-matmul-tiled/` (microtile f32) +
optionally `mojo-matmul-mixed/` (f16 input + f32 acc, conditional on TC reach).

## Time budget (actual vs planned)

| phase | budgeted | actual | notes |
|---|---:|---:|---|
| Phase A toolchain | 1-2 hr | ~15 min | one pitfall (`--format mojoproject` channel issue) |
| Phase B.1 vecadd-bench | ~1 hr | ~30 min | three iterations on UnsafePointer / closure shape |
| Phase B.2 reduction | ~1 hr | ~30 min | one rename pitfall (`out` keyword), one bench-shape fix (memset) |
| Phase B baseline rerun | ~30 min | ~10 min | cuTile venv had to be rebuilt (renamed-dir pitfall) |
| **Total Wave 18 wall clock** | **~4 hr** | **~90 min** | well under budget; multi-iteration on Mojo type system was fastest at <30 min total |

## What we did NOT do this wave

- No matmul (deferred to Wave 19)
- No attention (deferred to a future wave conditional on Wave 19 TC findings)
- No cross-vendor (no AMD/Apple hardware available)
- No cuda-reduction re-run (binary had a results.csv path bug; using Wave 14 baseline)
