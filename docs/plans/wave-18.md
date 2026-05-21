# Wave 18 plan — Mojo as a fifth GPU frontend

**Status:** ready for execution (pending user sign-off)
**Created:** 2026-05-20
**Predecessor:** Wave 17 in flight; Wave 18 is independent (different folders, different toolchain)
**Research grounding:** [docs/research/wave18-mojo-frontend.md](../research/wave18-mojo-frontend.md)

## Goal

Add a fifth frontend column — **Modular's Mojo** — to the cuda-exploration matrix, with the same SASS-level evidence discipline as cuTile/cuda-oxide/nvcc. Quantify whether Mojo's MLIR-based codegen produces meaningfully different SASS than libNVVM (cuTile path) or LLVM-NVPTX (cuda-oxide path) on the same RTX 5090 sm_120 hardware.

This is a **frontend-axis addition** (like Wave 12 cuTile), NOT a kernel-class addition (like Wave 15 GQA). Same phased rollout pattern as Wave 12.1.

## Why phased serial-then-parallel (not 5-subagent fan-out)

Mojo is a new toolchain. Install pitfalls, sm_120 specifics, the `@kernel` /
`DeviceContext.enqueue_function` API surface, and `.cubin` extraction recipe
all need to be discovered **before** we can write subagent prompts with the
file-ownership specificity the parallel-subagent-benchmarking discipline
requires. Wave 12.1 (cuTile) found two pip-naming pitfalls and a broken
launch syntax in the README *during* its smoke test; doing that in five
parallel subagents would have wasted four of them.

So Wave 18 is:
- **Phase A (serial, orchestrator):** install + 1M-element vecadd smoke test.
- **Phase B (serial, orchestrator):** memory-bound suite — once toolchain is known-good and SETUP.md is written, kernels are short enough that splitting them adds overhead.
- **Phase C (small parallel fan-out, 2-3 subagents):** compute-bound suite — naive matmul, tiled-microtile matmul, and (if Mojo TC works on sm_120) mixed-precision matmul, with file ownership analogous to Wave 12.3-13.1.
- **Phase D (deferred):** attention column, conditional on Phase C TC findings.

## Phase A — toolchain smoke test (orchestrator, ~1-2 hr)

### Tasks
1. Install pixi (if not present) and create `mojoproject` workspace at repo root: `mojo-workspace/` (mojoproject's `pixi.toml` lives here, kernels in subdirs reference it).
2. `pixi add mojo` — pin the version, record in SETUP.md.
3. `mojo --version` — capture in run.log.
4. Create `mojo-vecadd/` with:
   - `vecadd.mojo` — single-source host+device, 1M elements f32.
   - `run.sh` — `pixi run mojo vecadd.mojo` with output captured (`tee run.log`).
   - `SETUP.md` — install steps verbatim, version pinned, every pitfall hit.
   - `ANALYSIS.md` — correctness verification + GPU-detection trace.
5. **Acceptance:**
   - `DeviceContext().name()` reports "NVIDIA GeForce RTX 5090".
   - `max_abs_err = 0.0e+00` against CPU reference for 1M-element f32 vecadd.
   - SETUP.md lists at least: pixi version, mojo version, `MODULAR_NVPTX_COMPILER_PATH` (set or unset), driver version, any `MLRT_CUDA_DEBUG=1` output if first run failed.
6. **Stop criteria (BLOCKED.md, no further work):**
   - Mojo can't compile on Ubuntu 24.04 / WSL2 → docker fallback OR document and stop.
   - sm_120 unreachable from Mojo nightly → document and stop; this *is* the finding.
7. **Commit:** "Wave 18 Phase A: Mojo toolchain smoke test on RTX 5090 sm_120".

### Files (Phase A only)
| Path | Owner |
|---|---|
| `mojo-workspace/pixi.toml` | orchestrator |
| `mojo-workspace/.gitignore` | orchestrator |
| `mojo-vecadd/vecadd.mojo` | orchestrator |
| `mojo-vecadd/run.sh` | orchestrator |
| `mojo-vecadd/run.log` | orchestrator (gitignored if large; track if small) |
| `mojo-vecadd/SETUP.md` | orchestrator |
| `mojo-vecadd/ANALYSIS.md` | orchestrator |
| `mojo-vecadd/.gitignore` | orchestrator |

## Phase B — memory-bound suite (orchestrator, ~2 hr)

After Phase A passes, expand to the full memory-bound axis.

### Wave 18.B1: `mojo-vecadd-bench/`
- Sizes: N ∈ {1M, 16M, 64M, 256M}.
- Timing: cudaEvent (or Mojo's equivalent — investigate during Phase A).
- 1 warmup + 10 timed iters per size; report median GB/s.
- Sweep block size if Mojo exposes a tile primitive comparable to cuTile's TILE_SIZE.
- **Acceptance:** at N=256M (memory-bound regime), GB/s within ±3% of nvcc/oxide/cuTile baselines (1556-1573 GB/s).
- **Outside-range protocol:** document in ANALYSIS.md before claiming results — same discipline as Wave 1's outlier analysis.

### Wave 18.B2: `mojo-reduction/`
- 1 GB f32 sum-reduction (256M elements).
- Two implementation candidates if Mojo supports both:
  - warp-shuffle / parallel reduction (matches nvcc/oxide path → expected ~1520 GB/s parity)
  - tile-load-based reduction (matches cuTile path → potentially ~1700 GB/s with TMA)
- **Critical SASS check:** `cuobjdump --dump-sass mojo-reduction.cubin | grep -c UTMALDG`
  - >0 → Mojo lowers to TMA bulk loads (interesting — joins cuTile in this category)
  - =0 → Mojo lowers to warp-shuffle (parity with nvcc/oxide)
- This is the **single most informative datapoint** in Phase B for understanding Mojo's compiler character.

### Phase B re-bench discipline
Per Wave 12 thermal-drift lesson: when Phase B lands its results, **re-run
`cutile-vecadd-bench/`, `cutile-reduction/`, `oxide-vecadd-bench/`,
`oxide-reduction/`, `cuda-vecadd-bench/`, `cuda-reduction/` in the same
session on the same idle GPU**, and update their CSVs in the same commit.
Don't compare against weeks-old `results/scaling.csv` numbers.

`nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu` snapshot
captured at run start, recorded in run.log per existing convention.

### Phase B commit
"Wave 18 Phase B: Mojo memory-bound suite — vecadd N-sweep + reduction with SASS diff vs cuTile/nvcc/oxide".

## Phase C — compute-bound suite (small parallel fan-out, 2-3 subagents)

After Phase B lands a working Mojo kernel pattern + bench harness, Phase C
fans out to compute-bound work. Two-three disjoint folders, dispatched in one
`delegate_task(tasks=[...])` batch.

### File-ownership table (Phase C)

| Worker | Directory | Reads (RO) | Writes |
|---|---|---|---|
| C1 `mojo-matmul` | `mojo-matmul/` | `cuda-matmul/matmul.cu`, `oxide-matmul/src/main.rs` (algorithm reference), `mojo-vecadd/vecadd.mojo` (Mojo-API template), `mojo-vecadd/SETUP.md` (toolchain) | `mojo-matmul/{matmul.mojo, run.sh, run.log, ANALYSIS.md, .gitignore}` |
| C2 `mojo-matmul-tiled` | `mojo-matmul-tiled/` | C1's `matmul.mojo` (template), `oxide-matmul-tiled-microtile/src/main.rs` (microtile pattern), `cutile-matmul-tiled/main.py` (tile pattern) | `mojo-matmul-tiled/{matmul_tiled.mojo, run.sh, run.log, ANALYSIS.md, .gitignore}` |
| C3 `mojo-matmul-mixed` | `mojo-matmul-mixed/` | C2's `matmul_tiled.mojo`, `cutile-matmul-tiled-mixed/main.py` (mixed-precision pattern), `cublas-half-precision/main.cu` (cuBLAS hgemm baseline) | `mojo-matmul-mixed/{matmul_mixed.mojo, run.sh, run.log, ANALYSIS.md, .gitignore}` |

C3 is **conditional on Phase B finding** that Mojo can target sm_120's
4th-gen tensor cores (e.g. `mma.sync.m16n8k16` for f16). If Phase B's reduction
SASS shows Mojo emits no `mma.sync` family, C3 is **skipped** — file
`mojo-matmul-mixed/BLOCKED.md` describing the constraint, don't fake the
attempt.

### Per-cell acceptance test

| Cell | Correctness | Bench expected | Sanity |
|---|---|---|---|
| C1 `mojo-matmul` (naive f32) | max_abs_err vs CPU ≤ 1e-2 at N=4096 | TFLOPS ∈ [4, 50] (between nvcc 6.4 and tiled 38) | scalar FFMA path; `cuobjdump --dump-sass | grep -c FFMA > 0`; HMMA = 0 |
| C2 `mojo-matmul-tiled` (f32 microtile) | max_abs_err ≤ 1e-2 | TFLOPS ∈ [25, 50] (between nvcc-tiled 38 and oxide-microtile 45) | shared-memory + register tile; FFMA per inner-loop body = 8 or higher; HMMA = 0 |
| C3 `mojo-matmul-mixed` (f16 input + f32 acc) | max_abs_err ≤ 1e-1 (TC tolerances) | TFLOPS ∈ [80, 220] (cuTile 172, cuBLAS 218; expect Mojo somewhere in between) | HMMA > 0; HMMA-sanity-formula in [0.5, 2.0]; LDG.E for inputs |

### Phase C commit policy
Per-cell commits as Wave 17 does:
- `Wave 18 Phase C C1: mojo-matmul -- naive f32 baseline`
- `Wave 18 Phase C C2: mojo-matmul-tiled -- f32 register microtile`
- `Wave 18 Phase C C3: mojo-matmul-mixed -- f16 TC via mma.sync` (or `BLOCKED: no TC API on sm_120`)

## Phase D — attention column (deferred, conditional)

Only if Phase C confirms a working Mojo TC path on sm_120. If yes, mirror
the Wave 15 GQA / Wave 16 multi-frontend-attention pattern with `mojo-attn-gqa/`.
Avoid sink-aware FA until [issue #6198](https://github.com/modular/modular/issues/6198)
is fixed upstream.

If Phase C finds **no** Mojo TC on sm_120, Phase D is a no-op; Mojo stays in
the "no-TC ceiling" frontend category alongside cuda-oxide, and the README
table reflects that.

## Wave 18 budget

- **Phase A:** ~1-2 hr orchestrator wall-clock.
- **Phase B:** ~2 hr orchestrator wall-clock (kernels are short).
- **Phase C:** ~10-15 min wall-clock if 3 parallel subagents (~600s budget each),
  plus orchestrator-serial benchmarking after kernels land.
- **Phase D:** TBD, conditional.

## Out of scope for Wave 18

- MAX serving / model inference (different class of benchmark).
- AMD MI300X cross-vendor (no hosted access).
- Reproducing Modular's flagship sm_100a Blackwell matmul (datacenter-only).
- Mojo's CPU SIMD path (we are a GPU-frontend repo).

## After Wave 18 completes

1. Update README headline tables to add a fifth column "Mojo".
2. Update [`results/`] with `wave18-summary.md`.
3. Update the `rust-gpu-compute` skill (despite the misleading name — this skill
   is the de facto multi-frontend GPU-comparison knowledge base) with a Mojo
   section: install pitfalls, sm_120 caveats, API gotchas, our perf findings.
4. Optional: file an upstream issue against modular/modular if Phase B/C
   surfaces a clear codegen gap on sm_120 (e.g. matmul missing `mma.sync` on
   sm_120 even when explicitly targeting TC), with our SASS evidence.

## Open questions Wave 18 should answer

1. Does `mojo build --target-accelerator=sm_120` produce a single-arch native
   `.cubin` we can disassemble with `cuobjdump`?
2. Is there a Mojo equivalent of cuTile's `ct.load`/TMA primitive, and if so
   does it lower to `UTMALDG` on sm_120?
3. Does Mojo's MMA API reach the 4th-gen TCs on sm_120 (`mma.sync.m16n8k16` or
   newer family), or is its TC API gated to the same sm_100a tcgen05 path
   that issue #5707 documents?
4. What's the user-facing cost of Mojo's "single-source compiles to multi-vendor"
   promise on consumer Blackwell specifically — is it equivalent to nvcc with
   no porting effort, or are there sm_120-specific pessimizations vs. the
   datacenter blog-post numbers?
