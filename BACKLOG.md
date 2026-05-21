# cuda-exploration backlog

Vision: ship a comprehensive, third-party-citeable evaluation of NVlabs/cuda-oxide v0.1.0 vs CUDA C++ baselines on Blackwell. Quantify the cost of Rust safety, identify compiler gaps, document setup pitfalls. Independent third-party work, not affiliated.

## Status

- ✅ v0 shipped (initial commit `5065f3e`): naive 4096×4096 f32 matmul, three backends, per-folder ANALYSIS.md.
- ✅ Wave 1-3 shipped: cudaEvent timing, scaling sweep, cuBLAS + tiled, FMA escape hatch, libNVVM corrigendum.
- ✅ Wave 4 shipped: reduction (W4A), bandwidth bench (W4B), libNVVM causal isolation (W4C — inconclusive).
- ✅ Wave 5 shipped: SASS-level analysis identifying `LDG.E.CONSTANT` vs `LDG.E` as the residual-gap root cause.
- ✅ Wave 6 shipped: ran cuda-oxide's `gemm_sol`, `tma_copy`, `tcgen05_matmul` examples — characterizing consumer (sm_120) vs datacenter (sm_100) Blackwell support.
- ✅ Wave 7 shipped: register-microtile + fmuladd cuda-oxide tiled matmul; reaches nvcc-tiled parity at N=1024, 60% at N=4096.
- ✅ Wave 8 shipped: 2D Gaussian Splatting forward rasterizer (toy scene); kernel handled cleanly.
- ✅ Wave 8.5 shipped: procedural rings + smiley test scenes through the same kernel.
- ✅ Wave 9 shipped: real public 3DGS scene (Luigi figurine) rendered through the cuda-oxide kernel.
- ✅ Wave 10 shipped: canonical Utsuho scene (53,671 gaussians) rendered.
- ✅ Wave 11 shipped: nvcc CUDA C++ apples-to-apples reference; **byte-identical pixels** + arithmetically-identical SASS on the Utsuho scene.
- ✅ Repo cleanup: untracked regenerable build artifacts (kept evidence-cited PTX/LL files); standardized .gitignores; tracked Cargo.lock everywhere.
- Open question: SH-degree-3 view-dependent color (currently both pipelines use SH degree 0); tile-binning optimization for big scenes.

## Items

### P0 — methodology rigor (blocks any new claim)

- [ ] **M1: cudaEvent timing for cuda-oxide.** v0 used wall-clock + `stream.synchronize()` for cuda-oxide; nvcc uses `cudaEventRecord`. Fix to apples-to-apples. Eliminates ~5-50µs/iter sync overhead from the comparison.
  - Files: `oxide-matmul/src/main.rs`. Owner: wave 1 subagent A.
  - Acceptance: oxide bench reports `gpu_ms` (event-based) and `cpu_wall_ms` separately; results table shows `gpu_ms`.

- [ ] **M2: Size-scaling sweep.** Run all benches at N ∈ {1024, 2048, 4096, 8192}. One data point isn't a curve. Look for: does the safe-vs-unchecked gap scale with N (it should — more inner-loop iterations = more bounds checks)? Does cuda-oxide vs nvcc gap stay constant?
  - Files: introduce `--size N` flag in each binary. Owner: wave 1 subagent B (parallel with M1; different binaries don't conflict).
  - Acceptance: `results/scaling.csv` with columns `(impl, N, best_ms, median_ms, tflops)`.

### P1 — close the obvious gaps in cuda-oxide PTX

- [ ] **F1: Fast-math / FMA contraction flag.** PTX shows zero `fma.rn.f32`. Investigate: does cuda-oxide expose a `#[fast_math]` kernel attribute, a global `RUSTFLAGS` switch, or a `cargo oxide build --release-fast` mode? If yes, re-bench. If no, that's the upstream issue.
  - Files: `oxide-matmul/src/main.rs` adds a third kernel `matmul_fastmath` if the toolchain supports it; analysis writes results.
  - Acceptance: either we find a switch (and TFLOPS jumps) or we have a definitive "no, here are the issues we tried" paragraph for the upstream report.

- [ ] **F2: __restrict__ equivalent for `ld.global.nc`.** nvcc uses read-only cache via `__restrict__`. cuda-oxide takes `&[T]` references which should imply non-aliasing already. Investigate why PTX doesn't emit `ld.global.nc` and whether marking with raw `*const f32` + a hint changes it.
  - Owner: same wave as F1.

### P1 — broader comparison axes

- [ ] **C1: cuBLAS reference baseline.** Add `cublas-matmul/` that calls `cublasSgemm`. ~80-90 TFLOPS on RTX 5090. Quantifies how much of the gap is "naive algo fundamentally bad" vs "compiler gap." Without this, we don't know if our 6 TFLOPS naive baseline is even a meaningful comparison point.
  - Files: new folder `cublas-matmul/`. Owner: wave 2 subagent A.
  - Acceptance: `cublas-matmul/{matmul.cu,matmul,run.log,ANALYSIS.md}` with a TFLOPS number.

- [ ] **C2: Tiled shared-memory matmul.** Both cuda-oxide and nvcc, with 16×16 or 32×32 tiles. ~30-50 TFLOPS expected. Tests cuda-oxide's `SharedArray` API which is the next-most-important feature after kernels. Real apples-to-apples beyond the naive case.
  - Files: new folder `oxide-matmul-tiled/` and `cuda-matmul-tiled/`. Owner: wave 2 subagent B.
  - Acceptance: both binaries hit ≥20 TFLOPS, results filed.

### P2 — write-up + upstream

- [ ] **U1: NVlabs/cuda-oxide upstream issue.** With our PTX evidence, file an issue on `NVlabs/cuda-oxide` flagging the FMA + ld.global.nc gap. Be precise: link to commit, include PTX excerpts, propose `#[fast_math]` annotation if not already supported.
  - Files: draft in `docs/upstream-issue-fma.md`; user submits.
  - Acceptance: draft text reviewed by Phase 8.

- [ ] **W1: SUMMARY.md / final writeup.** A standalone results writeup with all scaling curves + the "if you're a Rust dev considering cuda-oxide today, what should you know?" guidance. Less of a README, more of a blog-post-shaped artifact.
  - Files: `SUMMARY.md`. Owner: wave 3.
  - Acceptance: cross-family-review-clean.

### P2 — lower priority / nice-to-have

- [ ] **N1: Multiple block sizes.** 8×8, 16×16, 32×8 etc. for the naive matmul. Occupancy effects.
- [x] **N2: Reduction kernel.** Different access pattern than matmul; tests warp-reduce primitives. Done in Wave 4 W4A: cuda-oxide hits 96% of nvcc on 1-GB sum-reduction (1451 vs 1517 GB/s).
- [ ] **N3: GitHub Actions CI.** Hard without a GPU runner. Document as not-applicable until self-hosted runner available, or use a lightweight "build-only" CI that catches Rust compile errors.

### P3 — opened by Wave 4-6

- [x] **N4: Memory-bandwidth bench at varying N.** Done in Wave 4 W4B: cuda-oxide and nvcc within 0.1% at N=64M (~90% of HBM peak).
- [x] **N5: SASS-level disassembly of remaining gap.** Done in Wave 5: `LDG.E.CONSTANT` vs `LDG.E` is the residual-gap root cause; both compilers unroll 8x.
- [x] **N6: Drop-in cuda-oxide `gemm_sol` / `tcgen05_matmul`.** Done in Wave 6: PTX builds, runtime fails on consumer sm_120 Blackwell — datacenter-only (sm_100/sm_100a). TMA works on sm_120.
- [ ] **U2: Upstream issue for `LDG.E.CONSTANT` / read-only cache hint.** Wave 5 SASS evidence and Wave 11 confirmation support drafting a second upstream issue alongside the FMA one. Suggested wording: "rustc-codegen-cuda doesn't emit `LDG.E.CONSTANT` for `&[T]` reads where the slice is shared and immutable; equivalent CUDA C++ with `const __restrict__` does." Patch space: NVPTX lowering of slice-deref to add `nontemporal!`/`invariant.load` hints, or expose a `#[restrict]` attribute on slice params.
- [x] **W2: Wave 7 candidate — quantify register-microtile lift in cuda-oxide tiled kernel.** Done in Wave 7: 4×4 microtile + fmuladd hits 27-28 TFLOPS at N=1024 (matches nvcc-tiled 24.5), 16-17 TFLOPS at N=4096 (60% of nvcc).
- [ ] **W3: Wave 8 candidate — try cuda-oxide on a Hopper SXM5 (H100) cloud instance** to run `gemm_sol` end-to-end and compare to upstream's claimed 868 TFLOPS on B200.

### P3 — opened by Wave 7-11

- [x] **G1: 2D Gaussian Splatting toy renderer in cuda-oxide.** Done in Wave 8.
- [x] **G2: Real-public 3DGS scene through cuda-oxide.** Done in Wave 9 (Luigi) + Wave 10 (Utsuho).
- [x] **G3: Apples-to-apples nvcc CUDA C++ port of the 3DGS pipeline.** Done in Wave 11. Pixel-identical, SASS-identical except for LDG.E.CONSTANT hint.
- [ ] **G4: SH degree 3 evaluation.** Both implementations currently use SH degree 0 (diffuse only). Adding view-dependent SH3 color (16 coefficients × 3 channels per gaussian) would make renders show specular variation. ~100 lines per implementation. Cheap and high-value.
- [ ] **G5: Tile-binning optimization.** Currently every pixel iterates every gaussian (O(N·W·H)). Real 3DGS bins gaussians into 16×16 screen tiles so each pixel only touches the ones overlapping its tile (O(K·W·H) where K is gaussians-per-tile, typically 50-200). Would 10-50× speed up rendering on 100k+-gaussian scenes. Significant work: GPU-side prefix sum, scatter, sort.
- [ ] **G6: Bigger scene** (Mip-NeRF 360 bicycle/garden, ~500k gaussians). Easy without tile-binning; useful with.
- [ ] **G7: Backward pass / training.** Would require gradient computation for all the projection math. Out of scope for a benchmark repo, but a natural Wave-X candidate if we ever wanted to claim "full 3DGS in cuda-oxide."

## Out of scope (for now)

- Multi-GPU
- ~~Mixed precision (fp16, bf16)~~ — **DONE** in Wave 13.1 (cuTile axis: f16/bf16/tf32 via ct.mma → tensor cores) and Wave 14.1 (cuBLAS hgemm/bgemm/sgemm-tf32 baselines)
- ~~Tensor Core / WGMMA~~ — investigated in Wave 14.4: cuda-oxide v0.1.0 has no usable TC API on RTX 5090 sm_120 (wgmma is Hopper-only and stubbed; tcgen05 is datacenter-only). cuTile DOES expose TC via ct.mma, comparison documented.
- Fixing wgpu on WSL — well-documented elsewhere; we already showed the limitation

## Wave 12-14 status (May 2026, the cuTile axis)

- ✅ **Wave 12.1**: cuTile vec-add bench, parity within 1% vs nvcc/oxide
- ✅ **Wave 12.2**: cuTile reduction, +11% vs nvcc/oxide (TMA bulk loads)
- ✅ **Wave 12.3**: cuTile naive matmul, broadcast-and-sum (4× slower than oxide naive)
- ✅ **Wave 12.4**: cuTile tiled matmul via `ct.mma`, 7.57 TF f32 (Wave 13 reframed)
- ✅ **Wave 13.1**: cuTile mixed-precision (f16/bf16/tf32), 172.5 TF f16 — TC engaged
- ✅ **Wave 13.2**: SASS analysis explaining reduction win + matmul f32 fallback
- ✅ **Wave 14.1**: cuBLAS hgemm/bgemm/sgemm-tf32 baselines for fair comparison
- ✅ **Wave 14.2**: 2 upstream issue drafts at `docs/upstream-issues/`
- ✅ **Wave 14.4**: cuda-oxide TC verdict — no usable TC API on consumer Blackwell

### Wave 15+ candidates

- [ ] **W15.1: file the upstream issues** at github.com/nvidia/cutile-python — drafts ready
- [ ] **W15.2: per-N cuTile/cuBLAS ratio writeup** — does the 79% ratio hold at smaller N?
- [ ] **W15.3: 3DGS rasterizer port to cuTile** — completeness alongside oxide and nvcc 3DGS
- [ ] **W15.4: revisit when cuda-oxide ships `cuda_device::mma`** — track NVlabs/cuda-oxide releases
- [ ] **G4-G7**: 3DGS deepening (SH3, tile-binning, larger scenes, backward pass)

### Wave 17+ in flight

- 🚧 **Wave 17 (research grounded, ADRs landed)**: KDA + oxide-MLA + GDN-other-frontends. See `docs/plans/wave-17.md`.
- 📋 **Wave 18 (research grounded, plan ready, pending sign-off)**: Mojo as a fifth frontend column. See `docs/plans/wave-18.md` and `docs/research/wave18-mojo-frontend.md`. Phased serial-then-parallel rollout (Phase A toolchain smoke test → Phase B memory-bound → Phase C compute-bound → Phase D conditional attention). Critical sm_120 question: does Mojo emit `mma.sync` for tensor cores, or is it gated to sm_100a-only tcgen05 like the MAX flagship matmul (issue #5707, PR #6059)?

## Wave plan (legacy, original Wave 1-3 outline)

- **Wave 1 (parallel, 2 subagents):** M1 + M2 — methodology fixes. Independent file ownership. Runs first because all later results depend on the new timing baseline.
- **Wave 2 (parallel, 2 subagents):** F1+F2 (compiler gaps) + C1+C2 (broader axes). Some cross-talk on results format but file-disjoint.
- **Wave 3 (parallel, 2 subagents):** U1 (upstream draft) + W1 (final writeup). Both consume wave-1 + wave-2 outputs.

## Budget

- 3 waves × ~3 parallel subagents/wave + Phase 8 (3-way review) = ~12 subagent calls. Token budget: ~150-300k summary tokens to orchestrator.
- Wall-clock: empirical runs themselves (<5 min each, batched) are smaller than the subagent reasoning time. Whole loop ~30-60 min realistic.

## Cross-cutting risks

- **WSL2 thermals / variability:** v0 saw 5-10% noise in median. With multiple sizes the cv may rise. Mitigation: 10+ iters, report median + IQR, not mean.
- **CUDA 12.0 PTX-JIT to sm_120:** every result is JIT'd, not native. Could mask Blackwell-specific behavior. Document as known limitation; don't try to fix in this loop (would require CUDA 13 install).
- **cuBLAS version skew:** ships with the toolkit. Whatever 12.0 has, that's the baseline. Document the version.
- **Subagent claims that they "ran" something but didn't:** common failure mode. Each wave's commit message must include the exact `./target/release/<bin>` invocation that produced the new run.log; reviewer confirms by reading the log header (which has timestamps) before signing off.
