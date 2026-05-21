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
- ✅ **Wave 17 W1 SHIPPED 2026-05-21**: 5-cell parallel attention-matrix expansion. cuda-attn-mla (24.17 TF native / 21.32 TF padded), oxide-attn-mla (24.70 TF best), cuda-attn-gdn (417.7 GB/s), oxide-attn-gdn (correctness only, no bench harness yet — see W22.6), cutile-attn-kda (344.7 GB/s best at small shape, large IQR — see W22.7). All correctness PASS within ADR-0004/0005/0006 thresholds. Headline cross-cell finding: cuTile's UTMALDG TMA path beats nvcc's LDG.E.128 vector loads on GDN despite W1c emitting LDG.E.128 as predicted (W22.8 investigation candidate). See [`results/wave17-summary.md`](results/wave17-summary.md).
- 📋 **Wave 17 W2 candidates**: W2c cublas-attn-mla (depends on W1a HMMA=20 + qk_pad choices, both now known); W2d cutile-attn-gqa BLOCK_M=128 sweep (close 24% gap to cuBLAS hgemm).
- 📋 **Wave 17 U candidates** (docs ready): U1+U2 upstream-issue drafts at `docs/upstream-issues-oxide/` for NVlabs/cuda-oxide (FMA contraction + LDG.E.CONSTANT). User submits.
- ✅ **Wave 18 Phase A+B SHIPPED 2026-05-20**: Mojo as a fifth frontend column, memory-bound axis. Commits `8d82d33` (research+plan), `8f70304` (Phase A toolchain smoke), `c111230` (Phase B vecadd-bench + reduction with SASS). Headline finding: Mojo joins the warp-shuffle club, NOT the TMA club — `block.sum` lowers to warp-shuffle (parity with nvcc/oxide at 1502 GB/s on N=256M reduction), 12% behind cuTile's TMA path. See `results/wave18-summary.md` and `mojo-reduction/ANALYSIS.md`.
- ✅ **Wave 19 — SHIPPED**: Mojo Phase C (compute-bound). `mojo-matmul/` naive f32 (7.1 TF, parity with nvcc/oxide naive) + `mojo-matmul-tc/` Tensor Core probe (**55.5 TFLOPS f32→TF32→f32 at 4096³**, 7.8× the naive baseline). **Hard question answered: Mojo CAN engage tensor cores on sm_120 via `from layout.tensor_core import TensorCore`** — SASS shows 64 × `HMMA.1684.F32.TF32`, no tcgen05 (correctly takes the legacy `mma.sync` path, not the sm_100a-only Blackwell tcgen05 path). Same-dtype constraint blocks bf16/f16 mixed-precision through the high-level wrapper. See [`results/wave19-summary.md`](results/wave19-summary.md), [`mojo-matmul-tc/`](mojo-matmul-tc/).
- ✅ **Wave 20 — SHIPPED (probe only, full harness deferred)**: Mojo bf16-in/f32-acc TC path proven on sm_120. `mojo-mma-probe/` shows `HMMA.16816.F32.BF16` × 1 on `.target sm_120a` from a single-warp `from std.gpu.compute.mma import mma` call. **The Wave 19 same-dtype constraint is in the `TensorCore` wrapper (BOTH `mma_op` and `store_d`, not just `store_d`), NOT in the hardware or `mma()` primitive.** Attempted re-using the Wave 19 kernel structure with `a_type=bf16, c_type=f32`: failed at `tensor_core.mojo:842:10` (`mma_op` internal rebind to bf16) AND at `tensor_core.mojo:781:9` (`store_d` same-dtype). See [`mojo-mma-probe/ANALYSIS.md`](mojo-mma-probe/ANALYSIS.md) and [`results/wave20-summary.md`](results/wave20-summary.md).
- ✅ **Wave 21 — SHIPPED**: Mojo bf16-in/f32-acc tiled matmul harness (closes Wave 20 gap). `mojo-matmul-bf16/` ships **79.3 TFLOPS @ M=N=K=4096** (median of 10 iters), 49.6% of cuTile bf16 (159.95 TF), 36.2% of cuBLAS bgemm (219.3 TF), +43% over Wave 19's TF32 path. Numerical correctness validated at M=64/256/4096 with tightened `atol=1e-2 + rtol=1e-3·|ref|` tolerance (R3 review caught the original atol=10 was 5500× looser than observed error). SASS shows `HMMA.16816.F32.BF16 × 16` on `.target sm_120a`, `UTMALDG × 0` (cp.async path; explains gap to cuTile's TMA path). **Key technique: hybrid `TensorCore[bf16, bf16, Index(16,8,16)]()` for load_a/load_b ONLY (no same-dtype constraint there) + raw `mma()` + hand-rolled epilogue per PTX 9.7.13.4.8.** See [`results/wave21-summary.md`](results/wave21-summary.md), [`mojo-matmul-bf16/`](mojo-matmul-bf16/).
- 📋 **Wave 22 candidates**:
  - W22.1: TMA loads via `std.gpu.sync.cp_async_bulk` to close the cp.async-vs-TMA gap (~120-130 TF expected)
  - W22.2: f16 lane at m16n8k16 (`HMMA.16816.F32.F16`) — same scaffolding, swap dtype
  - W22.3: Padded-smem layout to remove residual bank conflicts (~5-10% on top of W22.1)
  - W22.4: FP8 lane (e4m3/e5m2) at m16n8k32 — full hand-roll required (no LLVM intrinsic, inline asm only)
  - W22.5: Attention column with bf16 matmul + softmax (Phase D of Wave 18 lineage, now unblocked)
  - W22.6: oxide-attn-gdn timed bench harness (~50 LOC, mirror oxide-attn-gqa pattern) — Wave 17 W1d follow-up
  - W22.7: cutile-attn-kda larger-shape sweep (B=4 H=64 d_k=d_v=256) to saturate GPU and claim 8× state-traffic advantage — Wave 17 W1e follow-up
  - W22.8: cuda-attn-gdn TMA-vs-LDG.E.128 investigation — why does cuTile's UTMALDG beat thread-vectorized LDG.E.128 on Blackwell? — Wave 17 W1c hardware-level follow-up
- ✅ **Wave 22.2 SHIPPED 2026-05-21**: mojo-matmul-f16 (f16 lane) at **79.44 TF median @ 4096³**, essentially identical to Wave 21 bf16 (79.26 TF). Confirms Mojo's m16n8k16 path is dtype-agnostic on consumer Blackwell. See [`results/wave22-partial-summary.md`](results/wave22-partial-summary.md).
- ✅ **Wave 22.3 SHIPPED 2026-05-21 (correctness only)**: padded-smem variant of mojo-matmul-bf16. Critical finding: Mojo's `TensorCore.load_a/load_b` does NOT emit `ldmatrix`, only scalar `LDS.U16`. Wave 21 reviewer R2's "padded smem unlocks ldmatrix.x4" hypothesis cannot apply. Variant parked as counter-example.
- ✅ **Wave 22.6 SHIPPED 2026-05-21 (author only)**: oxide-attn-gdn `--bench` mode added with cudaEvent timing. Bench numbers TBD (orchestrator runs in next loop).
- ✅ **Wave 22.8 SHIPPED 2026-05-21**: Wave 17 W1c hypothesis REJECTED. Both nvcc and cuTile GDN kernels have ZERO TMA. cuTile's win is Blackwell async-barrier producer/consumer warp-specialization with 100KB smem + REG=255. nvcc weakness is TPB=16 (half-a-warp) launch geometry. Hardware-API gap, not compiler-quality. New W22.9 (`cuda::pipeline`) and W22.10 (`cuTensorMapEncodeTiled`) candidates added. See [`docs/research/wave17-w1c-tma-vs-ldg128-investigation.md`](docs/research/wave17-w1c-tma-vs-ldg128-investigation.md).
- ✅ **Wave 22.6 SHIPPED 2026-05-21 (full)**: oxide-attn-gdn `--bench` mode wired with cudaEvent timing. Bench result: **276.1 GB/s best** at qwen3_next_decode (66% of nvcc's 417.7 GB/s, 45% of cuTile's 610 GB/s). Confirms cuda-oxide no-TC ceiling pattern; FFMA-only path with two block-wide tree reductions per block, no LDG.E.128 auto-vectorization.
- ✅ **Wave 22.7 SHIPPED 2026-05-21**: cutile-attn-kda larger-shape sweep. **Saturation regime hits 1170 GB/s = 65% of HBM peak** (3.6× the W1e baseline 324 GB/s). At GDN-parity shape qwen3_next_decode (B=1 H=16 d_k=d_v=256): KDA=611.2 GB/s ≡ GDN's 610.6 GB/s. The "8× state-traffic" claim is a **per-step bytes-per-iter property** (kimi_linear's 4 MB state vs GDN-qwen3's 32 MB), NOT a bandwidth-vs-GDN claim. ADR-0006 framing validated.
- ✅ **Wave 22.9 SHIPPED 2026-05-21**: `cuda-attn-gdn-async/` using `cuda::pipeline` + `cuda::memcpy_async`. Bench result: **311.8 GB/s best — REGRESSION vs W1c's 417.7 GB/s** (-25%). cp.async overhead at TPB=16 hurts more than it helps. SASS confirms `LDGSTS.E.BYPASS.128 × 16` (vs W1c's 16 LDG.E.128). **Falsifies the "cuda::pipeline alone closes the gap" hypothesis** — TPB widening to 64+ with explicit warp roles is the dominant variable, validating Wave 22.8's hypothesis ranking. New W22.11 candidate added: cuda-attn-gdn-async-tpb128 (4-warp producer/consumer split).
- ✅ **W15.3 SHIPPED 2026-05-21**: `cutile-3dgs-real/` 4th frontend cell. Naive Approach A (per-pixel iter over all gaussians, no tile-binning, no early-termination). Correctness PASS: max u8 PPM diff = 1 vs cuda-3dgs-real cam A (well within ≤2 acceptance). 55.4 ms/cam median (~1.3× slower than cuda-3dgs-real's ~42 ms median per `cuda-3dgs-real/results.csv`, NOT 10× as initially mis-stated; corrected by Phase 7 reviewer). Single cuTile DSL feature gap: no `break` on tile-reduction within runtime `range()` for early-termination on transmittance < 1e-4. G5 (tile-binning) would be Approach B.
- 📋 **Wave 22.11 SHIPPED 2026-05-21**: `cuda-attn-gdn-async-tpb128/` (4-warp 1P+3C split + cuda::pipeline at TPB=128). **DEEPER REGRESSION: 245.3 GB/s best — WORSE than W22.9 (-21%) and W1c (-41%)**. SASS shows full cuTile-pattern signal (`SYNCS=106`, `BSSY=61`, `BSYNC=61`, `MBAR=18` — all the warp-specialized async-barrier instructions cuTile emits). **Two hypotheses now FALSIFIED**: (1) cuda::pipeline alone (W22.9), (2) TPB-widening + warp-specialization (W22.11). cuTile's 610 GB/s advantage is from something OTHER than the launch-geometry+SASS-pattern features identified in W22.8. New hypotheses to test: cuTile may use a different algorithmic decomposition (not 1:1 FFMA recurrence port), TENSORMAP descriptors without UTMALDG, or a substantially different smem layout strategy. See `cuda-attn-gdn-async-tpb128/ANALYSIS.md`.
- ✅ **Wave 22.5 SHIPPED 2026-05-21 (correctness only)**: `mojo-attn-bf16/` (3-kernel attention with bf16 matmul + softmax). max_abs_err = **0.0** BIT-EXACT vs CPU SDPA reference at small shape (B=1 n_h=4 S=128 qk=64 d_v=64). HMMA.16816.F32.BF16 = 32 in SASS on .target sm_120a. Expected ~20-25 TF at DeepSeek-V3 bench shape (3-kernel HBM round-trip ceiling, same as cuda-attn-mla 24 TF). Bench at full shape DEFERRED.
- ✅ **Phase 7 cross-family adversarial review SHIPPED 2026-05-21**: `docs/research/phase7-cross-family-review-2026-05-21.md`. Reviewer (GPT-5.5 via OpenRouter) caught 1 HIGH issue (3DGS gap was overstated as ~10× when actual is ~1.3×) + 3 MED issues (f16 tolerance unjustifiably loose, summary status drift, best/median reporting inconsistency). HIGH issue corrected in this commit. MED issues acknowledged for next loop.
- 📋 **Wave 22.10 candidate** (NEW priority — reframed by W22.11 result): `cuda-attn-gdn-tma/` — try explicit `cuTensorMapEncodeTiled` + `cp.async.bulk.tensor`. The W22.11 negative result rules out pure-launch-geometry as the answer; TMA descriptors are the next variable to test.
- 📋 **Wave 22.12 candidate**: investigate cuTile's actual algorithm via cubin disassembly + ptxas verbose log. The 610 GB/s number must come from somewhere — if not TMA + not warp-spec + not 4-warp launch, maybe it's a different decomposition (e.g. one CTA per (B, H) processing all of d_v at once, vs nvcc's per-(B, H, BV) tiling).
- ✅ **Wave 22.10 SHIPPED 2026-05-21** 🏆: `cuda-attn-gdn-tma/` using `cuTensorMapEncodeTiled` + `cp.async.bulk.tensor.2d`. **Bench result: 1032 GB/s best (887.6 mean) — BEATS cuTile (610) by +69%, beats W1c (417.7) by +148%, beats W22.9 (311.8) by +231%, hits 58% of HBM peak.** First nvcc kernel on this hardware/algorithm with `UTMALDG.2D × 2` in SASS (cuTile uses zero TMA per W22.8). Validates the W22.8 TMA hypothesis after all — TMA-with-simple-launch beats both W1c plain LDG.E.128 and cuTile's warp-spec-without-TMA pattern. See [`cuda-attn-gdn-tma/ANALYSIS.md`](cuda-attn-gdn-tma/ANALYSIS.md).
- ✅ **Wave 22.12 SHIPPED 2026-05-21**: `docs/research/wave22-12-cutile-launch-geometry.md`. Extracted cuTile cubin metadata: REQNTID=256 (8 warps), 99KB static smem, REG=255/thread (one CTA/SM ceiling), 3 named barriers vs nvcc W22.11's 128 threads + 70KB dyn smem + 40 regs/thread + 1 barrier. **Top hypothesis: only ~16 of W22.11's 128 threads do FFMA** (inherits W1c's BLOCK_V/4 lane structure). cuTile saturates 256 threads. W22.13 prescription: split outer-product across all consumer threads + opt into >49KB dyn smem for `outer_acc` tile + 3 named barriers.
- ✅ **Wave 22.4 SHIPPED 2026-05-21** ⚡: `mojo-matmul-fp8/` (e4m3 m16n8k32, hand-rolled). **`mma()` dispatcher already handles e4m3 natively** (no LLVM intrinsic, no inline asm needed). max_abs_err = **0.0 BIT-EXACT** at M=N=K=32. SASS shows 8× `QMMA.16832.F32.E4M3.E4M3` on `.target sm_120a`. **CORRECTION: Blackwell consumer emits `QMMA` (not `HMMA`) for sub-half inputs — pitfall added to `rust-gpu-compute` SKILL.** The "MOST RISKY" Wave 22 task was the cleanest path. Scaling to 4096³ unblocked.
- ✅ **G5 SHIPPED 2026-05-21** ⚡: `cutile-3dgs-real-binned/` tile-binned 3DGS rasterizer. **11.0× speedup** vs naive W15.3: 4.99 ms median vs 54.85 ms. **Now within ~7% of nvcc cuda-3dgs-real (5.4 ms)**. PPM diff vs cuda-3dgs-real cam A: max u8 diff = 1 (PASS, ≤2 acceptance). cuTile DSL fit confirmed: `for i in range(MAX)` with `mask = i < count_tile` is the canonical variable-length-loop pattern, MAX=4096 with sigma_k=4 the empirical sweet spot.
- 🚫 **Wave 22.1 BLOCKED 2026-05-21**: Mojo 1.0.0b1's `std.gpu.sync` exposes only mbarrier primitives — NO `cp_async_bulk` / TMA / `cuTensorMap` API path. Verified via 16 probe-compiles: `cp_async_bulk`, `tma_load`, `TensorMap`, etc. all return `does not contain`. Modules `std.gpu.tma`, `std.gpu.async_copy`, `std.gpu.bulk` ABSENT. Closest available: `mbarrier_arrive` (consumer side only). Defer until Modular ships TMA primitives. Workaround `inlined_assembly` PTX inline is high-risk (untested in 1.0.0b1, no `cuTensorMapEncodeTiled` host-side). See [`mojo-matmul-bf16-tma/BLOCKED.md`](mojo-matmul-bf16-tma/BLOCKED.md).

### Wave B (next-loop seeds) — 2026-05-21

- ✅ **Wave 22.13 SHIPPED 2026-05-21**: `cuda-attn-gdn-tma-warpspec/` combines W22.10's TMA path with W22.12's warp-spec prescription (TPB=128, 1 producer warp + 3 consumer warps, 3 named barriers, opted-into 99KB dyn smem, 4-stage pipeline). Bench result: **best 1032.0 GB/s = TIES W22.10**, mean 947.1 GB/s = **+6.7% over W22.10's 887.6 mean**. SASS shows BOTH UTMALDG.2D=2 (TMA preserved) AND BSSY/BSYNC=11/11 + SYNCS.PHASECHK=2 + SYNCS.EXCH=2 (warp-spec engaged). FFMA=120 (vs W22.10's 160). Correctness max_abs(o)=3.052e-05 bit-identical to W1c/W22.10. **Finding: TMA + warp-spec do NOT compose multiplicatively.** Once TMA loads saturate the HBM-issue path, warp-spec doesn't add throughput; its contribution is variance reduction (mean lift, not best lift). The W22.12 prescription helped at non-TMA W22.11 baseline (245→would need re-bench) but offers diminishing returns at the W22.10 1032 GB/s plateau. Hypothesis for future investigation: only at HIGHER input rates (multi-CTA TMA via `.cta_group::N` variants on sm_100a, OR pre-staged smem reuse across iterations) would warp-spec stack additively.
- ✅ **Wave 22.14 SHIPPED 2026-05-21** ⚡: `mojo-matmul-fp8/` scaled from M=N=K=32 (W22.4 correctness) to **M=N=K=4096 with full bench**. **TFLOPS_median = 113.4 (best 114.07)**, max_abs_err = **0.0 BIT-EXACT** vs CPU reference (1024 sampled cells × full K=4096 ref). 16× `QMMA.16832.F32.E4M3.E4M3` SASS on `.target sm_120a`. **+43% over Mojo bf16 (79.3 TF) and Mojo f16 (79.2 TF)** — real lift but NOT the optimistic 2× sketched, suggesting partial wins on both bandwidth (smem-load passes halved) AND compute (FLOPs/MMA doubled) compose to +43% rather than ×2. **First Mojo cell on this hardware to leapfrog cuTile bf16 (160 TF wins, but cuTile f16 172.5 still ahead).** 51.7% of cuBLAS hgemm. W22.4 single-block correctness preserved as `matmul_fp8_smoke.mojo`.
- ✅ **Wave 22.5b SHIPPED 2026-05-21** ⚡: scaled `mojo-attn-bf16/` from W22.5 correctness shape (B=1 n_h=4 S=128 qk=64 d_v=64, 0.0 bit-exact) to DeepSeek-V3 bench shape (B=1 n_h=128 S=2048 qk=192 d_v=128). **TFLOPS_median = 26.36 (best 26.93)**, max_abs_err = **0.0** BIT-EXACT (1024 samples vs CPU SDPA). HMMA.16816.F32.BF16 = 32 (16 in qkt + 16 in pv kernels). **+9% over cuda-attn-mla (24.17 TF), +7% over oxide-attn-mla (24.70 TF)** at the same shape — both fellow 3-kernel hand-rolled MMA implementations sharing the same HBM-roundtrip-between-stages structure. Mirrors the Wave 21 Mojo standalone matmul win. cublas-attn-mla (47 TF) still ahead via cuBLAS GEMMs; cuTile-MLA (112 TF) far ahead via fused FA-class kernel. **Pitfall corrected**: qk=192/BK=32 was claimed non-divisible in the task brief, but 32×6=192 — clean K-loop, no tail-handler needed.
- ✅ **Wave 22.15 SHIPPED 2026-05-21** 🏆: shape sweep of `cuda-attn-gdn-tma` vs cuTile across 5 shapes (tiny, small, qwen3_next_decode, large, wide). **The W22.10 +69% lift over cuTile GENERALIZES at medium-to-saturated grids (qwen3 1.63×, large 1.51×) but converges to 1.0× at launch-bound (tiny, n=4) and is structurally unsupported at D_K>256 (wide)**: `tiny` 1.00× (tie at launch-bound, n_blocks=4), `small` 2.0-2.4× variance band (n_blocks=16 — high inter-run noise per Phase 7c re-bench: 173-217 GB/s), `qwen3` 1.63× (1032 vs 634 GB/s, n_blocks=64), `large` 1.51× (1808.7 vs 1195.9 GB/s, n_blocks=1024 — saturated; refutes hypothesis that the lift would collapse at saturation). `wide` (D_K=512) BLOCKED on `cuTensorMapEncodeTiled` boxDim ≤ 256 hardware limit — next-loop seed: per-CTA two-tile assembly. **`large` shape best = 1808.7 GB/s = effective bytes/s of state traffic** (NOT direct DRAM bandwidth — without Nsight counters separating HBM vs L2 paths, the >1792-GB/s readings reflect L2 reuse across the 64 d_v-blocks per head fitting in 96 MB L2, not raw HBM throughput). **W22.15 MED issue (Phase 7c)**: small-shape ratio is variance-bounded; report range [2.0×, 2.4×] not point estimate. See [`cuda-attn-gdn-tma/SWEEP_W22_15.md`](cuda-attn-gdn-tma/SWEEP_W22_15.md) and [`docs/research/phase7c-cross-family-review-2026-05-21.md`](docs/research/phase7c-cross-family-review-2026-05-21.md).
- ✅ **Wave 23.1 SHIPPED 2026-05-21**: `mojo-3dgs/` — 5th frontend port of the 3DGS rasterizer (after cuda-3dgs-real, oxide-3dgs-real, cutile-3dgs-real, cutile-3dgs-real-binned). Naive per-pixel-iter-over-all-gaussians, 16×16 thread blocks → 50×50 grid for 800×800. **Correctness PASS first attempt** at utsuho_plush cam A: 99 of 640000 pixels differ by ≤1 u8, max diff = 1, mean = 0.0001, 0 pixels diff>2 (acceptance ≤2). Kernel time **38.5 ms/cam — slower than cuda/oxide/cutile-binned ~5-6 ms (no tile binning) but 30% faster than cutile-3dgs-real naive (54.85 ms)** at the same algorithmic complexity, attributed to Mojo restoring the `transmittance < 1e-4` early-out cuTile DSL had to drop. Full SH3 (degree-3, 16 coefs/channel, 45 f_rest fields) verified end-to-end. Pitfalls (Mojo 1.0.0b1): no String byte indexing → hard-coded PPM header as `UInt8` hex; file modes `r/w/rw/a` only (no `wb`/`rb`); `bitcast[DType.float32](u32)` works for LE blob decode; `fn` is deprecated emits warnings. Host-side PLY parse + 2D projection in Python (`prep.py`); pure-Mojo host preprocess extractable as W23.1c.
- ✅ **Wave 23.2 SHIPPED 2026-05-21**: `cuda-attn-mla-tma/` — applied W22.10's TMA recipe to the 3-kernel-decomposed MLA attention. **Correctness PASS at small shape** (B=1 n_h=4 S=128 qk=96 d_v=64): max_abs_err = 1.597e-4 (well within 1e-2 tol), both native (qk_eff=96) AND padded (qk_eff=128) paths pass. **SASS: UTMALDG.2D=4 + HMMA=20 + BSSY/BSYNC=13/13** — TMA loads of Q/K/V tiles, WMMA inner loops preserved, warp-spec engaged. **Zero LDG.E.128**: ALL gmem loads went through TMA. First cell on this hardware to combine TMA + WMMA in a 3-kernel MLA decomposition. Author + correctness only (no timed bench per task scope).

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
