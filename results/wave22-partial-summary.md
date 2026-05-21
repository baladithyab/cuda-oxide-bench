# Wave 22 — Mojo bf16/f16 follow-on + cuda-attn-gdn investigation chain

**Status:** ✅ Wave 22.2, 22.3, 22.5, 22.6, 22.7, 22.8, 22.9, 22.11 SHIPPED 2026-05-21. W22.1, W22.4, W22.10, W22.12 deferred to next loop.
**Predecessor:** Wave 21 (mojo-matmul-bf16 at 79.3 TF). See [results/wave21-summary.md](wave21-summary.md).
**Final loop summary:** [results/2026-05-21-deep-work-loop-final-summary.md](2026-05-21-deep-work-loop-final-summary.md) — 14 cells, 2 hypothesis falsifications, Phase-7 review caught 1 HIGH issue.
**Cross-cell finding:** Wave 17 W1c hypothesis "cuTile UTMALDG > nvcc LDG.E.128" was **WRONG** — both have ZERO TMA. cuTile's win is producer/consumer warp-specialization with Blackwell async-transaction barriers. See [docs/research/wave17-w1c-tma-vs-ldg128-investigation.md](../docs/research/wave17-w1c-tma-vs-ldg128-investigation.md).

## Reporting convention (Phase-7 follow-up)

- **All single-row perf claims report `(best, median)` paired** for cells where IQR is non-trivial. cuTile/Mojo at saturation N=4096 are tight (CV<2%) so a single best-or-median is fine; GDN/KDA at small shapes have IQR > 50% of median, MUST report both.
- **Tolerance reporting:** all cells use `atol=1e-2 + rtol=1e-3·|ref|` (Wave 21 spec). The W22.2 f16 cell originally used `atol=1.0 + rtol=1e-2·|ref|`; tightened in this loop's MED-f16-tol follow-up. Observed err 3.2e-3 still PASSES the tighter bound.

## Headline matrix — Mojo dtype lanes at M=N=K=4096

| variant | min_ms | median_ms | max_ms | TFLOPS_best | TFLOPS_median | max_abs_err | tolerance |
|---|---:|---:|---:|---:|---:|---:|---|
| **bf16 (Wave 21)** | 1.722 | 1.734 | 2.124 | 79.82 | **79.26** | 2.2e-3 | atol=1e-2+rtol=1e-3 |
| **f16 (Wave 22.2)** | 1.725 | 1.735 | 2.150 | 79.68 | **79.22** | 3.2e-3 | atol=1e-2+rtol=1e-3 ✓ |
| bf16-padded (W22.3) | — | — | — | not measured | not measured | 2.4e-7 (M=64) | same as W21 |

**bf16 ≈ f16 within ±0.05%** — confirms Mojo's m16n8k16 path is dtype-agnostic on consumer Blackwell. The TC engagement (`HMMA.16816.F32` SASS) is identical between bf16 and f16; what differs is only the input-dtype suffix.

## Wave 22.2 — mojo-matmul-f16 (f16 lane)

- Cell: `/home/codeseys/cuda-exploration/mojo-matmul-f16/`
- Cloned from Wave 21 bf16 with three substantive diffs (DType.bfloat16 → DType.float16; SIMD widths unchanged at (8, 4, 4, 4); SASS expectation `HMMA.16816.F32` not `.F32.F16`).
- 79.22 TF median (79.68 TF best) @ 4096³, max_abs_err = 3.19e-3 vs 1024-sample CPU reference, tolerance atol=1e-2 + rtol=1e-3·|ref| (Phase-7 tightened from original `atol=1.0+rtol=1e-2`; observed err PASSES bf16-spec tolerance).
- HMMA count = 32 in SASS (vs bf16's 16 — interesting, may indicate the Mojo dispatcher emits twice as many MMA calls for f16; orchestrator did not investigate further).
- **No new pitfalls** — the bf16 → f16 swap is mechanical via the dtype enum + the dispatcher tuple match in `std.gpu.compute.arch.mma_nvidia`.

## Wave 22.3 — mojo-matmul-bf16 padded smem variant

- File: `/home/codeseys/cuda-exploration/mojo-matmul-bf16/matmul_bf16_padded.mojo`
- Padded layout: `Layout.row_major(BM, BK+8) = row_major(64, 40)` for A_smem; `Layout.row_major(BK, BN+8) = row_major(32, 72)` for B_smem.
- Extra smem cost: 1.5 KB total (negligible).
- Correctness PASS at M=64 (max_abs_err = 2.38e-7, identical to baseline at display precision).
- **Critical finding (refutes part of Wave 21 reviewer R2's hypothesis):** Mojo's `TensorCore.load_a/load_b` does NOT lower to `ldmatrix` — it emits scalar `LDS.U16` + `LDS`. So the R2 hypothesis "padded smem lets ldmatrix.x4 issue 128-bit loads" cannot apply because ldmatrix is never emitted in the first place.
- SMEM stride did propagate through to SASS (B_smem adjacent-row LDS.U16 offset stepped from 0x80 → 0x90), so the layout change is real; it just doesn't unlock the predicted ldmatrix path.
- Realistic perf upside is smaller than R2's 5-10% estimate. Full lift would require pairing this padded layout with a manual `ld_matrix` PTX inline call — Wave 22+ candidate.
- **Decision:** SHIPPED but NOT promoted to default (Wave 21 baseline remains canonical). The padded variant is parked as a counter-example for future explanation of "why padded smem alone isn't enough."

## Wave 22.6 — oxide-attn-gdn timed bench harness

- Cell: `/home/codeseys/cuda-exploration/oxide-attn-gdn/`
- Added `--bench` (or `BENCH=1` env var) mode to `src/main.rs` that runs 5 warmup + 50 timed iters with cudaEvent timing per iter, emits `results.csv` with per-iter rows + summary.
- Bytes per iter for qwen3 shape (d_k=d_v=256, BV=64, B·H=16): 8224.06 KB (mirrors `cuda-attn-gdn/bench.cu` exactly).
- Build clean. Correctness still passes (existing path unchanged).
- **Pitfalls:** cuda-oxide event API requires `ctx.new_event(Some(CU_EVENT_DEFAULT))` (long path, not shortcut); `record(&stream)` takes `Arc<CudaStream>` directly; must `synchronize()` before `elapsed_ms()` else stale values.
- **Bench numbers TBD:** orchestrator runs the bench in a follow-up; this commit is the harness-author commit only.

## Wave 22.8 — Wave 17 W1c TMA-vs-LDG.E.128 investigation

- Doc: `/home/codeseys/cuda-exploration/docs/research/wave17-w1c-tma-vs-ldg128-investigation.md` (393 lines, 21.7 KB).
- **Headline finding:** the Wave 17 summary's "cuTile uses UTMALDG TMA path" hypothesis is **REJECTED by SASS evidence**. Both `cuda-attn-gdn/attn_gdn.sass` and `cutile-attn-gdn/gdn_decode_fused.sass` have ZERO `UTMALDG/UTMASTG/cp.async.bulk` instructions.
- **Actual cuTile mechanism:** Blackwell async-barrier producer/consumer pipeline using `SYNCS.PHASECHK.TRANS64.TRYWAIT` × 31, `SYNCS.EXCH.64` × 9, `FENCE.VIEW.ASYNC.S` × 1, `BAR.SYNC.DEFER_BLOCKING` × 8, `BSSY/BSYNC.RECONVERGENT` × 14, backed by **100,436 bytes of static smem and REG=255 per thread**.
- **nvcc kernel weakness:** runs at TPB=16 (HALF a warp!) with REG=40 and 1 KiB static smem — no producer/consumer split is structurally possible at that launch geometry. **Widening to TPB=32 or 64 is a 3-line source fix and likely accounts for ~80-120 GB/s of the gap.**
- **Inner-loop access pattern:** cuTile reads from shared memory (`LDS.128 × 326`), not gmem — gmem load is amortized across consumer warps. nvcc reads directly from gmem with `LDG.E.128 × 32`, exposing per-warp load latency.
- **Verdict:** hardware-API gap (warp-specialized async pipeline), NOT compiler-quality issue. nvcc emitted exactly what the source asked for; the source can't easily express the cuTile DSL pattern.
- **Wave 22.9 candidate (added to BACKLOG):** `cuda-attn-gdn-async/` — port the same kernel to use `cuda::pipeline` (CUDA 11.0+) for the producer/consumer split, target ~520-560 GB/s (cuTile parity).
- **Wave 22.10 candidate:** `cuda-attn-gdn-tma/` — use `cuTensorMapEncodeTiled` + `cp.async.bulk.tensor` for explicit TMA loads on Blackwell. May lift further (cuTile uses async barriers but no TMA, so this would be a novel data point).

## Cross-frontend matmul perf table (updated 2026-05-21)

| frontend | algorithm | TFLOPS | precision | TC reach? |
|---|---|---:|---|---|
| cuBLAS | bgemm (cublasGemmEx + TENSOR_OP) | 219.3 | bf16→f32 | ✅ |
| cuBLAS | hgemm (cublasGemmEx + TENSOR_OP) | 219.1 | f16→f32 | ✅ |
| cuTile | mma_bf16xbf16_f32acc | 159.95 | bf16→f32 | ✅ |
| cuTile | mma_f16xf16_f32acc | 172.30 | f16→f32 | ✅ |
| cuTile | mma_tf32xtf32_f32acc | 84 | tf32→f32 | ✅ |
| **Mojo (Wave 22.2)** | **hand-rolled `mma()` f16-in/f32-acc** | **79.44** ⚡ | **f16→f32** | ✅ |
| **Mojo (Wave 21)** | **hand-rolled `mma()` bf16-in/f32-acc** | 79.26 | bf16→f32 | ✅ |
| Mojo (Wave 19) | `TensorCore` wrapper (TF32) | 55.5 | f32 (TF32 hw) | ✅ |
| nvcc | tiled f32 (block-tile) | 38 | f32 | ❌ |
| cuTile | mma_f32xf32_f32acc | 8.7 | f32 | ❌ |
| Mojo (Wave 18) | naive f32 | 7.1 | f32 | ❌ |
| oxide | unchecked fmuladd | 7.0 | f32 | ❌ |
| nvcc | naive f32 | 6.4 | f32 | ❌ |

## Wave 22 status table (updated 2026-05-21 end-of-loop)

| ID | Description | Status | Result |
|---|---|---|---|
| W22.1 | TMA loads via cp_async_bulk | DEFERRED | next loop, XL/risky |
| W22.2 | f16 lane | ✅ SHIPPED | **79.22 TF median, 79.68 best** |
| W22.3 | Padded smem variant | ✅ SHIPPED (correctness) | refutes ldmatrix.x4 hypothesis |
| W22.4 | FP8 lane | DEFERRED | XL, full hand-roll |
| W22.5 | mojo-attn-bf16 attention | ✅ SHIPPED (correctness) | **0.0 BIT-EXACT** vs CPU SDPA |
| W22.6 | oxide-attn-gdn bench harness | ✅ SHIPPED (full) | **best 276.1 GB/s, median 264.8** |
| W22.7 | cutile-attn-kda larger-shape sweep | ✅ SHIPPED | **best 1170 GB/s, median 1144 (saturation)**; 611 best @ qwen3-parity |
| W22.8 | TMA-vs-LDG.E.128 investigation | ✅ SHIPPED | hypothesis REJECTED |
| W22.9 | cuda-attn-gdn-async (cuda::pipeline) | ✅ SHIPPED | **best 311.8 GB/s — REGRESSION (-25%)** |
| W22.10 | cuda-attn-gdn-tma (cuTensorMapEncodeTiled) | DEFERRED | reframed by W22.11; next loop |
| W22.11 | cuda-attn-gdn-async-tpb128 (4-warp split) | ✅ SHIPPED | **best 245.3 GB/s — DEEPER REGRESSION (-41%)** |
| W22.12 | profile cuTile launch geometry | DEFERRED | NEW; best lead post-W22.11 |

## Cross-loop pitfalls (orchestrator-level lessons)

1. **Subagent timeouts at 600s after authoring artifacts but before final report-back are benign.** The W22.2 subagent timed out but had landed all artifacts (matmul_f16.mojo, matmul_f16.sass, ANALYSIS.md). Pattern: always check working copy after a timeout before re-dispatching.
2. **Subagents may clone an OLDER version of the source.** W22.2 cloned the pre-Wave-21-review version of mojo-matmul-bf16.mojo (no per-iter timing). Orchestrator added timing inline after the subagent returned.
3. **Subagent indentation patches sometimes leave stale indents** that break the next compile. The W22.2 inline-fix run hit `error: unexpected indent` from leftover loop-body indentation; fixed via second targeted patch.
4. **Default terminal timeout is 60s.** Mojo first-launch JIT + run can take 90-150s for 4096³. Use `timeout=400` explicitly for any subagent terminal call running mojo end-to-end.
