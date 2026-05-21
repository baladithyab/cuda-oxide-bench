# Deep Work Loop Final Summary — 2026-05-21 (CONTINUED, post-deferred-XL batch)

**Loop scope:** Wave 17 (cross-frontend × cross-mechanism attention matrix) + Wave 22 (Mojo follow-on lanes + cuda-attn-gdn investigation chain) + Phase-7 follow-up MEDs + deferred-XL batch.
**Duration:** Single session, ~9 hours wall-clock from "keep going" to "loop done."
**Cells shipped:** 19 (15 kernel cells + 1 doc port + 4 docs/research + 2 cross-family reviews + 1 BLOCKED).
**Commits to master:** 7 (`ouvmkwzt 0fdcc177` → ... → `lysoumps fdd5f13e`, all pushed to origin).
**Backlog reduction:** from 18 active items → 0 ready-to-execute, 1 deferred (W22.13 = newly-added Wave 22.12 prescription), 2 BLOCKED on external resources, 1 BLOCKED on Mojo API gap.

## Cells shipped this loop (final count)

### Wave 17 — Cross-frontend × cross-mechanism attention matrix

| Cell | Status | Headline metric |
|---|---|---|
| W1a `cuda-attn-mla/` | ✅ | 24.17 useful_TF native, 21.32 padded; HMMA=20 |
| W1b `oxide-attn-mla/` | ✅ | 24.70 best TF @ deepseek_v3 |
| W1c `cuda-attn-gdn/` | ✅ | 417.7 GB/s best |
| W1d `oxide-attn-gdn/` | ✅ | correctness only initially → 276.1 GB/s in W22.6 |
| W1e `cutile-attn-kda/` | ✅ | 344.7 GB/s small shape; → 1170 GB/s saturation in W22.7 |
| W2c `cublas-attn-mla/` | ✅ | 47.1 useful_TF native |
| W2d `cutile-attn-gqa` BLOCK_M sweep | ✅ | REFUTED plan claim |
| U1+U2 `docs/upstream-issues-oxide/` | ✅ | Drafts ready |

### Wave 22 — Mojo follow-on + investigation chain

| Cell | Status | Headline metric |
|---|---|---|
| W22.1 `mojo-matmul-bf16-tma/` | 🚫 BLOCKED | Mojo 1.0.0b1 lacks `cp_async_bulk` API |
| W22.2 `mojo-matmul-f16/` | ✅ | 79.22 TF median (79.68 best), atol=1e-2+rtol=1e-3 (Phase-7 tightened) |
| W22.3 `mojo-matmul-bf16-padded/` | ✅ (correctness) | refutes ldmatrix.x4 hypothesis |
| **W22.4 `mojo-matmul-fp8/`** | ✅ ⚡ | **0.0 BIT-EXACT, 8× QMMA.16832.F32.E4M3.E4M3** |
| W22.5 `mojo-attn-bf16/` | ✅ (correctness) | 0.0 BIT-EXACT vs CPU SDPA |
| W22.6 `oxide-attn-gdn` `--bench` | ✅ | 276.1 GB/s |
| W22.7 `cutile-attn-kda` saturation | ✅ | **1170 GB/s = 65% HBM peak** |
| W22.8 W17 W1c TMA investigation | ✅ | hypothesis REJECTED |
| W22.9 `cuda-attn-gdn-async/` | ✅ | 311.8 GB/s — REGRESSION (-25%) |
| **W22.10 `cuda-attn-gdn-tma/`** | ✅ 🏆 | **1032 GB/s — BEATS cuTile by +69%** |
| W22.11 `cuda-attn-gdn-async-tpb128/` | ✅ | 245.3 GB/s — DEEPER REGRESSION |
| W22.12 cuTile launch geometry doc | ✅ | hypothesis: only 16/128 W22.11 threads do FFMA |

### Other

| Cell | Status | Headline metric |
|---|---|---|
| G4 SH3 in 3DGS | ✅ | oxide+cuda 99.99985% identical |
| W15.2 per-N ratio doc | ✅ | non-monotonic, sweet spot N=2048 |
| W15.3 `cutile-3dgs-real/` | ✅ | naive port, 55.4 ms/cam (1.3× nvcc) |
| **G5 `cutile-3dgs-real-binned/`** | ✅ ⚡ | **11.0× speedup, 4.99 ms (within 7% of nvcc)** |
| Phase 7 cross-family review | ✅ | 1 HIGH + 3 MED issues caught |
| Phase 7b cross-family review | ✅ | 0 HIGH + 0 MED + 5 LOW (all polish) |

## The Wave 22 investigation arc — narrative

The story of cuTile's 610 GB/s GDN advantage took FOUR waves to unravel:

| Wave | Hypothesis | Result | Conclusion |
|---|---|---|---|
| W22.8 | "cuTile uses UTMALDG TMA" | ❌ REJECTED — both nvcc and cuTile have ZERO UTMALDG | The TMA hypothesis was wrong AT THE TIME OF cuTile's design |
| W22.9 | "cuda::pipeline at TPB=16 closes the gap" | ❌ FALSIFIED (-25%, 312 GB/s) | cp.async overhead at small TPB hurts |
| W22.11 | "TPB=128 + 4-warp warp-spec closes the gap" | ❌ DEEPER FALSIFICATION (-41%, 245 GB/s) | SASS pattern matching ≠ perf parity |
| **W22.10** | **"explicit cuTensorMapEncodeTiled + cp.async.bulk.tensor.2d closes the gap"** | ✅ **OVERSHOT — 1032 GB/s, +69% over cuTile** | **TMA-with-simple-launch BEATS cuTile's warp-spec-without-TMA** |

The final picture: cuTile gets to 610 GB/s by warp-specializing 256 threads with 99 KB smem and full-register pressure (REG=255), exploiting math-density + smem-cooperation. nvcc's W22.10 path goes the *other* direction — keep TPB=16, lean on Blackwell's TMA hardware (UTMALDG.2D) for state-tile loads, do the FFMA recurrence in 16 threads. **Both work; on this shape, TMA wins.**

This also closes a generalization for the `rust-gpu-compute` skill: **emitting "the right SASS" doesn't guarantee perf parity** — but **emitting fundamentally different (better) SASS does**. W22.10 emits SASS no other cell in this repo emits (`UTMALDG.2D`), and that's what wins.

## Cross-frontend updated tables

### GDN @ B=1 H=16 d_k=d_v=256 (Qwen3-Next decode)

| Frontend / Variant | GB/s (best) | GB/s (median) | % HBM | vs W1c |
|---|---:|---:|---:|---:|
| **W22.10 cuda-attn-gdn-tma** | **1032.0** | **951.3** | 58% | +148% |
| cuTile fused (W16, re-benched today) | 610.6 | 566.6 | 34% | +46% |
| cuda-attn-gdn (W1c) | 417.7 | 297.8 | 23% | — |
| cuda-attn-gdn-async (W22.9) | 311.8 | 291.8 | 17% | -25% |
| oxide-attn-gdn (W22.6) | 276.1 | 264.8 | 15% | -34% |
| cuda-attn-gdn-async-tpb128 (W22.11) | 245.3 | 238.1 | 14% | -41% |

cuTile re-bench 2026-05-21 confirms the W16 baseline (Phase 7b LOW-1 follow-up).

### Matmul @ M=N=K=4096 (post-tightened-tolerance)

| Frontend | TFLOPS (best) | TFLOPS (median) | Precision |
|---|---:|---:|---|
| cuBLAS bgemm | 219.3 | — | bf16→f32 |
| cuTile mma_f16 | 172.5 | — | f16→f32 |
| cuTile mma_bf16 | 159.95 | — | bf16→f32 |
| Mojo W22.2 f16 | 79.68 | 79.22 | f16→f32 |
| Mojo W21 bf16 | 79.82 | 79.26 | bf16→f32 |
| Mojo W19 TF32 | 55.5 | — | TF32→f32 |

### 3DGS rasterizer @ utsuho_plush 800×800 cam A

| Frontend | ms/cam (median) |
|---|---:|
| nvcc cuda-3dgs-real | ~5.4 |
| **G5 cuTile-3dgs-real-binned** | **4.99** ⚡ |
| oxide-3dgs-real | ~6 |
| cuTile-3dgs-real (W15.3 naive) | 55.4 |

G5 binned cuTile is now within ~7% of nvcc. The 11× speedup over the naive cuTile port came from CPU-side tile-binning + cuTile's `for i in range(MAX) + mask = i < count_tile` pattern + sigma_k=4 gaussian extent.

## Skills + memory updated

- `mlops/rust-gpu-compute` SKILL.md (4 patches over the loop):
  - `cargo oxide -- --bench` arg-parsing trap, BENCH=1 env workaround
  - "SASS pattern matching ≠ perf parity" (W22.9 + W22.11 falsifications)
  - **NEW pitfall #20: Blackwell consumer emits `QMMA` (not `HMMA`) for sub-half inputs** (FP8/INT8). Grep `QMMA.16832.F32.E4M3.E4M3`, NOT `HMMA.*.E4M3`. Confirmed via PTXAS reverse-engineering reference.
  - Mojo m16n8k32 e4m3 row updated in `references/mojo-mma-shapes.md` (Wave 22.4)
- `software-development/deep-work-loop`:
  - new ref [`parallel-subagent-pitfalls-2026-05.md`](file:///home/codeseys/.hermes/skills/software-development/deep-work-loop/references/parallel-subagent-pitfalls-2026-05.md)

## Final loop verdict

✅ **Backlog ZERO ready-to-execute items remaining.** All XL items either shipped (W22.4, W22.10, W22.12, G5) or BLOCKED on external resources (W22.1 Mojo TMA API, W15.1 user creds, W15.4 upstream cuda-oxide release).

✅ **Cross-family Phase 7 + 7b reviews caught issues.** Phase 7 caught 1 HIGH (3DGS ratio) + 3 MED. Phase 7b caught 0 HIGH + 0 MED + 5 LOW. Both reviews accepted with corrections applied.

✅ **3 hypothesis chain results**:
- W22.8 hypothesis (UTMALDG-as-cuTile-mechanism) REJECTED on cuTile's own SASS evidence.
- W22.9 + W22.11 hypotheses (cuda::pipeline alone, then TPB-widening + warp-spec) FALSIFIED.
- W22.10 hypothesis (explicit TMA descriptors) **VALIDATED + OVERSHOT** — beats cuTile by 69%.

✅ **7 commits to master, all pushed** to `baladithyab/cuda-exploration` at `lysoumps fdd5f13e` + the cuTile re-bench addendum.

✅ **The cuda-exploration repo now has a kernel that beats every frontend baseline on its kernel/shape, ON consumer Blackwell, with reproducible SASS evidence (UTMALDG.2D × 2).**

The deep-work-loop closes cleanly. Next loop's seed item: W22.13 — apply W22.12's prescription (split outer-product across all 128 consumer threads + opt into >49KB dyn smem + 3 named barriers) ON TOP OF W22.10's TMA path. Headroom: 1032 → ~1500+ GB/s if the warp-spec + TMA stack adds rather than overlaps.
