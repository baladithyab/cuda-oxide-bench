# Phase 7 cross-family adversarial review — Wave 17 + Wave 22

**Reviewer model:** OpenAI GPT-5.5 via OpenRouter (Hermes Agent).  
**Scope:** audited on-disk Wave 17 and Wave 22 artifacts in `/home/codeseys/cuda-exploration` at git `6a75f0c` (working tree also had unrelated untracked `cuda-attn-gdn-async-tpb128/`; not inspected). No source files modified.

## CONFIRMED

- **W1c SASS vector-load claim is real.** `cuda-attn-gdn/attn_gdn.sass` contains `LDG.E.128=16`, `STG.E.128=16`, `HMMA=0`, `MUFU=0`, matching Wave 17's table. The Wave 22.8 correction is also supported: both `cuda-attn-gdn/attn_gdn.sass` and `cutile-attn-gdn/gdn_decode_fused.sass` have `UTMALDG=0`, `UTMASTG=0`, `LDGSTS=0`; cuTile instead shows `SYNCS.PHASECHK.TRANS64=31`, `FENCE.VIEW.ASYNC.S=1`, `LDS.128=326`.
- **GDN timing headlines reproduce from CSV.** `cuda-attn-gdn/results.csv`: best `0.020160 ms` = **417.73 GB/s**, median **297.81 GB/s**. `cuda-attn-gdn-async/results.csv`: best `0.027008 ms` = **311.81 GB/s**, median **291.76 GB/s**. `oxide-attn-gdn/results.csv`: best **276.149 GB/s**, median **264.759 GB/s**. These support the ranking cuTile (610 GB/s, prior cell) > nvcc W1c > async > oxide.
- **MLA Wave 17 numbers mostly reproduce.** `cuda-attn-mla/results.csv` best native useful TF = **24.168**, padded = **21.324**; `oxide-attn-mla/results.csv` best = **24.705 TF**. SASS count claims for W1c are plausible and directly checked.
- **Mojo f16 W22.2 performance reproduces from run.log.** `mojo-matmul-f16/run.log` reports `median_ms=1.73008`, **79.44 TF median**, `max_abs_err=0.003189`. SASS has bare `HMMA.16816.F32=32` and no `.BF16`, consistent with f16 being the implicit suffix. Current `mojo-matmul-bf16/run.log` reports **79.85 TF**, close enough to the summary's older 79.26/79.3 headline for dtype-parity conclusion.
- **KDA large-shape follow-on is supported, but by separate files.** `cutile-attn-kda/results_large.csv`: best **1170.128 GB/s**, median **1144.082 GB/s**. `results_qwen3_next_gdn_parity.csv`: best **611.195 GB/s**, median **532.172 GB/s**. This validates BACKLOG's 1170/611 claims; the older `results.csv` alone only contains W1e small-shape data.
- **W22.9 regression interpretation is broadly correct.** The async kernel passes correctness with the same residuals as W1c and replaces `LDG.E.128=16` with `LDGSTS.E.BYPASS.128=16`; no SASS evidence suggests a functional bug hiding true performance. The regression is credible as overhead/geometry: same TPB=16, extra smem/barrier traffic, and no `SYNCS`/mbarrier warp specialization.

## ISSUES

- **HIGH — cutile-3dgs-real vs nvcc gap is overstated/misstated in BACKLOG.** BACKLOG says `55.4 ms/cam median (~10× slower than nvcc, expected)`. On-disk `cuda-3dgs-real/results.csv` medians are ~41–44 ms (cam A median 43.42 in CSV; run.log older cam A 41.95), so the direct on-disk ratio is only **~1.25–1.35×**, not 10× or nvcc ~6 ms. The 3DGS nvcc ANALYSIS also describes 36–42 ms medians, not 6 ms. If a ~6 ms optimized baseline exists, it is not in `cuda-3dgs-real/` evidence and should not be used as an on-disk claim.
- **MED — f16 correctness tolerance is unjustifiably loose relative to evidence.** W22.2 documents `atol=1.0 + rtol=1e-2` because f16 has a narrower exponent than bf16. That rationale is backwards for this input distribution: f16 has more mantissa precision than bf16, and the observed 4096³ max error (**3.2e-3**) is well inside the Wave 21 bf16 tolerance (`1e-2 + 1e-3*|ref|`). The loose gate did not mask failure here, but future f16 cells should use the tighter bf16-style tolerance unless input range/overflow evidence proves otherwise.
- **MED — summary status drift in `results/wave22-partial-summary.md`.** It still says W22.6/W22.7 are deferred/TBD in several places, while BACKLOG and per-cell artifacts show full W22.6 bench results and W22.7 large-shape results. A reader using only the summary would miss shipped data.
- **LOW — Wave 17 summary uses best for headline and mean for W1c, while reviewer recomputation shows median differs materially.** W1c best is 417.7 GB/s but median is 297.8 GB/s; W1e best is 344.7 but median 210.9. The docs do mention jitter/mean in places, but headline comparisons should label best-vs-median consistently.
- **LOW — Mojo bf16 log drift.** Current bf16 run.log says 79.85 TF, while Wave 22 table says 79.26. This is a small rerun drift, not a conclusion-changing issue, but the source of truth should be frozen or annotated.

## QUESTIONS FOR ORCHESTRATOR

1. Where did the **nvcc ~6 ms** comparator for cutile-3dgs-real come from? It is not supported by `cuda-3dgs-real/results.csv` or `run.log` in this checkout.
2. Should f16 matmul acceptance be tightened to the same `atol=1e-2 + rtol=1e-3` used for bf16, given the observed f16 error is only `3.2e-3`?
3. Should `results/wave22-partial-summary.md` be regenerated after BACKLOG's later W22.6/W22.7/W22.9/W15.3 updates, or is BACKLOG intended to be the authoritative post-summary ledger?
4. For GDN, do we want future tables to report **best and median** side-by-side? Variability is large enough that best-only can overstate stable throughput.

## Union vs intersection verdict

**Intersection (high-confidence blockers):** only one material correction blocks citeable claims: the cutile-3dgs-real performance ratio must be fixed or sourced. On-disk evidence supports ~55 ms vs ~42 ms, not 55 ms vs ~6 ms / 10×.

**Union (worth fixing before final writeup):** tighten f16 tolerance, refresh Wave 22 partial summary status, and standardize best/median reporting for noisy GDN/KDA cells.

**Overall verdict:** Wave 17 + Wave 22 core conclusions mostly survive adversarial audit: W1c vectorized SASS exists; the TMA hypothesis was correctly rejected; W22.9 is a genuine negative result rather than an obvious bug; Mojo f16/bf16 parity is real; KDA saturation data exists. Ship with the 3DGS ratio corrected and the tolerance/status caveats addressed.
