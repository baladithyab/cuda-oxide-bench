# Phase 7c cross-family adversarial review — Wave B (2026-05-21)

**Reviewer model.** Hermes subagent on `openai/gpt-5.5` via OpenRouter. Scope:
commit `lkowzpym b5a7c1d6` (`b5a7c1d`), W22.13/14/5b/15. Audited on-disk
logs/CSV/SASS and fresh-ran W22.13 `./bench`, W22.14 `./run.sh`, W22.5b
`./run.sh`, W22.15 `./bench_sweep`. No source edits.

## CONFIRMED

### W22.13 `cuda-attn-gdn-tma-warpspec`

- Fresh `./bench`: best **1028.0 GB/s** (8.192 us), i.e. ties W22.10's 1032
  within timing quantization. Current 50-row CSV: best **1028.008**, mean of
  per-iter GB/s **888.34**, median **943.26**, median time **8.928 us**.
- Fresh SASS on `attn_gdn_tma_warpspec`: **UTMALDG.2D=2**, **BSSY=11**,
  **BSYNC=11**, **SYNCS=8**. On-disk SASS matches. Audit trap: dumping the
  `bench` executable itself gives zero counts; the signal is in the correctness
  binary.
- Best-tie claim is confirmed. The "6.7% mean lift" is more fragile because the
  fresh mean moved with outliers; report exact CSV convention if keeping it.

### W22.14 `mojo-matmul-fp8`

- Existing and fresh `matmul_fp8.sass`: exact
  `QMMA.16832.F32.E4M3.E4M3` count = **16**.
- Fresh run at M=N=K=4096: min/median/max **1.2057/1.2083/1.7736 ms**,
  **TFLOPS_median=113.7438**, best 113.9915; correctness `max_abs_err=0.0`,
  `max_rel_err=0.0`, PASSED. The shipped **113.4 TF** is confirmed.
- Sanity check: not bandwidth-bound. A+B FP8 inputs are ~33.6 MB, so HBM-only
  read floor at 1792 GB/s is ~18.7 us, far below observed 1.21 ms. GEMM reuse
  makes the prompt's `1792/(2 byte/elem)` framing an invalid roofline; 113 TF is
  a plausible compute-bound hand-QMMA result, well below peak tensor-core rates.

### W22.5b `mojo-attn-bf16`

- Fresh `run.log`: DeepSeek-V3 shape B=1,H=128,S=2048,qk=192,d_v=128;
  **max_abs_err=0.0**, PASSED against 1024 CPU SDPA samples.
- SASS HMMA count = **32** (`HMMA.16816`): 16 qkt + 16 pv.
- Fresh median **25.964 TF** (best 26.932), vs shipped log **26.360 TF**. The
  win over `cuda-attn-mla` holds: baseline `results.csv` native best useful
  **24.168 TF**, median **23.861 TF**; `bench.log` summary says best **24.17 TF**.
  Exact 26.36 is not stable, but direction is confirmed.

### W22.15 `cuda-attn-gdn-tma` sweep

- Fresh `./bench_sweep` confirms the critical saturation result: **large best
  1815.748 GB/s** at 74.208 us (shipped 1808.7 at 74.50 us); median large
  **1682.27 GB/s**, mean **1687.89**.
- CSV math is correct: `1815.748 GB/s * 0.074208 ms / 1000 = 0.134743 GB`, i.e.
  131585 KiB/iter. This is not a formula artifact.
- Plausibility: result is slightly above nominal 1792 GB/s HBM, but compatible
  with effective state-traffic bandwidth plus L2 help. For B=4,H=64,d=256,
  S_in or S_out is 64 MiB; per CTA 64 KiB; a CTA wave touches ~16 MiB, fitting
  in a 96 MiB L2. Do not call it direct measured DRAM throughput without counters.
- Fresh other bests: tiny **32.252** (shipped 32.5, OK), qwen3 **1032.039**
  (OK), small **173.816** (did **not** reproduce shipped 217.3).
- cuTile best baselines from `cutile-attn-gdn/sweep_results`: tiny 32.252,
  small 90.232, qwen3 634.145, large 1195.888, wide 557.361 GB/s. Fresh TMA
  ratios: tiny 1.00x, small 1.93x, qwen3 1.63x, large 1.52x; wide is TMA-NA.

## ISSUES

- **MED:** W22.15 generalization is overclaimed. The 1.8 TB/s large result is
  real, but "generalizes 1.5-2.4x across grids" fails for tiny (tie), wide
  (unsupported), and fresh small (173.8 not 217.3; 1.93x not 2.4x).
- **LOW:** 1808.7/1815.7 GB/s should be described as **effective bytes/s** from
  kernel traffic accounting, not direct DRAM bandwidth, unless Nsight counters
  separate HBM vs L2.
- **LOW:** W22.13 mean-lift and W22.5b exact 26.36 TF median drift on rerun;
  prefer pinned best+median tables over prose percentages mixing conventions.
- **LOW:** W22.14's 0.0 error is credible for the current deterministic input
  pattern, but not a broad FP8 accuracy claim without full-range/random e4m3.

## QUESTIONS

1. Should W22.15 small be re-run with more warmup/locked clocks to explain the
   217.3 -> 173.8 drift?
2. Can Nsight Compute DRAM/L2 counters validate the saturation explanation?
3. Can W22.14 add a second, full-dynamic-range e4m3 correctness distribution?

## Intersection-vs-union verdict

**Ship core wins with hedges.** No HIGH falsification of the two critical claims:
W22.14's **113.4 TF** FP8 result is confirmed, and W22.15's **1808.7 GB/s**
saturation is confirmed/plausible as effective bandwidth. Union risk is one MED
issue: W22.15's shape-sweep generalization needs narrower wording.
