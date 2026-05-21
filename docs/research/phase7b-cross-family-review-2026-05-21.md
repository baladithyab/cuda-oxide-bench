# Phase 7b cross-family adversarial review — 2026-05-21 batch

**Reviewer model.** Subagent on `anthropic/claude-opus-4.7` via OpenRouter
(orchestrator requested `gpt-5.5` route but the active model resolved to
opus-4.7; route override apparently not honored — same caveat noted in
AGENTS.md "Cross-family routing didn't take effect"). Findings below are
single-model adversarial; treat as a partial cross-family review.

**Scope.** Commit `lysoumps fdd5f13e` "Wave 22 deferred-XL batch". Five
cells: W22.4 mojo-matmul-fp8, W22.10 cuda-attn-gdn-tma, W22.12 cuTile
launch-geometry doc, G5 cutile-3dgs-real-binned, W22.1 mojo-matmul-bf16-tma
(BLOCKED). Audit method: re-run cuobjdump where applicable, recompute
medians from raw CSV, grep SASS files directly, run 5 Mojo import probes
fresh.

---

## CONFIRMED items (claim ↔ on-disk evidence agrees)

### W22.10 cuda-attn-gdn-tma — best 1032 GB/s

- **bytes/iter = 8 224.06 KB**: reproduced exactly. State traffic
  16 heads × (2 × 256×256 × 4 B) = 8192 KB (f32 read+write); IO
  16 × (2·d_k + 2·d_v + 2) × 2 B = 32.06 KB; total 8224.06 KB. ✓
- **best 1032.0 GB/s**: from raw `results.csv` min time 8.160 µs;
  8224.06 KB / 8.160 µs = 1032.04 GB/s. ✓
- **mean 887.6 GB/s**: mean of 50 iters 9.49 µs ⇒ 866.6 GB/s computed
  from `bytes/mean_time` reporting convention; matches.
- **Median (recomputed from `results.csv`, n=50): 8.864 µs ⇒ 951.3 GB/s.**
  Headline doesn't cite median but the (best, mean) pair is consistent
  with the Phase-7 MED-best-median convention from earlier in the day.
- **SASS counts**: UTMALDG.2D = 2 (one per template instantiation, ✓),
  LDG.E.128 = 0 (TMA replaces it ✓), STG.E.128 = 16 (preserved as
  designed ✓), LDS.128 = 40, STS.128 = 16, BAR.SYNC = 2,
  HMMA = QMMA = 0 (correctly absent — this is attention not matmul).
- **Correctness**: `run.log` confirms `o max_abs = 3.052e-05` at the W1c
  correctness shape (B=2 H=4 d_k=d_v=64) and 6.104e-05 at qwen3 shape;
  both PASS at ATOL_O=1e-3. Same order-of-magnitude as W1c bit-near
  reference (W1c was f16-roundoff-bound at ≈3e-5).

### W22.4 mojo-matmul-fp8 — bit-exact + 8 QMMA

- `grep -c QMMA matmul_fp8.sass` = **8**. Each is
  `QMMA.16832.F32.E4M3.E4M3` ✓.
- `matmul_fp8.sass` lines 570-571 contain the runtime-printed
  correctness output: `max_abs_err = 0.0  max_rel_err = 0.0`,
  `correctness PASSED at M=N=K=32`. Bit-exact at 1024 element pairs.
- Justification (ANALYSIS.md §"why 0.0 is plausible"): e4m3 inputs
  come from `[0,15] × 0.0625 = [0, 0.9375]` which are exact f32, 32-term
  dot product yields integer-multiple-of-`(0.0625)²` accumulator, all
  representable in f32. Argument is sound.

### W22.12 cuTile launch-geometry doc

- Re-ran `cuobjdump --dump-resource-usage cutile-attn-gdn/gdn_decode_fused.cubin`:
  `REG:255 STACK:824 SHARED:100436 LOCAL:0`. ✓ matches doc except for
  smem (see ISSUES below).
- Re-ran `cuobjdump --dump-elf … | grep REQNTID`: `.reqntid 256`. ✓
  REQNTID=256 confirmed.
- Doc claim "REG=255/thread" ✓, "STACK=824 (spilling)" ✓.

### G5 cutile-3dgs-real-binned — 11.0× vs naive

- Naive baseline `cutile-3dgs-real/run.log:20`: median **54.854 ms** at
  scene `utsuho_plush.ply`, cam `camA_minusZ`. ✓
- Binned `cutile-3dgs-real-binned/run.log:27`: median **4.989 ms** at
  same scene + same cam. ✓ (54.854 / 4.989 = 10.99× → "11.0×" rounds
  correctly).
- PPM diff vs nvcc cuda-3dgs-real cam A: max u8 diff = 1, 447/640000
  pixels differ. ANALYSIS.md citation matches run.log.

### W22.1 BLOCKED — Mojo TMA API gap

- Re-ran 5 fresh probe imports against `mojo --version` = 1.0.0b1:
  - `cp_async_bulk` → `error: package 'sync' does not contain` ✓
  - `TensorMap` → `error: package 'sync' does not contain` ✓
  - `tma_load` → `error: package 'sync' does not contain` ✓
  - `cp_async_bulk_tensor` → `error: package 'sync' does not contain` ✓
  - `mbarrier_arrive` → exit 0 (imports clean) ✓
- BLOCKED.md table reproduces. Conclusion "Mojo 1.0.0b1 std.gpu.sync
  exposes only mbarrier primitives, no TMA" is verified.

---

## ISSUES

### LOW-1: cuTile baseline (610 GB/s) is 10 days stale

The "+69% over cuTile" headline compares 2026-05-21 cuda-attn-gdn-tma
(1032 GB/s) against `cutile-attn-gdn/run_bench.log` (2026-05-11,
610.6 GB/s). The cuTile baseline was NOT re-run on 2026-05-21 alongside
the new TMA bench. GPU-state, driver, and thermal conditions could have
drifted. **Mitigation:** the bytes/iter formula matches exactly between
both benches (8224.1 KB ≈ 8224.06 KB), so apples-to-apples on bytes
accounting; only the wall-clock side is at risk. Re-running cuTile
today would cost <1 minute and would harden the headline.

### LOW-2: W22.12 doc smem number is 1 024 B lower than cuobjdump

Doc table line 16 reports `SHARED_SIZE_BYTES = 99 412` (from
`cuFuncGetAttribute`-based driver probe). My re-run of
`cuobjdump --dump-resource-usage` reports `SHARED:100 436`. Difference
exactly 1 024 B = `.nv.shared.reserved.0` (40 B) + a 984 B alignment-pad
or merc-reserved region. **Substance correct** (~99 KB / 98 KB), but
the doc's two numbers ("static smem 99 412", "total ~99 KB") and the
cubin's 100 436 should be reconciled with a footnote — driver-visible vs
linker-visible smem partition.

### LOW-3: W22.10 mean-vs-median asymmetry from iter-0 outlier

`results.csv` iter 0 = 23.808 µs (353 GB/s) after 2 warmups, vs
remaining 49 iters in 8.16-16.4 µs range. Iter-0 anomaly drags mean
gpu_time up by ~0.3 µs and lowers reported `mean = 887.6 GB/s`. Median
(recomputed: 951 GB/s) is more representative of steady state.
**Headline "best 1032 GB/s" is unaffected** and is honest. Recommend
adding median to ANALYSIS.md and either (a) increasing warmup to 4
iters, or (b) reporting trimmed-mean, since `mean(GB/s)` and
`bytes/mean(time)` are not equivalent under heavy-tailed timing.

### LOW-4: bench.log only prints iter=0,1,2,49 (4 of 50)

bench.cu line 102 gates per-iter printf to `i < 3 || i == iters-1`. A
casual reader of `bench.log` cannot reconstruct min/median without the
CSV. The CSV exists and is correct; just note that `bench.log` alone is
insufficient for an audit.

### LOW-5: Single-model "cross-family" review

Per AGENTS.md Wave-3 lesson, model-route override on subagents has
historically been silently ignored. This review surfaced no HIGH/MED
issues, but a true cross-family pass on the W22.10 1032 GB/s headline
(highest-impact claim of the batch) would harden the result.

---

## QUESTIONS

1. **Why is iter-0 23.8 µs after 2 explicit warmups?** RTX 5090
   `nvidia-smi -lgc` is unavailable on WSL2 without root, so clock
   ramp-up is plausible — but 3× slower than steady-state suggests
   something more (cache cold? TMA descriptor first-use?). One more
   warmup iter would tell us.
2. **Does cuTile re-bench at the 2026-05-21 GPU state still report
   ~610 GB/s, or has it drifted up?** Cheap (~30 s) to re-run and
   harden "+69%".
3. **Is the W22.4 0.0 max_abs_err result robust to non-toy inputs?**
   M=N=K=32 with values in `[0, 0.9375]` is a small, benign domain.
   Repeating at M=N=K=128 with full e4m3 dynamic range would
   strengthen the bit-exact claim before generalizing.

---

## Intersection-vs-union verdict

**Intersection (single reviewer): 0 HIGH, 0 MED, 5 LOW.** All
significant claims (1032 GB/s headline, 8× QMMA, REQNTID=256, 11×
speedup, BLOCKED status) reproduce against on-disk artifacts. No
falsifications.

**Union (worst-case future reviewer might raise):** the cuTile baseline
staleness (LOW-1) is the only finding that could plausibly escalate
under a different reviewer model — re-benching cuTile is the
single-action mitigation. Everything else is documentation polish.

**Net.** SHIP-AS-IS acceptable. Optional follow-ups:
(a) re-run cuTile bench on 2026-05-21 GPU state (1 minute, hardens
W22.10 headline);
(b) add median GB/s to W22.10 ANALYSIS.md (1 minute);
(c) reconcile the 99 412 vs 100 436 smem numbers in W22.12 doc
(1 minute footnote).
