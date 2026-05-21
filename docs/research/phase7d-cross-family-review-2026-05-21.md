# Phase 7d Cross-Family Adversarial Review — Wave C (12 cells)

Date: 2026-05-21  
Reviewer/runtime: openai/gpt-5.5 via openrouter  
Repo: `/home/codeseys/cuda-exploration` @ `725d117e13e6b36f11926313adcebabd42972f4c`

## Method

Audited all Wave C cells at artifact level, with deep checks on the highest-risk claims: `mojo-attn-gqa` vs `cuda-attn-gqa`, `oxide-attn-gdn-tma` TMA reach, `cutile-attn-gdn-tma` byte-identical cubin falsification, `mojo-matmul-tiled` rerun, and WGPU/LLVMPIPE framing. Spot-checked C2.1/C2.2/C2.6 headline CSV/SASS claims.

## Confirmed

### C1.5 `mojo-attn-gqa` vs `cuda-attn-gqa`

Direct CSV recomputation confirms shape parity and the performance ratio:

- Mojo CSV: `(B=1, S=2048, Nq=32, Nkv=8, D=128)`, median **25.650171 TFLOPS**, best **28.973066 TFLOPS**.
- CUDA CSV: same shape, 10 timed rows, computed median **23.374294 TFLOPS**.
- Ratio: `25.650171 / 23.374294 = 1.0974x` = **9.74% faster**.

Verdict: “+10% / 1.10x over nvcc” is fair after rounding. The `BIT-EXACT` wording is supported by the cell’s sampled CPU-reference check (`max_abs_err=0.0`), but should ideally be phrased as sampled bit-exact rather than exhaustive full-output proof.

### C2.4 `oxide-attn-gdn-tma`

Fresh disassembly of `oxide_attn_gdn_tma.cubin`:

```text
UTMALDG      2
UTMALDG.2D   2
LDG.E.128    0
LDG.E        72
FFMA         64
HMMA         0
MBAR         2
```

The fresh SASS md5 exactly matched the existing `oxide_attn_gdn_tma.sass` (`a5ad08051af6f5921ca52ef7078941c4`). `results.csv` bench rows peak at **278.192 GB/s**. Verdict: cuda-oxide TMA reach is proven; the near-FFMA-baseline throughput is an algorithm/parallelization limitation, not a TMA-availability failure.

### C2.5 `cutile-attn-gdn-tma`

MD5/size check:

```text
cubin_tma_on_qwen3_next_decode.cubin   372888 B  c102d09456668b2590d97f2198f60070
cubin_tma_off_qwen3_next_decode.cubin  372888 B  c102d09456668b2590d97f2198f60070
cubin_tma_on_large.cubin               372888 B  c102d09456668b2590d97f2198f60070
```

All three SASS dumps have `UTMA=0`, `UTMALDG=0`, `LDG.E.128=0`. Verdict: the DSL falsification is solid for this kernel: `allow_tma=True`/`False` are byte-identical on qwen3 and the explicit large-shape TMA leg is also identical. Correct framing: the TMA hint is accepted but is a no-op for these GDN tile shapes, not that cuTile can never emit TMA.

### C1.4 `mojo-matmul-tiled`

Reran `bash run.sh` successfully:

```text
min_ms=18.737119 median_ms=18.949567 max_ms=19.610176
TFLOPS_median=7.252880948255969 TFLOPS_best=7.335116645840804
correctness: max_abs_err=0.0 max_rel_err=0.0
```

The original **7.69 TF median** did not reproduce exactly, but the rerun is within ~6% and consistent with WSL/no-clock-lock variance. The architectural interpretation is intact: FFMA-only, no Tensor Core, no register microtile, roughly 5x below `cuda-matmul-tiled` because CUDA uses a 4x4 register microtile. Bit-exact sampled correctness is confirmed.

### C2.1/C2.2/C2.6 spot checks

- `cuda-matmul-tc-bf16`: SASS has **64 `HMMA.16816.F32.BF16`**, **0 `FFMA`**, **8 `LDGSTS`**; CSV best **147.527 TF**, median **146.771 TF**.
- `cutile-matmul-tc-bf16`: CSV best **160.572 TF**, median **159.117 TF**.
- `cuda-attn-kda`: large-shape CSV best **568.415 GB/s**, median **557.476 GB/s**. `cutile-attn-kda/results_large.csv` best **1170.128 GB/s**, so CUDA is **48.58%** of cuTile saturation, matching the 48.6% claim.

## WGPU / LLVMPIPE integrity

All rerun wgpu cells enumerate only llvmpipe CPU adapters. None should be presented as RTX 5090 GPU throughput.

- `wgpu-vecadd`: fresh run used `llvmpipe ... type=Cpu`; N=16M best **16.00 GB/s**, `max_abs_err=0`. Original 17.26 GB/s is within variance and explicitly CPU/LLVMPIPE.
- `wgpu-reduction`: fresh run best **2.3 GB/s**, median **2.2 GB/s**, `rel_err=1.799e-7`; original 1.9 GB/s is reproducible enough and explicitly CPU/LLVMPIPE.
- `wgpu-matmul-tiled`: default 50-iter run is impractical on llvmpipe; 1-iter rerun completed at **0.006 TFLOPS**, `max_abs_err=1.621e-5`, confirming original value and caveat.
- `wgpu-attn-mla`: fresh run warns “TFLOPS below is NOT GPU perf”; correctness shape has `max_abs_err=1.192e-7`. Fresh timing was lower than the older log (0.0006 TF vs prompted 0.0046 TF), and canonical `deepseek_v3` skips due to llvmpipe’s 128 MiB binding cap versus a 2 GiB scores buffer.

Documentation check: `wgpu-vecadd`, `wgpu-reduction`, and `wgpu-matmul-tiled` have strong `ANALYSIS.md` llvmpipe/CPU caveats. **`wgpu-attn-mla` has no `ANALYSIS.md`**, though its run log and source comments contain the right warnings.

## Issues

- **MEDIUM — C1.6 `wgpu-attn-mla` missing `ANALYSIS.md`.** This fails the task’s exact “all wgpu-* cells disclose LLVMPIPE-CPU in ANALYSIS.md” criterion. Add an analysis file that labels all TFLOPS as llvmpipe CPU-only, notes canonical shape skip, and records the correctness result.
- **LOW — C1.4 rerun drift.** Current rerun median **7.25 TF** vs analysis **7.69 TF**. Not a correctness issue; use a range or footnote WSL variance if publishing exact numbers.
- **LOW — C1.5 +10% is rounded.** Exact uplift is **9.74%**; “~10%”/“1.10x” is best.
- **LOW — C1.6 wgpu timing unstable.** Since it is CPU/llvmpipe/JIT-sensitive and not GPU perf, avoid treating 0.0046 TF as a stable performance headline.

## Questions

1. Should the Rosetta matrix keep WGPU entries as `wgpu/llvmpipe CPU fallback`, or hold them until native Linux/Windows real-GPU reruns?
2. Should “BIT-EXACT” be standardized to “sampled bit-exact vs CPU reference” for attention cells?

## Intersection-vs-union verdict

**Intersection:** the high-impact claims survive: Mojo GQA is ~1.10x CUDA GQA at identical shape; cuda-oxide reaches TMA (`UTMALDG.2D=2`, `LDG.E.128=0`); cuTile GDN-TMA is falsified by byte-identical cubins.

**Union caveats:** WGPU numbers are WSL/llvmpipe CPU characterizations only; one WGPU cell lacks the required analysis document; rerun variance affects exact Mojo/WGPU numbers but not the architectural conclusions.

**Recommendation:** accept Wave C after adding `wgpu-attn-mla/ANALYSIS.md` (or marking C1.6 documentation incomplete). Do not publish any WGPU number as RTX 5090 GPU throughput.
