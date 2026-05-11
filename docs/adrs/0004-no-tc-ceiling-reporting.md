# ADR 0004: No-TC ceiling cells report HMMA=0 explicitly

**Status:** accepted (2026-05-11)

**Context.** Wave 14+ introduced multiple frontends (cuda-oxide, wgpu) that have NO usable Tensor Core path on consumer Blackwell sm_120. Wave 16 confirmed:

- `oxide-attn-gqa`: 24.15 TF f32 with HMMA=0, FFMA=157 in SASS — this is the **non-TC ceiling** for that algorithm on this GPU.
- `wgpu-attn-gqa`: 0.0026 TF on llvmpipe (CPU fallback because WSL2 has no Vulkan) — neither a TC ceiling nor a non-TC ceiling.

When a reader compares oxide's 24 TF to cuTile-fused's 165 TF, the right framing is "compiler-quality ceiling absent tensor cores" vs "with tensor cores," not "oxide is 7× slower than cuTile" — the latter conflates frontend choice with hardware-feature access. Wave 16 summary already does this in prose; this ADR codifies it as policy for Wave 17+.

**Decision.** Every Wave-17+ cell that reports a TFLOPS or GB/s number MUST also report the SASS instruction-mix evidence in its `ANALYSIS.md`:

| Field | Required | How to obtain |
|---|---|---|
| HMMA count | Always | `/usr/local/cuda/bin/cuobjdump --dump-sass <cubin> \| grep -c HMMA` |
| FFMA count | When HMMA=0 | Same, with `FFMA` |
| Vectorized-load evidence (LDG.E.128 / UTMALDG / LDS.128) | When kernel is memory-bound | Same, with the relevant op |
| MUFU/SFU count | For any cell with softmax / exp / reciprocal | `cuobjdump --dump-sass <cubin> \| grep -c MUFU` — required for ALL Wave 17 attention cells (GQA, MLA, GDN, KDA all run softmax or exp-gate) |
| TC-availability statement | Always | One sentence: "this frontend has TC available on sm_120" / "this frontend has no TC API on sm_120 in v<version>" / "this frontend's TC path was tested; HMMA emitted N times" |
| HMMA-count sanity formula | When HMMA > 0 | Compute expected `HMMA_count ≈ FLOPs / (HMMA_fragment_flops × grid_iters)` and report measured-vs-expected ratio in ANALYSIS.md. Ratio outside [0.5, 2.0] requires explanation. HMMA fragment FLOPs: f16 16×16×16 = 8192, tf32 16×16×8 = 4096. |

**Cross-frontend headline tables** in `results/wave17-summary.md` MUST split into two columns when both TC and no-TC cells exist for the same mechanism:

```
| Mechanism | TC ceiling (best frontend) | no-TC ceiling (best frontend) | TC vs no-TC ratio |
```

A "no-TC ceiling" claim requires HMMA=0 in SASS. A "TC ceiling" claim requires HMMA > 0 AND the count is consistent with the reported FLOPS (e.g., 165 TF f16 GQA @ HMMA=256 = sane; 165 TF f16 @ HMMA=4 = lying).

**Consequences.**

- **Positive:** readers can't conflate frontend-choice with hardware-feature ceiling. The cuda-oxide vs cuTile comparison becomes meaningful instead of misleading.
- **Negative:** every cell needs a SASS step. Already done for Waves 12-16; codifying makes it non-optional going forward.
- **Negative:** when a frontend gains a TC API (cuda-oxide upstream lands `cuda_device::mma`), we'll need a v2 cell to update the ceiling — but that's correct, not a defect.

**Pre-mortem (3-month-out failure scenarios).**

1. *Reader sees "oxide GDN 540 GB/s, cuTile GDN 610 GB/s" and assumes oxide is missing TC.* GDN is memory-bound and has 0 HMMA in BOTH frontends — TC is irrelevant. Mitigation: when HMMA=0 in BOTH cells, the ANALYSIS.md MUST include "this kernel is memory-bound; tensor cores are not applicable" so the no-TC framing isn't misapplied.
2. *Cell reports "no HMMA" but uses MUFU.EX2 / MUFU.RCP for softmax, hitting SFU peak.* HMMA=0 ≠ "scalar FMA only." For attention/softmax cells, also report SFU instruction count when reasoning about a peak.
3. *cuda-oxide v0.2.0 ships `cuda_device::mma` and oxide-attn-gqa-v2 hits 100 TF.* Old v0.1.0 cell stays as the "no-TC ceiling at the time" historical record; we add v2 as a separate cell. Don't mutate published numbers.
