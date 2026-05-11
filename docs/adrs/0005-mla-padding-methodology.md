# ADR 0005: MLA per-frontend padding policy and FLOPS accounting

**Status:** accepted (2026-05-11)

**Context.** DeepSeek-V3 MLA has Q/K head dim qk = 192 (128 nope + 64 rope). This is **not** a friendly tile dimension for tensor-core kernels — HMMA's 16×16 fragment shape means 192 doesn't tile cleanly. Wave 16's `cutile-attn-mla` padded qk = 192 → 256 to keep `ct.mma` happy; the cell pays a measured 25% wasted compute (192/256) but still hit 112.4 TF.

For Wave 17 we need to:
1. Decide the padding policy per frontend (cuTile, oxide, nvcc, cuBLAS).
2. Decide how to count FLOPS — useful work or padded work? Different choices look bad on different cells.

**Decision.**

### Padding policy by frontend

| Frontend | Policy | Rationale |
|---|---|---|
| cuTile | Pad 192 → 256 | `ct.mma` v1.3.0 wants **power-of-two inner dims** for HMMA lane packing (Wave 13.1 finding). 192 is NOT a power of two; 256 is the smallest power-of-two pad. The 25% waste is a frontend limitation, not a hardware constraint. |
| nvcc + WMMA | **192-native** (NO pad) | WMMA 16×16×16 fragments tile 192 cleanly along K (12 trips of K=16). Power-of-two is a cuTile-specific lane-packing requirement, NOT a WMMA hardware requirement. Per ADR 0005 Pre-mortem #3, 192-native nvcc+WMMA is the apples-to-apples baseline that exposes cuTile's padding as a frontend cost. |
| nvcc + scalar FMA (no WMMA) | 192-native | No alignment requirement; padding wastes registers and DRAM |
| cuBLAS | Pad 192 → 256 OR 192-native | cuBLAS hgemm accepts 192 directly; padded comparison is fairer to TC frontends but masks cuBLAS's actual capability. **We do BOTH cells** when the cost is small (1 extra `cublasGemmEx` call) and report both numbers. |
| cuda-oxide | 192-native | Per `docs/research/wave17-oxide-mla-design.md` finding — oxide emits scalar `fmuladdf32`, no K-loop alignment requirement, padding only imports cuTile's 25% waste with no benefit. |

### FLOPS accounting

**Headline rule (one number per cell):** the cross-frontend comparison number for any MLA cell is `useful_flops / wall_time`, where `useful_flops = 2 * B * S² * (qk + d_v) * n_h` with qk = 192. This is the SAME numerator regardless of whether the kernel internally padded — what changes between cells is the wall_time, which captures the padding cost honestly.

**Diagnostic column (in ANALYSIS.md only, NOT headline):** `padded_flops / wall_time` where `padded_flops = 2 * B * S² * (qk_pad + d_v) * n_h`. Label it "hardware-throughput, not comparable across padding policies." This is the "what the hardware actually computed per second" number — useful for sanity-checking against HMMA peak but NOT a fair cross-frontend metric.

**DRAM vs register padding sanity check:** every MLA cell MUST report `LDG.E.*` byte count for K/V loads (from SASS) and compare to:
- `B · S · n_h · qk · 4B` (unpadded reference)
- `B · S · n_h · qk_pad · 4B` (DRAM-padded)

If LDG bytes match the unpadded reference, padding is register-only. If LDG bytes match the padded reference, the cell wastes DRAM bandwidth proportional to the pad ratio.

### Cross-frontend headline framing

```
| Frontend | TFLOPS (useful, qk=192) | TFLOPS (padded, qk_pad) | Padding overhead |
| cuTile   | 112.4                   | 149.9 (qk_pad=256)       | 25%              |
| nvcc-WMMA | TBD                    | TBD                      | TBD              |
| oxide    | TBD (qk_pad=qk=192)     | same                     | 0%               |
```

When two frontends both pad to 256, their **padded TFLOPS** are directly comparable (same compute work). When one pads and one doesn't, **useful TFLOPS** is the apples-to-apples number.

**Consequences.**

- **Positive:** cuTile-MLA's 112.4 TF doesn't get framed as "slower than it could be" — the 25% is a hardware-feature constraint, not a kernel bug.
- **Positive:** oxide-attn-mla can be honestly reported as "192-native, 0% padding overhead, but 4× lower TFLOPS than cuTile" — separating padding from TC-access in the comparison.
- **Negative:** every MLA cell now has 2 FLOPS numbers in its headline. We accept the ceremony for the comparison clarity.
- **Risk:** future readers might quote padded FLOPS thinking it's the headline number. Mitigation: useful FLOPS is always FIRST in tables and bolded.

**Pre-mortem.**

1. *cuBLAS-MLA cell shows 218 TF padded vs 164 TF useful and the headline says "cuBLAS wins."* The 218 TF is doing 33% useless work. Mitigation: `wave17-summary.md` headline table uses **useful** column for the rank ordering; padded column is in a sub-table.
2. *We measure useful FLOPS but the kernel actually loaded only 192 bytes per head and stored only 192-element row.* If a frontend doesn't waste DRAM (only register-padded), useful = padded for memory-bound regions. Document per cell whether padding is register-only or also touches DRAM.
3. *Reviewer asks "why didn't you also try pad-to-200 (smaller waste)?"* 200 isn't a multiple of 16 either; 192 → 256 is the smallest WMMA-compatible pad. We don't measure intermediate pads.
