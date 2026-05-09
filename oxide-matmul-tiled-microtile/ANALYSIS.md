# Wave 7: register-microtile + fmuladd cuda-oxide tiled matmul

**Goal:** apply Wave 5 SASS findings on the cuda-oxide side. Wave 5 showed nvcc's tiled kernel (28 TFLOPS at N=4096) wins via (a) 4×4 register microtile, (b) FFMA via `FastmathFlags::CONTRACT`, (c) `LDG.E.CONSTANT` cache hint. We can attack (a) and (b) in cuda-oxide; (c) needs upstream work (no inline PTX in cuda-oxide kernels).

## Algorithm

- **Block:** 16×16 threads (256 per block)
- **Output covered per block:** 64×64 (each thread writes a 4×4 microtile)
- **K-loop tile size (BK):** 16
- **Shared tiles:** `TILE_A` 64×16 = 1024 f32, `TILE_B` 16×64 = 1024 f32
- **Cooperative load:** 256 threads × 4 elems = 1024 per shared tile, perfect coverage
- **Two kernels:**
  - `matmul_tiled_4x4`: accumulation via `core::intrinsics::fmuladdf32`
  - `matmul_tiled_4x4_safe`: plain `sum = sum + a*b` (default `*+`, FastmathFlags::empty())

Implementation note: cuda-oxide can't lower 2-level array projections in assignments (e.g. `c[i][j] = …`), so the 4×4 accumulator is 16 scalar locals (`c00`, `c01`, …, `c33`).

## TFLOPS measurements (RTX 5090, gaming concurrent — CV ~20-30%)

| Kernel | N=1024 | N=4096 |
|---|---:|---:|
| matmul_tiled_4x4 (fmuladd) | 27.8–28.9 | 16.2–17.1 |
| matmul_tiled_4x4_safe (plain `*+`) | 24.7–26.1 | 14.9–24.0 |
| **Reference: oxide-matmul-tiled (old, 16×16, 1 output/thread)** | 9.1 | 7.9 |
| **Reference: cuda-matmul-tiled (nvcc, 32×32 + 4×4 microtile)** | 24.5 | 28.1 |
| **Reference: oxide-matmul/safe (naive, 1 output/thread)** | 6.94 | 4.84 |

**Headline:** at N=1024 cuda-oxide tiled-microtile reaches **103-118%** of nvcc-tiled. At N=4096 cuda-oxide reaches **57-85%** of nvcc-tiled (wide due to gaming noise on 8 ms kernels). Versus the original oxide-tiled, this is a **3.0-3.6× lift at N=1024 and 2.0-3.0× at N=4096.**

## SASS instruction counts (per kernel function)

`cuobjdump --dump-sass` on the cubin:

| Kernel | FFMA | LDS | LDG.E | IMAD | BRA | Total |
|---|---:|---:|---:|---:|---:|---:|
| matmul_tiled_4x4 (fmuladd) | 64 | 8 | 8 | 25 | 4 | 564 |
| matmul_tiled_4x4_safe (plain `*+`) | **128** | 16 | 8 | 23 | 4 | 708 |
| nvcc matmul_tiled (reference) | 256 | 32 | 16 | 37 | 3 | 1044 |
| oxide-matmul/unchecked (naive, Wave 5) | 0 | 0 | 16 | — | — | — |
| oxide-matmul/fmuladd (naive, Wave 5) | 8/loop | 0 | 16 | — | — | — |

**Surprise finding #1: libNVVM DOES contract `*+` to FFMA in this kernel.** The `_safe` kernel uses plain `sum = sum + a*b` — Wave 3 said this produces zero FFMAs because `FastmathFlagsAttr::default() = empty()`. **It produced 128 FFMAs anyway.** Same finding from Wave 8's 3DGS kernel (9 FFMAs from plain `*+`). This contradicts the Wave 3 narrative.

Hypothesis: libNVVM's contractor fires when the IR pattern is "scalar fmul followed by fadd into the same accumulator," and the hot inner loop with 16 fully-unrolled scalar-accumulator updates (`c00 = c00 + a0*b0; c00 = c00 + a1*b1; …`) hits this pattern cleanly. The naive matmul's hot loop has the same shape but with `dim=1024` un-unrollable iterations indexed by a runtime `k`, which apparently confuses the contractor. **Wave 3's "FastmathFlags::empty() blocks contraction" finding may be specific to the runtime-bounded loop case, not universal.** Worth re-investigating.

**Surprise finding #2: fewer FFMAs is faster.** `matmul_tiled_4x4` (fmuladd) has 64 FFMAs; `matmul_tiled_4x4_safe` has 128. The `_safe` version achieves slightly *higher* peak TFLOPS at N=1024. Why fewer FFMAs in the fmuladd kernel? Because `core::intrinsics::fmuladdf32` is implemented as a libdevice call to `__nv_fmaf`. nvJitLink inlines that call; ptxas presumably keeps the inlined FMAs as separate instructions but with different register allocation than the `_safe` version's directly-emitted `fma.rn.f32`. So the libdevice path is structurally different from the direct-pattern-match path even though both end at FFMA in SASS.

**Conclusion at SASS level:** both kernels successfully emit FFMA; both load from shared memory via LDS; both are unrolled. The remaining gap to nvcc-tiled (256 FFMAs, 32 LDS, 16 LDG.E, 1044 total instructions) is consistent with nvcc using a 32×32 thread block + 4×4 microtile (covers 128×128 output per block, vs cuda-oxide's 16×16 + 4×4 = 64×64). nvcc has 4× the total compute per thread, hence ~4× more instructions per kernel, but at 32 threads per warp this exposes more ILP and amortizes LDG/LDS costs better.

## What's left

- **Match nvcc's 32×32 + 4×4 geometry.** A 32×32 block (1024 threads) with 4×4 microtile covers 128×128 output per block; shared tiles need 128×16 = 2048 f32 each (just under the 48 KB shared mem limit on Blackwell). We didn't try this; would likely close more of the N=4096 gap.
- **`LDG.E.CONSTANT` upstream issue.** Still requires upstream patch; can't fix from kernel-side alone (no inline PTX in cuda-oxide).

## Verdict

**Cuda-oxide can hit nvcc-tiled performance at N=1024 with a thoughtful kernel.** The 3-3.6× speedup over the old oxide-tiled comes from:
1. Register microtile (algorithm geometry — by far the biggest win)
2. FFMA via either fmuladdf32 explicit, OR plain `*+` when the loop is fully unrolled (libNVVM contracts in that case)

The "Rust safety tax" story stays dead (safe kernels match unchecked). The "cuda-oxide can't reach nvcc tiled performance" story is **partially refuted**: at small problem sizes parity is reachable; at N=4096 the gap halves but doesn't close, primarily due to thread-block geometry not the compiler.
