# Wave 5: SASS-level analysis of N=4096 naive-matmul gap

## Hypothesis (entering)

At N=4096, nvcc naive matmul lands ~6.23 TFLOPS while cuda-oxide unchecked is
5.67 TFLOPS and cuda-oxide fmuladd is 5.31. Wave 3 closed the FMA question at
PTX level. The new hypothesis for the residual: **nvcc unrolls the K-loop
and cuda-oxide does not** — we should see ~4–8 back-to-back FFMAs in nvcc's
SASS and a single FFMA/iteration loop in oxide's SASS.

## Method

- `nvcc -ccbin clang-14 -O3 -arch=sm_120 -cubin` → `matmul.cubin` (8,408 B)
- Oxide path: `oxide_matmul.ll` → `llvm-link` against libdevice.10.bc → `opt
  -O2` → `llc -mcpu=sm_120 -mattr=+ptx87` → `ptxas -arch=sm_120 -O3` →
  `oxide_matmul.cubin` (21,288 B; contains all three kernel variants).
- `cuobjdump --dump-sass` on both, per-kernel extraction with awk.

## Instruction counts (one unrolled hot-loop iteration)

| Metric                     | nvcc `matmul` | oxide `matmul_unchecked` | oxide `matmul_fmuladd` |
|----------------------------|---------------|---------------------------|-------------------------|
| FFMA                       | **8**         | 0                         | **8**                   |
| FMUL                       | 0             | **8**                     | 0                       |
| FADD                       | 0             | **8**                     | 0                       |
| LDG (global loads)         | 16            | 16                        | 16                      |
| LDG cache variant          | **LDG.E.CONSTANT** | LDG.E                 | LDG.E                   |
| LDS (shared)               | 0             | 0                         | 0                       |
| Total insns in hot block   | 49            | 44                        | 36                      |
| Unroll factor              | **8×**        | **8×**                    | **8×**                  |

Kernel totals (whole function): nvcc 15 FFMA / 30 LDG; oxide_unchecked 17
FMUL + 17 FADD / 34 LDG; oxide_fmuladd 17 FFMA / 34 LDG. All three share
the prologue/epilogue pattern (main 8-wide unrolled body + scalar tail for
the remainder loop — 4-wide, 2-wide, 1-wide phases in nvcc; similar in oxide).

## SASS excerpts

**nvcc hot loop (one unrolled body, 8 FFMAs chained on `R17`/`R12`):**

```
/*0240*/ LDG.E.CONSTANT R13, desc[UR16][R4.64+-0x10]   ;  // 16 hoisted loads
/*0290*/ LDG.E.CONSTANT R15, desc[UR16][R4.64+-0xc]    ;  // all .CONSTANT
...      (16 LDG.E.CONSTANTs total, pointer-stride from A and B)
/*0490*/ FFMA R12, R13, R12, R0    ;                     // acc (R0) -> R12
/*04a0*/ FFMA R12, R15, R14, R12   ;                     // chained
/*04b0*/ FFMA R17, R17, R16, R12   ;
/*04c0*/ FFMA R17, R24, R20, R17   ;
/*04d0*/ FFMA R17, R26, R22, R17   ;
/*04e0*/ FFMA R17, R28, R18, R17   ;
/*04f0*/ FFMA R8,  R30, R8,  R17   ;
/*0500*/ FFMA R0,  R21, R10, R8    ;                     // final -> R0 acc
/*0510*/ BRA.U UP0, 0x210          ;                     // loop back
```

**oxide `matmul_unchecked` hot loop (8×, FMUL+FADD interleaved with LDGs):**

```
/*0440*/ FMUL R31, R25, R24                ;
/*0460*/ LDG.E R19, desc[UR4][R16.64+0x18] ;   // no .CONSTANT suffix
/*0470*/ FADD R18, R31, R28                ;   // split into two ops
/*0490*/ FMUL R35, R29, R26                ;
/*04f0*/ FADD R35, R18, R35                ;
/*0500*/ FMUL R32, R33, R32                ;
/*0520*/ FMUL R37, R37, R20                ;
/*0530*/ FADD R32, R35, R32                ;
/*0550*/ FADD R32, R32, R37                ;
/*0570*/ FMUL R23, R28, R36                ;
/*0580*/ FADD R23, R32, R23                ;
/*0590*/ FMUL R22, R29, R22                ;
/*05a0*/ FADD R22, R23, R22                ;
/*05b0*/ FMUL R21, R26, R21                ;
/*05c0*/ FMUL R28, R19, R30                ;
/*05d0*/ FADD R21, R22, R21                ;
/*05f0*/ FADD R28, R21, R28                ;
/*0600*/ BRA  @!P1 0x350                   ;   // loop back
```

oxide `matmul_fmuladd` matches nvcc's FFMA chain (8 FFMAs) but still emits
plain `LDG.E` — no `.CONSTANT`.

## Conclusion — hypothesis verdict

**REJECTED.** Both compilers unroll the K-loop 8× (same trip count per hot
iteration, same 16 LDGs hoisted). The residual gap decomposes into two
orthogonal SASS-level deltas:

1. **FP instruction count (unchecked/checked only):** oxide emits
   `FMUL` + `FADD` pairs instead of fused `FFMA`, doubling FP issue pressure
   in the hot loop (16 FP insns vs 8). This is the Wave 3 FastmathFlags
   issue, now visible at ISA level.
2. **Memory-descriptor class (all oxide variants):** nvcc uses
   `LDG.E.CONSTANT` — routes through the read-only / uniform cache path,
   enabled by `__restrict__` + `const float*` annotations on the nvcc
   source. cuda-oxide's NVPTX backend emits plain `LDG.E`, missing the
   `ldg` intrinsic / `!invariant.load` metadata that would unlock the same
   promotion. This alone is consistent with the `fmuladd → nvcc` residual
   (~15% gap even with FFMA parity).

So the unrolling story from orchestrator intuition was wrong; the real
remaining levers are (a) FMA contraction (already documented) and (b)
load-class promotion to the read-only cache. (b) is a new finding and
a plausible upstream patch target for cuda-oxide's lowering.
