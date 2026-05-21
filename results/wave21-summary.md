# Wave 21 — Mojo bf16-in/f32-acc tiled matmul (hand-rolled, harness for Wave 20 path)

**Status:** ✅ SHIPPED 2026-05-21
**Date:** 2026-05-21 (single-session continuation of Waves 18-20)
**GPU:** RTX 5090, sm_120, driver 596.21
**Mojo:** 1.0.0b1 (a9591de6)
**Predecessor:** Wave 20 (probe-only finding that `HMMA.16816.F32.BF16 × 1` is reachable on `.target sm_120a` via raw `mma()` from a single-warp probe). See [results/wave20-summary.md](wave20-summary.md).

## Headline finding

**Mojo's bf16-in/f32-acc tiled matmul on consumer Blackwell sm_120 ships at
79.3 TFLOPS at M=N=K=4096** (median of 10 iters, best 79.8 TF).

This closes the API-coverage gap proven in Wave 20: the path was reachable
from the low-level `from std.gpu.compute.mma import mma` primitive but had
no harness. Wave 21 builds the harness and lands the perf number.

## Cross-frontend matmul comparison (RTX 5090 sm_120, all M=N=K=4096, all on the same idle GPU 2026-05-21)

| frontend | path | TFLOPS | precision | TC reach? |
|---|---|---:|---|---|
| cuBLAS | `bgemm` (cublasGemmEx + TENSOR_OP) | **219.3** | bf16 in/out, f32 acc | ✅ |
| cuBLAS | `hgemm` (cublasGemmEx + TENSOR_OP) | 219.1 | f16 in/out, f32 acc | ✅ |
| cuTile | `mma_bf16xbf16_f32acc` | 159.95 | bf16 in/out, f32 acc | ✅ |
| **Mojo (Wave 21)** | hand-rolled `mma()` + epilogue | **79.3** | **bf16 in, f32 acc** | ✅ |
| Mojo (Wave 19) | `TensorCore` wrapper TF32 path | 55.5 | f32 in/out (TF32 hw) | ✅ |
| nvcc | tiled f32 (block-tile) | 38 | f32 | ❌ |
| cuTile | `mma_f32xf32_f32acc` | 8.7 | f32 | ❌ |
| Mojo (Wave 18) | naive f32 | 7.1 | f32 | ❌ |
| oxide | unchecked fmuladd | 7.0 | f32 | ❌ |
| nvcc | naive f32 | 6.4 | f32 | ❌ |

**Wave 21 vs cuTile bf16:** 49.6% (gap explained — see SASS analysis).
**Wave 21 vs cuBLAS bgemm:** 36.2% (cuBLAS uses internally-tuned cubins; user code rarely hits this).
**Wave 21 vs Mojo TF32 (Wave 19):** +43% (bf16 has 4× the MMA flops/instr at m16n8k16 vs TF32 at m16n8k4).
**Wave 21 vs Mojo naive f32 (Wave 18):** +1017% (10.2× — TC engagement vs scalar f32).

## SASS evidence

`mojo-matmul-bf16/matmul_bf16.sass` (298 lines):

- ✅ `.target sm_120a` (consumer Blackwell, native arch)
- ✅ `HMMA.16816.F32.BF16 × 16` in inner loop (TC engaged, m16n8k16 bf16 dispatch)
- ✅ `UTMALDG × 0` (no TMA — Mojo `cp.async` path; explains gap to cuTile)
- ✅ `LDGSTS × 32` (cp.async DRAM→SMEM loads engaged)
- ✅ Zero `tcgen05` instructions (correct — sm_120 doesn't support those, MAX flagship matmul gates them off via PR #6059)

The `HMMA × 16` count comes from the inner unroll: `comptime for` over `BK/MMA_K × WM/MMA_M × WN/MMA_N = 2 × 2 × 4 = 16` MMA positions per block per K-tile-pass. Total dynamic MMAs across the full launch = 16 × (K/BK) × (M/BM) × (N/BN) × NUM_WARPS = 16 × 128 × 64 × 64 × 4 = 33.5M, matching `2·M·N·K / (m·n·k) = 2·4096³ / (16·8·16) = 33.5M` ✓.

## Numerical correctness

**Validated at three problem sizes:**

| M=N=K | check | max_abs_err | max_rel_err | tolerance | result |
|---|---|---|---|---|---|
| 64 | full CPU ref | 2.4e-7 | 3.0e-7 | atol=1, rtol=1e-2 | ✅ PASS |
| 256 | full CPU ref | 5.7e-6 | 7.0e-7 | atol=1, rtol=1e-2 | ✅ PASS |
| 4096 | 1024-sample CPU ref | **2.2e-3** | 1.87e-5 | **atol=1e-2, rtol=1e-3** | ✅ PASS |

**Tolerance for the M=4096 sampled check was tightened to atol=1e-2, rtol=1e-3** during Phase 4 review (R3 mathematically proved the original atol=10 was 5500× looser than observed error and would have passed kernels returning 85% of the correct value). The 1024-sample version uses HIGH bits of a Knuth-multiplier hash for both `i` and `j` to avoid the low-bits-of-multiplier near-½ Weyl clustering that the original `i = seed % M` had.

The reported `max_abs_err = 2.2e-3` at K=4096 is consistent with f32-accumulator differential summation between the kernel (tiled MMA order) and the CPU reference (left-to-right scalar order). Wilkinson f32 worst case for K=4096 sum to magnitude ~67 is `K · ε_f32 · |S| ≈ 1.6e-2`. Observed error sits at 13% of that bound, comfortably inside.

## Key technique: `TensorCore` wrapper for fragment loads + raw `mma()` + hand-rolled epilogue

The plan's primary path was "drop completely to raw `mma()` and `ld_matrix`", but during execution we found a cleaner hybrid:

1. **`TensorCore[bf16, bf16, Index(16, 8, 16)]()`** for `load_a` and `load_b` ONLY. The wrapper's `A.dtype == C.dtype` constraint bites in `mma_op` (line 842 of `tensor_core.mojo`) and `store_d` (line 781), but NOT in `load_a/load_b` — those are happy to load uniform-bf16 fragments.
2. **Raw `mma(d_frag, a_frag, b_frag, c_frag)`** from `std.gpu.compute.mma`, with bf16 SIMD-8/SIMD-4 inputs and f32 SIMD-4 accumulator. Bypasses the wrapper's mixed-precision constraint.
3. **Hand-rolled epilogue** per PTX 9.7.13.4.8 m16n8 distribution:
   ```mojo
   var lane = Int(lane_id())
   var group_id = lane >> 2
   var tid_in_grp = lane & 3
   comptime for i in range(4):
       var row = group_id + (i >> 1) * 8
       var col = (tid_in_grp << 1) + (i & 1)
       C_mma_tile[row, col] = c_reg_tile[0, i]
   ```

This three-step strategy was **not** in the plan's primary or fallback path. R1's spec-compliance review flagged this as a deviation; R2's code-quality review confirmed the structural correctness (epilogue formula matches PTX byte-for-byte, no data races, no bank conflicts on `ldmatrix` against the row-major bf16 layout).

The deviation is **principled and recommended for future bf16/f16/FP8 lanes** — the wrapper's load_a/load_b are well-tested and produce optimal `ldmatrix.x4` SASS, while only the mma+store path needs hand-rolling.

## Why 79 TF instead of 100-130 TF (plan target)

R1 flagged this as MAJOR. The plan's Goal target (100-130 TF) was based on cuTile's 159 TF being primarily a TMA-vs-cp.async gap with secondary register-allocation gains. SASS analysis shows two compounding factors:

1. **No TMA** (`UTMALDG × 0`). cuTile's bf16 path emits TMA bulk loads via `ct.load`. Mojo's `copy_dram_to_sram_async` lowers to `cp.async` (pre-Hopper async-copy path). The TMA path has lower LSU pressure on Blackwell since the SM doesn't have to issue per-warp loads. Estimated gap: ~30-40 TF.
2. **No swizzled smem layout.** Wave 19's `Layout.row_major(BM, BK)` was used as-is. cuTile's tiled matmul uses an internal swizzled smem layout that cooperates with `ldmatrix` to reduce bank conflicts. R2 noted no critical bank-conflicts with `ldmatrix.x4` on bf16 row-major(64,32) but a 5-10% lift might be available with padded smem.
3. **The `TensorCore.load_a/load_b` wrapper allocates intermediate registers** that may not fold cleanly across the `mma_k ∈ {0,1}` inner unroll. Manual `ld_matrix` calls might shave a few percent.

So the gap is **predominantly TMA**, with secondary register-allocation effects, **all out of scope for closing in this wave**. Wave 22 candidate.

## Files added in `/home/codeseys/cuda-exploration/`

- `mojo-matmul-bf16/matmul_bf16.mojo` (~360 lines) — the kernel + harness with timing + sampled correctness check
- `mojo-matmul-bf16/matmul_bf16.sass` (298 lines) — captured SASS evidence
- `mojo-matmul-bf16/run.log` (gitignored) — full run output
- `mojo-matmul-bf16/run.sh`, `.gitignore`, `ANALYSIS.md`
- `docs/plans/wave-21.md` — implementation plan (added Wave 21 task list)
- `results/wave21-summary.md` — this file

## Pitfalls discovered (added to skill)

1. **`LayoutTensor[i, j]` returns `SIMD[dtype, element_size]`, not `Scalar[dtype]`.** For `element_size==1` (the typical case from `loader.load_a` / `loader.load_b`), grab the scalar lane via `a_lt[0, k][0]`. The documented `load_scalar` method exists but its `Tys: Indexer` parameter doesn't accept `IntLiteral` directly in Mojo 1.0.0b1.
2. **`ref` is a Mojo keyword** (sibling to `mut`/`read`/`out`). Using it as a variable name produces a confusing parse error pointing at the colon. Rename to `expected`.
3. **`copy_dram_to_sram_async` thread_layout × vectorize must match the DRAM tile's element count** — `(4, 8) + vectorize[1, 4]` works for both 64×32 (A) and 32×64 (B) at this tile shape, by way of the kernel internally cycling threads through multiple passes. Wave 19's pattern transfers to bf16 cleanly. A `(8, 16) + vectorize[1, 8]` layout that *also* totals 1024 elements/pass produces an `CUDA_ERROR_ILLEGAL_ADDRESS` at runtime — the (4, 8) shape is what the stdlib expects.
4. **`$(pwd)` inside `(cd $WORKSPACE && pixi run mojo $(pwd)/file.mojo)` evaluates AFTER the cd**, so the existing `mojo-matmul-tc/run.sh` and `mojo-mma-probe/run.sh` use a broken pattern (silently looking for the source file in the workspace dir, not the cell dir). The fix is to capture `HERE="$(pwd)"` in the outer shell before forking the subshell.

## Methodology notes from cross-model review

Three parallel reviewers (R1 spec-compliance, R2 code-quality, R3 numerical-correctness) ran via `delegate_task`. R3's mathematical analysis caught a real BLOCKER in the original validation block:

- **Original tolerance was atol=10, rtol=2e-2** — would have passed a kernel returning `0.85 × correct`. Tightened to `atol=1e-2, rtol=1e-3` per R3's Wilkinson-bound analysis.
- **Original sample size 256** — R3 proved 5-80% miss rate for realistic localized bugs. Bumped to 1024 samples.
- **Original LCG used low bits of multiplier** for `i` (1969 mod 4096 ≈ near-½ Weyl stride → 2-stripe clustering). Switched both `i` and `j` to high-bit windows of the Knuth multiplier output.
- **Original report was mean-of-10-iters** — plan specified median. Replaced with per-iter timing + insertion-sort + median, which actually *improved* the headline number from 73.1 to 79.3 TF (the mean was depressed by occasional 2.1ms outliers).

R2 confirmed: kernel's epilogue formula matches PTX 9.7.13.4.8 byte-for-byte; no data races; no missing barriers; smem layout has no critical bank-conflicts on `ldmatrix`.

R1 found an honest deviation: kernel uses `TensorCore` for `load_a/load_b` despite the plan saying "drop completely to raw `mma()` and `ld_matrix`". As detailed above, this hybrid is principled (the wrapper's load functions don't enforce the same-dtype constraint) and is the recommended pattern for future bf16/f16/FP8 lanes.

## Wave 22 candidates

- **W22.1: TMA loads via `std.gpu.sync.cp_async_bulk`** — close the cp.async-vs-TMA gap. Expected lift to ~120-130 TF (cuTile's level minus the swizzle-layout lift).
- **W22.2: f16 lane** at m16n8k16 (`HMMA.16816.F32.F16`) — same scaffolding, swap dtype. Should land ~80 TF (parity with bf16, both engage TC at the same shape).
- **W22.3: Padded-smem layout** to remove residual bank conflicts. ~5-10% on top of W22.1.
- **W22.4: FP8 lane** (e4m3 / e5m2) at m16n8k32. Different SIMD widths; full hand-roll required since `_mma_nvidia` uses inline asm not LLVM intrinsic for FP8.
- **W22.5: Attention column** with bf16 matmul + softmax — Phase D of the Wave 18 lineage now unblocked since Wave 19+21 confirm bf16 TC reach.
