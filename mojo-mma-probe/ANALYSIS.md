# mojo-mma-probe — Mojo bf16 mma.sync probe on sm_120

**Wave 20 W1.** Single-warp probe testing whether
`from std.gpu.compute.mma import mma` can emit
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` on consumer
Blackwell (RTX 5090, sm_120).

Wave 19 finding #11 had been: the high-level `from layout.tensor_core
import TensorCore` wrapper requires `A.dtype == C.dtype`, so bf16-in/
f32-acc was unreachable through it. This probe asks the lower-level
question: does the *underlying* mma() call work?

## Result

**Yes, the bf16 MMA path works on sm_120.** SASS evidence (`mma_probe.sass`):

```
.target sm_120a
...
HMMA.16816.F32.BF16 R4, R4, R12, R8 ;   ← single instance
```

A single `HMMA.16816.F32.BF16` SASS instruction in the warp body, on
the sm_120a target, with no fallback to scalar code or compile error.
**The bf16-in/f32-acc m16n8k16 tensor-core path is reachable from Mojo
on consumer Blackwell.**

This is a definitive answer to the Wave 19 / Wave 20 hypothesis: the
"same-dtype constraint" lives in the `TensorCore` wrapper, NOT in the
underlying hardware path or the `mma()` primitive.

## What this probe does

A 32-thread (single warp) kernel constructs trivial bf16 fragments
(8×bf16 for A, 4×bf16 for B, 4×f32 for C) seeded by `thread_idx.x` so
the compiler can't constant-fold them away, calls `mma(d, a, b, c)`,
and writes the 4×f32 output back to global memory. No tiling, no
shared memory, no benchmark — purely an "does this compile and emit
HMMA?" probe.

```mojo
var a_frag = SIMD[DType.bfloat16, 8](...)  # per-lane bf16 × 8
var b_frag = SIMD[DType.bfloat16, 4](...)  # per-lane bf16 × 4
var c_frag = SIMD[DType.float32,  4](...)  # per-lane f32 × 4
var d_frag = SIMD[DType.float32,  4](0,0,0,0)
mma(d_frag, a_frag, b_frag, c_frag)
out_ptr[Int(thread_idx.x) * 4 + i] = d_frag[i]
```

The SIMD widths come straight from the `_mma_nvidia` source for the
m16n8k16 BF16 dispatch lane (`mma_nvidia.mojo`):

```
elif _has_type[(BF16, BF16, F32, F32)]
     and _has_shape[(8, 4, 4, 4)]:
    var r = llvm_intrinsic[
        "llvm.nvvm.mma.m16n8k16.row.col.bf16", ...](...)
```

So `_has_shape[(8, 4, 4, 4)]` is the key dispatch condition — A.size=8,
B.size=4, C.size=D.size=4.

## What this probe does NOT do

- **No correctness check.** The output values printed are not compared
  against a reference. The probe's only purpose is to generate SASS
  showing the HMMA instruction. Per-lane fragment data is arbitrary;
  the warp-wide A and B "virtual" 16×16 / 16×8 matrices that the MMA
  computes against are essentially garbage from any matmul perspective.
- **No tiling, no harness for full matmul.** Building a real
  bf16-in/f32-acc tiled matmul harness is a separate, much larger
  project — see "Why no full matmul yet" below.

## Why no full bf16-in/f32-acc matmul yet (Wave 21+ candidate)

We attempted the easiest path: re-use the Wave 19
`tensor_core_matrix_multiplication` kernel structure but with
`a_type=bf16, c_type=f32`. **It fails at compile time** at TWO points
in `tensor_core.mojo`:

1. `mma_op.mma_op(...)` on line 814 — internal rebind from `f32` to
   `bf16` fails because the wrapper assumes uniform dtype across A/C/D.
2. `mma_op.store_d(C_mma_tile, c_reg_m_n)` on line 781 — same-dtype
   constraint, same as Wave 19 finding #11.

So even if you replace just `store_d` with a hand-rolled epilogue, the
`mma_op` itself rejects mixed-precision. **The whole `TensorCore`
struct is fundamentally same-dtype.** To get bf16-in/f32-acc on
Mojo, the harness needs to:

1. Hand-roll `ld_matrix` calls (or manual smem→reg gathers) per the
   m16n8k16 fragment layout
2. Call `mma()` directly (not through `TensorCore`)
3. Hand-roll the f32 epilogue write per PTX section 9.7.13.4.8
   distribution (groupID/tid_in_grp, 4 outputs per lane)

That's roughly the same complexity as the Wave 19 TF32 kernel but with
explicit fragment layout management instead of wrapper-managed.
Reasonable for Wave 21; out of scope for Wave 20.

## What this means for the user-facing comparison

The Wave 19 README claim that **"Mojo cannot reach the bf16-in/f32-acc
lane"** needs to be split into a two-tier statement:

- **High-level (`from layout.tensor_core import TensorCore`)**: cannot
  reach mixed-precision today. The wrapper has same-dtype constraints
  in both `mma_op` and `store_d`.
- **Low-level (`from std.gpu.compute.mma import mma`)**: CAN reach
  mixed-precision on sm_120. The hardware path works, the Mojo
  primitive emits the right instruction, and the path is documented in
  `mma_nvidia.mojo`. The cost is hand-rolling the tile/fragment/epilogue
  scaffolding.

## Files

- `mma_probe.mojo` (~3.9KB) — single-warp kernel + harness
- `mma_probe.sass` (82 lines, ~3KB) — captured SASS, **`HMMA.16816.F32.BF16` × 1**
- `run.sh` — reproduce
- `run.log` — fresh run output

## Next (Wave 21 candidate)

Build the full hand-rolled bf16-in/f32-acc tiled matmul:

1. `mojo-matmul-bf16/` cell using `from std.gpu.compute.mma import mma`
2. Hand-rolled `ld_matrix` for A/B fragments
3. Manual epilogue write per PTX m16n8 distribution
4. Numerical correctness check vs `vendor_blas.matmul`
5. Expected target: 100-130 TFLOPS (60-80% of cuTile's 159 TF, given
   no TMA loads in the Mojo path)

Reference materials:
- `mma_nvidia.mojo` — exact SIMD widths and intrinsic mappings
- The Mojo manual's `tensor_core_matrix_multiplication` (Wave 19) — same
  tile structure, replace `TensorCore` calls with raw `mma()`/`ld_matrix`
- `cutile-matmul-tiled-mixed/` — reference bf16 TC matmul to compare against
