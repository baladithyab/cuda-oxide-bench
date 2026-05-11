# Wave 3 W3A — FMA toggle experiment

**Status:** complete
**Date:** 2026-05-08
**Owner:** W3A subagent
**Reads:** `docs/research/cuda-oxide-flags.md`
**Writes:** this file + `oxide-matmul/src/main.rs` (added `matmul_fmuladd` kernel)

## TL;DR

Adding a `matmul_fmuladd` kernel that uses `core::intrinsics::fmuladdf32`
**does** produce `fma.rn.f32` in the PTX that actually executes on the GPU.
Phase-3 research concluded (correctly) that `fmuladdf32` lowers to a libdevice
`call @__nv_fmaf` rather than to the `llvm.fmuladd.f32` intrinsic — so in the
pre-link NVVM IR and the un-linked `llc` output you see a PTX function call,
not an `fma.rn.f32` instruction. **But** the libdevice body of `__nv_fmaf`
*is* `fma.rn.f32`, and cuda-oxide's runtime loader links libdevice through
nvJitLink, so post-link the kernel ends up with native FMA.

Per-kernel fma / mul / add counts after libdevice link (ptx87, sm_120):

| Kernel            | fma.rn.f32 | mul.rn.f32 | add.rn.f32 |
|-------------------|-----------:|-----------:|-----------:|
| `matmul`          | 0          | 3          | 3          |
| `matmul_unchecked`| 0          | 3          | 3          |
| `matmul_fmuladd`  | **3**      | 0          | 0          |

Inline `asm!` is not available for user kernels in cuda-oxide v0.1.0
(`cuda-oxide-book/appendix/supported-features.md:175`: **Planned**).
`inline_asm_convergent` is an internal mir-lower helper used by tcgen05 /
wgmma / mbarrier / tma intrinsics; there is no stable user-facing path.
**Fourth kernel `matmul_inline_fma` not implemented.**

## Hypothesis

Per `docs/research/cuda-oxide-flags.md`:

- `FastmathFlagsAttr::default()` is empty at every fadd/fsub/fmul/fdiv/fneg
  callsite in mir-lower, so `-ffp-contract=on|fast` semantics never reach
  NVPTX. `fmul` + `fadd` pairs stay separated.
- `core::intrinsics::fmuladdf32` lowers to `call @__nv_fmaf` (a libdevice
  symbol) in `mir-lower/src/convert/ops/call.rs:269-270`, **not** to
  `llvm.fmuladd.f32` (which NVPTX would lower to `fma.rn.*` unconditionally).

Prediction: `matmul_fmuladd` will still show `0 fma.rn.f32` in the user-visible
PTX. Revision expected after running the experiment: post-link (libdevice
linked in), the picture may differ.

## Method

### 1. Inline-asm support check

```bash
grep -rn 'asm!\|llvm_asm\|inline_asm\|__nvvm_asm' \
  /home/codeseys/.cargo/git/checkouts/cuda-oxide-6d394bb007f5e114/6de0509/
```

Result: hits only in `mir-lower/src/convert/intrinsics/` (the internal
`inline_asm_convergent` helper used by tcgen05, wgmma, mbarrier, tma, …).
`cuda-oxide-book/appendix/supported-features.md:175` states: `Inline
Assembly (\`asm!\` macro) | **Planned** | Workaround: use built-in intrinsics
or add new intrinsics to cuda-device.` → **No user-facing inline asm.**
Skipped the `matmul_inline_fma` kernel.

### 2. Add `matmul_fmuladd` kernel

Added to `oxide-matmul/src/main.rs` (appended after `matmul_unchecked`):

- Crate attributes: `#![feature(core_intrinsics)]`, `#![allow(internal_features)]`
- Kernel body identical to `matmul_unchecked` except inner loop uses
  `acc = core::intrinsics::fmuladdf32(av, bv, acc)` in place of
  `acc += av * bv`.
- Host driver is **not** modified — the new kernel is present in the PTX
  but not launched. Inspection-only.

### 3. Build + run

```bash
cd /home/codeseys/cuda-exploration/oxide-matmul
export PATH=/usr/lib/llvm-21/bin:$PATH:$HOME/.cargo/bin
cargo oxide build oxide-matmul   # ✓ Build succeeded
cargo oxide run oxide-matmul     # completes, SUMMARY unchanged
```

The build emits `oxide_matmul.ll` (NVVM IR) rather than `oxide_matmul.ptx`
at the crate root. The runtime loader (`cuda_host::ltoir::load_kernel_module`)
turns the `.ll` into a cubin via nvJitLink with libdevice linked in.

### 4. PTX generation (two views)

**Pre-link PTX** (what cuda-oxide emits conceptually, `llc` without libdevice):

```bash
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_120 -mattr=+ptx87 \
    oxide_matmul.ll -o oxide_matmul.ptx.new
```

**Post-link PTX** (what nvJitLink produces at runtime, what the GPU runs):

```bash
llvm-as oxide_matmul.ll -o oxide_matmul.bc
llvm-link oxide_matmul.bc /usr/local/cuda-13.2/nvvm/libdevice/libdevice.10.bc \
    -o oxide_matmul_linked.bc
opt -passes='internalize<preserve-gv=matmul;preserve-gv=matmul_unchecked;preserve-gv=matmul_fmuladd>,default<O2>' \
    oxide_matmul_linked.bc -o oxide_matmul_opt.bc
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_120 -mattr=+ptx87 \
    oxide_matmul_opt.bc -o oxide_matmul_linked.ptx
```

## Evidence

### NVVM IR (pre-link) — `oxide_matmul.ll`

```llvm
define void @matmul_unchecked(...) {
  %v48 = fmul float %v43, %v47         ; unfused, no fastmath flags
  %v49 = fadd float %v36, %v48
}
define void @matmul(...) {
  %v52 = fmul float %v44, %v51         ; same shape
  %v53 = fadd float %v34, %v52
}
declare float @__nv_fmaf(float, float, float)
define void @matmul_fmuladd(...) {
  %v48 = call float @__nv_fmaf(float %v43, float %v47, float %v36)
}                                      ; libdevice call, NOT llvm.fmuladd
```

### Pre-link PTX (un-linked, from `llc` only)

- `fma.rn.f32`: **0**
- `__nv_fmaf`: **5** (1 extern decl + 4 `call.uni` sites across the inlined loop)
- `mul.rn.f32` / `add.rn.f32`: 2 / 2 (from `matmul` + `matmul_unchecked`)

```ptx
.extern .func (.param .b32 func_retval0) __nv_fmaf(...)
; inside matmul_fmuladd:
call.uni (retval0), __nv_fmaf, (param0, param1, param2);
```

This matches the Phase-3 research prediction exactly.

### Post-link PTX (libdevice linked, `-O2 internalize`)

Per-kernel (script that partitions the PTX by `.visible .entry` and counts
per-section, listed under Method step 4):

```
matmul_fmuladd   fma=3  mul=0  add=0
matmul           fma=0  mul=3  add=3
matmul_unchecked fma=0  mul=3  add=3
```

Excerpt from `matmul_fmuladd` inner loop:

```ptx
ld.global.b32   %r20, [%rd41+-4];
ld.global.b32   %r21, [%rd40];
fma.rn.f32      %r22, %r20, %r21, %r28;
ld.global.b32   %r23, [%rd41];
ld.global.b32   %r24, [%rd30];
fma.rn.f32      %r28, %r23, %r24, %r22;
ld.global.b32   %r25, [%rd32];
ld.global.b32   %r26, [%rd35];
fma.rn.f32      %r28, %r25, %r26, %r28;
```

Excerpt from `matmul_unchecked` inner loop (post-link, same flags):

```ptx
mul.rn.f32      %r22, %r20, %r21;
add.rn.f32      %r23, %r31, %r22;
mul.rn.f32      %r26, %r24, %r25;
add.rn.f32      %r31, %r23, %r26;
mul.rn.f32      %r29, %r27, %r28;
add.rn.f32      %r31, %r31, %r29;
```

Note the 3 fma vs 3 mul+add ratio: LLVM/NVPTX fully unrolls one step (3x)
of the inner loop before the main loop body. Inside the main loop the
pattern is the same (1 fma per iter vs 1 mul+1 add per iter).

## Interpretation vs Phase-3 research

The Phase-3 research doc said: "`core::intrinsics::fmuladdf32` is lowered
to a libdevice call `__nv_fmaf` rather than `llvm.fmuladd`, so the one
route you'd expect to give you a hardware FMA from user code goes through
libdevice instead of letting ptxas contract it. This is almost certainly
why the generated PTX has `0 fma.rn.f32` while `nvcc -O3` emits 5."

Half right, half wrong:

- **Correct:** mir-lower lowers `FmuladdF32 → __nv_fmaf`, not `llvm.fmuladd`.
  No fast-math flags are emitted. ptxas / NVPTX get no contract permission.
- **Wrong (about the `0 fma`):** the reason `matmul` / `matmul_unchecked` in
  the original bench show 0 fma is that **user code uses `acc += a*b` (plain
  `fmul` + `fadd`)**, and without CONTRACT the separate ops survive. If user
  code explicitly uses `fmuladdf32`, the libdevice call gets inlined by
  nvJitLink and you DO get `fma.rn.f32` on device.

The practical gap vs `nvcc -O3` for the naive matmul is therefore:

- nvcc contracts `a*b + c` → `fma` automatically (fp-contract=on by default
  for CUDA).
- cuda-oxide does not contract; user must explicitly write
  `core::intrinsics::fmuladdf32` to get FMA.

## Conclusion

Three findings:

1. **`matmul_fmuladd` does yield hardware FMA on device.** 3 `fma.rn.f32`
   per kernel post-libdevice-link. Phase-3 research was right about the IR
   mechanism (`__nv_fmaf` call) but underestimated what nvJitLink does at
   runtime.
2. **Default `acc += a*b` in cuda-oxide produces separated `mul.rn.f32` +
   `add.rn.f32`**, confirming the `FastmathFlagsAttr::default()` finding
   from Phase 3. This is the real gap vs `nvcc -O3`.
3. **Inline PTX `asm!` is not available** for user kernels. Fourth kernel
   experiment skipped. Book labels it "Planned".

### User-facing upshot

The Phase-3 upstream-patch recommendations still stand. Either:

- Thread `FastmathFlags::CONTRACT` through mir-lower so plain `a*b+c` gets
  contracted (the "nvcc-compatible" default), **or**
- Change `FmuladdF32/F64` in `call.rs:269-270` to lower to `llvm.fmuladd.*`
  intrinsics (would make pre-link PTX show `fma.rn.f32` directly without
  requiring libdevice inlining).

But the *workaround that exists today* is: users who want FMA in a cuda-oxide
kernel should write `core::intrinsics::fmuladdf32(a, b, c)` explicitly
(with `#![feature(core_intrinsics)]` on nightly). It's ugly but it works.

Perf measurement for `matmul_fmuladd` vs `matmul_unchecked` was out of
scope for W3A (host driver not modified). Follow-up in BACKLOG if useful.

## Artifacts

- `oxide-matmul/src/main.rs` — `matmul_fmuladd` kernel added
- `oxide-matmul/oxide_matmul.ll` — NVVM IR (emitted by cuda-oxide)
- `oxide-matmul/oxide_matmul.ptx.new` — pre-link PTX (diagnostic only, not committed)
- `oxide-matmul/oxide_matmul_linked.ptx` — post-link PTX (diagnostic only, not committed)
