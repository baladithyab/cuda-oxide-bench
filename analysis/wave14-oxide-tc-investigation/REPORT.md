# Wave 14.4 — cuda-oxide tensor-core API surface on RTX 5090 (sm_120)

**Upstream pinned:** `cuda-oxide` v0.1.0 @ commit `44abb0717610f5420f98bebee74c27be3a2c186b`
(NVlabs/cuda-oxide, this is what every `cuda-exploration/*` crate's `Cargo.lock`
resolves to).

**Question:** Can we add an `oxide-matmul-tiled-tf32/` (or any half-precision
TC) baseline on RTX 5090 sm_120 today using the cuda-oxide v0.1.0 APIs?

**TL;DR verdict:** **No.** cuda-oxide v0.1.0 exposes exactly two tensor-core
paths — `cuda_device::wgmma::*` (Hopper-only, sm_90) and
`cuda_device::tcgen05::*` (datacenter Blackwell, sm_100a-only). Neither
targets the **consumer Blackwell** tensor-core ISA (`mma.sync.*` /
`HMMA.16816.F32`) that sm_120 requires. There is **no Ampere/Ada/sm_120-style
MMA wrapper** in the device crate, and cuda-oxide does not support inline
`asm!` in kernel code. An `oxide-matmul-tiled-tf32` baseline would require
adding a new `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` lowering
path upstream.

---

## 1. cuda-oxide TC API surface as of v0.1.0

Device-side module list (from
`crates/cuda-device/src/lib.rs`):

```
atomic  barrier  clc  cluster  cooperative_groups  cusimd  debug
disjoint  fence  grid  shared  tcgen05  thread  tma  warp  wgmma
```

Only `wgmma` and `tcgen05` are tensor-core modules. The full cuda-oxide tree
contains **zero** references to `mma.sync`, `mma.m16n8*`, `mma.m16n16*`,
`ldmatrix`, or HMMA:

```
$ grep -rn 'mma\.sync\|mma\.m16n8\|mma\.m16n16\|mma\.m8n8\|ldmatrix' \
    /home/codeseys/.cargo/git/checkouts/cuda-oxide-*/44abb07/crates/
# (no output)
```

### 1.1 `cuda_device::wgmma` — Hopper sm_90 only (329 lines)

Exports `make_smem_desc`, `wgmma_fence`, `wgmma_commit_group`,
`wgmma_wait_group::<N>()`. The example `examples/wgmma/src/main.rs` only
exercises the sync primitives and writes the SMEM descriptor back to a u64
buffer — it does **not** actually issue a `wgmma.mma_async`. PTX lowering is
in `crates/mir-lower/src/convert/intrinsics/wgmma.rs`, which emits inline
PTX strings like `wgmma.fence.sync.aligned;` and notes
`"// wgmma.mma placeholder"` at line 141 — i.e. the MMA issue path is
**stubbed**, not wired. Even on Hopper hardware, you cannot yet do a full
WGMMA matmul through cuda-oxide's public API.

### 1.2 `cuda_device::tcgen05` — datacenter Blackwell sm_100a only (2,285 lines)

This is the one rich MMA wrapper in the crate. Public surface (from
`crates/cuda-device/src/tcgen05.rs`):

| Category | Items |
|---|---|
| Types | `TensorMemoryHandle`, `TmemAddress`, `TmemGuard<State, N>`, `TmemUninit`, `TmemReady`, `TmemDeallocated`, `TmemF32x4`, `TmemF32x32` |
| Descriptors | `Tcgen05InstructionDescriptor` (+ builder), `Tcgen05SmemDescriptor` (+ builder), `Tcgen05MmaShape`, `Tcgen05ElementType { F16, BF16, TF32 }`, `Tcgen05AccumulatorType { F16, F32 }`, `Tcgen05SwizzleMode`, `CollectorUsage` |
| TMEM alloc | `tcgen05_alloc`, `tcgen05_dealloc`, `tcgen05_relinquish_alloc_permit` (+ `_cg2` variants) |
| Sync | `tcgen05_fence_before_thread_sync`, `tcgen05_fence_after_thread_sync`, `tcgen05_commit`, `tcgen05_commit_shared_cluster`, `tcgen05_load_wait`, `tcgen05_store_wait` |
| MMA | `tcgen05_mma_f16`, `tcgen05_mma_ws_f16`, `tcgen05_mma_ws_bf16`, `tcgen05_mma_ws_tf32`, `tcgen05_mma_ws_f16_with_collector`, `tcgen05_mma_f16_cg2` |
| TMEM↔register/SMEM | `tcgen05_ld_16x256b_pure`, `tcgen05_ld_16x256b_x8_pure`, `tcgen05_cp_smem_to_tmem`, `stmatrix_m8n8_x2/x4` (+ `_trans`) |
| Helpers | `cvt_f32x2_bf16x2`, `f32_to_bf16_rne`, `pack_bf16_pair`, `pack_f16_pair`, `f32_pair_to_packed_bf16` |

Every MMA-issuing function's doc-comment says explicitly:

> Must be called from within a CUDA kernel context on **sm_100a+**.

The emitted PTX carries `.target sm_100a` (verified against
`oxide-tcgen05-matmul/tcgen05_matmul.ptx:6`). Codegen is real — the
`mir-lower/src/convert/intrinsics/tcgen05.rs` file wires each intrinsic to
an inline `tcgen05.*` PTX asm template.

### 1.3 How intrinsics get lowered (no user-facing `asm!`)

Per the `rust-gpu-compute` skill and confirmed here: `asm!` is not supported
inside cuda-oxide `#[kernel]` functions in v0.1.0. The path is:

1. User calls e.g. `tcgen05_mma_f16(...)` in Rust.
2. `cuda-device/src/tcgen05.rs` function body is `unreachable!(...)` — this
   is a **fake body** for host-side compilation.
3. The codegen backend (`rustc-codegen-cuda`) recognises the symbol and
   substitutes a `nvvm.tcgen05_*` MIR op.
4. `mir-lower/src/convert/intrinsics/tcgen05.rs` lowers the op to an inline
   PTX asm string inside the emitted LLVM IR.
5. libNVVM compiles IR → PTX; the final `.target sm_100a` directive forces
   the driver to reject the module on non-sm_100a devices.

So **the only way to get tensor-core PTX out of cuda-oxide today is via the
`cuda_device::wgmma` or `cuda_device::tcgen05` modules**. There is no
"escape hatch" in kernel code.

## 2. Hardware-class gating — what runs on sm_120 vs sm_100

| Module        | Builds on sm_120 | Runs on sm_120 | Runs on sm_100a |
|---------------|------------------|----------------|-----------------|
| `tma`         | ✅ yes           | ✅ yes (sm_90+) | ✅ yes          |
| `barrier`     | ✅ yes           | ✅ yes          | ✅ yes          |
| `cluster`     | ✅ yes           | ✅ yes          | ✅ yes          |
| `wgmma`       | ✅ yes           | ❌ no (sm_90 only) | ❌ no (Hopper-only) |
| `tcgen05`     | ✅ yes (PTX `.target sm_100a`) | ❌ `CUDA_ERROR_INVALID_PTX` | ✅ yes |

Verified on this RTX 5090 box against existing artifacts (no new GPU work
required):

- `oxide-tcgen05-matmul/build.log` → build succeeds in 58s, emits
  `.target sm_100a` PTX.
- `oxide-tcgen05-matmul/run.log` → host code detects `sm_120`, short-
  circuits before `cuModuleLoad`. Wave-6 `ANALYSIS.md` documents that
  bypassing the guard (or passing `--arch=sm_100a`) produces
  `CUDA_ERROR_INVALID_PTX` at module-load time — the SASS encoder on
  Blackwell-consumer does not recognise the `tcgen05.*` opcodes.
- `oxide-gemm-sol/run.log` → same story, identical guard + failure mode
  across all 8 speed-of-light kernel variants.

The wgmma example never reaches MMA execution because the issue path is
stubbed; it exits after the SMEM-descriptor write on any hardware.

## 3. Verdict — can we add an `oxide-matmul-tiled-tf32` baseline on RTX 5090 today?

**No.** Concretely:

1. RTX 5090 is **sm_120**. The TC instruction class it accepts is
   `mma.sync.aligned.m16n8k*` (classic Ampere/Ada/Blackwell-consumer
   tensor cores, SASS `HMMA.16816.F32` / `HMMA.16816.F32.BF16`).
2. cuda-oxide v0.1.0 has **zero** wrappers, intrinsics, or lowerings for
   that instruction family. The one TF32 function it exposes
   (`tcgen05_mma_ws_tf32`) targets the sm_100a TMEM-based 5th-gen MMA, a
   different instruction class entirely.
3. `asm!` is not permitted in kernel code, so we cannot work around the
   gap with inline PTX inside a `#[kernel]` function.
4. `wgmma` is stubbed (no MMA issue wired), and it is sm_90-only anyway.

The existing `oxide-matmul-tiled-microtile` result (45 TFLOPS f32, all
CUDA cores, zero TC engagement — confirmed by Wave-13 SASS: only `FFMA` in
the hot loop, no HMMA) is therefore **very likely the ceiling for cuda-oxide
on sm_120 using only v0.1.0 APIs**.

**What would unblock a TC baseline on sm_120:**

- (A) An `mma.sync` wrapper upstream. Concretely: a new
  `cuda_device::mma` (or `cuda_device::mma_sync`) module exposing
  `mma_sync_m16n8k16_f16`, `mma_sync_m16n8k8_tf32`, `ldmatrix_*`,
  `cp_async` (Ampere-generation), mirroring the tcgen05 module shape
  but targeting sm_80+ / sm_120 without `.target sm_100a`. Would need
  matching inline-PTX lowering in `mir-lower/src/convert/intrinsics/`.
- (B) Or: user-facing inline `asm!` support inside `#[kernel]` functions.
  This is a rustc-codegen-cuda feature gap; once present, a user could
  hand-roll `mma.sync` without upstream API changes.
- (C) Or: wait for cuda-oxide to wire the cuTile-style `ct.mma` tile DSL
  through its MIR layer (there is no sign of this in v0.1.0; cuTile is a
  cuTile-PTX-frontend feature, not a MIR-level concept).

None of (A)/(B)/(C) is a half-day fix; the cleanest is (A), a ~1-2k-line
upstream contribution mirroring the structure of `tcgen05.rs` (and sharing
its codegen machinery).

## 4. Contrast with cuTile on the same sm_120 hardware

cuTile (`cutile-matmul-tiled-mixed` in this repo) emits the following SASS
on sm_120 (verified from `cutile-matmul-tiled-mixed/*.sass`):

| cuTile dtype combo         | Hot-loop TC instruction     | Instruction class                        |
|----------------------------|-----------------------------|-------------------------------------------|
| f16 × f16, f32 accumulator | `HMMA.16816.F32`            | consumer-Blackwell `mma.sync.m16n8k16`    |
| bf16 × bf16, f32 acc       | `HMMA.16816.F32.BF16`       | same class, bf16 variant                  |
| f32 × f32, f32 acc         | `HFMA2` (no tensor core)    | CUDA cores; cuTile does not auto-promote f32→TF32 |

Wave 13 measured the f16 case at **172.5 TFLOPS** at N=4096 — clearly TC-
engaged. The f32 case in Wave 13 (`cutile_matmul_tiled.sass`) also shows
only `HFMA2`, confirming cuTile does not engage TC without an explicit low-
precision dtype.

**What this tells us about the two approaches:**

- **cuTile is a Python-hosted tile DSL** that emits its own PTX with
  `.target sm_120a` and **knows the consumer-Blackwell MMA encoding**. Its
  `ct.mma` operator resolves to `HMMA.16816.F32` on sm_120 and presumably
  to `tcgen05.mma` on sm_100a — the dtype + target combo picks the right
  encoding.
- **cuda-oxide is a rustc backend + a thin Rust-side wrapper layer.** The
  Rust wrapper only exposes the wrappers that upstream chose to add. They
  added Hopper (`wgmma`) and datacenter-Blackwell (`tcgen05`) because that
  is what NVIDIA's tcgen05 example corpus targets. The **entire
  sm_80/sm_89/sm_90a-consumer/sm_100/sm_120a `mma.sync` family** is
  missing from v0.1.0 — not because of any codegen limitation (libNVVM
  supports it fine, and the mir-lower machinery could lower to inline PTX
  the same way tcgen05 does) but because **no one has written the wrapper
  yet.**
- So the gap is a **surface-area gap, not a capability gap.** The libNVVM
  backend, the PTX emitter, and the driver can all produce and load
  `mma.sync` PTX on sm_120 today. What is missing is a ~1-2k-line Rust
  module (and its MIR lowering) mirroring `tcgen05.rs` but targeting the
  classic MMA ISA. Until that lands upstream (or until `asm!` is allowed
  in kernels), cuda-oxide is strictly a **CUDA-core-only** matmul tool on
  RTX 5090, while cuTile is a full TC tool.

This is the cleanest one-sentence framing: **cuTile's DSL design made TC
engagement part of the core abstraction; cuda-oxide's wrapper-per-ISA
design means TC engagement is gated on each ISA family having a hand-
written Rust wrapper, and the consumer-Blackwell wrapper does not exist
in v0.1.0.**

## 5. Recommendation

Do **not** add an `oxide-matmul-tiled-tf32` axis to the comparison in a
follow-up wave targeting v0.1.0. The underlying primitive is not
accessible from Rust. Instead:

1. Write a negative-result note in the SUMMARY: "cuda-oxide v0.1.0 has no
   sm_120 TC wrapper; cuTile vs oxide-tiled-microtile is a tensor-core-
   vs-CUDA-core comparison, not apples-to-apples."
2. File an upstream feature request for a `cuda_device::mma` module
   covering `mma.sync.aligned.m16n8k{8,16}` (tf32, f16, bf16 variants
   minimum) — point at the tcgen05 module as the structural template.
3. If/when upstream adds inline `asm!` for kernels (issue-tracker
   recommended), revisit; a ~300-line crate could then wrap the needed
   PTX by hand.

## Appendix — sources cited

- `cuda-device/src/lib.rs` — module list
- `cuda-device/src/tcgen05.rs` — TC API (lines 460, 612, 645, 1450, 1490, 1504, 1526, etc.)
- `cuda-device/src/wgmma.rs` — Hopper API (329 lines total)
- `mir-lower/src/convert/intrinsics/tcgen05.rs` — inline-PTX lowering
- `mir-lower/src/convert/intrinsics/wgmma.rs:141` — "// wgmma.mma placeholder"
- `rustc-codegen-cuda/examples/tcgen05_matmul/src/main.rs` — upstream usage example
- `rustc-codegen-cuda/examples/wgmma/src/main.rs` — Hopper-only, verify-PTX fallback on non-Hopper
- `oxide-tcgen05-matmul/tcgen05_matmul.ptx:6` — `.target sm_100a`
- `oxide-tcgen05-matmul/ANALYSIS.md` — Wave 6 runtime verification (`CUDA_ERROR_INVALID_PTX` on sm_120)
- `oxide-gemm-sol/ANALYSIS.md` — same, all 8 gemm_sol variants
- `cutile-matmul-tiled-mixed/mma_f16xf16_f32acc.sass` — cuTile `HMMA.16816.F32` on sm_120a
- `analysis/wave13-sass/cutile_matmul_tiled.sass` — cuTile f32 case is `HFMA2` only
- `analysis/wave13-sass/oxide_matmul_tiled_microtile.sass` — oxide microtile is `FFMA` only
