# Title: FMA contraction never emitted — `FastmathFlagsAttr::default()` blocks all fast-math bits in mir-lower

## Summary

Generated PTX from cuda-oxide kernels contains **zero** `fma.rn.f32` instructions, while equivalent `nvcc -O3` PTX contains many. Every fp op lowered in `crates/mir-lower/src/convert/ops/arithmetic.rs` attaches `FastmathFlagsAttr::default()` (= `FastmathFlags::empty()`), so ptxas/NVVM never gets the `contract` bit and can't fuse `fmul`+`fadd`. No CLI flag, env var, or `#[kernel(...)]` parameter changes this today. It's the largest compiler delta we measure vs CUDA C++.

## Repro

Repo: <https://github.com/baladithyab/cuda-exploration> @ `<COMMIT_HASH_TBD>`. Artifacts: `oxide-matmul/oxide_matmul.ptx`, `oxide-matmul-tiled/oxide_matmul_tiled.ptx` (cuda-oxide, `sm_120`); `cuda-matmul/matmul.ptx`, `cuda-matmul-tiled/matmul.ptx` (nvcc 13.2 `-O3 -arch=sm_120`); `docs/research/cuda-oxide-flags.md`; `results/scaling-summary.md`.

## Evidence

**PTX instruction counts** (`grep -o 'fma\.rn\.f32' … | wc -l`):

- `oxide-matmul/oxide_matmul.ptx`: **0**
- `oxide-matmul-tiled/oxide_matmul_tiled.ptx`: **0**
- `cuda-matmul/matmul.ptx` (nvcc `-O3`): **15**
- `cuda-matmul-tiled/matmul.ptx` (nvcc `-O3`): **256**

**Source evidence** (cuda-oxide checkout at commit `6de0509`):

- `crates/mir-lower/src/convert/ops/arithmetic.rs:97-102` — `add_fastmath_flags` always constructs `FastmathFlagsAttr::default()`.
- `crates/dialect-llvm/src/attributes.rs:121-124` — `impl Default for FastmathFlagsAttr` returns `FastmathFlags::empty()`.
- Other FMF callsites pass the same default: `arithmetic.rs:122, 148, 174, 200, 227, 542`; `cast.rs:453`; `dialect-llvm/src/ops/comparison.rs:209`.
- `crates/mir-lower/src/convert/ops/call.rs:269-270` — `FmuladdF32`/`FmuladdF64` (i.e. `f32::mul_add`, `core::intrinsics::fmuladdf32`) lower to libdevice `__nv_fmaf`/`__nv_fma` rather than `llvm.fmuladd.*`. **Empirically this DOES still produce hardware FMA** in the final SASS, because libdevice's `__nv_fmaf` body itself contains `fma.rn.f32` and nvJitLink resolves the call at module load. We verified this by inspecting post-link PTX of the `matmul_fmuladd` kernel (`docs/experiments/fma-toggle.md`); the kernel emits 3 `fma.rn.f32` instructions per iteration of the unrolled loop. So `core::intrinsics::fmuladdf32` IS a working escape hatch today. The remaining issue is that **default `*+` chains still don't fuse** because of the FastmathFlags::default() above — making explicit FMA the only path, when it should also be the default.

**Perf consequence** (RTX 5090, sm_120 native, `cudaEvent` timing, 10 iters, median):

- N=4096 naive SGEMM: cuda-oxide unchecked = **4.96 TFLOPS**, nvcc naive = **6.23 TFLOPS** → **0.80×**.
- N=4096 tiled SGEMM: oxide-tiled = **7.95 TFLOPS**, nvcc register-tiled = **28.07 TFLOPS** → **0.28×**. Not identical-kernel (nvcc adds a 4×4 register micro-tile), but PTX `fma.rn.f32` counts 0 vs 256 confirm a compiler-level gap independent of tile geometry.

## Proposed fix

Smallest useful patch is steps (1)+(2) from our research doc:

1. Thread a `LoweringOptions { fast_math: FastmathFlags }` through `Context` so `add_fastmath_flags` (`arithmetic.rs:98`), `cast.rs:453`, and the fneg site at `arithmetic.rs:542` emit non-empty FMF. Expose via `--fast-math`/`--fp-contract=on|fast|off` in `crates/cargo-oxide/src/main.rs` plus a `CUDA_OXIDE_FAST_MATH` env var forwarded in `commands.rs`. An MVP could set only `CONTRACT`, keeping IEEE semantics except FMA fusion.
2. Change `call.rs:269-270` so `FmuladdF32/F64` lower to `llvm.fmuladd.f32/f64` instead of `__nv_fmaf/__nv_fma`. NVPTX lowers `llvm.fmuladd` to `fma.rn.*` regardless of `-ffp-contract`, giving per-call opt-in via `a.mul_add(b, c)` without a global flag.

Optional follow-ups (per-kernel `#[kernel(fast_math)]`, `ReadOnlyPtr<T>` / `llvm.nvvm.ldg.global.*`) are in the research doc but out of scope here.

## Workaround question

Is there a path we missed — undocumented `RUSTFLAGS`, codegen flag, env var, or attribute that flips `CONTRACT` on today? Our grep over `cargo-oxide`, `rustc-codegen-cuda`, `mir-lower`, `cuda-macros`, the book, and README found nothing, but we'd rather hear "use X" than duplicate work.

## Repro commands

```bash
git clone https://github.com/baladithyab/cuda-exploration
cd cuda-exploration
cat docs/research/cuda-oxide-flags.md                          # full investigation
grep -o 'fma\.rn\.f32' oxide-matmul/oxide_matmul.ptx | wc -l   # -> 0
grep -o 'fma\.rn\.f32' cuda-matmul/matmul.ptx       | wc -l    # -> 15
grep -o 'fma\.rn\.f32' cuda-matmul-tiled/matmul.ptx | wc -l    # -> 256
```

## Author / context

Independent third-party benchmark. Evaluating cuda-oxide for a Rust-native GPU project; FMA is the largest compiler delta we measured vs hand-written CUDA C++. Happy to send a PR for (1) and/or (2) — want to confirm design preferences (context-threaded config vs per-function attribute vs env-var only) first.
