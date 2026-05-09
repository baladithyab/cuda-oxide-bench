# cuda-oxide-bench: how does NVlabs/cuda-oxide v0.1.0 compare to CUDA C++?

> **TL;DR (3 bullets, hedged per Phase 8 review):**
> 1. **At N=1024, cuda-oxide is statistically indistinguishable from nvcc on naive matmul** — all four naive kernels (oxide safe/unchecked/fmuladd, nvcc) hit 6.80-6.88 TFLOPS, within 1% of each other and within run-to-run noise. **Caveat:** at this size kernels run in 0.3 ms and ~5-50 µs of launch+timer overhead is a non-trivial fraction of measurement; this result is consistent with both "compiler equivalence" and "all kernels are launch-overhead-bound." At N=4096 (>20 ms kernel time) cuda-oxide naive is **0.85-0.91× of nvcc**, a real but modest gap.
> 2. **The "Rust safety tax" we initially measured at 2.5× was not real** — it was an artifact of the system `libnvvm.so.4` (libNVVM 7.0.1 from CUDA 12.0 era) shadowing the modern libNVVM 22.0.0 from CUDA 13.2. After setting `CUDA_HOME=/usr/local/cuda`, slice-indexed kernels run within ±5% of raw-pointer kernels at all N. Caveat: we changed both libNVVM version AND target arch (compute_89→compute_120) when fixing this; the ~2.5× delta is fully attributable to one of the two but our archived artifacts can't isolate which. The compile-time fix (libNVVM understanding compute_120) is rock-solid; the perf-quality attribution to libNVVM is correlation not causation.
> 3. **Tiling exposes a real, multi-cause gap.** nvcc-tiled (32×32 block + 4×4 register microtile + K-loop unroll) achieves 28.07 TFLOPS at N=4096; oxide-tiled (pure 16×16 SharedArray, 1 output per thread) achieves 7.91. Cause is **not** primarily FMA contraction — it's algorithm geometry first (register tiling), unrolling second, FMA fusion third. The cuda-oxide-side patch space includes all three. Working FMA escape hatch today: `core::intrinsics::fmuladdf32` lowers to libdevice's `__nv_fmaf` which contains hardware FMA, resolved by nvJitLink at module load. Doesn't change naive-matmul perf in our bench (the 4-instruction inner loop is too short to benefit).

## Methodology in one paragraph

We benchmark naive 4096×4096 f32 matrix-multiply (137.44 GFLOP/iter) at N ∈ {1024, 2048, 4096} across **8 (impl, kernel) configurations**: nvcc CUDA C++ (naive + register-tiled), cuBLAS sgemm with `CUBLAS_PEDANTIC_MATH` (no TF32), cuda-oxide naive (safe slice-indexed + unchecked raw-ptr + fmuladd), cuda-oxide tiled (safe + unchecked SharedArray). All builds target `sm_120` (Blackwell native) using CUDA 13.2. All timings via `cudaEventRecord` / `cuEventRecord` (ADR-0001) for kernel-only timing, with CPU wall-clock retained for visibility. 1 warmup + 10 timed iterations per config; report best, median, p95. Inputs deterministic: `(i % 7) * 0.01f` and `(i % 11) * 0.01f`. Correctness spot-checked at (0,0), (N/2, N/2), (N-1, N-1) per config per N. Detail in [`METHODOLOGY.md`](METHODOLOGY.md), [`docs/adrs/`](docs/adrs/), and per-folder `ANALYSIS.md`.

## Master results

TFLOPS (median, gpu_ms event-timed, RTX 5090 Blackwell, sm_120 native, CUDA 13.2):

| impl/kernel \ N | 1024 | 2048 | 4096 |
|---|---:|---:|---:|
| cublas-matmul/sgemm | **33.94** | **62.98** | **59.83** |
| cuda-tiled/matmul_tiled | 24.47 | 33.44 | 28.07 |
| oxide-tiled/unchecked | 9.09 | 7.22 | 7.91 |
| oxide-tiled/safe | 8.89 | 7.99 | 7.67 |
| **oxide/fmuladd** | **6.92** | **5.58** | **5.70** |
| **cuda-matmul/matmul** | **6.88** | **6.33** | **6.23** |
| **oxide/unchecked** | **6.95** | **5.81** | **5.13** |
| **oxide/safe** | **6.94** | **6.37** | **4.84** |

Full per-N tables in [`results/scaling-summary.md`](results/scaling-summary.md). Raw data in [`results/scaling.csv`](results/scaling.csv) (240 rows).

## Seven findings

### 1. cuda-oxide naive is within 1% of nvcc — at the right N

At N=1024, **all four naive kernels land in a 6.88-6.95 TFLOPS band, a <1% spread.** cuda-oxide's safe slice-indexed kernel (6.94), unchecked raw-pointer (6.95), `core::intrinsics::fmuladdf32` (6.92), and nvcc's CUDA C++ (6.88) are statistically indistinguishable. The "compiler quality gap" we expected to find for naive matmul is essentially absent at this size.

### 2. The "Rust safety tax" was an artifact

Initial benchmarks (v0 README) reported `oxide/safe` running 2.5× slower than `oxide/unchecked`. That number was generated with `libNVVM 7.0.1` (CUDA 12.0 era) which produced poor codegen for slice bounds-check predicates. After diagnosing the [libNVVM shadowing bug](docs/experiments/libnvvm-corrigendum.md) and forcing the modern `libNVVM 22.0.0` from CUDA 13.2, the safety tax collapses: at N=1024 the ratio is 1.00×, at N=2048 it's 0.91×, at N=4096 it's 1.06×. **Bounds-checked Rust slice indexing is essentially free** in correctly-configured cuda-oxide.

### 3. Tiling is where the compiler gap reopens — and widens

The picture changes when the algorithm uses shared-memory tiling. cuda-oxide's tiled kernel achieves 7.91 TFLOPS at N=4096; nvcc-tiled (with register micro-tiling on top of shared mem) achieves 28.07. The tiling speedup is **4.5× for nvcc but only 1.5× for cuda-oxide**. PTX inspection at [`oxide-matmul-tiled/oxide_matmul_tiled.ptx`](oxide-matmul-tiled/oxide_matmul_tiled.ptx) and [`cuda-matmul-tiled/matmul.ptx`](cuda-matmul-tiled/matmul.ptx) shows the cause: zero `fma.rn.f32` instructions in oxide vs 256 in nvcc, and no K-loop unrolling. The shared-memory mechanism itself works (`ld.shared.b32` and `bar.sync 0` are present); cuda-oxide just doesn't capitalize on the additional compute opportunity tiling exposes.

### 4. cuBLAS shows how much both naive baselines leave on the floor

cuBLAS sgemm with `CUBLAS_PEDANTIC_MATH` (strict f32, no TF32 path) hits 60-72 TFLOPS — about **10× our naive nvcc baseline.** This contextualizes the cuda-oxide vs nvcc comparison: both the C++ compiler and cuda-oxide leave ~90% of single-precision throughput unexposed without algorithmic optimization. The interesting gap is between cuda-oxide and nvcc-with-the-same-algorithm, which is where the FMA + unrolling story plays out.

### 5. wgpu can't reach NVIDIA on WSL2

The `wgpu-matmul/` folder documents what happens when you try to use the cross-vendor stack: only `llvmpipe` (Mesa CPU rasterizer) enumerates as a Vulkan adapter. The system has the NVIDIA driver passthrough at `/dev/dxg` and `nvidia_icd.json`, but the latter points to `libGLX_nvidia.so.0` (the OpenGL driver, not a Vulkan driver). DX12 backend on Linux/WSL needs `libdxcore`/`mesa-vulkan-drivers-microsoft` glue not packaged in Ubuntu 24.04. Net result: 0.005 TFLOPS at N=4096 on the CPU. **For Rust+GPU on WSL2 with NVIDIA hardware, cuda-oxide is the only working option.** See [`wgpu-matmul/ANALYSIS.md`](wgpu-matmul/ANALYSIS.md).

### 6. Memory-bound kernels reach parity; the matmul gap is a compute-side compiler quality issue

We added two new kernel classes in Wave 4 to test whether the matmul gap generalizes. It does not.

**Sum-reduction (1 GB input, warp-shuffle two-stage):** cuda-oxide hits 1451 GB/s vs nvcc's 1517 GB/s — **96% parity**, both at ~85% of the 1.79 TB/s HBM peak. See [`oxide-reduction/`](oxide-reduction/) and [`cuda-reduction/`](cuda-reduction/).

**Memory-bandwidth bench (3-buffer streaming `c = a + b`):** at the bandwidth-bound size N=64M, cuda-oxide-safe achieves 1608 GB/s vs nvcc's 1609 GB/s — **0.1% gap**, both at 90% of HBM peak. See [`oxide-vecadd-bench/`](oxide-vecadd-bench/) and [`cuda-vecadd-bench/`](cuda-vecadd-bench/).

The implication: the cuda-oxide matmul gap at N=4096 is **not** a memory-subsystem issue or a kernel-launch issue. It is specifically a compute-throughput / instruction-scheduling issue that manifests when the inner loop is FMA-heavy. Memory-bound code reaches parity; compute-bound code reveals the compiler delta.

### 7. SASS-level root cause: `LDG.E.CONSTANT` vs `LDG.E`, plus FMUL+FADD vs FFMA

Wave 5 went one level below PTX into the actual SASS instructions emitted by ptxas, using `cuobjdump --dump-sass`. The hot K-loop in our naive matmul:

| Metric                  | nvcc `matmul`      | oxide `matmul_unchecked` | oxide `matmul_fmuladd` |
|-------------------------|--------------------|---------------------------|-------------------------|
| FFMA per unrolled body  | **8**              | 0                         | **8**                   |
| FMUL + FADD             | 0                  | 8 + 8                     | 0                       |
| LDG count               | 16                 | 16                        | 16                      |
| LDG cache variant       | **LDG.E.CONSTANT** | LDG.E                     | LDG.E                   |
| K-loop unroll           | 8×                 | 8×                        | 8×                      |

Two findings, both new vs the PTX-level analysis from Wave 3:

**(a) Both compilers unroll the K-loop 8x.** The "missing unroll" hypothesis is rejected. cuda-oxide's loop unroller is doing its job.

**(b) nvcc emits `LDG.E.CONSTANT` (read-only / uniform-cache hint), cuda-oxide emits plain `LDG.E`.** This is the SASS confirmation of what the upstream FMA issue draft hinted at PTX level. The `__restrict__` + `const` annotations on nvcc's pointer args promote loads to the read-only cache path; cuda-oxide's NVPTX lowering doesn't emit this hint even when the slice is shared and immutable. Two tractable upstream patches: thread `FastmathFlags::CONTRACT` through mir-lower (fixes finding (b)'s related FMA issue from Wave 3), and emit `LDG.E.CONSTANT` for `&[T]` reads from non-overlapping disjoint slices.

See [`docs/experiments/sass-analysis.md`](docs/experiments/sass-analysis.md).

## Wave 6: cuda-oxide on consumer vs datacenter Blackwell

We tried to run three of cuda-oxide's bundled advanced examples on our RTX 5090 (sm_120, consumer Blackwell):

| Example | Build on sm_120 | Run on sm_120 | Notes |
|---|---|---|---|
| `tma_copy` | ✅ | ✅ **Works** | Real `cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes` instructions in PTX. TMA (Tensor Memory Accelerator) is supported on consumer Blackwell. See [`oxide-tma-copy/ANALYSIS.md`](oxide-tma-copy/ANALYSIS.md). |
| `gemm_sol` (cuda-oxide's flagship 868-TFLOPS-on-B200 example) | ✅ | ❌ **CUDA_ERROR_INVALID_PTX** | All 8 kernel variants depend on `tcgen05` (5th-gen Tensor Cores), which exists only on **datacenter Blackwell sm_100 / sm_100a**, not consumer sm_120. The example correctly detects this and prints a clear message. See [`oxide-gemm-sol/ANALYSIS.md`](oxide-gemm-sol/ANALYSIS.md). |
| `tcgen05_matmul` | ✅ | ❌ Same as above | Same reason. PTX has 64 `tcgen05.*` instructions; consumer 5090 has no TMEM and no tcgen05 hardware to execute them. See [`oxide-tcgen05-matmul/ANALYSIS.md`](oxide-tcgen05-matmul/ANALYSIS.md). |

**The Blackwell consumer/datacenter SM split matters for cuda-oxide today.** sm_120 (consumer: RTX 50-series) has 4th-gen Tensor Cores, FP4/FP8, and TMA — but no `tcgen05` instructions and no TMEM. cuda-oxide's flagship perf example only targets datacenter Blackwell (sm_100, B200/B100). For now, RTX 5090 owners can run cuda-oxide's basic + TMA examples but not the headline gemm_sol / tcgen05_matmul kernels. This is a hardware-feature gap, not a cuda-oxide gap — same constraint applies to writing tcgen05 in CUDA C++.

## Wave 4 W4C: causal isolation of the libNVVM finding (inconclusive)

The Phase 8 review flagged that the libNVVM corrigendum confounded two variables: libNVVM version (7.0.1 → 22.0.0) AND target arch (compute_89 → compute_120). We attempted to isolate them by forcing `CUDA_OXIDE_TARGET=sm_89` with the modern libNVVM. Result: modern libNVVM 22.0.0 **rejects rustc-codegen-cuda's IR for any arch ≤ sm_100** with `(13, 30): parse expected type` — a parse error on opaque pointers. The two variables are mechanically coupled in this toolchain; the experiment cannot be run as designed. **Implication for cuda-oxide:** the modern compiler frontend cannot target Ampere-or-earlier GPUs without rustc-codegen-cuda IR changes. This is a previously-undocumented portability constraint. See [`docs/experiments/libnvvm-causal-isolation.md`](docs/experiments/libnvvm-causal-isolation.md).

## Wave 7: closing the matmul gap (register microtile + fmuladd)

[`oxide-matmul-tiled-microtile/`](oxide-matmul-tiled-microtile/) — Wave 7 implements a 16×16-block + 4×4-register-microtile cuda-oxide tiled matmul. **Result: at N=1024 cuda-oxide hits 27-28 TFLOPS, matching or slightly exceeding nvcc-tiled (24.5 TF).** A 3.0-3.6× lift over the old oxide-tiled (9.1 TF). At N=4096 the gap halves to ~60% of nvcc (16-17 TF vs 28 TF), residual primarily due to thread-block geometry (nvcc uses 32×32+4×4 = 128×128 output/block; cuda-oxide here is 16×16+4×4 = 64×64).

**Surprise**: libNVVM **does** contract `*+` to FFMA in tiled kernels (128 FFMAs in `_safe`), and in 3DGS too. Wave 3's "FastmathFlags::empty() blocks contraction" finding was specific to runtime-bounded loops — not universal. Re-investigation pending.

## Wave 8: rudimentary 2D Gaussian Splatting in cuda-oxide

[`oxide-3dgs-mini/`](oxide-3dgs-mini/) — Wave 8 ports a forward-only 2D Gaussian Splatting rasterizer to cuda-oxide as a complexity stress-test. 256×256 image, 512 gaussians, per-pixel kernel iterates all gaussians with front-to-back alpha-blend and early-exit on transmittance < 1e-4. **Builds and runs cleanly: 75 µs/frame, ~9 TFLOPS effective on the work that ran.**

Verdict: **cuda-oxide handled the kernel cleanly.** 12-arg kernel signature with mixed `&[f32]` / `DisjointSlice<f32>` / `u32`, `core::intrinsics::expf32` lowers through libdevice to hardware MUFU.EX2, multi-branch early-exit produces sane SASS. No API friction. Wave 8.5 added two procedural test scenes (concentric rings, smiley face) for visual sanity-check; both render correctly via the unchanged kernel.

## Wave 9-10: real public 3DGS scenes through the cuda-oxide pipeline

[`oxide-3dgs-real/`](oxide-3dgs-real/) extends the toy renderer with a full PLY parser and 3D→2D projection so it can consume real public 3DGS data:

- **Wave 9** rendered the 14,526-gaussian Luigi figurine (`dylanebert/3dgs/luigi.ply`, 988 KB) at ~10 ms/frame. Standard pipeline: quaternion → R, `Σ_3d = R · diag(s²) · Rᵀ`, perspective Jacobian → `Σ_2d`, conic = inv(Σ_2d), SH-degree-0 color = `0.28209479 · f_dc + 0.5`, sigmoid opacity, depth sort by camera-space z.
- **Wave 10** rendered the canonical 53,671-gaussian Utsuho plush scene (`solaaaa/sample-gaussian-splats`, 13 MB, SH degree 3 .ply but rendering at SH degree 0). Recognizable chibi character figurine at ~37 ms/frame.

The kernel is byte-identical to the Wave 8 toy — only the host-side scene generator changed. Validates that cuda-oxide can drive a non-trivial real-world graphics pipeline end-to-end.

## Wave 11: byte-identical pixels in cuda-oxide and CUDA C++ on the same 3DGS scene

The strongest single result in the repo. We ported the 3D Gaussian Splatting forward rasterizer to nvcc CUDA C++ at [`cuda-3dgs-real/`](cuda-3dgs-real/) as an apples-to-apples reference. Same PLY parser, same camera, same kernel algorithm — line-by-line port to a single .cu file.

**Pixel-level result on Wave 10's Utsuho scene (53,671 gaussians, 800×800):**

| Camera | cuda-oxide md5 | nvcc md5 | Diff |
|---|---|---|---|
| A (canonical) | `9f45b235168305e4b3dad2abe8f50db4` | `9f45b235168305e4b3dad2abe8f50db4` | **byte-identical** |
| C | `e3a3fd9056f3da3f1c512c7a268b777e` | `e3a3fd9056f3da3f1c512c7a268b777e` | **byte-identical** |
| D | (different) | (different) | **3 of 640,000 pixels differ by 1 intensity level** |

Cam A and C are bit-identical; cam D shows sub-ULP clang-vs-rustc FMA reordering on three pixels.

**SASS-level comparison (kernel body only):**

| Instruction | nvcc | cuda-oxide |
|---|---:|---:|
| FFMA | 9 | 9 |
| FMUL | 9 | 9 |
| FADD | 5 | 5 |
| MUFU (for `expf`) | 1 | 1 |
| LDG.E.CONSTANT | **9** | 0 |
| LDG.E (plain) | 0 | **9** |

Arithmetic mix is **identical**. The only SASS-level difference is the read-only-cache hint — exactly the Wave 5 LDG.E.CONSTANT finding that drove the matmul-tiled gap. On this kernel it doesn't move runtime (kernel is MUFU/branch-bound, not memory-bound).

**Kernel timing** (3 iters, 800×800, gaming concurrent on the same GPU):
- nvcc: 36.5–42.0 ms median
- cuda-oxide: 37.1–42.0 ms median
- ±15% noise band, no winner

**Verdict.** On the Utsuho 3DGS rasterizer, cuda-oxide produces byte-identical pixels and arithmetically-identical SASS. The matmul-scale codegen gap **does not generalize** to splat rasterization. The only SASS-level delta is the LDG.E.CONSTANT hint, which is a single-line patch on the upstream NVPTX lowering side and doesn't currently affect runtime on this workload. **This is the single most concrete piece of evidence in the study that cuda-oxide is production-viable for non-trivial real-world workloads.**

## Compiler gap deep-dive

Two separate effects compound to produce the cuda-oxide vs nvcc gap:

**(a) Default fast-math flags are empty.** `crates/dialect-llvm/src/attributes.rs:121-124` defines `FastmathFlags { NNAN, NINF, NSZ, ARCP, CONTRACT, AFN, REASSOC, FAST }` and `crates/mir-lower/src/convert/ops/arithmetic.rs:97-102` calls `add_fastmath_flags` on every `fadd/fsub/fmul/fdiv/frem/fneg`, but **every callsite passes `FastmathFlagsAttr::default()` (= `FastmathFlags::empty()`)**. Without `CONTRACT`, ptxas/NVVM cannot fuse `fmul`+`fadd` chains into `fma.rn.f32`. There is no CLI flag, env variable, or `#[kernel(...)]` parameter to enable this today.

**(b) `fmuladd` lowers to libdevice, not `llvm.fmuladd`.** `crates/mir-lower/src/convert/ops/call.rs:269-270` lowers `FmuladdF32`/`FmuladdF64` (i.e. `f32::mul_add`, `core::intrinsics::fmuladdf32`) to a call to libdevice's `__nv_fmaf`/`__nv_fma`. **However**, libdevice's `__nv_fmaf` body itself contains `fma.rn.f32`, so when nvJitLink resolves the call at module-load time, the final SASS does contain hardware FMA. We verified this in [`docs/experiments/fma-toggle.md`](docs/experiments/fma-toggle.md). So `core::intrinsics::fmuladdf32` IS a working escape hatch today — the `oxide/fmuladd` kernel demonstrates this. The default `*+` codegen still doesn't fuse, which is the larger missed opportunity.

We've drafted an upstream issue at [`docs/upstream-issue-fma.md`](docs/upstream-issue-fma.md) detailing both findings and proposing a 2-line patch to thread a `FastmathFlags` config through mir-lower. The user submits manually.

## Setup gotchas you'll hit

1. **`/usr/bin/nvcc` may be a stale apt-package shim** even if you have CUDA 13.2 at `/usr/local/cuda`. The system shim doesn't recognize `-arch=sm_120`. Always use `/usr/local/cuda/bin/nvcc` explicitly. See [`docs/adrs/0002-native-sm120.md`](docs/adrs/0002-native-sm120.md).
2. **The libNVVM shadow bug.** cuda-oxide's `libnvvm-sys` tries `libnvvm.so.4` against the system loader before falling back to `CUDA_HOME`. If you have an older `libnvvm.so.4` from a previous apt CUDA package at `/usr/lib/x86_64-linux-gnu/libnvvm.so.4`, you'll silently use stale codegen. Always set `CUDA_HOME=/usr/local/cuda` and `LIBNVVM_PATH=/usr/local/cuda/nvvm/lib64/libnvvm.so` before any `cargo oxide` invocation. See [`docs/experiments/libnvvm-corrigendum.md`](docs/experiments/libnvvm-corrigendum.md).
3. **LLVM 21 with NVPTX target is required**, not LLVM 18 or 19. cuda-oxide pins to LLVM 21+ explicitly. Install via `wget -qO- https://apt.llvm.org/llvm.sh | sudo bash -s -- 21 all`.
4. **Rust nightly with `rust-src`, `rustc-dev`, `llvm-tools-preview` components.** Your project's `rust-toolchain.toml` (auto-generated by `cargo oxide new`) pins a specific date; install that exact nightly.
5. **wgpu/WGSL on WSL2 falls back to CPU.** No fix in this benchmark; documented in `wgpu-matmul/ANALYSIS.md`.

Full setup at [`SETUP.md`](SETUP.md).

## For Rust developers considering cuda-oxide today

It works. It works well enough that the headline performance number for naive kernels is **within 1% of CUDA C++**, which was not what we expected going in. The remaining gap is in tiled / compute-bound kernels where missing FMA-contraction-by-default costs 4-5× of the achievable speedup. Until that lands upstream, you can route around it via `core::intrinsics::fmuladdf32` for hot inner loops.

| You are... | GPU class | Recommendation |
|---|---|---|
| Greenfield Rust + NVIDIA project, naive-ish kernels, memory-bound work, reductions | Any sm_100+ Blackwell, sm_90 Hopper | **Use cuda-oxide.** v0.1.0 is alpha but on memory-bound work it hits parity with CUDA C++; on compute-bound naive kernels it's within 1% at small N and 85-90% at large N; on reductions it hits 96% of nvcc. The safety story is real, cargo/rustc-codegen-cuda integrates cleanly. |
| Want to use cuda-oxide's flagship `gemm_sol` (868 TFLOPS-class TMA + tcgen05 pipeline) | sm_100 / sm_100a only (B200, B100, datacenter Blackwell) | **Hardware-gated.** Consumer RTX 50-series (sm_120) lacks tcgen05 / TMEM and cannot run the headline kernels. TMA itself works on sm_120; tcgen05 does not. |
| Need Tensor Cores / WGMMA / TMA, peak performance today | sm_90+ but not on cuda-oxide's most advanced path | **Wait** for cuda-oxide v0.2 or use CUDA C++ for hot kernels + cuda-oxide for everything else. The tiled-kernel + LDG-cache gap is real (see Wave 5 SASS analysis). |
| Targeting Ampere or earlier (sm_86, sm_80) | sm_89 and below | **Cannot use cuda-oxide today.** Modern libNVVM 22.0.0 rejects rustc-codegen-cuda's IR for any arch ≤ sm_100 (Wave 4 W4C). Stuck with CUDA C++ until upstream fix. |
| Need cross-vendor (AMD/Intel/NVIDIA), one codebase | Any | **Use wgpu** or **CubeCL**, not cuda-oxide. cuda-oxide is NVIDIA-only by design. |

## What's next (followups)

From [`BACKLOG.md`](BACKLOG.md), items deferred:

- **U1**: Upstream issue submission to `NVlabs/cuda-oxide` (draft ready at `docs/upstream-issue-fma.md`; needs human signoff to submit)
- **N1**: Block-size sensitivity sweep (8×8, 16×16, 32×8, 32×16, 32×32) on naive matmul. Likely small effect; documented as low-priority.
- **N2**: Reduction kernel as a second algorithm class (different access pattern than matmul; tests warp-reduce primitives)
- **N3**: GitHub Actions CI — needs self-hosted GPU runner; not a fit for a public bench repo today.
- **Tensor Core / WGMMA path** for cuda-oxide once the API stabilizes.
- **Bare-metal Linux re-bench** to settle the wgpu story.

## Acknowledgments / disclaimer

- **NVlabs cuda-oxide team** for shipping v0.1.0 and the deep architecture documentation that made the FastmathFlags investigation tractable.
- **NVIDIA CUDA team** for the toolkit + libnvjitlink that resolves libdevice's `__nv_fmaf` to hardware FMA.
- **gfx-rs/wgpu** for the cross-vendor Rust+GPU compute story even where it can't run on WSL2 today.

This is independent third-party work. We are not affiliated with NVlabs, NVIDIA, gfx-rs, or any vendor. Single-machine benchmark on RTX 5090 (Blackwell, sm_120) under WSL2 Ubuntu 24.04. Different hardware, different drivers, different CUDA toolchain versions may produce materially different results — please rerun if claims of yours depend on the numbers here.
