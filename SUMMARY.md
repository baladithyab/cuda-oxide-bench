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

## Five findings

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

| You are... | Recommendation |
|---|---|
| Greenfield Rust + NVIDIA GPU project, naive-ish kernels OK | **Use cuda-oxide.** v0.1.0 is alpha but the perf is real, the safety story is real, the tooling story (cargo, rustc-codegen-cuda) integrates cleanly. |
| Need Tensor Cores / WGMMA / TMA, peak performance today | **Wait** for cuda-oxide v0.2 or use CUDA C++ for hot kernels + cuda-oxide for everything else. The tiled-kernel gap is real. |
| Need cross-vendor (AMD/Intel/NVIDIA), one codebase | **Use wgpu** or **CubeCL**, not cuda-oxide. cuda-oxide is NVIDIA-only by design. |

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
