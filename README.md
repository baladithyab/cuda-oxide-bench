# cuda-oxide-bench

A fair-comparison benchmark of [**cuda-oxide**](https://github.com/NVlabs/cuda-oxide) — NVIDIA's brand-new Rust-to-PTX compiler (v0.1.0, released 2026-05-07) — against raw **CUDA C++** (nvcc) and **wgpu/WGSL**, running the identical naive 4096×4096 f32 matrix-multiply algorithm on an RTX 5090 (Blackwell, sm_120) under WSL2. The goal is not to crown a winner, but to quantify how much performance a Rust-first CUDA frontend costs today, where the cost comes from, and where it matches or closes on the C++ baseline. Every kernel implements the same inner loop (`for k in 0..dim: C[r,c] += A[r,k] * B[k,c]`) with the same 16×16 thread block; no shared-memory tiling, no Tensor Cores, no cuBLAS — just the compiler and the driver.

## Headline results

Full scaling sweep — 7 `(impl, kernel)` configurations × N ∈ {1024, 2048, 4096} × 10 iters, `cudaEvent` timed, nvcc 13.2 native `-arch=sm_120`.

**TFLOPS vs N (median of 10 iterations):**

| impl/kernel \ N | 1024 | 2048 | 4096 |
|---|---:|---:|---:|
| cublas-matmul/sgemm | **33.94** | **62.98** | **59.83** |
| cuda-tiled/matmul_tiled | 24.47 | 33.44 | 28.07 |
| oxide-tiled/unchecked | 9.09 | 7.22 | 7.91 |
| oxide-tiled/safe | 8.89 | 7.99 | 7.67 |
| oxide/fmuladd | 6.92 | 5.58 | 5.70 |
| cuda-matmul/matmul | 6.88 | 6.33 | 6.23 |
| oxide/unchecked | 6.95 | 5.81 | 5.13 |
| oxide/safe | 6.94 | 6.37 | 4.84 |

**Key ratios (median):**

- **cuda-oxide naive vs nvcc CUDA C++:** at N=1024, all four naive kernels (oxide safe/unchecked/fmuladd, nvcc) land within 1% of each other (6.88-6.95 TFLOPS). The compiler is 99% of the way there for naive kernels.
- **Rust safety tax:** ~0% at N=1024, varying to a small effect at larger N driven by thermal noise. The 2.5× initially reported in v0 was a libNVVM environment artifact (see [`docs/experiments/libnvvm-corrigendum.md`](docs/experiments/libnvvm-corrigendum.md)).
- **Tiling speedup (tiled / naive):** nvcc gets 3.6×/5.3×/4.5×; cuda-oxide gets 1.3×/1.2×/1.6× — compiler gap widens with tiling (missing default-FMA contraction + K-loop unroll).
- **wgpu/WGSL on WSL2:** falls back to llvmpipe CPU, ~1000× slower; boundary is WSL2 Vulkan passthrough, not wgpu.

**→ See [SUMMARY.md](SUMMARY.md) for the full writeup**, or [`results/scaling-summary.md`](results/scaling-summary.md) for per-size best/median/p95 tables. The libNVVM correction is in [`docs/experiments/libnvvm-corrigendum.md`](docs/experiments/libnvvm-corrigendum.md). PTX-level deltas in per-folder `ANALYSIS.md` files. SASS-level deep-dive in [`docs/experiments/sass-analysis.md`](docs/experiments/sass-analysis.md).

## Wave 7-8: closing the gap + 3D Gaussian Splatting

- **Wave 7 — register microtile + fmuladd:** [`oxide-matmul-tiled-microtile/`](oxide-matmul-tiled-microtile/) implements a 4×4 register microtile in cuda-oxide. **At N=1024: 27-28 TFLOPS, matching nvcc-tiled (24.5 TF).** At N=4096: 16-17 TF vs nvcc 28 TF (60%, gap halved from old oxide-tiled).
- **Wave 8 — rudimentary 3DGS rasterizer:** [`oxide-3dgs-mini/`](oxide-3dgs-mini/) ports a forward 2D Gaussian Splatting rasterizer (256×256 image, 512 gaussians, alpha-blend with early-exit) to cuda-oxide. Builds and runs at 75 µs/frame. cuda-oxide handles a complex 12-arg kernel cleanly. Rendered PPM saved.

**Notable surprise**: libNVVM **does** contract plain `*+` to FFMA in some kernels (3DGS's per-pixel kernel; the `_safe` variant in Wave 7's tiled-microtile). Wave 3's "FastmathFlags::empty() blocks contraction" finding may be specific to runtime-bounded inner loops, not universal. Worth re-investigating.

## Wave 4-6 follow-up findings (additional kernel classes + advanced features)

Beyond matmul we ran reduction, memory-bandwidth, and three of cuda-oxide's bundled advanced examples. Highlights:

- **Reduction (1 GB warp-shuffle):** cuda-oxide hits **96% of nvcc** (1451 vs 1517 GB/s, both ~85% of HBM peak). See [`oxide-reduction/`](oxide-reduction/), [`cuda-reduction/`](cuda-reduction/).
- **Memory bandwidth (3-buffer streaming):** **0.1% gap** at N=64M (1608 vs 1609 GB/s, 90% of HBM peak). See [`oxide-vecadd-bench/`](oxide-vecadd-bench/), [`cuda-vecadd-bench/`](cuda-vecadd-bench/).
- **SASS root cause for the matmul gap:** nvcc emits `LDG.E.CONSTANT` (read-only cache hint) where cuda-oxide emits plain `LDG.E`. Both unroll 8×. See [`docs/experiments/sass-analysis.md`](docs/experiments/sass-analysis.md).
- **Consumer vs datacenter Blackwell:** cuda-oxide's flagship `gemm_sol` and `tcgen05_matmul` examples build but **fail at runtime** on RTX 5090 (sm_120) — they require sm_100/sm_100a (datacenter Blackwell, B100/B200). `tma_copy` works on consumer Blackwell. See [`oxide-gemm-sol/`](oxide-gemm-sol/), [`oxide-tcgen05-matmul/`](oxide-tcgen05-matmul/), [`oxide-tma-copy/`](oxide-tma-copy/).
- **Older arch unreachable:** modern libNVVM 22.0.0 rejects rustc-codegen-cuda IR for any arch ≤ sm_100. Cannot target Ampere/Ada with current cuda-oxide. See [`docs/experiments/libnvvm-causal-isolation.md`](docs/experiments/libnvvm-causal-isolation.md).

## Three findings

### 1. cuda-oxide hits 99% of nvcc on naive kernels (when libNVVM is correctly configured)
At N=1024 all four naive kernels — `oxide/safe` 6.94, `oxide/unchecked` 6.95, `oxide/fmuladd` 6.92, `cuda-matmul/matmul` 6.88 — land in a 1% band. The compiler quality story is much better than v0 suggested. The previous 2.5× "safety tax" was driven entirely by an outdated `libNVVM 7.0.1` shadowing the modern `libNVVM 22.0.0` from CUDA 13.2. See [`docs/experiments/libnvvm-corrigendum.md`](docs/experiments/libnvvm-corrigendum.md). With `CUDA_HOME=/usr/local/cuda` set, slice bounds-check codegen is essentially free.

### 2. The Rust safety tax was a libNVVM artifact, not a real cost
v0 reported a 2.5× slowdown for safe slice-indexed kernels vs unchecked. After diagnosing the [libNVVM shadow bug](docs/experiments/libnvvm-corrigendum.md) (the system `/usr/lib/x86_64-linux-gnu/libnvvm.so.4` was an outdated CUDA 12.0-era binary capping at compute_90), and forcing `CUDA_HOME=/usr/local/cuda` so the build picks up libNVVM 22.0.0 from CUDA 13.2, the safe/unchecked ratio drops to 1.00× at N=1024. The v0 tax was a misconfigured-environment artifact. Modern libNVVM optimizes Rust slice bounds checks to near-free.

### 3. The compiler gap reopens with tiling
Even with libNVVM fixed, cuda-oxide's tiled kernel (16×16 SharedArray) gets only 1.3-1.6× speedup from tiling vs nvcc-tiled's 4.5-5.3×. PTX inspection confirms the cause: zero `fma.rn.f32` instructions in oxide-tiled vs 256 in nvcc-tiled. Cuda-oxide's `FastmathFlagsAttr::default()` (= empty) blocks fast-math contraction in the default codegen path. Working escape hatch today: `core::intrinsics::fmuladdf32` lowers to libdevice's `__nv_fmaf` which itself contains `fma.rn.f32`, resolved by nvJitLink at module load. We've drafted an [upstream issue](docs/upstream-issue-fma.md). PTX evidence in [`oxide-matmul-tiled/ANALYSIS.md`](oxide-matmul-tiled/ANALYSIS.md).

### 4. wgpu cannot reach the NVIDIA GPU on WSL2, only llvmpipe (CPU)
The WSL2 VM exposes `/dev/dxg` (DirectX to CUDA passthrough) but no native Vulkan ICD that wgpu's Vulkan backend will accept as a discrete GPU adapter. wgpu falls back to **llvmpipe**, Mesa's software rasteriser, which runs the compute shader on the AMD Ryzen 7 7700X — ~25 seconds per iteration vs ~23 ms on the real GPU. This is not a wgpu performance issue; it is a WSL2 Vulkan-passthrough issue. See [`wgpu-matmul/ANALYSIS.md`](wgpu-matmul/ANALYSIS.md).

## Repo layout

- [`oxide-vecadd/`](oxide-vecadd/) — end-to-end smoke test (1024-element vector add) to verify the cuda-oxide toolchain works — [ANALYSIS.md](oxide-vecadd/ANALYSIS.md)
- [`oxide-matmul/`](oxide-matmul/) — naive cuda-oxide matmul (safe + unchecked + fmuladd kernels), 10 timed iterations × 3 sizes — [ANALYSIS.md](oxide-matmul/ANALYSIS.md)
- [`oxide-matmul-tiled/`](oxide-matmul-tiled/) — cuda-oxide tiled matmul (SharedArray, safe + unchecked) — [ANALYSIS.md](oxide-matmul-tiled/ANALYSIS.md)
- [`cuda-matmul/`](cuda-matmul/) — naive nvcc CUDA C++ reference, `cudaEventRecord`-timed — [ANALYSIS.md](cuda-matmul/ANALYSIS.md)
- [`cuda-matmul-tiled/`](cuda-matmul-tiled/) — register-tiled nvcc CUDA C++ (32×32 block + 4×4 microtile) — [ANALYSIS.md](cuda-matmul-tiled/ANALYSIS.md)
- [`cublas-matmul/`](cublas-matmul/) — cuBLAS sgemm reference (`CUBLAS_PEDANTIC_MATH`) — [ANALYSIS.md](cublas-matmul/ANALYSIS.md)
- [`wgpu-matmul/`](wgpu-matmul/) — wgpu/WGSL portable fallback, demonstrates the WSL2 Vulkan-passthrough limitation — [ANALYSIS.md](wgpu-matmul/ANALYSIS.md)
- [`docs/`](docs/) — research reports, ADRs, experiment writeups, upstream issue draft
- [`results/`](results/) — master scaling.csv (240 rows) + scaling-summary.md
- [`system-info/`](system-info/) — full hardware / driver / toolchain dump ([`system-spec.txt`](system-info/system-spec.txt))

## Quick start

```bash
git clone https://github.com/<you>/cuda-oxide-bench.git
cd cuda-oxide-bench
cat SETUP.md
```

Each of the four benchmarks lives in its own folder with its own run command:

- `cd oxide-vecadd && cargo oxide run oxide-vecadd`
- `cd oxide-matmul && cargo oxide run oxide-matmul`
- `cd cuda-matmul && /usr/local/cuda/bin/nvcc -ccbin clang-14 -O3 -arch=sm_120 -lstdc++ matmul.cu -o matmul && ./matmul`
- `cd wgpu-matmul && cargo run --release`

**REMEMBER:** before any `cargo oxide` command, set `CUDA_HOME=/usr/local/cuda` so cuda-oxide loads the correct libNVVM. See [SETUP.md](SETUP.md) Step 4 for details.

Full toolchain installation (LLVM 21, Rust nightly, CUDA 13.2, cargo-oxide) is in [SETUP.md](SETUP.md).

## Cross-references

- [METHODOLOGY.md](METHODOLOGY.md) — workload, inputs, timing methodology, threats to validity
- [SETUP.md](SETUP.md) — reproducible install steps and tested toolchain versions
- [system-info/system-spec.txt](system-info/system-spec.txt) — exact hardware and driver versions
- [analysis/ptx-stats.txt](analysis/ptx-stats.txt) — PTX-level instruction counts behind the deltas

## Acknowledgments and links

- **cuda-oxide** — [github.com/NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide) — the compiler under test, v0.1.0 (commit `6de0509`), released 2026-05-07
- **The cuda-oxide book** — <https://nvlabs.github.io/cuda-oxide/> — official documentation and language reference
- **wgpu** — [github.com/gfx-rs/wgpu](https://github.com/gfx-rs/wgpu) — the portable GPU API used for the WGSL baseline
- **NVIDIA CUDA Toolkit** — <https://developer.nvidia.com/cuda-toolkit> — nvcc reference compiler

This is **independent third-party work**. It is not affiliated with, endorsed by, or reviewed by NVIDIA, NVlabs, or the cuda-oxide authors. All opinions and findings are the author's own. Errors are mine; corrections welcome via issues or PRs.
