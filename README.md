# cuda-oxide-bench

A fair-comparison benchmark of [**cuda-oxide**](https://github.com/NVlabs/cuda-oxide) — NVIDIA's brand-new Rust-to-PTX compiler (v0.1.0, released 2026-05-07) — against raw **CUDA C++** (nvcc) and **wgpu/WGSL**, running the identical naive 4096×4096 f32 matrix-multiply algorithm on an RTX 5090 (Blackwell, sm_120) under WSL2. The goal is not to crown a winner, but to quantify how much performance a Rust-first CUDA frontend costs today, where the cost comes from, and where it matches or closes on the C++ baseline. Every kernel implements the same inner loop (`for k in 0..dim: C[r,c] += A[r,k] * B[k,c]`) with the same 16×16 thread block; no shared-memory tiling, no Tensor Cores, no cuBLAS — just the compiler and the driver.

## Headline results

4096×4096 f32 matmul (137.44 GFLOP/iter), 16×16 thread block, identical naive algorithm:

- **nvcc CUDA C++** `-arch=sm_89` — best 20.64 ms, median 21.53 ms, **6.39 TFLOPS**, 1.00×
- **cuda-oxide (unchecked, raw ptr)** — best 22.43 ms, median 23.15 ms, **5.94 TFLOPS**, **0.93×**
- **cuda-oxide (safe slice idx)** — best 51.82 ms, median 56.86 ms, 2.42 TFLOPS, 0.38×
- **wgpu/WGSL via llvmpipe CPU** — best ~25 000 ms, median ~25 700 ms, 0.005 TFLOPS, ~0.001×

(Full results and PTX instruction counts in [`analysis/ptx-stats.txt`](analysis/ptx-stats.txt).)

## Three findings

### 1. cuda-oxide hits 93% of nvcc when you use `unsafe` raw pointers
On a raw-pointer inner loop (`*a_base.add(r*dim+k)`), cuda-oxide lands at **0.93×** the throughput of nvcc's CUDA C++ — 5.94 vs 6.39 TFLOPS. The remaining ~10% gap is explained by PTX instruction selection: cuda-oxide emits separate `mul.rn` + `add.rn` pairs where nvcc emits 5 fused `fma.rn` instructions, and nvcc uses `ld.global.nc` (read-only cache, `__restrict__` / `__ldg`) on 10 loads where cuda-oxide uses plain `ld.global`. See [`oxide-matmul/ANALYSIS.md`](oxide-matmul/ANALYSIS.md) for the side-by-side PTX.

### 2. The 'safety tax' is a 2.46× slowdown from per-iter slice bounds checks
The only difference between the safe and unchecked kernels is the inner-loop read: `a[r*dim+k]` vs `*a_base.add(r*dim+k)`. That one change moves the kernel from 23.15 ms to 56.86 ms — a **2.46× slowdown**. In PTX this materialises as a `setp.ge.u64 … @%p bra` predicate pair inside the hot loop (8 `setp` in safe vs what would be ~0 if bounds were elided). Today, reaching nvcc-competitive throughput in cuda-oxide requires either `unsafe` raw pointers or compiler-side elision of bounds checks that LLVM's NVPTX path does not yet perform. Detailed breakdown: [`analysis/ptx-stats.txt`](analysis/ptx-stats.txt).

### 3. wgpu cannot reach the NVIDIA GPU on WSL2, only llvmpipe (CPU)
The WSL2 VM exposes `/dev/dxg` (DirectX to CUDA passthrough) but no native Vulkan ICD that wgpu's Vulkan backend will accept as a discrete GPU adapter. wgpu falls back to **llvmpipe**, Mesa's software rasteriser, which runs the compute shader on the AMD Ryzen 7 7700X — ~25 seconds per iteration vs ~23 ms on the real GPU. This is not a wgpu performance issue; it is a WSL2 Vulkan-passthrough issue. See [`wgpu-matmul/ANALYSIS.md`](wgpu-matmul/ANALYSIS.md).

## Repo layout

- [`oxide-vecadd/`](oxide-vecadd/) — end-to-end smoke test (1024-element vector add) to verify the cuda-oxide toolchain works — [ANALYSIS.md](oxide-vecadd/ANALYSIS.md)
- [`oxide-matmul/`](oxide-matmul/) — main cuda-oxide benchmark with both safe and unchecked kernels, 10 timed iterations — [ANALYSIS.md](oxide-matmul/ANALYSIS.md)
- [`cuda-matmul/`](cuda-matmul/) — nvcc CUDA C++ reference baseline, `cudaEventRecord`-timed, 5 iterations — [ANALYSIS.md](cuda-matmul/ANALYSIS.md)
- [`wgpu-matmul/`](wgpu-matmul/) — wgpu/WGSL portable fallback, demonstrates the WSL2 Vulkan-passthrough limitation — [ANALYSIS.md](wgpu-matmul/ANALYSIS.md)
- [`analysis/`](analysis/) — PTX instruction counts and compiler-level explanation of the deltas ([`ptx-stats.txt`](analysis/ptx-stats.txt))
- [`system-info/`](system-info/) — full hardware / driver / toolchain dump ([`system-spec.txt`](system-info/system-spec.txt))

## Quick start

```bash
git clone https://github.com/<you>/cuda-oxide-bench.git
cd cuda-oxide-bench
cat SETUP.md
```

Each of the four benchmarks lives in its own folder with its own run command:

- `cd oxide-vecadd && cargo oxide run`
- `cd oxide-matmul && cargo oxide run`
- `cd cuda-matmul && nvcc -O3 -arch=sm_89 matmul.cu -o matmul && ./matmul`
- `cd wgpu-matmul && cargo run --release`

Full toolchain installation (LLVM 21, Rust nightly, CUDA 12.x, cargo-oxide) is in [SETUP.md](SETUP.md).

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
