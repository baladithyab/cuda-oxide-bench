# Setup

Reproducible install steps for cuda-exploration. Expect ~30 minutes end-to-end on a warm system, most of it downloading LLVM and building cargo-oxide.

## System requirements

- **OS** — Linux x86_64 (tested on Ubuntu 24.04 LTS, native and WSL2). Other distros should work; steps below are apt-centric.
- **GPU** — NVIDIA GPU with compute capability **sm_70 or newer** (Volta, Turing, Ampere, Ada, Hopper, Blackwell). Tested on RTX 5090 (sm_120).
- **Disk** — ~5 GB free (LLVM 21 is ~3 GB, Rust nightly + build targets ~1 GB, CUDA toolkit ~1 GB if not already installed).
- **Install time** — ~30 minutes on a fresh machine, most of it LLVM.

## Step 1 — LLVM 21 (with NVPTX target)

cuda-oxide compiles Rust MIR → LLVM IR → PTX using `llc` from LLVM 21. The Ubuntu archive's `llvm` packages are often older; use the official LLVM apt script to get a clean 21 install that includes the `nvptx64` backend:

```bash
wget -qO- https://apt.llvm.org/llvm.sh | sudo bash -s -- 21 all
```

Verify the NVPTX backend is registered:

```bash
llc-21 --version | grep -i nvptx
# Expected:
#   nvptx   - NVIDIA PTX 32-bit
#   nvptx64 - NVIDIA PTX 64-bit
```

If `nvptx64` is missing, cargo-oxide will fail at link time. Re-run the installer with `all` to pull every target.

## Step 2 — Rust nightly with required components

cuda-oxide uses `#[kernel]` proc-macros and nightly-only `rustc_private` APIs. Install nightly and add the three components needed to compile kernel crates:

```bash
rustup install nightly
rustup component add --toolchain nightly rust-src rustc-dev llvm-tools-preview
```

`rust-src` gives you the standard-library sources cargo-oxide recompiles for the NVPTX target. `rustc-dev` exposes the compiler internals the `#[kernel]` macro depends on. `llvm-tools-preview` ships `llvm-objcopy` / `llvm-ar` that the build driver invokes.

## Step 3 — CUDA toolkit (nvcc 12.x)

Needed for the `cuda-matmul` baseline and for the CUDA runtime libs that cargo-oxide links against. This repo was built and tested with **nvcc 12.0** from the Ubuntu system package. The toolkit typically installs to `/usr/local/cuda`.

On Ubuntu:
```bash
sudo apt install nvidia-cuda-toolkit
# or, for a newer version, follow https://developer.nvidia.com/cuda-downloads
```

Make sure `/usr/local/cuda/bin` is on `PATH` and `/usr/local/cuda/lib64` is on `LD_LIBRARY_PATH`.

**Blackwell note.** nvcc 12.0 does not know about `sm_120` (RTX 5090). Compile the CUDA C++ baseline with `-arch=sm_89` (Ada) and let the driver's PTX JIT re-target to the physical device at load time. This works because the PTX ISA is forward-compatible; the cost is a one-time JIT at first launch (amortised by the warmup iteration).

## Step 4 — cargo-oxide

cargo-oxide is the build driver that wraps `cargo` for cuda-oxide projects. Install it from the upstream repository:

```bash
cargo +nightly install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
```

**CRITICAL: Set `CUDA_HOME` before any `cargo oxide` invocation** to avoid a libNVVM shadow bug:

```bash
export CUDA_HOME=/usr/local/cuda
export LIBNVVM_PATH=/usr/local/cuda/nvvm/lib64/libnvvm.so
```

Without this, cuda-oxide's `libnvvm-sys` may load `/usr/lib/x86_64-linux-gnu/libnvvm.so.4` (libNVVM 7.0.1 from an older Ubuntu CUDA package, capped at compute_90 / Hopper) instead of the modern libNVVM 22.0.0 from CUDA 13.2. Symptom: builds fail with `-arch=compute_120 is an unsupported option`, OR build silently produces poor PTX for slice indexing. See [`docs/experiments/libnvvm-corrigendum.md`](docs/experiments/libnvvm-corrigendum.md). This is the difference between `oxide/safe` running at 6.94 TFLOPS (correct) vs 2.80 TFLOPS (broken libNVVM) at N=1024.

From inside any cargo-oxide project (e.g. `oxide-vecadd/`), verify the toolchain is healthy:

```bash
cd oxide-vecadd
cargo oxide doctor
```

You should see green checkmarks for: nightly toolchain detected, `rust-src` present, `rustc-dev` present, `llvm-tools-preview` present, `llc-21` found, NVPTX target registered, CUDA driver reachable.

**Note:** `doctor` reports "libNVVM 2.0" — that's the IR version, not the toolkit version. To confirm you're getting the modern libNVVM 22.0.0, run:

```bash
strings /usr/local/cuda/nvvm/lib64/libnvvm.so | grep "compute_120"
# Should print: -arch=compute_120, -arch=compute_120a, -arch=compute_120f
```

If any check is red, fix it before moving on — the build will fail with cryptic linker errors otherwise.

## Step 5 — Run the benchmarks

Each benchmark lives in its own folder with its own run command:

```bash
# 1. cuda-oxide smoke test (1024-elem vector add)
cd oxide-vecadd && cargo oxide run

# 2. cuda-oxide matmul (main benchmark, safe + unchecked kernels)
cd ../oxide-matmul && cargo oxide run

# 3. nvcc CUDA C++ reference
cd ../cuda-matmul && nvcc -O3 -arch=sm_89 matmul.cu -o matmul && ./matmul

# 4. wgpu/WGSL portable baseline
cd ../wgpu-matmul && cargo run --release
```

Expected total runtime: under 5 minutes once everything is compiled. Each harness prints per-iteration timings, median, best, correctness-check status, and PTX output path where applicable.

## Known issues on WSL2

The `wgpu-matmul` run will **not reach the NVIDIA GPU** on WSL2. wgpu uses Vulkan, and WSL2 does not expose an NVIDIA Vulkan adapter to Linux guests — only a software fallback (`llvmpipe`). Expect ~25 seconds per iteration (vs ~23 ms on a real GPU). This is a WSL2 infrastructure limitation, not a wgpu bug. On a native Linux host with `nvidia-driver-*` + `libvulkan1` the wgpu bench will pick the RTX 5090 directly. Details in [`wgpu-matmul/ANALYSIS.md`](wgpu-matmul/ANALYSIS.md).

## Toolchain versions tested

Exact versions from [`system-info/system-spec.txt`](system-info/system-spec.txt):

- **OS** — Ubuntu 24.04.4 LTS (Noble Numbat), kernel `6.6.87.2-microsoft-standard-WSL2`
- **CPU** — AMD Ryzen 7 7700X (8C/16T, znver4)
- **GPU** — NVIDIA GeForce RTX 5090, Blackwell, 32 GiB, VBIOS `98.02.2e.00.03`
- **NVIDIA driver** — `596.21` (Windows host), `NVIDIA-SMI 595.58.03` (guest)
- **CUDA** — driver advertises CUDA 13.2; toolkit / nvcc is 12.0 (`V12.0.140`, built 2023-01-06)
- **libcuda / libnvvm** — WSL passthrough `/usr/lib/wsl/lib/libcuda.so`, `libnvvm.so.4.0.0`
- **Rust** — `rustc 1.88.0 (6b00bc388 2025-06-23)` stable; `nightly-2026-04-03` also installed
- **LLVM / clang** — `21.1.8` with NVPTX target registered
- **cargo-oxide** — `0.1.0`, commit `6de0509` (2026-05-07)
