# wgpu-matmul — Analysis

## 1. Goal

This test is the cross-vendor Rust-on-GPU leg of the benchmark. The other three
tests (`cuda-matmul`, `oxide-matmul` safe, `oxide-matmul` unchecked) all reach
the RTX 5090 through the proprietary CUDA driver. The question here is narrow
and fair: given the **same kernel** (naive O(N³) f32 matmul), the **same shape**
(4096×4096), and the **same input pattern**, does a wgpu + WGSL program —
configured to use whatever backend the host will give it (Vulkan first, then
DX12, then GL) — actually reach the same physical GPU?

If yes, we'd have a three-way comparison: CUDA driver (vendor) vs. Vulkan
compute (cross-vendor, same hardware). If no, the failure mode is itself the
finding.

## 2. Methodology

- `wgpu` 22 with the `dx12` feature enabled in `Cargo.toml` so the DX12 backend
  is compiled in alongside Vulkan/GL.
- WGSL compute shader (`src/matmul.wgsl`): one thread per output element,
  `@compute @workgroup_size(16, 16, 1)`, dispatched as `(N/16, N/16, 1)` — the
  same thread-block geometry as the CUDA and cuda-oxide kernels.
- Four bind-group entries at `@group(0)`: binding 0 = A (read-only storage),
  binding 1 = B (read-only storage), binding 2 = C (read_write storage),
  binding 3 = `dim` (uniform u32).
- 1 warmup + 5 timed iterations. Where `TIMESTAMP_QUERY` is supported, times
  are derived from GPU-side begin/end timestamps multiplied by
  `queue.get_timestamp_period()`. Otherwise we fall back to `Instant::now()`
  around `queue.submit` + `device.poll(Wait)` (CPU wall-clock, includes
  dispatch overhead).
- Inputs are generated with the same `(i % 7) * 0.01`, `(i % 11) * 0.01`
  pattern used by the CUDA paths so the kernel does identical arithmetic.

## 3. Adapter selection on WSL2 — what we tried, why it failed

The adapter loop in `src/main.rs` enumerates every adapter across all backends
and picks the first non-`Cpu` one, falling back to a CPU adapter only if that
is all that exists. On this WSL2 host, only CPU adapters exist.

**Vulkan backend.** Enumerates exactly one device:
`llvmpipe (LLVM 20.1.2, 256 bits)`, `device_type = Cpu`. That is Mesa's
software rasterizer — the compute pipeline runs on the znver4 CPU cores. The
system does have `/usr/share/vulkan/icd.d/nvidia_icd.json`, but that ICD
manifest points at `libGLX_nvidia.so.0`, which is NVIDIA's OpenGL/GLX library,
not a Vulkan ICD. On bare-metal Linux, NVIDIA ships `libnvidia-vulkan-*.so`
as part of the proprietary driver; on WSL, the Linux-side NVIDIA driver stack
is a thin shim over the Windows host driver via `/dev/dxg`, and it does not
expose a Linux Vulkan ICD. Net result: Vulkan on WSL sees the software
rasterizer and nothing else.

**DX12 backend.** We enabled it explicitly — `wgpu = { version = "22", features = ["dx12"] }`
— because `/dev/dxg` is present and `/usr/lib/wsl/lib/libd3d12.so` is
installed (that is the whole point of WSLg / WSL GPU compute). wgpu's DX12
backend, however, still did not enumerate the GPU on this host. The missing
piece is Microsoft's Mesa "Dozen" driver — the Vulkan-on-D3D12 translation
layer shipped as `dzn_icd.x86_64.json` in `mesa-vulkan-drivers-microsoft`.
That package is published by Microsoft, not in Ubuntu 24.04's default repos,
and we did not install it. Without it (or a direct DX12 path that wgpu's
Linux build can drive), the kernel has no route from `wgpu::Backends::DX12`
to the Windows D3D12 runtime.

**GL backend.** Enumerates the same `llvmpipe` as a `Gl` adapter. Same CPU.

The resulting warning in `run.log`:

```
[wgpu] candidate: llvmpipe (LLVM 20.1.2, 256 bits) (Vulkan, type=Cpu)
[wgpu] candidate: llvmpipe (LLVM 20.1.2, 256 bits) (Gl, type=Cpu)
[wgpu] using: llvmpipe (LLVM 20.1.2, 256 bits) (Vulkan, type=Cpu)
[wgpu] !! WARNING: only CPU adapter available (WSL Vulkan limitation).
```

## 4. What we ran instead

A CPU run, via Vulkan→llvmpipe. From `run.log`:

```
[wgpu] matmul 4096x4096 f32, 137.44 GFLOP/iter
[wgpu] warmup 0: gpu_ts=25576.33 ms (0.005 TFLOPS)
[wgpu] iter   1: gpu_ts=25448.37 ms (0.005 TFLOPS)
[wgpu] iter   2: gpu_ts=28436.62 ms (0.005 TFLOPS)
```

Median ≈ **25,738 ms, 0.005 TFLOPS**. That is roughly **1,100× slower** than
cuda-oxide's unchecked path (23.15 ms / 5.94 TFLOPS) on the exact same shape.
These numbers say nothing about wgpu-on-a-real-GPU; they only tell us what
llvmpipe does with 137 GFLOPs of naive matmul on 24 Zen 4 cores.

## 5. Why this is itself a finding

If you are on WSL2 with an NVIDIA GPU and you pick wgpu over cuda-oxide for
compute, you get zero GPU access by default. That is the comparison, even if
it is not the comparison anyone wanted. cuda-oxide wins here by being able to
**run on the GPU at all** — it talks to the Windows-side NVIDIA driver via
the CUDA shim (`libcuda.so.1` on WSL), which is a supported path.

On a different host this would look different. On bare-metal Linux with a
working `libnvidia-vulkan`, or on Windows with wgpu's DX12 backend, the same
code would hit the GPU and we would get a real Vulkan/DX12-vs-CUDA number.
Published wgpu/Vulkan-compute benchmarks on kernel workloads typically show
Vulkan trailing vendor compute APIs by roughly 5–10× for naive kernels (more
for tuned libraries like cuBLAS). We did not measure that here — we cannot
from this machine.

## 6. Reading the WGSL

```wgsl
@group(0) @binding(0) var<storage, read>       a : array<f32>;
@group(0) @binding(1) var<storage, read>       b : array<f32>;
@group(0) @binding(2) var<storage, read_write> c : array<f32>;
@group(0) @binding(3) var<uniform>             dim : u32;

@compute @workgroup_size(16, 16, 1)
fn matmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= dim || col >= dim) { return; }
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < dim; k = k + 1u) {
        acc = acc + a[row * dim + k] * b[k * dim + col];
    }
    c[row * dim + col] = acc;
}
```

`@group(0) @binding(N)` matches the `BindGroupLayoutEntry` at index `N` in
the Rust side: 0/1 are read-only storage buffers (A, B), 2 is a read-write
storage buffer (C), 3 is a uniform buffer holding `dim`. The kernel does one
explicit grid-boundary check (`row >= dim || col >= dim`), identical to what
the CUDA kernels do.

Semantics worth calling out vs. the Rust/cuda-oxide side: WGSL has **no
Rust-style slice-bounds-check** at the language level — no hidden `setp` +
branch per indexed read. What it does have is **mandatory robust-access
semantics**: out-of-bounds reads on storage/uniform buffers return zero, and
out-of-bounds writes are silently dropped. Conceptually this sits *closer to
cuda-oxide's `unchecked` path* than to the `safe` (slice-indexed) path —
there is no panic, no early return, just clamping. The actual cost of that
guarantee depends entirely on the translation stack: Naga emits SPIR-V (or
HLSL for DX12), and it is the downstream driver compiler that decides whether
it can prove indices in-range and omit bounds enforcement. On llvmpipe, none
of this matters — the whole kernel runs on the CPU via LLVM JIT and the
f32 multiply-add throughput is bounded by SIMD, not by predicate branches.

## 7. Limitations & how to actually do this comparison

Three supported paths would turn this from a WSL adapter-selection note into a
real benchmark:

a. **Bare-metal Linux with NVIDIA's Vulkan driver.** Install `nvidia-driver-*`
   such that `libnvidia-vulkan-*.so` is present and the ICD manifest points
   at it (not `libGLX_nvidia.so.0`). wgpu's Vulkan backend will then enumerate
   the RTX 5090 directly.

b. **Windows host with wgpu DX12.** Build and run the same crate on Windows.
   The DX12 backend on Windows is a first-class wgpu path and will hit the
   GPU through D3D12 without any ICD gymnastics.

c. **Install Dozen / Mesa-Vulkan-on-D3D12 in WSL.** Add Microsoft's
   `mesa-vulkan-drivers-microsoft` (which installs the `dzn` ICD at
   `dzn_icd.x86_64.json`) so Vulkan calls in WSL tunnel through `/dev/dxg`
   to the Windows D3D12 runtime. This would still go through a translation
   layer, so it is a weaker comparison than (a) or (b), but it would exercise
   real GPU silicon.

We did **none** of these three on this run. Everything here is adapter
selection + a CPU-side reproducer.

## 8. Reproducing on this machine

```
cargo run --release --manifest-path=wgpu-matmul/Cargo.toml
```

From `~/cuda-oxide-bench`. On this WSL2 host the run will print the `!!
WARNING: only CPU adapter available` line and then spend roughly 25–30
seconds per iteration (≈ 2–3 minutes for warmup + 5 iters) on the 24-core
znver4 CPU via llvmpipe. The `run.log` in this directory is a trimmed capture
of one such run.
