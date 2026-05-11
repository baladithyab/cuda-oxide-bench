# cuda-matmul: nvcc CUDA C++ Reference Baseline

**Role in the benchmark:** this folder is the *reference* against which
`cuda-oxide` (Rust) and `wgpu` (WGSL) are judged. Hardware: RTX 5090
(sm_120, Blackwell), WSL2, CUDA 12.0 toolkit. Problem: 4096×4096 f32
naive matmul, 16×16 thread block, one output element per thread, no
shared-memory tiling.

## 1. Why a reference baseline matters

"Naive matmul TFLOPS" is not a fixed number — it is whatever the
optimizer behind the language makes of the same three nested loops. The
same C-like kernel can land anywhere from ~2 TFLOPS (strict-FP, with
bounds checks) to ~7 TFLOPS (fully contracted, read-only cache) on this
GPU. Without a reference we only know *relative* numbers between two
Rust variants; we have no way to say "cuda-oxide is within X% of what
NVIDIA's own compiler can do on this exact algorithm." That is the
explicit goal of this folder: run the same algorithm through the most
optimizing path NVIDIA ships — `nvcc` → NVVM (LLVM 7) → `ptxas` → driver
JIT — and record the number. Every other implementation in the repo is
measured against this row.

## 2. Methodology

Timing is done with `cudaEventRecord` / `cudaEventSynchronize`, not
wall-clock. Events are enqueued on the stream and measured GPU-side, so
host scheduling jitter does not contaminate the number. Protocol: **1
warmup launch** + **5 timed iterations**; we report best and median (not
mean) so a stray outlier cannot skew the result.

Inputs are bit-identical to `oxide-matmul` so the comparison is fair:

```cpp
for (int i = 0; i < N*N; ++i) { hA[i] = (i % 7) * 0.01f; hB[i] = (i % 11) * 0.01f; }
```

The kernel is the textbook triple loop, with `__restrict__` on all three
pointers — this is the keyword that lets nvcc emit the read-only
(non-coherent) cache path in PTX:

```cpp
__global__ void matmul(const float* __restrict__ A, const float* __restrict__ B,
                       float* __restrict__ C, int dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dim || col >= dim) return;
    float acc = 0.0f;
    for (int k = 0; k < dim; ++k) {
        acc += A[row * dim + k] * B[k * dim + col];
    }
    C[row * dim + col] = acc;
}
```

## 3. Compile flags

```
nvcc -ccbin clang-14 -O3 -arch=sm_89 -o matmul matmul.cu
```

* **`-arch=sm_89`** — CUDA 12.0 predates Blackwell and does **not** know
  `sm_120`. The recommended forward-compat path is to emit PTX for the
  newest arch your toolchain supports (Ada / sm_89) and let the driver
  **JIT the PTX to `sm_120` at load time**. The PTX header confirms
  this: `.target sm_89`. At runtime the driver reports
  `device: NVIDIA GeForce RTX 5090 (sm_120)` and translates the PTX
  into native SASS for Blackwell.
* **`-ccbin clang-14`** — nvcc 12.0's host-compiler allow-list does not
  include the gcc/g++ shipping on modern Ubuntu/WSL; clang-14 is the
  simplest supported host compiler to install and point nvcc at.
* **`-O3`** — enables full host + device optimization, including the
  loop unroll we see in the PTX below.

## 4. Results

From `run.log` (identical to the `nvcc` row in `/tmp/bench-context.txt`):

- **best:** 20.64 ms — **6.66 TFLOPS**
- **median:** 21.53 ms — **6.39 TFLOPS**
- workload: 137.44 GFLOP per iteration (2·N³)

| impl                      | best ms | median ms | TFLOPS (med) | vs nvcc |
|---------------------------|--------:|----------:|-------------:|--------:|
| nvcc CUDA C++ `-arch=sm_89` |  20.64 |   21.53   |     6.39     |  1.00×  |

## 5. PTX deep-dive

The inner hot loop lives at `$L__BB0_4` in `matmul.ptx`. The compiler
unrolled it **4× automatically** (one iteration of the PTX loop performs
four original k-iterations), and every multiply-add is contracted into a
single FMA:

```
$L__BB0_4:
    ld.global.nc.f32    %f12, [%rd32];
    ld.global.nc.f32    %f13, [%rd31+-8];
    fma.rn.f32          %f14, %f13, %f12, %f29;
    add.s64             %rd23, %rd32, %rd5;
    ld.global.nc.f32    %f15, [%rd23];
    ld.global.nc.f32    %f16, [%rd31+-4];
    fma.rn.f32          %f17, %f16, %f15, %f14;
    add.s64             %rd24, %rd23, %rd5;
    ld.global.nc.f32    %f18, [%rd24];
    ld.global.nc.f32    %f19, [%rd31];
    fma.rn.f32          %f20, %f19, %f18, %f17;
    add.s64             %rd25, %rd24, %rd5;
    add.s64             %rd32, %rd25, %rd5;
    ld.global.nc.f32    %f21, [%rd25];
    ld.global.nc.f32    %f22, [%rd31+4];
    fma.rn.f32          %f29, %f22, %f21, %f20;
    add.s32             %r28, %r28, 4;
    ...
    @%p6 bra            $L__BB0_4;
```

Two instructions matter:

* **`fma.rn.f32 %f14, %f13, %f12, %f29`** — fused multiply-add with
  round-to-nearest-even. One dispatch, one rounding, a single cycle of
  throughput on FP32-capable SMs. The alternative — separate `mul.rn.f32`
  then `add.rn.f32` — costs two cycles and two roundings. `cuda-oxide`'s
  PTX emits exactly that separated pair (see `analysis/ptx-stats.txt`:
  0 FMAs, 4 `mul.rn + add.rn`) because LLVM's default strict-FP mode
  will not contract across a `*` / `+` boundary without the right flag.
* **`ld.global.nc.f32`** — the **non-coherent** (read-only) global load.
  Because `__restrict__` promised the compiler that `A` and `B` never
  alias `C`, nvcc is free to route these reads through the read-only
  data cache (the same path `__ldg` exposes). No cache-coherency traffic,
  higher effective bandwidth, better hit rate for the reused rows/cols
  of a matmul. `cuda-oxide`'s PTX has **zero** `ld.global.nc` — every
  load is a plain `ld.global`.

Between `fma` contraction, read-only loads, and the 4× unroll, nvcc is
squeezing every easy win out of the naive algorithm.

## 6. Why nvcc still leaves performance on the table

The RTX 5090's peak FP32 is roughly **104 TFLOPS**. 6.4 TFLOPS is about
**6% of SoL**. That is not an nvcc failure — it is the algorithm:

* Every output element performs a 4096-step dot product.
* No shared-memory tiling, so each row of A and column of B is re-read
  from global memory by many threads.
* The kernel is therefore **memory-bound**, and the FMA units sit mostly
  idle waiting for loads.

A properly tiled kernel with shared memory reaches **30–50 TFLOPS**.
cuBLAS `sgemm` on the same shape hits **60–90 TFLOPS**. Tensor-core
paths (TF32/FP16) blow past **1000 TFLOPS**. This bench deliberately
uses the naive algorithm so we are comparing **language/runtime overhead
on identical work**, not framework-level matmul cleverness.

## 7. Reproducing

```bash
cd ~/cuda-exploration/cuda-matmul
nvcc -ccbin clang-14 -O3 -arch=sm_89 -o matmul matmul.cu
./matmul
```

## 8. Limitations

* No `nsys` / `ncu` profile — we only have event timings; per-SM
  occupancy, warp-stall reasons, and DRAM bandwidth utilization are
  unmeasured.
* Single GPU (RTX 5090, sm_120, WSL2). Numbers are not portable across
  architectures; on Ada or Hopper the TFLOPS and the FMA-vs-mul+add gap
  can shift.
* PTX is emitted for **sm_89** and JIT'd to sm_120 by the driver. The
  JIT is good, but it cannot synthesize Blackwell-specific instructions
  the sm_89 PTX never referenced — a native sm_120 compile, once CUDA
  12.8+ is installed, would likely shave a few percent off.