# [bug] `ct.mma(f32, f32, f32)` generates unfused scalar FMUL+FADD; missing FFMA contraction (RTX 5090 sm_120, v1.3.0)

## Summary

On RTX 5090 (Blackwell consumer, `sm_120`, cuda-tile 1.3.0), `ct.mma(a, b, acc)`
with **f32 × f32 → f32acc** operands lowers to a per-element scalar `FMUL`
followed by `FADD` with **no FMA fusion at all** — the resulting cubin
contains **zero `FFMA`** instructions. The same kernel body with f16, bf16,
or tf32 inputs lowers to `HMMA` tensor-core instructions correctly, and
`nvcc` compiling the same shared-memory-tiled algorithm at f32 emits 256
`FFMA`s. So this is specifically an issue with the **f32 lowering path of
`ct.mma`** missing the libNVVM FMA contractor, not a hardware-capability
issue (Blackwell consumer has no f32 tensor-core MMA — that part is
correct) and not a generic cuTile codegen issue.

Observed end-to-end perf: 8.7 TFLOPS at N=4096 vs 38.4 TFLOPS for the same
algorithm hand-written in CUDA C++ and compiled by `nvcc` on the same GPU
(**4.4× slower**). The f32 CUDA-core fallback path is leaving at minimum
FFMA contraction on the table.

## Environment

| item | value |
|---|---|
| cuda-tile (cutile-python) | **1.3.0** |
| cupy | cupy-cuda13x 14.0.1 |
| CUDA toolkit | 13.2 (`/usr/local/cuda/bin/nvcc` → `release 13.2`) |
| NVIDIA driver | as shipped with CUDA 13.2 on this box (WSL2 host passthrough) |
| GPU | NVIDIA GeForce RTX 5090 |
| compute capability | `sm_120` (Blackwell consumer) |
| OS | Windows 11 host, WSL2 Ubuntu |
| libNVVM | `/usr/local/cuda/nvvm/lib64/libnvvm.so` (CUDA 13.2, libNVVM 22.0.0) |

## Minimal reproducer

Kernel body below compiles, runs, and produces numerically correct results
— the bug is purely in the SASS quality of the lowered `ct.mma` at f32.

```python
import cuda.tile as ct
import cupy
import numpy as np

BM = BN = 128
BK = 16

@ct.kernel
def matmul_f32(A, B, C):
    i, j = ct.bid(0), ct.bid(1)
    a_view = A.tiled_view((BM, BK), padding_mode=ct.PaddingMode.ZERO)
    b_view = B.tiled_view((BK, BN), padding_mode=ct.PaddingMode.ZERO)
    acc = ct.zeros((BM, BN), ct.float32)
    for k in range(a_view.num_tiles(1)):
        tx = a_view.load((i, k))
        ty = b_view.load((k, j))
        acc = ct.mma(tx, ty, acc)          # f32 × f32 → f32acc
    ct.store(C, (i, j), acc.astype(C.dtype))

N = 4096
rng = np.random.default_rng(0xC0FFEE)
a = cupy.asarray(rng.random((N, N), dtype=np.float32))
b = cupy.asarray(rng.random((N, N), dtype=np.float32))
c = cupy.zeros((N, N), dtype=cupy.float32)
stream = cupy.cuda.get_current_stream()
ct.launch(stream.ptr, (N // BM, N // BN), matmul_f32, (a, b, c))
cupy.cuda.runtime.deviceSynchronize()

# Export the cubin for SASS inspection:
from cuda.tile.compilation import (
    ArrayConstraint, CallingConvention, KernelSignature, export_kernel)
def _ac():
    return ArrayConstraint(
        dtype=ct.float32, ndim=2, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=(), may_alias_internally=False,
        stride_constant=(None, 1), stride_divisible_by=1,
        shape_divisible_by=1, base_addr_divisible_by=1)
sig = KernelSignature(
    parameters=[_ac(), _ac(), _ac()],
    calling_convention=CallingConvention.cutile_python_v1())
export_kernel(matmul_f32, [sig], "matmul_f32.cubin",
              gpu_code="sm_120", output_format="cubin")
```

Then:

```bash
/usr/local/cuda/bin/cuobjdump --dump-sass matmul_f32.cubin > matmul_f32.sass
grep -c FFMA matmul_f32.sass   # → 0   (expected: non-zero)
grep -c FMUL matmul_f32.sass   # → 2051
grep -c FADD matmul_f32.sass   # → 2176
```

Full reproducer (includes the other three dtype variants used for the
comparison table below) lives in the cuda-exploration repo:

- <https://github.com/baladithyab/cuda-exploration/blob/master/cutile-matmul-tiled-mixed/main.py>

## Expected behavior

At minimum, the libNVVM FMA pattern-match contractor should fuse each
`FMUL` + `FADD` pair into an `FFMA`, matching what `nvcc` does on the
same algorithm expressed in CUDA C++. There is no hardware f32 MMA on
Blackwell consumer (we don't expect `HMMA`) — but the CUDA-core fallback
path should still go through FFMA, not scalar FMUL+FADD pairs.

Concrete numeric expectation, taken from the reference nvcc compile of
the same shared-memory-tiled algorithm on the same hardware:

- `nvcc -arch=sm_120` on the same tiled matmul at f32 → **256 `FFMA`**,
  0 `FMUL`, 0 `FADD` (the FFMAs cover the K-loop)
- cuda-oxide's rust-cuda pipeline on a similar register-microtile kernel
  at f32 → **192 `FFMA`** with `.reuse` register hints

## Observed behavior — SASS instruction counts

All four cubins produced from the **same kernel body** differing only in
operand dtype (export path identical, `gpu_code='sm_120'`, CUDA 13.2
cuobjdump):

| variant | `HMMA` shape | HMMA | `FFMA` | `FMUL` | `FADD` | TFLOPS @ N=4096 |
|---|---|---:|---:|---:|---:|---:|
| `mma_f16xf16_f32acc`   | `HMMA.16816.F32`        | 64  | 0 | 2    | 0    | **172.5** |
| `mma_bf16xbf16_f32acc` | `HMMA.16816.F32.BF16`   | 64  | 0 | 2    | 0    |   159.8  |
| `mma_tf32xtf32_f32acc` | `HMMA.1688.F32.TF32`    | 128 | 0 | 3    | 0    |    84.0  |
| `mma_f32xf32_f32acc`   | *(none — no f32 MMA HW)* | 0  | **0** | **2051** | **2176** | **8.7** |

Cross-stack comparison for the **same shared-mem-tiled algorithm at f32**
on the same GPU/session:

| implementation | FFMA | FMUL | FADD | HMMA | TFLOPS @ N=4096 |
|---|---:|---:|---:|---:|---:|
| **cuTile `ct.mma` f32×f32**                 | **0** | 2051 | 2176 | 0 | 8.7 |
| nvcc `-arch=sm_120` shared-mem-tiled f32    | 256   | 0    | 0    | 0 | 38.4 |
| cuda-oxide register-microtile f32 (rust-cuda) | 192 | 0    | 0    | 0 | 45.0 |
| cuBLAS sgemm (tensor-core via internal TF32)  | —   | —    | —    | — | 73.6 |

So cuTile's f32 `ct.mma` is ~**4.4× slower** than the same hand-tiled
algorithm through `nvcc`, and ~**8-12× lighter on useful FP-issue
instructions** than it should be. The other three dtype variants (f16,
bf16, tf32) lower to `HMMA` cleanly — they are not affected and in fact
reach excellent throughput (f16 hits 172.5 TFLOPS, which is a real
tensor-core win from Python via cuTile).

## Independently-verifiable evidence

All in the cuda-exploration repo
(<https://github.com/baladithyab/cuda-exploration>):

- `cutile-matmul-tiled-mixed/main.py` — 4-variant reproducer (f16/bf16/tf32/f32)
- `cutile-matmul-tiled-mixed/mma_f32xf32_f32acc.sass` — the problem cubin
- `cutile-matmul-tiled-mixed/mma_f16xf16_f32acc.sass` — f16, 64 HMMAs (for contrast)
- `cutile-matmul-tiled-mixed/mma_bf16xbf16_f32acc.sass` — bf16, 64 HMMAs
- `cutile-matmul-tiled-mixed/mma_tf32xtf32_f32acc.sass` — tf32, 128 HMMAs
- `cutile-matmul-tiled-mixed/results.csv` — per-iter gpu_ms / TFLOPS, 10 iters × 3 sizes × 4 dtypes
- `cutile-matmul-tiled-mixed/ANALYSIS.md` — Wave 13.1 writeup
- `analysis/wave13-sass/cuda_matmul_tiled.sass` — nvcc reference at f32 (256 FFMAs)
- `analysis/wave13-sass/oxide_matmul_tiled_microtile.sass` — cuda-oxide reference at f32 (192 FFMAs)
- `results/wave13-summary.md` — cross-stack summary

Grep commands that reproduce the headline counts:

```bash
grep -c HMMA mma_f16xf16_f32acc.sass             # → 64
grep -c HMMA mma_f32xf32_f32acc.sass             # → 0
grep -c FFMA mma_f32xf32_f32acc.sass             # → 0
grep -c FMUL mma_f32xf32_f32acc.sass             # → 2051
grep -c FADD mma_f32xf32_f32acc.sass             # → 2176
grep -c FFMA analysis/wave13-sass/cuda_matmul_tiled.sass   # → 256
```

We also verified there are **no register spills** (`grep -c STL\\b
mma_f32xf32_f32acc.sass` → 0) — the kernel isn't spill-bound, it's simply
unfused. An earlier hypothesis about STL spills did not reproduce and is
noted in `ANALYSIS.md`.

## Suggested investigation

Without seeing the internal IR, the pattern that best fits the evidence
is that the NVVM IR emitted for `ct.mma` at the f32 path is *not* marked
with `FastmathFlags::CONTRACT` (or the equivalent `contract` fast-math
flag on the `fmul`/`fadd` ops), so libNVVM's FMA contractor declines to
fuse. Two places worth looking at:

1. The lowering of `ct.mma(a, b, acc)` when neither operand is a
   tensor-core-eligible dtype — it currently appears to expand to a
   scalar `a_ij * b_jk + acc_ik` loop nest without the contract flag.
2. Whether the cuTile compiler globally sets `contract` / `fast` on the
   generated NVVM module, independent of the per-op lowering — a module-
   wide `nvvm.annotations`-style fast-math toggle would fix this path
   without a per-op audit.

A quick experiment that would confirm the root cause is to manually run
the exported kernel's LLVM/NVVM IR through `nvvm-ir` (or equivalent) with
`-fmad=true` / contract-on and diff the resulting SASS.

## Workarounds known to the filing repo

- **Use f16, bf16, or tf32 inputs** (with an f32 accumulator). These paths
  engage `HMMA` correctly and reach 159-172 TFLOPS on the same kernel.
  This is the cuTile-idiomatic answer for matmul-shaped compute-bound work.
- If f32 inputs are required, avoid `ct.mma` and hand-write the inner
  product with `ct.sum(a_tile * b_tile, axis=...)` style — but this does
  not engage the FFMA contractor either in our tests.
- For peak f32 matmul today, cuBLAS sgemm remains the right answer.

## Context / cooperative note

This finding comes from **cuda-exploration**, a public third-party
benchmark comparing Rust (cuda-oxide / rust-cuda), CUDA C++ (nvcc), and
cuTile as GPU programming frontends on RTX 5090. The project turned up
**two good things about cuTile** alongside this bug — the `ct.mma` f16
path reaches 172.5 TFLOPS from Python which is an excellent result, and
`ct.reduce_sum` lowers to `UTMALDG.1D` bulk-TMA loads (which nvcc and
rust-cuda do not emit), beating both by 10-12% on memory-bound reductions.
So this issue is narrow: the f32 fallback path of `ct.mma` specifically.

The full repo is at <https://github.com/baladithyab/cuda-exploration>.
Happy to provide additional SASS dumps, NVVM IR dumps, or rerun the
experiment with any suggested flag changes if that would help.
