# Master scaling results (Waves 1+2+3, post-Phase-8 amendments)

Source: `results/scaling.csv` (240 rows: 8 (impl,kernel) configs × 3 N × 10 iters)

All gpu_ms via `cudaEventRecord` / `cuEventRecord` (ADR-0001).
`-arch=sm_120` native (ADR-0002), nvcc 13.2.78, cuBLAS 13.4.0 with `CUBLAS_PEDANTIC_MATH`.
**libNVVM via `CUDA_HOME=/usr/local/cuda`** to avoid the system `libnvvm.so.4` 7.0.1 shadow (see `docs/experiments/libnvvm-corrigendum.md`).

**Variance disclosure (Phase 8 review):** All cells include CV% and IQR. At sub-ms kernels (N=1024) CVs are 0.5-3% — tight. At N=4096 with thermal noise on this WSL2 hosting, CVs widen to 5-15%, and iter-by-iter drift is visible. We do not lock GPU clocks (no permissions on this shared system); readers should treat single-decimal-place TFLOPS at N=4096 as having ±0.5 TF uncertainty.

## Backends
- `cuda-matmul / matmul` — naive nvcc, 1 thread = 1 output, no shared mem
- `cuda-tiled / matmul_tiled` — register-tiled nvcc (32×32 block + 4×4 register micro-tile + K=16 unroll)
- `cublas-matmul / sgemm` — cuBLAS sgemm (`CUBLAS_PEDANTIC_MATH`, no TF32)
- `oxide / safe` — cuda-oxide naive, slice-indexed (Rust bounds-checked)
- `oxide / unchecked` — cuda-oxide naive, raw-pointer reads
- `oxide / fmuladd` — cuda-oxide naive with `core::intrinsics::fmuladdf32` (post-link emits `fma.rn.f32`)
- `oxide-tiled / safe` — cuda-oxide 16×16 SharedArray tile, slice-indexed (no register tiling)
- `oxide-tiled / unchecked` — cuda-oxide 16×16 SharedArray tile, raw-pointer (no register tiling)

## N = 1024  (2.15 GFLOP/iter)

| impl | kernel | best ms | median ms | p95 ms | IQR ms | CV % | TFLOPS (med) | × naive nvcc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cublas-matmul | sgemm | 0.058 | 0.063 | 0.077 | 0.004 | 7.6 | 33.94 | 4.93× |
| cuda-tiled | matmul_tiled | 0.084 | 0.088 | 0.135 | 0.004 | 16.7 | 24.47 | 3.56× |
| oxide-tiled | unchecked | 0.233 | 0.236 | 0.237 | 0.002 | 0.5 | 9.09 | 1.32× |
| oxide-tiled | safe | 0.240 | 0.242 | 0.246 | 0.001 | 0.7 | 8.89 | 1.29× |
| cuda-matmul | matmul | 0.309 | 0.312 | 0.325 | 0.003 | 1.5 | 6.88 | 1.00× |
| oxide | safe | 0.305 | 0.314 | 0.335 | 0.005 | 2.5 | 6.84 | 0.99× |
| oxide | fmuladd | 0.307 | 0.315 | 0.326 | 0.005 | 1.8 | 6.82 | 0.99× |
| oxide | unchecked | 0.311 | 0.316 | 0.404 | 0.019 | 8.6 | 6.80 | 0.99× |

## N = 2048  (17.18 GFLOP/iter)

| impl | kernel | best ms | median ms | p95 ms | IQR ms | CV % | TFLOPS (med) | × naive nvcc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cublas-matmul | sgemm | 0.262 | 0.273 | 0.289 | 0.007 | 2.6 | 62.98 | 9.95× |
| cuda-tiled | matmul_tiled | 0.500 | 0.514 | 1.441 | 0.282 | 46.7 | 33.44 | 5.28× |
| oxide-tiled | safe | 1.768 | 2.150 | 2.826 | 0.502 | 15.8 | 7.99 | 1.26× |
| oxide-tiled | unchecked | 1.732 | 2.380 | 2.878 | 0.946 | 20.2 | 7.22 | 1.14× |
| oxide | safe | 2.284 | 2.404 | 4.978 | 0.289 | 32.4 | 7.15 | 1.13× |
| oxide | fmuladd | 2.298 | 2.573 | 4.811 | 0.626 | 34.6 | 6.68 | 1.06× |
| oxide | unchecked | 2.289 | 2.590 | 4.806 | 0.633 | 27.3 | 6.63 | 1.05× |
| cuda-matmul | matmul | 2.320 | 2.715 | 3.644 | 0.852 | 16.0 | 6.33 | 1.00× |

## N = 4096  (137.44 GFLOP/iter)

| impl | kernel | best ms | median ms | p95 ms | IQR ms | CV % | TFLOPS (med) | × naive nvcc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cublas-matmul | sgemm | 1.889 | 2.297 | 3.256 | 0.766 | 18.9 | 59.83 | 9.60× |
| cuda-tiled | matmul_tiled | 3.639 | 4.897 | 5.393 | 0.876 | 12.9 | 28.07 | 4.50× |
| oxide-tiled | unchecked | 15.882 | 17.371 | 18.559 | 0.532 | 4.1 | 7.91 | 1.27× |
| oxide-tiled | safe | 17.030 | 17.917 | 21.452 | 2.055 | 8.2 | 7.67 | 1.23× |
| cuda-matmul | matmul | 19.423 | 22.057 | 25.179 | 2.177 | 8.2 | 6.23 | 1.00× |
| oxide | unchecked | 22.050 | 24.236 | 31.155 | 3.041 | 12.5 | 5.67 | 0.91× |
| oxide | safe | 22.158 | 25.245 | 32.171 | 8.005 | 14.9 | 5.44 | 0.87× |
| oxide | fmuladd | 22.782 | 25.879 | 29.191 | 2.569 | 7.9 | 5.31 | 0.85× |

## TFLOPS vs N (median)

| impl/kernel \ N | 1024 | 2048 | 4096 |
|---|---:|---:|---:|
| cublas-matmul/sgemm | 33.94 | 62.98 | 59.83 |
| cuda-tiled/matmul_tiled | 24.47 | 33.44 | 28.07 |
| oxide-tiled/unchecked | 9.09 | 7.22 | 7.91 |
| oxide-tiled/safe | 8.89 | 7.99 | 7.67 |
| cuda-matmul/matmul | 6.88 | 6.33 | 6.23 |
| oxide/fmuladd | 6.82 | 6.68 | 5.31 |
| oxide/unchecked | 6.80 | 6.63 | 5.67 |
| oxide/safe | 6.84 | 7.15 | 5.44 |

## Naive cuda-oxide vs nvcc (with variance)

> Phase 8 reviewer #3: at N=1024 the kernels run in 0.3 ms and ~5-50 µs of launch+timer overhead
> may be a non-trivial fraction of the measurement. Treat the 'all four within 1%' result as
> evidence of *kernel parity* (or being launch-overhead-bound), NOT proof of compiler equivalence.
> N=4096 (where the kernel runs >20 ms) is more diagnostic of compiler differences.

| N | nvcc median TF | oxide/safe TF | oxide/unchecked TF | oxide/fmuladd TF | safe/nvcc | unchk/nvcc |
|---|---:|---:|---:|---:|---:|---:|
| 1024 | 6.88 | 6.84 | 6.80 | 6.82 | 0.99× | 0.99× |
| 2048 | 6.33 | 7.15 | 6.63 | 6.68 | 1.13× | 1.05× |
| 4096 | 6.23 | 5.44 | 5.67 | 5.31 | 0.87× | 0.91× |

## Tiling speedup (median TFLOPS, tiled / naive)

> **Caveat (Phase 8 review #2): the nvcc-tiled vs oxide-tiled gap has 4 components, in approximate order of magnitude:**
> 1. **Block geometry + register microtile**: nvcc-tiled = 32×32 block + 4×4 register tile per thread; oxide-tiled = 16×16 block, 1 output per thread. (~16× more FLOPs per thread for nvcc.)
> 2. **K-loop unrolling**: nvcc unrolls 16-deep; cuda-oxide does not unroll.
> 3. **Default fast-math**: cuda-oxide's `FastmathFlagsAttr::default()` is empty (no `contract` bit), so plain `*+` chains don't fuse.
> 4. **No `ld.global.nc` (read-only cache)** in oxide PTX vs nvcc's restrict-marked loads.
> Calling 'missing FMA' the cause is misleading — fixing only FMA wouldn't close most of the gap. The full algorithm-vs-compiler-vs-hand-tuning attribution is in `oxide-matmul-tiled/ANALYSIS.md`.

| backend | N=1024 | N=2048 | N=4096 |
|---|---:|---:|---:|
| nvcc CUDA C++ | 3.56× | 5.28× | 4.50× |
| cuda-oxide unchecked | 1.34× | 1.09× | 1.40× |
| cuda-oxide safe | 1.30× | 1.12× | 1.41× |
