# Master scaling results (Waves 1+2)

Source: `results/scaling.csv` (210 rows: 7 (impl,kernel) configs × 3 N × 10 iters)

All gpu_ms via cudaEventRecord/cuEventRecord (ADR-0001), nvcc 13.2.78 -arch=sm_120 native (ADR-0002).
Backends:
- `cuda-matmul / matmul` — naive nvcc (1 thread = 1 output, no shared mem)
- `cuda-tiled / matmul_tiled` — register-tiled nvcc (32×32 block + 4×4 register micro-tile)
- `cublas-matmul / sgemm` — cuBLAS sgemm with `CUBLAS_PEDANTIC_MATH` (no TF32)
- `oxide / safe` — cuda-oxide naive, slice-indexed (bounds-checked)
- `oxide / unchecked` — cuda-oxide naive, raw-pointer (no bounds checks)
- `oxide-tiled / safe` — cuda-oxide 16×16 SharedArray tile, slice-indexed
- `oxide-tiled / unchecked` — cuda-oxide 16×16 SharedArray tile, raw-pointer

## N = 1024  (2.15 GFLOP/iter)

| impl | kernel | best ms | median ms | p95 ms | TFLOPS (med) | × naive nvcc |
|---|---|---:|---:|---:|---:|---:|
| cublas-matmul | sgemm | 0.058 | 0.063 | 0.077 | 33.94 | 4.93× |
| cuda-tiled | matmul_tiled | 0.084 | 0.088 | 0.135 | 24.47 | 3.56× |
| oxide-tiled | unchecked | 0.233 | 0.238 | 0.362 | 9.02 | 1.31× |
| oxide-tiled | safe | 0.240 | 0.243 | 0.265 | 8.83 | 1.28× |
| cuda-matmul | matmul | 0.309 | 0.312 | 0.325 | 6.88 | 1.00× |
| oxide | unchecked | 0.327 | 0.330 | 0.332 | 6.51 | 0.95× |
| oxide | safe | 0.709 | 0.766 | 1.172 | 2.80 | 0.41× |

## N = 2048  (17.18 GFLOP/iter)

| impl | kernel | best ms | median ms | p95 ms | TFLOPS (med) | × naive nvcc |
|---|---|---:|---:|---:|---:|---:|
| cublas-matmul | sgemm | 0.262 | 0.273 | 0.289 | 62.98 | 9.95× |
| cuda-tiled | matmul_tiled | 0.500 | 0.514 | 1.441 | 33.44 | 5.28× |
| oxide-tiled | unchecked | 1.751 | 2.602 | 4.300 | 6.60 | 1.04× |
| cuda-matmul | matmul | 2.320 | 2.715 | 3.644 | 6.33 | 1.00× |
| oxide-tiled | safe | 1.780 | 3.092 | 4.034 | 5.56 | 0.88× |
| oxide | unchecked | 2.399 | 3.492 | 4.157 | 4.92 | 0.78× |
| oxide | safe | 6.945 | 7.146 | 8.377 | 2.40 | 0.38× |

## N = 4096  (137.44 GFLOP/iter)

| impl | kernel | best ms | median ms | p95 ms | TFLOPS (med) | × naive nvcc |
|---|---|---:|---:|---:|---:|---:|
| cublas-matmul | sgemm | 1.889 | 2.297 | 3.256 | 59.83 | 9.60× |
| cuda-tiled | matmul_tiled | 3.639 | 4.897 | 5.393 | 28.07 | 4.50× |
| oxide-tiled | unchecked | 16.120 | 17.295 | 18.155 | 7.95 | 1.28× |
| oxide-tiled | safe | 16.962 | 17.878 | 18.211 | 7.69 | 1.23× |
| cuda-matmul | matmul | 19.423 | 22.057 | 25.179 | 6.23 | 1.00× |
| oxide | unchecked | 23.893 | 27.699 | 36.017 | 4.96 | 0.80× |
| oxide | safe | 63.792 | 68.297 | 70.395 | 2.01 | 0.32× |

## TFLOPS vs N (median)

| impl/kernel \ N | 1024 | 2048 | 4096 |
|---|---:|---:|---:|
| cublas-matmul/sgemm | 33.94 | 62.98 | 59.83 |
| cuda-tiled/matmul_tiled | 24.47 | 33.44 | 28.07 |
| oxide-tiled/unchecked | 9.02 | 6.60 | 7.95 |
| oxide-tiled/safe | 8.83 | 5.56 | 7.69 |
| cuda-matmul/matmul | 6.88 | 6.33 | 6.23 |
| oxide/unchecked | 6.51 | 4.92 | 4.96 |
| oxide/safe | 2.80 | 2.40 | 2.01 |

## Tiling speedup (median TFLOPS, tiled / naive)

| backend | N=1024 | N=2048 | N=4096 |
|---|---:|---:|---:|
| nvcc CUDA C++ | 3.56× | 5.28× | 4.50× |
| cuda-oxide unchecked | 1.39× | 1.34× | 1.60× |
| cuda-oxide safe | 3.15× | 2.31× | 3.82× |
