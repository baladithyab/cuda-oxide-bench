# Wave 1 size-scaling results
Source: `results/scaling.csv` (90 rows: 6 (impl,kernel) configs × 3 N × 5-10 iters)
All gpu_ms values measured via cudaEventRecord/cuEventRecord (ADR-0001).
Backend: cuda = nvcc 13.2.78 -arch=sm_120 native; oxide = cuda-oxide v0.1.0.

## N = 1024  (2.15 GFLOP/iter)

| impl | kernel | best ms | median ms | p95 ms | TFLOPS (med) | vs nvcc |
|---|---|---:|---:|---:|---:|---:|
| cuda | matmul | 0.309 | 0.312 | 0.325 | 6.879 | 1.00× |
| oxide | unchecked | 0.327 | 0.330 | 0.332 | 6.507 | 0.95× |
| oxide | safe | 0.709 | 0.766 | 1.172 | 2.804 | 0.41× |

## N = 2048  (17.18 GFLOP/iter)

| impl | kernel | best ms | median ms | p95 ms | TFLOPS (med) | vs nvcc |
|---|---|---:|---:|---:|---:|---:|
| cuda | matmul | 2.320 | 2.715 | 3.644 | 6.328 | 1.00× |
| oxide | unchecked | 2.399 | 3.492 | 4.157 | 4.920 | 0.78× |
| oxide | safe | 6.945 | 7.146 | 8.377 | 2.404 | 0.38× |

## N = 4096  (137.44 GFLOP/iter)

| impl | kernel | best ms | median ms | p95 ms | TFLOPS (med) | vs nvcc |
|---|---|---:|---:|---:|---:|---:|
| cuda | matmul | 19.423 | 22.057 | 25.179 | 6.231 | 1.00× |
| oxide | unchecked | 23.893 | 27.699 | 36.017 | 4.962 | 0.80× |
| oxide | safe | 63.792 | 68.297 | 70.395 | 2.012 | 0.32× |

## TFLOPS vs N (median)

| impl/kernel \ N | 1024 | 2048 | 4096 |
|---|---:|---:|---:|
| cuda/matmul | 6.879 | 6.328 | 6.231 |
| oxide/safe | 2.804 | 2.404 | 2.012 |
| oxide/unchecked | 6.507 | 4.920 | 4.962 |
