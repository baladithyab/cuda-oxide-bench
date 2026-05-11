# Wave 14.1 — cuBLAS half-precision baselines vs cuTile (Wave 13.1)

**Date:** 2026-05-11
**GPU:** NVIDIA GeForce RTX 5090 (sm_120)
**cuBLAS:** 13.4.0 (from CUDA 13.2 toolkit)
**Build:** `/usr/local/cuda/bin/nvcc -ccbin clang-14 -O3 -arch=sm_120 -lcublas`
**Protocol:** N ∈ {1024, 2048, 4096}, 1 warmup + 10 timed iters, `cudaEvent` timing.
Correctness: spot-check at (0,0), (N/2,N/2), (N-1,N-1) vs double-precision CPU reference.
All 27 checks passed (9 per variant).

## 1. Headline table — best TFLOPS per variant per N

| N    | hgemm (f16, f32 acc) | bgemm (bf16, f32 acc) | sgemm-tf32 (f32 i/o, tf32 int) |
|------|---------------------:|----------------------:|-------------------------------:|
| 1024 |   128.56             |   142.48              |    78.67                       |
| 2048 |   167.67             |   170.71              |    89.87                       |
| 4096 |**218.41**            | **219.24**            |   **104.23**                   |

Median TFLOPS @ N=4096: hgemm 212.5, bgemm 217.4, sgemm-tf32 104.1 (low variance for tf32, ~3% variance for hgemm/bgemm driven by single slow iter each).

## 2. The apples-to-apples comparison (N=4096, best iter)

| dtype class     | cuTile (W13.1) | cuBLAS (this wave) | cuTile / cuBLAS |
|-----------------|---------------:|-------------------:|----------------:|
| f16  → f32 acc  |      172.5 TF  |       **218.4 TF** |    **79.0%**    |
| bf16 → f32 acc  |      159.8 TF  |       **219.2 TF** |    **72.9%**    |
| tf32 internal   |       84.0 TF  |       **104.2 TF** |    **80.6%**    |
| pure f32 / pedantic | 8.7 TF (cuTile) | 73.6 TF (W2A sgemm PEDANTIC) | 11.8% |

**Bottom line:** at matched dtype on N=4096, cuTile sits at **73–81% of hand-tuned cuBLAS** for the tensor-core dtype classes. This is the number the task asked for.

## 3. Verdict per dtype class

- **f16 (`cuTile 172.5 / cuBLAS 218.4 = 79.0%`).** This is the headline win and the honest framing for cuTile: within 21% of a library that has seen decades of tuning on NVIDIA tensor cores. Competitive, not dominant. The delta is almost certainly inner-loop scheduling + autotuned tile shapes + cuBLAS-specific algo selection (cuBLAS picks from ~dozens of GEMM kernels per shape). cuTile's single generic codegen path hitting 79% of that is a real accomplishment.

- **bf16 (`cuTile 159.8 / cuBLAS 219.2 = 72.9%`).** Gap slightly larger than f16 (6pp worse). Possible causes: cuTile's bf16 accumulation path may be less mature, or the conversion to/from f32 in the epilogue is less fused. Worth a follow-up microbenchmark — but still clearly in the competitive regime.

- **tf32 (`cuTile 84.0 / cuBLAS 104.2 = 80.6%`).** Nearly identical ratio to f16. TF32 on sm_120 caps at ~104 TF best-iter vs f16's ~218 TF — about half the tensor-core throughput per expected architecture roofline (f16 TC have 2× the per-cycle MMA shape vs tf32 TC). cuTile scales proportionally, suggesting its codegen is doing the right thing relative to hardware, not a tf32-specific bug.

- **pure f32 without TF32 (pedantic).** cuTile's f32 path is still ~9× off cuBLAS sgemm (8.7 vs 73.6 TF), because cuTile's f32 codegen goes through CUDA cores, while the cuBLAS **pedantic** sgemm at 73.6 TF is also forced onto CUDA cores — so the ~9× gap is pure CUDA-core kernel quality, not tensor-core engagement. This is consistent with the cuda-matmul vs cublas-sgemm gap documented in earlier waves.

## 4. Are tensor cores actually engaged?

Yes. Multiple signals confirm:

1. **Raw throughput.** 218 TF on f16 is ~66% of the RTX 5090's marketed ~318 TF f16 tensor-core peak (dense, no sparsity), well above the ~105 TF CUDA-core f16 peak. hgemm on CUDA cores would be ~100 TF max; we're 2× that.
2. **sgemm-tf32 at 104 TF vs pedantic sgemm at 73.6 TF** (W2A baseline). The 42% lift from switching math mode shows the TF32 tensor path is distinct from the pure-f32 CUDA-core path.
3. **bgemm matches hgemm within 0.4 TF at N=4096** (219.2 vs 218.4). Same tensor-core unit, same per-cycle MMA shape — expected.
4. **Numerical pattern.** bf16's relative error (~3e-3 at N=4096) is 10× looser than f16's (~2e-4), as expected from the 7 vs 10 mantissa bits. Both drop accumulator precision from the CPU reference in the right direction, confirming tensor-core accumulation is in play.

## 5. Methodology notes / caveats

- We used `cublasGemmEx` with `CUBLAS_COMPUTE_32F` + `CUBLAS_GEMM_DEFAULT_TENSOR_OP` for both hgemm and bgemm. `CUBLAS_COMPUTE_32F` means f32 accumulator (matches cuTile W13.1's `f32acc` config).
- `CUBLAS_DEFAULT_MATH` on CUDA 13.x + sm_120 already permits tensor-core use for reduced-precision GEMM; the explicit `TENSOR_OP` algo selector just pins the choice.
- For the tf32 variant, `cublasSgemm` + `CUBLAS_TF32_TENSOR_OP_MATH` is the officially supported path. The existing W2A baseline uses `CUBLAS_PEDANTIC_MATH` which disables TF32 — so the 73.6 TF "sgemm" number in the repo is not a tf32 number, it's a strict-f32 number. This wave clarifies that tf32 mode gives ~104 TF, not 73.6.
- Variance: one slow hgemm iter at N=2048 (0.143 ms vs ~0.107 median, +34%) and one slow bgemm iter at N=4096 (0.693 ms vs ~0.632, +10%) — consistent with known WSL2 / unmodified-clock variance documented in `cuda-exploration/AGENTS.md`. Best-iter numbers are the right metric for peak-throughput comparison; median numbers are still within 3% of best.
- Both hgemm and bgemm exceed cuTile in best-iter AND median at every N ≥ 2048. sgemm-tf32 does too. Verdict is stable under metric choice.

## 6. Files produced

- `matmul.cu` — single binary, three benchmarks (hgemm, bgemm, sgemm-tf32), mirrors `cublas-matmul/matmul.cu` structure.
- `results.csv` — 90 rows (3 variants × 3 N × 10 iters), schema `impl,kernel,N,iter,gpu_ms,tflops`.
- `run.log` — full stdout, grep-able for per-iter numbers.
- `.gitignore` — binary + build artifacts.
