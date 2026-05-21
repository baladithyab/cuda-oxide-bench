# mojo-matmul-bf16 — hand-rolled bf16-in/f32-acc tiled matmul on RTX 5090 sm_120

**Status:** ✅ Wave 21 SHIPPED 2026-05-21.
**Headline:** 79.3 TFLOPS @ M=N=K=4096 (median of 10 iters), correctness PASSED.

See [results/wave21-summary.md](../results/wave21-summary.md) for the full writeup.

## Quick repro

```bash
cd /home/codeseys/cuda-exploration/mojo-matmul-bf16
bash run.sh 2>&1 | tee run.log
```

Expected output:
- `GPU: NVIDIA GeForce RTX 5090`
- `Mojo 1.0.0b1 (a9591de6)`
- `[mojo-matmul-bf16] M=N=K= 4096 ... TFLOPS_median= 79.X`
- `[mojo-matmul-bf16] correctness: max_abs_err= ~2e-3`
- `[mojo-matmul-bf16] correctness PASSED at M=N=K= 4096`

## Files

- `matmul_bf16.mojo` — kernel + harness (~360 lines)
- `matmul_bf16.sass` — captured SASS, shows `HMMA.16816.F32.BF16 × 16` on `.target sm_120a`
- `run.sh` — repro script
- `run.log` — gitignored, regenerable

## Strategy (the plan-deviation worth knowing)

Plan said "drop completely to raw `mma()` and `ld_matrix`". Execution found a cleaner hybrid:

1. `TensorCore[bf16, bf16, Index(16, 8, 16)]()` for `load_a` and `load_b` ONLY. (The same-dtype constraint bites in `mma_op` and `store_d` but NOT in `load_a/load_b`.)
2. Raw `mma(d, a, b, c)` from `std.gpu.compute.mma` with bf16-in / f32-out.
3. Hand-rolled epilogue per PTX 9.7.13.4.8 m16n8 distribution.

This is the recommended pattern for future bf16/f16/FP8 lanes — the wrapper's load functions are well-tested and produce optimal `ldmatrix.x4`, only the mma+store path needs hand-rolling.

## Cross-frontend position at 4096³

Mojo bf16 (this) is **49.6% of cuTile bf16 (159.95 TF)** and **36.2% of cuBLAS bgemm (219.3 TF)**. Gap explained by `cp.async`-vs-TMA loads (Mojo doesn't auto-emit `UTMALDG` through `copy_dram_to_sram_async`); see SASS analysis in the wave summary.
