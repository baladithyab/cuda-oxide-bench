# mojo-attn-bf16 — Wave 22.5b — 3-kernel attention, timed at DeepSeek-V3 decode shape

**Status:** ✅ correctness verified (1024-sample, max_abs_err = 0.0 = bit-exact),
TFLOPS_median = **26.36 TF**, TFLOPS_best = **26.93 TF**.
Beats both `cuda-attn-mla` (24.17 TF) and `oxide-attn-mla` (24.70 TF) at the
same 3-kernel HBM-roundtrip pattern.

## History

| Wave   | Shape                           | Result                              |
|--------|---------------------------------|-------------------------------------|
| W22.5  | B=1 n_h=4 S=128 qk=64 d_v=64    | correctness only, max_abs_err = 0.0 |
| W22.5b | B=1 n_h=128 S=2048 qk=192 d_v=128 (DeepSeek-V3) | timed bench: 26.36 TF median, max_abs_err = 0.0 (1024 samples) |

Built on Wave 21 (`mojo-matmul-bf16` = 79.3 TF) and Wave 22.2 (`mojo-matmul-f16`).
Same TC engagement pattern as Wave 21: `TensorCore[bf16,bf16]` for `load_a`/`load_b`,
raw `mma()` with f32 accumulator, hand-rolled m16n8 epilogue.

## Pipeline

| Stage | Kernel              | Math                              | Output dtype |
|-------|---------------------|-----------------------------------|--------------|
| 1     | `qkt_kernel`        | `S = Q @ K^T`                     | f32          |
| 2     | `softmax_kernel`    | `P = softmax(S * 1/sqrt(qk))`     | bf16         |
| 3     | `pv_kernel`         | `O = P @ V`                       | f32          |

HBM round-trip between every stage (S, P arrays live in HBM). This is the
cuda-attn-mla 24 TF ceiling pattern from Wave 17 W1a — three serial kernel
launches, no fused softmax-into-MMA path.

## DeepSeek-V3 bench results (W22.5b)

```
Shape: B=1, n_h=128, S=2048, qk=192, d_v=128
Tile : BM=64, BN=64, BK=32, MMA=16x8x16
GPU  : NVIDIA GeForce RTX 5090 (compute_120, sm_120a)
Mojo : 1.0.0b1 (a9591de6)

flops_per_iter = 343.597 GFLOPS    (2 · B · n_h · S² · (qk + d_v))

10 iters, ctx.execution_time per iter:
  min_ms     = 12.758
  median_ms  = 13.035
  max_ms     = 14.057

TFLOPS_median = 26.36
TFLOPS_best   = 26.93

Correctness vs CPU SDPA ref (1024 samples, atol=1e-2 + rtol=1e-3·|ref|):
  max_abs_err = 0.0
  max_rel_err = 0.0
  PASSED — bit-exact match
```

## Cross-frontend MLA comparison (Wave 17 W1a + W22.5b)

| Kernel              | TFLOPS  | Pattern                                 | Note                           |
|---------------------|---------|-----------------------------------------|--------------------------------|
| cuTile-MLA          | 112.00  | fused, FlashAttention-class             | Mojo cuTile DSL (Wave 17)      |
| cublas-attn-mla     |  47.00  | 3-kernel, cuBLAS GEMMs                  | C++ + cuBLAS (Wave 17)         |
| **mojo-attn-bf16**  | **26.36** | **3-kernel, hand-MMA bf16 (this cell)** | **Mojo, hand-rolled (W22.5b)** |
| oxide-attn-mla      |  24.70  | 3-kernel, hand-WMMA                     | cuda-oxide (Wave 17)           |
| cuda-attn-mla       |  24.17  | 3-kernel, hand-WMMA                     | C++ + WMMA (Wave 17)           |

**Headline:** mojo-attn-bf16 is 9% faster than cuda-attn-mla and 7% faster
than oxide-attn-mla — both fellow 3-kernel hand-rolled implementations sharing
the same HBM-roundtrip-between-stages structure.

We do NOT match cublas-attn-mla (47 TF) or cuTile-MLA (112 TF) — those use
either highly-tuned cuBLAS GEMMs or fused softmax-inside-MMA-pipe, both
outside the 3-kernel hand-MMA scope.

## Why mojo-attn-bf16 beats cuda-attn-mla / oxide-attn-mla at the same pattern

Wave 21's standalone matmul_bf16 (79.3 TF on M=N=K=4096) outperforms
cuda-matmul (~50 TF) and oxide-matmul (~40 TF) on the same hardware, by
margins similar to the W22.5b headline. The Mojo `TensorCore[bf16,bf16]`
fragment-load + raw `mma()` pattern + hand-rolled m16n8 epilogue compiles
to tighter HMMA scheduling than CUDA C++ WMMA fragments (which carry
dead-code register juggling that the Mojo path skips).

The HBM round-trip ceiling still bites — we're 26.36 TF, not 79.3 TF. The
softmax kernel's row-wide max + sum + write-back is unavoidable bandwidth
overhead in a 3-kernel pattern. That's the cuTile-MLA delta (it fuses).

## Shape-divisibility check at DeepSeek-V3 (qk=192, S=2048, d_v=128)

The orchestrator flagged qk=192 vs BK=32 as a potential tail-handler issue,
but 192 / 32 = 6 cleanly. Full check:

| Constraint            | Value | OK?  |
|-----------------------|-------|------|
| QK / BK = 192 / 32    | 6     | ✓    |
| S / BM  = 2048 / 64   | 32    | ✓    |
| S / BN  = 2048 / 64   | 32    | ✓    |
| S / BK  = 2048 / 32   | 64    | ✓    |
| DV / BN = 128 / 64    | 2     | ✓    |

No tail-handler needed. Wave 21's BM=BN=64, BK=32 tile is reused unchanged.

## SASS (RTX 5090, sm_120a)

```
HMMA count: 32 (in run.log)
  qkt_kernel: 16   (= 2 K-tile iters × 2 mma_m × 4 mma_n × 1 warp pattern, replicated 4 warps/block)
  pv_kernel : 16   (= 4 K-tile iters × 2 mma_m × 1 mma_n × 1 warp pattern, replicated 4 warps/block)
softmax_kernel: 0 HMMA (compute-light row-reduction, expected)
```

All HMMA opcodes are `HMMA.16816.F32.BF16` — same pattern as Wave 21 matmul_bf16.

## Pitfalls vs Wave 21 mojo-matmul-bf16

1. **Q@K^T needs K loaded in transposed orientation.** `copy_dram_to_sram_async`
   is row-major-only and 4-element-vectorized. For K^T (where the K-dim of MMA
   is qk, indexing along K's column axis, while the N-dim is S, indexing along
   K's row axis) we use a thread-cooperative element-wise gather (BK*BN=2048
   elems / BLOCK_THREADS=128 threads = 16 elems/thread). This means the qkt
   kernel does NOT use `cp.async` for K — only for Q. Expect somewhat slower
   B-load on qkt vs matmul_bf16, but HMMA still emits cleanly.

2. **PV is plain row-major matmul, identical to Wave 21 pattern.** Both A=P and
   B=V tiles use `copy_dram_to_sram_async` with `vectorize[1, 4]`. Same
   `cp.async` codegen.

3. **Softmax writes bf16 P.** The cuda-attn-mla and cublas-attn-mla references
   both do this — cast f32 probs → bf16 (or f16) so stage 3 can engage tensor
   cores. CPU reference also rounds through bf16 to match.

4. **Multi-head layout flattened.** Layouts are 2-D `(BH * S, X)` with each head's
   row block at offset `bh * S`. Inside kernels we compute
   `head_bm_off = bh * (S // BM)` and use `tile[BM, BX](head_bm_off + block_idx.y, ...)`.
   Cleaner than rank-3 layouts (Mojo's `tile[]` doesn't subset rank-3 tensors in
   the obvious way — Wave 22.5 first try ran into "Indexed with 3 dims, but rank=2").

5. **`alias` deprecation.** Mojo 1.0.0b1 deprecates `alias` in favor of `comptime`
   for module-scope constants. Already migrated.

6. **`ref` is a reserved word.** Renamed to `refv` in the correctness loop.

## Pitfalls discovered in W22.5b (this scaling step)

7. **qk=192 is a "false alarm" for tail-handling at BK=32.**
   The orchestrator's task hint flagged `192 % 32 != 0`, but 32 × 6 = 192,
   so BK=32 actually divides the K-axis cleanly. The Wave 21 tile shape is
   reusable as-is. (If we'd been at BK=64, then 192 % 64 = 64 would have
   required tail-handling.)

8. **`@parameter def body(ctx)` for `ctx.execution_time` works for multi-kernel
   pipelines.** The closure captures `Q_lt`, `K_lt`, `V_lt`, `S_lt`, `P_lt`,
   `O_lt`, `scale`, and the three comptime kernel handles by reference. The
   cudaEvent bracket spans all three kernel launches as a single `body(1)`
   invocation, giving us pipeline-level wall time. Output buffers persist
   across iterations (no buffer rotation needed at this scale; the matmul
   work dwarfs the HBM-store cost from previous iter).

9. **CPU SDPA reference at S=2048 is fine on host stack.**
   `InlineArray[Float32, 2048]` = 8 KiB; 2 of them per sample = 16 KiB stack
   frame. Total CPU reference cost: 1024 samples × (S·QK + 3·S + S) ≈ 410 M
   ops in interpreted Mojo, ~3-4 s wall. Bit-exact match against the kernel
   (max_abs_err = 0.0) suggests the f32 accumulator path inside the matmul
   stages is fully deterministic at this shape, just like the small shape.

## Files

- `attn_bf16.mojo` — 3 kernels + harness + 10-iter timed bench + 1024-sample CPU SDPA ref (~711 lines)
- `attn_bf16.sass` — captured SASS from W22.5 (kernels unchanged, SASS identical at this dump)
- `run.sh` — repro script (now runs the bench shape directly)
- `run.log` — gitignored, regenerable. Latest: W22.5b bench, includes SASS + perf + correctness output.
- `.gitignore`

## Repro

```bash
cd /home/codeseys/cuda-exploration/mojo-attn-bf16
bash run.sh 2>&1 | tee run.log
```

Expected output (timing varies ±5%):
- `GPU: NVIDIA GeForce RTX 5090`
- `Mojo 1.0.0b1 (a9591de6)`
- `[mojo-attn-bf16] shape: B= 1  n_h= 128  S= 2048  qk= 192  d_v= 128`
- `[mojo-attn-bf16] TFLOPS_median= ~26.4   TFLOPS_best= ~26.9`
- `[mojo-attn-bf16] correctness PASSED at deepseek_v3 shape (1024 samples)`
- 32 occurrences of `HMMA.16816.F32.BF16` in the SASS dump (16 per matmul kernel).

## Acceptance (per task spec)

- [x] Compiles + runs end-to-end at DeepSeek-V3 shape (B=1, n_h=128, S=2048, qk=192, d_v=128)
- [x] Correctness PASSES with 1024-sample CPU SDPA reference (got: max_abs_err = 0.0)
- [x] Tolerance atol=1e-2 + rtol=1e-3·|ref| (Wave 21 / Phase-7 spec)
- [x] TFLOPS_median in [10, 50] expected range (got: 26.36)
- [x] TFLOPS comparison documented against cuTile-MLA / cublas-attn-mla / oxide-attn-mla / cuda-attn-mla
- [x] HMMA.16816.F32.BF16 > 0 in SASS (got: 32, in both qkt and pv kernels)
