# mojo-attn-bf16 — Wave 22.5 — 3-kernel attention with bf16 matmul stages

**Status:** ✅ correctness verified at small shape, SASS shows HMMA.16816.F32.BF16 in both matmul kernels, end-to-end pipeline runs cleanly.
**Authoring + correctness only.** Orchestrator runs the timed bench at the DeepSeek-V3 decode shape (B=1, n_h=128, S=2048, qk=192, d_v=128) serially.

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

## Correctness shape and result

- Shape: B=1, n_h=4, S=128, qk=64, d_v=64
  (all multiples of BM/BN/BK = 64/64/32, no boundary guards needed)
- 256 samples checked against an inline numpy-style SDPA reference
  (Q@K^T → softmax → cast through bf16 → @V, identical algorithm)
- **max_abs_err = 0.0** — bit-exact match against the inline reference.
  The reference rounds the post-softmax probs through bf16 to match the
  kernel's intermediate dtype, and the f32 accumulators of the matmul stages
  match between GPU and CPU at this small problem size.

## SASS (RTX 5090, sm_120a)

```
HMMA count: 32
  qkt_kernel: 16   (= 2 K-tile iters × 2 mma_m × 4 mma_n × 1 warp pattern, replicated 4 warps/block)
  pv_kernel : 16   (= 4 K-tile iters × 2 mma_m × 1 mma_n × 1 warp pattern, replicated 4 warps/block)
softmax_kernel: 0 HMMA (compute-light row-reduction, expected)
```

All HMMA opcodes are `HMMA.16816.F32.BF16` — same pattern as Wave 21 matmul_bf16.

## TFLOPS estimate at DeepSeek-V3 shape

Per-shape FLOPS = `2 · B · n_h · S² · (qk + d_v) = 2 · 1 · 128 · 2048² · (192+128) ≈ 343.6 GFLOPS`.

| Reference at this shape   | TFLOPS |
|---------------------------|--------|
| cuda-attn-mla (WMMA)      | 24.17  |
| cuBLAS-attn-mla           | 47     |
| cuTile-attn-mla (fused)   | 112    |

This kernel uses the same 3-kernel HBM-round-trip structure as cuda-attn-mla.
Expected: **~20–25 TF** — we share the cuda-attn-mla ceiling. We MIGHT do
slightly better than cuda-attn-mla because Wave 21's matmul_bf16 standalone
(79.3 TF) outperforms cuda-matmul (~50 TF) at the bare-matmul level, but the
softmax phase + the qkt's transposed-K manual gather cap the upside.

We will NOT match cuTile-attn-mla (112 TF) — that requires fused
softmax-inside-MMA-pipe, which is outside the 3-kernel scope.

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

## Files

- `attn_bf16.mojo` — 3 kernels + harness + inline numpy-style CPU reference (~600 lines)
- `attn_bf16.sass` — captured SASS, shows `HMMA.16816.F32.BF16 × 32` on `.target sm_120a`
- `run.sh` — repro script
- `run.log` — gitignored, regenerable
- `.gitignore`

## Repro

```bash
cd /home/codeseys/cuda-exploration/mojo-attn-bf16
bash run.sh 2>&1 | tee run.log
```

Expected output:
- `GPU: NVIDIA GeForce RTX 5090`
- `Mojo 1.0.0b1 (a9591de6)`
- `[mojo-attn-bf16] shape: B= 1  n_h= 4  S= 128  qk= 64  d_v= 64`
- `[mojo-attn-bf16] correctness: max_abs_err= 0.0  max_rel_err= 0.0`
- `[mojo-attn-bf16] correctness PASSED at small shape`
- 32 occurrences of `HMMA.16816.F32.BF16` in the SASS dump (16 per matmul kernel).

## Acceptance (per task spec)

- [x] max_abs_err ≤ 1e-2 vs numpy-style SDPA reference (got: 0.0)
- [x] HMMA.16816.F32.BF16 > 0 in SASS (got: 32, in both qkt and pv kernels)
- [x] Kernel runs end-to-end at small shape
