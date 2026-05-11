# Wave 15.1 — cuTile fused GQA attention — ANALYSIS

**Cell:** `cutile-attn-gqa/`
**Frontend:** cuda-tile 1.3.0 (Python tile DSL)
**Mechanism:** GQA (Grouped-Query Attention), Llama-2/3 style
**Target HW:** NVIDIA RTX 5090 (sm_120, Blackwell consumer)
**Bench shape:** Llama-3-8B — `batch=1, seq=2048, n_q=32, n_kv=8, d_head=128`
**Status:** ✅ correctness OK, ✅ TC-engaged, **165 TFLOPS best / 164.9 TFLOPS median**

## Headline

| metric                         | value                      |
|--------------------------------|---------------------------:|
| best  gpu_ms (10 iters)        | **0.415 ms**               |
| median gpu_ms (10 iters)       | **0.417 ms**               |
| best  TFLOPS                   | **165.5 TFLOPS**           |
| median TFLOPS                  | **164.9 TFLOPS**           |
| ratio vs cuBLAS hgemm (218 TF) | **75.9%**                  |
| vs cuda-attn-gqa (nvcc)        | *TBD by orchestrator*      |
| vs cublas-attn-gqa (3-kernel)  | *TBD by orchestrator*      |
| HMMA.16816.F32 instructions    | 256 (in cubin)             |
| MUFU.EX2 instructions          | 68 (softmax exp path)      |
| correctness vs PyTorch SDPA    | max_abs 4.5e-5 @ bench shape (f16 tol 5e-3) |

**One-paragraph verdict:** cuTile's fused-attention path lands at ~76% of
cuBLAS hgemm peak on a single Llama-3-8B forward-attention launch. The
entire FlashAttention-2 algorithm — QK^T via ct.mma, online rescaling
softmax in registers, PV via ct.mma, final normalize — compiles cleanly
into one kernel with **256 HMMA.16816.F32 tensor-core instructions** and
**zero intermediate HBM traffic** for the (seq × seq) attention matrix.
All pitfalls were anticipated from Waves 12–14; first-compile correctness
and first-run perf were both within target.

## Kernel structure

One `@ct.kernel` function, built by a Python-closure factory
`make_gqa_kernel(BLOCK_M, BLOCK_N, D_HEAD, SEQ, N_Q, N_KV)` so that
tile-shape integers are captured as Python constants (the
`ct.Constant[int]` launch-arg path is broken for tile shapes in cuTile
1.3.0 — this was the Wave 13.1 reproduction).

### Layout

Q/K/V/O are passed as 2D arrays flattened over `(batch, head, seq)`:
- `Q : (B·n_q·S, D)` f16
- `K : (B·n_kv·S, D)` f16
- `V : (B·n_kv·S, D)` f16
- `O : (B·n_q·S, D)` f16

The GQA head-remap `h_kv = h_q // groups` becomes a single integer divide
inside the kernel on `ct.bid(0)`. No dynamic array slicing, no pointer
arithmetic — the 2D `tiled_view((BLOCK_M, D))` handles everything through
a static tile shape, and each head's `S/BLOCK_M` rows in the tile space
are addressed by `bid0 * SEQ_TILES + block_idx`.

### Grid

- `bid0 ∈ [0, B·n_q)`  (flattened batch × query-head index)
- `bid1 ∈ [0, S/BLOCK_M)` (query-block index)

At bench shape: `grid = (32, 32)` → 1024 CTAs, which fills the 5090
perfectly (170 SMs × 6 concurrent blocks ≈ 1020 max concurrency).

### Online softmax (the heart of FlashAttention-2)

```
m_i = -inf  (BLOCK_M, 1)   # running row max
l_i = 0     (BLOCK_M, 1)   # running row sum-of-exp
O_i = 0     (BLOCK_M, D)   # running output accumulator

for each K/V block:
    S        = Q @ K^T * scale                  (BLOCK_M, BLOCK_N) f32
    m_new    = max(m_i, rowmax(S))              (BLOCK_M, 1)  f32
    alpha    = exp(m_i - m_new)                 (BLOCK_M, 1)  f32   <- rescale factor
    P        = exp(S - m_new)                   (BLOCK_M, BN) f32
    l_i      = alpha * l_i + rowsum(P)
    O_i      = alpha * O_i  +  P_f16 @ V         <- PV via ct.mma
    m_i      = m_new

O = O_i / l_i                                    (final normalize)
```

Why this is stable and lossless relative to naive softmax: the running
max `m_i` is monotonically non-decreasing, `exp(m_i - m_new) ≤ 1`, and
subtracting `m_new` before exp guarantees the exp argument is `≤ 0`. No
overflow possible, no NaNs. The rescale of `O_i` and `l_i` by `alpha`
compensates for the max-shift, so the final `O_i / l_i` is mathematically
identical to a single-pass softmax on the full row — just computed
block-wise in registers.

### TC engagement (cubin evidence)

Export path via `export_kernel(..., gpu_code="sm_120", output_format="cubin")`
lifted from `cutile-matmul-tiled-mixed/main.py`. Disassembly (`cuobjdump
--dump-sass gqa_fwd_fused.cubin > gqa_fwd_fused.sass`):

- **256 `HMMA.16816.F32`** — the 16×8×16 tensor-core instruction at
  f16×f16 → f32 accumulator. Present for both the `Q @ K^T` and `P @ V`
  matmuls inside the K/V loop. Count lines up with 32 iterations of the
  inner loop × 4 mmas per iteration × 2 for QK^T and PV.
- **68 `MUFU.EX2`** — the exp2 special-function instruction used by
  `ct.exp` in the softmax. Two per K/V block for the `alpha = exp(m_i -
  m_new)` and `P = exp(S - m_new)` calls, amortized across register
  tiles.
- **608 `FFMA` / 262 `FMUL` / 214 `FADD`** — the scalar-math glue for
  rescaling, scaling-by-inv-sqrt-d, and the final divide.

Ratio HMMA/total-instr ≈ 2% by count, but HMMA dominates wall-clock (each
HMMA retires 128 f16 FMA = 256 flops in one issue slot).

## Ratios

- **cuBLAS hgemm peak** was 218 TFLOPS (Wave 13, `cublas-half-precision/`).
  cuTile fused GQA hits **75.9% of that** — impressive given that it's
  also doing softmax (expensive special-functions) inside the same kernel
  that cuBLAS's hgemm doesn't have to do at all.
- **cuTile matmul peak (Wave 13.1 mixed f16×f16→f32acc)** was 172.5 TF.
  Fused attention hits **95.9% of single-matmul cuTile peak** — i.e.,
  the softmax work is almost free relative to the mma pipeline. This is
  the whole promise of FlashAttention-2: once the QK^T result is in
  registers, keeping it there through softmax and using it as the PV
  LHS costs one 10× cheaper exp + FMA rescale, not a full HBM round
  trip.
- **cuda-attn-gqa / cublas-attn-gqa**: *TBD by orchestrator* (those cells
  run in parallel during this session; see the wave-15 summary for the
  cross-frontend table).

## Pitfalls encountered (for future-wave reference)

1. **`ct.store(o_view, ...)` with a tiled_view argument fails** — the
   store primitive expects the raw `Array`, not a `TiledView`. Use
   `o_view.store((row, 0), tile)` *or* `ct.store(O, (row, 0), tile)` with
   the bare array + tile-space index. Both are in the 1.3.0 API.

2. **`ct.max(..., axis=1, keepdims=True)` works and returns the
   (BLOCK_M, 1) f32 shape needed for broadcast.** No manual
   `expand_dims` required. Same for `ct.sum(..., axis=1, keepdims=True)`.

3. **Broadcasting a `(BLOCK_M, 1)` scalar-per-row tile against a
   `(BLOCK_M, D)` tile in arithmetic ops is implicit** — cuTile follows
   NumPy broadcasting rules without a manual `ct.broadcast_to`. Both
   `alpha * O_i` and `S - m_new` compiled and ran correctly on first
   attempt.

4. **`ct.transpose(k_tile)` before the QK^T mma works** — cuTile sees
   through the transpose to pick the TC-layout, no need to pre-transpose
   K in host memory.

5. **P must be cast to f16 before the PV mma** — cuTile's `ct.mma`
   requires matching input dtypes and doesn't auto-promote. `p_f16 =
   p.astype(ct.float16)` is the one-liner.

6. **Kernel compile time is ~1.2 s first launch** (twice the single-matmul
   cost from Wave 13.1). Two warmup iters absorb it; reported TFLOPS
   exclude compile cost by construction.

## What would close the remaining 24% gap to cuBLAS hgemm peak?

- **Larger BLOCK_M** — we tried 128×64 (perf drop to 46 TF, likely
  register spill) and 128×128 (perf drop to 31 TF, same story). The
  sweet spot at 64×64 is register-budget-limited; getting to 128×64
  without spills would require explicit register-reuse hints that
  cuTile 1.3.0 doesn't yet expose.
- **Persistent-kernel / pipelined K/V loads**. Current kernel does a
  plain synchronous load inside the loop; a pipelined or persistent
  variant would overlap HBM load with HMMA compute. cuTile has no
  user-facing API for this in 1.3.0 (everything is implicit through
  the compiler's scheduler).
- **Epilogue fusion of the final divide**. `O_i / l_i` runs as a
  separate pass over the D-dim after the loop; merging it into the
  store-time epilogue would save one register pass. Marginal at d=128.

None of these are necessary to ship the cell. At 165 TFLOPS / 0.4 ms
per launch, cuTile fused GQA is within 2× of what cuBLAS-3-kernel is
expected to produce (130–180 TF, per the Wave 15 PLAN predictions) and
within 1.1× of single-kernel cuTile hgemm.

## Files

- `main.py` — kernel factory, smoke, bench, cubin export.
- `run.sh` — one-shot: smoke → bench → cubin → SASS-grep.
- `results.csv` — per-iter bench data (schema: `impl, kernel, batch, seq, n_q, n_kv, d_head, block_m, block_n, iter, gpu_ms, tflops`).
- `run.log` — stdout/stderr from the `./run.sh` invocation.
- `gqa_fwd_fused.cubin` — compiled sm_120 kernel (~483 KB).
- `gqa_fwd_fused.sass` — full disassembly (~12.5k lines).
