# cuda-matmul-tc-bf16 — Wave C2.1 analysis

Hand-rolled CUDA C++ WMMA bf16 matmul. Closes the cuda-matmul TC column of
the Rosetta-Stone matmul matrix (alongside the naive FFMA cell, the
register-tiled FFMA cell, the cublas/cublasLt cells, and the Mojo /
oxide / cuTile cells in their respective columns).

**Headline (RTX 5090, sm_120, CUDA 13.2, 50 timed iters @ N=M=K=4096):**

| metric             | value    |
|--------------------|----------|
| best TFLOPS        | **147.53** |
| median TFLOPS      | 146.77   |
| best ms            | 0.932    |
| median ms          | 0.936    |
| correctness        | 1024 / 1024 sampled cells pass |
| worst abs error    | 6.5e-3   (got=6.1520, want=6.1455 at (746,1839)) |
| worst rel error    | 1.1e-3   |
| HMMA.16816.F32.BF16 ops in cubin | **64** |
| FFMA ops in cubin  | **0**    |
| LDGSTS (cp.async)  | 8        |

Acceptance gate (TFLOPS in [50, 220], correctness within atol=2e-1 / rtol=5e-2)
is **PASS**.

## Comparison to the Wave C2 column neighbors

| cell                       | best TF | ratio vs cublas-bf16 (219.24) | ratio vs mojo-matmul-bf16 (79.85) |
|----------------------------|---------|-------------------------------|-----------------------------------|
| cuda-matmul (naive FFMA)   | ~3      | 0.014×                        | 0.038×                            |
| cuda-matmul-tiled (FFMA µ-tile) | ~38 | 0.173×                        | 0.476×                            |
| **cuda-matmul-tc-bf16 (this cell)** | **147.5** | **0.673×** | **1.848×**                |
| cublas-half-precision (bf16) | 219.24 | 1.000×                       | 2.745×                            |

(cuda-matmul / cuda-matmul-tiled numbers are f32 not bf16 — listed for
column-progression context only; the ratios are bf16 vs cublas-bf16.)

So the hand-WMMA cell lands at **67% of cublas-bf16** and **1.85× the Mojo
hand-MMA cell** — squarely between Mojo and cublas as expected for a
clean classical-WMMA implementation that doesn't yet pipeline cp.async
across stages or use Hopper-specific TMA / wgmma.

## Geometry

- **CTA tile**: BM=BN=128, BK=32.
- **Warps per CTA**: 4 (2×2 over the 128×128 output tile).
- **Per-warp output tile**: 64×64.
- **WMMA fragment shape**: m16n16k16, BF16 inputs, F32 accumulator.
- **Per-warp fragments**: 4 (rows) × 4 (cols) = 16 accumulators; per K-tile
  pass each accumulator gets BK/MMA_K = 2 mma_sync calls; total per warp
  per K-tile = 32 HMMA. With 128 K-tiles in a 4096-K matmul that's
  4096 mma_sync calls per warp per CTA, ×4 warps × 32×32 grid = ~16.8 G
  HMMA.16816.F32.BF16 lane-instructions, well-matched to the 64 distinct
  HMMA opcodes in the cubin (= 16 fragments × 2 K-fragments × 2 unrolled
  passes from the compiler).
- **Threads per CTA**: 128, with `__launch_bounds__(128, 2)` to encourage
  occupancy of 2 CTAs/SM.

## Memory staging

A and B are staged from global to shared via `cp.async.cg.shared.global`
with 16-byte (= 8 bf16 element) chunks. Per CTA per K-tile:

- A tile = BM·BK = 4096 bf16 = 8 KiB. 512 16-byte chunks; 128 threads ×
  4 chunks each.
- B tile = BK·BN = 4096 bf16 = 8 KiB. Same 4 chunks per thread.

Single-buffered: `cp.async.commit_group` then `cp.async.wait_all` at the
top of each K-tile pass. This is simpler than a true 2-stage pipeline and
leaves measurable headroom — the next obvious win for closing the gap to
cublas would be double-buffering with `cp.async.wait_group N`.

## SASS shape

`/usr/local/cuda/bin/cuobjdump --dump-sass matmul_tc_bf16`:

- `HMMA.16816.F32.BF16`: **64 occurrences** — Blackwell's BF16-input,
  F32-accumulator tensor-core opcode. Confirms the wmma path lowered to
  the BF16 tensor-core instruction class, not to FFMA emulation.
- `LDGSTS`: 8 — that's the four A and four B `cp.async` calls per thread,
  hoisted out of the inner loop and unrolled.
- No `FFMA`, no scalar accumulation in the inner loop.
- No `LDSM` — the WMMA `load_matrix_sync` path on this arch doesn't emit
  the explicit `ldmatrix` opcode the way a hand-rolled mma.sync kernel
  with `__nv_bfloat162` 8-lane gathers would; the loads land as standard
  `LDS` against the staged shared tile. That's expected for the WMMA
  abstraction layer on sm_120 and is one reason cublas (which uses
  ldmatrix.x4 directly) pulls ahead.

## Numerical behavior

Inputs follow the same `(i % 7)·0.01` / `(i % 11)·0.01` deterministic
pattern as cuda-matmul-tiled, rescaled into bf16's representable range
(values ≤ 0.06). Per-cell f32 partial sums stay well under 2¹², which
is comfortably inside bf16's exponent / f32-accumulator window.

Sample-1024 result: every sample passes the (atol=2e-1 OR rtol=5e-2) gate;
the worst absolute error is 6.5e-3 — three orders of magnitude inside
the 2e-1 tolerance. Tightening to atol=1e-2 would still pass; we keep the
loose gate per the task spec because BF16 is the gating dtype across the
whole row.

## Pitfalls encountered / things to watch

1. **`cp.async.wait_all` vs `wait_group N`.** Going single-buffered means
   the second tile's loads can't overlap the first tile's compute. Two-
   stage doubles the smem footprint (32 KiB combined) but should give
   another ~10-25 TF on this kernel.
2. **No swizzling on shared layout.** `As[BM][BK]` with BK=32 → 64 B per
   row, which means 16-bank conflicts on the WMMA `load_matrix_sync` for
   A (matrix_a row-major). Adding a +8-bf16 pad column or applying the
   Ampere-style XOR swizzle is the next non-pipelining win.
3. **WMMA on sm_120 doesn't emit `ldmatrix`.** Verified empirically
   (`grep LDSM` returns 0). Hand-rolled `mma.sync.aligned.m16n8k16` with
   explicit `ldmatrix.sync.aligned.x4.m8n8.shared.b16` would be the path
   to close the rest of the cublas gap; that's a separate cell (and the
   Mojo one is roughly that shape, but slower because of the m16n8 vs
   m16n16 frag size).
4. **Kept the `b2f` and `lane` compiler warnings.** Both are intentional
   and harmless (`b2f` is reachable for future debug hooks; `lane` is
   captured because future variants will need it for ldmatrix lane
   addressing). Could be silenced with `-diag-suppress=177` in the
   Makefile if the warnings ever break a CI pipeline.
5. **Variance between iters.** Iters bimodally cluster around 0.93 ms
   (~147 TF) and 1.15-1.25 ms (~115 TF). This matches the AGENTS.md note
   about WSL2 + non-clock-locked sm_120. The headline number takes the
   best-of-50 to dodge the desktop-contention slow lane.

## Reproduction

```bash
cd /home/codeseys/cuda-exploration/cuda-matmul-tc-bf16
./run.sh
```

That builds with `-arch=sm_120`, runs 5 warmups + 50 timed iters at
N=4096, samples 1024 cells against an f32 CPU reference, dumps SASS, and
prints HMMA / FFMA / LDGSTS counts.

## Files

- `matmul_tc_bf16.cu` — kernel + bench + correctness, single binary.
- `Makefile`           — `make` builds `matmul_tc_bf16` for sm_120.
- `run.sh`             — driver (build + run + SASS dump + opcode counts).
- `.gitignore`         — ignores artifacts.
- `ANALYSIS.md`        — this file.
- (generated) `matmul_tc_bf16` (binary), `matmul_tc_bf16.sass`,
  `results.csv`, `run.log`, `build.log`.
