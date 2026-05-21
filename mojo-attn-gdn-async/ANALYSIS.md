# Wave C3.4 — `mojo-attn-gdn-async` analysis

**Status.** ✅ SHIPPED 2026-05-21. Compiles, correctness PASSES on both `o`
(max_abs_err = 3.81e-6 vs atol=1e-3) and `S_out` (max_abs_err = 3.73e-9 vs
atol=5e-3). 50-iter timed bench reports **best 610.6 GB/s, median 586.1 GB/s**
(stable across 4 reruns: 607.8–610.6 best).

## Headline result

| frontend            | impl                         | best GB/s |  vs sync baseline | vs cuda-async W22.9 |
|---|---|---:|---:|---:|
| Mojo 1.0.0b1        | mojo-attn-gdn (sync FFMA)    |     320.5 | 1.00× |       1.03× |
| nvcc CUDA 13.2 C++  | cuda-attn-gdn-async (W22.9)  |     311.8 | — |       1.00× |
| **Mojo 1.0.0b1**    | **mojo-attn-gdn-async**      | **610.6** | **1.91× ↑** | **1.96× ↑** |
| (reference)         | cutile-attn-gdn              |     610.0 | — | — |
| (reference)         | cuda-attn-gdn-tma (W22.10)   |    1032.0 | — | — |

**The async-pipe primitive in Mojo is dramatically positive for GDN, lifting
the cell to cuTile's saturation level (610 GB/s) — the same ceiling that
warp-specialized async-transaction barriers hit, achieved with a pure
collective `cp.async` + drain pattern.** This is a 2× speedup over the
synchronous Mojo baseline and reverses the C++ async-pipe -25% regression
into a +96% **lift**.

## Why Mojo wins where cuda::pipeline lost

The two implementations differ at the API/codegen layer in ways that turn
out to matter:

| dimension                      | cuda-attn-gdn-async (W22.9)              | mojo-attn-gdn-async (this)                |
|---|---|---|
| async primitive                | `cuda::pipeline<thread_scope_thread>`    | `copy_dram_to_sram_async` (collective)    |
| pipeline scope                 | per-thread                               | CTA-cooperative                            |
| inflight strategy              | **4-stage ring**, producer/consumer interleave | **single bulk load + drain** before compute |
| smem footprint (state)         | 64 KiB tile + **1 KiB ring**             | **64 KiB tile only** (in-place scale)     |
| sync model                     | `pipe.consumer_wait_prior<N>()` per row  | `async_copy_wait_all()` once              |
| inner-loop overhead            | wait/release tokens every iter           | none — inner loop is pure FFMA            |

**The Mojo win is structural, not magical.** The W22.9 regression came
from two compounding costs at TPB=16:

1. **Ring-buffer smem round-trip per row.** cuda-attn-gdn-async issued one
   `LDGSTS.E.BYPASS.128` per state row, then read it back with `LDS.128` —
   that's an extra smem hop the sync baseline didn't pay (W1c reads the
   row directly from gmem into the FFMA chain).
2. **Per-thread pipeline coordination.** With TPB=16 (only ½ warp), the
   `consumer_wait_prior<N>` token-bookkeeping and `pipe.producer_commit()`
   per K-step inflated the inner-loop instruction count, eating the
   latency-hiding benefit.

Mojo's `copy_dram_to_sram_async` is a CTA-collective lowering: one giant
async DMA of the whole (D_K, BLOCK_V) tile (4096 vectorized 16-B loads
distributed across 16 threads = 256 LDGSTS/thread), one `async_copy_wait_all`
drain, then the K-loop is **pure FFMA over smem** — no token traffic,
no ring index arithmetic. The scaled-S in-place write avoids the
double-buffer (sync baseline used a separate `sm_S` cache; we eliminate
it because each thread owns its 4 cols in `sm_S`).

This appears to be the same architectural pattern cuTile uses to hit
610 GB/s, just expressed at a higher-level API. The `cp.async` SASS class
(`LDGSTS.E.BYPASS.128`) is presumably identical to W22.9's; the gap is
**how the load is scheduled**, not the load instruction itself.

## Numerics

Bit-clean: max_abs_err on `o` is 3.8e-6 (10× tighter than f16 round-trip
quantization at this magnitude), max_abs_err on `S_out` is 3.7e-9 (pure f32
roundoff). The async-load reordering doesn't perturb FFMA accumulation
order — pass-1 still reduces over D_K in-order, just with all the source
S values prefetched up front rather than interleaved.

## Implementation notes

- **TPB=16, BLOCK_V=64** — same as sync baseline. The shape was deliberately
  preserved so the only changed variable is the load pattern.
- **In-place scale.** Sync baseline kept `sm_S_raw` (post-LDG) and `sm_S`
  (post-α) as two separate smem tiles; that hits the 96 KiB Blackwell smem
  cap when both are 64 KiB. We collapse to a single tile by overwriting
  with the α-scaled value during pass-1 (race-free because each thread
  owns 4 disjoint cols).
- **Tile construction.** The gmem tile uses `LayoutTensor[D_K, D_V](s_in + bh_offset).tile[D_K, BLOCK_V](0, bv)` — preserves stride D_V between rows. `vectorize[1, 4]` packs 4 contig f32 cols into a 16-B vec for the cp.async lowering. `thread_layout=Layout.row_major(TPB, 1)` distributes 16 row-threads over 256 rows (16 rows/thread).

## Pitfalls / gotchas

1. **Smem cap.** First attempt kept both raw and scaled S tiles (128 KiB)
   and ptxas rejected with `0x20808 bytes > 0x18c00 max`. In-place scale
   collapses footprint to 64 KiB. Future BLOCK_V=128 (or D_K>256) attempts
   would need dynamic smem promotion — not exposed in Mojo 1.0.0b1.
2. **`UnsafePointer.offset()` doesn't exist** in Mojo 1.0.0b1. Use
   pointer arithmetic `s_in + bh_offset` instead.
3. **`Layout` constructor with explicit stride** isn't the cleanest API to
   slice a non-contig sub-tile; the idiomatic path is the
   `LayoutTensor.tile[ROWS, COLS](row_idx, col_idx)` method (matches
   matmul-bf16's pattern).
4. **Median CV is high.** Best 610 / median 580–586 — about 4–5%
   coefficient of variation across the 50-iter window. Consistent with
   the cuda-exploration AGENTS.md note about WSL+desktop-contention; not
   a methodology issue. Use **best** as the headline number, median as
   the conservative number.
5. **Don't attempt warp-specialized split with this primitive.**
   `copy_dram_to_sram_async` is intentionally a collective, not a
   producer/consumer building block. If we wanted W22.10-class numbers
   (>1 TB/s with TMA), we'd need either a Mojo TMA primitive (not
   available) or a hand-rolled `cp.async.bulk.tensor` PTX-asm path.

## Hypothesis status

**Lift hypothesis CONFIRMED.** Mojo's async-pipe primitive avoids the
W22.9 cuda::pipeline regression class entirely because:
- It's CTA-collective, not per-thread (no token bookkeeping in inner loop).
- It's bulk-then-drain, not ring-buffered (no per-iter wait/release).
- It composes with in-place compute (no double-buffer smem cost).

The empirical result (610.6 GB/s = 1.96× the cuda-async best, 1.91× the
sync baseline) matches the qualitative prediction in the task brief
("if Mojo's primitive is just per-thread async-load-then-await (no
producer/consumer split), it might NOT regress"). The actual primitive
isn't even per-thread — it's CTA-collective — and the result is far
better than "not regress": it's the **largest single-frontend GDN lift in
the cuda-exploration repo so far** outside the TMA path.

## Files

- [`attn_gdn_async.mojo`](attn_gdn_async.mojo) — kernel + main + bench harness
- [`run.sh`](run.sh) — reproduce script
- [`run.log`](run.log) — last-known output
- [`results.csv`](results.csv) — single-row best/median GB/s

## Next steps (out of scope this cell)

- **Multi-stage software pipeline.** This cell does one bulk load + drain.
  A 2-stage pipeline (load tile-N+1 while computing tile-N) might lift
  further if D_K could be tiled — but D_K=256 fits in one slab already.
  More relevant for batched / multi-timestep variants.
- **Mojo TMA primitive.** If/when Modular exposes `cp.async.bulk.tensor`
  or equivalent in Mojo, reattempt at the cuda-attn-gdn-tma 1032 GB/s level.
- **Warp-specialized variant.** Hand-roll the producer/consumer split with
  `barrier_arrive` / `barrier_wait` directly to test if Mojo allows the
  cuTile-style 100 KiB pipeline pattern at higher granularity.
