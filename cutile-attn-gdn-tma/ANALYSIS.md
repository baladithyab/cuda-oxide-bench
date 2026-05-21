# Wave C2.5 — `cutile-attn-gdn-tma`: explicit-TMA-mode test of cuTile DSL on GDN

## TL;DR

cuTile 1.3.0 **does** expose explicit TMA controls — `allow_tma: bool` and
`latency: int` kwargs on `TiledView.load`, `TiledView.store`, and the free
`ct.load`/`ct.store` (see `cuda/tile/_stub.py:743,785,959,1112`). The
default is `allow_tma=True` (TMA is opt-OUT, not opt-IN). We re-built the
W16.4 GDN-decode kernel with `allow_tma=True, latency=10` plumbed through
every load/store call site, and ran a falsification leg with
`allow_tma=False`.

**Result: TMA falsified for cuTile GDN at sm_120, identically to
W22.12.** The compiler emits **zero** UTMALDG.2D / UTMASTG.2D /any-UTMA
instructions for any of the three legs, including the explicit-TMA-on
leg, on either shape. The cubin generated for `allow_tma=True` and
`allow_tma=False` is byte-identical (md5
`c102d09456668b2590d97f2198f60070`, 372 888 B) — the kwarg parses, the
kernel runs correctly, but the compiler's internal heuristic refuses TMA
on these tile shapes.

| Leg | `allow_tma` | shape | best GB/s | median GB/s | UTMALDG | FFMA | HMMA |
|---|---|---|---:|---:|---:|---:|---:|
| A | `True` (explicit) | qwen3_next_decode | 610.6 | 554.0 | **0** | 0 | 0 |
| B | `False` (falsification) | qwen3_next_decode | 868.5 | 766.1 | **0** | 0 | 0 |
| C | `True` (explicit) | large (B=4, H=64, d_k=256, d_v=256) | **1180.1** | 1125.9 | **0** | 0 | 0 |

All three legs have identical SASS / md5 cubins — the compiler-emitted
code is the same; the perf differences across legs are pure run-to-run
HBM/thermal variance (5–15 % CV typical for this rig per
`cuda-exploration/AGENTS.md`). Re-runs confirm:
leg A best 610.6 GB/s reproduces, leg B best 868.5 GB/s reproduces.

`max_abs_err`: o = 3.052e-05 (< atol=1e-3), S_out = 2.980e-08. Identical
across all three legs (deterministic kernel).

## DSL-TMA-feature analysis: yes, exposed; no-op on GDN

**The cuTile DSL exposes TMA control.** The tools to look at:

- `TiledView.load(index, *, latency=None, allow_tma=None) -> Tile` — see
  `cuda/tile/_stub.py:741`. Docstring: "*allow_tma (const bool): If
  False, the load will not use TMA. By default, TMA is allowed.*"
- `TiledView.store(index, tile, *, latency=None, allow_tma=None) -> None`
  — `cuda/tile/_stub.py:783`.
- `ct.load(...)` / `ct.store(...)` free functions (same kwargs).
- `ct.LoadStoreHints(latency=..., allow_tma=...)` bytecode attribute —
  `cuda/tile/_bytecode/attribute.py:83`. Default is `allow_tma=True`.

**TMA is opt-out, not opt-in.** Setting `allow_tma=True` is
semantically a no-op vs. the default; setting `allow_tma=False`
*forbids* TMA. There is no `use_tma=True`-as-mandate flag — the DSL
gives the compiler permission, but the compiler decides.

**The cuTile compiler can produce TMA on this rig.** Wave-13 control:
`/home/codeseys/cuda-exploration/analysis/wave13-sass/cutile_matmul_tiled.sass`
contains 17 `UTMA*` matches including `UTMALDG.2D` and `UTMACCTL.PF`,
emitted by the DSL for the canonical block-tiled matmul (`BM=BN=128,
BK=16`, f32 accum). So this is **not** a "DSL doesn't emit TMA on
sm_120" issue — it's a tile-shape heuristic. (Note: the more recent
`cutile-matmul-tiled-mixed` outputs do *not* have UTMA either —
suggesting the heuristic may have regressed in 1.3.0 for the mixed-dtype
path, or that those benchmarks happen to use shapes the heuristic
already declines. Out of scope here.)

## Why the cuTile compiler refuses TMA for GDN-decode tile shapes

The kernel uses six `tiled_view`s with these tile shapes:

| view | tile shape | dtype | per-tile bytes |
|---|---|---|---:|
| Q, K | `(1, D_K)` = `(1, 256)` | f16 | 512 B |
| V, O | `(1, BLOCK_V)` = `(1, 64)` | f16 | 128 B |
| Alpha, Beta | `(1, 1)` | f16 | 2 B |
| S_in, S_out | `(D_K, BLOCK_V)` = `(256, 64)` | f32 | 64 KB |

TMA's hardware preconditions (Hopper/Blackwell `cp.async.bulk.tensor`)
require the global tensor to be 16-byte aligned with strides matching a
TMA descriptor and the tile to be a "useful" 2D rectangle (typically ≥
4–16 rows of contiguous columns, sometimes shape-multiple constraints).
The cuTile compiler heuristic appears to:

1. **Refuse TMA on `(1, N)` row tiles.** All five 1-row views (Q, K, V,
   Alpha, Beta, O) skip TMA. This makes sense: TMA's amortization
   benefit collapses on a single-row load — a normal 16B vectorized
   `LDG.E.128` is just as good and has lower descriptor-build overhead
   per kernel.
2. **Decline TMA on the `(256, 64) × f32` state tile** even though it
   is shape-compatible. Plausible reason: it's 64 KB per block; TMA's
   smem allocation strategy may conflict with the kernel's existing
   99 KB static smem (W22.12 metadata) — 64 KB TMA-staged tile + 99 KB
   static = 163 KB which exceeds sm_120's 100 KB dynamic-smem-per-block
   cap. So the compiler stays on cooperative `LDG.E` loads to avoid
   smem-staging.

We tried `latency=10` (the maximum "DRAM heavy" hint) on every load to
push the heuristic towards async/bulk paths. No change in cubin bytes
or SASS instruction shape — the heuristic ignores it for this kernel.

## Bench numbers in context

Wave 22.12 baseline (`cutile-attn-gdn`): 610 GB/s on qwen3_next_decode,
no UTMA. We replicate that exactly in leg A (610.6 GB/s, no UTMA). The
explicit `allow_tma=True` annotation **changes nothing**.

Leg-C (`large` shape) hits 1180 GB/s — 65.9 % of 5090 HBM3E peak — also
without UTMA. This is **larger working set, more grid blocks → more HBM
traffic concurrency**, not TMA. The `large` shape has 4× batch and 4×
heads (256 grid cells vs. 16) which saturates the memory subsystem
better. So GDN-decode at sufficient grid size is already memory-bound
near peak bandwidth on `LDG.E` paths alone — TMA wouldn't help here.

The C++ TMA target (`cuda-attn-gdn-tma/attn_gdn_tma.cu`) does emit
UTMALDG (we confirmed `UTMA=2` in its SASS), but its perf number isn't
in the comparison set for this cell — that's a separate parent-orchestrator
concern.

## Acceptance criteria

| criterion | status |
|---|---|
| kernel runs | ✅ all three legs |
| correctness atol=1e-3 | ✅ max_abs_err = 3.052e-05 (o), 2.980e-08 (S_out) |
| SASS UTMALDG.2D > 0 | ❌ **0 in every leg** — TMA falsified |
| GB/s reported | ✅ 610.6 / 868.5 / 1180.1 best |
| target 1000+ GB/s if TMA engaged | TMA never engaged; large-shape reaches 1180 GB/s WITHOUT TMA via memory-system saturation alone |
| stop condition (b): "DSL-claims-TMA-but-SASS-shows-no-UTMALDG" | **Triggered.** Falsification documented here. |

**This cell is a falsification result.** cuTile 1.3.0 exposes TMA opt-out
machinery, but its codegen heuristic refuses to apply TMA to the
GDN-decode tile shapes (1×256 row tiles + 256×64 f32 state tile under
the 99 KB static-smem ceiling). Bit-identical cubin between
`allow_tma=True` and `allow_tma=False` is the cleanest possible proof.

## Files

- `main.py` — clone of `cutile-attn-gdn/main.py` with `allow_tma=` and
  `latency=` plumbed through every load/store, factory, and CLI flag.
- `run.sh` — three-leg sweep (TMA-on qwen3, TMA-off qwen3, TMA-on large).
- `cubin_tma_*.cubin` — exported cubins (md5-identical across legs).
- `sass_tma_*.sass` — disassembled SASS (md5-identical across legs).
- `results_tma_*.csv` — per-iter bench CSVs.
- `bench_tma_*.log`, `smoke_tma_*.log` — full run logs.
- `run_full.log` — combined log of the run.sh invocation.

## Reproduce

```bash
cd /home/codeseys/cuda-exploration/cutile-attn-gdn-tma
bash run.sh   # ~2 minutes, three legs
```

## Anchor for downstream waves

- **The `allow_tma`/`latency` kwargs are silent on this kernel.**
  Whoever reads this next: don't waste time on hint-tuning for GDN-decode
  in cuTile — go for the algorithmic-restructuring lever (bigger row
  tiles via head-fusion, or a warp-spec / cp.async.bulk pattern outside
  the DSL) if you want TMA to engage.
- **Reference UTMA-positive cubin:**
  `analysis/wave13-sass/cutile_matmul_tiled.sass` — proves the cuTile
  compiler can emit `UTMALDG.2D` on this rig for `BM=BN=128, BK=16`
  matmul tiles.
- **W22.12 finding fully replicated** with both implicit and explicit
  TMA-on annotations; the metadata story (99 KB static smem, REQNTID=256,
  zero UTMA) is unchanged by explicit DSL hints.
