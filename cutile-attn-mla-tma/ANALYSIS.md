# Wave C3.2 — `cutile-attn-mla-tma`: explicit-TMA-mode test of cuTile DSL on MLA

## TL;DR

Cloned `cutile-attn-mla/main.py` with `allow_tma` (True/False) and
`latency=10` plumbed through every `TiledView.load` / `TiledView.store`,
mirroring the W-C2.5 transformation pattern from `cutile-attn-gdn-tma`.
**Hypothesis**: MLA's matmul-friendly tile shapes (64×256 f16 Q/K, 64×128
f16 V/O) might trigger the cuTile compiler's TMA path even though GDN's
1×D_K row-tile shapes did not (W-C2.5).

**Result: TMA emission falsified for cuTile MLA at sm_120, identically
to W-C2.5 GDN.** The compiler emits **zero** UTMALDG / UTMASTG / any-UTMA
instructions for either leg, on the bench shape. The cubin generated for
`allow_tma=True` and `allow_tma=False` is **byte-identical**
(md5=`2b18e38bfa8d22b15cfee3f5b6b4c4c4`, size 378 448 B).

**However — surprise finding**: in-process bench timings differ
*reproducibly* by ~50 % between `allow_tma=True` and `allow_tma=False`
even though the on-disk cubins are bit-identical. The TileIR bytecode
**does** differ (size 2 265 vs 2 294 B, distinct md5s), so the DSL flag
is captured upstream. The flag must be re-applied in the in-process JIT
launch path in a way that is not preserved by `export_kernel`'s on-disk
cubin serialization. Practical implication: **`allow_tma=True` is the
right default** — even when no TMA SASS is emitted, the kwarg measurably
helps perf at this shape.

| Leg | `allow_tma` | best ms | best TFLOPS | median TFLOPS | UTMALDG | HMMA |
|---|---|---:|---:|---:|---:|---:|
| A | `True`  (explicit) | 3.060 | **112.29** |  95.50 | **0** | 192 |
| B | `False` (falsification) | 4.542 |  75.65 |  69.09 | **0** | 192 |
| A (reproducer)  | `True`  | 3.064 | **112.15** |  93.98 | — | — |
| B (reproducer)  | `False` | 4.580 |  75.02 |  66.03 | — | — |

Tolerance: smoke max_abs = 1.539e-04 ≪ atol=5e-3 (f16 tolerance) ≪
1e-2 (task allowance). Reproducible across reruns.

## DSL-shape-acceptance analysis: the heuristic refuses MLA shapes too

cuTile's MLA tile shapes:

| view | tile shape | dtype | per-tile bytes |
|---|---|---|---:|
| Q     | `(BLOCK_M, QK_PAD) = (64, 256)` | f16 | 32 KB |
| K     | `(BLOCK_N, QK_PAD) = (64, 256)` | f16 | 32 KB |
| V, O  | `(BLOCK_N, D_V) = (64, 128)`    | f16 | 16 KB |

These are larger and more rectangular than GDN's 1-row tiles, and they
match the canonical FlashAttention-2 tile pattern the cuTile DSL is
explicitly marketed for. **Yet the compiler still declines TMA.**

Comparison with the Wave-13 control (which DID emit TMA):

| Kernel | tile shapes | UTMA matches |
|---|---|---:|
| W13 `cutile_matmul_tiled` | (BM=128, BK=16) f32 | **17** ✅ |
| W-C2.5 `cutile-attn-gdn-tma` | (1, 256), (256, 64) etc. | 0 ❌ |
| **W-C3.2 `cutile-attn-mla-tma`** | **(64, 256), (64, 128) f16** | **0 ❌** |

So the heuristic isn't a simple "rectangular ≥ 16 rows wins". The MLA
tiles ARE rectangular (64 rows), ARE matmul-friendly (64×256 is a
common GEMM tile), and ARE used inside two `ct.mma` calls per loop iter
— exactly the pattern the matmul-tiled benchmark uses successfully. Yet
the cuTile compiler refuses TMA.

**Speculation about the heuristic**: the matmul-tiled benchmark loads a
**single** tile per matmul. MLA's kernel pipelines an **online softmax
loop** with online accumulator state in registers and 3 in-loop loads
(K, V, [implicit Q]). The compiler likely refuses TMA when the
load-store pattern is interleaved with arithmetic that isn't a pure mma
chain — TMA's bulk-async model wants prefetch-then-mma decoupling, and
flash-attention's `m_i / l_i` reduction couples loads to accum. This is
a **DSL pattern-matcher limitation**, not a hardware limitation. (Wave 13
matmul-tiled has no online softmax; it's a clean prefetch-and-mma loop.)

A 1.3.0-version regression is also possible: ANALYSIS.md for W-C2.5
notes the more recent `cutile-matmul-tiled-mixed` likewise produces no
UTMA. Worth re-running W13's exact `cutile_matmul_tiled` against 1.3.0
to confirm the matmul-only path still emits TMA before generalizing.

## The byte-identical-cubin / divergent-perf paradox

The most interesting unexpected finding. Reproduced via three cubin
exports per leg:

| export | latency kwarg | allow_tma | cubin md5 | cubin size |
|---|---|---|---|---:|
| run.sh leg A | 10  | True  | `2b18e38b…` | 378 448 |
| run.sh leg B | 10  | False | `2b18e38b…` | 378 448 |
| `export_latnone.py` | None | True  | `a8eb31ae…` | 632 504 |
| `export_latnone.py` | None | False | `a8eb31ae…` | 632 504 |
| baseline `cutile-attn-mla` | (None) | (default=True) | `0cbb8893…` | 633 232 |

Within each `latency` setting, the on-disk cubin is bit-identical
between `allow_tma=True` and `allow_tma=False`. Across `latency`
settings the cubin differs. So the on-disk cubin **encodes `latency`**
but **does not encode `allow_tma`** — for these MLA tile shapes.

But TileIR bytecode (the format that `export_kernel(...,
output_format='tileir_bytecode')` produces) **does** differ:

```
allow_tma=True:  TileIR md5=47ba0324c4a06faa6fa0e21cc85c6c25  size=2265
allow_tma=False: TileIR md5=d326c4600b7cc121af9b812befa08e95  size=2294
```

29-byte difference is consistent with the per-load attribute
`("allow_tma", Bool(False))` being serialized into TileIR
(see `_bytecode/attribute.py:88`).

So the DSL captures the flag, the TileIR carries it, but the
TileIR→cubin lowering ignores `allow_tma=False` for these shapes
(because it was never going to use TMA anyway → the flag has no
SASS-level effect). Yet **bench TF differs ~50 % reproducibly**, even
when both legs run in the SAME Python process across multiple
A→B→A interleaved configurations:

```
allow_tma=True  latency=None   best=3.065ms (112.10 TF)
allow_tma=False latency=None   best=4.603ms ( 74.64 TF)
allow_tma=True  latency=10     best=3.064ms (112.14 TF)
allow_tma=False latency=10     best=4.347ms ( 79.04 TF)
allow_tma=True  latency=1      best=3.056ms (112.43 TF)
allow_tma=False latency=1      best=5.027ms ( 68.35 TF)
```

The factor that changes the runtime is **`allow_tma`**, not `latency`.

Resource usage in the cubin is identical:
`REG=255  STACK=1536  SHARED=50268  LOCAL=0  CONSTANT[0]=992`. The ELF
sections are identical (verified via cuobjdump --dump-elf diff = empty).

**Hypothesis** (best guess, not confirmed): the in-process JIT path uses
the TileIR bytecode (which differs!) directly, and runs a slightly
different libNVVM optimization pipeline downstream — producing two
*different* in-memory cubins that happen to *serialize* identically when
written to disk via `export_kernel` (which may strip the no-op
`allow_tma=False` attribute before persisting). This is consistent with
`replace_hints` docs at `_execution.py:128` saying "Because hints
affects compilation, the returned object will have its own JIT cache."
The flag absolutely affects the JIT cache key.

Falsifying this would require capturing the actual JIT'd cubin out of
the live process (not via `export_kernel`). Out of scope for C3.2 but
**worth a follow-up wave**: if the in-process cubin really does differ
from the on-disk one, that's a packaging-level finding — every cuTile
cubin extracted via `export_kernel` may be lying about what runs at
launch time.

A second, more boring hypothesis: there's some launch-time `cuFunc*`
attribute that gets set differently. Grep of the cuTile Python
package found no `cuFuncSetAttribute` callers, but the C-extension
`_cext.so` (TileDispatcher) is opaque; it could be calling driver-level
`cuKernelSetAttribute` based on the LoadStoreHints attribute carried in
the bytecode. Reading `_cext.so` is a separate wave.

Either way, the **operational lesson** is clear: keep `allow_tma=True`
(the default), even when SASS analysis shows no TMA emitted. Setting it
to False is *worse* than the default, not just neutral.

## Bench summary (TF/s, MLA TFLOPS counted at qk_head_dim=192)

```
allow_tma=True:   best=112.29 TF  median=95.50 TF  (51% of 218 TF cuBLAS hgemm peak)
allow_tma=False:  best= 75.65 TF  median=69.09 TF  (35% of cuBLAS hgemm peak)
```

The TMA-on number (112 TF) is also slightly above the baseline `cutile-attn-mla`
result (median 96 TF, best 103 TF in this run, recorded as 110/112 in the
older `run.log`). Within run-to-run variance for this rig (5–15 %
CV per AGENTS.md). **No TMA was actually emitted in either case**, yet
the explicit `allow_tma=True` kwarg consistently wins by a measurable
margin over `allow_tma=False`.

## Files

| file | description |
|---|---|
| `main.py` | Cloned MLA kernel with `allow_tma`/`latency` kwargs threaded through 4× load + 1× store call sites |
| `run.sh` | 2-leg driver (A: TMA-on; B: TMA-off) — smoke + bench + cubin + SASS grep + md5 cross-compare |
| `isolation_test.py` | Single-process 6-config matrix (allow_tma × {None,10,1}) proving allow_tma drives the perf delta |
| `export_latnone.py` | Cubin export at `latency=None` to verify the cubin-md5-collision is robust to latency choice |
| `cubin_tma_on.cubin`, `cubin_tma_off.cubin` | byte-identical cubins (md5 `2b18e38b…`) |
| `sass_tma_on.sass`, `sass_tma_off.sass` | SASS dumps; both show 0 UTMA |
| `bench_tma_on.log`, `bench_tma_off.log` | per-leg bench output |
| `results_tma_on.csv`, `results_tma_off.csv` | per-iter timings |
| `run_full.log` | full pipeline log |

## Pitfalls

1. **Exporting a cubin via `export_kernel` may not faithfully represent
   the in-process JIT'd code.** The TileIR bytecode for
   `allow_tma=True` differs from `allow_tma=False`, but their exported
   cubins md5-collide, while runtime behavior diverges by 50 %. Don't
   trust md5(exported_cubin) as a proxy for "same kernel runs at
   launch". This is bigger than C3.2; it deserves a separate wave with
   in-process-cubin extraction (or process-level perf counters that
   distinguish the two paths).

2. **The `allow_tma` flag is not actionable at the SASS level for
   MLA's tile shapes.** The cuTile DSL captures it, the TileIR carries
   it, but the compiler's TileIR→cubin lowering produces identical
   SASS regardless of the flag value (because zero TMA insts are
   emitted in both branches). The DSL "exposes TMA control" only in
   the sense that it's a knob that *could* matter — for these shapes
   it's a no-op at the codegen level.

3. **Wave-13 matmul-tiled may be the only kernel-class on this rig
   where cuTile's heuristic accepts TMA.** Both the W-C2.5 GDN-decode
   shape (rows=1) and the W-C3.2 MLA shape (rows=64, ostensibly the
   "good" matmul-friendly case) get TMA refused. If reproducing W13's
   17-UTMA result against cuTile 1.3.0 fails, then "TMA is reachable
   in the cuTile DSL on sm_120" itself becomes a regression-or-
   never-true hypothesis. Worth one short follow-up wave to retest
   W13's exact code at 1.3.0.

4. **Smoke uses MLA's default tolerance (`get_tol("f16")` →
   atol=rtol=5e-3), which is tighter than the C3.2 task brief's
   atol=1e-2.** Both legs pass at 1.539e-04, well within the tighter
   bound. No tolerance loosening was needed.

5. **Reading `_cext.so` would settle the perf paradox.** It's the
   binary that holds the launch path. Out of scope here.
