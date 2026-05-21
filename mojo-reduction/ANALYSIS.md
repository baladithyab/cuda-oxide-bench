# mojo-reduction

Wave 18 Phase B.2 — Mojo 1 GB f32 sum-reduction on RTX 5090 sm_120.
Mirrors `oxide-reduction`, `cuda-reduction`, `cutile-reduction`:

- sizes: 1M, 16M, 256M elements
- 1 warmup + 10 timed iters per N
- cudaEvent timing via `ctx.execution_time`
- Block-wide reduction via Mojo's `std.gpu.primitives.block.sum` + atomicAdd into out[0]
- block=256 threads, grid-stride loop with grid=4096 fixed

## Results (Mojo 1.0.0b1, fresh idle GPU 2026-05-20)

| N | warmup correctness | avg µs/iter | GB/s | regime |
|---|---|---:|---:|---|
| 1M (1,048,576) | rel_err = 0.0 | 14.9 | 280 | overhead-bound |
| 16M (16,777,216) | rel_err = 0.0 | 18.9 | 3549 | L2-cache-resident |
| **256M** (268,435,456) | rel_err = 0.0 | 714.5 | **1502** | **canonical memory-bound** |

## Cross-frontend comparison @ N=256M (canonical)

Re-benched on the same idle GPU 2026-05-20 thermal-window:

| frontend | GB/s | vs Mojo | reduction strategy |
|---|---:|---:|---|
| cuda-oxide (warp-shuffle) | 1507 | +0.3% | hand-written `warp::shuffle_xor_f32` chain |
| **Mojo** (`block.sum`) | **1502** | — | stdlib block.sum (warp-shuffle internally) |
| **cuTile** (`ct.sum`) | **1691** ⚡ | **+12.6%** | TMA bulk loads (`UTMALDG.1D`) |
| nvcc (Wave 14) | 1605 | +6.9% | hand-written warp-shuffle |

## Critical SASS-level finding

This is the most informative single datapoint of Phase B for understanding
Mojo's compiler character.

The question coming into Phase B was: **does Mojo lower its reduction to
the warp-shuffle path (parity with nvcc/oxide) or to TMA bulk loads
(parity with cuTile, +11% lift)?**

Verified via Mojo's `_dump_sass=True` kwarg on `enqueue_function`:

```
=== Mojo reduction SASS analysis (target sm_120a) ===
  UTMALDG (TMA bulk load):     0
  TMA-related instructions:    0
  LDG.E (regular global load): 3
  SHFL.BFLY (warp shuffle):    45
  REDG.E.ADD.F32 (HW atomic):  3
```

Inner-loop body in SASS (5 instructions, identical pattern to nvcc and
cuda-oxide on this kernel):

```
.L_x_2:
    LDG.E R13, desc[UR6][R6.64] ;   # global load
    IADD.64 R8, R4, R8 ;            # i += stride high
    ISETP.GE.S64.AND P0, PT, R8, UR8, PT ; # i < n ?
    IADD.64 R6, R6, R10 ;           # ptr += stride
    FADD R0, R13, R0 ;              # acc += data[i]
    @!P0 BRA `(.L_x_2) ;
```

**Conclusion: Mojo joins the warp-shuffle path club (nvcc + cuda-oxide),
NOT the TMA club (cuTile).** This explains the perf parity with nvcc/oxide
(±0.3%) and the 12% gap to cuTile.

This is a *finding*, not a *bug*. cuTile's `ct.load(buffer, index, shape)`
explicitly invokes the tile-loading abstraction that lowers to TMA on
Blackwell. Mojo's `std.gpu.primitives.block.sum` is a one-line block-wide
reduction; it consumes regular global-load values from each thread's
arithmetic. The user *could* in principle write a Mojo kernel that uses TMA
explicitly via lower-level primitives (cf. `std.gpu.sync.cp_async_bulk`),
but the high-level `block.sum` doesn't take that path.

## API discoveries

1. **`std.gpu.primitives.block.sum[block_size=BLOCK, broadcast=False](val)`** —
   one-line block-wide reduction. Returns the sum to thread 0 (when
   `broadcast=False`) or all threads (when `broadcast=True`, default).
   Internally implements warp-shuffle + smem two-stage reduction. Comparable
   to writing the chain by hand in cuda-oxide (~30 lines).
2. **`std.atomic.Atomic.fetch_add[ordering=Ordering.RELAXED](ptr, value)`** —
   static method, returns the previous value. Lowers to `REDG.E.ADD.F32` on
   sm_120 (Blackwell hardware-accelerated atomic-reduction-in-global).
3. **`ctx.enqueue_memset(buffer, value)`** — fully GPU-side memset, used for
   resetting the output scalar between iters. Critical for clean reduction
   benchmarks; using `enqueue_copy` from a host scalar instead inflates the
   timed window with a PCIe round-trip and tanks the apparent throughput
   (we measured ~1160 GB/s with copy-reset → 1502 GB/s with memset-reset).
4. **`_dump_sass=True`** on `enqueue_function` is undocumented in 1.0.0b1
   but works. Dumps SASS to stdout. Unlike cuTile (where you have to
   monkey-patch `compile_tile`) and unlike cuda-oxide (where you `cuobjdump
   --dump-sass` the cubin), Mojo gives you SASS directly from the bench
   harness. Best dev-loop ergonomics of the four frontends here.
5. **`out` keyword conflict**: parameter named `out` triggers a syntax
   error. Mojo uses `out` for output-argument convention (sibling to
   `mut`/`read`/`ref`). Use `result` or any other name.

## Files

- `reduction.mojo` — single-source kernel + harness
- `reduction.sass` — full SASS dump from `_dump_sass=True` (499 lines)
- `run.log` — captured run output

## Reproducibility

```bash
cd /home/codeseys/cuda-exploration/mojo-workspace
pixi run mojo /home/codeseys/cuda-exploration/mojo-reduction/reduction.mojo
```

## Cross-reference

The Wave 13 cuTile reduction analysis (`analysis/wave13-sass/`) showed cuTile's
`UTMALDG.1D × 7` instruction count vs nvcc/oxide's zero. Mojo joins the latter
group with `UTMALDG = 0`. This adds a third independent "warp-shuffle path"
data point and confirms the cuTile reduction win is genuinely from TMA,
not from any general "DSL/MLIR vs hand-written CUDA C++" lift.
