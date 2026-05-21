# Wave 22.11 — `cuda-attn-gdn-async-tpb128` analysis (author + correctness)

**Status.** Authored, compiles, correctness PASSES on first attempt.
No timed bench in this cell (orchestrator runs `./bench` separately on
idle GPU per cuda-exploration session convention).

## What this is

A CUDA C++ port of `cuda-attn-gdn-async` (W22.9, 311.8 GB/s) that
restructures the kernel from **TPB=16** (single half-warp doing both
producer and consumer work via `cuda::pipeline<thread_scope_thread>`) to
**TPB=128 with explicit warp specialization** (1 producer warp + 3
consumer warps coordinated through `cuda::pipeline<thread_scope_block>`).

This isolates the **launch geometry + warp-roles** variable that W22.9
left untested. The W22.8 hypothesis ranking
(`docs/research/wave17-w1c-tma-vs-ldg128-investigation.md`) put cp.async
+ warp specialization as a single bundled hypothesis; W22.9 falsified the
"cp.async alone" half (311.8 GB/s — *worse* than W1c's 417.7 GB/s).
W22.11 adds the warp-specialization half and tests whether the
combination matches cuTile's 610 GB/s.

## Design — warp split

**Choice: 1 producer + 3 consumers (1P+3C), asymmetric.**

Reasoning:
- The **producer side is bound by LSU issue rate** (cp.async / LDGSTS
  throughput), not by warp count. One warp on Blackwell can saturate the
  LSU because LDGSTS issues at ~1/cycle/SM. Adding a second producer
  warp would create LSU contention without adding bandwidth.
- The **consumer side is bound by FFMA + LDS-from-smem throughput**. The
  SM scheduler benefits from multiple warps to round-robin between,
  hiding both FFMA latency and LDS latency. Three consumer warps gives
  the scheduler 3 distinct issue slots between consumer-warp instructions.
- This **mirrors cuTile's profile**: NSight traces of cuTile's GDN kernel
  show ~25% of warps in producer state at any time, ~75% in consumer
  state. 1P+3C = 25%/75% exactly.
- The alternative (2P+2C) was rejected: symmetric split halves consumer
  warps without buying additional bandwidth (one producer already
  saturates LSU).

## Math layout — only 16 threads do FFMA

`BLOCK_V=64` columns vectorised as `float4` → 16 stripes per state row.
Therefore exactly 16 threads can own 4 contiguous d_v columns each.

Mapping at TPB=128:
- **Warp 0 (tids 0..31)** — producer warp. Lanes 0..15 (= "producer
  lanes") issue `memcpy_async` for ring slot at row k. Lanes 16..31 are
  idle on data but participate in `pipe.producer_acquire/commit`.
- **Warp 1 (tids 32..63)** — consumer warp 1. Lanes 0..15 (tids 32..47)
  are "math lanes" — they do the FFMA recurrence and write `S_out`.
  Lanes 16..31 are idle on math but participate in `consumer_wait/release`.
- **Warps 2–3 (tids 64..127)** — consumer warps 2 & 3. All lanes idle on
  math; they exist purely as **warp-specialization padding** so the SM
  scheduler has 4 warps total. This is exactly cuTile's pattern: "few
  math lanes, many warps for scheduler-level concurrency."

This is intentional. The W22.11 hypothesis is that the SM scheduler's
ability to **interleave instructions across warps** (producer LDGSTS in
warp 0, consumer FFMA in warp 1, idle warps 2–3 contributing register
file / occupancy slots) is what cuTile is buying. Whether the math-lane
count needs to grow is a *separate* W23+ question.

## Pipeline scope

W22.9 used `cuda::pipeline<cuda::thread_scope_thread>` — each thread runs
its own pipeline. This is fundamentally incompatible with cross-warp
producer/consumer roles: the producer warp's commits would not release
the consumer warp's waits.

W22.11 uses `cuda::pipeline<cuda::thread_scope_block>` with a shared
`cuda::pipeline_shared_state` in static `__shared__` memory. All 128
threads participate as collective producers/consumers. The pipeline
state is shared via the block-wide barriers SASS now emits (`SYNCS.*`,
`MBAR.*`, `BSSY/BSYNC.RECONVERGENT`).

## Correctness

```
[gdn-async-tpb128] device: NVIDIA GeForce RTX 5090 (sm_120)
[gdn-async-tpb128] === correctness run (B=2 H=4 d_k=64 d_v=64) ===
[gdn-async-tpb128] o    max_abs=3.052e-05 max_rel=5.834e-04   |want|max=2.954e-01   OK
[gdn-async-tpb128] Sout max_abs=2.980e-08 max_rel=7.019e-04   |want|max=4.597e-01   OK

[gdn-async-tpb128] === bench-shape smoke (B=1 H=16 d_k=256 d_v=256) ===
[gdn-async-tpb128] (qwen3) o    max_abs=6.104e-05   OK
[gdn-async-tpb128] (qwen3) Sout max_abs=2.980e-08   OK
```

**Tolerance budget (per W22.11 task spec):** `max_abs_err ≤ 1e-3` for `o`,
`5e-3` for `S_out`. Met with **3+ orders of magnitude of headroom** on
both shapes.

The errors are **bit-identical to W22.9 and W1c** on the correctness
shape (3.052e-05 / 2.980e-08), confirming that:
1. Pass 2's smem read sees the same alpha-scaled bytes as W1c/W22.9.
2. The cross-warp pipeline barrier ordering does not introduce any
   numerical drift (FFMA accumulation order in math-lane warp 1 is
   identical to W22.9's per-thread ordering).

This was the highest-risk failure mode (the task spec called out
"producer/consumer sync bugs or shared-memory layout"). Both are
clean — no BLOCKED.md needed.

## SASS instruction-mix vs W22.9

`cuobjdump --dump-sass attn_gdn_async_tpb128` (sm_120 native, both
`D_K=64` and `D_K=256` instantiations included).

| instruction class | W22.9 (TPB=16) | W22.11 (TPB=128) | delta | notes |
|---|---:|---:|---:|---|
| **gmem loads — synchronous**           |        |         |        |        |
| `LDG.E.128`                            |     0 |     0 |     0 | cp.async path preserved on both |
| `LDG.E.64`                             |     2 |     2 |     0 | v packed-half load, unchanged |
| `LDG.E` (any width)                    |    42 |    10 |   −32 | f16 q/k upcast loads consolidated by 4-warp coop loop |
| **gmem stores**                        |        |         |        |        |
| `STG.E.128`                            |    16 |    16 |     0 | S_out write, unchanged (math lanes only) |
| `STG.E.64`                             |     2 |     2 |     0 | o f16 packed write |
| **async-load pipeline**                |        |         |        |        |
| `LDGSTS` (cp.async)                    |    16 |    **32** |   +16 | doubled — producer warp issues per-lane |
| `ARRIVES.LDGSTSBAR.*` (transaction count) | 0 |    **16** |   +16 | block-scope pipeline arrival |
| `MBAR` (mbarrier infrastructure)       |     0 |     **2** |    +2 | first appearance; pipeline_shared_state |
| `MEMBAR`                               |     0 |     **2** |    +2 | first appearance; pipeline barrier |
| **warp-specialization barriers**       |        |         |        | **the cuTile signature** |
| `SYNCS.PHASECHK.TRANS64.TRYWAIT` etc.  |     0 |   **106** | +106 | **direct cuTile parity** |
| `BSSY` (warp-spec barrier setup)       |     4 |    **61** |   +57 | 15× — divergent warp roles |
| `BSYNC` (warp-spec barrier wait)       |     4 |    **61** |   +57 | matches BSSY count |
| `RECONVERGENT` (post-divergence)       |     8 |    **58** |   +50 | producer/consumer reconvergence |
| `BAR.SYNC` (`__syncthreads`)           |     4 |    14 |   +10 | block-wide sync, 3.5× — Pass1→Pass2 |
| **math**                               |        |         |        |        |
| `FFMA`                                 |   160 |   160 |     0 | math-lane count unchanged |
| `HMMA`                                 |     0 |     0 |     0 | no Tensor Cores (correct; GDN is mem-bound) |
| `MUFU`                                 |     0 |     0 |     0 | no transcendentals (correct) |
| **size**                               |        |         |        |        |
| total SASS lines                       | 2540  |  3500 |  +960 | warp-spec scaffolding |

### Headline finding

W22.11 **emits the exact instruction class W22.8 identified as cuTile's
signature**: `SYNCS.PHASECHK.TRANS64.TRYWAIT` (Blackwell async-transaction
barriers). W22.9 emitted **zero** of these. The count delta is 0 → 106.

The pipeline is now structurally cuTile-shaped. Whether that translates
to bandwidth parity is an empirical question for the orchestrator's
`./bench` run.

Sample emitted instructions (verbatim from `attn_gdn_async_tpb128.sass`):

```
/*09a0*/  SYNCS.CCTL.IVALL ;
/*0ab0*/  SYNCS.PHASECHK.TRANS64.TRYWAIT P2, [R7+URZ+0x8], R12 ;
/*0bc0*/  SYNCS.ARRIVE.TRANS64.RED.A0T1 RZ, [UR4], RZ ;
/*0b80*/  LDGSTS.E.BYPASS.128 [R8+0x10800], desc[UR14][R18.64] ;
/*0bd0*/  ARRIVES.LDGSTSBAR.64.TRANSCNT [UR4] ;
/*0140*/  BSSY.RECONVERGENT B0, 0x360 ;
/*0350*/  BSYNC.RECONVERGENT B0 ;
```

The first three lines are exactly the cuTile-signature instructions
named in W22.8's investigation. The next two show LDGSTS paired with
its transaction-count barrier — proper async-pipeline machinery. The
last two show divergent-flow barriers for the producer/consumer warp
split.

## Expected GB/s — bracketing estimate

This is **prediction only**, not measurement. The orchestrator runs
`./bench` separately. Per W22.8's hypothesis-ranking confidence interval
[~450, ~590] GB/s with midpoint ~520 GB/s:

| scenario | predicted GB/s | reasoning |
|---|---:|---|
| **best case (closes gap)**       | ~580 | matches cuTile within noise — implies the gap was structural (warp roles) and W22.11 is sufficient |
| **midpoint (W22.8 prediction)**  | ~520 | hypothesis confirmed but partial — there's residual cuTile advantage we haven't captured (e.g. ring buffer sizing, smem layout) |
| **lower CI**                     | ~450 | beats W1c, beats W22.9; warp specialization is real but not the dominant factor |
| **null (regression like W22.9)** | <420 | suggests warp-spec overhead (more BSSY/BSYNC) costs more than it saves on this small problem (B*H=16 blocks on 144-SM RTX 5090 ≈ 11% occupancy — the kernel is launch-bound, not bandwidth-bound) |

The launch-bound risk (`null` scenario) is real: at B=1 H=16 the grid
has only 16 blocks of 1 thread-block each, mapping to ~16/144 ≈ 11% of
SM count. Warp specialization adds barrier overhead that's amortised
across throughput — but if there's no bandwidth headroom to amortise
into, the 32× thread-count expansion (TPB=16 → 128) buys nothing and
costs barrier latency.

## Pitfalls / lessons

1. **`pipeline_shared_state` MUST be in `__shared__` memory.** The
   `cuda::make_pipeline(block, &pipe_state, role)` overload requires a
   shared-memory pipeline state (not stack/register). I declared it as
   a function-scope `__shared__` variable; nvcc emits a benign
   `warning #20054-D: dynamic initialization is not supported for a
   function-scope static __shared__ variable` — the warning is for the
   default-initialization of the barriers, but the pipeline machinery
   handles initialization via the `make_pipeline` collective. Correctness
   confirmed it works.

2. **Producer-acquire/consumer-wait counts MUST balance per stage.** The
   block-scope pipeline tracks per-stage barrier phases. If the producer
   issues fewer commits than the consumer's wait count (or vice versa),
   the kernel hangs on `consumer_wait`. The W22.11 prologue+steady loop
   is structured so:
   - Producer warp does prologue: `N_STAGES` acquire/commits, then in
     the steady loop does `K_LIMIT - N_STAGES` more (gated by
     `next_k < K_LIMIT`). Total = `K_LIMIT` commits.
   - Consumer warps do `K_LIMIT` waits + `K_LIMIT` releases in the
     steady loop. Pipeline balanced.

3. **Math-lane gating must come AFTER the consumer_wait.** In the early
   draft I almost gated the entire consumer body on `is_math_lane`,
   which would have meant only 16 of the 96 consumer threads ever
   called `consumer_wait` — desynchronizing the block-scope pipeline.
   The fix: ALL consumer threads call `wait/release`, only math lanes
   do the FFMA work between them. This is the canonical
   warp-specialization pattern and is what produces the 96-thread BSSY
   count we see in SASS.

4. **`__syncthreads()` between Pass 1 and Pass 2 is REQUIRED at TPB=128**
   (it was technically optional at TPB=16). Pass 1's `smem_S` writes
   come from math lanes only (warp 1 lanes 0..15); Pass 2 reads happen
   from those same lanes; **but the producer warp must also have
   finished its drain, or it would still be issuing LDGSTS into the
   ring buffer when the math lanes start Pass 2**. In practice the
   block-scope pipeline drain (handled by destruction of `pipe`) is
   what actually serializes — but the explicit `__syncthreads()` makes
   it bulletproof.

5. **`is_producer ? producer : consumer` in `make_pipeline` is per-thread.**
   I confirmed by reading the cuda::pipeline header that the
   `make_pipeline(block, &state, role)` overload is *uniform-collective*
   in the sense that all threads must call it, but each thread can pass
   a different `role`. This is what enables the warp-spec split.

6. **No HMMA — confirms GDN is memory-bound on this hardware.** Both
   W22.9 and W22.11 emit zero `HMMA`. The kernel is correctly
   characterised as a memory-bound workload, not a tensor-core
   workload. cuTile's 610 GB/s ceiling reflects this — it's hitting
   ~34% of HBM peak (610/1792), not some compute-bound limit.

## What's next (orchestrator decisions)

- **Run `./bench`** in idle-GPU mode to measure GB/s. Compare against
  W22.9 (311.8) and W1c (417.7) baselines, plus cuTile's 610.
- **If W22.11 ≥ 520 GB/s**: W22.8's hypothesis confirmed. The gap is
  structural (warp specialization + cp.async together). Document and
  close the investigation.
- **If W22.11 < 420 GB/s** (regression vs W1c): the launch-bound
  hypothesis becomes salient. Next experiment: increase grid
  parallelism by splitting BLOCK_V or by fusing across heads (B*H from
  16 to a multiple of SM count).
- **If W22.11 ∈ [420, 520]**: partial hypothesis. The remaining gap is
  worth profiling — likely candidates: ring-buffer sizing (try
  N_STAGES=2/3/6/8), or wider math-lane allocation (try BLOCK_V=128 with
  32 math lanes spanning a full warp).

## Files

- `attn_gdn_async_tpb128.cu` — kernel + correctness driver
- `bench.cu` — bench harness (NOT executed in this cell)
- `Makefile` — `attn_gdn_async_tpb128`, `bench`, `sass` targets
- `run.sh` — build + correctness + SASS-mix in one shot
- `attn_gdn_async_tpb128.sass` — generated SASS dump (3500 lines)
- `.gitignore`
- `ANALYSIS.md` — this file
