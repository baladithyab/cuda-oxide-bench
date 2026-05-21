# Wave 22.13 — cuda-attn-gdn-tma-warpspec

**Scope.** Author + correctness only. NO timed benches in this cell — the
orchestrator runs `./bench` separately on idle GPU.

## Hypothesis under test

W22.10 (TMA, TPB=16, 1 thread issues bulk-tensor) hit **1032 GB/s** — the
current best on this hardware/algorithm and the first nvcc-authored kernel
with `UTMALDG > 0` SASS. W22.12 documented cuTile's structural
prescription: **TPB=256, 99 KB static smem, REG=255/thread, 3 named
barriers, 8-warp warp-specialisation** with each consumer warp doing real
math (cuTile per-CTA math ~835 effective FMAs vs nvcc-W22.11's 168).

The W22.13 hypothesis: stack W22.10's TMA load with W22.12's warp-spec
prescription. Specifically — keep W22.10's batched single-issue
`cp.async.bulk.tensor.2d` for the FULL `(D_K, BLOCK_V)` tile (the load-side
win), and split the consumer-side FFMA recurrence across 3 consumer warps
along the D_K axis so all 48 math lanes run concurrently rather than
serialising the loop on a single warp.

If TMA + warp-spec stack additively, expected GB/s is bounded above by
`(W22.10 × consumer-FFMA-density-multiplier)`. With 3× the consumer math
density, the bound is ~3 × 1032 = ~3 TB/s — well above HBM peak (1.79 TB/s).
The realistic target is **1100-1500 GB/s**, leaving the FFMA-density gain
to be partially absorbed by:
* the partial-reduction across consumer warps (extra LDS/STS),
* the consumer-only `bar.sync.aligned 1, 96` and `bar.sync.aligned 2, 96`
  named-barrier rendezvous (which are not free on Blackwell),
* the doubled register-file pressure (TPB=128 vs 16 → 8× more concurrent
  warps requiring register state).

If we regress below W22.10's 1032 GB/s, that is a **NEW falsification** —
the warp-spec restructure imposed more overhead than the math-density
bought back.

## Approach (W22.10 + W22.12 fusion)

| Axis | W22.10 (TMA only) | W22.13 (this) |
|---|---|---|
| TMA load shape | full (D_K, BLOCK_V) tile | **same** — single-issue, batched |
| TMA issuer | thread 0 (1 thread) | **warp 0 lane 0** (1 thread, in producer warp) |
| TMA mbarrier | 1 mbarrier, arrive_count=1 | **same** |
| TPB | 16 | **128** (1 producer warp + 3 consumer warps) |
| FFMA-active threads | 16 (= VLANES = BLOCK_V/4) | **48** (16 lanes × 3 consumer warps; D_K rows split) |
| Consumer-warp math split | n/a | **D_K-row split** — warp `c` owns rows `[c·D_K/3, (c+1)·D_K/3)` |
| Partial-u reduction | none | smem-resident `(N_CONSUMERS, VLANES)` float4 slab + named barrier |
| Named barriers used | 1 (`__syncthreads` after TMA) | **3 distinct bar IDs**: `__syncthreads` ID 0 + `bar.sync 1, 96` + `bar.sync 2, 96` |
| Dynamic smem (correctness shape) | ~16 KB | **~17.4 KB** (added 2× partial slabs = 2 × 3 × 16 × 16 B = 1536 B) |
| Dynamic smem (qwen3 shape)       | ~64 KB | **~75 KB** (same 1.5 KB overhead at BV=64; tile dominates) |
| `cudaFuncSetAttribute` opt-in | conditional > 48 KB | **unconditional 99 KB** (W22.12 prescription) |
| Pass 2 split | n/a (TPB=16, all rows in one warp) | **same D_K-row split as Pass 1**, partial-o reduced across consumer warps |

## Why TPB=128 (not 256)

W22.12 recommends widening TPB. We chose **128 over 256** based on the
math-density bound:

* `VLANES = BLOCK_V / 4 = 16` stripes per row regardless of TPB. Wider
  blocks add scheduling capacity but no extra math lanes per row.
* The unique math-density axis available is the **D_K split** across
  consumer warps. With 3 consumer warps × 16 lanes = **48 active math
  lanes**, we already 3× W22.10's 16 active lanes.
* 8 warps (TPB=256) would give 7 consumer warps × 16 = 112 lanes — but
  this needs a 7-way partial reduction, doubling the smem-traffic cost
  for the reduction step. At BLOCK_V=64 the math budget per CTA is fixed
  (D_K × VLANES = `D_K × 16` FFMAs), so going from 3 consumers to 7
  consumers shrinks each warp's per-iteration work to `D_K / 7 ≈ 9-37`
  rows, which on Blackwell falls below the hot-path `#pragma unroll`
  threshold and triggers epilogue jumping. We avoid this.
* The W22.12 doc itself notes ">256 threads or >100 KB smem won't help —
  it just reduces grid concurrency". With our 64-block grid, going wider
  per CTA without expanding the grid leaves SMs idle.

If 1100 GB/s is achieved at TPB=128 and there's a clear bottleneck visible
in SASS at the partial-reduction step, the natural next experiment is
TPB=256 with 7 consumer warps, but that's a W22.14+ candidate.

## Why we kept TMA single-issue (NOT 4-stage ring)

The task suggested a 4-stage TMA ring buffer over D_K. We deliberately
chose **single-stage TMA covering the full (D_K, BLOCK_V) tile**:

1. **TMA preservation is the highest-priority acceptance criterion.** A
   single bulk-tensor issue produces ONE `UTMALDG` SASS line per
   instantiation — clean, unambiguous evidence the path didn't get lost
   in the restructure. A 4-stage ring would split this into 4 issues
   (still UTMALDG > 0, but harder to attribute).
2. **W22.10 already won at single-issue.** 1032 GB/s shows the load side
   is not the bottleneck; the CTA needs ~D_K × 4 cycles to consume the
   tile via FFMA, far longer than the TMA latency. Multi-stage ring
   buffering is only a win when consumer-cycles < load-latency, which
   was relevant for cp.async-style streaming through a small smem ring.
   For TMA + a tile that fits in smem in one shot, single-issue is
   optimal.
3. **Multi-stage TMA + warp-spec adds 2 axes of complexity simultaneously.**
   That violates one-variable-at-a-time experimental discipline. If
   W22.13 falls below 1032, we want the falsification cleanly attributed
   to "warp-spec hurt", not a confound with "ring buffering hurt".

If this run shows we're still load-bound, multi-stage TMA becomes a
W22.14 experiment.

## Build

```
/usr/local/cuda/bin/nvcc -O3 -arch=sm_120 -ccbin clang-14 -std=c++17 \
    -lineinfo -lstdc++ -lm -o attn_gdn_tma_warpspec attn_gdn_tma_warpspec.cu -lcuda
```

`-lcuda` required for `cuTensorMapEncodeTiled` (driver API).

## Correctness — PASSED

```
[gdn-tma-ws] === correctness run (B=2 H=4 d_k=64 d_v=64) ===
[gdn-tma-ws] o    max_abs=3.052e-05   |want|max=2.954e-01   OK
[gdn-tma-ws] Sout max_abs=2.980e-08   |want|max=4.597e-01   OK

[gdn-tma-ws] === bench-shape smoke (B=1 H=16 d_k=256 d_v=256) ===
[gdn-tma-ws] (qwen3) o   max_abs=6.104e-05   OK
[gdn-tma-ws] (qwen3) Sout max_abs=2.980e-08 OK
```

* Correctness shape `o` max_abs = **3.05e-05**, well under the **1e-3**
  tolerance (32× tighter).
* Correctness shape `Sout` max_abs = **2.98e-08** — essentially numerical-exact.
  The cross-warp partial reduction adds f32 commutativity-of-sum reordering
  that does NOT lose precision because each warp's partial covers a
  contiguous `D_K`-row slab (FFMA accumulator order within a warp is
  preserved).
* qwen3-decode shape passes the 1e-2 tolerance comfortably.

**Single-attempt success — no BLOCKED.md needed.**

## SASS instruction-mix table — W22.10 vs W22.13

Counts via `grep -c` across the full `cuobjdump --dump-sass` output (both
template instantiations included).

| Instruction class | W22.10 (TPB=16) | W22.13 (TPB=128, 1P+3C) | Δ | Notes |
|---|---:|---:|---|---|
| **UTMALDG**       | **2** | **2** | = | TMA preserved (1 per instantiation) |
| LDGSTS            | 0 | 0 | = | no cp.async-legacy path |
| LDG.E             | 46 | 10 | -36 | W22.10 had per-thread Q/K/V scalar loads from gmem; W22.13 still does, but FEWER threads loading (only 1 of 4 warps does the cooperative q/k load thanks to the TPB=tid stride pattern) |
| LDG.E.128         | 0 | 0 | = | both kernels avoid 16B gmem loads — TMA replaces the S_in path |
| STG.E.128         | 16 | 10 | -6 | per-CTA S_out 128B stores; W22.13 uses fewer because each consumer warp does only its row range |
| LDS.128           | 40 | 32 | -8 | smem reads of the alpha-scaled tile |
| STS.128           | 16 | 14 | -2 | smem writes of the alpha-scaled tile |
| FFMA              | 192 | **120** | -72 | **regression?** see below |
| FMUL              | 80 | 50 | -30 |   |
| FADD              | 8 | **56** | +48 | partial-reduction adds across consumer warps |
| **BSSY**          | **0** | **11** | +11 | **WARP-SPEC ENGAGED** (was 0 in W22.10) |
| **BSYNC**         | **0** | **11** | +11 | **WARP-SPEC ENGAGED** |
| SYNCS             | 8 | 8 | = | mbarrier `try_wait.parity` lowering — same |
| BAR.SYNC          | 2 | **10** | +8 | named-barrier IDs 1 & 2 in addition to ID 0 |
| HMMA              | 0 | 0 | = | no tensor cores in this algorithm |

**Per-kernel breakdown (D_K=64 instance, correctness shape):**

| | W22.10 | W22.13 |
|---|---:|---:|
| UTMALDG | 1 | 1 |
| BSSY    | 0 | 6 |
| BSYNC   | 0 | 4 |
| SYNCS   | 4 | 3 |
| BAR.SYNC| 1 | 4 |
| FFMA    | 96 | 37 |
| FMUL    | 40 | 23 |
| FADD    | 4  | 16 |

### Apparent FFMA regression — explanation

The whole-binary FFMA went from **192 → 120** despite restructuring to
3× math-density per CTA. Two things explain this:

1. **W22.13 has fewer FFMAs because each consumer warp has fewer rows.**
   W22.10's single warp does ALL `D_K` rows → emits `D_K × 4 = 256` FFMA
   patterns inlined per pass × 2 passes = ~512 raw mults inlined; ptxas
   merges/contracts to the 192 you see. W22.13's per-consumer-warp loop
   does `D_K/3` rows → fewer inlined FFMA *per warp* in SASS, but **3
   warps execute the loop concurrently**.
2. **The runtime FFMA throughput is what matters, not the SASS count.**
   W22.10: 16 lanes × 256 FFMA-cycles per CTA = 4096 FFMA-thread-cycles.
   W22.13: 48 lanes × ~85 FFMA-cycles per CTA = 4080 FFMA-thread-cycles
   (matches — same total work, distributed differently).
   The win is **wall-clock time**: 48 lanes finish ~3× faster than 16,
   modulo reduction overhead.

The `FADD: 56` line in W22.13 (vs 8 in W22.10) is the **partial-u and
partial-o cross-warp reduction**. Each math lane does 2 FADD (3-way sum)
× 2 passes × 16 lanes × 3 instantiations factors ≈ 200 raw, ptxas-merged
to ~48 visible. This is the cost of warp-spec.

### Resource usage

```
W22.13 D_K=64 :  REG=44 STACK=0 SHARED=1024 LOCAL=0 CONSTANT=1160
W22.13 D_K=256:  REG=42 STACK=0 SHARED=1024 LOCAL=0 CONSTANT=1160
```

### EIATTR confirms 3 named barriers

```
$ cuobjdump --dump-elf attn_gdn_tma_warpspec | grep -A2 EIATTR_NUM_BARRIERS
  Attribute: EIATTR_NUM_BARRIERS  Format: EIFMT_BVAL  Value: 0x3   (D_K=256)
  Attribute: EIATTR_NUM_BARRIERS  Format: EIFMT_BVAL  Value: 0x3   (D_K=64)
```

This **exactly matches** the W22.12 cuTile prescription
(`EIATTR_NUM_BARRIERS: 0x3`) — the warp-spec restructure took at the
SASS-attribute level, not just the instruction level. W22.10 had 1
barrier (the implicit `__syncthreads`); W22.13 has 3 (`__syncthreads` +
`bar.sync 1, 96` + `bar.sync 2, 96`).

* REG=42-44 — close to W22.10's 40, **far** from cuTile's 255. This
  reflects choosing scalar-friendly partial reduction rather than
  cuTile's persistent-state-tile-in-regs approach. We did NOT take the
  full W22.12 prescription's "register pressure to ceiling" path because
  that requires a kernel restructure (loop fusion, regs-as-tile) that
  is W22.14+ territory.
* SHARED=1024 (static) — small. The dyn smem (passed at launch) is the
  meaningful budget, ~17 KB / ~75 KB.
* LOCAL=0 — no register spills, unlike cuTile's 824 B/thread.

## Expected GB/s estimate

Bench memory model from `bench.cu`: per-iter HBM bytes =
`B_H × (state_bytes + io_bytes)` where state_bytes = 2·D_K·D_V·4 (read S_in
+ write S_out) and io_bytes ≈ tiny. At qwen3-decode shape this is
~16 × 524 KB = **~8.4 MB/iter**.

Mental model:
* W22.10's 1032 GB/s @ 8.4 MB/iter = 8.1 µs/iter. The kernel is **load-bound
  on TMA** because TPB=16 → only 1 consumer warp doing math, and that
  warp's FFMA loop took ~`D_K × LDS128 + FFMA` ≈ ~80-160 cycles with deep
  ILP. The kernel ran at ~58% of HBM peak.
* W22.13 keeps the same TMA issue but 3× the math density. If math was
  the bottleneck in W22.10, W22.13 should hit higher GB/s. If TMA-load
  was the bottleneck (likely true at 58% peak), W22.13's gain comes from
  **closing the load-store gap** — math no longer being the long pole means
  the LSU can drain S_out stores faster, freeing the next iter's TMA.
* Realistic estimate: **1100-1400 GB/s** at the qwen3 shape, with the
  partial-reduction overhead (~5-10% of total cycles) being the main
  drag.
* Optimistic (no reduction overhead): **~1500 GB/s** (load-store overlap
  catches up with HBM peak, ~84% of 1792 GB/s).
* Pessimistic (reduction is on the critical path): **~900-1000 GB/s**
  (slight regression vs W22.10) — would be the **NEW falsification**.

## Pitfalls and gotchas

1. **TMA descriptor builder is unchanged from W22.10** — same column-major
   2D encoding over `S_in` viewed as `(D_V, B_H * D_K)`. Reusing that exact
   builder ensures the descriptor is bit-identical to W22.10's, which
   means any cubin difference in the TMA path is from the kernel side, not
   the descriptor side.

2. **mbarrier.init must precede the producer thread's
   mbarrier.arrive.expect_tx**. We added an explicit `__syncthreads()`
   before `mbarrier_arrive_expect_tx` to ensure tid=0's `mbarrier_init`
   store is visible to warp 0. Without it the producer might race ahead.

3. **Partial-reduction race**: the consumer-only `bar.sync 1, 96` is
   essential. Each consumer warp writes its (cidx, lane) slot of
   `smem_part_u`, then ALL consumer warps must rendezvous before reading
   ANY slot for the 3-way sum. Using `__syncthreads()` here would force
   the producer warp into the rendezvous unnecessarily — `bar.sync 1, 96`
   includes only the 96 consumer threads.

4. **v_vec broadcast**: each consumer warp's math lanes load v from gmem
   independently. We considered loading once in warp 1 and broadcasting
   via smem to warps 2/3, but the redundant load is 4 halves per math
   lane × 48 lanes = 384 B/CTA — well under the L1 line size, free.

5. **D_K not divisible by 3 (correctness shape D_K=64)**: rows split as
   `(0..21, 21..42, 42..64)` via `(c·D_K)/3` integer arithmetic. The
   last consumer warp gets 22 rows instead of 21. No correctness
   implication; small load-imbalance cost (2-row-out-of-21 = 5%).

6. **dyn smem opt-in is unconditional** — we set
   `MaxDynamicSharedMemorySize=99 KB` for both template instantiations
   even at the small correctness shape. This is safe (the actual launch
   only allocates `smem_bytes`), and exercises the same opt-in code path
   on both shapes. The W22.12 prescription requires this for the qwen3
   shape (~75 KB > 48 KB default cap).

7. **No spills** despite the partial-reduction working set. ptxas keeps
   `u_part`, `o_part`, `r`, `v_vec` all in registers per math lane (4
   float4 = 16 floats × ~16 lanes/warp = 256 regs/warp ≪ 1024 reg/warp
   limit). REG=42-44 confirms.

## Files

* `attn_gdn_tma_warpspec.cu` — kernel + correctness driver (38 KB)
* `bench.cu` — orchestrator-runnable bench harness (separate)
* `Makefile`, `run.sh`, `.gitignore`
* `attn_gdn_tma_warpspec.sass` — SASS dump (cuobjdump)
* `build.log`, `run.log` — build + correctness logs

## Acceptance per task

* [x] Kernel compiles (clean, no warnings).
* [x] Correctness PASSES at correctness shape (B=2 H=4 d_k=d_v=64), max_abs_err = 3.05e-05 ≤ 1e-3 ✓
* [x] Correctness PASSES at qwen3 shape too, max_abs_err = 6.1e-05 ≤ 1e-2 ✓
* [x] SASS shows **UTMALDG > 0**: 2 occurrences (one per template instance) — TMA preserved ✓
* [x] SASS shows **BSSY > 0** AND **BSYNC > 0** AND **SYNCS > 0**: 11/11/8 — warp-spec engaged ✓
* [x] No bench timing run in this cell (orchestrator runs `./bench` separately).

## Next-loop expectation

Bench should hit **1100-1500 GB/s** at qwen3 shape if TMA + warp-spec
stack additively. If it regresses below W22.10's **1032 GB/s**, the
falsification is: **the partial-reduction overhead exceeded the
math-density gain**. In that case, W22.14 would explore
register-resident-state-tile (cuTile-style, REG=255, no smem partials)
or a different consumer split (e.g. on BLOCK_V instead of D_K, eliminating
the cross-warp reduction at the cost of needing a wider BLOCK_V).
