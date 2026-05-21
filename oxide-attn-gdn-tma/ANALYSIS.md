# oxide-attn-gdn-tma — Analysis

**Wave C2.4: Rust-CUDA TMA path for GDN — testing cuda-oxide's TMA reach.**

## TL;DR

| Implementation | GB/s (best, qwen3_next_decode) | UTMALDG count | LDG.E.128 count | Notes |
|---|---|---|---|---|
| oxide-attn-gdn (FFMA baseline) | 276 | 0 | high | one-thread-per-d_k-row, scalar LDS reductions |
| **oxide-attn-gdn-tma (this cell)** | **278** | **2** | **0** | TMA loads S_in slab; same algorithm shape |
| cuda-attn-gdn-tma (C++ ref) | 1032 | >0 | 0 | float4 lanes + per-thread D_K loop |

Compiles ✓.  Correctness ✓ on both the (d_k=64 / d_v=64) correctness shape
and the (d_k=256 / d_v=256) qwen3_next_decode shape:
`max_abs(o) ≤ 1.7e-4`, `max_abs(S_out) ≤ 2.5e-4` — well under the 1e-3 tol.

**cuda-oxide's TMA reach is real and complete** — both the device-side intrinsic
(`cp_async_bulk_tensor_2d_g2s` lowered through cuda-oxide's MIR/NVVM path,
**not** raw inline PTX) and the host-side
`cuTensorMapEncodeTiled` driver-API binding land on Blackwell sm_120 in one
build. Going the C++ TMA-style perf gain (~3.7×) is **not** a TMA-availability
problem — it's an algorithm-restructure problem (see "Why GB/s didn't move").

## What changed vs oxide-attn-gdn

* Added kernel parameter `tensor_map_s_in: *const TmaDescriptor`.
* Replaced the per-thread S_in row LDG with one TMA bulk-tensor load
  per block. Tile shape:
    * dk64  : (64 rows, 32 cols) f32 = 8 KiB / block
    * dk256 : (256 rows, 32 cols) f32 = 32 KiB / block
* Added a single `Barrier` per block. Thread 0 issues
  `cp_async_bulk_tensor_2d_g2s` and `mbarrier_arrive_expect_tx(1, tile_bytes)`;
  every other thread does `mbarrier_arrive`. All threads spin on
  `mbarrier_try_wait`. Mirrors `oxide-tma-copy` exactly.
* Host-side: `encode_s_in_descriptor()` calls `cuTensorMapEncodeTiled` from
  `cuda_core::sys` with the same column-major-innermost-first convention as
  `cuda-attn-gdn-tma/attn_gdn_tma.cu`. The 128-byte descriptor is
  `transmute_copy`-ed into a `[u8; 128]` and uploaded as a `DeviceBuffer`
  (cuda-oxide kernels need device-resident pointers for `*const TmaDescriptor`
  args; cf. oxide-tma-copy line 272 — same trick).
* Two-kernel split (dk64, dk256) preserved — same launch shape as the FFMA
  baseline.

## SASS evidence

`oxide_attn_gdn_tma.sass` (cuobjdump --dump-sass on the cubin emitted by
cuda-oxide):

```
HMMA insts  : 0
FFMA insts  : 64
UTMALDG ins : 2          ← TMA reaches the metal
LDG.E.128   : 0          ← was the dominant traffic in the FFMA baseline
LDG.E (any) : 72         ← only the scalar q/k/v/alpha/beta header loads
STG.E       : 66
LDS         : 1964       ← reading the TMA-deposited tile from shared
MBAR        : 2          ← mbarrier instructions present
```

The two `UTMALDG.2D` opcodes are exactly the two kernels' single bulk loads:

```
UTMALDG.2D [UR4], [UR10] ;
```

`UR10` is the uniform register holding the descriptor pointer; `UR4` is the
uniform shared-memory destination address. This is the canonical
post-ptxas SASS form. cuda-oxide → libNVVM → ptxas all cooperated; no
inline-PTX-in-Rust hacks were necessary.

## Why GB/s didn't move

The C++ `cuda-attn-gdn-tma` kernel reaches 1032 GB/s by combining TMA
**with** a fundamentally different parallelization:

* **C++**: `blockDim = BLOCK_V/4 = 16`, each thread owns 4 d_v columns
  (`float4`), and iterates the D_K=256 loop in registers. Two
  `LDS.128`/`STS.128` per (k, lane) plus one `STG.128` per (k, lane) —
  vectorized loads/stores, zero block-wide reductions.
* **oxide here**: `blockDim = D_K = 256`, one thread per d_k row, BV=32
  scalar columns owned per thread, two block-wide tree reductions per BV
  column (256 → 128 → … → 1) per BV iteration. That's
  `BV * 2 * log2(D_K) = 32 * 2 * 8 = 512` `__syncthreads` per block plus
  ~16k LDS for the partial-product writes/reads.

The FFMA baseline was 276 GB/s and **wasn't** HBM-bound — it was
shared-memory-reduction-bound and barrier-bound. Adding TMA on the front
end:
* Saves: ~one LDG.E.128/thread × 256 threads × 8 cycles ≈ trivial vs the
  reductions.
* Costs: extra mbarrier init + arrive_expect_tx + try_wait spin + a
  proxy fence + the LDS replays of the deposited tile.

Net: ~278 vs ~276 GB/s. **TMA is not the bottleneck for this algorithm
shape** — the reduction structure is. To get the 1032 GB/s win we'd need
to also restructure the kernel along C++'s lines: smaller blocks, per-thread
register accumulators over the D_K dim, vectorized LDS. That's a
follow-up cell ("oxide-attn-gdn-tma-vlanes" or similar), not a TMA-reach
question.

## Expected upper bound if restructured

If oxide adopts the C++ float4-lane structure and keeps TMA, projecting from
the FFMA→C++-TMA gap is:

* C++ baseline (cuda-attn-gdn) was 417.7 GB/s with float4 lanes; TMA brought
  it to 1032 GB/s — **2.47×**.
* Oxide FFMA baseline w/ the *same* float4-lane structure would likely sit
  in the 350–450 GB/s range (FFMA quality matches nvcc here, but cuda-oxide
  still emits scalar LDS where nvcc emits LDS.128 — Wave 4 finding).
* Apply 2.47× → ~870–1110 GB/s as the realistic ceiling for a
  cuda-oxide-with-TMA-and-vlanes follow-up.

So **oxide is not blocked from reaching ≥800 GB/s; it just needs the
algorithm restructure as well.** TMA is *necessary but not sufficient*.

## Pitfalls hit / avoided

1. **Descriptor lifetime.** `CUtensorMap` is a 128-byte opaque host blob.
   You cannot pass it to a cuda-oxide kernel as `&descriptor` — the kernel
   parameter must be a *device* pointer. We `transmute_copy` to `[u8; 128]`
   and `DeviceBuffer::from_host` it, then cast the device pointer to
   `*const TmaDescriptor`. This is the exact pattern from
   `oxide-tma-copy/src/main.rs:272`.

2. **Proxy fence after `mbarrier_init`.** Without
   `fence_proxy_async_shared_cta()` after init, the TMA async proxy can read
   the not-yet-visible barrier and deadlock. The first build had this fence
   from copying the oxide-tma-copy template; removing it would silently
   break under load.

3. **`mbarrier_init(arrive_count = block_size)`.** All threads of the block
   participate in the barrier (one with `arrive_expect_tx`, the rest with
   `arrive`). If you use `arrive_count = 1` (issuer-only), the other threads
   would race ahead and read the tile before TMA lands.

4. **TMA descriptor coordinate convention.** Innermost dim first
   (D_V = 256 here), outer next (B*H * D_K). Box dims `[BV, D_K]` in the
   same order. Coords passed to the device intrinsic are `(bv*BV, bh*D_K)`.
   This matches the C++ `cuda-attn-gdn-tma` exactly — straight port.

5. **`load_kernel_module` filename.** cuda-oxide expects the PTX/cubin to
   be named after the crate with hyphens → underscores:
   `oxide_attn_gdn_tma`. Mismatch silently fails at runtime.

6. **No `static mut SharedArray` warnings to fix.** The 28 build warnings
   (`uses SharedArray, which cuda-oxide lowers to per-block CUDA shared
   memory`) are cuda-oxide telling you the lowering happened *correctly*.
   They are noise; the FFMA baseline has the same set.

## Files written

* `Cargo.toml` — three cuda-oxide deps
* `rust-toolchain.toml` — `nightly-2026-04-03`
* `src/main.rs` — two TMA kernels (dk64, dk256) + host harness with
  TMA descriptor encoder + correctness + bench
* `run.sh` — build+run+SASS analysis
* `oxide_attn_gdn_tma.cubin`, `.ll`, `.ltoir`, `.ptx` — emitted by the build
* `oxide_attn_gdn_tma.sass` — disassembly with UTMALDG = 2
* `results.csv`, `build.log`, `run.log`, `bench.log`
* `ANALYSIS.md` — this file

## Reproduce

```bash
cd /home/codeseys/cuda-exploration/oxide-attn-gdn-tma
./run.sh                  # build + correctness run + SASS scan
BENCH=1 ./run.sh          # adds the 50-iter bench on qwen3_next_decode
```
