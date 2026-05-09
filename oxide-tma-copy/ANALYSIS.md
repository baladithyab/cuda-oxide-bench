# Wave 6 W6B — cuda-oxide TMA Copy on RTX 5090

## What TMA is

The **Tensor Memory Accelerator (TMA)** is a Hopper-and-newer hardware unit (sm_90+, includes Blackwell sm_100/sm_120) that performs bulk asynchronous copies of multi-dimensional tensor tiles between global and shared memory. A single thread issues a `cp.async.bulk.tensor.Nd` instruction referencing a pre-encoded `CUtensorMap` descriptor; the TMA engine then streams the tile, freeing the warp for compute and signalling completion via an `mbarrier`. This removes the need for hand-rolled per-thread async-copy pipelines and is a prerequisite for efficient Hopper/Blackwell GEMMs (e.g. `tcgen05.mma`, wgmma feeder paths).

## What this example does

`oxide-tma-copy` is cuda-oxide's TMA smoke test. Two kernels:

- `tma_copy_2d_test` — copies a 64×64 f32 tile from a 256×256 global tensor into shared memory via `cp_async_bulk_tensor_2d_g2s`, waits on an `mbarrier` initialised with block size (arrive-expect-tx pattern, `fence.proxy.async.shared::cta` before kicking off), then each thread writes one tile element back to a global output buffer for host verification.
- `tma_pipeline_test` — same 2D TMA fetch, smaller 32×32 tile, all 256 threads flag success.

Host verifies every element against the known `i as f32` pattern.

## Build/run result on RTX 5090 (sm_120)

- `cargo oxide run oxide-tma-copy` **builds cleanly** (3 benign warnings about `SharedArray` in TMA addresses — expected) and **both tests PASS**: `All 4096 values match!` + `All 256 threads completed successfully!`. See `run.log`.

## PTX evidence of TMA instructions

`oxide_tma_copy.ptx` (target `.target sm_100`, `.version 8.6`) emits the TMA instructions directly — not just intrinsics in the .ll:

```
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%rd13], [%rd8, {%r4, %r5}], [%rd14];
mbarrier.init.shared.b64        [__shared_mem_0], %r2;
fence.proxy.async.shared::cta;
mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %rd25, [%rd16], %r8;
mbarrier.try_wait.shared.b64    p, [%rd17], %rd25;
```

Both kernels contain a matching `cp.async.bulk.tensor.2d…` (lines 50 and 146). The driver JIT upgrades the sm_100 PTX to sm_120 SASS at module load — no sm_120-specific failures observed. No Blackwell-only features (TMA::A multicast, cluster-scope) exercised here; that lives in `tma_multicast` / `tcgen05` examples.
