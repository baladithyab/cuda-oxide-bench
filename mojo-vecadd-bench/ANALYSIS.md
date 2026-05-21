# mojo-vecadd-bench

Wave 18 Phase B.1 — Mojo's vecadd benchmarked on RTX 5090 sm_120 across
N ∈ {1M, 16M, 64M, 256M}, using `ctx.execution_time` for cudaEvent-based GPU
timing (1 warmup + 10 timed iters, GB/s reported across the full window).

## Results (Mojo 1.0.0b1, fresh idle GPU 2026-05-20)

| N | avg µs/iter | GB/s | regime |
|---|---:|---:|---|
| 1M (1,048,576) | 5.4 | **2339** | L2-cache-resident |
| 16M (16,777,216) | 114.8 | **1754** | DRAM peak |
| 64M (67,108,864) | 548.1 | **1469** | DRAM, contention |
| 256M (268,435,456) | 2048.8 | **1572** | canonical memory-bound |

## Cross-frontend comparison @ N=256M (canonical)

Re-benched on the same idle GPU 2026-05-20 thermal-window (per Wave 12
discipline):

| frontend | GB/s | vs Mojo |
|---|---:|---:|
| nvcc CUDA C++ | 1404 (today) | -10.7% |
| cuda-oxide safe | 1572 | 0% |
| cuda-oxide unchecked | 1575 | +0.2% |
| cuTile (TILE=1024) | 1565 | -0.4% |
| **Mojo** | **1572** | — |

**Result: Mojo joins cuda-oxide at 1572 GB/s on the canonical memory-bound
regime, within ±1% of cuTile.** All four hit ~90% of HBM peak (~1750 GB/s
on RTX 5090). The "Mojo column" lands cleanly in the parity zone — there
is no language tax for Mojo on streaming memory.

The N=1M datapoint at 2339 GB/s is L2-cache-resident (1M × 4B × 3 buffers =
12 MB ≪ 96 MB L2). All frontends report >2 GB/s in this regime; the number
is real but isn't a HBM-bandwidth measurement.

## Algorithm

3-buffer streaming `c[i] = a[i] + b[i]`, fp32. Memory traffic = 12 bytes/elem
(2 reads + 1 write × 4 bytes). Block size = 256 threads. Grid = `ceildiv(N, 256)`.

Kernel:

```mojo
def vector_addition(
    a: UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    c: UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    n: Int,
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid < n:
        c[tid] = a[tid] + b[tid]
```

## API discoveries

1. **`UnsafePointer[Scalar[T], MutAnyOrigin]`** is the right kernel-arg
   shape for raw mutable device pointers. The `_` unbind for origin
   defaults to immutable, so explicit `MutAnyOrigin` is required when the
   buffer is written.
2. **`@parameter def body(ctx) raises -> None:`** is the canonical Mojo
   capturing-closure pattern. The `@parameter` decorator is what makes it
   match the `def(DeviceContext) raises capturing -> None` signature
   `ctx.execution_time` expects.
3. **`fn` is deprecated** as of Mojo 1.0.0b1 — the compiler suggests
   replacing with `def`. Old tutorials using `fn closure() capturing`
   need this conversion.
4. **`out` is a Mojo keyword** (output convention) — can't be used as a
   parameter name. Use `result` or another name.
5. **`ctx.execution_time[body](num_iters) -> Int`** returns total elapsed
   time in nanoseconds across all iters. Internally records cudaEvent
   start/end pairs (per Mojo docs: "functionally equivalent to recording
   start and end events, then calculating the elapsed time"). Same
   methodology as our nvcc / cuda-oxide harnesses.

## Files

- `vecadd_bench.mojo` — single-source kernel + harness
- `run.log` — captured output

## Reproducibility

```bash
cd /home/codeseys/cuda-exploration/mojo-workspace
pixi run mojo /home/codeseys/cuda-exploration/mojo-vecadd-bench/vecadd_bench.mojo
```
