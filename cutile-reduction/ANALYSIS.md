# cuTile reduction — port of oxide-reduction

## Kernel structure

Single-pass f32 sum reduction with one atomic output slot:

```python
@ct.kernel
def reduce_sum(a, out):
    bid = ct.bid(0)
    a_t = ct.load(a, index=(bid,), shape=(TILE_SIZE,),
                  padding_mode=ct.PaddingMode.ZERO)
    partial = ct.sum(a_t)        # tile-wide reduction → scalar
    ct.atomic_add(out, (0,), partial)
```

Launch: `grid = (cdiv(n, TILE_SIZE),)`, TILE_SIZE = 1024 f32 elements per block.
Each block loads one tile, reduces it to a scalar, and atomics that scalar
into `out[0]`.

## cuTile primitives used

| primitive              | role                                              |
|------------------------|---------------------------------------------------|
| `ct.bid(0)`            | block index in the 1-D grid                       |
| `ct.load`              | vectorised tile load with `PaddingMode.ZERO`       |
| `ct.sum(tile)`         | tile-wide reduction — compiler emits the warp-shuffle + smem 2-stage scheme that oxide-reduction writes by hand |
| `ct.atomic_add`        | single-location atomic f32 add for cross-block accumulation |

The key insight is that `ct.sum` over a whole tile **is** the cuTile
equivalent of the warp-shuffle butterfly + smem fan-in that oxide-reduction
hand-writes with `warp::shuffle_xor_f32` and `SharedArray`. Letting the
compiler generate that code is the whole point of the tile-level API.

## Pitfalls hit

- Initial draft tried a grid-stride loop at the tile level (match the fixed
  `GRID=4096` in the oxide baseline). cuTile runtime while-loops inside
  kernels are awkward; the idiomatic shape is one tile per block with a
  variable grid. Switched to `grid = cdiv(n, TILE_SIZE)` and relied on
  `PaddingMode.ZERO` to handle the tail when N isn't an exact multiple.
  oxide-reduction fixes `GRID=4096`; we fix `TILE_SIZE=1024` instead — same
  total work, different parallel decomposition, same atomic contention
  pattern.
- `ct.atomic_add` signature is `(array, indices, update)` where `indices`
  is a tuple matching the array rank (see cuTile help). For the 1-D output
  buffer we pass `(0,)`, not `0`.
- Launch syntax: `ct.launch(stream.ptr, grid, kernel, args_tuple)` — the
  README's `kernel[(grid,)](args)` sugar is broken in v1.3.0.

## Correctness

Smoke test on N = 1 Mi f32 random input: `rel_err = 5.962e-08`, well below
the 1e-2 tolerance. FP32 sum across 1 Mi elements with atomic fan-in only
loses ~1 ULP compared to cupy.sum — expected.

## Bench status

`--bench` code path is wired (same SIZES = [1M, 16M, 256M] and per-iter
CSV schema `impl,kernel,n,iter,gpu_ms,gbps` as oxide-reduction) but **not
executed in this task** — the orchestrator will run the timed sweep.
WARMUP=1, ITERS=10, cudaEvent timing.
