# cuTile naive matmul — analysis

Port of `oxide-matmul` (naive C[r,c] = sum_k A[r,k] * B[k,c]) into cuTile.
Correctness only in this wave; timed numbers come from the orchestrator's
serial bench phase.

## Kernel structure

cuTile's unit of work is a CTA (tile), not a thread. One CTA computes a
**16×16 output tile** of C — matching oxide-matmul's 16×16 thread block that
computes 16×16 output elements, just with the threads-within-a-block handled
by the cuTile compiler rather than explicitly.

```
acc = zeros((16, 16), f32)
for k in range(N // 16):
    a_tile = load(A, index=(bi, k),  shape=(16, 16))   # (BM, BK)
    b_tile = load(B, index=(k,  bj), shape=(16, 16))   # (BK, BN)
    prod   = a_tile[:, :, None] * b_tile[None, :, :]   # (16, 16, 16)
    acc    = acc + sum(prod, axis=1)                   # reduce K
store(C, index=(bi, bj), tile=acc)
```

Grid: `(N // 16, N // 16)`. Launch: `ct.launch(stream.ptr, grid, kernel, (a, b, c))`.

## cuTile primitives used

- `ct.bid(0)`, `ct.bid(1)` — block indices in the 2D grid.
- `ct.load(array, index, shape)` — tile load with 2D tile-space indexing.
- `ct.store(array, index, tile)` — tile store.
- `ct.zeros(shape, dtype)` — fresh f32 accumulator.
- `ct.sum(tile, axis=...)` — K-axis reduction inside the loop.
- Tile broadcasting via NumPy-style `x[:, :, None]` / `x[None, :, :]` indexing
  (documented in `expand_dims` stub: *"This can also be done via the NumPy-style
  syntax: `x[:, None]`"*).
- Plain `+` and `*` on tiles — no `ct.mma` / `ct.matmul` (intentional: we're
  stressing compiler quality on a hand-written accumulate, per the oxide parity
  goal).

## Design choices

1. **Kernel factory `make_kernel(n)`.** cuTile unrolls Python `for` loops at
   trace time, so the K-loop trip count must be a compile-time constant.
   Closing over `K_TILES = n // 16` in the factory gives one JIT specialization
   per matrix size, exactly matching cutile-vecadd-bench's `make_kernel(tile_size)`
   pattern.

2. **Closed-over constants (`BM`, `BN`, `BK`).** Same reason — cutile-vecadd-bench
   established this as the canonical pattern for constants inside `@ct.kernel`.

3. **Broadcast-and-sum instead of `ct.mma`.** The task explicitly asks for no
   `ct.mma` / `ct.matmul` for the naive port. Broadcast-and-sum is the most
   direct way to write `C += A @ B` from elementwise primitives: element
   `(r, c)` of `sum(a[:, :, None] * b[None, :, :], axis=1)` is
   `sum_k a[r, k] * b[k, c]`, which is the naive inner product.

## Compiler issues hit

None. The kernel compiled cleanly on first try at N=512. The main risk flagged
in the task ("16×16 may stress per-thread register count") didn't fire — cuTile
evidently handles the (16, 16, 16) intermediate without problems at this size.
A partial worry that the `:, None` broadcasting syntax might not be supported
in a traced kernel was resolved by the `expand_dims` docstring confirming
NumPy-style is valid.

## Correctness

Smoke test at **N=512** vs `cupy.matmul`:

```
max_rel_err=4.583e-07  mean_rel_err=7.403e-08  max_abs_err=6.104e-05
OK
```

Well under the 1e-3 relative-error bar; difference is just FP associativity
between cuTile's accumulation order and cuBLAS's.

## What comes next

Orchestrator runs `python main.py --bench` to get the N ∈ {1024, 2048, 4096}
sweep (1 warmup + 10 timed iters, `cupy.cuda.Event` timing, TFLOPS = 2N³/time).
We're **not** running that here.
