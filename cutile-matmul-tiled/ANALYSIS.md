# cutile-matmul-tiled — Analysis (Wave 12)

## TL;DR

**Both cuTile matmul kernels compile and produce correct results on sm_120
(RTX 5090) with cuda-tile 1.3.0.** `ct.mma` — the tensor-core accumulation
primitive — lowers cleanly and is the canonical way to express a block-tiled
matmul in cuTile. **This is the kernel class where cuTile shines and the
finding is: it works, unmodified, at f32 × f32.**

## Kernels

Two variants, both built on the same pattern (canonical from `cuda.tile.tune._tune`'s
own matmul example):

    @ct.kernel
    def matmul(A, B, C):
        i, j = ct.bid(0), ct.bid(1)
        a_view = A.tiled_view((TM, TK), padding_mode=ct.PaddingMode.ZERO)
        b_view = B.tiled_view((TK, TN), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.zeros((TM, TN), ct.float32)
        for k in range(a_view.num_tiles(1)):
            tx = a_view.load((i, k))
            ty = b_view.load((k, j))
            acc = ct.mma(tx, ty, acc)
        ct.store(C, (i, j), acc.astype(C.dtype))

1. **`matmul_tiled`** — BM=BN=128, BK=16. Mainstream Blackwell block-tile shape.
2. **`matmul_tiled_simple`** — TS=16 (both M, N, K tile == 16). Mirrors the
   oxide-matmul-tiled 16×16 shape. Fallback / direct comparison point.

Both use the factory-closure pattern: tile dims are Python locals captured
by `@ct.kernel` so they become compile-time constants in the IR (see pitfall
below).

## Smoke test result

    cuda-tile version: 1.3.0
    device: NVIDIA GeForce RTX 5090
    compute capability: sm_120

    [cutile-matmul_tiled]        N=512 BM=128 BN=128 BK=16  rel_err=4.005e-07  OK
    [cutile-matmul_tiled_simple] N=512 TS=16                rel_err=4.005e-07  OK

Reference is `cupy.matmul(a, b)`. Both kernels match to `rel_err < 1e-3`
(actual ~4e-7, essentially bit-level with single-rounding rearrangement).

## Pitfall encountered — `Constant[int]` launch args

First attempt passed tile sizes as kernel args typed `ct.Constant[int]`,
following the doc example in `cuda/tile/tune/_tune.py`:

    @ct.kernel
    def matmul_tiled(A, B, C,
                     tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]):
        a_view = A.tiled_view((tm, tk), padding_mode=ct.PaddingMode.ZERO)

That failed with:

    TileTypeError: Invalid argument "tile_shape" of _m_array_tiled_view():
        Expected a constant integer tuple, but given value is not constant

i.e. the `Constant[int]` specialization is not being propagated deeply
enough for `tiled_view((tm, tk))` in cuda-tile 1.3.0 — the tuple is seen
as a runtime value. The workaround (closure over Python ints) is easy
and matches the `make_kernel(tile_size)` pattern used in our
cutile-vecadd-bench. Worth filing upstream.

## cuTile primitives proven usable on sm_120

| Primitive                    | Works? | Notes |
| ---------------------------- | ------ | ----- |
| `ct.bid(axis)`               | yes    | unchanged from vecadd |
| `Array.tiled_view(shape, …)` | yes    | needs *compile-time-constant* shape (see pitfall) |
| `ct.zeros((M, N), dtype)`    | yes    | acc tile initialization |
| `TiledView.num_tiles(axis)`  | yes    | K-loop bound |
| `TiledView.load((i, k))`     | yes    | no explicit TMA hint; compiler chooses |
| `ct.mma(x, y, acc)`          | yes    | f32 acc. Tensor-core path assumed |
| `tile.astype(dtype)`         | yes    | dtype bridge to output |
| `ct.store(A, idx, tile)`     | yes    | shape inferred from tile |
| `ct.launch(stream, grid, k, args)` | yes | 2D grid works |
| `ct.PaddingMode.ZERO`        | yes    | boundary handling for non-multiple N |

No unsupported-feature errors; no sm_120 guardrails hit. First compile
warms slow (~several seconds) for the (128,128,16) kernel; subsequent
launches reuse the cached module. Cache dir: `.cuda_tile_cache/`.

## What remains for Wave 12 follow-up

- **Timed sweep.** Code path is present behind `--bench` (not run per task
  instructions). N ∈ {1024, 2048, 4096}, 1 warmup + 10 iters, CSV columns
  `impl,kernel,n,iter,gpu_ms,tflops`. This is where the real comparison
  lives: cuTile's ct.mma versus oxide's hand-tiled shared-memory loop.
- **Dtype sweep.** f32 `ct.mma` is in use; f16 / bf16 / tf32 would plausibly
  deliver a much bigger Tensor-Core uplift and the dtype surface is
  available per `help(ct.mma)`. Out of scope for W12.1 parity with
  oxide-matmul-tiled (f32), noted for later.
- **Tile-shape tuning.** `ct.tune.exhaustive_search` exists and the docs
  use it precisely for this kernel (`tm,tn,tk,num_ctas` grid). Good W12.2
  material.

## Frank assessment

`ct.matmul` / `ct.mma` on sm_120 RTX 5090 in cuTile 1.3.0 is **usable**.
The compile pipeline is clean, the docs provide a canonical template,
and correctness matches cupy to float-rounding precision on both a 128²
block tile and a 16² block tile. The only rough edge is the `Constant[int]`
launch-arg route; using a Python-closure factory sidesteps it entirely
and is the idiomatic cuTile pattern anyway. Whether the *performance* is
competitive with cuBLAS / oxide's register-microtile matmul is the next
question — and is deferred to the W12 bench pass.
