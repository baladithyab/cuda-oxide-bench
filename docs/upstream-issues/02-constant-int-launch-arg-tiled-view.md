# [bug/docs] `ct.Constant[int]` launch arg cannot be used as `tiled_view` shape (TileTypeError) in v1.3.0

## Summary

In cuda-tile 1.3.0, passing tile sizes as kernel arguments typed
`ct.Constant[int]` and then referencing them inside
`array.tiled_view((tm, tk))` fails with:

```
TileTypeError: Invalid argument "tile_shape" of _m_array_tiled_view():
    Expected a constant integer tuple, but given value is not constant
```

This is the pattern used by the matmul example in `cuda/tile/tune/_tune.py`
(the docstring example for `ct.tune.exhaustive_search`), so the published
example does not run as-is. The `Constant[int]` specialization seemingly
is not being propagated deeply enough for `tiled_view` to see the tuple
as a compile-time constant.

The **Python-closure factory workaround** (capturing tile dims as Python
ints at kernel-definition time) works reliably and is already the idiom
used elsewhere in cuTile, so the impact is medium (working workaround
exists, but the docs-authoritative example breaks on a first copy-paste).

## Environment

| item | value |
|---|---|
| cuda-tile (cutile-python) | **1.3.0** |
| cupy | cupy-cuda13x 14.0.1 |
| CUDA toolkit | 13.2 |
| GPU | NVIDIA GeForce RTX 5090 |
| compute capability | `sm_120` |
| OS | Windows 11 host, WSL2 Ubuntu |

## Minimal reproducer — failing pattern (mirrors docs example)

```python
import cuda.tile as ct
import cupy, numpy as np

@ct.kernel
def matmul(A, B, C,
           tm: ct.Constant[int],
           tn: ct.Constant[int],
           tk: ct.Constant[int]):
    i, j = ct.bid(0), ct.bid(1)
    a_view = A.tiled_view((tm, tk), padding_mode=ct.PaddingMode.ZERO)  # ← fails here
    b_view = B.tiled_view((tk, tn), padding_mode=ct.PaddingMode.ZERO)
    acc = ct.zeros((tm, tn), ct.float32)
    for k in range(a_view.num_tiles(1)):
        tx = a_view.load((i, k))
        ty = b_view.load((k, j))
        acc = ct.mma(tx, ty, acc)
    ct.store(C, (i, j), acc)

N = 512
a = cupy.asarray(np.random.rand(N, N).astype(np.float32))
b = cupy.asarray(np.random.rand(N, N).astype(np.float32))
c = cupy.zeros((N, N), dtype=cupy.float32)
stream = cupy.cuda.get_current_stream()

# Fails at compile time on the first launch with TileTypeError:
ct.launch(stream.ptr, (N // 128, N // 128), matmul, (a, b, c, 128, 128, 16))
```

### Exact error

```
TileTypeError: Invalid argument "tile_shape" of _m_array_tiled_view():
    Expected a constant integer tuple, but given value is not constant
```

Raised from the `A.tiled_view((tm, tk), ...)` call inside the kernel body.
The `Constant[int]` annotation appears to make `tm` / `tn` / `tk`
individually recognized as compile-time-constant scalars, but wrapping
them into a Python tuple literal `(tm, tk)` loses that constant-ness at
the `_m_array_tiled_view` interface.

## Working workaround — Python-closure factory

```python
def make_matmul(tm: int, tn: int, tk: int):
    @ct.kernel
    def matmul(A, B, C):
        i, j = ct.bid(0), ct.bid(1)
        a_view = A.tiled_view((tm, tk), padding_mode=ct.PaddingMode.ZERO)
        b_view = B.tiled_view((tk, tn), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.zeros((tm, tn), ct.float32)
        for k in range(a_view.num_tiles(1)):
            tx = a_view.load((i, k))
            ty = b_view.load((k, j))
            acc = ct.mma(tx, ty, acc)
        ct.store(C, (i, j), acc)
    return matmul

matmul = make_matmul(128, 128, 16)   # Python ints captured by closure
ct.launch(stream.ptr, (N // 128, N // 128), matmul, (a, b, c))
```

This compiles and runs correctly at numerical parity with `cupy.matmul`
(`rel_err ≈ 4e-7` at N=512 on f32). The factory pattern is used
throughout the cuda-exploration repo:

- `cutile-vecadd-bench/main.py`
- `cutile-matmul/main.py`
- `cutile-matmul-tiled/main.py`
- `cutile-matmul-tiled-mixed/main.py`

## Expected behavior

Either:

1. **Fix:** the `Constant[int]` specialization should propagate through
   Python tuple construction so that `(tm, tk)` where both are
   `Constant[int]` kernel arguments is itself recognized as a constant
   integer tuple at `_m_array_tiled_view`'s type-check; or
2. **Docs-only fix:** update the `cuda/tile/tune/_tune.py` example (and
   any other example that uses `ct.Constant[int]` launch args for
   tile-shape parameters) to use the Python-closure factory pattern
   instead, and add a note in the user guide that `Constant[int]` launch
   args cannot currently be used as elements of a `tiled_view` shape
   tuple.

Our slight preference is (1), because the `Constant[int]` docstring
example is a natural expression of "I want to tune these at launch
time," which is exactly the thing `ct.tune.exhaustive_search` is set up
to do. But (2) would unblock users immediately.

## Evidence / references

- cuda-exploration `cutile-matmul-tiled/ANALYSIS.md`, section
  **"Pitfall encountered — `Constant[int]` launch args"** — reproduces
  this exact error against cuda-tile 1.3.0:
  <https://github.com/baladithyab/cuda-exploration/blob/master/cutile-matmul-tiled/ANALYSIS.md>
- The working workaround is used in four separate benchmarks in that
  repo, all passing correctness vs `cupy.matmul` / `cupy.add` references.

## Related smaller pitfalls hit in the same wave (FYI, not part of this issue)

These are already documented in `cutile-matmul-tiled-mixed/ANALYSIS.md`
and are not a request for change — just listing them so the same line of
investigation can cover them if they share a root cause:

- `CallingConvention.cutile_python_v1` is a factory method, not a value
  (must call `CallingConvention.cutile_python_v1()` — the bound method
  itself raises `TypeError: Unsupported calling convention`).
- Python-level dtype-identity branching inside `@ct.kernel` fails:
  `if dtype is ct.tfloat32: ...` raises
  `TileTypeError: Operator 'is' expects one of the operands to be None`.
  Workaround: separate kernel factories per dtype path.

## Context / cooperative note

This finding comes from **cuda-exploration**
(<https://github.com/baladithyab/cuda-exploration>), a public third-party
benchmark comparing cuda-oxide (rust-cuda), nvcc, and cuTile as GPU
programming frontends on RTX 5090 Blackwell consumer hardware. The repo
turned up a healthy picture for cuTile overall (172.5 TFLOPS at f16 matmul
from Python, TMA-bulk-load reduction wins) — this is just the single
docs-example friction we hit.

Happy to test a candidate fix or provide more context if that would help.
