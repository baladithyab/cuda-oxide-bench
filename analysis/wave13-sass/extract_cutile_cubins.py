"""Extract cuTile cubins by monkeypatching compile_tile to dump cubins.

Approach: import cuda.tile, wrap compile_tile so each call dumps result.cubin
to a file named after the kernel. Then trigger compilation by importing the
existing main.py modules and calling their kernels (smoke test path).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

OUT_DIR = Path("/home/codeseys/cuda-exploration/analysis/wave13-sass")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Monkeypatch cuda.tile._compile.compile_tile
from cuda.tile import _compile as _ct_compile

_orig_compile_tile = _ct_compile.compile_tile

_dump_count = {}

def _patched_compile_tile(*args, **kwargs):
    res = _orig_compile_tile(*args, **kwargs)
    # args[0] is the AnnotatedFunction; try to read its name
    try:
        afunc = args[0]
        name = getattr(afunc, "name", None) or getattr(afunc, "_name", None) \
               or getattr(getattr(afunc, "_function", None), "__name__", "kernel")
    except Exception:
        name = "kernel"
    if getattr(res, "cubin", None):
        idx = _dump_count.get(name, 0)
        _dump_count[name] = idx + 1
        suffix = "" if idx == 0 else f"_{idx}"
        out_path = OUT_DIR / f"cutile_{name}{suffix}.cubin"
        Path(out_path).write_bytes(res.cubin)
        print(f"  [dump] {name} -> {out_path} ({len(res.cubin)} bytes)", flush=True)
    return res

_ct_compile.compile_tile = _patched_compile_tile

# Now invoke the smoke-test paths to trigger compilation.

# 1) cutile-reduction
print("=== cutile-reduction ===", flush=True)
sys.path.insert(0, "/home/codeseys/cuda-exploration/cutile-reduction")
import importlib
if "main" in sys.modules:
    del sys.modules["main"]
import main as red_main  # type: ignore
try:
    ok = red_main.run_correctness()
    print(f"  reduction smoke ok={ok}", flush=True)
except Exception as e:
    print(f"  reduction smoke FAILED: {type(e).__name__}: {e}", flush=True)
sys.path.remove("/home/codeseys/cuda-exploration/cutile-reduction")
del sys.modules["main"]

# 2) cutile-matmul-tiled
print("=== cutile-matmul-tiled ===", flush=True)
sys.path.insert(0, "/home/codeseys/cuda-exploration/cutile-matmul-tiled")
import main as mm_main  # type: ignore
try:
    res = mm_main.run_correctness(mm_main.CORRECTNESS_N)
    print(f"  matmul smoke results: {res}", flush=True)
except Exception as e:
    print(f"  matmul smoke FAILED: {type(e).__name__}: {e}", flush=True)

print("=== done ===", flush=True)
print("Cubins dumped:")
for f in sorted(OUT_DIR.glob("cutile_*.cubin")):
    print(f"  {f} ({f.stat().st_size} bytes)")
