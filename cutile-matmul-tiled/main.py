"""Wave 12 — cuTile tiled matmul. Port of oxide-matmul-tiled.

Two kernel variants:

  1. matmul_tiled         — block-level tile, BM × BN output, BK K-step, built on
                            ct.mma. This is cuTile's home turf; ct.mma lowers to
                            Tensor-Core path on Blackwell (sm_120) and is the
                            canonical matmul pattern from the cuda-tile docs
                            (see tune/_tune.py example).
  2. matmul_tiled_simple  — manual 16×16 shared tile, hand-written k-loop inside
                            the tile using element-wise ops. Mirrors the oxide
                            `matmul_tiled` (16×16 tile, acc += a[tile[ty,k]] *
                            b[tile[k,tx]] — but expressed as ct.matmul on a 16×16
                            tile. Fallback comparison point if `ct.mma` path has
                            trouble.

Correctness only (--smoke). Bench sweep is code-complete but gated behind
--bench (NOT RUN in this task).

Reference oxide implementation:
  - oxide-matmul-tiled/src/main.rs   (BM=BN=BK=16 shared tile)
  - oxide-matmul-tiled-microtile/src/main.rs (64×64 block, 4×4 reg microtile)
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time

import cuda.tile as ct
import cupy
import numpy as np

# Default tile sizes — bigger block tiles are Blackwell-friendly; BK=16 is the
# safe starting point for f32 mma lowering.
BM = 128
BN = 128
BK = 16

# Simple (small) variant tile size — matches oxide's 16×16.
SIMPLE_TILE = 16

# Correctness N.
CORRECTNESS_N = 512

# Bench sweep sizes (NOT RUN in this task; code path gated behind --bench).
BENCH_SIZES = [1024, 2048, 4096]
WARMUP = 1
ITERS = 10


# ────────────────────────────────────────────────────────────────────
# Kernels are built by a factory so the tile dims become module-level
# constants captured in the kernel closure (same pattern as the vecadd
# bench). This avoids the `ct.Constant[int]` launch-arg path which in
# cuda-tile 1.3.0 doesn't propagate the int into `tiled_view(shape)`.
# ────────────────────────────────────────────────────────────────────


def make_matmul_tiled(tm: int, tn: int, tk: int):
    """Block-tile matmul: block (i, j) computes C[i*tm:(i+1)*tm, j*tn:(j+1)*tn]
    by iterating over the K dim in chunks of tk and accumulating via ct.mma.
    This is the canonical cuTile matmul pattern."""

    @ct.kernel
    def matmul_tiled(A, B, C):
        i, j = ct.bid(0), ct.bid(1)
        a_view = A.tiled_view((tm, tk), padding_mode=ct.PaddingMode.ZERO)
        b_view = B.tiled_view((tk, tn), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.zeros((tm, tn), ct.float32)
        for k in range(a_view.num_tiles(1)):
            tx = a_view.load((i, k))
            ty = b_view.load((k, j))
            acc = ct.mma(tx, ty, acc)
        ct.store(C, (i, j), acc.astype(C.dtype))

    return matmul_tiled


def make_matmul_tiled_simple(ts: int):
    """Small square-tile variant (ts×ts). Closer to oxide-matmul-tiled's
    16×16 block tile shape. Fallback / comparison point."""

    @ct.kernel
    def matmul_tiled_simple(A, B, C):
        i, j = ct.bid(0), ct.bid(1)
        a_view = A.tiled_view((ts, ts), padding_mode=ct.PaddingMode.ZERO)
        b_view = B.tiled_view((ts, ts), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.zeros((ts, ts), ct.float32)
        for k in range(a_view.num_tiles(1)):
            tx = a_view.load((i, k))
            ty = b_view.load((k, j))
            acc = ct.mma(tx, ty, acc)
        ct.store(C, (i, j), acc.astype(C.dtype))

    return matmul_tiled_simple


# Instantiate the two kernel shapes we will test.
matmul_tiled = make_matmul_tiled(BM, BN, BK)
matmul_tiled_simple = make_matmul_tiled_simple(SIMPLE_TILE)


# ────────────────────────────────────────────────────────────────────
# Correctness check
# ────────────────────────────────────────────────────────────────────

def run_correctness(n: int) -> dict[str, bool]:
    """Run both kernels at N=n and verify against cupy.matmul. Returns
    dict of kernel_name -> pass/fail. Kernels that fail to compile are
    recorded separately."""
    stream = cupy.cuda.get_current_stream()

    rng = cupy.random.default_rng(0xC0FFEE)
    a = rng.random((n, n), dtype=cupy.float32)
    b = rng.random((n, n), dtype=cupy.float32)
    expected = cupy.matmul(a, b)

    results: dict[str, bool] = {}
    errors: dict[str, str] = {}

    # ── Variant 1: matmul_tiled (BM×BN, BK=16) ──────────────────────
    if n % BM == 0 and n % BN == 0 and n % BK == 0:
        out1 = cupy.zeros((n, n), dtype=cupy.float32)
        grid1 = (n // BM, n // BN)
        try:
            ct.launch(stream.ptr, grid1, matmul_tiled, (a, b, out1))
            cupy.cuda.runtime.deviceSynchronize()
            max_err = float(cupy.max(cupy.abs(out1 - expected)))
            ref_mag = float(cupy.max(cupy.abs(expected)))
            rel = max_err / max(ref_mag, 1e-6)
            ok = rel < 1e-3
            results["matmul_tiled"] = ok
            print(f"[cutile-matmul_tiled] N={n} BM={BM} BN={BN} BK={BK}  "
                  f"max_abs_err={max_err:.3e} rel_err={rel:.3e}  {'OK' if ok else 'FAIL'}")
        except Exception as e:
            results["matmul_tiled"] = False
            errors["matmul_tiled"] = f"{type(e).__name__}: {e}"
            print(f"[cutile-matmul_tiled] COMPILE/RUNTIME ERROR: {type(e).__name__}: {e}",
                  file=sys.stderr)
    else:
        print(f"[cutile-matmul_tiled] skip: N={n} not divisible by BM={BM}/BN={BN}/BK={BK}")

    # ── Variant 2: matmul_tiled_simple (16×16) ──────────────────────
    if n % SIMPLE_TILE == 0:
        out2 = cupy.zeros((n, n), dtype=cupy.float32)
        grid2 = (n // SIMPLE_TILE, n // SIMPLE_TILE)
        try:
            ct.launch(stream.ptr, grid2, matmul_tiled_simple, (a, b, out2))
            cupy.cuda.runtime.deviceSynchronize()
            max_err = float(cupy.max(cupy.abs(out2 - expected)))
            ref_mag = float(cupy.max(cupy.abs(expected)))
            rel = max_err / max(ref_mag, 1e-6)
            ok = rel < 1e-3
            results["matmul_tiled_simple"] = ok
            print(f"[cutile-matmul_tiled_simple] N={n} TS={SIMPLE_TILE}  "
                  f"max_abs_err={max_err:.3e} rel_err={rel:.3e}  {'OK' if ok else 'FAIL'}")
        except Exception as e:
            results["matmul_tiled_simple"] = False
            errors["matmul_tiled_simple"] = f"{type(e).__name__}: {e}"
            print(f"[cutile-matmul_tiled_simple] COMPILE/RUNTIME ERROR: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
    else:
        print(f"[cutile-matmul_tiled_simple] skip: N={n} not divisible by TS={SIMPLE_TILE}")

    del a, b, expected
    cupy.get_default_memory_pool().free_all_blocks()

    if errors:
        print("\n=== Errors ===")
        for k, v in errors.items():
            print(f"  {k}: {v}")

    return results


# ────────────────────────────────────────────────────────────────────
# Bench — code path present but NOT RUN in this task.
# ────────────────────────────────────────────────────────────────────

def run_bench(csv_path: str) -> None:
    """Timed sweep over SIZES. DO NOT RUN per task instructions —
    guarded behind --bench flag. Present so Wave 12 continuation can
    enable it without a rewrite."""
    stream = cupy.cuda.get_current_stream()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["impl", "kernel", "n", "iter", "gpu_ms", "tflops"])

        for n in BENCH_SIZES:
            rng = cupy.random.default_rng(0xC0FFEE)
            a = rng.random((n, n), dtype=cupy.float32)
            b = rng.random((n, n), dtype=cupy.float32)
            total_flops = 2.0 * (n ** 3)

            variants = []
            if n % BM == 0 and n % BN == 0 and n % BK == 0:
                variants.append(("matmul_tiled", matmul_tiled,
                                 (n // BM, n // BN)))
            if n % SIMPLE_TILE == 0:
                variants.append(("matmul_tiled_simple", matmul_tiled_simple,
                                 (n // SIMPLE_TILE, n // SIMPLE_TILE)))

            for name, kernel, grid in variants:
                out = cupy.zeros((n, n), dtype=cupy.float32)
                for _ in range(WARMUP):
                    ct.launch(stream.ptr, grid, kernel, (a, b, out))
                cupy.cuda.runtime.deviceSynchronize()

                starts = [cupy.cuda.Event() for _ in range(ITERS)]
                ends = [cupy.cuda.Event() for _ in range(ITERS)]
                for i in range(ITERS):
                    starts[i].record(stream)
                    ct.launch(stream.ptr, grid, kernel, (a, b, out))
                    ends[i].record(stream)
                    stream.synchronize()
                for i in range(ITERS):
                    gpu_ms = cupy.cuda.get_elapsed_time(starts[i], ends[i])
                    tflops = (total_flops / 1e12) / (gpu_ms / 1000.0)
                    print(f"[cutile-{name}] N={n} iter={i} gpu_ms={gpu_ms:.3f} tflops={tflops:.3f}")
                    writer.writerow(["cutile", name, n, i,
                                     f"{gpu_ms:.6f}", f"{tflops:.6f}"])
                del out

            del a, b
            cupy.get_default_memory_pool().free_all_blocks()


# ────────────────────────────────────────────────────────────────────
# Entrypoint
# ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", default=True,
                        help="correctness check only (default)")
    parser.add_argument("--bench", action="store_true", default=False,
                        help="run timed sweep (NOT invoked in Wave 12 task 1)")
    parser.add_argument("--n", type=int, default=CORRECTNESS_N,
                        help="size for correctness check")
    parser.add_argument("--csv-out", default="results.csv")
    args = parser.parse_args()

    print(f"cuda-tile version: {ct.__version__}")
    print(f"cupy: {cupy.__version__}")
    props = cupy.cuda.runtime.getDeviceProperties(0)
    print(f"device: {props['name'].decode()}")
    print(f"compute capability: sm_{props['major']}{props['minor']}")
    print(f"BM={BM} BN={BN} BK={BK}  SIMPLE_TILE={SIMPLE_TILE}")
    print(f"correctness N={args.n}")
    print()

    if args.bench:
        print("Running bench sweep (timed iters)…")
        run_bench(args.csv_out)
        return 0

    # smoke path (default)
    results = run_correctness(args.n)
    print()
    print("=" * 48)
    if not results:
        print("NO KERNELS RAN — something is wrong with the setup.")
        return 2
    any_ok = any(results.values())
    all_ok = all(results.values())
    for k, ok in results.items():
        print(f"  {k}: {'PASS' if ok else 'FAIL'}")
    print("=" * 48)
    if all_ok:
        print("SMOKE TEST OK")
        return 0
    elif any_ok:
        print("SMOKE TEST PARTIAL — some kernel(s) failed")
        return 1
    else:
        print("SMOKE TEST FAILED — no kernels passed")
        return 2


if __name__ == "__main__":
    sys.exit(main())
