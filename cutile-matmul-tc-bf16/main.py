"""Wave C2.2 — cuTile DSL bf16 tensor-core matmul, single-dtype.

Sister cell to cuda-matmul-tc-bf16 (C2.1, hand-written CUDA WMMA bf16) and a
focused single-variant cousin to cutile-matmul-tiled-mixed (W13.1, four
mixed-precision dtypes). This cell makes the BF16 input commitment explicit:
one kernel, bf16 × bf16 → f32 accumulator → bf16 store, M=N=K=4096, 50 iters.

cuTile DSL pattern (mirroring cutile-matmul-tiled-mixed exactly):
  - import cuda.tile as ct
  - @ct.kernel decorator
  - ct.zeros((BM,BN), ct.float32) for the accumulator
  - ct.mma(a, b, acc) inside K-loop; reassign returned tile to acc
  - ct.store(C, (i,j), acc.astype(C.dtype))  ← cast accumulator to bf16
  - ct.launch(stream.ptr, grid, kernel, args_tuple)

The DSL exposes no separate `cute.tensor_core_mma` user-level API. ct.mma is
the front-door; tensor-core engagement is dtype-conditional and verified for
bf16 in W13.1 (HMMA.16816.F32.BF16, 64 instructions emitted, no FFMA fallback).

Correctness: sampled vs CPU f32 reference (200 random elements), atol=2e-1
+ rtol=5e-2 — looser than cuda-matmul-tc-bf16 because the cuTile output is
stored back to bf16, so we lose ~7 mantissa bits at the store on top of the
bf16-input rounding.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time

import cuda.tile as ct
import cupy
import ml_dtypes
import numpy as np

# ── Tile shape (matches W13.1 mixed-precision cell exactly) ─────────
BM = 128
BN = 128
BK = 16

# ── Bench shape & protocol ──────────────────────────────────────────
N = 4096
WARMUP = 5
ITERS = 50

# ── Correctness sampling vs CPU f32 reference ───────────────────────
NUM_SAMPLES = 200
ATOL = 2e-1
RTOL = 5e-2


# ────────────────────────────────────────────────────────────────────
# Kernel
# ────────────────────────────────────────────────────────────────────


def make_matmul_bf16(bm: int, bn: int, bk: int):
    """BF16 × BF16 → F32 accumulator → BF16 store. Tile-level MMA."""

    @ct.kernel
    def matmul_bf16(A, B, C):
        i, j = ct.bid(0), ct.bid(1)
        a_view = A.tiled_view((bm, bk), padding_mode=ct.PaddingMode.ZERO)
        b_view = B.tiled_view((bk, bn), padding_mode=ct.PaddingMode.ZERO)
        # F32 accumulator — the *only* dtype Blackwell HMMA supports as
        # accumulator for bf16 inputs (HMMA.16816.F32.BF16).
        acc = ct.zeros((bm, bn), ct.float32)
        for k in range(a_view.num_tiles(1)):
            tx = a_view.load((i, k))   # bf16 tile
            ty = b_view.load((k, j))   # bf16 tile
            acc = ct.mma(tx, ty, acc)  # ct.mma returns a new tile; reassign
        # Down-cast accumulator to bf16 at store. The output dtype is bf16
        # (matches the C array dtype on the host side).
        ct.store(C, (i, j), acc.astype(C.dtype))

    return matmul_bf16


matmul_bf16 = make_matmul_bf16(BM, BN, BK)


# ────────────────────────────────────────────────────────────────────
# Inputs (deterministic seed) and CPU reference (sampled)
# ────────────────────────────────────────────────────────────────────


def make_inputs(n: int, seed: int = 0xC0FFEE):
    rng = np.random.default_rng(seed)
    a_np = rng.random((n, n), dtype=np.float32).astype(ml_dtypes.bfloat16)
    b_np = rng.random((n, n), dtype=np.float32).astype(ml_dtypes.bfloat16)
    a = cupy.asarray(a_np)
    b = cupy.asarray(b_np)
    return a, b, a_np, b_np


def sample_reference_rows(a_np, b_np, sample_rows, sample_cols):
    """Compute exactly the requested (row, col) entries of A @ B in f32 on the
    CPU. We promote operands to f32 before the matmul so the reference is
    high-precision."""
    a32 = a_np.astype(np.float32)
    b32 = b_np.astype(np.float32)
    refs = np.empty(len(sample_rows), dtype=np.float32)
    for k, (r, c) in enumerate(zip(sample_rows, sample_cols)):
        refs[k] = float(np.dot(a32[r, :], b32[:, c]))
    return refs


def run_correctness(n: int, kernel) -> tuple[bool, float, float, int]:
    """Execute the kernel once at size n and check NUM_SAMPLES random output
    entries vs an f32 CPU reference. Returns (ok, max_abs, max_rel, n_fail)."""
    stream = cupy.cuda.get_current_stream()
    a, b, a_np, b_np = make_inputs(n)
    out = cupy.zeros((n, n), dtype=ml_dtypes.bfloat16)

    grid = (n // BM, n // BN)
    ct.launch(stream.ptr, grid, kernel, (a, b, out))
    cupy.cuda.runtime.deviceSynchronize()

    out_np = cupy.asnumpy(out).astype(np.float32)

    rng = np.random.default_rng(0xBEEF)
    rows = rng.integers(0, n, size=NUM_SAMPLES)
    cols = rng.integers(0, n, size=NUM_SAMPLES)
    refs = sample_reference_rows(a_np, b_np, rows, cols)
    got = out_np[rows, cols]

    abs_err = np.abs(got - refs)
    rel_err = abs_err / np.maximum(np.abs(refs), 1e-6)
    # combined atol+rtol gate (numpy.allclose semantics, elementwise)
    fail_mask = abs_err > (ATOL + RTOL * np.abs(refs))
    n_fail = int(fail_mask.sum())
    ok = n_fail == 0
    return ok, float(abs_err.max()), float(rel_err.max()), n_fail


# ────────────────────────────────────────────────────────────────────
# Bench
# ────────────────────────────────────────────────────────────────────


def run_bench(n: int, kernel, csv_path: str) -> dict:
    stream = cupy.cuda.get_current_stream()
    a, b, _, _ = make_inputs(n)
    out = cupy.zeros((n, n), dtype=ml_dtypes.bfloat16)

    grid = (n // BM, n // BN)
    flops = 2.0 * (n ** 3)

    # Warmup (drops JIT + first-launch overhead).
    for _ in range(WARMUP):
        ct.launch(stream.ptr, grid, kernel, (a, b, out))
    cupy.cuda.runtime.deviceSynchronize()

    # Timed iters with cudaEvent pairs (matches W13.1 protocol).
    starts = [cupy.cuda.Event() for _ in range(ITERS)]
    ends = [cupy.cuda.Event() for _ in range(ITERS)]
    cpu_t0 = time.perf_counter()
    for i in range(ITERS):
        starts[i].record(stream)
        ct.launch(stream.ptr, grid, kernel, (a, b, out))
        ends[i].record(stream)
    stream.synchronize()
    cpu_wall_s = time.perf_counter() - cpu_t0

    rows = []
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["impl", "kernel", "n", "iter", "gpu_ms", "tflops"])
        for i in range(ITERS):
            gpu_ms = cupy.cuda.get_elapsed_time(starts[i], ends[i])
            tflops = (flops / 1e12) / (gpu_ms / 1000.0)
            w.writerow(["cutile", "matmul_tc_bf16", n, i,
                        f"{gpu_ms:.6f}", f"{tflops:.6f}"])
            rows.append((gpu_ms, tflops))

    ms_sorted = sorted(r[0] for r in rows)
    tf_sorted = sorted(r[1] for r in rows)
    summary = {
        "n": n,
        "iters": ITERS,
        "best_ms": ms_sorted[0],
        "median_ms": ms_sorted[ITERS // 2],
        "worst_ms": ms_sorted[-1],
        "best_tflops": tf_sorted[-1],
        "median_tflops": tf_sorted[ITERS // 2],
        "worst_tflops": tf_sorted[0],
        "cpu_wall_s": cpu_wall_s,
    }
    return summary


# ────────────────────────────────────────────────────────────────────
# Entrypoint
# ────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", default=False,
                        help="correctness check only, default N=512")
    parser.add_argument("--bench", action="store_true", default=False,
                        help="full bench at N=4096 (default mode)")
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--csv-out", default="results.csv")
    args = parser.parse_args()
    if not args.smoke and not args.bench:
        args.bench = True  # default

    print(f"cuda-tile version: {ct.__version__}")
    print(f"cupy: {cupy.__version__}")
    props = cupy.cuda.runtime.getDeviceProperties(0)
    print(f"device: {props['name'].decode()}")
    print(f"compute capability: sm_{props['major']}{props['minor']}")
    print(f"BM={BM} BN={BN} BK={BK}  N={args.n}  ITERS={ITERS}")
    print()

    # Correctness gate at the bench size (sampled).
    print(f"=== correctness (sampled, N={args.n}) ===")
    ok, mae, mre, n_fail = run_correctness(args.n, matmul_bf16)
    print(f"  max_abs={mae:.4e}  max_rel={mre:.4e}  "
          f"failed={n_fail}/{NUM_SAMPLES}  atol={ATOL}  rtol={RTOL}")
    if not ok:
        print("  CORRECTNESS FAIL", file=sys.stderr)
        return 2
    print("  CORRECTNESS OK")
    print()

    if args.smoke and not args.bench:
        return 0

    print(f"=== bench (N={args.n}, {WARMUP} warm + {ITERS} iters) ===")
    summary = run_bench(args.n, matmul_bf16, args.csv_out)
    print()
    print("─" * 60)
    print(f"  best   : {summary['best_ms']:8.3f} ms   "
          f"{summary['best_tflops']:8.2f} TFLOPS")
    print(f"  median : {summary['median_ms']:8.3f} ms   "
          f"{summary['median_tflops']:8.2f} TFLOPS")
    print(f"  worst  : {summary['worst_ms']:8.3f} ms   "
          f"{summary['worst_tflops']:8.2f} TFLOPS")
    print(f"  CPU wall: {summary['cpu_wall_s']:.3f} s for "
          f"{ITERS} dispatches")
    print("─" * 60)

    # Reference numbers from prior waves on this exact RTX 5090 box.
    print()
    print("  Reference @ N=4096 (RTX 5090, sm_120):")
    print("    cuBLAS bgemm (W14.1, bf16→f32acc) :  219.24 TF best /"
          " 217.4 TF median")
    print("    cuBLAS hgemm (W14.1, f16→f32acc)  :  218.41 TF best")
    print("    cuTile mixed-bf16 (W13.1)         :  159.8  TF best /"
          "  ~160 TF median")
    return 0


if __name__ == "__main__":
    sys.exit(main())
