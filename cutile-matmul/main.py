"""cuTile naive matmul — port of oxide-matmul.

Algorithm: C[r, c] = sum_k A[r, k] * B[k, c] for f32 square matrices.

Oxide version: 1 thread per output element, 16x16 threads per block, inner k-loop
iterates the K dimension. Each thread loads `dim` elements of A and `dim` elements
of B via pointer arithmetic and accumulates a single f32 output.

cuTile natural form: each CTA computes a 16x16 output tile, not a single element.
We load 16x16 tiles of A and B along the K dimension (step 16) and accumulate
into a 16x16 output accumulator. To match the "naive" spirit of oxide-matmul
we do NOT use `ct.mma` or `ct.matmul` — instead we express the tile-level
multiply-accumulate via broadcast-and-sum, which is the most direct way to
write `C += A @ B` from elementwise primitives only.

Per the task spec, launch is `ct.launch(stream.ptr, (N//16, N//16), kernel, args)`.
Constants (BLOCK) are closed over from enclosing scope (see cutile-vecadd-bench).
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

# ──── Problem parameters ──────────────────────────────────────────────────
BLOCK = 16                       # 16x16 output tile per CTA; matches oxide BS=16
SIZES = [1024, 2048, 4096]       # N sweep for bench path only
WARMUP = 1
ITERS = 10
SMOKE_N = 512                    # small N for correctness smoke test


# ──── Kernel factory ──────────────────────────────────────────────────────
# We make the kernel a function of N (known at launch time via tile loop
# bounds). cuTile JIT-specializes per (kernel, shapes, dtypes); `N` itself
# is a runtime arg but the K-loop trip count depends on it, so we close it
# over by making the kernel accept `n` as a scalar argument and using it
# to bound the Python `for` loop. Since cuTile unrolls Python for-loops at
# trace time, `n` must be a compile-time constant — so we bake it into the
# factory (one specialization per size).
def make_kernel(n: int):
    # These constants are closed-over from enclosing scope so they appear
    # as compile-time constants inside the kernel (see cutile-vecadd-bench).
    BM = BLOCK
    BN = BLOCK
    BK = BLOCK
    K_TILES = n // BK

    @ct.kernel
    def matmul_naive(a, b, c):
        # Block indices in the 2D grid: (N/BM, N/BN)
        bi = ct.bid(0)   # row-block index   -> output rows bi*BM .. bi*BM+BM
        bj = ct.bid(1)   # col-block index   -> output cols bj*BN .. bj*BN+BN

        acc = ct.zeros((BM, BN), dtype=ct.float32)

        # Naive inner k-loop: load 16x16 tiles of A and B along K and
        # accumulate via broadcast-and-sum (no ct.mma / ct.matmul).
        for k in range(K_TILES):
            a_tile = ct.load(a, index=(bi, k),  shape=(BM, BK))
            b_tile = ct.load(b, index=(k,  bj), shape=(BK, BN))
            # a_tile: (BM, BK), b_tile: (BK, BN)
            # outer product per k: a_tile[:, :, None] * b_tile[None, :, :]
            # -> shape (BM, BK, BN); reduce K axis by summing.
            prod = a_tile[:, :, None] * b_tile[None, :, :]
            acc = acc + ct.sum(prod, axis=1)

        ct.store(c, index=(bi, bj), tile=acc)

    return matmul_naive


# ──── Correctness smoke test ──────────────────────────────────────────────
def smoke(n: int = SMOKE_N) -> int:
    assert n % BLOCK == 0, f"N={n} must be divisible by BLOCK={BLOCK}"
    stream = cupy.cuda.get_current_stream()

    print(f"cuda-tile version: {ct.__version__}")
    print(f"cupy: {cupy.__version__}")
    print(f"device: {cupy.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    cc_major = cupy.cuda.runtime.getDeviceProperties(0)["major"]
    cc_minor = cupy.cuda.runtime.getDeviceProperties(0)["minor"]
    print(f"compute capability: sm_{cc_major}{cc_minor}")
    print(f"smoke test: N={n}, BLOCK={BLOCK}x{BLOCK}")

    rng = cupy.random.default_rng(0xC0FFEE)
    a = rng.random((n, n), dtype=cupy.float32)
    b = rng.random((n, n), dtype=cupy.float32)
    c = cupy.zeros((n, n), dtype=cupy.float32)

    kernel = make_kernel(n)
    grid = (n // BLOCK, n // BLOCK)

    # Launch
    try:
        ct.launch(stream.ptr, grid, kernel, (a, b, c))
        cupy.cuda.runtime.deviceSynchronize()
    except Exception as e:
        print(f"FAIL: kernel launch raised: {e}", file=sys.stderr)
        return 1

    expected = cupy.matmul(a, b)
    # Relative error check: |got - expected| / max(|expected|, 1e-6) < 1e-3
    diff = cupy.abs(c - expected)
    denom = cupy.maximum(cupy.abs(expected), cupy.float32(1e-6))
    rel_err = diff / denom
    max_rel = float(cupy.max(rel_err))
    mean_rel = float(cupy.mean(rel_err))
    max_abs = float(cupy.max(diff))
    print(f"max_rel_err={max_rel:.3e}  mean_rel_err={mean_rel:.3e}  max_abs_err={max_abs:.3e}")

    if max_rel < 1e-3:
        print("OK")
        return 0
    else:
        # Also print a sample to help debug.
        print("FAIL", file=sys.stderr)
        print(f"  sample c[0,0]={float(c[0,0])}  expected[0,0]={float(expected[0,0])}")
        print(f"  sample c[{n//2},{n//2}]={float(c[n//2,n//2])}  expected[{n//2},{n//2}]={float(expected[n//2,n//2])}")
        return 1


# ──── Bench driver (NOT run in smoke mode; orchestrator runs it later) ────
def bench(csv_path: str) -> int:
    stream = cupy.cuda.get_current_stream()

    print(f"cuda-tile version: {ct.__version__}")
    print(f"cupy: {cupy.__version__}")
    print(f"device: {cupy.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    cc_major = cupy.cuda.runtime.getDeviceProperties(0)["major"]
    cc_minor = cupy.cuda.runtime.getDeviceProperties(0)["minor"]
    print(f"compute capability: sm_{cc_major}{cc_minor}")
    print(f"sizes: {SIZES}")
    print(f"warmup: {WARMUP}, iters: {ITERS}")
    print()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["impl", "kernel", "n", "iter", "gpu_ms", "tflops"])

        for n in SIZES:
            print(f"---- N = {n} ----")
            rng = cupy.random.default_rng(0xC0FFEE)
            a = rng.random((n, n), dtype=cupy.float32)
            b = rng.random((n, n), dtype=cupy.float32)
            c = cupy.empty((n, n), dtype=cupy.float32)

            kernel = make_kernel(n)
            grid = (n // BLOCK, n // BLOCK)

            # Warmup (also triggers JIT compile of this specialization)
            for _ in range(WARMUP):
                ct.launch(stream.ptr, grid, kernel, (a, b, c))
            cupy.cuda.runtime.deviceSynchronize()

            # Correctness check
            expected = cupy.matmul(a, b)
            rel = float(
                cupy.max(
                    cupy.abs(c - expected)
                    / cupy.maximum(cupy.abs(expected), cupy.float32(1e-6))
                )
            )
            if rel > 1e-2:  # looser at large N; matmul is FP-noisy
                print(f"[cutile-naive] N={n} CORRECTNESS FAIL rel={rel:.3e}", file=sys.stderr)
                return 1

            total_flops = 2.0 * (n ** 3)
            starts = [cupy.cuda.Event() for _ in range(ITERS)]
            ends = [cupy.cuda.Event() for _ in range(ITERS)]
            for i in range(ITERS):
                starts[i].record(stream)
                ct.launch(stream.ptr, grid, kernel, (a, b, c))
                ends[i].record(stream)
                stream.synchronize()

            gpu_times_ms = [
                cupy.cuda.get_elapsed_time(starts[i], ends[i]) for i in range(ITERS)
            ]
            for i, gpu_ms in enumerate(gpu_times_ms):
                tflops = (total_flops / 1e12) / (gpu_ms / 1000.0)
                print(f"[cutile-naive] N={n} iter={i} gpu_ms={gpu_ms:.4f} TFLOPS={tflops:.3f}")
                writer.writerow(["cutile", "naive", n, i, f"{gpu_ms:.6f}", f"{tflops:.6f}"])

            best = min(gpu_times_ms)
            med = statistics.median(gpu_times_ms)
            med_tf = (total_flops / 1e12) / (med / 1000.0)
            print(
                f"[cutile-naive] N={n} correctness OK  best={best:.3f}ms med={med:.3f}ms "
                f"({med_tf:.3f} TFLOPS median)"
            )

            del a, b, c, expected
            cupy.get_default_memory_pool().free_all_blocks()

    print(f"\nResults written to {csv_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", default=True,
                        help="Run correctness smoke test only (default).")
    parser.add_argument("--bench", action="store_true", default=False,
                        help="Run the full benchmark sweep (NOT used in smoke mode).")
    parser.add_argument("--smoke-n", type=int, default=SMOKE_N,
                        help=f"N for the smoke test (default {SMOKE_N}).")
    parser.add_argument("--csv-out", default="results.csv",
                        help="CSV path for bench results.")
    args = parser.parse_args()

    if args.bench:
        return bench(args.csv_out)
    return smoke(args.smoke_n)


if __name__ == "__main__":
    sys.exit(main())
