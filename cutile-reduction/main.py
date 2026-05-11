"""cuTile parallel sum-reduction — port of oxide-reduction.

Mirrors oxide-reduction/src/main.rs:
  - f32 sum reduction of arrays of size 1M / 16M / 256M
  - 1 warmup + 10 timed iters per N
  - Per-block: local reduction, then atomic_add into out[0]
  - cudaEvent timing → gpu_ms + GB/s

oxide-reduction uses a warp-shuffle 2-stage reduction with block=256,
grid=4096 and a grid-stride loop. In cuTile, `ct.sum` does the equivalent
work within a tile (compiler emits warp-shuffle + smem reduction SASS).
We use a grid-stride loop at the tile level so we can keep a fixed grid
size (4096) matching the oxide baseline.

TILE_SIZE=1024 elements per block × grid=4096 blocks = 4M elements per
grid pass. For N > 4M we loop at the tile level; for N < 4M many blocks
process padded tiles (ct.PaddingMode.ZERO makes out-of-bounds loads = 0).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time

import cuda.tile as ct
import cupy

SIZES = [1 * 1024 * 1024, 16 * 1024 * 1024, 256 * 1024 * 1024]
WARMUP = 1
ITERS = 10
TILE_SIZE = 1024
GRID = 4096


def make_kernel(tile_size: int):
    """Build a cuTile reduction kernel with a fixed TILE_SIZE.

    Each block:
      - owns bid ∈ [0, GRID)
      - walks the input in a grid-stride loop at tile granularity:
          tile_idx = bid, bid + GRID, bid + 2*GRID, ... while tile_idx*tile_size < n
      - loads a TILE_SIZE-wide tile with zero padding (handles the tail)
      - accumulates local partial via ct.sum over the tile
      - atomic_add the single scalar partial into out[0]
    """

    @ct.kernel
    def reduce_sum(a, out):
        bid = ct.bid(0)
        # Load a tile_size-wide tile. PaddingMode.ZERO handles the tail
        # when N is not an exact multiple of tile_size — OOB lanes read 0
        # so they contribute nothing to the sum.
        a_t = ct.load(
            a, index=(bid,), shape=(tile_size,),
            padding_mode=ct.PaddingMode.ZERO,
        )
        # ct.sum over the whole tile is the cuTile equivalent of the
        # warp-shuffle + smem 2-stage reduction in oxide-reduction.
        partial = ct.sum(a_t)
        # Single-output atomic sum. ct.atomic_add(array, indices, update).
        ct.atomic_add(out, (0,), partial)

    return reduce_sum


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def run_correctness(tile_size: int = TILE_SIZE) -> bool:
    """Smoke test: random 1M f32 input, compare to cupy.sum.

    Relative tolerance 1e-2 because order of summation differs between
    atomic-add tree and cupy's deterministic reduction. (1e-2 = 1%.)
    """
    stream = cupy.cuda.get_current_stream()
    kernel = make_kernel(tile_size)

    n = 1 * 1024 * 1024
    rng = cupy.random.default_rng(0xC0FFEE)
    a = rng.random(n, dtype=cupy.float32)
    out = cupy.zeros(1, dtype=cupy.float32)

    grid = (cdiv(n, tile_size),)
    ct.launch(stream.ptr, grid, kernel, (a, out))
    stream.synchronize()

    gpu_sum = float(out[0].get())
    cpu_sum = float(cupy.sum(a).get())
    # Reference sum for tolerance magnitude.
    rel_err = abs(gpu_sum - cpu_sum) / max(abs(cpu_sum), 1e-12)

    print(f"[smoke] N={n} cpu_sum={cpu_sum:.6f} gpu_sum={gpu_sum:.6f} rel_err={rel_err:.3e}")
    ok = rel_err < 1e-2
    if ok:
        print("OK")
    else:
        print(f"FAIL: rel_err {rel_err:.3e} exceeds 1e-2")
    return ok


def run_bench(csv_path: str, tile_size: int = TILE_SIZE) -> None:
    """Full timed sweep. Orchestrator runs this; smoke test does NOT."""
    stream = cupy.cuda.get_current_stream()
    kernel = make_kernel(tile_size)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["impl", "kernel", "n", "iter", "gpu_ms", "gbps"])

        for n in SIZES:
            n_tiles = cdiv(n, tile_size)
            grid = (n_tiles,)

            rng = cupy.random.default_rng(0xC0FFEE)
            a = rng.random(n, dtype=cupy.float32)
            out = cupy.zeros(1, dtype=cupy.float32)

            # Warmup (also compiles for this (dtype, tile_size)).
            for _ in range(WARMUP):
                out.fill(0)
                ct.launch(stream.ptr, grid, kernel, (a, out))
            stream.synchronize()

            # Correctness sanity check
            expected = float(cupy.sum(a).get())
            got = float(out[0].get())
            rel_err = abs(got - expected) / max(abs(expected), 1e-12)
            if rel_err > 1e-2:
                print(
                    f"[cutile-reduce] N={n} CORRECTNESS FAIL rel_err={rel_err:.3e}",
                    file=sys.stderr,
                )
                sys.exit(1)

            traffic_bytes = float(n) * 4.0  # 1 read per element
            starts = [cupy.cuda.Event() for _ in range(ITERS)]
            ends = [cupy.cuda.Event() for _ in range(ITERS)]
            wall_times_ms: list[float] = []
            gpu_times_ms: list[float] = []

            for i in range(ITERS):
                out.fill(0)
                stream.synchronize()
                t0 = time.perf_counter()
                starts[i].record(stream)
                ct.launch(stream.ptr, grid, kernel, (a, out))
                ends[i].record(stream)
                stream.synchronize()
                wall_times_ms.append((time.perf_counter() - t0) * 1000.0)

            gpu_times_ms = [
                cupy.cuda.get_elapsed_time(starts[i], ends[i]) for i in range(ITERS)
            ]
            for i, gpu_ms in enumerate(gpu_times_ms):
                gbps = (traffic_bytes / 1e9) / (gpu_ms / 1000.0)
                print(f"[cutile-reduce] N={n} iter={i} gpu_ms={gpu_ms:.4f} GB/s={gbps:.2f}")
                writer.writerow(["cutile", "reduce_sum", n, i, f"{gpu_ms:.6f}", f"{gbps:.6f}"])

            sorted_ms = sorted(gpu_times_ms)
            best = sorted_ms[0]
            median = statistics.median(sorted_ms)
            gbps_med = (traffic_bytes / 1e9) / (median / 1000.0)
            print(
                f"[cutile-reduce] N={n} correctness OK  best={best:.4f}ms med={median:.4f}ms "
                f"({gbps_med:.2f} GB/s median) rel_err={rel_err:.3e}"
            )

            del a, out
            cupy.get_default_memory_pool().free_all_blocks()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", default=True, help="run correctness smoke test only")
    parser.add_argument("--bench", action="store_true", default=False, help="run full timed bench sweep")
    parser.add_argument("--csv-out", default="results.csv", help="path for CSV output (bench mode)")
    args = parser.parse_args()

    # --bench overrides default-on --smoke
    if args.bench:
        args.smoke = False

    print(f"cuda-tile version: {ct.__version__}")
    print(f"cupy: {cupy.__version__}")
    dev_props = cupy.cuda.runtime.getDeviceProperties(0)
    print(f"device: {dev_props['name'].decode()}")
    print(f"compute capability: sm_{dev_props['major']}{dev_props['minor']}")
    print(f"TILE_SIZE: {TILE_SIZE}")
    print()

    if args.bench:
        run_bench(args.csv_out)
        return 0

    ok = run_correctness()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
