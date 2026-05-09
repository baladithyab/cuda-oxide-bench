# Methodology

How the numbers in [README.md](README.md) were measured, and what they do and do not mean.

## 1. Workload

Every backend runs the same kernel: a **naive 4096×4096 f32 matrix multiply**, `C = A × B`, with a 16×16 thread block (256 threads per block, 65 536 blocks). Each iteration performs

`2 × 4096³ = 137 438 953 472 ≈ 137.44 GFLOP`

using the textbook inner loop:

```
for k in 0..dim:
    C[r, c] += A[r, k] * B[k, c]
```

No shared-memory tiling. No Tensor Cores (`mma.sync`, `wgmma`). No cuBLAS. No vectorised loads (`float4`). The algorithm is deliberately simple so that the delta between backends reflects the **compiler and driver**, not the programmer's cleverness.

## 2. Why naive matmul

Naive matmul is nearly the ideal micro-benchmark for a compiler-vs-compiler study:

- It has real compute (`4096³` FMAs) and real memory access (two `O(N²)` operands, one `O(N²)` output, total ~200 MiB).
- It scales `O(N³)`, so at N=4096 the kernel runs for tens of milliseconds — well past launch overhead, well within the 32 GiB of VRAM.
- It has no data-dependent control flow, no reductions, no atomics — so any perf delta is due to instruction selection (FMA contraction, read-only cache hints, bounds-check elision), not algorithmic accident.

In short: the math is fixed, the blocking is fixed, the problem size is fixed. Everything that varies is the toolchain.

## 3. Inputs

Inputs are **bit-identical across all three benchmarks**, generated from a deterministic closed form:

```
a[i] = (i % 7)  * 0.01    // values in [0.00, 0.06]
b[i] = (i % 11) * 0.01    // values in [0.00, 0.10]
```

The ranges are chosen so that partial sums stay bounded and the final product has no `NaN` / `Inf`. No RNG, no seeds, no variance across runs.

## 4. Measurement

Each backend is timed with the most appropriate primitive available to it. Reported values are **median** and **best** of the timed iterations, after a warmup iteration that is discarded.

- **nvcc CUDA C++** — `cudaEventRecord` / `cudaEventElapsedTime` on GPU-side events (nanosecond precision, no CPU/driver jitter). 1 warmup + 5 timed iterations.
- **cuda-oxide** — CPU wall-clock via `std::time::Instant::now()` bracketing `stream.synchronize()`. 1 warmup + 10 timed iterations. Wall-clock is used because the public cuda-oxide v0.1.0 surface does not yet expose CUDA events as first-class types.
- **wgpu / WGSL** — `TIMESTAMP_QUERY` feature when the adapter advertises it; otherwise CPU wall-clock around `queue.submit(...)` + `device.poll(Maintain::Wait)`. 1 warmup + 5 timed iterations.

For each backend we report two statistics:
- **best** = `min(timed_iters)` — an estimate of the compute-bound lower bound when system noise is excluded.
- **median** = `timed_iters.sorted()[n/2]` — a robust point estimate of the typical per-iteration time under the harness's actual overheads.

## 5. Correctness check

Before any timing is reported, each backend's output is verified against a CPU oracle (a plain Rust / C++ triple loop over the same inputs). Three spot indices are checked:

- `C[0, 0]`
- `C[100, 100]`
- `C[4095, 4095]`

Tolerance is **1e-3 relative** — loose enough to accommodate FMA vs non-FMA ordering and any intermediate-precision differences, tight enough to catch genuine bugs. All four kernels pass on every run.

## 6. Why median and best?

Both numbers are useful and they answer different questions:

- **best** approximates what the GPU can do when the OS scheduler, driver, and PCIe bus all cooperate. It is the closest number to a pure compute-bound measurement and the fairest cross-compiler comparison.
- **median** is what a user would actually observe in a loop. It includes the real cost of the host-side harness — stream synchronisation, event recording, Rust's `Instant` granularity, the driver's per-launch bookkeeping — and is robust to a single outlier iteration.

Reporting both makes it obvious when the two agree (system is quiet, numbers are trustworthy) versus when they disagree (outlier, thermal excursion, or noisy host).

## 7. What we did NOT do

These are deliberately out of scope for v0.1 of this study, and are logged as next steps:

- **Profiling** — no `nsys` / `ncu` runs, no occupancy or memory-throughput counters. All root-cause claims are backed by PTX diffs, not by hardware counters.
- **Block-size sweeps** — 16×16 only. 8×8, 32×8, 32×32 would likely shift absolute numbers.
- **Matrix-size sweeps** — 4096×4096 only.
- **Multi-run, multi-boot stability** — each configuration was run once in a single session.
- **Multi-GPU** — single RTX 5090.
- **Release-mode vs debug-mode compiler-flag sweeps** — `cargo oxide run` and `nvcc -O3` defaults only.

## 8. Threats to validity

Things that could move the numbers and that the reader should be aware of:

- **a) Wall-clock overhead for cuda-oxide.** `stream.synchronize()` adds roughly 5–50 µs of host-side overhead per iteration. At ~25 ms per iteration this is **<0.5%** and well below the observed spread between runs. For a kernel in the tens of microseconds it would matter; for this workload it does not.
- **b) PTX JIT cache.** The first launch on a fresh driver session pays JIT cost. All reported numbers are post-warmup, so JIT is amortised.
- **c) WSL2 timer granularity.** `Instant` on WSL2 is backed by a virtual TSC and has sub-microsecond granularity — more than adequate for millisecond-scale measurements.
- **d) Thermals.** Over the full benchmark session `nvidia-smi` reported the RTX 5090 at ~30% fan, 43–50 °C, 80 W of a 575 W TDP. No clock throttling was observed. The GPU is essentially idle relative to its design envelope.
