# ADR 0001: Standardize timing on `cudaEvent` for all CUDA backends

**Status:** accepted (2026-05-08, decided in deep-work-loop run 2)

**Context.** v0 used `Instant::now()` + `stream.synchronize()` for cuda-oxide and `cudaEventRecord` for nvcc. Wall-clock includes ~5-50µs of host-side stream synchronize overhead per iteration; for fast kernels this distorts the comparison. Even for ~25ms kernels, it adds noise that masks small (~5%) compiler-quality differences.

**Decision.** All CUDA-backend benches in this repo (cuda-oxide, nvcc, cuBLAS) report two distinct timings:

1. **`gpu_ms`** — measured via `cudaEventRecord` / `cudaEventSynchronize` at the kernel boundary. This is the headline number used in result tables.
2. **`cpu_wall_ms`** — measured via `std::time::Instant` around `stream.synchronize()`. Reported in the log for debug visibility but not in the headline table.

For wgpu, use `TIMESTAMP_QUERY` where available (already in v0), else CPU wall-clock with a clear warning in the log.

**Why event-based.** GPU timestamp deltas are nanosecond-resolution, measured by the GPU itself, and exclude host-side launch + synchronize overhead. They give a clean kernel-only number. The host-side wall-clock difference is the kernel time + everything else; useful for whole-pipeline benchmarks but not for compiler comparisons.

**Implementation.** cuda-oxide doesn't expose `cudaEventRecord` directly — its safe API is `stream.synchronize()`-based. We need to call into `cuda-bindings` (the raw FFI crate) for the event APIs. This is ugly but unavoidable until cuda-oxide adds a higher-level event API.

**Alternatives considered.**

- *Keep wall-clock, run more iters.* Variance averages out but the systematic bias (sync overhead per iter) does not.
- *Use `cuStreamPollEvent` polling.* Same data as event-record but more code.
- *Use a CUDA Graph capture.* Overkill for single-launch kernels.

**Consequences.**

- All v1 results are slightly faster than v0 results for the same kernel (sync overhead removed). Document the methodology change in `SUMMARY.md`.
- Comparison fairness improves: cuda-oxide vs nvcc gap narrows by whatever fraction of v0 wall-clock was sync overhead. Current ~7% gap on unchecked may shrink to 3-5% or stay where it is — that's the experiment.
- Reproducing v0 numbers requires reverting to wall-clock; document this in METHODOLOGY.md.

**Acceptance test.** Each updated bench logs both `gpu_ms=<x>` and `cpu_wall_ms=<y>` per iter; result tables in ANALYSIS.md show `gpu_ms` median + best.
