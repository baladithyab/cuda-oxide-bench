# Wave 4: reduction, bandwidth, causal isolation

**Status:** ready to execute
**Subagents:** 3 in parallel, file-disjoint
**Budget:** ~30 min wall-clock
**Acceptance:** three new test folders or experiments, results aggregated into `results/scaling.csv` if applicable, ANALYSIS.md per folder.

## File-ownership

| Subagent | Owns | Reads |
|---|---|---|
| W4A reduction | `oxide-reduction/`, `cuda-reduction/` (new) | sharedmem + warp_reduce examples in cuda-oxide source |
| W4B bandwidth | `oxide-vecadd-bench/`, `cuda-vecadd-bench/` (new) | existing oxide-vecadd as template |
| W4C causal isolation | `docs/experiments/libnvvm-causal-isolation.md` + `oxide-matmul/run.log.compute89-newvm` | rebuilds oxide-matmul with arch override |

## W4A — Reduction kernel

Sum-reduction: input N×f32, output 1×f32. Exercises cuda-oxide's `warp::*` shuffle primitives + shared-memory atomics if needed. The two-stage reduce-per-block pattern is the canonical benchmark.

Two implementations:
- **`cuda-reduction/`** — nvcc CUDA C++ canonical reduction (warp shuffle + per-block partial → host or atomic to device-side single-cell)
- **`oxide-reduction/`** — same algorithm in cuda-oxide. Use `warp::shuffle_xor`, `warp::all_reduce`, or whatever cuda-oxide's API actually exposes (check `crates/cuda-device/src/warp.rs`).

Sizes: N ∈ {1M, 16M, 256M} elements (= 4 MB, 64 MB, 1 GB).

Output: `*-reduction/results.csv` with `impl,kernel,N_elems,iter,gpu_ms,GB_per_s`.

Acceptance: both produce numerically-identical sum (relative error < 1e-3), GB/s in the 100-1500 range (RTX 5090 has ~1.79 TB/s peak DRAM).

## W4B — Memory bandwidth bench (vec-add scaling)

Pure memory-bound `c[i] = a[i] + b[i]` at varying N. Tells us the achievable HBM bandwidth ceiling.

Two implementations:
- **`cuda-vecadd-bench/`** — nvcc, identical pattern to existing oxide-vecadd but parameterized N
- **`oxide-vecadd-bench/`** — cuda-oxide same

Sizes: N ∈ {1M, 16M, 64M, 256M} f32 = {4MB, 64MB, 256MB, 1GB} per buffer (3 buffers). 1GB stays under 32GB VRAM.

Output: `*-vecadd-bench/results.csv` with `impl,N_elems,iter,gpu_ms,GB_per_s` (3 buffers × N × 4B / time).

Acceptance: oxide and nvcc within 5% of each other, both within 5% of cublas's effective sgemm bandwidth ceiling (sanity check on roofline — if they don't, our vec-add kernel is launch-overhead-bound or we have a bug).

## W4C — libNVVM causal isolation

Reviewer #2's request: when we fixed the libNVVM shadow bug we changed *both* (a) libNVVM version (7.0.1 → 22.0.0) and (b) target arch (sm_89 PTX-JIT → sm_120 native). Can't isolate which.

Experiment: force `-arch=compute_89` (older arch) with the modern libNVVM (CUDA_HOME=/usr/local/cuda set). If the safe-vs-unchecked gap stays at ~1.0×, then libNVVM version is what mattered. If it widens back toward 2.5×, the arch is what matters.

Cuda-oxide's arch is set via `CUDA_OXIDE_ARCH=sm_89` env (per `cuda-host/src/ltoir.rs` source). Set it, rebuild, rerun, compare.

Output: `docs/experiments/libnvvm-causal-isolation.md` with hypothesis, method, results, conclusion. ≤200 words.

Acceptance: doc exists with at least the four bench numbers (safe/unchecked × old/new arch) and a verdict.

## Concurrent review (1 reviewer, low-risk batch)

After all three subagents finish, single cross-family reviewer reads all three deliverables, verifies the GB/s numbers are in plausible range, the reduction sums match, the causal experiment was actually run.

## Reflexion checklist

- [ ] All new folders or docs committed
- [ ] AGENTS.md not updated (lessons-learned phase deferred to Wave 6)
- [ ] BACKLOG.md updated: N2 (reduction) → ✅; new items added if found
- [ ] Push after Wave 4 reviewer signoff
