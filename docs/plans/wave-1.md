# Wave 1: cudaEvent timing + size scaling

**Status:** ready to execute
**ADR refs:** 0001 (cudaEvent timing), 0002 (sm_120 native), 0003 (scope)
**Subagents:** 3 in parallel
**Budget:** ~30 min wall-clock, ~80k summary tokens to orchestrator
**Acceptance test:** all three subagent artifacts exist; `results/scaling.csv` has rows for all three backends × N ∈ {1024, 2048, 4096}; commits made; push successful.

## File-ownership table (concurrency safety)

| Subagent | Owns (writes) | Reads (no-write) |
|---|---|---|
| W1A — oxide bench update | `oxide-matmul/src/main.rs`, `oxide-matmul/run.log`, `oxide-matmul/results.csv`, `oxide-matmul/oxide_matmul.ptx` (regenerated) | docs/research/*, docs/adrs/*, system-info/* |
| W1B — nvcc bench update | `cuda-matmul/matmul.cu`, `cuda-matmul/matmul`, `cuda-matmul/matmul.ptx`, `cuda-matmul/run.log`, `cuda-matmul/results.csv` | docs/* |
| W1C — results aggregator (runs AFTER W1A & W1B) | `results/scaling.csv`, `results/scaling-summary.md` | both other subagents' .csv |

W1A and W1B run in parallel; W1C runs after both finish.

## W1A — cuda-oxide: cudaEvent timing + size sweep

**Goal:** Update `oxide-matmul/src/main.rs` to (a) report kernel-only timing via `cudaEventRecord` (ADR-0001) and (b) sweep N ∈ {1024, 2048, 4096}. Output `oxide-matmul/results.csv` with columns `impl,kernel,N,iter,gpu_ms,cpu_wall_ms,tflops`.

**Implementation hints:**
- cuda-oxide's safe API has `CudaStream::synchronize()` but no high-level event wrapper. Reach into `cuda-bindings` (raw FFI, already a dep transitively). Call `cudaEventCreate`, `cudaEventRecord(ev, stream.raw_stream())`, `cudaEventSynchronize`, `cudaEventElapsedTime`.
- Verify that cuda-oxide's `CudaStream` exposes a raw stream pointer (check `crates/cuda-core/src/stream.rs`). If not, use `cuStreamCreate` from raw bindings and pass it everywhere — but that requires more rework. Simplest first: use the default stream's raw handle if accessible; else fall back to `Instant`-based + a clear warning that 0001 wasn't fully applied for cuda-oxide.
- Keep both `gpu_ms` (event) and `cpu_wall_ms` (Instant) per iter. Print both. CSV column should be `gpu_ms`.
- Sizes loop: outer `for n in [1024, 2048, 4096]`, inner 1 warmup + 10 timed iters per kernel (safe, unchecked).
- Don't allocate new buffers per N if you can avoid it — allocate the largest (4096²) once, sub-slice for smaller.
- Correctness check at each N at indices (0,0), (n/2, n/2), (n-1, n-1).
- Final summary table to stdout AND to results.csv.

**Run:** `cd oxide-matmul && cargo oxide run oxide-matmul 2>&1 | tee run.log`

**Acceptance:** results.csv has 3 N × 2 kernels × 10 iters = 60 rows + header; run.log shows pass; both old kernels still produce 3/3 correct spot-check.

## W1B — nvcc CUDA C++: native sm_120 + size sweep

**Goal:** Rebuild `cuda-matmul` against native `-arch=sm_120` (ADR-0002). Add size sweep. Use `cudaEventRecord` (already in v0).

**Implementation hints:**
- Replace any hardcoded `nvcc` invocation with `/usr/local/cuda/bin/nvcc -ccbin clang-14 -O3 -arch=sm_120`.
- Add `argc` parsing for optional `--size N` arg, default 4096. Or just loop `for (int N : {1024, 2048, 4096})` inside main and run the suite for each.
- Same input pattern as oxide: `(i % 7) * 0.01f`, `(i % 11) * 0.01f`.
- 1 warmup + 10 timed iters per N (was 5; bump to match W1A).
- Output `cuda-matmul/results.csv` with `impl,kernel,N,iter,gpu_ms,tflops`.

**Acceptance:** matmul binary built, runs, results.csv populated, run.log shows native sm_120 build. PTX dump reflects the new arch.

## W1C — Results aggregator (sequential, after W1A & W1B)

**Goal:** Combine the per-folder CSVs into `results/scaling.csv` and emit `results/scaling-summary.md` with a table per N showing best/median/TFLOPS for each backend.

**Implementation:** Pure data work, can be done in `execute_code` Python or a small bash script. No new code.

**Acceptance:** scaling.csv has all rows from both backends; scaling-summary.md has 3 tables (one per N) with backends as rows and (best_ms, median_ms, TFLOPS, ratio_to_nvcc) as columns.

## Wave 1 Concurrent reviewers

After W1A & W1B finish (before W1C), dispatch 1 cross-family reviewer to confirm:
- gpu_ms and cpu_wall_ms are both populated and reasonable (gpu_ms ≤ cpu_wall_ms for every iter)
- run.log timestamps are recent (today)
- size sweep loop didn't accidentally OOM
- new oxide_matmul.ptx still has the same instruction shape (no regression)

Reviewer prompt is concise: 250-line max output, single REJECT/AMEND/PASS verdict + justification.

## Reflexion checklist (post-wave)

- [ ] Both W1A and W1B committed independently (one commit per subagent)
- [ ] AGENTS.md updated with one-liner: "ALWAYS use /usr/local/cuda/bin/nvcc; system /usr/bin/nvcc is a stale 12.0 shim."
- [ ] BACKLOG.md updated: M1, M2 → ✅; ADR-0002 reflected in any prereq docs
- [ ] Push to origin master after both subagents
