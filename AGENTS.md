# cuda-oxide-bench — agent-of-record notes

Lessons from running the deep work loop on this repo. These are operational
patterns specific to this codebase; general agent guidance lives in skills.

## Wave 1 (2026-05-08)

- **Always use `/usr/local/cuda/bin/nvcc` (13.2), never the `/usr/bin/nvcc`
  shim (12.0). The system shim doesn't recognize `-arch=sm_120` and silently
  falls back. Confirmed via `version.json` showing CUDA 13.2.20260407 vs
  shim's `release 12.0, V12.0.140`.

- **Build for native `sm_120` on this RTX 5090 box.** Going from `sm_89`
  PTX-JIT to `sm_120` native gave a ~6% speedup at N=4096 (20.6 → 19.4 ms
  best). Smaller than expected; for naive matmul without Tensor Core ops
  the Blackwell-specific instructions don't apply. But it's the correct
  baseline.

- **`cudaEvent` timing for cuda-oxide works** via `CudaContext::new_event` +
  `CudaEvent::record(&stream)` + `elapsed_ms`. The ctx-bound stream is
  required (`stream.context().bind_to_thread()` if calling raw driver
  bindings). cpu_wall_ms is consistently ~0.5-1ms above gpu_ms — small but
  not zero.

- **At N=1024, cuda-oxide unchecked = 95% of nvcc.** At N=4096 it's 80%.
  The compiler gap is size-dependent — most likely the FMA-omission cost
  is amortized over small inner loops but compounds for large ones.

- **High variance at N=4096 in oxide-unchecked.** Iters 7-9 jumped from
  ~25ms to ~33ms. Thermal contention with desktop apps probably; could
  also be GPU power-state transitions. Mitigation in future runs: lock
  the GPU clock with `nvidia-smi -lgc <max>` before benching, or ignore
  outliers via IQR.

## Phase 3 research nuggets

- **cuda-oxide has FastmathFlags plumbing but always uses `::default()`
  (= empty).** Two-line patch upstream would unlock FMA contraction. See
  `docs/research/cuda-oxide-flags.md`.

- **`fmuladd` lowers to libdevice `__nv_fmaf` instead of `llvm.fmuladd`.**
  Even explicit `core::intrinsics::fmuladdf32` won't hit hardware FMA
  without a deeper fix.

- **N=8192 naive matmul will run ~180s on this hardware and trip WSL2
  WDDM TDR.** Bench harness must clamp at N=4096. Per
  `docs/research/occupancy-and-scaling.md`.

## Subagent discipline

- **Subagent timeouts at 600s usually mean it stalled on a large `write_file`
  after building extensive context.** When this happens, the orchestrator
  has enough context from the task prompt + reference reads to finish the
  artifact inline. We did this for Wave 1 W1A — subagent's source code was
  correct on inspection, just needed orchestrator to run the bench.

- **One commit per coherent unit of work, not per subagent.** Wave 1 has
  three commits: W1B (cuda-matmul), W1A+W1C (oxide-matmul + aggregator).
  Commit messages cite the wave + which ADRs apply.

- **Per-folder file ownership prevents merge issues.** Wave 1's two parallel
  subagents touched disjoint folders (oxide-matmul vs cuda-matmul); no
  conflict possible.

## Variance and noise

- **Run iters: 10 timed + 1 warmup is usually sufficient.** Median is robust;
  reporting best for lower-bound + p95 for upper-bound + IQR for variance
  width is the cleanest way to present.

- **gpu_ms vs cpu_wall_ms:** 0.3-1ms differential. Trustworthy as a
  measure of host-side stream-sync overhead. Larger differentials
  (>5ms) indicate sync bug — investigate.
