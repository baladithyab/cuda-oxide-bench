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

## Wave 3 (2026-05-08, late)

- **The libNVVM shadow bug.** cuda-oxide's `libnvvm-sys` tries `libnvvm.so.4`
  against the system loader before falling back to `CUDA_HOME/nvvm/lib64`.
  On systems with a stale apt-installed CUDA at `/usr/lib/x86_64-linux-gnu/libnvvm.so.4`,
  this loads libNVVM 7.0.1 (capped at compute_90), masking the modern libNVVM
  22.0.0 from CUDA 13.2. Symptom: `libnvvm: -arch=compute_120 is an unsupported option`,
  OR silently bad codegen for slice-bounds-check predicates. Fix: always export
  `CUDA_HOME=/usr/local/cuda` AND `LIBNVVM_PATH=/usr/local/cuda/nvvm/lib64/libnvvm.so`
  before any `cargo oxide` invocation. See `docs/experiments/libnvvm-corrigendum.md`.

- **`core::intrinsics::fmuladdf32` is a working FMA escape hatch.** Lowers to
  libdevice `__nv_fmaf` which itself contains `fma.rn.f32` — nvJitLink resolves
  at module load, final SASS has hardware FMA. Verified in
  `docs/experiments/fma-toggle.md`. So the upstream FMA-contraction issue is
  about default `*+` codegen, not about whether any FMA is reachable.

- **Phase-8 reviewers caught real overclaim risks.** The "99% of nvcc" headline
  was true at N=1024 but misleading at larger sizes; the libNVVM causal claim
  conflated two variables (libNVVM version + target arch); the tiling-gap "missing
  FMA" framing understated the role of register-microtile + unrolling. All three
  hedged in the final docs. Lesson: **Phase 8's intersection (3+ reviewers
  flagging the same issue) is a much stronger signal than any single reviewer's
  union list.** The intersection here was tiny but high-value.

- **Cross-family routing didn't take effect** in our `delegate_task` calls.
  All three reviewers ran on the orchestrator's primary model. Reviews still
  produced orthogonal findings (different file-reading paths, different angles)
  but the model-diversity guarantee wasn't there. Open question: was the model
  override silently ignored, or was it routed but logs not surfaced? Worth
  investigating before relying on cross-family review for high-stakes work.

## Methodology lessons

- **Always run with output captured** (`tee run.log`). Wave 1 W1A's "successful"
  log was actually a crash log because we ran the binary inline without `tee`.
  Phase 8 reviewer #1 caught this only because they read the file directly.

- **Lock GPU clocks if you can** (`nvidia-smi -lgc`). Without root permission
  we can't do this in this environment; CV at N=4096 is 5-15% as a result.
  Future runs on a hosted self-locked machine should re-bench.

- **Save the broken-state artifacts**, not just the fixed-state ones, when
  diagnosing a bug. We don't have broken-era PTX preserved, so we can't isolate
  whether the libNVVM perf delta was due to compute_120 vs compute_89 codegen
  quality or something else. A 60-second experiment (force old libnvvm + same arch)
  would settle it.

- **Numerical drift between aggregator and per-folder CSVs.** Wave 1 W1C used
  the per-folder CSVs as input; later waves regenerated against `results/scaling.csv`.
  The aggregator must be the single source of truth — and headline tables
  must be regenerated from that source after any rerun. We had ~3% drift between
  README claims and CSV medians until Phase 8 caught it.
