# ADR 0003: Comparison scope — what we benchmark and what we don't

**Status:** accepted (2026-05-08)

**Context.** As the bench grows beyond the v0 4096×4096 naive matmul, scope creep is the biggest risk. With infinite time we'd tile, retune blocks, swap precisions, multi-GPU, and write a paper. With finite time we pick what answers the central question (*how does cuda-oxide compare to CUDA C++ today?*) and skip everything else.

**Decision.** Scope is fixed to the following axes:

### IN SCOPE

| Axis | Values | Rationale |
|---|---|---|
| Algorithm | Naive matmul (1 thread = 1 output element) | Memory-bound, well-understood, fair across compilers |
| Algorithm | Tiled shared-memory matmul (16×16 tile) | Compute-bound, exercises `SharedArray`, real comparison axis |
| Backend | nvcc CUDA C++ | Reference baseline (oracle) |
| Backend | cuda-oxide v0.1.0 (safe + unchecked) | Subject of evaluation |
| Backend | cuBLAS sgemm | Real-world SoL upper bound |
| Backend | wgpu/WGSL (CPU fallback only on WSL) | Cross-vendor stack reference; documented limitation |
| Precision | f32 | Cleanest cross-stack comparison; both Tensor Cores and shader cores support it |
| Sizes | N ∈ {1024, 2048, 4096} | 8192 omitted (WSL2 WDDM TDR risk per Phase 3 research) |
| Block size | 16×16 only | Per Phase 3 research, retuning is a separate study |
| Iterations | 1 warmup + 5-10 timed; report median + best | |

### OUT OF SCOPE

| Axis | Excluded values | Why |
|---|---|---|
| Algorithm | Tensor Core / WGMMA matmul | Requires `cuda-oxide`'s wgmma API; ~50% of our remaining budget for marginal cross-comparison value vs cuBLAS |
| Algorithm | Reductions, scans, FFT | Different kernel shapes; matmul tells us 90% of what we want to know about the compiler |
| Backend | rust-cuda (legacy) | Different design (Rust-on-GPU not CUDA-in-Rust); separate evaluation |
| Backend | OpenCL, SYCL, HIP | Different ecosystem; cuda-oxide is NVIDIA-only by design |
| Precision | f16, bf16, fp64 | Single precision is the cleanest cross-compiler test |
| GPU | Multi-GPU / distributed | Single 5090 is enough to surface compiler issues |
| Hardware | Other NVIDIA GPUs | We have one Blackwell; document that and move on |
| Tooling | `nsys` / `ncu` profiling | Useful but a 10x time investment; ANALYSIS.md notes it as next-step |
| Tooling | Occupancy tuning, register pressure tuning | Per Phase 3 research, not the bottleneck for naive matmul |
| Tooling | Self-hosted GitHub Actions runner | Out of scope; nice-to-have |
| Methodology | Multiple runs across reboots / thermal cycling | Single run with 5-10 iters captured the variance |

### Negotiable (could move into IN SCOPE if a wave finishes early)

- Adding N=8192 with a per-iter timeout guard (avoid TDR)
- Reduction kernel as a second algorithm class
- Multiple block sizes (8×8, 32×8) for the tiled variant only

**Decision rule for scope creep.** During waves, if a subagent finds an interesting tangent (e.g., "we should also try fp16"), they mark it in `docs/followups.md` and continue with the in-scope task. Phase 8 review reads `followups.md` and surfaces the highest-value items to the user as suggested next steps. **No silent expansion.**

**Consequences.**

- Some readers will want fp16 / Tensor Core numbers; we point them at `followups.md` and the cuBLAS comparison.
- The repo's central claim ("here's how cuda-oxide compares to CUDA C++ on f32 naive + tiled matmul") is bounded and defensible.
- Any future addition is opt-in via a separate folder (e.g., `oxide-tensor-cores/`) so the existing scope stays clean.
