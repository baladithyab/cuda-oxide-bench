# Wave 18 research — Mojo as a fifth GPU frontend

**Status:** research grounding (not yet implemented)
**Created:** 2026-05-20
**Predecessor:** Wave 17 in flight (KDA + oxide-MLA + GDN-other-frontends)
**Goal:** evaluate adding Modular's **Mojo** language as a fifth frontend column to the cuda-oxide / cuTile / nvcc / wgpu / cuBLAS matrix.

## Why Mojo is interesting for this repo

The current matrix asks *"what is the best non-CUDA-C++ way to write a GPU kernel
on consumer Blackwell today?"* Our four existing answers each have a clear shape:

| frontend | language | tensor cores on sm_120? | strength |
|---|---|---|---|
| nvcc CUDA C++ | C++ | yes (hand-written `mma.sync`) | full HW access, the reference |
| cuda-oxide | Rust → PTX | **no** (no `mma.sync` in v0.1.0; wgmma stubbed; tcgen05 sm_100a-only) | safe Rust, MIR→PTX |
| cuTile | Python DSL | yes (`ct.mma`, ~79% of cuBLAS hgemm) | tile-level, TMA bulk loads |
| wgpu | Rust + WGSL | n/a (broken on WSL2 NVIDIA) | cross-vendor portability |

**Mojo fills a different niche than any of these:**

- **MLIR-based, single-source host+device** like cuda-oxide, but Python-syntax-adjacent like cuTile.
- **Multi-vendor**: NVIDIA + AMD + Apple Silicon from one codebase (the only stack besides wgpu/CubeCL that claims this — and unlike wgpu, with first-class GPU compute, not graphics-shader-shaped).
- **Modular's own "we beat cuBLAS" claim** for B200 matmul ([blog series, Aug 2025](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction)) — directly comparable to our existing cuBLAS hgemm baseline.
- **Distinct compiler stack**: not LLVM-NVPTX (cuda-oxide path) or libNVVM-via-Python (cuTile path). Mojo lowers through MLIR → PTX via Modular's own toolchain (with the option to hand off to a system `ptxas` on older drivers).

So Mojo gives us a **fourth independent compiler axis** for the same algorithm
classes we already benchmark, plus — uniquely — the option to later cross-check
the same Mojo source against AMD MI300X if we ever get hosted access.

## What's actually shipping as of May 2026

### Language & GPU model

Mojo's GPU API is in `from gpu.host import DeviceContext` and
`from gpu import block_idx, thread_idx, block_dim, ...`. Kernels are plain
functions launched with `ctx.enqueue_function[name=...]`. Single-source
file, host code calls into kernel via `ctx.enqueue_function[kernel](...)`
with grid+block dim args. Memory: `DeviceBuffer[DType.float32](N)` →
`buffer.unsafe_ptr()` for kernel signatures.

Pixi-managed install is the recommended path:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
pixi init --format mojoproject myproj
cd myproj
pixi add mojo
pixi run mojo my_kernel.mojo            # JIT
pixi run mojo build my_kernel.mojo      # AOT to a binary
```

Single-binary shells (`magic install mojo` etc.) also exist but pixi is the
official endorsement.

### sm_120 (RTX 5090) status — the critical part

**Headline:** Mojo officially lists RTX 50-series as **"Known compatible"**
under arch target `sm_120` ([requirements page](https://docs.modular.com/mojo/requirements)).
But the picture beneath that label is exactly the same shape as the cuda-oxide
gemm_sol vs consumer-Blackwell story — and we should expect it.

Three concrete known issues on sm_120 as of late Q1 2026:

#### 1. **MAX flagship matmul kernels were sm_100a-only until March 2026**

Tracked in [issue #5707 (Dec 2025)](https://github.com/modular/modular/issues/5707):
running any FP8/FP4-using model (e.g. `Qwen3-8B`) on RTX 5090 fails with

```
constraint failed: The tcgen05 instructions are only applicable on
nVidia Blackwell (sm_100a, sm_101a) hardware.
```

The same hardware-class boundary cuda-oxide hits with `gemm_sol` and
`tcgen05_matmul`. This is **CUDA-C++-equivalent** — tcgen05 is genuinely
unavailable on consumer Blackwell, not a Mojo bug.

[PR #6059 (Mar 4, 2026)](https://github.com/modular/modular/pull/6059) added
host-side dispatch gating: on sm_120, MAX now falls back to "naive matmul
kernels" instead of the tcgen05 path. Brad Larson (Modular) said an
"internally robust" version of this fix would land in the next nightly after
Mar 6. **If we install latest nightly, dispatch should work — but the
matmul will be a naive fallback, not the flagship kernel.**

The Modular blog series' headline numbers (~95% of cuBLAS on B200) are
**all on sm_100a**. They do not transfer to sm_120 today.

#### 2. **Sink-aware flash attention is numerically wrong on sm_120**

[Issue #6198 (Mar 2026, still open)](https://github.com/modular/modular/issues/6198):
`test_flash_attention_sink_kernel` returns `1.0` instead of `~0.3013`.
[PR #6209](https://github.com/modular/modular/pull/6209) in flight to fix.
**Implication:** if we want to bench Mojo attention, we must verify the
specific kernel path works on sm_120 *before* claiming numbers — same
verification discipline as everywhere else in this repo.

#### 3. **Driver and ptxas requirements**

- Minimum NVIDIA driver: 580. Our system has 596.21 ✓.
- Older drivers need `MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas`
  to bypass Mojo's bundled compiler version check. We don't need this, but
  it is a useful diagnostic if Mojo refuses to compile despite the driver
  being recent enough — pointing at our CUDA-13.2 ptxas costs nothing and
  matches our existing toolchain story for nvcc.

### What works on sm_120 and what we should expect to find

Based on the issues above and Mojo's actual GPU surface, our best prior on
the matrix outcome is:

| kernel class | expected status on sm_120 | reason |
|---|---|---|
| vec-add | ✅ works, parity ±1% with nvcc/oxide/cuTile | streaming memory has no SM-class dependency |
| reduction (warp-shuffle) | ✅ works | same |
| reduction with TMA | ❓ depends on whether Mojo lowers to TMA on sm_120 (cuTile does) | needs SASS check |
| naive f32 matmul | ✅ works | scalar FFMA path, no MMA needed |
| mixed-precision matmul | ❓ probably works via Mojo's TC API → `mma.sync.m16n8k16` (4th-gen TC on RTX 5090 supports it; cuTile reaches 172 TF here) | needs HMMA SASS verification |
| MAX-flagship f8/f4 matmul | ❌ tcgen05 = sm_100a-only | hardware boundary |
| flash-attention (sink-aware) | ❌ buggy on sm_120 (issue #6198) | known regression |
| flash-attention (basic) | ❓ unknown | need to test |

So Mojo plausibly slots between cuTile and cuda-oxide on the matrix:
**TC-capable on sm_120 like cuTile (via Mojo's MMA intrinsics), but
single-source compiled like cuda-oxide.**

The most interesting question this answers for the repo's vision: **does
Mojo's MLIR-based codegen produce different SASS than libNVVM (cuTile) or
LLVM-NVPTX (cuda-oxide) for the same algorithm?** That's a clean comparison
of three independent compiler paths emitting code for the same hardware.

## Phased plan (mirrors Wave 12.1 cuTile rollout)

The cuTile axis was added in three phases, lowest-risk first. Same structure
applies for Mojo:

**Phase A — toolchain smoke test** (1-2 hours, high info-per-effort)
- Install Mojo via pixi.
- Verify `from gpu.host import DeviceContext; DeviceContext().name()` reports
  the RTX 5090.
- Hello-world vecadd kernel, 1M elements, max_abs_err = 0 vs CPU.
- Document install-time pitfalls in `mojo-vecadd/SETUP.md` (cuTile pattern).
- Phase A **deliverables**: working pixi env, working binary, SETUP.md.
- Phase A **fail mode**: if Mojo can't compile or run on sm_120, file
  `BLOCKED.md` describing the install state and stop. Don't try to patch
  Modular's compiler.

**Phase B — memory-bound axis**
- `mojo-vecadd-bench/` (mirrors `cutile-vecadd-bench/`, `oxide-vecadd-bench/`):
  N ∈ {1M, 16M, 64M, 256M}, cudaEvent timing via Mojo's `DeviceContext`
  timing API. Tile-size sweep if Mojo exposes one.
- `mojo-reduction/` (mirrors `cutile-reduction`, `cuda-reduction`,
  `oxide-reduction`): 1 GB sum-reduction. **Critical question**: does Mojo
  lower the reduction to TMA bulk loads like cuTile (winning +11% on
  sm_120) or to warp-shuffle like nvcc/oxide (parity)?
- Phase B **deliverable**: a fifth column in the headline memory-bound
  table at the top of the README, with absolute parity expected within 1-3%.
- SASS evidence: `cuobjdump --dump-sass mojo-reduction.cubin | grep
  -c UTMALDG` — ports the cuTile-vs-nvcc methodology directly.

**Phase C — compute-bound axis**
- `mojo-matmul/` (naive f32, mirrors `cuda-matmul`, `oxide-matmul`,
  `cutile-matmul`): scalar FFMA path. Expected near-parity with nvcc.
- `mojo-matmul-tiled/` (mixed-precision, mirrors `cutile-matmul-tiled-mixed`,
  `cublas-half-precision/`): try to engage tensor cores via Mojo's MMA API
  on sm_120. **If HMMA-count > 0 in SASS**, this becomes a meaningful
  data point (Mojo TC on consumer Blackwell, where cuda-oxide has zero TC
  reach and cuTile reaches 172 TF f16). **If HMMA-count = 0**, document
  the constraint in ANALYSIS.md and report the f32 fallback number with
  ADR-0004-style "no-TC ceiling" framing.
- Phase C **deliverable**: a fifth column in the headline compute-bound
  table.

**Phase D — optional, attention column** (depends on Phase C outcome)
- `mojo-attn-gqa/` mirroring `oxide-attn-gqa`. Only if Mojo TC works on
  sm_120 in Phase C; otherwise this is a no-TC ceiling probe like
  `oxide-attn-gqa` (24 TF) and not the comparison we want.
- **Avoid sink-aware flash attention** until issue #6198 / PR #6209 is
  resolved upstream — we don't want to bench broken numerics.

**Out of scope for Wave 18:**
- MAX serving / model inference (Qwen3-8B, etc.) — that's MAX, not Mojo,
  and post-PR #6059 it serves at 95 tok/s on RTX 5090 with a fallback
  matmul; an interesting but different class of benchmark.
- AMD MI300X cross-vendor comparison — would need hosted access.
- Modular's flagship Blackwell matmul reproduction — sm_100a-only.

## Reference points (existing repo data we'll be comparing against)

From the latest README headline table (N=4096, RTX 5090 sm_120):

| metric | nvcc | cuda-oxide | cuTile | cuBLAS |
|---|---:|---:|---:|---:|
| vec-add 256M (GB/s) | 1568 | 1573 | 1559 | — |
| reduce_sum 256M (GB/s) | 1522 | 1519 | **1696** ⚡ | — |
| matmul 4096 f32 (TFLOPS) | 38.4 (tiled) | 45.0 (microtile) | 8.7 ⚠️ | 73.6 / 104.2 (tf32) |
| matmul 4096 f16 TC (TFLOPS) | — | (no API) | **172.5** | **218.4** |

A successful Wave 18 adds a "Mojo" row to all four with fresh data on the
same idle GPU (per Wave 12 lesson: re-bench baselines in the same session
as a new entrant to avoid thermal-state-induced false leads).

## Risk register

| risk | mitigation |
|---|---|
| Mojo install fails on Ubuntu 24.04 / WSL2 | Try pixi first; fall back to docker container with Ubuntu 22.04 base if glibc mismatch. Phase A budget caps this at 1-2 hours. |
| Mojo's `DeviceContext` doesn't pick up CUDA driver | Set `MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas`, set `MLRT_CUDA_DEBUG=1` and read its output. |
| sm_120 path crashes despite #6059 | Pin nightly version that includes the post-#6059 fix; document version in SETUP.md. |
| Mojo's reduction is not TMA-lowered | That's a finding, not a failure. Document the SASS, show why Mojo took a different path. |
| Mixed-precision MMA not engaged on sm_120 | Same — document the SASS instruction count, file as ADR-0004-style ceiling note. |
| Tiled matmul "obviously beats cuTile/cuda-oxide" | Re-bench all three on the same idle GPU in the same session; use cudaEvent timing identical to existing harnesses; double-check cuobjdump path is `/usr/local/cuda/bin/cuobjdump`. Wave 12 thermal-drift lesson applies here. |

## Open questions to resolve while implementing

1. Does Mojo expose a tile/TMA primitive comparable to cuTile's `ct.load`?
   If yes, we should test it; if no, the reduction comparison will be a
   pure language-vs-language test of the same warp-shuffle algorithm.
2. What's Mojo's equivalent of `cuda_launch!` (cuda-oxide) /
   `ct.launch(stream, grid, kernel, args)` (cuTile)? Probably
   `ctx.enqueue_function[kernel](args, grid_dim=..., block_dim=...)`.
3. Does `mojo build --target-accelerator=sm_120` give us the same kind of
   single-arch native code we get from `nvcc -arch=sm_120` and
   `cargo oxide build --arch sm_120`?
4. Does Mojo emit a usable `.cubin` we can disassemble with
   `cuobjdump --dump-sass`? (cuTile does, with the recipe in
   `references/cutile-cubin-extraction.md`.)

These are Phase A / Phase B exit-criteria questions. Answering them is part
of the wave deliverable.

## Decision

This wave is structurally a **frontend-axis addition** (like Wave 12 cuTile),
not a kernel-class addition (like Wave 15 GQA). Right phasing is the cuTile
phased rollout: smoke-test → memory-bound → compute-bound → optional
attention.

It is **not appropriate to dispatch a 5-subagent fan-out at the start** the
way Wave 17 does for new mechanism cells. Mojo is a new toolchain; its
install pitfalls, sm_120 specifics, and API surface need to be discovered
serially before we can write subagent prompts with the level of file-ownership
specificity the repo's parallel-subagent-benchmarking discipline requires.

Wave 18 starts serial (Phase A by orchestrator), and only opens up to
parallel subagent work once the toolchain is known-good and we can split
"author Mojo kernel + harness in folder X" into clean units.
