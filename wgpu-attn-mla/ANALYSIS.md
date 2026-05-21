# wgpu-attn-mla — Analysis (Wave C1.6, Rosetta Stone)

## Headline (LLVMPIPE CPU under WSL — NOT GPU PERF)

**Critical caveat**: All TFLOPS numbers in this cell are measured on **Mesa
LLVMPIPE (Vulkan, type=Cpu)**, the only adapter wgpu can find on this WSL host.
WSL does NOT expose the RTX 5090 to Vulkan, so wgpu cannot reach the GPU.
**Treat these numbers as a CPU-emulation lower bound, not as WebGPU
performance characterization.** To get true wgpu-on-GPU numbers, reruns are
needed on:

- A native Linux host with an NVIDIA Vulkan ICD installed
- A native Windows host (DX12 backend would also work)
- A remote-GPU server with Vulkan exposed

| metric | correctness shape | medium shape | deepseek_v3 |
|---|---:|---:|---:|
| shape | (B=1, H=4, S=128, qk=96, d_v=64) | (B=1, H=4, S=512, qk=96, d_v=64) | (B=1, H=128, S=2048, qk=192, d_v=128) |
| best ms | 4.556 | 41.202 | SKIP |
| best TFLOPS | 0.0046 | 0.0081 | SKIP |
| median ms | 5.045 | 44.338 | SKIP |
| median TFLOPS | 0.0042 | 0.0076 | SKIP |
| max_abs_err | **1.192e-7** ✅ | (perf-only) | SKIP |
| atol | 1e-2 | — | — |

**deepseek_v3 SKIPPED**: `B·n_h·S²·4B = 1·128·2048²·4 = 2 GiB` scores buffer
exceeds llvmpipe's 128 MiB `max_storage_buffer_binding_size`. To run, would
need a real GPU adapter (real binding caps are ~32 GB) or a fused/
FlashAttention-style scores-tile streaming rewrite that never materializes
the full attention scores matrix.

## What this cell is

A WGSL+Rust port of MLA attention (used in DeepSeek-V3) for the Rosetta Stone
WebGPU column. Implements the canonical 3-kernel decomposition:

1. **qkt_kernel**: `Q @ K^T → scores[B,H,S,S]`
2. **softmax_kernel**: row-wise softmax with scale and causal mask
3. **pv_kernel**: `softmax(scores) @ V → output[B,H,S,d_v]`

Each kernel is a separate `wgpu::ComputePass` dispatch with a global memory
round-trip via the scores buffer between stages. This is the same structure
used by `cuda-attn-mla`, `oxide-attn-mla`, `mojo-attn-bf16`. **WGSL has no
tensor-core primitive** — the matmul stages are FFMA-class (single-precision
fused multiply-add), so the closest peer for cross-frontend perf comparison
is `oxide-attn-mla` (24.70 TF, FFMA + no TC) or `cuda-attn-mla` (24.17 TF,
WMMA but FFMA-class accumulation).

## Cross-frontend Rosetta context

| frontend | TFLOPS @ DeepSeek-V3 (B=1 H=128 S=2048) | mechanism | notes |
|---|---:|---|---|
| cuTile-MLA fused | 112 | online-softmax FA-class | fused single-pass |
| cublas-attn-mla | 47 | 3-kernel + cuBLAS bgemm | vendor library |
| **mojo-attn-bf16** | **26.36** | 3-kernel + hand-WMMA bf16 | beats cuda+oxide |
| oxide-attn-mla | 24.70 | 3-kernel + hand-FFMA, no TC | closest peer |
| cuda-attn-mla | 24.17 | 3-kernel + WMMA bf16 (FFMA accum) | C++ baseline |
| **wgpu-attn-mla (THIS, on real GPU)** | **expected 5-15** | 3-kernel + FFMA, no TC | extrapolation |
| **wgpu-attn-mla (THIS, LLVMPIPE)** | **0.0046** | 3-kernel + FFMA, on CPU | NOT GPU perf |

The expected real-GPU performance class for WGSL FFMA-no-TC attention is
roughly 5-15 TF on RTX 5090 (slightly below `oxide-attn-mla`'s 24.70 TF
because of Vulkan-via-wgpu translation overhead vs. native CUDA driver). On
LLVMPIPE CPU, the kernel runs at ~5,000× slower because LLVMPIPE is a
software rasterizer, not a SIMD-optimized CPU compute backend.

## WGSL idioms used (Rosetta Stone reference)

- `var<workgroup>` arrays for per-tile shared memory
- `workgroupBarrier()` between K-loop iterations
- `@compute @workgroup_size(16, 16)` for the qkt and pv 2D thread layouts
- `@compute @workgroup_size(256)` for softmax (1D row-wise)
- Storage buffer alignment: WGSL forces `array<f32>` to 4-byte stride; no
  `array<f16>` available without `enable f16` extension (not on llvmpipe).

## Pitfalls (Mojo 1.0.0b1 → WGSL parity issues)

1. **No FP16 / BF16 in WGSL** without the experimental `enable f16` extension.
   This cell uses f32 inputs; if the algorithm needs bf16 inputs, the host
   must cast on CPU before staging.
2. **No tensor cores**. WGSL has no `mma()` primitive. All matmul stages are
   FFMA-bound. WGSL spec doesn't expose hardware MMA at all.
3. **No subgroup primitives** at the wgpu 22 stability level. Reduce/scan
   patterns must use shared memory + `workgroupBarrier()`, not warp shuffles.
4. **Storage-buffer binding cap on llvmpipe = 128 MiB**. Real GPUs allow ~32 GB.
   This affects shape coverage on this WSL host.
5. **timestamp_query feature**: present on llvmpipe (lucky). On older wgpu
   versions or restricted contexts, fall back to `Instant::now()` host loops.
6. **No async pipe / TMA equivalent**. WebGPU has no `cuda::pipeline` or TMA
   analog; the closest pattern is overlapping `wgpu::Queue::submit` calls
   from a host-side scheduler — coarse-grained, not register/smem-level.

## Reproduction

```bash
cd /home/codeseys/cuda-exploration/wgpu-attn-mla
./run.sh
```

Inspect `run.log` for adapter list (you should see `Vulkan llvmpipe ... type=Cpu`)
and per-shape TFLOPS / correctness numbers.

## Verdict

This cell SHIPS for the Rosetta Stone WGSL column with the explicit caveat
that the perf numbers are LLVMPIPE CPU emulation, not RTX 5090 GPU
throughput. The correctness number (max_abs_err=1.192e-7, well within
1e-2 tolerance) is real and would carry over to a real-GPU rerun unchanged.

## Phase 7d MED resolution

Phase 7d cross-family review (gpt-5.5/openrouter, 2026-05-21) flagged
this cell as missing `ANALYSIS.md`. This file fixes that MED. The Phase 7d
verdict otherwise confirmed all C1.6 critical claims:

- Fresh run.log shows correct LLVMPIPE warning and `max_abs_err=1.192e-7`
- deepseek_v3 SKIP behavior is consistent and correctly documented in source
- Per-stage breakdown (qkt 31% / sm 34% / pv 26%) matches the W17 cross-
  frontend MLA structural pattern.
