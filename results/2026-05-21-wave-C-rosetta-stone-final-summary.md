# Wave C continuation: Rosetta Stone — Final summary

## Headline

The cuda-exploration repo is now a **6-frontend Rosetta Stone** for GPU
programming: cuda (NVIDIA C++/PTX), cublas (vendor lib), cutile (NVIDIA's
Python tile DSL), oxide (Rust-CUDA via cuda_std), mojo (Modular's language),
wgpu (WebGPU/WGSL).

## Wave C totals

- **3 commits to master**: C1 (5cf994b9), C2 (725d117e), C3 (this commit)
- **18 cells shipped** (C1×6, C2×6, C3×4 + Phase 7d + Rosetta doc)
- **Cross-frontend matrix** advanced from ~30 cells in 5 frontends → 60+ cells
  in 6 frontends with explicit Rosetta mapping document

## Cell inventory (Wave C)

### C1: vecadd / reduction / matmul-tiled / attn-mla / attn-gqa
- C1.1 wgpu-vecadd: 17.26 GB/s LLVMPIPE, BIT-EXACT
- C1.2 wgpu-reduction: 1.9 GB/s LLVMPIPE, rel_err=1.8e-7
- C1.3 wgpu-matmul-tiled: 0.006 TF LLVMPIPE
- C1.4 mojo-matmul-tiled: 7.69 TF FFMA (no microtile, no MMA), BIT-EXACT
- C1.5 mojo-attn-gqa: 28.97 TF best, BIT-EXACT, **+10% over cuda-attn-gqa**
- C1.6 wgpu-attn-mla: 0.0046 TF LLVMPIPE, max_abs_err=1.192e-7

### C2: TC matmul + GDN/KDA expansion
- C2.1 cuda-matmul-tc-bf16: **147.5 TF** hand-WMMA, 64× HMMA, 67% of cuBLAS
- C2.2 cutile-matmul-tc-bf16: **160.6 TF** DSL, 73% of cuBLAS
- C2.3 mojo-attn-gdn: **320.5 GB/s**, 5th GDN frontend, beats oxide
- C2.4 oxide-attn-gdn-tma: TMA reach proven (UTMALDG=2 in SASS)
- C2.5 cutile-attn-gdn-tma: DSL TMA falsified (byte-identical cubins)
- C2.6 cuda-attn-kda: **568.4 GB/s** at large saturation

### C3: Advanced TMA/async + Rosetta doc
- C3.1 docs/rosetta-stone.md: 42,659 chars, 8 sections, 10 algorithms
- C3.2 cutile-attn-mla-tma: 112.3 TF, DSL TMA falsified again (consistent w/ C2.5)
- C3.3 oxide-attn-mla-tma: TMA + WMMA-tile-layout matched; UTMALDG=3 across 3 kernels
- C3.4 mojo-attn-gdn-async: 🏆 **610.6 GB/s — +91% over sync**

## The biggest finding: Mojo's `copy_dram_to_sram_async` doubles GDN throughput

Wave C3.4 vs Wave 22.9 contrast:
- W22.9 cuda-attn-gdn-async (`cuda::pipeline<thread_scope_thread>`): 311.8 GB/s = **−25% regression** vs FFMA baseline 417 GB/s
- C3.4 mojo-attn-gdn-async (`copy_dram_to_sram_async` CTA-collective): **610.6 GB/s = +91% lift** over Mojo sync 320.5

**Why**: cuda::pipeline does per-thread token bookkeeping in a 4-stage ring
buffer. Mojo's primitive is CTA-collective bulk-load-then-drain — no
per-thread tokens, no ring buffer, in-place compute compatible. Where C++
chose the wrong async abstraction, Mojo's higher-level API picked the right
one.

## DSL falsification: cuTile TMA refuses attention shapes

Two cells in Wave C tested cuTile DSL's TMA support on attention:
- C2.5 cutile-attn-gdn-tma: `allow_tma=True` parses but produces byte-identical cubin to `allow_tma=False`. UTMALDG=0 in SASS.
- C3.2 cutile-attn-mla-tma: same falsification — UTMALDG=0, byte-identical cubins, even though the matmul-friendly tile shapes (BM=BN=128 BK=16) used inside `ct.mma` calls.

Control: cuTile DOES emit TMA for `cutile-matmul-tiled` (W13 SASS shows 17 UTMA matches). The compiler refuses TMA specifically for the FlashAttention-2 / state-recurrence patterns (likely heuristic conflict with online softmax + 99KB static smem + sm_120's 100KB cap).

## Cuda-oxide TMA reach proven (independent of perf lift)

Two Wave C cells transferred the TMA recipe from C++ to Rust-CUDA:
- C2.4 oxide-attn-gdn-tma: cuda_device::tma::cp_async_bulk_tensor_2d_g2s lowers to UTMALDG.2D in SASS. Bench tied at 278 GB/s with FFMA baseline (parallelization-shape limited, not TMA-availability limited).
- C3.3 oxide-attn-mla-tma: same recipe extended to 3-kernel MLA (UTMALDG=2 in qkt + 1 in pv). Correctness PASS first attempt.

Pattern: descriptor encoded host-side via `cuTensorMapEncodeTiled` FFI through `cuda_core::sys`, `transmute_copy` to `[u8; 128]`, `DeviceBuffer::from_host`, kernel param as device pointer. Mirror the C2.4/C3.3 solution for any future cuda-oxide TMA work.

## Phase 7d cross-family review

Reviewer: openai/gpt-5.5 via openrouter (model override routed correctly,
consistent with Phase 7c finding).

**Verdict: 0 HIGH, 1 MED (resolved this commit), 4 LOW.**

Critical claims confirmed:
- C1.5 +10% over cuda-attn-gqa: actual 9.74%, fair rounding
- C2.4 oxide TMA reach: UTMALDG.2D=2, LDG.E.128=0 in fresh disassembly
- C2.5 byte-identical cubins: MD5 c102d09456668b2590d97f2198f60070 across all 3 legs
- C1.4 mojo-matmul-tiled BIT-EXACT: rerun max_abs_err=0.0, 7.25 TF (within 6% of claim)
- C2.1 64× HMMA, C2.2 160.572 TF, C2.6 568.415 GB/s @ large

MED resolved this commit: wgpu-attn-mla/ANALYSIS.md added with
explicit LLVMPIPE-CPU-vs-GPU caveat.

## Cross-frontend matrix (post Wave C)

| algorithm | cuda | cutile | oxide | mojo | wgpu | cublas | rosetta-essential? |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| vecadd | ✅ | ✅ | ✅ | ✅ | ✅ | N/A | ✅ ROSETTA |
| reduction | ✅ | ✅ | ✅ | ✅ | ✅ | N/A | ✅ ROSETTA |
| matmul-naive | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ ROSETTA |
| matmul-tiled | ✅ | ✅ | ✅ | ✅ | ✅ | N/A | ✅ ROSETTA |
| matmul-tc-bf16 | ✅ | ✅ | ✅ | ✅ | N/A | ✅ | ✅ ROSETTA |
| matmul-tc-fp8 | — | — | — | ✅ | N/A | — | research-only |
| matmul-tma | — | (W13 implicit) | — | ✅ | N/A | — | research-only |
| attn-mla | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ ROSETTA 6/6 |
| attn-mla-tma | ✅ | ✅ (falsified) | ✅ | — | N/A | — | research |
| attn-gqa | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ ROSETTA 6/6 |
| attn-gdn | ✅ | ✅ | ✅ | ✅ | — | — | 5/6 |
| attn-gdn-tma | ✅ | ✅ (falsified) | ✅ | — | N/A | — | research |
| attn-gdn-async | ✅ (regressed) | — | — | ✅ (won) | N/A | — | research |
| attn-kda | ✅ | ✅ | — | — | — | — | 2/6 |
| 3dgs | ✅ | ✅ | ✅ | ✅ | — | N/A | 4/5 |
| 3dgs-binned | — | ✅ | — | — | — | N/A | 1/5 |

**Rosetta-essential rows are 6/6 frontend-complete** for vecadd, reduction,
matmul-naive, matmul-tiled, matmul-tc-bf16 (where applicable), attn-mla,
attn-gqa.

## Hardware-constraint summary

1. **WSL + wgpu = LLVMPIPE CPU only**. RTX 5090 not exposed to Vulkan from
   WSL. ALL wgpu-* numbers are CPU emulation, not GPU. Native Linux/Windows
   needed for true wgpu-on-GPU perf.
2. **Mojo 1.0.0b1 lacks TMA primitives** (W22.1 BLOCKED.md). cp_async_bulk,
   cuTensorMap, async_copy, bulk modules ABSENT. Closest available:
   mbarrier_arrive (consumer side only). Defer until Modular ships them.
3. **cuTile DSL TMA is shape-conditional**: emits TMA for matmul-tiled
   (W13: 17 UTMA matches), refuses for GDN/MLA attention shapes (C2.5 + C3.2
   falsifications). Compiler heuristic, not user-controllable.
4. **cuda-oxide has no usable WMMA on sm_120** (Wave 14.4 finding). Stuck on
   FFMA microtile. TMA reach works (C2.4, C3.3) but no TC ceiling lift.
5. **WGSL has no FP16/BF16** without `enable f16` extension. No tensor cores.
   No subgroup primitives at wgpu 22 stability level. No async pipe/TMA
   equivalent.
6. **TMA boxDim ≤ 256** hardware limit (cuTensorMapEncodeTiled). D_K=512
   shapes (W22.15 wide) need per-CTA two-tile assembly.

## Decision matrix: when to use which frontend

- **Max performance**: cuda + cublas. cublas is 219 TF (cuBLAS hgemm/bgemm), the
  ceiling. cuda hand-WMMA is 147 TF (67% of cublas). Hand-rolled `mma.sync` +
  `ldmatrix` could close that gap further.
- **Compact research code**: cuTile DSL. 160 TF in 100 lines of Python with TC
  + autotuned launch geometry. Best balance of perf-per-LOC. But TMA is
  shape-conditional (research caveat for attention).
- **Type-safe accelerated**: Mojo. Beats cuda+oxide MLA/GQA at the same
  algorithm, doubles GDN throughput via async-pipe primitive. Lacks TMA
  primitives in 1.0.0b1, hand-MMA mna() works for FP8 e4m3.
- **Systems-Rust on GPU**: cuda-oxide. TMA reach proven, no WMMA on sm_120.
  Stuck at FFMA-class ceiling but offers the Rust-CUDA workflow.
- **Portable cross-platform**: wgpu/WGSL. No tensor cores, no async pipe, no
  TMA. Works wherever wgpu+Vulkan/DX12/Metal are available — but on WSL,
  forces CPU fallback which is disqualifying for perf.

## Next-loop seeds

- **Wave 23.4** mojo-attn-mla-async: apply C3.4's `copy_dram_to_sram_async`
  recipe to 3-kernel MLA. Hypothesis: the same +91% lift might apply to MLA's
  qkt and pv stages, lifting mojo-attn-bf16 from 26.36 TF toward 50+ TF.
- **Wave 23.5** mojo-attn-gqa-async: same recipe for GQA (currently 28.97 TF).
- **Wave 23.6** wgpu-* on native Linux/Windows: rerun all wgpu-* cells on a
  host with NVIDIA Vulkan ICD to characterize true WebGPU GPU performance.
- **Wave 23.7** oxide-attn-gdn-tma-vlanes: pivot from one-thread-per-d_k-row
  to 4-vlane-per-thread (matching W22.10's parallelization shape) to push
  past 870-1110 GB/s ceiling.
- **Wave 23.8** mojo-attn-kda: 3rd KDA frontend (closing the kda column).
- **Wave 23.9** cublas-attn-gdn: vendor library baseline for the gdn column.
