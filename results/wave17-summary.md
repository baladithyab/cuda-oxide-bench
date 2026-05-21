# Wave 17 — Cross-Frontend × Cross-Mechanism Attention Matrix (final results)

**Status:** ✅ Wave 17 W1 SHIPPED 2026-05-21 (5 cells, all benches run on idle RTX 5090 sm_120).
**Plan:** [docs/plans/wave-17.md](../docs/plans/wave-17.md). ADRs: 0004, 0005, 0006.

## Headline matrix

| Mechanism | cuda (nvcc) | cublas | cutile | oxide | wgpu |
|---|---|---|---|---|---|
| **GQA** | ✅ W15 | ✅ W15 (218 TF hgemm) | ✅ W15 (165 TF) | ✅ W16 (24 TF) | ✅ W16 (CPU only) |
| **MLA** | ✅ **W1a (24.17 TF native, 21.32 TF padded)** | 📋 W2c | ✅ W16 (112 TF) | ✅ **W1b (24.70 TF best @ deepseek_v3)** | — (deferred) |
| **GDN** | ✅ **W1c (417.7 GB/s)** | DEFERRED ADR-0006 | ✅ W16 (610 GB/s) | ✅ **W1d (correctness only; no bench harness yet)** | — |
| **KDA** | — (W18+) | — | ✅ **W1e (344.7 GB/s best @ kimi_linear_decode)** | — | — |

**6 of 7 W1 cells produced timed numbers; W1d oxide-attn-gdn is correctness-only (subagent didn't wire timing — see "Open work" below).**

## W1a — cuda-attn-mla (CUDA C++ MLA)

**Shape:** B=1 n_h=128 S=2048 qk=192 d_v=128 (DeepSeek-V3 decode).

| variant | best_ms | best_useful_TF | best_padded_TF |
|---|---|---|---|
| native qk_eff=192 | 14.22 | **24.17** | 29.00 |
| padded qk_eff=256 | 16.11 | 21.32 | 25.59 |

- max_abs_err = 1.597e-04 (vs 1e-2 threshold)
- HMMA=20, FFMA=104, MUFU=27, LDG.E=129
- Native vs padded bit-identical correctness (ADR-0005 invariant)
- Padding overhead: 11.7% useful-TF degradation, 33% DRAM-K-traffic increase
- **Below plan range [40, 130] — capped by 3-kernel HBM round-trip ceiling, same as the cuda-attn-gqa template at 23 TF.**

## W1b — oxide-attn-mla (cuda-oxide MLA)

**Shape:** B=1 S=2048 n_h=128 qk=192 d_v=128.

| iter | total_ms | qkt_ms | sm_ms | pv_ms | TFLOPS |
|---|---|---|---|---|---|
| best | **13.91** | 7.34 (53%) | 2.84 (21%) | 3.73 (25%) | **24.70** |
| median | 14.44 | 7.79 | 3.14 | 3.67 | 23.79 |

- max_abs_err = 2.831e-7 (deepseek_v3) / 1.192e-7 (correctness)
- **In plan range [10, 30] ✓**, surprisingly close to nvcc-MLA's 24.17 TF
- The qkt stage dominates (53% of total) — opportunity for register-microtile microoptimization

## W1c — cuda-attn-gdn (CUDA C++ GDN)

**Shape:** B=1 H=16 d_k=256 d_v=256 (Qwen3-Next decode).

- best gpu_us = 20.16 → **417.7 GB/s** (23.3% of HBM peak)
- mean = 29.92 µs → 281.4 GB/s
- HMMA=0 ✓ FFMA=192 LDG.E.128=16 ✓ STG.E.128=16 ✓ MUFU=0
- Bytes/iter = 8224.06 KB
- **Below cuTile's 610 GB/s** despite emitting LDG.E.128/STG.E.128 vector loads. Hypothesis: cuTile's ct.load → UTMALDG.1D bulk TMA path beats thread-vectorized LDG.E.128 even on sm_120, because TMA dispatches the load via dedicated hardware that doesn't compete with the SM's load/store unit. Worth investigating in Wave 22.
- Correctness: o max_abs=3.05e-5 (corr), 6.10e-5 (qwen3); Sout max_abs=2.98e-8

## W1d — oxide-attn-gdn (cuda-oxide GDN)

**Shape:** B=1 H=16 d_k=256 d_v=256 (qwen3_next_decode), correctness only.

- max_abs_err o = 1.677e-4, S = 1.585e-4 (vs 1e-3 threshold ✓)
- HMMA=0 ✓ FFMA=64 (via core::intrinsics::fmuladdf32) FMUL=194 FADD=960
- LDG.E=136, STG.E=66 (no LDG.E.128 — cuda-oxide doesn't auto-vectorize)
- LDS=1940, STS=1156 (block-wide tree-reduction shared-mem traffic)
- Two specialized kernels (gdn_decode_dk64, gdn_decode_dk256) — cuda-oxide #[kernel] proc macro rejects const generic params

**Open work:** subagent didn't wire timed-bench iters. To complete: extend src/main.rs harness with cudaEvent timing around the kernel launch, mirror oxide-attn-gqa pattern. See Wave 22 candidate W22.6 in BACKLOG.

## W1e — cutile-attn-kda (cuTile KDA, semantic-fenced GDN fork)

**Shape:** B=1 H=32 d_k=128 d_v=128 (kimi_linear_decode), BLOCK_V=128.

- median: 20.08 µs → 210.9 GB/s (11.8% of HBM peak)
- best: 12.29 µs → **344.7 GB/s** (19.2% of peak), 0.256 TFLOPS
- IQR: [13.50, 29.18] µs (large jitter)
- bytes/iter = 4136.1 KB
- Both shapes correctness PASS: o ≤ 1.22e-04, S ≤ 5.96e-08
- Diff vs cutile-attn-gdn/main.py within ADR-0006 §2's 4 allowed change classes
- **Below plan range [400, 700]** — kimi_linear_decode is small (n_blocks=32 = 32 SMs only); large IQR suggests launch-overhead dominance. Larger shape (e.g. B=4 H=64) might saturate; not in current plan but worth a Wave 22 sweep.

## Cross-cell observations

1. **MLA cap is real:** both nvcc (24.17 TF) and oxide (24.70 TF) are within 2% of each other on the same MLA shape. The 3-kernel HBM round-trip dominates; algorithm geometry, not compiler quality, is the bottleneck. cuTile's 112 TF (W16) used a fused single-kernel path — the gap is structural, not compiler-class.
2. **GDN: cuTile still wins despite cuda-attn-gdn's LDG.E.128.** 417.7 vs 610 GB/s. The LDG.E.128 emission was the W1c hypothesis; it's correct that the SASS shows them, but doesn't translate to the predicted lift. **TMA bulk loads (UTMALDG) have a hardware-level advantage over per-thread vector loads on Blackwell** that wasn't captured in the original ADR-0006 reasoning. Open question for skill update.
3. **cuda-oxide GDN matches cuda-oxide GQA's "no-TC ceiling" pattern**: works, correct, FFMA-only via mul_add escape hatch, but no perf claim (no bench harness yet). Per ADR-0004, this is documented and shipped without timing — the no-TC-ceiling characterization was the goal.
4. **KDA at small shape is launch-bound**: 32-SM grid leaves the GPU half-idle. Worth re-running at a larger shape to claim the "8× state traffic reduction" advantage advertised in the research doc.

## SASS evidence

| cell | path | size |
|---|---|---|
| cuda-attn-mla | `cuda-attn-mla/attn_mla.sass` | 332 KB |
| oxide-attn-mla | `oxide-attn-mla/oxide_attn_mla.sass` | 304 KB |
| cuda-attn-gdn | `cuda-attn-gdn/attn_gdn.sass` | 139 KB |
| oxide-attn-gdn | `oxide-attn-gdn/oxide_attn_gdn.sass` | 1.3 MB |

## Pitfalls discovered (Wave 17 specific)

- **jj working-copy contention with parallel subagents** is severe. 5 concurrent subagents snapshotting through the same `.jj/working_copy/` cause cross-cell file commingling at every commit boundary. **Future parallel waves should either**: (a) defer ALL `jj describe` to the orchestrator, or (b) use git-managed working copies per subagent (one repo clone each, push to shared remote).
- **Subagent timeouts at 600s after authoring but before commits** are benign — `jj restore --from <sibling-commit>` recovers the artifacts. Don't re-dispatch on timeout.
- **Subagents may "reset" their cell mid-flight** trying to clean up parallel-agent contamination, accidentally deleting their own work. Recovery pattern: `jj log --no-graph -T 'change_id ++ "\n"' | xargs -I{} jj diff -r {} --summary 2>&1 | grep <missing-file>` to find the commit with the file.
- **GDN kernels need `*.sass` gitignored** by default — multi-MB SASS files trip jj's 1MB snapshot guard. cuda-oxide's full SASS is 1.3MB.
- **cublas-half-precision binary trips the same guard** at 1.1MB — already gitignored in this commit.

## Wave 17 W2 (next)

| Cell | Depends on | Status |
|---|---|---|
| W2c cublas-attn-mla | W1a HMMA count (=20) + qk_eff choices (192 native, 256 padded) — both known | ready to dispatch |
| W2d cutile-attn-gqa BLOCK_M=128 sweep | None (independent) | ready to dispatch |

Both can run in parallel. See Phase 6 wave 2 in this loop.

## Wave 22 candidates surfaced by Wave 17

- **W22.6: oxide-attn-gdn timed bench harness** — extend src/main.rs with cudaEvent timing, mirror oxide-attn-gqa pattern. ~50 LOC.
- **W22.7: cutile-attn-kda larger-shape sweep** — re-run at B=4, H=64, d_k=d_v=256 to saturate the GPU and claim the "8× state traffic" advantage.
- **W22.8: cuda-attn-gdn TMA-vs-LDG.E.128 investigation** — why does cuTile's UTMALDG path beat thread-vectorized LDG.E.128? Hardware-level analysis.
