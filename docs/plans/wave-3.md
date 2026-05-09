# Wave 3: FMA investigation + write-up + upstream issue draft

**Status:** queued (depends on Wave 1 + Wave 2)
**ADR refs:** 0003 (scope)
**Subagents:** 3 in parallel
**Budget:** ~30 min wall-clock, ~80k summary tokens
**Acceptance test:** SUMMARY.md, docs/upstream-issue-fma.md, AGENTS.md updated; final commit pushed.

## File-ownership table

| Subagent | Owns | Reads |
|---|---|---|
| W3A — FMA toggle experiment | `docs/experiments/fma-toggle.md` | docs/research/cuda-oxide-flags.md, all wave outputs |
| W3B — Upstream issue draft | `docs/upstream-issue-fma.md` | research + experiment + PTX dumps |
| W3C — SUMMARY.md final write-up | `SUMMARY.md`, updates to `README.md` headline table | all results CSVs, all ANALYSIS.mds, BACKLOG.md |

## W3A — FMA toggle experiment

**Goal:** Empirically test whether we can coax cuda-oxide into emitting `fma.rn.f32`. Three approaches:

1. **`core::intrinsics::fmuladdf32`** — explicit FMA intrinsic in source. Per Phase 3 research, this lowers to `__nv_fmaf` libdevice call, NOT to `llvm.fmuladd`. We expect this to NOT improve. Verify.
2. **Manual LLVM IR patch** — too invasive for this loop.
3. **Inline PTX `fma.rn.f32`** — does cuda-oxide support `asm!()` blocks targeting PTX? Check the source. If yes, write a kernel using inline PTX FMA, bench it.

**Steps:**
1. Add a kernel `matmul_fmuladd` to `oxide-matmul/src/main.rs` that uses `core::intrinsics::fmuladdf32` (requires `#![feature(core_intrinsics)]`). Bench. Inspect resulting PTX. Document whether `fma.rn.f32` appeared (we predict no).
2. Search cuda-oxide source for inline-asm support: `grep -rn "asm!" /home/codeseys/.cargo/git/checkouts/cuda-oxide-*/`. If found, add `matmul_inline_fma` kernel.
3. Document the experiment in `docs/experiments/fma-toggle.md`: hypothesis, method, results, conclusion.

**Acceptance:** experiment doc written; if any approach yields fma.rn.f32 in PTX, results.csv updated; if none did, conclusion is "upstream patch needed."

## W3B — Upstream issue draft

**Goal:** Author a precise, evidence-based GitHub issue for `NVlabs/cuda-oxide` flagging the FMA + read-only-cache gap. Draft only — user submits manually.

**Structure:**
1. Title: "FMA contraction not emitted on f32 mul+add chains; FastmathFlags::default() blocks all fast-math"
2. Repro: link to this benchmark repo + commit + ANALYSIS.md
3. Evidence: PTX excerpts showing 0 fma.rn vs nvcc's 5; the `FastmathFlagsAttr::default()` lines from `mir-lower` (cite file:line).
4. Proposed fix: thread a config through. Two-line change pattern (see `docs/research/cuda-oxide-flags.md` "What we'd need to add upstream" section).
5. Workaround question: is there an undocumented escape-hatch we missed? Polite request for guidance.

**Acceptance:** draft is ≤500 words, all claims have file/line citations, no speculation about NVIDIA's internal priorities.

## W3C — SUMMARY.md final write-up

**Goal:** A single document that someone reading just this one file can use to evaluate cuda-oxide.

**Structure:**
1. **TL;DR** (3 bullets max): cuda-oxide hits X% of nvcc on naive matmul / Y% with tiling / safety tax is Zx
2. **Methodology** (link to METHODOLOGY.md, brief summary)
3. **Results** (the master table from results/scaling-summary.md)
4. **Three findings** (carry forward from README, expand with new wave-1+wave-2 data)
5. **Compiler gaps** (FMA, ld.global.nc; link to upstream issue)
6. **Setup gotchas** (sm_120 vs sm_89, /usr/bin/nvcc shim, WSL2 wgpu)
7. **What's next** (the BACKLOG followups)
8. **For Rust developers considering cuda-oxide today** (1-paragraph guidance)
9. **Acknowledgments**

**Also:** update `README.md` headline table with the new event-based numbers.

**Acceptance:** SUMMARY.md is internally consistent (no claim contradicting an ANALYSIS.md or research doc); README headline matches SUMMARY headline.

## Wave 3 Concurrent review

After W3A finishes but before W3B/W3C: one cross-family reviewer reviews W3A's experiment doc — did the experiment actually run, or was it claimed-to-run? Verify run.log timestamps + PTX inspection.

## Phase 8 review (separately scheduled)

Phase 8's 3-way scatter happens after Wave 3 completes. Reviews the entire delta of waves 1-3 in one pass. Different framing: not per-wave but cross-cutting.

## Reflexion checklist

- [ ] All three artifacts committed
- [ ] BACKLOG.md flagged: U1, W1, F1, F2 → ✅
- [ ] AGENTS.md updated with consolidated lessons
- [ ] Push
