# ADR 0006: KDA-decode is implemented as a vector-gate extension of GDN

**Status:** accepted (2026-05-11)

**Context.** KDA (Kimi Delta Attention, used in Kimi-Linear, NOT in K2.x) and GDN (Gated DeltaNet, used in Qwen3-Next) differ by exactly one design dimension — gate granularity:

- **GDN gate:** `α_t ∈ ℝ` per (batch, head, time). Scalar gate broadcasts uniformly across the recurrent state.
- **KDA gate:** `α_t ∈ ℝ^{d_k}` per (batch, head, time). Per-channel gate; each row of the (d_k × d_v) state matrix has its own decay.

Recurrence comparison (verbatim from `docs/research/wave17-kda-spec.md`):
```
GDN: S_t = (I − β_t k_t k_tᵀ) · α_t       · S_{t-1} + β_t k_t v_tᵀ
KDA: S_t = (I − β_t k_t k_tᵀ) · Diag(α_t) · S_{t-1} + β_t k_t v_tᵀ
```

The decay is applied as a separate **S-rescale step** before the rank-1 update, exactly as in GDN. This is NOT a commutativity property of `Diag(α)` with `(I − β k kᵀ)` — those operators do NOT commute in general. **No operator reordering is permitted in the implementation.** The fork from `cutile-attn-gdn` keeps the inner-loop ordering byte-identical and only changes the broadcast shape on the rescale line.

**Decision.** Wave 17.2 implements `cutile-attn-kda` as a **direct fork** of `cutile-attn-gdn`, NOT as a separate from-scratch design.

### Concrete fork policy

1. **File ownership:** `cutile-attn-kda/` is a sibling directory; it copies `cutile-attn-gdn/`'s `main.py`, `bench.py`, `correctness.py`, then changes:
   - Gate tensor shape: `g[B, S, H]` → `g[B, S, H, d_k]`
   - Inside the kernel inner loop: `s_scaled = s_tile * α_f32` → `s_scaled = s_tile * exp_g.reshape(-1, 1)` (broadcasting `(d_k,)` against `(d_k, BLOCK_V)`)
   - PyTorch reference: `naive_recurrent_gdn` → `naive_recurrent_kda` (already in fla/ops/kda/naive.py)
   - Bench shape: GDN's Qwen3-Next (H=16, d_k=d_v=256) → KDA's Kimi-Linear-48B-A3B (H=32, d_k=d_v=128)
   - Headline table: report both TFLOPS-equivalent and GB/s; KDA's per-step state traffic is ~4 MB (8× less than GDN-256 because d_k=d_v=128 not 256)

2. **Diff target:** semantic fence (NOT a LOC cap): the diff vs `cutile-attn-gdn/main.py` may contain ONLY (a) shape-constant changes, (b) the gate-tensor shape change (`g[B,S,H]` → `g[B,S,H,d_k]`), (c) the broadcast-rescale change (`s_tile * α_f32` → `s_tile * exp_g.reshape(-1, 1)`), and (d) the bench-shape change (Qwen3-Next → Kimi-Linear-48B-A3B). NO new kernel functions, NO changes outside the gate-rescale block, NO inner-loop reorderings. If a needed change falls outside this list, stop and amend this ADR before proceeding.

3. **Naming:** `cutile-attn-kda/main.py` retains the GDN function names (`gdn_decode_kernel` becomes `kda_decode_kernel` but the inner-loop variable names stay the same to make the diff readable).

4. **Documentation:** `cutile-attn-kda/ANALYSIS.md` MUST include a section "Diff vs cutile-attn-gdn" that:
   - Lists every changed line with rationale
   - Reports the LOC delta
   - Notes which differences are KDA-semantic (gate broadcast) vs cosmetic (shape constants)

### What this ADR explicitly REJECTS

- Implementing KDA as a from-scratch new kernel without referencing GDN. Wastes effort, hides the gate-granularity story, and produces unfair comparisons (different tile choices, different load patterns).
- Implementing the chunkwise/training-mode KDA. The FlashKDA CUTLASS reference is a 2-kernel CHUNK=16 design that's 10-100× larger than decode. We stay decode-only this wave.
- Cross-frontend KDA cells (oxide-attn-kda, nvcc-attn-kda) **this wave**. They're cheap once cutile-attn-kda is committed but Wave 17 keeps KDA single-frontend to validate the fork policy first. Whether to extend the fork policy to oxide/nvcc is a future-wave decision recorded in `followups.md`, not a forward commitment of this ADR.

**Consequences.**

- **Positive:** KDA cell ships in ~50 LOC of real new code, enabling clean attribution of cost: "the per-channel gate adds X% to register pressure, Y% to runtime."
- **Positive:** future readers see the GDN→KDA delta as a clear ablation, not two unrelated kernels.
- **Negative:** the cell looks "easy" in LOC count, which may undersell the methodological value. We address this in the ANALYSIS.md narrative by quoting the recurrence equations side-by-side.
- **Risk:** if FlashKDA's CUTLASS path ships an optimization the decode kernel can't do (e.g., chunked decay accumulation), our decode-only comparison underestimates KDA's true ceiling. Documented as a known limitation.

**Pre-mortem.**

1. *We discover mid-implementation that the decay-commute claim is wrong for some inner-loop ordering.* The research doc cites the recurrence equation but doesn't formally prove commutativity in Triton-style block layouts. Mitigation: smoke test against `naive_recurrent_kda` BEFORE writing the bench harness; if max_err > 1e-3 vs the PyTorch oracle, we have the wrong loop structure.
2. *Bench shape (H=32, d_k=128) is too small to saturate HBM and we measure overhead-bound numbers.* Per-step state traffic is 4 MB across H=32 heads = 128 KB per head; SM sees this in registers. Mitigation: report both single-step and 64-step batched numbers; if single-step is launch-overhead-bound, headline the 64-step.
3. *The `(d_k, 1)`-broadcast in cuTile pulls a different SASS shape and changes the FFMA count compared to GDN's scalar broadcast.* That's the point of the comparison. SASS diff goes in ANALYSIS.md as evidence.
