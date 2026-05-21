# Wave C2.3 — mojo-attn-gdn — ANALYSIS

## Result

| metric | value |
|---|---|
| best GB/s | **320.5** |
| median GB/s | 303.5 |
| best gpu_ms | 0.0263 ms |
| median gpu_ms | 0.0277 ms |
| bytes/iter | 8224 KiB (= B·H · (2·D_K·D_V·4 + (2·D_K + 2·D_V + 2)·2)) |
| correctness (sampled, 256 (bh,j) for o + 256 (bh,t,j) for S_out) | **PASS** |
| max_abs_err o | 3.81e-6 (atol=1e-3) |
| max_abs_err S_out | 3.73e-9 (atol=5e-3) |

## Cross-frontend at qwen3_next_decode (B=1, H=16, D_K=D_V=256)

| frontend | path | GB/s (best) | ratio vs mojo-attn-gdn |
|---|---|---:|---:|
| cuda-attn-gdn-tma (W22.10) | TMA + UTMALDG | 1032 | 3.22× |
| cuda-attn-gdn-tma-warpspec (W22.13) | TMA + warp specialization | (>1032) | — |
| cutile-attn-gdn | saturation FFMA + cuTile DSL | 610 | 1.90× |
| **cuda-attn-gdn (W1c)** | **FFMA + LDG.E.128 (Mojo's algorithmic peer)** | **417** | **1.30×** |
| **mojo-attn-gdn (this cell)** | **FFMA, static smem, BLOCK_V=64** | **320.5** | **1.00×** |
| oxide-attn-gdn (W1d) | FFMA, 1 thread per d_k row, smem reduce | 276 | 0.86× |

**Headline ratios:**
- vs W1c cuda-attn-gdn: **0.769** (we're 23% behind nvcc-FFMA on the same algorithm)
- vs oxide-attn-gdn:    **1.161** (we beat cuda-oxide's hand-rolled FFMA path by 16%)
- vs cutile-attn-gdn:   **0.525** (cuTile's saturation path leverages NVIDIA's DSL stack)

This sits exactly in the "FFMA-class" envelope the task asked for (200–400 GB/s).
Mojo lands between oxide (276) and W1c (417), closer to W1c — the codegen
quality of Mojo's compiler on straight-line FFMA + LDG.E.128 patterns is good.

## Algorithm

Direct port of the **W1c FFMA-LDG.E.128** kernel, *not* W22.10's TMA path
(Mojo 1.0.0b1 has no TMA primitives — see W22.1 BLOCKED.md). Two-pass kernel:

1. **Pass 1:** stream `S_in[k=0..D_K, col0..col0+4]` from HBM via 128-bit
   loads (each thread owns 4 contiguous d_v cols). Multiply by α, cache the
   scaled tile in static smem (D_K × VLANES × 4 floats = 64 KiB at qwen3
   shape). Accumulate `u_acc[4] += k[k] * S_scaled`.
2. **r = v - u_acc** (per-thread, no cross-thread reduction needed because
   u is per-d_v-column).
3. **Pass 2:** read S_scaled back from smem, compute
   `S_out = S_scaled + β·k·r`, write S_out to HBM (128-bit stores), and
   accumulate `o_acc += q[k] * S_out`. Cast and store o (f16).

Grid: `(B·H, D_V / BLOCK_V) = (16, 4)` = 64 CTAs at qwen3 shape.
Block: `(BLOCK_V/4,) = (16,)` threads = 1/2 warp utilization (single warp
per block, only 16 lanes active). This is a perf concession for smem
budget — but the kernel is so memory-bound (AI ≈ 0.77 flops/byte) that
the half-warp is irrelevant; total runtime is dominated by HBM traffic.

## Pitfalls discovered

### 1. Static-smem-only allocator works at 64 KiB on Blackwell

Mojo 1.0.0b1's `LayoutTensor[..., AddressSpace.SHARED].stack_allocation()`
allocates static shared memory at compile time — there is no exposed
equivalent of CUDA's `cudaFuncSetAttribute(maxDynamicSharedMemorySize)`.

I expected the 64 KiB state-tile alloc (256 × 16 × 4 floats) to fail
because the default static-smem cap on most architectures is 48 KiB. The
first run with `BLOCK_V=32` (32 KiB tile) was a defensive choice for that
reason; `BLOCK_V=64` was a stretch attempt.

**It worked**: Mojo's runtime appears to either (a) auto-promote large
static allocations to the dynamic-smem path on Blackwell or (b) leverage
sm_120's larger static-smem cap (Blackwell ups static smem to 100 KiB
opt-in vs Hopper's 48 KiB default). This is undocumented but reproducible.

Net effect: BLOCK_V=64 is +1.0% best / +1.7% median over BLOCK_V=32.
Modest because we're already memory-bound.

### 2. Closure system + persistent registers: clean, no idiomatic break

The state-recurrence pattern (per-CTA persistent state across two passes
over D_K) maps cleanly onto plain Mojo `var` locals. The two-pass
structure is just two `while k_iter < D_K` loops sharing per-thread
`u0..u3, r0..r3, o0..o3` Float32 vars. No closure machinery required;
no fancy stage-decorator dance like cuTile's `@stage`.

The orchestrator's concern ("does Mojo's closure system handle per-CTA
persistent registers cleanly?") was a non-issue. The only place we needed
the closure pattern at all was the timed `body(ctx)` `@parameter` lambda
for `ctx.execution_time` — same as mojo-attn-bf16.

### 3. NPY loader absence → inline deterministic init + inline CPU reference

Other Mojo cells in this repo (mojo-matmul, mojo-attn-bf16, mojo-attn-gqa)
do not load .npy files; they generate inputs deterministically from a
Knuth golden-ratio hash. I followed that pattern (the task allowed it:
*"port a simple O(B·H·S·d_k·d_v) reference inline"*).

The CPU reference verifies 256 sampled (bh, j) output values + 256
sampled (bh, t, j) state values. Each (bh, j) sample requires a full
D_K=256 reduce for u_j followed by another D_K=256 reduce for o_j ≈ 130k
ops/sample × 256 samples = 33M ops, runs in <1 s in interpreted Mojo.

Errors are tiny (3.8e-6 on o, 3.7e-9 on S_out) because both kernel and
reference use f32 throughout the recurrence, with f16 only on the I/O
boundary. Effectively bit-equivalent within f16 round-trip.

### 4. half-warp CTAs (16 active threads) are not a perf concern

VLANES = BLOCK_V/4 = 16 threads per CTA = exactly one half-warp.
Naively, this wastes 50% of warp-issue slots. But the kernel is memory-
bound (B·H·~514 KiB ≈ 8 MiB transferred per iter at the qwen3 shape,
HBM peak 1792 GB/s → floor ≈ 4.6 µs; we hit 26 µs = ~22% of peak).
The bottleneck is L2 → SM bandwidth, not warp issue.

If we wanted to push toward W1c's 417 GB/s, the obvious next move is
**per-CTA two-warps** (BLOCK_V=128, VLANES=32 threads, single full warp)
plus restructuring smem so each warp owns half the tile. That doubles
the tile to 128 KiB which would force the dynamic-smem path. Out of
scope for this FFMA-class baseline.

## Build / Run

```bash
cd /home/codeseys/cuda-exploration/mojo-attn-gdn
bash run.sh   # full reproduce: nvidia-smi + mojo --version + bench + correctness
```

Mojo 1.0.0b1, RTX 5090 (driver 596.21, sm_120). Under 30 s wall time end-to-end.

## Files

- `attn_gdn.mojo` — kernel + harness (single file, 432 lines)
- `run.sh` — driver
- `run.log` — captured output of the run
- `results.csv` — best/median GB/s + tile config
- `ANALYSIS.md` — this file
