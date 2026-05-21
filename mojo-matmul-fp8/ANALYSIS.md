# Wave 22.14 — `mojo-matmul-fp8`: Tiled e4m3 FP8 matmul at M=N=K=4096

**Status: ✅ Bench + correctness COMPLETE on first compile/run.**

Builds on Wave 22.4's bit-exact correctness baseline (M=N=K=32, single-block,
single-warp, 8 QMMAs). Scaled the kernel up to a full-size 4096³ tiled
matmul, mirroring the W21 bf16 scaffolding (tile + warp + K-loop + async
DRAM→SMEM copy) but with the m16n8k32 MMA shape and e4m3 inputs.

## Headline numbers (RTX 5090 / sm_120a, 4096³, Mojo 1.0.0b1)

| metric | value |
|---|---|
| TFLOPS_median | **113.42** |
| TFLOPS_best | **114.07** |
| min_ms / median_ms / max_ms | 1.205 / 1.212 / 1.770 |
| max_abs_err | **0.0** (bit-exact!) |
| max_rel_err | 0.0 |
| Tolerance applied | atol=2e-1, rtol=1e-1 (FP8-appropriate) |
| Correctness | **PASSED** (1024 sampled cells, full K=4096 CPU ref each) |
| QMMA.16832.F32.E4M3.E4M3 count in SASS | **16** (= 8 MMAs/K-step × 2 K-steps per inner loop) |
| HMMA count | 0 |
| `.target` | `sm_120a` |

## Comparison to baselines (same RTX 5090, same Mojo 1.0.0b1, same 4096³)

| lane | TFLOPS_median | vs Mojo bf16 (W21) | vs cuBLAS hgemm |
|---|---|---|---|
| **Mojo FP8 e4m3** (this wave, W22.14) | **113.4** | **+43%** | 51.7% |
| Mojo bf16 (W21) | 79.3 | (baseline) | 36.2% |
| Mojo f16 (W22.2) | 79.2 | -0.04% | 36.2% |
| cuBLAS bgemm/hgemm | 219 | +176% | (baseline) |

### Where the FP8 lift went

The task prompt sketched two mental models:

- **Bandwidth-bound regime**: FP8 gives 2× the input-bandwidth advantage of
  16-bit types, so a bandwidth-bound kernel should roughly double from
  79.3 → ~160 TF.
- **Compute-bound regime**: m16n8k32 has 2× the FLOPs-per-K-step of m16n8k16
  (same 16×8 output tile, twice the K span), so a compute-bound kernel
  with a fixed warp count should roughly double the same way.

What actually happened: **+43% (113.4 vs 79.3 TF)**, not +100%.

The shortfall vs the optimistic 2× projection is consistent with the kernel
being only **partially** improved by FP8:

1. **Smem load cost is halved.** A bf16 BK=32 tile is 64×32×2 = 4096 bytes per
   slab; this kernel's BK=64 slab is also 64×64×1 = 4096 bytes per slab.
   So per-K-element of work, the smem footprint is identical and the
   `cp.async` cost per tile is the same in absolute terms. The K-step count
   doubles for bf16 (BK/16=2 inner steps × M=4096/BK=128 outer = 256 total)
   vs FP8 (BK/32=2 × 4096/BK=64 = 128 total). Net: FP8 does **half** the
   smem-load passes, but each pass moves the **same** number of bytes and
   does **twice** the FLOPs. So the FLOP/byte ratio of the kernel doubled,
   which is the lift we got.
2. **Compute density per warp didn't double.** Both kernels do 8 MMAs per
   warp per inner pass (WM/MMA_M × WN/MMA_N = 2×4 in both cases), but each
   FP8 MMA does **2× the FLOPs** of a bf16 MMA at the same fragment slot.
   So absolute compute-issue throughput per warp doubled; SM occupancy
   (4 warps/block, 64 blocks at 4096³) is identical to W21 bf16.
3. **The K-loop trip count halved.** bf16 inner: BK/MMA_K = 32/16 = 2
   K-steps × 64 outer = 128. FP8 inner: 64/32 = 2 × 64 outer = 128. Same
   (we kept BK=64 here vs BK=32 in W21). So loop overhead and barrier
   count are identical; the only thing that changed is the dtype of the
   fragment loads + which MMA instruction was emitted.

So the +43% lift is real bandwidth-and-compute coupling, but we're hitting
a regime that's **neither purely bandwidth-bound nor purely compute-bound**:
something else is the limiter. Most likely candidates (**not** investigated
in this wave):

- **Register pressure.** FP8 A-frag is `SIMD[e4m3, 16]`, 16 bytes/lane vs
  bf16's 16 bytes/lane (`SIMD[bf16, 8]`). Register footprint per-lane is
  the same in bytes, but FP8 packs them differently — 4 groups of 4 e4m3
  vs 4 groups of 2 bf16 — and the MMA expects them in 32-bit-packed
  registers. Compiler may be issuing more shuffle/byte-extract instructions.
- **Async-copy throughput.** `cp.async` is byte-rate-limited; both kernels
  drive the same byte rate. FP8 doesn't help here unless we also widen
  BK to use the available time.
- **Smem bandwidth.** Same byte rate, fewer K-steps, so smem reads per
  block are roughly halved. Possibly slack on the smem side that more
  warps could absorb.

A natural follow-up: **try BK=128** (which fits since 64×128×1 = 8KB per
tile, well within smem capacity) and see if more compute per smem load
unsticks the limiter. That experiment is W22.15 territory.

### Why 0.0 max_abs_err at K=4096 is plausible (not a false positive)

The W22.4 ANALYSIS already documented this for K=32: A and B init values
are integer multiples of 0.0625 in [0, 0.9375]. Each product is a multiple
of 0.0625² = 1/256 in [0, 0.879]. Both the device kernel and the CPU
reference loop read from `a_host`/`b_host` which are bf16-cast e4m3 values,
so both consume the same already-quantized values.

At K=4096, the sum is **at most** 4096 × 0.879 ≈ 3601, and **mean** sum
~K × 0.47² × 0.5 ≈ 452. Multiples of 1/256 sum exactly in f32 as long as
the running sum stays below 2^24 / 256 = 65536; we're well under that.
So bit-exact (max_abs_err = 0.0) is the expected result here — different
summation order in the MMA vs the CPU loop is the only potential noise
source, and at this magnitude all partial sums are exactly f32-representable
so reordering doesn't change the result.

The atol=2e-1 + rtol=1e-1 tolerance is what we'd actually need on a
production-scale FP8 kernel where inputs are unconstrained e4m3 values —
quantization noise dominates Wilkinson rounding by orders of magnitude,
and per-product roundoff is ~12% (3-bit mantissa). We applied that
tolerance and beat it by 13+ orders of magnitude because the inputs are
representable exactly.

## Tile geometry (this wave)

- **BM = BN = 64, BK = 64** (vs W21 bf16's BK=32; m16n8k32 demands BK
  ≥ 32 to fit at least one K-step, and BK=64 gives 2 K-steps per inner
  pass at the same output footprint).
- **WM = WN = 32**, 4 warps/block (BM/WM × BN/WN = 2×2). 128 threads/block.
- MMA shape **m16n8k32**. Per-warp inner: 2×4 = **8 QMMAs per K-step**.
- Per-tile-pass K-loop: **BK/MMA_K = 2** outer K iterations. So per block
  per K-tile-pass: 8 × 2 = **16 QMMAs** (matches SASS count exactly).
- Outer K-loop: K/BK = 4096/64 = **64 iterations**.
- Grid: (4096/64)² = **64 × 64 = 4096 blocks**.
- Per-block FLOPs: 64 × 64 × 2 × 64 = 524288 (per K-tile-pass) × 64 K-iters
  = 3.36e7 FLOPs/block × 4096 blocks = 1.37e11 FLOPs/iter (matches the
  2·M·N·K = 1.374e11 we use for TFLOPS).

## Hand-rolled fragment loads (PTX 9.7.13.4.7, 8-bit m16n8k32)

The W22.4 formulas were reused **verbatim** since they're already validated
bit-exact at M=N=K=32, just applied per-MMA-position with proper smem
slicing:

```mojo
# A: SIMD[e4m3, 16] / lane = 4 sub-blocks × 4 K-elems each
# row_off = (sub & 1) * 8
# col_off = (sub >> 1) * 16
# row     = group_id + row_off
# col     = tid_in_grp * 4 + col_off + elem
comptime for sub in range(4):
    comptime for elem in range(4):
        var row_off = (sub & 1) * 8
        var col_off = (sub >> 1) * 16
        var row = group_id + row_off
        var col = tid_in_grp * 4 + col_off + elem
        a_frag[sub * 4 + elem] = A_mma_tile[row, col][0]

# B: SIMD[e4m3, 8] / lane = 2 sub-blocks × 4 K-elems each
# row = sub * 16 + tid_in_grp * 4 + elem
# col = group_id
comptime for sub in range(2):
    comptime for elem in range(4):
        var row = sub * 16 + tid_in_grp * 4 + elem
        var col = group_id
        b_frag[sub * 4 + elem] = B_mma_tile[row, col][0]
```

The C/D output epilogue (PTX 9.7.13.4.8, m16n8) is unchanged from the
bf16 path — same as W21.

## What changed vs Wave 22.4

| element | W22.4 (smoke) | W22.14 (this) |
|---|---|---|
| File | `matmul_fp8.mojo` (now `matmul_fp8_smoke.mojo`) | new `matmul_fp8.mojo` |
| M = N = K | 32 | **4096** |
| Block tile (BM, BN, BK) | (32, 32, 32) (single block) | **(64, 64, 64)** |
| Warp tile (WM, WN) | (32, 32) (whole block) | **(32, 32)** |
| Warps per block | 1 | **4** |
| Smem-load source | host-loaded once | **`copy_dram_to_sram_async` inside K-loop** |
| K outer loop | 0 (single K-step) | **64 iters** |
| MMAs per block | 8 | **16 per K-tile-pass × 64 = 1024 per block** |
| Total grid | 1×1 | **64×64 = 4096 blocks** |
| Bench harness | none | **10 iters × `execution_time[body](1)` → median** |
| Correctness | full M·N=1024 pairs (atol=1e-1, rtol=5e-2) | **1024 sampled pairs (atol=2e-1, rtol=1e-1)** |

The smoke kernel is preserved at `matmul_fp8_smoke.mojo` for future
correctness regression checks (it runs in seconds and reproduces 0.0 error
unconditionally).

## SASS structure (matmul_fp8.sass key fragments)

```text
.target sm_120a
.elftype @"ET_EXEC"
...
# 16 QMMA instructions in the inner block (8 per K-step × 2 K-steps);
# the K-outer loop is a runtime loop so we see only the unrolled body.
QMMA.16832.F32.E4M3.E4M3 R36, R20, R48, R36 ;
QMMA.16832.F32.E4M3.E4M3 R40, R20, R46, R40 ;
QMMA.16832.F32.E4M3.E4M3 R4,  R20, R44, R4  ;
QMMA.16832.F32.E4M3.E4M3 R32, R16, R48, R32 ;
QMMA.16832.F32.E4M3.E4M3 R28, R16, R46, R28 ;
QMMA.16832.F32.E4M3.E4M3 R24, R16, R44, R24 ;
QMMA.16832.F32.E4M3.E4M3 R8,  R20, R2.reuse, R8 ;
QMMA.16832.F32.E4M3.E4M3 R12, R16, R2,  R12 ;
QMMA.16832.F32.E4M3.E4M3 R36, R16, R44, R36 ;
QMMA.16832.F32.E4M3.E4M3 R32, R20, R44, R32 ;
... (16 total)
```

Note the C-input register is **non-RZ** (e.g. `R36, R40, R4, R32 ...`) —
that's the per-MMA accumulator carrying state across both inner K-steps
and the outer K-loop. This is the canonical "C is reused" SASS pattern,
and it's the right one for a tiled matmul (vs W22.4's smoke kernel which
showed `RZ` because there was only one K-step).

## Pitfalls hit and avoided

- **None hit on first compile/run.** The W22.4 fragment formulas + W21
  scaffolding combined cleanly; first compile produced a working binary
  with bit-exact correctness and a sensible TFLOPS number. The cleanest
  scale-up in the matmul series so far.
- **`thread_layout` for vec-loads matters more at FP8.** With FP8 (1 byte/elem)
  the natural vector width is 16 elements (16 bytes = 128 bits, the largest
  cp.async transaction). For a 64×64 tile that's 4 vec-cols × 64 rows = 256
  vec-loads, which split over 128 threads is 2 per thread. Used
  `thread_layout=Layout.row_major(32, 4)` (32 rows × 4 vec-cols = 128 threads).
  bf16 had `Layout.row_major(4, 8)` for 32 vec-loads / 8 threads-per-row
  with vec-width 4 (i.e. 4 bf16 = 8 bytes); the choice is dtype-dependent.
- **No `TensorCore.load_a` / `load_b` for FP8.** As in W22.4, we don't have
  evidence the wrapper validates `float8_e4m3fn` for fragment loads, so
  we hand-rolled. The W22.4 formulas being bit-exact at K=32 made this
  decision essentially free.

## What's NOT in scope for Wave 22.14

- **No BK sweep.** BK=64 was chosen as the minimum that fits 2 K-steps;
  BK=128 (4 K-steps, doubled smem footprint) is the obvious next experiment.
  Adjacent waves can take that on.
- **No comparison vs cuBLAS FP8** (`cublasGemmEx` with `CUDA_R_8F_E4M3`).
  cuBLAS FP8 typically targets sm_90 (Hopper) or sm_120 (Blackwell-DC) —
  consumer Blackwell sm_120a is best-effort, no published SOL number.
  The 51.7% of cuBLAS hgemm comparison above uses **half-precision** cuBLAS
  as the reference, not FP8 cuBLAS, so it's not a direct dtype comparison;
  it's a "fraction-of-the-machine" comparison.
- **No compile-time SASS analysis.** The QMMA count + .target line are
  the only SASS signals checked; deeper analysis (occupancy, register
  pressure, smem bank conflicts) is W22.15+ territory.

## Files (this wave)

| Path | Purpose |
|---|---|
| `matmul_fp8.mojo` | **NEW** — M=N=K=4096 tiled bench kernel (this wave) |
| `matmul_fp8_smoke.mojo` | RENAMED from old `matmul_fp8.mojo` — M=32 correctness baseline (W22.4) |
| `matmul_fp8.sass` | SASS dump from this wave's run (16 × QMMA, .target sm_120a) |
| `matmul_fp8.stderr` | Compile warnings (deprecations, no errors) |
| `mma_probe_fp8.mojo` | UNCHANGED — single-warp probe (W22.4) |
| `mma_probe_fp8.sass` | UNCHANGED |
| `run.sh` | UPDATED — runs probe + smoke + tiled bench |
| `ANALYSIS.md` | UPDATED — this file |

## Reproduce

```bash
cd /home/codeseys/cuda-exploration/mojo-matmul-fp8
./run.sh
```

Expected output tail:

```
[mojo-matmul-fp8] M=N=K= 4096  a_type=e4m3 c_type=f32  MMA= 16 x 8 x 32  BM= 64  BN= 64  BK= 64  min_ms= 1.205  median_ms= 1.212  max_ms= 1.770  TFLOPS_median= 113.4  TFLOPS_best= 114.1
[mojo-matmul-fp8] correctness: max_abs_err= 0.0  max_rel_err= 0.0  (atol= 0.2  rtol= 0.1 )
[mojo-matmul-fp8] correctness PASSED at M=N=K= 4096
```

QMMA count in SASS: **16**.
