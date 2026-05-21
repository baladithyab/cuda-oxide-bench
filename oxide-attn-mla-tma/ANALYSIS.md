# oxide-attn-mla-tma — Wave C3.3 analysis

**Cell:** `cuda-exploration/oxide-attn-mla-tma/`
**Wave:** C3.3 (oxide-attn-mla-tma) — author + correctness only.
**GPU:** RTX 5090, sm_120, CUDA 13.2, libNVVM 22.0.0.

## What this cell is

cuda-oxide MLA attention with TMA-loaded Q/K/V tiles. Lifts the C2.4
oxide-attn-gdn-tma recipe (host-side `cuTensorMapEncodeTiled` +
`transmute_copy::<CUtensorMap, [u8;128]>` + `DeviceBuffer::from_host`;
device-side `cp_async_bulk_tensor_2d_g2s` + `Barrier` /
`mbarrier_arrive_expect_tx` / `mbarrier_try_wait`) into the 3-kernel MLA
decomposition:

- `mla_qkt_kernel` (TMA-loaded Q + K tiles)
- `softmax_kernel` (byte-for-byte copy from oxide-attn-mla; no TMA)
- `mla_pv_kernel`  (TMA-loaded V tile; P stays as cooperative LDG)

The FFMA microtile (16×16 threads, each computing a 4×4 output, K-tile=16)
and the launch grids are unchanged from the FFMA baseline — only the gmem
tile loads of Q/K/V are replaced with TMA. This isolates the change to the
load path, matching the task's "author + correctness" scope.

## TMA descriptors (3, encoded host-side once)

All three are 2D row-major f32 tensors with `globalDim = [inner, outer]` in
TMA's innermost-first convention.

| Descr | Tensor view             | inner | outer    | box (inner, outer) | Issued by |
|-------|-------------------------|-------|----------|--------------------|-----------|
| Q     | (B*n_h*S, qk)           | qk    | B*n_h*S  | (16, 64)           | qkt       |
| K     | (B*n_h*S, qk)           | qk    | B*n_h*S  | (16, 64)           | qkt       |
| V     | (B*n_h*S, d_v)          | d_v   | B*n_h*S  | (64, 16)           | pv        |

Per-block coords (in elements):
- Q: `coord_x = k_off`, `coord_y = bh*S + by*64`
- K: `coord_x = k_off`, `coord_y = bh*S + bx*64`
- V: `coord_x = bx*64`, `coord_y = bh*S + k_off`

The descriptors are encoded once on the host with `cuTensorMapEncodeTiled`,
copied into a `DeviceBuffer<u8>` of length 128 via `transmute_copy::<CUtensorMap,
[u8;128]>`, and passed to the kernel as a `*const TmaDescriptor`.

## SMEM layout match — the key correctness pivot

The FFMA baseline kept K's smem tile **transposed** (`TILE_K[kk*64+cc] =
K[col0+cc, k_off+kk]`) so the inner FFMA loop could read K with the same kk
that drove Q. TMA cannot transpose during a load.

Solution: TMA-load K **naturally** as 64 rows × 16 cols
(`TILE_K[r*16+kk] = K[col0+r, k_off+kk]`, same shape as Q's tile), and rewrite
the inner FFMA's K read from `TILE_K[kk*64 + tx4+i]` to `TILE_K[(tx4+i)*16 +
kk]`. Same data, swapped indexing — no layout transform, no extra smem traffic.

Q and V tile layouts already matched their FFMA-baseline smem layouts
(64×16 row-major and 16×64 row-major respectively), so neither needed any
inner-loop changes.

## Acceptance results

```
SHAPE_CORRECTNESS: B=1 n_h=4 S=128 qk=96 d_v=64
correctness_mla correctness: max_abs=1.192e-7 max_rel=9.017e-4 (atol=0.01) -> PASS
```

`max_abs = 1.192e-7` is well under the 1e-2 acceptance bound and matches
f32-on-f32 round-off noise (the FFMA baseline's max_abs at this shape was
similar). The change touches only the load path; the arithmetic is
bit-identical to the FFMA baseline.

### SASS UTMALDG counts (per-kernel)

```
  mla_pv_kernel                            1
  mla_qkt_kernel                           2
  softmax_kernel                           0
```

Total: **3 UTMALDG** instructions across the 3 kernels — exactly the spec
("qkt: 2 UTMALDG for Q+K; pv: 1 UTMALDG for V"). softmax_kernel correctly
contains zero (pure per-row reduction, no TMA).

### Auxiliary SASS counts

```
HMMA insts  : 0     (no tensor cores; cuda-oxide has no usable WMMA on sm_120)
FFMA insts  : 349   (FFMA microtile preserved)
LDG.E insts : 19    (down from 31 in the FFMA baseline — Q/K/V loads now TMA)
LDS  insts  : 52    (smem reads in the FFMA inner loop)
```

The LDG.E delta confirms that Q (qkt), K (qkt), and V (pv) global loads
were successfully replaced with TMA. The residual LDG.E come from P (probs)
loads in pv_kernel and small scalar boilerplate.

## Expected-TFLOPS estimate (no timed bench)

Per task: optional bench skipped; algorithm pattern is identical to the
W23.2 cuda-attn-mla-tma C++ reference. The TMA path's win in C2.4 was a
3.7× single-kernel speedup over its FFMA baseline (276 GB/s → 1032 GB/s).
The MLA bench shape (B=1, S=2048, n_h=128, qk=192, d_v=128 — DeepSeek-V3)
is ~50 TFLOPS of useful compute.

The FFMA baseline (oxide-attn-mla) hits ~24 TF on the GQA-equivalent shape
(per Wave 16.1 / Wave 17 W1b notes). Same FFMA microtile here, with TMA
replacing the gmem load path. Realistic expectation:

- **No tensor-core uplift** (cuda-oxide has no usable WMMA on sm_120 — the
  W23.2 C++ reference uses WMMA for ~4-5× over FFMA; we cannot match that).
- TMA reduces the load issue rate (1 cp.async.bulk per tile instead of 4
  cooperative LDG.E per warp) but the FFMA loop itself remains bound by
  scalar throughput. **Expected uplift: 1.0×–1.3× over FFMA-baseline MLA**,
  i.e. ~24–32 TFLOPS at the DeepSeek-V3 bench shape, NOT WMMA-class
  throughput. The "no-TC ceiling" framing from Wave 17 still applies.
- The TMA path's bigger payoff is for memory-bound kernels (GDN's 3.7×).
  MLA at the bench shape is compute-bound on FFMA, so TMA's load-rate
  savings are mostly absorbed by the FFMA inner loop's already-amortized
  smem traffic.

A timed bench across the DeepSeek-V3 shape would settle this; per the task
spec ("Optional bench. … no timed bench needed; W23.2 in C++ uses identical
algorithm pattern") we leave it as future work.

## Pitfalls encountered / things that bit

1. **K tile shape vs FFMA-baseline transposed layout.** The original cell
   loaded K transposed into smem so the FFMA inner loop could share `kk`
   with Q. TMA cannot transpose. Solution above (natural-load + swapped
   index inside the FFMA loop) is the cleanest fix; index swap is the
   *only* device-code change to the inner loop relative to the FFMA
   baseline. This mirrors the W23.2 cell's resolution of the same issue.

2. **Three descriptors, three pointers — but only two TMAs per qkt block.**
   The qkt kernel issues both Q and K TMAs in the same block iteration,
   sharing a single barrier with `arrive_expect_tx(total_bytes = Q_bytes +
   K_bytes)`. This avoids two separate `try_wait` spins per K-tile. Same
   pattern the W23.2 C++ reference uses.

3. **No bench shape exercised.** The task's acceptance shape (B=1, n_h=4,
   S=128, qk=96, d_v=64) is small enough that 1 warmup iter suffices for
   correctness. The full bench shape (DeepSeek-V3) was deferred per the
   "author + correctness only" scope.

4. **WMMA gap (NOT hit, documented).** The task flagged a possible WMMA
   fragment-vs-TMA-tile-layout mismatch as a stop condition. cuda-oxide
   v0.1.0 has **no usable WMMA / mma.sync API on sm_120** (Wave 14.4
   finding: zero `mma.sync` in cuda-oxide source; wgmma is a placeholder;
   tcgen05 is sm_100a-only). So this cell is FFMA-microtile + TMA-load,
   never WMMA-fragment + TMA-tile. The W23.2 C++ cell uses WMMA and had
   to solve fragment/tile alignment; we side-step that entirely by staying
   on FFMA. This is the documented "no-TC ceiling" path.

5. **Cargo.toml edition 2024.** The C2.4 cell uses edition 2024 (required
   for `&raw mut` and a few of the cuda-oxide bindings). The FFMA baseline
   used edition 2021. We bumped to 2024 to match the GDN-TMA pattern.

## Files written

- `oxide-attn-mla-tma/Cargo.toml` (edition 2024, cuda-oxide deps)
- `oxide-attn-mla-tma/src/main.rs` (3 kernels + host driver, ~700 lines)
- `oxide-attn-mla-tma/run.sh` (build + run + per-kernel UTMALDG SASS report)
- `oxide-attn-mla-tma/rust-toolchain.toml` (carried over from baseline)
- `oxide-attn-mla-tma/ANALYSIS.md` (this file)

Build/run artifacts (regenerable):
- `build.log`, `run.log`
- `oxide_attn_mla_tma.cubin` / `.sass` / `.ll` / `.ltoir`
- `results.csv`
