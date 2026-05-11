# cuda-attn-gqa — Wave 15.1: nvcc CUDA C++ GQA (3-kernel naive, WMMA tensor cores)

**Date:** 2026-05-11
**GPU:** NVIDIA GeForce RTX 5090 (sm_120)
**Toolchain:** `/usr/local/cuda/bin/nvcc` (CUDA 13.2), `-ccbin clang-14`
**Build:** `/usr/local/cuda/bin/nvcc -ccbin clang-14 -O3 -arch=sm_120 -lstdc++ -lm -o attn_gqa attn_gqa.cu`

## What this is

Naive 3-kernel Grouped-Query Attention forward pass as the non-PyTorch
correctness oracle for Wave 15. Tensor cores engaged via WMMA
(`nvcuda::wmma`) at `m16n16k16`, f16 inputs, f32 accumulator, f16 output —
matching the cuBLAS hgemm configuration from Wave 14.1.

## Kernel structure

```
Q [B, Nq,  S, D]                   K [B, Nkv, S, D]
         \                                /
          \-- gqa_qkt_kernel (WMMA f16→f32) scores *= 1/sqrt(D)
                    ↓
              S [B, Nq, S, S] f32
                    ↓
             softmax_kernel (block-reduction, row-wise; f32→f16 out)
                    ↓
              P [B, Nq, S, S] f16
                    ↓                       V [B, Nkv, S, D] f16
                    \                      /
                     \-- gqa_pv_kernel (WMMA f16→f32, write f16)
                                ↓
                           O [B, Nq, S, D] f16
```

**GQA broadcasting** (`n_q=32, n_kv=8, groups=4`) is done implicitly:
each kernel computes `h_kv = h_q / groups` in-place — we never expand
K,V in memory. The entire KV cache touches 4 MB, not 16.

**Kernel 1 — QKt (WMMA).** Grid `(S/16, S/16, B*Nq)`, one warp per
16×16 output tile. A-fragment is Q-row (row-major, ld=D); B-fragment is
K viewed as col-major (`wmma::col_major`) with ld=D so we effectively
multiply by K^T without a transpose. Scale applied via fragment
element loop before `store_matrix_sync`.

**Kernel 2 — Softmax.** One block per `(B*Nq, row)`, 128 threads, two
reduction passes (max then sum of exp) in shared memory, then a write
pass that stores f16. Using f16 output means kernel 3 can consume P as
a WMMA matrix_a directly.

**Kernel 3 — PV (WMMA).** Grid `(S/16, D/16, B*Nq)`. Both A (P) and B
(V) are row-major. Accumulator is f32, cast to f16 on store via a
shared-memory staging tile.

## Tensor-core verification

```
$ /usr/local/cuda/bin/cuobjdump --dump-sass attn_gqa | grep -c HMMA
20
$ /usr/local/cuda/bin/cuobjdump --dump-sass attn_gqa | grep HMMA | head -1
/*0c80*/    HMMA.16816.F32 R20, R12, R4, R20 ;
```

20 `HMMA.16816.F32` instructions total (across the two matmul kernels,
counting the unrolled variants nvcc emits). `.16816.F32` is exactly the
m16n16k16 f16→f32 shape we asked for. ✅ Tensor cores engaged.

## Correctness

Shape `b=1, s=128, n_q=4, n_kv=2, d=64`. Compared f16 output against
PyTorch reference (`gqa_naive` in `pytorch_reference.py`) expected-f32
tensor from `inputs/gqa_correctness_expected_f32.npy`.

```
max_abs_err = 1.559e-04
max_rel_err = 3.095e+01    (denominator near zero on one element)
expected_max_abs = 2.162e-01
tolerance (f16) = atol 5e-3, rtol 5e-3
result: OK (combined |e| ≤ atol + rtol*|y| passes for all elements)
```

The large `max_rel_err` is a small-denominator artifact (one output
element is ~5e-6 where f16 rounds to a slightly different value); the
absolute error ceiling of 1.6e-4 is ~30× tighter than the 5e-3 f16
tolerance, so we pass comfortably.

## Performance (Llama-3-8B shape)

`b=1, s=2048, n_q=32, n_kv=8, d=128`. FLOPS per iter from
`reference/flops.py`: `4 * B * Nq * S^2 * D = 68.72 GFLOPS`.

| stat    | gpu_ms | TFLOPS |
| ------- | -----: | -----: |
| best    | 2.929  |  23.46 |
| median  | 2.939  |  23.38 |
| worst   | 3.448  |  19.93 |

**Reference ceiling:** cuBLAS hgemm @ N=4096 = 218 TFLOPS (Wave 14.1).
Our 3-kernel naive GQA reaches **23.46 / 218 = 10.8 %** of that peak.

## Where the 89 % is going

At this early MVP stage the gap is expected and attributable to:

1. **No fusion.** Three kernels mean S and P (f32 scores, f16 probs,
   each 268–537 MB for the bench shape) round-trip through HBM. A fused
   FlashAttention keeps them in SRAM and eliminates most of this
   traffic. Planned for Wave 15.5.
2. **One warp per 16×16 tile** — tiny tiles have poor compute/memory
   overlap. cuBLAS hgemm uses 128×128 or larger output tiles with many
   WMMA instructions per block.
3. **No shared-memory tiling.** Every WMMA load hits HBM directly; we
   don't double-buffer or cooperative-load a larger tile per block.
4. **Softmax is a 3-pass kernel** (max, sum, normalize) that rereads
   the full scores tensor three times; the online softmax (Flash-style)
   would fuse it and cut bandwidth by ~3×.

All four are MVP-concessions that the fused Wave 15.5 kernel is designed
to fix. The current role of this cell is the **non-PyTorch correctness
oracle**, which it delivers: tensor-core engagement, byte-identical
output format, sub-tolerance match against the f32 PyTorch reference.

## Pitfalls hit

- **Linker needed `-lm` explicitly.** On this Ubuntu 24.04 + clang-14
  setup, nvcc's default link flags don't drag in libm, so `sqrtf` in
  device-side `expf` compiled fine but `sqrtf` called from host code
  failed to link. Added `-lm` to Makefile.
- **WMMA col_major for K^T.** To avoid materializing K^T, loaded K as
  `matrix_b` with `col_major` layout and ld=D. This gave the right
  logical shape `(K=D) × (N=S)` for the B fragment.
- **f16 probabilities → f16 output of softmax.** Writing P as f16
  lets kernel 3 use P as a WMMA `matrix_a` directly. At correctness
  tolerance this is lossless; the reference f32 output matches.
- The `max_rel_err` metric is misleading when expected values hover
  near zero; absolute error is the useful signal here.

## Files

- `attn_gqa.cu` — 3 kernels + bench driver + NPY reader (~410 LOC)
- `Makefile`
- `.gitignore`
- `results.csv` — per-iter timings at llama3_8b shape
- `run.log` — captured stdout
