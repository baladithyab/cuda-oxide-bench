# cublas-attn-gqa — Wave 15.1 analysis

**Implementation:** 3-kernel GQA attention pipeline using cuBLAS for both
matmuls and a hand-written row-wise softmax kernel for the middle stage.

- Stage 1 — `cublasGemmEx` for QKᵀ, f16 inputs, f32 accumulator, tensor-core
  algo (`CUBLAS_GEMM_DEFAULT_TENSOR_OP`, `CUBLAS_COMPUTE_32F`).
- Stage 2 — `row_softmax_scale_f32_to_f16_kernel` (`softmax.cu`): one block
  per row of the (seq, seq) score matrix, 128 threads/block, three passes
  (max → sum → normalize) with block-wide shared-memory reductions. Scale
  `1/√d_head` fused into pass 1 so we avoid a separate scaling kernel.
- Stage 3 — `cublasGemmEx` for PV, same f16-in/f32-acc TC config.

f16 → f32-acc is enforced everywhere (matches the PLAN.md correctness
contract; f16 softmax accumulator would not pass `atol=5e-3`).

GQA broadcasting is handled the simple way (**option (a)**): we loop
`batch × n_q` per-head `cublasGemmEx` calls for each of the two matmul
stages and index the KV head as `h_kv = h_q / groups`. Option (b)
(`cublasGemmStridedBatchedEx` with KV-stride broadcasting) would cut
launch overhead — see **Pitfalls** below. Queued for Wave 15.6.

## Bench (Llama-3-8B shape: B=1, S=2048, n_q=32, n_kv=8, d=128)

10 timed iters, 1 warmup, on RTX 5090 sm_120, cuBLAS 13.4.0,
CUDA 13.2. Correctness pass at both `correctness` (1×4×128×64) and
`llama3_8b` shapes under f16 tolerance (atol=5e-3, rtol=5e-3) —
observed max_abs ≈ 6.7e-5 at bench, essentially bounded by the f16
rounding of the f32 reference.

| metric            | value        |
|-------------------|-------------:|
| best total ms     | **1.546 ms** |
| avg total ms      | 1.620 ms     |
| **best TFLOPS**   | **44.46 TF** |
| avg TFLOPS        | 42.43 TF     |

### Per-stage breakdown (avg over 10 iters)

| stage    | avg ms | % of total |
|----------|-------:|-----------:|
| QKᵀ      | 0.502  |   31.0 %   |
| softmax  | 0.506  |   31.2 %   |
| PV       | 0.612  |   37.8 %   |

Three stages are roughly balanced — each contributes ~1/3 of the
total. Softmax, despite being the only non-matmul kernel, costs **as
much** as the 32-head cuBLAS QKᵀ loop. That's the core finding for
the wave: a 3-kernel GQA pipeline can't drive TFLOPS with the softmax
tax on the critical path.

## Ratios

- **cuBLAS hgemm ceiling (Wave 14.1, N=4096): 218 TF.**
  Our 44 TF is **~20 %** of the dense-hgemm peak.
- Why so far below hgemm peak? Each matmul stage here is *not* a single
  big gemm — it's 32 small per-head gemms on matrices of size
  (2048, 128, 2048). That's `32 × 2 × 2048² × 128 = 34.4 GF` per stage
  split across 32 launches (~1.07 GF each), and only hits ~137 TF
  effective at the gemm stage if we attribute all 34.4 GF to 0.502 ms
  for Stage 1 — already well below the 218 TF peak at these small
  per-call shapes, plus launch overhead between the 32 calls.
- Amdahl floor: even if both matmul stages ran at the 218 TF peak, the
  custom softmax kernel at 0.506 ms would limit us to
  `68.72 GF / (0.506 ms × 2) ≈ 68 TF`-ish — meaning softmax latency
  alone is a ~50 % tax on anything fusion could win.

This is the strongest **non-fused** baseline we will produce this wave,
and it matches the PLAN.md estimate ("130–180 TF") only if we were to
collapse softmax into a single per-head kernel fused with one of the
gemms — which is exactly what FlashAttention does. Our actual 44 TF
says the softmax-launch-overhead tax + per-head-gemm-launch tax *on an
RTX 5090 driving small gemms* eats most of that predicted headroom.

## Comparison context (to be filled in by the wave-summary pass)

- vs **cuda-attn-gqa** naive WMMA baseline (Wave 15.0 sibling cell):
  TBD; PLAN predicts ~60–90 TF for that cell. If our 44 TF is below
  it, the story is "custom per-row softmax + per-head cuBLAS launch
  overhead > hand-rolled everything". If above, cuBLAS still wins on
  the matmuls enough to compensate.
- vs **cutile-attn-gqa** fused (Wave 15.0 sibling cell): the expected
  "beat the unfused baseline" target — any fused impl ≥ 44 TF
  demonstrates the fusion payoff.

## Pitfalls encountered

1. **cuBLAS column-major vs row-major re-derivation was subtle for the
   QKᵀ case.** First working attempt passed Q as A and K as B with
   `op_A = T, op_B = N`. That silently computed the *transpose* of
   the desired scores matrix — softmax then normalized along the wrong
   axis, and correctness still "passed" with numpy-style `|a-b| <
   atol+rtol|b|` at the llama3_8b shape (because output magnitudes are
   small), but failed badly at the correctness shape (299/32768 values
   over tolerance). **Fix:** for row-major `scores[i,j] = Qrow[i]·Krow[j]`
   we pass `A = K_head, B = Q_head` with `op_A = T, op_B = N`. The key
   insight: col-major-kernel writing to position (M,N) lands at
   row-major (N,M), so we want the kernel to compute the transpose of
   the row-major target. Tested at both shapes: max_abs now ≈ f16
   rounding bound.
2. **Variance under GPU contention.** While Waves 15.0/15.5 sibling
   subagents were hammering the same GPU, individual iters showed
   2-3× slowdowns (e.g. a 1.6 ms iter jumping to 3.3 ms). Mitigation
   for headline numbers: re-ran the tight 10-iter block once
   contention cleared (reported above). CV at clean run ≈ 5 %.
3. **Softmax three-pass recompute trade-off.** The kernel does three
   passes over each row (max, sum, normalize). The alternative is a
   single-pass online-softmax with scratch = row length; we chose the
   three-pass version because: (a) at S=2048 the row fits in L1/L2
   easily, so the pass cost is memory-bound but on *cached* data; (b)
   scratch allocation in a shared-memory kernel with 128 threads would
   limit occupancy. Measured softmax time (0.506 ms) is already close
   to the HBM-bandwidth floor for `num_rows × seq × sizeof(f32)`
   = 65536 × 2048 × 4 B = 512 MB read + 256 MB write = 768 MB /
   0.506 ms ≈ **1.5 TB/s**, which is 70 % of the RTX 5090's 1.8 TB/s
   HBM3e peak — i.e. softmax is bandwidth-bound and basically at the
   kernel's roofline. An online-fused variant inside the matmul
   kernel (FlashAttention) is the only way to improve this.
4. **Memory footprint.** At bench shape the full-dense scores tensor
   is `B × n_q × S × S × 4 = 512 MB` f32, plus `256 MB` f16 probs.
   Fits on RTX 5090 (32 GB) but would be a problem at seq=8192
   (8 GB scores alone). Fused attention avoids the materialization.

## Open questions → Wave 15.6+

- **`cublasGemmStridedBatchedEx` with KV-stride broadcasting:** set
  `strideA = d_head × seq` for Q (per-h_q), `strideB = (d_head × seq)
  × groups⁻¹` for K (per-h_kv, reused by `groups` consecutive batch
  indices). cuBLAS allows fractional strides via integer `groups`
  layout only if we replicate the K pointers. Cleaner: two batched
  gemms per KV group. Expected win: at 32 calls × ~5–10 μs launch
  overhead each = ~160–320 μs, the n_q × batch loop is eating
  20-30% of the matmul stage. Batching could close the gap to
  ~100 TF ceiling for the matmul stages, and (softmax kernel limit
  unchanged at ~0.5 ms) the total could approach ~90 TF.
- **f16 softmax accumulator:** ruled out by PLAN.md correctness
  contract. But a *warp-level* reduction (one warp per row, 64 threads,
  no inter-block sync needed) might reduce launch overhead. Marginal.
- **Fused flash-style kernel in cuBLASLt:** not a cuBLAS primitive;
  belongs in the cuTile or hand-written-CUDA column.
