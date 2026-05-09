# Wave 2: cuBLAS baseline + tiled matmul

**Status:** queued (depends on Wave 1 results)
**ADR refs:** 0001, 0002, 0003
**Subagents:** 3 in parallel
**Budget:** ~45 min wall-clock, ~100k summary tokens
**Acceptance test:** `cublas-matmul/` and `oxide-matmul-tiled/` and `cuda-matmul-tiled/` exist with run.log + ANALYSIS.md per folder; results aggregated.

## File-ownership table

| Subagent | Owns (writes) | Reads |
|---|---|---|
| W2A — cuBLAS | `cublas-matmul/` (entire new folder) | docs/research/cublas-setup.md (verbatim sample provided there) |
| W2B — tiled CUDA C++ | `cuda-matmul-tiled/` (entire new folder) | cuda-matmul/matmul.cu (template) |
| W2C — tiled cuda-oxide | `oxide-matmul-tiled/` (entire new folder), regen its `*.ptx` | oxide-matmul/src/main.rs (template), cuda-matmul-tiled/matmul.cu (algorithm reference) |

All three are file-disjoint, full parallel.

## W2A — cuBLAS sgemm baseline

**Goal:** Materialize the sample matmul.cu from `docs/research/cublas-setup.md` into `cublas-matmul/`. Run it. Document.

**Steps:**
1. `mkdir -p cublas-matmul`; `cp` the sample matmul.cu from the research doc.
2. Build: `/usr/local/cuda/bin/nvcc -ccbin clang-14 -O3 -arch=sm_120 -lcublas -o cublas-matmul/matmul cublas-matmul/matmul.cu`.
3. Run for N ∈ {1024, 2048, 4096}. Log `gpu_ms`, TFLOPS, cuBLAS version, math mode.
4. Set math mode to `CUBLAS_PEDANTIC_MATH` (per ADR-0003, fair f32 comparison; no TF32).
5. Write `cublas-matmul/ANALYSIS.md` with: methodology, results table, expected vs measured (research doc said 60-90 TFLOPS; if we get ~150+ TFLOPS something is wrong, likely TF32 path slipped in).
6. Add `cublas-matmul/results.csv` with same columns as the W1 CSVs.

**Acceptance:** TFLOPS in the 50-100 range for N=4096 (anything outside means math mode wrong); spot-check correct; ANALYSIS.md written.

## W2B — tiled CUDA C++ matmul

**Goal:** Add `cuda-matmul-tiled/matmul.cu` with a 16×16 shared-memory tiled SGEMM. Bench at the same sizes.

**Algorithm:** Standard tile-and-reduce. Each block computes a 16×16 output tile by streaming K/16 tiles of A and B through shared memory.

```c
__shared__ float sA[16][16];
__shared__ float sB[16][16];
// per tile: load sA, sB, __syncthreads(), inner-product accum, __syncthreads()
```

**Steps:**
1. Write the tiled kernel.
2. Build with same nvcc flags as W1B (sm_120 native).
3. Bench N ∈ {1024, 2048, 4096}, 1 warmup + 10 iters, cudaEventRecord.
4. Write `cuda-matmul-tiled/ANALYSIS.md` with: kernel source quoted, tradeoff (4096 elements / 16-wide tile = 256 tiles per row, perfect divisibility), expected ~25-40 TFLOPS.
5. results.csv same shape.

**Acceptance:** TFLOPS for N=4096 ≥ 4× the naive nvcc number from W1B (i.e., ≥ 25 TFLOPS). If less, the kernel is wrong.

## W2C — tiled cuda-oxide matmul

**Goal:** Same algorithm as W2B but in cuda-oxide. Exercises `SharedArray<f32, 256>` API.

**Implementation hints:**
- See `crates/rustc-codegen-cuda/examples/sharedmem/src/main.rs` for the SharedArray usage pattern (cuda-oxide source). Two kernels there: `shared_test` and `shared_dual`. They show `static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;` + `unsafe { TILE[tid] = ...; thread::sync_threads(); }`.
- Define `TILE_A: SharedArray<f32, 256>` and `TILE_B: SharedArray<f32, 256>` (256 = 16*16).
- Mirror the C++ tiled algorithm. Write the safe version with slice indexing AND an unchecked version with raw pointers.
- Block size: 16×16, same as W2B.
- Use cudaEvent timing same as W1A.
- Sizes: same as W2B.

**Acceptance:** at least the unchecked variant lands ≥ 50% of W2B's TFLOPS. (If it lands at 100%, surprising; if at 20%, look at what `SharedArray` lowered to in PTX.)

## Wave 2 Concurrent reviewers (1 reviewer, mid-risk batch)

Single cross-family reviewer (per run-discipline §2): one of {gpt-5.5, gemini-3-pro, kimi-k2.6}. Different family from any used in Wave 1's reviewer. Reviews:
- All three folders' run.log timestamps are post-W2-start.
- TFLOPS values for tiled are ≥ 4× naive (sanity check).
- cuBLAS results aren't accidentally TF32.
- ANALYSIS.mds quote actual code, not generic descriptions.

## Reflexion checklist

- [ ] Three new folders committed
- [ ] BACKLOG.md updated: F1, F2 (compiler gaps) deferred to W3 (cuBLAS+tiled was higher value first); C1, C2 → ✅
- [ ] Aggregated results: oxide tiled vs nvcc tiled vs cuBLAS = three new rows in results/scaling.csv
- [ ] AGENTS.md note: any cuda-oxide pitfalls discovered during SharedArray work
- [ ] Push
