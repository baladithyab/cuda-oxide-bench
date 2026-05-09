# Wave 2 W2C: Tiled cuda-oxide Matmul — Analysis

## SharedArray API used

Per the upstream `sharedmem` example, block-level shared memory in cuda-oxide is
declared as a `static mut` with the `SharedArray<T, N>` type and initialized to
`SharedArray::UNINIT`. Indexed reads and writes require `unsafe`.

```rust
use cuda_device::{SharedArray, kernel, thread};

#[kernel]
pub fn matmul_tiled_unchecked(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>, dim: u32) {
    static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;
    // ... 16×16 block, ty*16+tx indexing, tile over K in steps of 16 ...
    unsafe {
        let a_val = *a_base.add(row_us * dim_us + a_col);
        let b_val = *b_base.add(b_row  * dim_us + col_us);
        TILE_A[local] = a_val;
        TILE_B[local] = b_val;
    }
    thread::sync_threads();          // load → compute barrier
    let mut k: usize = 0;
    while k < 16 {
        unsafe { acc += TILE_A[ty_us*16 + k] * TILE_B[k*16 + tx_us]; }
        k += 1;
    }
    thread::sync_threads();          // compute → next-tile barrier
}
```

Thread intrinsics: `thread::threadIdx_x/y()`, `thread::blockIdx_x/y()`,
`thread::sync_threads()`. Block is `(16, 16, 1)`; grid `⌈N/16⌉²`. Per tile each
thread stores exactly 1 f32 to each of TILE_A / TILE_B, then loops K=16 using
shared reads.

## Results (own run)

| kernel                 |    N | best gpu_ms | median gpu_ms | best TF | median TF |
|------------------------|-----:|------------:|--------------:|--------:|----------:|
| oxide-tiled safe       | 1024 |       0.240 |         0.243 |   8.95  |   8.83    |
| oxide-tiled unchecked  | 1024 |       0.233 |         0.238 |   9.21  |   9.02    |
| oxide-tiled safe       | 2048 |       1.780 |         3.092 |   9.65  |   5.56    |
| oxide-tiled unchecked  | 2048 |       1.751 |         2.602 |   9.81  |   6.60    |
| oxide-tiled safe       | 4096 |      16.962 |        17.878 |   8.10  |   7.69    |
| oxide-tiled unchecked  | 4096 |      16.120 |        17.295 |   8.53  |   7.95    |

Correctness: 3/3 at (0,0), (N/2, N/2), (N−1, N−1) for every (kernel, N). All 60
CSV rows written; 10 timed iters per kernel per N.

**Acceptance: FAILED.** Target was ≥12 TFLOPS at N=4096 unchecked; we got 8.53
best / 7.95 median. The deficit is compiler-side, not algorithmic (PTX analysis
below).

## Comparison — naive oxide vs tiled oxide vs cuda-tiled

At N=4096, best gpu_ms:

- `oxide` (naive, W1A):         ~20 ms  → ~6.7 TF
- `oxide-tiled` unchecked:      16.12 ms → 8.53 TF  (~1.27× naive)
- `cuda-tiled` (nvcc, W2B):      3.64 ms → 37.8 TF best; median ~29 TF

The cuda-oxide tiled implementation is **~4.4× slower** than the equivalent
nvcc-compiled tiled kernel at N=4096. The tile algorithm still gives a real
speedup over naive (1.27×) because shared-memory reuse cuts DRAM traffic by
BS=16, but the inner-loop codegen leaves most of that benefit on the floor.

## PTX examination

oxide-tiled (`oxide_matmul_tiled.ptx`, 263 lines, 2 entries):

- `st.shared.b32`: **4** (2 per kernel — one per tile store, inside the outer
  tile loop; not unrolled)
- `ld.shared.b32`: **4** (2 per kernel — one pair inside the K-loop, not
  unrolled)
- `bar.sync 0`: **4** (2 per kernel — load/compute fence and compute/next-tile
  fence)
- `fma.rn.f32`: **0**. K-accumulate uses `mul.rn.f32` + `add.rn.f32`
  (2 each per kernel).

cuda-tiled (`matmul.ptx`, nvcc sm_120):

- `ld.shared`: **128** (fully unrolled K=16 across 2 entries and both operands)
- `bar.sync`: **2**
- `fma.rn`: **256** (inner FMA chain fully unrolled)

So the shared-memory machinery is correct in the oxide output — the
`SharedArray` API lowers to `st.shared`/`ld.shared` against named
`__shared_mem_0` / `__shared_mem_1` symbols, and `sync_threads()` lowers to
`bar.sync 0`. Both fences fire in the right places. What's missing:

1. **No FMA contraction.** cuda-oxide sets
   `FastmathFlags::default()` (empty) everywhere (see repo
   `docs/research/cuda-oxide-flags.md`), so `mul + add` on consecutive lines
   never fuses. This alone halves peak FP32 throughput vs nvcc.
2. **K-loop not unrolled.** nvcc emits 16 back-to-back FMAs with no branch
   between; oxide emits a 7-instruction loop body (2 `ld.shared`,
   `mul`, `add`, 3 pointer bumps, 1 `bra.uni`) — 2 branches per inner
   iteration are serialization and scheduling hazards.

Both issues are upstream-compiler gaps, not fixable from kernel source. The
tiled algorithm, shared-memory usage, and barriers are all structurally
correct; cuda-oxide today just can't exploit them.

cf. `cuda-matmul-tiled/results.csv` and its ANALYSIS.md for the nvcc reference.
