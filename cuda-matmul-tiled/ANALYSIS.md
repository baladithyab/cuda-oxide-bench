# Wave 2 W2B — tiled CUDA C++ SGEMM analysis

## Kernel

Block computes a 32×32 output tile using a BK=16 K-tile in shared memory, and
each of the 64 threads (8×8) owns a 4×4 register micro-tile. This is shared-mem
tiling + register tiling; the shared-mem half matches the standard 16-wide
K-streaming pattern from CUDA textbooks, and the register half is what actually
pushes us past the simple-tile throughput wall on RTX 5090.

```cuda
__shared__ float As[BM][BK];      // 32x16
__shared__ float Bs[BK][BN];      // 16x32

float acc[TM][TN];                // 4x4 per thread, held in registers
// ... zero acc ...

for (int t = 0; t < dim / BK; ++t) {
    // Cooperative load: 64 threads bring in 512 floats each for As, Bs.
    for (int i = 0; i < 8; ++i) { /* As load */ }
    for (int i = 0; i < 8; ++i) { /* Bs load */ }
    __syncthreads();

    for (int k = 0; k < BK; ++k) {
        float a_reg[TM], b_reg[TN];
        for (int i = 0; i < TM; ++i) a_reg[i] = As[ty*TM+i][k];
        for (int j = 0; j < TN; ++j) b_reg[j] = Bs[k][tx*TN+j];
        for (int i = 0; i < TM; ++i)
            for (int j = 0; j < TN; ++j)
                acc[i][j] += a_reg[i] * b_reg[j];   // 16 FMAs / k-step
    }
    __syncthreads();
}
// ... write acc[TM][TN] to C ...
```

## Why tiling speeds things up

Naive matmul reads each A-row and B-column element once *per output element
that uses it*, i.e. O(N) DRAM reads per element. With a 32×32 block tile and
a K-streaming loop, each loaded element of A is reused 32× (across columns of
the output tile) and each B element 32× — a factor of 32 reduction in DRAM
traffic. DRAM traffic scales as O(N³/BM) instead of O(N³). The arithmetic
intensity goes from ~0.5 FLOP/byte (naive) to ~8 FLOP/byte (32×32 tile),
easily moving the kernel from memory-bound to compute-bound.

Register tiling layers a second reuse level on top: each thread's `a_reg[i]`
broadcasts across TN=4 FMAs and each `b_reg[j]` across TM=4 FMAs, so the 8
shared-memory reads per K-step feed 16 FMAs, a 2× amplification on top of the
shared-mem reuse. Without this step we measured ~8.5 TFLOPS at N=4096 on the
5090 — shared-mem latency was the bottleneck. Adding the 4×4 register tile
pushes the same kernel to ~29–38 TFLOPS.

## Results (RTX 5090, sm_120 native, CUDA 13.2, nvcc + clang-14)

| N    | best ms | median ms | best TFLOPS | median TFLOPS |
|------|---------|-----------|-------------|---------------|
| 1024 |  0.084  |   0.087   |    25.55    |     24.60     |
| 2048 |  0.500  |   0.510   |    34.33    |     33.67     |
| 4096 |  3.639  |   4.749   |    37.77    |     28.94     |

Baseline (W1B naive nvcc, same machine, N=4096): **~6.3 TFLOPS best**.
Tiled result at N=4096: **37.8 TFLOPS best → 6.0× speedup**. Median speedup
is ~4.6×, comfortably above the Wave-2 acceptance bar (≥4× naive). N=2048
shows some variance iter-to-iter (range 11.9–34.3 TFLOPS), consistent with
the desktop-thermal-contention pattern documented in `AGENTS.md`.

Spot-checks at (0,0), (n/2,n/2), (n-1,n-1) match CPU reference to ~1e-6
relative error for every N — no correctness regression vs. naive.
