# cuda-reduction — Wave 4 W4A Analysis

## Kernel

Block size **256 (8 warps)**. 2-stage reduction: each warp butterfly-reduces
via `__shfl_xor_sync`, lane-0 writes to `__shared__ float partials[8]`,
warp 0 reduces the 8 partials and `atomicAdd`s into a single global `f32`.
A grid-stride loop lets a fixed `grid=4096` cover any N.

```cpp
__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 4);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    return v;
}

__global__ void reduce_sum_kernel(const float* data, float* out, uint64_t n) {
    __shared__ float partials[WARPS_PER_BLOCK];
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (uint64_t i = gid; i < n; i += stride) acc += data[i];
    acc = warp_reduce_sum(acc);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) partials[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < WARPS_PER_BLOCK) ? partials[lane] : 0.0f;
        v += __shfl_xor_sync(0xffffffff, v, 4);
        v += __shfl_xor_sync(0xffffffff, v, 2);
        v += __shfl_xor_sync(0xffffffff, v, 1);
        if (lane == 0) atomicAdd(out, v);
    }
}
```

## Results (best / median of 10 iters, 1 warmup)

| N (elems) | bytes | best ms | med ms | best GB/s | med GB/s | rel_err |
|-----------|-------|---------|--------|-----------|----------|---------|
| 1,048,576   |   4 MB | 0.009 | 0.011 |   454 |   396 | 2.7e-6 |
| 16,777,216  |  64 MB | 0.016 | 0.019 |  4145 |  3543 | 3.8e-6 |
| 268,435,456 |   1 GB | 0.708 | 0.713 |  1517 |  1505 | 2.5e-5 |

## Notes

- **1 GB case hits 1517 GB/s = 85 % of the 1.79 TB/s RTX 5090 DRAM peak.**
  This is the canonical bandwidth-bound baseline for any reduction.
- **64 MB case reads at 4145 GB/s — well above DRAM peak** because it is
  L2-resident after the warmup iter (RTX 5090 has ~88 MB L2). Not a fair
  DRAM measurement, but a useful "hot-cache" data-point.
- Variance is low (< 10 % CV) at 1 GB where each iter is long enough to
  average out kernel-launch jitter. At 4 MB / 64 MB the per-iter time is
  ≤ 20 µs, so CPU-side launch overhead and GPU-clock transitions dominate;
  one outlier at N=16M iter 5 (39 µs vs ~17 µs typical).
- Correctness: GPU single-precision tree-sum vs CPU double-precision Kahan
  reference. Relative error grows with N as expected (~√N mantissa accumulation);
  2.5e-5 at 256 M still three decades below the 1e-3 gate.
- `atomicAdd` contention is trivial: only 4096/256 = 16 blocks × one atomic
  per block ≤ 64 atomics total per launch.
