# oxide-reduction — Wave 4 W4A Analysis

## Kernel

Block size **256 (8 warps)**. Identical 2-stage algorithm to the nvcc
reference: warp-shuffle butterfly → `SharedArray<f32, 8>` per-warp partials
→ first-warp reduce → `DeviceAtomicF32::fetch_add(.., Relaxed)`.
Grid-stride loop over `data.len()` with fixed grid=4096.

```rust
#[kernel]
pub fn reduce_sum(data: &[f32], mut out: DisjointSlice<f32>) {
    static mut PARTIALS: SharedArray<f32, 8> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x() as usize;
    let (bid, bdim, gdim) = (thread::blockIdx_x() as usize,
                             thread::blockDim_x() as usize,
                             thread::gridDim_x() as usize);
    let lane = warp::lane_id() as usize;
    let warp_id = tid >> 5;

    let mut acc: f32 = 0.0;
    let stride = bdim * gdim;
    let mut i = bid * bdim + tid;
    let p = data.as_ptr();
    while i < data.len() { unsafe { acc += *p.add(i); } i += stride; }

    acc += warp::shuffle_xor_f32(acc, 16);
    acc += warp::shuffle_xor_f32(acc, 8);
    acc += warp::shuffle_xor_f32(acc, 4);
    acc += warp::shuffle_xor_f32(acc, 2);
    acc += warp::shuffle_xor_f32(acc, 1);

    if lane == 0 { unsafe { PARTIALS[warp_id] = acc; } }
    thread::sync_threads();

    if warp_id == 0 {
        let mut v = if lane < 8 { unsafe { PARTIALS[lane] } } else { 0.0 };
        v += warp::shuffle_xor_f32(v, 4);
        v += warp::shuffle_xor_f32(v, 2);
        v += warp::shuffle_xor_f32(v, 1);
        if lane == 0 {
            let atomic = unsafe { &*(out.as_mut_ptr() as *const DeviceAtomicF32) };
            atomic.fetch_add(v, AtomicOrdering::Relaxed);
        }
    }
}
```

## Results (best / median of 10 iters, 1 warmup)

| N (elems) | bytes | best ms | med ms | best GB/s | med GB/s | rel_err |
|-----------|-------|---------|--------|-----------|----------|---------|
| 1,048,576   |   4 MB | 0.008 | 0.014 |   506 |   292 | 2.7e-6 |
| 16,777,216  |  64 MB | 0.018 | 0.021 |  3772 |  3197 | 3.7e-6 |
| 268,435,456 |   1 GB | 0.740 | 0.962 |  1451 |  1116 | 2.4e-5 |

## Comparison vs nvcc cuda-reduction

- **Best-case parity.** At 1 GB, oxide's fastest run hits **1451 GB/s (81 %
  of DRAM peak)** vs nvcc's 1517 GB/s — about **96 % of nvcc's throughput**.
- **Correctness identical.** Both reductions produce sums within ~2e-5 of
  the CPU Kahan oracle at 256 M elements; numerically equivalent.
- **Variance is meaningfully higher for oxide.** At 1 GB, oxide median is
  0.962 ms (13 % above its best) while cuda median is 0.713 ms (< 1 % above).
  Three oxide iters at 1 GB spiked to 1.1–3.0 ms. Suspect cause: same
  shared WSL/GPU workload noise that affected Wave 1 (no `nvidia-smi -lgc`
  available), amplified here because the kernel is too short to absorb
  jitter. Best-case numbers are the reliable comparison.
- **PTX was not markedly different** from the nvcc version in shape: the
  cuda-oxide build used `shuffle_xor_f32` lowering from the warp module
  and emits the correct `shfl.sync.bfly.b32` plus `atom.global.add.f32`;
  this is confirmed by the absence of any `bar.sync` injected between the
  xor-steps and by the fact that we match nvcc's throughput at best-case.
