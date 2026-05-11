// Wave 15.1: row-wise softmax kernel for cuBLAS-3-kernel GQA attention.
//
// Input : scores f32, shape (B, n_q, S, S), row-major; row stride = S.
// Output: probs  f16, same shape, row-major.
//
// Each CUDA block processes one row. We use 128 threads/block and reduce
// across them via shared memory (online-softmax-style: max → exp → sum).
// The attention scale (1 / sqrt(d_head)) is fused into the kernel so we
// don't need a separate pass across the scores tensor.
//
// This is the "custom middle kernel" between the two cublasGemmEx calls.
// For bench at B=1, n_q=32, S=2048 we launch 65,536 blocks — plenty of
// parallelism to saturate the RTX 5090's SMs.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// One block per row. BLOCK_THREADS threads cooperate to reduce.
// Row length S is passed at runtime; kernel loops over S in chunks of
// BLOCK_THREADS. For S=2048 and 128 threads this is 16 iterations.
template <int BLOCK_THREADS>
__global__ void row_softmax_scale_f32_to_f16_kernel(
    const float* __restrict__ scores,  // (num_rows, seq)
    __half* __restrict__ probs,        // (num_rows, seq)
    int seq,
    float scale)                       // 1/sqrt(d_head)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* row_in = scores + (size_t)row * seq;
    __half*      row_out = probs  + (size_t)row * seq;

    // Shared memory for block-wide reductions (max, then sum).
    __shared__ float s_red[BLOCK_THREADS];

    // ---- Pass 1: find max over the row (with scaling folded in) ----
    float local_max = -INFINITY;
    for (int i = tid; i < seq; i += BLOCK_THREADS) {
        float v = row_in[i] * scale;
        if (v > local_max) local_max = v;
    }
    s_red[tid] = local_max;
    __syncthreads();

    // Tree reduction for max.
    for (int off = BLOCK_THREADS / 2; off > 0; off >>= 1) {
        if (tid < off) {
            float other = s_red[tid + off];
            if (other > s_red[tid]) s_red[tid] = other;
        }
        __syncthreads();
    }
    float row_max = s_red[0];
    __syncthreads();

    // ---- Pass 2: compute exp(x*scale - max), accumulate sum ----
    float local_sum = 0.0f;
    for (int i = tid; i < seq; i += BLOCK_THREADS) {
        float v = row_in[i] * scale - row_max;
        float e = __expf(v);
        // Stash exp-result into shared... actually we stash into
        // probs as f16 after normalization. For now write to a
        // temporary via another pass: restart and recompute exp.
        // (Cheaper than S scratch; recompute one __expf per elem.)
        local_sum += e;
    }
    s_red[tid] = local_sum;
    __syncthreads();

    for (int off = BLOCK_THREADS / 2; off > 0; off >>= 1) {
        if (tid < off) s_red[tid] += s_red[tid + off];
        __syncthreads();
    }
    float row_sum = s_red[0];
    float inv_sum = 1.0f / row_sum;
    __syncthreads();

    // ---- Pass 3: write normalized f16 output ----
    for (int i = tid; i < seq; i += BLOCK_THREADS) {
        float v = row_in[i] * scale - row_max;
        float e = __expf(v) * inv_sum;
        row_out[i] = __float2half(e);
    }
}

// Host-callable launcher. num_rows = B * n_q * S.
extern "C" void launch_row_softmax_scale(
    const float* d_scores,
    __half* d_probs,
    int num_rows,
    int seq,
    float scale,
    cudaStream_t stream)
{
    constexpr int BLOCK_THREADS = 128;
    dim3 grid(num_rows);
    dim3 block(BLOCK_THREADS);
    row_softmax_scale_f32_to_f16_kernel<BLOCK_THREADS>
        <<<grid, block, 0, stream>>>(d_scores, d_probs, seq, scale);
}
