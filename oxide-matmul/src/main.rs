// cuda-oxide naive matmul: same algorithm as wgpu and raw CUDA reference.
// Two kernels: `matmul` (safe slice indexing, bounds-checked) and
// `matmul_unchecked` (raw pointer reads, no bounds checks). The unchecked
// version isolates how much of the perf delta vs nvcc CUDA C++ comes from
// Rust's per-iteration bounds checks. Both produce identical results.

use cuda_core::{CudaContext, CudaModule, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::{cuda_launch, load_kernel_module};
use std::sync::Arc;
use std::time::Instant;

const N: usize = 4096;
const BS: u32 = 16;

#[kernel]
pub fn matmul(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>, dim: u32) {
    let row = thread::blockIdx_y() * thread::blockDim_y() + thread::threadIdx_y();
    let col = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
    if row >= dim || col >= dim {
        return;
    }
    let dim_us = dim as usize;
    let r = row as usize;
    let c_idx = col as usize;
    let mut acc: f32 = 0.0;
    let mut k: usize = 0;
    while k < dim_us {
        acc += a[r * dim_us + k] * b[k * dim_us + c_idx];
        k += 1;
    }
    unsafe {
        *c.as_mut_ptr().add(r * dim_us + c_idx) = acc;
    }
}

#[kernel]
pub fn matmul_unchecked(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>, dim: u32) {
    let row = thread::blockIdx_y() * thread::blockDim_y() + thread::threadIdx_y();
    let col = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
    if row >= dim || col >= dim {
        return;
    }
    let dim_us = dim as usize;
    let r = row as usize;
    let c_idx = col as usize;
    let a_base = a.as_ptr();
    let b_base = b.as_ptr();
    let mut acc: f32 = 0.0;
    let mut k: usize = 0;
    while k < dim_us {
        // SAFETY: row, col < dim and 0 <= k < dim_us — both reads in bounds.
        unsafe {
            let av = *a_base.add(r * dim_us + k);
            let bv = *b_base.add(k * dim_us + c_idx);
            acc += av * bv;
        }
        k += 1;
    }
    unsafe {
        *c.as_mut_ptr().add(r * dim_us + c_idx) = acc;
    }
}

fn run_bench(
    name: &str,
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    a_dev: &DeviceBuffer<f32>,
    b_dev: &DeviceBuffer<f32>,
    c_dev: &mut DeviceBuffer<f32>,
    dim_arg: u32,
    use_unchecked: bool,
) -> (f64, f64) {
    let mut c_dev = c_dev;
    let total_flops = 2.0_f64 * (N as f64).powi(3);
    let cfg = LaunchConfig {
        grid_dim: ((N as u32).div_ceil(BS), (N as u32).div_ceil(BS), 1),
        block_dim: (BS, BS, 1),
        shared_mem_bytes: 0,
    };
    let s = stream.clone();
    let m = module.clone();
    // warmup
    if use_unchecked {
        cuda_launch! {
            kernel: matmul_unchecked, stream: s, module: m, config: cfg,
            args: [slice(a_dev), slice(b_dev), slice_mut(c_dev), dim_arg]
        }
        .unwrap();
    } else {
        cuda_launch! {
            kernel: matmul, stream: s, module: m, config: cfg,
            args: [slice(a_dev), slice(b_dev), slice_mut(c_dev), dim_arg]
        }
        .unwrap();
    }
    stream.synchronize().unwrap();

    let mut times: Vec<f64> = Vec::new();
    for i in 0..10 {
        let s = stream.clone();
        let m = module.clone();
        let t0 = Instant::now();
        if use_unchecked {
            cuda_launch! {
                kernel: matmul_unchecked, stream: s, module: m, config: cfg,
                args: [slice(a_dev), slice(b_dev), slice_mut(c_dev), dim_arg]
            }
            .unwrap();
        } else {
            cuda_launch! {
                kernel: matmul, stream: s, module: m, config: cfg,
                args: [slice(a_dev), slice(b_dev), slice_mut(c_dev), dim_arg]
            }
            .unwrap();
        }
        stream.synchronize().unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let tf = (total_flops / 1e12) / (ms / 1000.0);
        println!("[{name}] iter {i}: {ms:.2} ms ({tf:.3} TFLOPS)");
        times.push(ms);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let best = times[0];
    let median = times[times.len() / 2];
    let best_tf = (total_flops / 1e12) / (best / 1000.0);
    let med_tf = (total_flops / 1e12) / (median / 1000.0);
    println!("[{name}] BEST   {best:.2} ms  {best_tf:.3} TFLOPS");
    println!("[{name}] MEDIAN {median:.2} ms  {med_tf:.3} TFLOPS\n");
    (best, median)
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = load_kernel_module(&ctx, "oxide_matmul").expect("load");

    println!(
        "[oxide] matmul {N}x{N} f32 ({:.2} GFLOP/iter)\n",
        2.0 * (N as f64).powi(3) / 1e9
    );

    let a_host: Vec<f32> = (0..N * N).map(|i| ((i % 7) as f32) * 0.01).collect();
    let b_host: Vec<f32> = (0..N * N).map(|i| ((i % 11) as f32) * 0.01).collect();
    let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N * N).unwrap();
    let dim_arg = N as u32;

    let (s_best, s_med) = run_bench(
        "oxide-safe",
        &stream,
        &module,
        &a_dev,
        &b_dev,
        &mut c_dev,
        dim_arg,
        false,
    );
    let (u_best, u_med) = run_bench(
        "oxide-unchk",
        &stream,
        &module,
        &a_dev,
        &b_dev,
        &mut c_dev,
        dim_arg,
        true,
    );

    println!("=== summary ===");
    println!("safe       BEST {:.2} ms  MEDIAN {:.2} ms", s_best, s_med);
    println!("unchecked  BEST {:.2} ms  MEDIAN {:.2} ms", u_best, u_med);
    println!("speedup (median): {:.2}x", s_med / u_med);

    let c_host = c_dev.to_host_vec(&stream).unwrap();
    let dim = N;
    let mut ok = 0;
    for &(r, c) in &[(0usize, 0usize), (100, 100), (4095, 4095)] {
        let mut expect = 0.0_f32;
        for k in 0..dim {
            expect += a_host[r * dim + k] * b_host[k * dim + c];
        }
        if (c_host[r * dim + c] - expect).abs() / expect.abs().max(1e-6) < 1e-3 {
            ok += 1;
        }
    }
    println!("[oxide] correctness spot-check: {ok}/3");
}
