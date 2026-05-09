// Wave 7: cuda-oxide tiled matmul with 4x4 register microtile + fmuladdf32.
// Applies Wave 5 SASS findings (nvcc's 28 TFLOPS path uses block-level
// register microtile + FFMA via FastmathFlags::CONTRACT).
//
// Block dim: 16x16 threads (256).
// Each thread computes a 4x4 OUTPUT microtile -> block covers 64x64 output.
// K-loop tile size BK = 16. Shared tiles: A = 64x16, B = 16x64 (1024 f32 each).
//
// Two kernels:
//   matmul_tiled_4x4       -- accumulation via core::intrinsics::fmuladdf32
//   matmul_tiled_4x4_safe  -- plain `sum = sum + a*b` (no FFMA, for isolation)
//
// NB: cuda-oxide (current) can't lower 2-level array projections in
// assignments, so accumulators are 16 scalar locals rather than [[f32;4];4].
//
// Host: 1 warmup + 3 timed iters per kernel at N in {1024, 4096}. Gaming
// on GPU concurrently -- timings are ±20-30% noisy. Primary evidence is PTX.

#![feature(core_intrinsics)]
#![allow(internal_features)]

use cuda_core::{CudaContext, CudaEvent, CudaModule, CudaStream, DeviceBuffer, LaunchConfig, sys};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::{cuda_launch, load_kernel_module};
use std::sync::Arc;
use std::time::Instant;

const N_MAX: usize = 4096;
const SIZES: [usize; 2] = [1024, 4096];
const BX: u32 = 16;
const BY: u32 = 16;
const BM: usize = 64;
const WARMUP: usize = 1;
const ITERS: usize = 3;

#[kernel]
pub fn matmul_tiled_4x4(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>, dim: u32) {
    static mut TILE_A: SharedArray<f32, 1024> = SharedArray::UNINIT;
    static mut TILE_B: SharedArray<f32, 1024> = SharedArray::UNINIT;

    let tx = thread::threadIdx_x() as usize;
    let ty = thread::threadIdx_y() as usize;
    let bx = thread::blockIdx_x() as usize;
    let by = thread::blockIdx_y() as usize;
    let dim_us = dim as usize;

    let row0 = by * 64 + ty * 4;
    let col0 = bx * 64 + tx * 4;
    let tid = ty * 16 + tx;
    let num_tiles = dim_us / 16;

    // 4x4 scalar accumulators (flattened — cuda-oxide can't lower [[f32;4];4] assigns).
    let mut c00: f32 = 0.0; let mut c01: f32 = 0.0; let mut c02: f32 = 0.0; let mut c03: f32 = 0.0;
    let mut c10: f32 = 0.0; let mut c11: f32 = 0.0; let mut c12: f32 = 0.0; let mut c13: f32 = 0.0;
    let mut c20: f32 = 0.0; let mut c21: f32 = 0.0; let mut c22: f32 = 0.0; let mut c23: f32 = 0.0;
    let mut c30: f32 = 0.0; let mut c31: f32 = 0.0; let mut c32: f32 = 0.0; let mut c33: f32 = 0.0;

    let a_base = a.as_ptr();
    let b_base = b.as_ptr();

    let mut t: usize = 0;
    while t < num_tiles {
        let k_base = t * 16;

        // Cooperative load: 256 threads x 4 elements = 1024 per tile.
        let mut li: usize = 0;
        while li < 4 {
            let idx = tid + li * 256;
            let r = idx / 16;
            let k = idx & 15;
            let gr = by * 64 + r;
            let gk = k_base + k;
            unsafe {
                let v = *a_base.add(gr * dim_us + gk);
                TILE_A[idx] = v;
            }
            li += 1;
        }
        let mut lj: usize = 0;
        while lj < 4 {
            let idx = tid + lj * 256;
            let k = idx / 64;
            let cc = idx & 63;
            let gk = k_base + k;
            let gc = bx * 64 + cc;
            unsafe {
                let v = *b_base.add(gk * dim_us + gc);
                TILE_B[idx] = v;
            }
            lj += 1;
        }
        thread::sync_threads();

        let ty4 = ty * 4;
        let tx4 = tx * 4;
        let mut k: usize = 0;
        while k < 16 {
            let a0: f32; let a1: f32; let a2: f32; let a3: f32;
            let b0: f32; let b1: f32; let b2: f32; let b3: f32;
            unsafe {
                a0 = TILE_A[(ty4 + 0) * 16 + k];
                a1 = TILE_A[(ty4 + 1) * 16 + k];
                a2 = TILE_A[(ty4 + 2) * 16 + k];
                a3 = TILE_A[(ty4 + 3) * 16 + k];
                b0 = TILE_B[k * 64 + tx4 + 0];
                b1 = TILE_B[k * 64 + tx4 + 1];
                b2 = TILE_B[k * 64 + tx4 + 2];
                b3 = TILE_B[k * 64 + tx4 + 3];
                c00 = core::intrinsics::fmuladdf32(a0, b0, c00);
                c01 = core::intrinsics::fmuladdf32(a0, b1, c01);
                c02 = core::intrinsics::fmuladdf32(a0, b2, c02);
                c03 = core::intrinsics::fmuladdf32(a0, b3, c03);
                c10 = core::intrinsics::fmuladdf32(a1, b0, c10);
                c11 = core::intrinsics::fmuladdf32(a1, b1, c11);
                c12 = core::intrinsics::fmuladdf32(a1, b2, c12);
                c13 = core::intrinsics::fmuladdf32(a1, b3, c13);
                c20 = core::intrinsics::fmuladdf32(a2, b0, c20);
                c21 = core::intrinsics::fmuladdf32(a2, b1, c21);
                c22 = core::intrinsics::fmuladdf32(a2, b2, c22);
                c23 = core::intrinsics::fmuladdf32(a2, b3, c23);
                c30 = core::intrinsics::fmuladdf32(a3, b0, c30);
                c31 = core::intrinsics::fmuladdf32(a3, b1, c31);
                c32 = core::intrinsics::fmuladdf32(a3, b2, c32);
                c33 = core::intrinsics::fmuladdf32(a3, b3, c33);
            }
            k += 1;
        }
        thread::sync_threads();
        t += 1;
    }

    let c_ptr = c.as_mut_ptr();
    unsafe {
        *c_ptr.add((row0 + 0) * dim_us + col0 + 0) = c00;
        *c_ptr.add((row0 + 0) * dim_us + col0 + 1) = c01;
        *c_ptr.add((row0 + 0) * dim_us + col0 + 2) = c02;
        *c_ptr.add((row0 + 0) * dim_us + col0 + 3) = c03;
        *c_ptr.add((row0 + 1) * dim_us + col0 + 0) = c10;
        *c_ptr.add((row0 + 1) * dim_us + col0 + 1) = c11;
        *c_ptr.add((row0 + 1) * dim_us + col0 + 2) = c12;
        *c_ptr.add((row0 + 1) * dim_us + col0 + 3) = c13;
        *c_ptr.add((row0 + 2) * dim_us + col0 + 0) = c20;
        *c_ptr.add((row0 + 2) * dim_us + col0 + 1) = c21;
        *c_ptr.add((row0 + 2) * dim_us + col0 + 2) = c22;
        *c_ptr.add((row0 + 2) * dim_us + col0 + 3) = c23;
        *c_ptr.add((row0 + 3) * dim_us + col0 + 0) = c30;
        *c_ptr.add((row0 + 3) * dim_us + col0 + 1) = c31;
        *c_ptr.add((row0 + 3) * dim_us + col0 + 2) = c32;
        *c_ptr.add((row0 + 3) * dim_us + col0 + 3) = c33;
    }
}

#[kernel]
pub fn matmul_tiled_4x4_safe(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>, dim: u32) {
    static mut TILE_A: SharedArray<f32, 1024> = SharedArray::UNINIT;
    static mut TILE_B: SharedArray<f32, 1024> = SharedArray::UNINIT;

    let tx = thread::threadIdx_x() as usize;
    let ty = thread::threadIdx_y() as usize;
    let bx = thread::blockIdx_x() as usize;
    let by = thread::blockIdx_y() as usize;
    let dim_us = dim as usize;

    let row0 = by * 64 + ty * 4;
    let col0 = bx * 64 + tx * 4;
    let tid = ty * 16 + tx;
    let num_tiles = dim_us / 16;

    let mut c00: f32 = 0.0; let mut c01: f32 = 0.0; let mut c02: f32 = 0.0; let mut c03: f32 = 0.0;
    let mut c10: f32 = 0.0; let mut c11: f32 = 0.0; let mut c12: f32 = 0.0; let mut c13: f32 = 0.0;
    let mut c20: f32 = 0.0; let mut c21: f32 = 0.0; let mut c22: f32 = 0.0; let mut c23: f32 = 0.0;
    let mut c30: f32 = 0.0; let mut c31: f32 = 0.0; let mut c32: f32 = 0.0; let mut c33: f32 = 0.0;

    let a_base = a.as_ptr();
    let b_base = b.as_ptr();

    let mut t: usize = 0;
    while t < num_tiles {
        let k_base = t * 16;

        let mut li: usize = 0;
        while li < 4 {
            let idx = tid + li * 256;
            let r = idx / 16;
            let k = idx & 15;
            let gr = by * 64 + r;
            let gk = k_base + k;
            unsafe {
                let v = *a_base.add(gr * dim_us + gk);
                TILE_A[idx] = v;
            }
            li += 1;
        }
        let mut lj: usize = 0;
        while lj < 4 {
            let idx = tid + lj * 256;
            let k = idx / 64;
            let cc = idx & 63;
            let gk = k_base + k;
            let gc = bx * 64 + cc;
            unsafe {
                let v = *b_base.add(gk * dim_us + gc);
                TILE_B[idx] = v;
            }
            lj += 1;
        }
        thread::sync_threads();

        let ty4 = ty * 4;
        let tx4 = tx * 4;
        let mut k: usize = 0;
        while k < 16 {
            let a0: f32; let a1: f32; let a2: f32; let a3: f32;
            let b0: f32; let b1: f32; let b2: f32; let b3: f32;
            unsafe {
                a0 = TILE_A[(ty4 + 0) * 16 + k];
                a1 = TILE_A[(ty4 + 1) * 16 + k];
                a2 = TILE_A[(ty4 + 2) * 16 + k];
                a3 = TILE_A[(ty4 + 3) * 16 + k];
                b0 = TILE_B[k * 64 + tx4 + 0];
                b1 = TILE_B[k * 64 + tx4 + 1];
                b2 = TILE_B[k * 64 + tx4 + 2];
                b3 = TILE_B[k * 64 + tx4 + 3];
            }
            c00 = c00 + a0 * b0; c01 = c01 + a0 * b1; c02 = c02 + a0 * b2; c03 = c03 + a0 * b3;
            c10 = c10 + a1 * b0; c11 = c11 + a1 * b1; c12 = c12 + a1 * b2; c13 = c13 + a1 * b3;
            c20 = c20 + a2 * b0; c21 = c21 + a2 * b1; c22 = c22 + a2 * b2; c23 = c23 + a2 * b3;
            c30 = c30 + a3 * b0; c31 = c31 + a3 * b1; c32 = c32 + a3 * b2; c33 = c33 + a3 * b3;
            k += 1;
        }
        thread::sync_threads();
        t += 1;
    }

    let c_ptr = c.as_mut_ptr();
    unsafe {
        *c_ptr.add((row0 + 0) * dim_us + col0 + 0) = c00;
        *c_ptr.add((row0 + 0) * dim_us + col0 + 1) = c01;
        *c_ptr.add((row0 + 0) * dim_us + col0 + 2) = c02;
        *c_ptr.add((row0 + 0) * dim_us + col0 + 3) = c03;
        *c_ptr.add((row0 + 1) * dim_us + col0 + 0) = c10;
        *c_ptr.add((row0 + 1) * dim_us + col0 + 1) = c11;
        *c_ptr.add((row0 + 1) * dim_us + col0 + 2) = c12;
        *c_ptr.add((row0 + 1) * dim_us + col0 + 3) = c13;
        *c_ptr.add((row0 + 2) * dim_us + col0 + 0) = c20;
        *c_ptr.add((row0 + 2) * dim_us + col0 + 1) = c21;
        *c_ptr.add((row0 + 2) * dim_us + col0 + 2) = c22;
        *c_ptr.add((row0 + 2) * dim_us + col0 + 3) = c23;
        *c_ptr.add((row0 + 3) * dim_us + col0 + 0) = c30;
        *c_ptr.add((row0 + 3) * dim_us + col0 + 1) = c31;
        *c_ptr.add((row0 + 3) * dim_us + col0 + 2) = c32;
        *c_ptr.add((row0 + 3) * dim_us + col0 + 3) = c33;
    }
}

fn upload_f32(stream: &CudaStream, dst: &DeviceBuffer<f32>, src: &[f32]) {
    use cuda_core::IntoResult;
    let num_bytes = std::mem::size_of_val(src);
    assert!(num_bytes <= dst.num_bytes());
    stream.context().bind_to_thread().expect("bind ctx");
    unsafe {
        sys::cuMemcpyHtoDAsync_v2(
            dst.cu_deviceptr(),
            src.as_ptr() as *const _,
            num_bytes,
            stream.cu_stream(),
        )
        .result()
        .expect("htod");
    }
    stream.synchronize().unwrap();
}

#[allow(clippy::too_many_arguments)]
fn run_sweep(
    label: &str,
    n: usize,
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    a_dev: &DeviceBuffer<f32>,
    b_dev: &DeviceBuffer<f32>,
    mut c_dev: &mut DeviceBuffer<f32>,
    which: u8,
) -> Vec<(f64, f64)> {
    let dim_arg = n as u32;
    let total_flops = 2.0_f64 * (n as f64).powi(3);
    let gx = (n as u32).div_ceil(BM as u32);
    let gy = (n as u32).div_ceil(BM as u32);
    let cfg = LaunchConfig { grid_dim: (gx, gy, 1), block_dim: (BX, BY, 1), shared_mem_bytes: 0 };

    for _ in 0..WARMUP {
        let s = stream.clone();
        let m = module.clone();
        match which {
            0 => cuda_launch! {
                kernel: matmul_tiled_4x4, stream: s, module: m, config: cfg,
                args: [slice(a_dev), slice(b_dev), slice_mut(c_dev), dim_arg]
            }.unwrap(),
            _ => cuda_launch! {
                kernel: matmul_tiled_4x4_safe, stream: s, module: m, config: cfg,
                args: [slice(a_dev), slice(b_dev), slice_mut(c_dev), dim_arg]
            }.unwrap(),
        }
        stream.synchronize().unwrap();
    }

    let mut out = Vec::with_capacity(ITERS);
    for i in 0..ITERS {
        let start: CudaEvent = ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT)).unwrap();
        let stop: CudaEvent = ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT)).unwrap();
        let s = stream.clone();
        let m = module.clone();
        let t0 = Instant::now();
        start.record(stream).unwrap();
        match which {
            0 => cuda_launch! {
                kernel: matmul_tiled_4x4, stream: s, module: m, config: cfg,
                args: [slice(a_dev), slice(b_dev), slice_mut(c_dev), dim_arg]
            }.unwrap(),
            _ => cuda_launch! {
                kernel: matmul_tiled_4x4_safe, stream: s, module: m, config: cfg,
                args: [slice(a_dev), slice(b_dev), slice_mut(c_dev), dim_arg]
            }.unwrap(),
        }
        stop.record(stream).unwrap();
        stream.synchronize().unwrap();
        let _ = t0.elapsed();
        let gpu_ms = start.elapsed_ms(&stop).unwrap() as f64;
        let tflops = (total_flops / 1e12) / (gpu_ms / 1000.0);
        println!("[{label}] N={n} iter={i} gpu_ms={gpu_ms:.3} tflops={tflops:.3}");
        out.push((gpu_ms, tflops));
    }
    out
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = load_kernel_module(&ctx, "oxide_matmul_tiled_microtile").expect("load");

    println!("[oxide-tiled-microtile] sizes={:?} warmup={} iters={} (noisy: user gaming)", SIZES, WARMUP, ITERS);

    let a_host_max: Vec<f32> = (0..N_MAX * N_MAX).map(|i| ((i % 7) as f32) * 0.01).collect();
    let b_host_max: Vec<f32> = (0..N_MAX * N_MAX).map(|i| ((i % 11) as f32) * 0.01).collect();
    let a_dev = DeviceBuffer::from_host(&stream, &a_host_max).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_host_max).unwrap();
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N_MAX * N_MAX).unwrap();

    for &n in &SIZES {
        println!("---- N = {n} ----");
        let a_host: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.01).collect();
        let b_host: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.01).collect();
        upload_f32(&stream, &a_dev, &a_host);
        upload_f32(&stream, &b_dev, &b_host);

        for &(label, which) in &[("fmuladd", 0u8), ("safe", 1u8)] {
            let _ = run_sweep(label, n, &ctx, &stream, &module, &a_dev, &b_dev, &mut c_dev, which);

            let c_host = c_dev.to_host_vec(&stream).unwrap();
            let mut ok = 0;
            for &(r, col) in &[(0usize, 0usize), (n / 2, n / 2), (n - 1, n - 1)] {
                let mut expect = 0.0_f32;
                for k in 0..n {
                    expect += a_host[r * n + k] * b_host[k * n + col];
                }
                let got = c_host[r * n + col];
                if (got - expect).abs() / expect.abs().max(1e-6) < 1e-3 {
                    ok += 1;
                }
            }
            println!("[{label}] N={n} correctness {ok}/3");
        }
    }
}
