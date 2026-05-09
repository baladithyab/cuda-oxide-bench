// cuda-oxide tiled matmul (Wave 2 W2C).
// 16x16 block tile. K dimension iterated in tiles of BS=16. Each iteration:
//   1) cooperatively load one BS×BS tile of A and one of B into shared memory
//   2) sync_threads
//   3) each thread accumulates BS partial products from its (ty, tx) slot
//   4) sync_threads before loading next tile
// Two kernels: `matmul_tiled` (safe slice indexing) and `matmul_tiled_unchecked`
// (raw pointer reads). Same host harness, CSV, and timing pattern as oxide-matmul.
//
// SharedArray API reference: sharedmem example at
// cuda-oxide/crates/rustc-codegen-cuda/examples/sharedmem/src/main.rs

use cuda_core::{CudaContext, CudaEvent, CudaModule, CudaStream, DeviceBuffer, LaunchConfig, sys};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::{cuda_launch, load_kernel_module};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::time::Instant;

const N_MAX: usize = 4096;
const SIZES: [usize; 3] = [1024, 2048, 4096];
const BS: u32 = 16;
const BS_US: usize = 16;
const TILE_LEN: usize = 256; // BS*BS
const WARMUP: usize = 1;
const ITERS: usize = 10;

#[kernel]
pub fn matmul_tiled(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>, dim: u32) {
    static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

    let tx = thread::threadIdx_x();
    let ty = thread::threadIdx_y();
    let bx = thread::blockIdx_x();
    let by = thread::blockIdx_y();

    let row = by * 16 + ty;
    let col = bx * 16 + tx;

    let tx_us = tx as usize;
    let ty_us = ty as usize;
    let dim_us = dim as usize;
    let row_us = row as usize;
    let col_us = col as usize;
    let local = ty_us * 16 + tx_us;

    let num_tiles = dim_us / 16; // dim is multiple of 16 for our sweep
    let mut acc: f32 = 0.0;
    let mut t: usize = 0;
    while t < num_tiles {
        let a_col = t * 16 + tx_us;
        let b_row = t * 16 + ty_us;
        // Load tile. Bounds-checked reads via slice indexing.
        let a_val = if row_us < dim_us && a_col < dim_us {
            a[row_us * dim_us + a_col]
        } else {
            0.0
        };
        let b_val = if b_row < dim_us && col_us < dim_us {
            b[b_row * dim_us + col_us]
        } else {
            0.0
        };
        unsafe {
            TILE_A[local] = a_val;
            TILE_B[local] = b_val;
        }
        thread::sync_threads();

        // Compute on tile.
        let mut k: usize = 0;
        while k < 16 {
            unsafe {
                acc += TILE_A[ty_us * 16 + k] * TILE_B[k * 16 + tx_us];
            }
            k += 1;
        }
        thread::sync_threads();
        t += 1;
    }

    if row_us < dim_us && col_us < dim_us {
        unsafe {
            *c.as_mut_ptr().add(row_us * dim_us + col_us) = acc;
        }
    }
}

#[kernel]
pub fn matmul_tiled_unchecked(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>, dim: u32) {
    static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

    let tx = thread::threadIdx_x();
    let ty = thread::threadIdx_y();
    let bx = thread::blockIdx_x();
    let by = thread::blockIdx_y();

    let row = by * 16 + ty;
    let col = bx * 16 + tx;

    let tx_us = tx as usize;
    let ty_us = ty as usize;
    let dim_us = dim as usize;
    let row_us = row as usize;
    let col_us = col as usize;
    let local = ty_us * 16 + tx_us;

    let a_base = a.as_ptr();
    let b_base = b.as_ptr();

    let num_tiles = dim_us / 16;
    let mut acc: f32 = 0.0;
    let mut t: usize = 0;
    while t < num_tiles {
        let a_col = t * 16 + tx_us;
        let b_row = t * 16 + ty_us;
        // SAFETY: for dim multiple of 16, all indices are in bounds.
        unsafe {
            let a_val = *a_base.add(row_us * dim_us + a_col);
            let b_val = *b_base.add(b_row * dim_us + col_us);
            TILE_A[local] = a_val;
            TILE_B[local] = b_val;
        }
        thread::sync_threads();

        let mut k: usize = 0;
        while k < 16 {
            unsafe {
                acc += TILE_A[ty_us * 16 + k] * TILE_B[k * 16 + tx_us];
            }
            k += 1;
        }
        thread::sync_threads();
        t += 1;
    }

    unsafe {
        *c.as_mut_ptr().add(row_us * dim_us + col_us) = acc;
    }
}

fn upload_f32(stream: &CudaStream, dst: &DeviceBuffer<f32>, src: &[f32]) {
    use cuda_core::IntoResult;
    let num_bytes = std::mem::size_of_val(src);
    assert!(num_bytes <= dst.num_bytes(), "upload exceeds dst capacity");
    stream
        .context()
        .bind_to_thread()
        .expect("bind ctx for htod");
    unsafe {
        sys::cuMemcpyHtoDAsync_v2(
            dst.cu_deviceptr(),
            src.as_ptr() as *const _,
            num_bytes,
            stream.cu_stream(),
        )
        .result()
        .expect("cuMemcpyHtoDAsync_v2");
    }
    stream.synchronize().expect("sync after htod");
}

#[allow(clippy::too_many_arguments)]
fn run_kernel_sweep(
    kernel_name: &str,
    n: usize,
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    a_dev: &DeviceBuffer<f32>,
    b_dev: &DeviceBuffer<f32>,
    mut c_dev: &mut DeviceBuffer<f32>,
    use_unchecked: bool,
    csv: &mut BufWriter<File>,
) -> Vec<(f64, f64, f64)> {
    let dim_arg = n as u32;
    let total_flops = 2.0_f64 * (n as f64).powi(3);
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(BS), (n as u32).div_ceil(BS), 1),
        block_dim: (BS, BS, 1),
        shared_mem_bytes: 0,
    };

    for _ in 0..WARMUP {
        let s = stream.clone();
        let m = module.clone();
        if use_unchecked {
            cuda_launch! {
                kernel: matmul_tiled_unchecked, stream: s, module: m, config: cfg,
                args: [slice(a_dev), slice(b_dev), slice_mut(c_dev), dim_arg]
            }
            .unwrap();
        } else {
            cuda_launch! {
                kernel: matmul_tiled, stream: s, module: m, config: cfg,
                args: [slice(a_dev), slice(b_dev), slice_mut(c_dev), dim_arg]
            }
            .unwrap();
        }
        stream.synchronize().unwrap();
    }

    let mut results: Vec<(f64, f64, f64)> = Vec::with_capacity(ITERS);
    for i in 0..ITERS {
        let start: CudaEvent = ctx
            .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
            .expect("new_event start");
        let stop: CudaEvent = ctx
            .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
            .expect("new_event stop");

        let s = stream.clone();
        let m = module.clone();

        let t0 = Instant::now();
        start.record(stream).expect("record start");
        if use_unchecked {
            cuda_launch! {
                kernel: matmul_tiled_unchecked, stream: s, module: m, config: cfg,
                args: [slice(a_dev), slice(b_dev), slice_mut(c_dev), dim_arg]
            }
            .unwrap();
        } else {
            cuda_launch! {
                kernel: matmul_tiled, stream: s, module: m, config: cfg,
                args: [slice(a_dev), slice(b_dev), slice_mut(c_dev), dim_arg]
            }
            .unwrap();
        }
        stop.record(stream).expect("record stop");
        stream.synchronize().unwrap();
        let cpu_wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let gpu_ms = start.elapsed_ms(&stop).expect("elapsed_ms") as f64;
        let tflops = (total_flops / 1e12) / (gpu_ms / 1000.0);

        println!(
            "[oxide-tiled-{kernel_name}] N={n} iter={i} gpu_ms={gpu_ms:.3} cpu_wall_ms={cpu_wall_ms:.3} tflops={tflops:.3}"
        );
        writeln!(
            csv,
            "oxide-tiled,{kernel_name},{n},{i},{gpu_ms:.6},{cpu_wall_ms:.6},{tflops:.6}"
        )
        .expect("csv write");
        results.push((gpu_ms, cpu_wall_ms, tflops));
    }
    results
}

fn median(sorted: &[f64]) -> f64 {
    sorted[sorted.len() / 2]
}

fn main() {
    let _ = BS_US; // silence unused-const lint if optimized away
    let _ = TILE_LEN;
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = load_kernel_module(&ctx, "oxide_matmul_tiled").expect("load");

    let csv_path = concat!(env!("CARGO_MANIFEST_DIR"), "/results.csv");
    let csv_file = File::create(csv_path).expect("create results.csv");
    let mut csv = BufWriter::new(csv_file);
    writeln!(csv, "impl,kernel,N,iter,gpu_ms,cpu_wall_ms,tflops").unwrap();

    println!(
        "[oxide-tiled] matmul size-sweep {:?} f32, {} warmup + {} timed iters per (kernel, N)",
        SIZES, WARMUP, ITERS
    );
    println!("[oxide-tiled] 16x16 tile, SharedArray<f32, 256> A & B tiles");
    println!();

    let a_host_max: Vec<f32> = (0..N_MAX * N_MAX).map(|i| ((i % 7) as f32) * 0.01).collect();
    let b_host_max: Vec<f32> = (0..N_MAX * N_MAX).map(|i| ((i % 11) as f32) * 0.01).collect();
    let a_dev = DeviceBuffer::from_host(&stream, &a_host_max).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_host_max).unwrap();
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N_MAX * N_MAX).unwrap();

    let mut summary: Vec<(String, usize, f64, f64, f64, f64)> = Vec::new();

    for &n in &SIZES {
        println!("---- N = {n} ----");
        let a_host: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.01).collect();
        let b_host: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.01).collect();
        upload_f32(&stream, &a_dev, &a_host);
        upload_f32(&stream, &b_dev, &b_host);

        for &(kname, use_unchk) in &[("safe", false), ("unchecked", true)] {
            let iters = run_kernel_sweep(
                kname,
                n,
                &ctx,
                &stream,
                &module,
                &a_dev,
                &b_dev,
                &mut c_dev,
                use_unchk,
                &mut csv,
            );
            let mut gpu_times: Vec<f64> = iters.iter().map(|(g, _, _)| *g).collect();
            gpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let best = gpu_times[0];
            let med = median(&gpu_times);
            let total_flops = 2.0_f64 * (n as f64).powi(3);
            let best_tf = (total_flops / 1e12) / (best / 1000.0);
            let med_tf = (total_flops / 1e12) / (med / 1000.0);
            summary.push((kname.to_string(), n, best, med, best_tf, med_tf));

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
            println!(
                "[oxide-tiled-{kname}] N={n} correctness {ok}/3  best={best:.3}ms median={med:.3}ms  ({med_tf:.3} TFLOPS median)"
            );
        }
        println!();
    }

    csv.flush().unwrap();

    println!("================== SUMMARY ==================");
    println!(
        "{:<12} {:>6} {:>12} {:>12} {:>10} {:>10}",
        "kernel", "N", "best_gpu_ms", "med_gpu_ms", "best_TF", "med_TF"
    );
    println!("{}", "-".repeat(68));
    for (k, n, b, m, btf, mtf) in &summary {
        println!(
            "{:<12} {:>6} {:>12.3} {:>12.3} {:>10.3} {:>10.3}",
            k, n, b, m, btf, mtf
        );
    }
    println!("=============================================");
}
