// cuda-oxide naive matmul: same algorithm as wgpu and raw CUDA reference.
// Two kernels: `matmul` (safe slice indexing, bounds-checked) and
// `matmul_unchecked` (raw pointer reads, no bounds checks). The unchecked
// version isolates how much of the perf delta vs nvcc CUDA C++ comes from
// Rust's per-iteration bounds checks. Both produce identical results.
//
// Wave 1 W1A (ADR-0001): kernel-only timing via cuEventRecord on stream 0
// (default stream). CPU wall-clock also retained for debug visibility.
// Size sweep over N in {1024, 2048, 4096}. CSV output for downstream tooling.

use cuda_core::{CudaContext, CudaEvent, CudaModule, CudaStream, DeviceBuffer, LaunchConfig, sys};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::{cuda_launch, load_kernel_module};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::time::Instant;

const N_MAX: usize = 4096;
const SIZES: [usize; 3] = [1024, 2048, 4096];
const BS: u32 = 16;
const WARMUP: usize = 1;
const ITERS: usize = 10;

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

/// Copy `src` (host f32 slice) into the first `src.len()` elements of `dst` (device).
/// Uses the raw driver API via cuda-oxide's re-exported `cuda_bindings` crate.
fn upload_f32(stream: &CudaStream, dst: &DeviceBuffer<f32>, src: &[f32]) {
    use cuda_core::IntoResult;
    let num_bytes = std::mem::size_of_val(src);
    assert!(num_bytes <= dst.num_bytes(), "upload exceeds dst capacity");
    // Bind ctx before driver call.
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

/// One (warmup + timed) sweep for a given kernel at a given dim.
/// Returns list of (gpu_ms, cpu_wall_ms, tflops) per iter.
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

    // Warmup
    for _ in 0..WARMUP {
        let s = stream.clone();
        let m = module.clone();
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
    }

    let mut results: Vec<(f64, f64, f64)> = Vec::with_capacity(ITERS);
    for i in 0..ITERS {
        // Create timing-enabled events per iter (cheap).
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
        stop.record(stream).expect("record stop");
        stream.synchronize().unwrap();
        let cpu_wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // elapsed_ms internally syncs both events (idempotent after sync above).
        let gpu_ms = start.elapsed_ms(&stop).expect("elapsed_ms") as f64;
        let tflops = (total_flops / 1e12) / (gpu_ms / 1000.0);

        println!(
            "[oxide-{kernel_name}] N={n} iter={i} gpu_ms={gpu_ms:.3} cpu_wall_ms={cpu_wall_ms:.3} tflops={tflops:.3}"
        );
        writeln!(
            csv,
            "oxide,{kernel_name},{n},{i},{gpu_ms:.6},{cpu_wall_ms:.6},{tflops:.6}"
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
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = load_kernel_module(&ctx, "oxide_matmul").expect("load");

    // Open CSV fresh (truncate) and write header.
    let csv_path = concat!(env!("CARGO_MANIFEST_DIR"), "/results.csv");
    let csv_file = File::create(csv_path).expect("create results.csv");
    let mut csv = BufWriter::new(csv_file);
    writeln!(csv, "impl,kernel,N,iter,gpu_ms,cpu_wall_ms,tflops").unwrap();

    println!(
        "[oxide] matmul size-sweep {:?} f32, {} warmup + {} timed iters per (kernel, N)",
        SIZES, WARMUP, ITERS
    );
    println!("[oxide] timing: gpu_ms via cuEventRecord (ADR-0001), cpu_wall_ms via Instant");
    println!();

    // Allocate buffers ONCE for max N. Populated with N_MAX-sized host data up-front;
    // per-N inner loop re-uploads N²-sized host data into the first N² elements.
    let a_host_max: Vec<f32> = (0..N_MAX * N_MAX).map(|i| ((i % 7) as f32) * 0.01).collect();
    let b_host_max: Vec<f32> = (0..N_MAX * N_MAX).map(|i| ((i % 11) as f32) * 0.01).collect();
    let a_dev = DeviceBuffer::from_host(&stream, &a_host_max).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_host_max).unwrap();
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N_MAX * N_MAX).unwrap();

    // Summary accumulator: (kernel_name, N) -> (best_gpu_ms, median_gpu_ms, best_tf, med_tf)
    let mut summary: Vec<(String, usize, f64, f64, f64, f64)> = Vec::new();

    for &n in &SIZES {
        println!("---- N = {n} ----");
        // Build fresh N×N host data with stride = n (so correctness-check math matches
        // what the kernel actually reads given dim_arg = n).
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

            // Correctness at (0,0), (n/2, n/2), (n-1, n-1).
            let c_host = c_dev.to_host_vec(&stream).unwrap();
            let mut ok = 0;
            for &(r, col) in &[(0usize, 0usize), (n / 2, n / 2), (n - 1, n - 1)] {
                let mut expect = 0.0_f32;
                for k in 0..n {
                    expect += a_host[r * n + k] * b_host[k * n + col];
                }
                // Column index must use stride = n in the device buffer too, since the
                // kernel reads with stride = dim = n and writes at r*n + col.
                let got = c_host[r * n + col];
                if (got - expect).abs() / expect.abs().max(1e-6) < 1e-3 {
                    ok += 1;
                }
            }
            println!(
                "[oxide-{kname}] N={n} correctness {ok}/3  best={best:.3}ms median={med:.3}ms  ({med_tf:.3} TFLOPS median)"
            );
        }
        println!();
    }

    csv.flush().unwrap();

    // === Summary table ===
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
