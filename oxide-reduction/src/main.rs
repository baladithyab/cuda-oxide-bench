// Wave 4 W4A: cuda-oxide parallel sum-reduction.
//
// 2-stage reduction: warp-shuffle within warp -> smem across warps -> atomicAdd
// to a single device-side f32.  Block size = 256 threads = 8 warps.
// Grid-stride loop so we can cap grid_dim at a modest value.
//
// Template: examples/warp_reduce/src/main.rs (warp API) +
//           examples/atomics/src/main.rs     (DeviceAtomicF32::fetch_add)
// Host timing: oxide-matmul (cuEventRecord pattern).

#![allow(internal_features)]

use cuda_core::{CudaContext, CudaEvent, DeviceBuffer, LaunchConfig, sys};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32};
use cuda_device::shared::SharedArray;
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::{cuda_launch, load_kernel_module};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

const BLOCK: u32 = 256;
const WARPS_PER_BLOCK: usize = 8;
const GRID: u32 = 4096;
const WARMUP: usize = 1;
const ITERS: usize = 10;

const SIZES: [usize; 3] = [1024 * 1024, 16 * 1024 * 1024, 256 * 1024 * 1024];

#[kernel]
pub fn reduce_sum(data: &[f32], mut out: DisjointSlice<f32>) {
    // Per-block shared storage for 8 warp partials.
    static mut PARTIALS: SharedArray<f32, 8> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    let bid = thread::blockIdx_x() as usize;
    let bdim = thread::blockDim_x() as usize;
    let gdim = thread::gridDim_x() as usize;
    let n = data.len();

    let lane = warp::lane_id() as usize;
    let warp_id = tid >> 5;

    // Grid-stride load + local accumulate.
    let mut acc: f32 = 0.0;
    let stride = bdim * gdim;
    let mut i = bid * bdim + tid;
    let data_ptr = data.as_ptr();
    while i < n {
        // SAFETY: bounds-checked by the while condition.
        unsafe { acc += *data_ptr.add(i); }
        i += stride;
    }

    // Warp-level butterfly reduce via shuffle-xor.
    acc += warp::shuffle_xor_f32(acc, 16);
    acc += warp::shuffle_xor_f32(acc, 8);
    acc += warp::shuffle_xor_f32(acc, 4);
    acc += warp::shuffle_xor_f32(acc, 2);
    acc += warp::shuffle_xor_f32(acc, 1);

    // Lane 0 of each warp writes its partial.
    if lane == 0 {
        unsafe { PARTIALS[warp_id] = acc; }
    }
    thread::sync_threads();

    // First warp reduces the 8 partials.
    if warp_id == 0 {
        let mut v: f32 = if lane < WARPS_PER_BLOCK {
            unsafe { PARTIALS[lane] }
        } else {
            0.0
        };
        v += warp::shuffle_xor_f32(v, 4);
        v += warp::shuffle_xor_f32(v, 2);
        v += warp::shuffle_xor_f32(v, 1);
        if lane == 0 {
            // Atomic-add the block partial into out[0].
            // SAFETY: out has >= 1 element; device-scope atomic.
            let atomic = unsafe { &*(out.as_mut_ptr() as *const DeviceAtomicF32) };
            atomic.fetch_add(v, AtomicOrdering::Relaxed);
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = load_kernel_module(&ctx, "oxide_reduction").expect("load module");

    let csv_path = concat!(env!("CARGO_MANIFEST_DIR"), "/results.csv");
    let csv_file = File::create(csv_path).expect("create results.csv");
    let mut csv = BufWriter::new(csv_file);
    writeln!(csv, "impl,kernel,N_elems,iter,gpu_ms,GB_per_s").unwrap();

    println!(
        "[oxide-reduction] sum-reduction sweep, block={}, grid={}, 1 warmup + {} iters per N",
        BLOCK, GRID, ITERS
    );

    // Allocate a max-sized input + single-f32 output.
    let n_max = *SIZES.iter().max().unwrap();
    let h_max: Vec<f32> = (0..n_max).map(|i| ((i % 7) as f32) * 0.01).collect();
    let data_dev = DeviceBuffer::from_host(&stream, &h_max).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, 1).unwrap();

    struct Row {
        n: usize,
        best_ms: f64,
        med_ms: f64,
        best_gbs: f64,
        med_gbs: f64,
        cpu_sum: f64,
        gpu_sum: f64,
        rel_err: f64,
    }
    let mut summary: Vec<Row> = Vec::new();

    for &n in &SIZES {
        // Re-upload first `n` elements if n != n_max (pattern depends on i%7, already correct).
        // Actually h_max was built for the max; first n elements are identical to a fresh length-n build.
        // So nothing to do.

        // CPU oracle (Kahan/double).
        let mut cpu_sum: f64 = 0.0;
        let mut c_err: f64 = 0.0;
        for i in 0..n {
            let x = h_max[i] as f64;
            let y = x - c_err;
            let t = cpu_sum + y;
            c_err = (t - cpu_sum) - y;
            cpu_sum = t;
        }

        let cfg = LaunchConfig {
            grid_dim: (GRID, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        // A "length-n" view we'll pass as slice(data_dev) — but the kernel uses data.len().
        // We actually need the slice to report exactly n elements so the grid-stride loop
        // knows when to stop.  Build a sub-buffer view by re-using the same underlying
        // device pointer with manually-set length.  Easiest: rely on `data.len()` via
        // DeviceBuffer::from_host for the *max*, then for smaller N we rebuild a device
        // buffer (cheap -- single memcpy).
        let view_dev: DeviceBuffer<f32> =
            DeviceBuffer::from_host(&stream, &h_max[..n]).unwrap();

        // Warmup
        for _ in 0..WARMUP {
            // Zero output
            let zero = vec![0.0_f32];
            let tmp = DeviceBuffer::from_host(&stream, &zero).unwrap();
            // Copy that single 0 into out_dev by swap-style trick; use stream.synchronize.
            // Simpler: just use cuMemsetD32Async via sys.
            stream.context().bind_to_thread().unwrap();
            unsafe {
                use cuda_core::IntoResult;
                sys::cuMemsetD32Async(
                    out_dev.cu_deviceptr(),
                    0,
                    1,
                    stream.cu_stream(),
                )
                .result()
                .expect("memset");
            }
            drop(tmp);
            let s = stream.clone();
            let m = module.clone();
            cuda_launch! {
                kernel: reduce_sum, stream: s, module: m, config: cfg,
                args: [slice(view_dev), slice_mut(out_dev)]
            }
            .unwrap();
            stream.synchronize().unwrap();
        }

        let mut ms_list: Vec<f64> = Vec::with_capacity(ITERS);
        let mut final_gpu: f32 = 0.0;
        for it in 0..ITERS {
            // Reset out_dev to 0.
            stream.context().bind_to_thread().unwrap();
            unsafe {
                use cuda_core::IntoResult;
                sys::cuMemsetD32Async(
                    out_dev.cu_deviceptr(),
                    0,
                    1,
                    stream.cu_stream(),
                )
                .result()
                .expect("memset");
            }

            let start: CudaEvent = ctx
                .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
                .expect("start event");
            let stop: CudaEvent = ctx
                .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
                .expect("stop event");

            let s = stream.clone();
            let m = module.clone();
            let t0 = Instant::now();
            start.record(&stream).expect("rec start");
            cuda_launch! {
                kernel: reduce_sum, stream: s, module: m, config: cfg,
                args: [slice(view_dev), slice_mut(out_dev)]
            }
            .unwrap();
            stop.record(&stream).expect("rec stop");
            stream.synchronize().unwrap();
            let _cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let gpu_ms = start.elapsed_ms(&stop).expect("elapsed") as f64;
            let bytes = (n as f64) * 4.0;
            let gbs = (bytes / 1.0e9) / (gpu_ms / 1000.0);

            let host_out = out_dev.to_host_vec(&stream).unwrap();
            final_gpu = host_out[0];

            writeln!(csv, "oxide,reduce_sum,{n},{it},{gpu_ms:.6},{gbs:.6}").unwrap();
            ms_list.push(gpu_ms);
            println!("[oxide-reduction] N={n} iter={it} gpu_ms={gpu_ms:.3} GB/s={gbs:.1}");
        }

        let mut sorted = ms_list.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let best = sorted[0];
        let med = sorted[sorted.len() / 2];
        let best_gbs = ((n as f64) * 4.0 / 1.0e9) / (best / 1000.0);
        let med_gbs = ((n as f64) * 4.0 / 1.0e9) / (med / 1000.0);
        let rel_err = ((final_gpu as f64 - cpu_sum).abs()) / cpu_sum.abs().max(1e-12);
        println!(
            "[oxide-reduction] N={n}  cpu_sum={cpu_sum:.6} gpu_sum={:.6} rel_err={rel_err:.3e}",
            final_gpu as f64
        );
        summary.push(Row {
            n,
            best_ms: best,
            med_ms: med,
            best_gbs,
            med_gbs,
            cpu_sum,
            gpu_sum: final_gpu as f64,
            rel_err,
        });
    }

    csv.flush().unwrap();

    println!("\n================== SUMMARY ==================");
    println!(
        "{:<14} {:>14} {:>14} {:>14} {:>14} {:>12}",
        "N_elems", "best_ms", "med_ms", "best_GB/s", "med_GB/s", "rel_err"
    );
    println!("{}", "-".repeat(84));
    for r in &summary {
        println!(
            "{:<14} {:>14.3} {:>14.3} {:>14.1} {:>14.1} {:>12.3e}",
            r.n, r.best_ms, r.med_ms, r.best_gbs, r.med_gbs, r.rel_err
        );
    }
    println!("=============================================");
    // Silence unused warnings.
    let _ = (data_dev, summary.iter().map(|r| r.cpu_sum + r.gpu_sum).sum::<f64>());
}
