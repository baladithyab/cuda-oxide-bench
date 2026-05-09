// Rudimentary 2D Gaussian Splatting forward rasterizer in cuda-oxide.
// Wave 8: proof-of-life test of cuda-oxide's expressiveness on a non-matmul
// non-reduction kernel with control flow, early-exit, and libdevice math.
//
// Algorithm: per pixel, iterate over all N gaussians in pre-sorted order,
// front-to-back alpha blend. No tile binning, no SH, no 3D projection.
// See ANALYSIS.md for simplifications.

#![feature(core_intrinsics)]
#![allow(internal_features)]

use cuda_core::{CudaContext, CudaEvent, DeviceBuffer, LaunchConfig, sys};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::{cuda_launch, load_kernel_module};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

const W: u32 = 256;
const H: u32 = 256;
const N: usize = 512;
const BS: u32 = 16;

#[kernel]
pub fn rasterize_2dgs(
    means_x: &[f32],
    means_y: &[f32],
    conic_xx: &[f32],
    conic_xy: &[f32],
    conic_yy: &[f32],
    opacity: &[f32],
    color_r: &[f32],
    color_g: &[f32],
    color_b: &[f32],
    n_gaussians: u32,
    width: u32,
    height: u32,
    mut out_r: DisjointSlice<f32>,
    mut out_g: DisjointSlice<f32>,
    mut out_b: DisjointSlice<f32>,
) {
    let px = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
    let py = thread::blockIdx_y() * thread::blockDim_y() + thread::threadIdx_y();
    if px >= width || py >= height {
        return;
    }
    let pxf = px as f32;
    let pyf = py as f32;
    let pidx = (py * width + px) as usize;
    let n = n_gaussians as usize;

    let mut accum_r: f32 = 0.0;
    let mut accum_g: f32 = 0.0;
    let mut accum_b: f32 = 0.0;
    let mut transmittance: f32 = 1.0;

    let mut i: usize = 0;
    while i < n {
        let dx = pxf - means_x[i];
        let dy = pyf - means_y[i];
        let power = -0.5
            * (conic_xx[i] * dx * dx + 2.0 * conic_xy[i] * dx * dy + conic_yy[i] * dy * dy);
        if power <= 0.0 {
            // exp(power) via libdevice __nv_expf (cuda-oxide lowers expf32 to this).
            let alpha = opacity[i] * unsafe { core::intrinsics::expf32(power) };
            if alpha >= 1.0 / 255.0 {
                let alpha_clamped = if alpha > 0.99 { 0.99 } else { alpha };
                let weight = alpha_clamped * transmittance;
                accum_r = accum_r + weight * color_r[i];
                accum_g = accum_g + weight * color_g[i];
                accum_b = accum_b + weight * color_b[i];
                transmittance = transmittance * (1.0 - alpha_clamped);
                if transmittance < 0.0001 {
                    // Write + early-exit.
                    unsafe {
                        *out_r.as_mut_ptr().add(pidx) = accum_r;
                        *out_g.as_mut_ptr().add(pidx) = accum_g;
                        *out_b.as_mut_ptr().add(pidx) = accum_b;
                    }
                    return;
                }
            }
        }
        i += 1;
    }

    unsafe {
        *out_r.as_mut_ptr().add(pidx) = accum_r;
        *out_g.as_mut_ptr().add(pidx) = accum_g;
        *out_b.as_mut_ptr().add(pidx) = accum_b;
    }
}

// ---------- Host side ----------

// Tiny LCG so builds are deterministic without pulling `rand`.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f01(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f01() * (hi - lo)
    }
}

#[derive(Clone, Copy)]
struct Gaussian {
    mx: f32, my: f32,
    sx: f32, sy: f32,              // stddevs (diagonal covariance)
    opacity: f32,
    r: f32, g: f32, b: f32,
    depth: f32,
}

fn generate_gaussians() -> Vec<Gaussian> {
    let mut rng = Lcg::new(0xC0FFEE_1234_5678);
    let mut v: Vec<Gaussian> = (0..N).map(|_| {
        Gaussian {
            mx: rng.range(0.0, W as f32),
            my: rng.range(0.0, H as f32),
            sx: rng.range(3.0, 15.0),
            sy: rng.range(3.0, 15.0),
            opacity: rng.range(0.3, 1.0),
            r: rng.next_f01(),
            g: rng.next_f01(),
            b: rng.next_f01(),
            depth: 0.0,
        }
    }).collect();
    // Plant a "known" dominant gaussian at (128,128): big sigma, full opacity, red.
    v[0] = Gaussian {
        mx: 128.0, my: 128.0, sx: 30.0, sy: 30.0,
        opacity: 1.0, r: 1.0, g: 0.05, b: 0.05, depth: -1e9,
    };
    for g in v.iter_mut() { g.depth = g.my + 0.1 * g.mx; }
    v.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());
    v
}

fn save_ppm(path: &str, pixels_r: &[f32], pixels_g: &[f32], pixels_b: &[f32], w: u32, h: u32) {
    let f = File::create(path).expect("create ppm");
    let mut bw = BufWriter::new(f);
    writeln!(bw, "P6").unwrap();
    writeln!(bw, "{} {}", w, h).unwrap();
    writeln!(bw, "255").unwrap();
    let n = (w * h) as usize;
    let mut buf: Vec<u8> = Vec::with_capacity(n * 3);
    for i in 0..n {
        let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        buf.push(to_u8(pixels_r[i]));
        buf.push(to_u8(pixels_g[i]));
        buf.push(to_u8(pixels_b[i]));
    }
    bw.write_all(&buf).unwrap();
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = load_kernel_module(&ctx, "oxide_3dgs_mini").expect("load module");

    // Host data.
    let gs = generate_gaussians();
    let mut mx = Vec::with_capacity(N);
    let mut my = Vec::with_capacity(N);
    let mut cxx = Vec::with_capacity(N);
    let mut cxy = Vec::with_capacity(N);
    let mut cyy = Vec::with_capacity(N);
    let mut op  = Vec::with_capacity(N);
    let mut cr  = Vec::with_capacity(N);
    let mut cg  = Vec::with_capacity(N);
    let mut cb  = Vec::with_capacity(N);
    for g in &gs {
        // Σ = [[sx² , 0], [0, sy²]]; b=0, a=sx², c=sy², det = sx²*sy²
        let a = g.sx * g.sx;
        let c = g.sy * g.sy;
        let b = 0.0_f32;
        let det = a * c - b * b;
        mx.push(g.mx); my.push(g.my);
        cxx.push(c / det);
        cxy.push(-b / det);
        cyy.push(a / det);
        op.push(g.opacity);
        cr.push(g.r); cg.push(g.g); cb.push(g.b);
    }

    let d_mx  = DeviceBuffer::from_host(&stream, &mx).unwrap();
    let d_my  = DeviceBuffer::from_host(&stream, &my).unwrap();
    let d_cxx = DeviceBuffer::from_host(&stream, &cxx).unwrap();
    let d_cxy = DeviceBuffer::from_host(&stream, &cxy).unwrap();
    let d_cyy = DeviceBuffer::from_host(&stream, &cyy).unwrap();
    let d_op  = DeviceBuffer::from_host(&stream, &op).unwrap();
    let d_cr  = DeviceBuffer::from_host(&stream, &cr).unwrap();
    let d_cg  = DeviceBuffer::from_host(&stream, &cg).unwrap();
    let d_cb  = DeviceBuffer::from_host(&stream, &cb).unwrap();

    let pixels = (W * H) as usize;
    let mut d_out_r = DeviceBuffer::<f32>::zeroed(&stream, pixels).unwrap();
    let mut d_out_g = DeviceBuffer::<f32>::zeroed(&stream, pixels).unwrap();
    let mut d_out_b = DeviceBuffer::<f32>::zeroed(&stream, pixels).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (W.div_ceil(BS), H.div_ceil(BS), 1),
        block_dim: (BS, BS, 1),
        shared_mem_bytes: 0,
    };
    let n_arg: u32 = N as u32;

    // Warmup.
    {
        let s = stream.clone();
        let m = module.clone();
        cuda_launch! {
            kernel: rasterize_2dgs, stream: s, module: m, config: cfg,
            args: [slice(&d_mx), slice(&d_my),
                   slice(&d_cxx), slice(&d_cxy), slice(&d_cyy),
                   slice(&d_op),
                   slice(&d_cr), slice(&d_cg), slice(&d_cb),
                   n_arg, W, H,
                   slice_mut(&mut d_out_r), slice_mut(&mut d_out_g), slice_mut(&mut d_out_b)]
        }.unwrap();
        stream.synchronize().unwrap();
    }

    // Timed iters.
    let mut times_ms = Vec::<f64>::new();
    for i in 0..5 {
        let start: CudaEvent = ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT)).unwrap();
        let stop:  CudaEvent = ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT)).unwrap();
        let s = stream.clone();
        let m = module.clone();
        let t0 = Instant::now();
        start.record(&stream).unwrap();
        cuda_launch! {
            kernel: rasterize_2dgs, stream: s, module: m, config: cfg,
            args: [slice(&d_mx), slice(&d_my),
                   slice(&d_cxx), slice(&d_cxy), slice(&d_cyy),
                   slice(&d_op),
                   slice(&d_cr), slice(&d_cg), slice(&d_cb),
                   n_arg, W, H,
                   slice_mut(&mut d_out_r), slice_mut(&mut d_out_g), slice_mut(&mut d_out_b)]
        }.unwrap();
        stop.record(&stream).unwrap();
        stream.synchronize().unwrap();
        let gpu_ms = start.elapsed_ms(&stop).unwrap() as f64;
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("iter {i}: gpu_ms={gpu_ms:.3} cpu_wall_ms={cpu_ms:.3}");
        times_ms.push(gpu_ms);
    }

    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times_ms[times_ms.len() / 2];
    let best = times_ms[0];
    // ~20 flops/eval * N evals per pixel * W*H pixels.
    let flops = 20.0 * (N as f64) * (W as f64) * (H as f64);
    let tflops = (flops / 1e12) / (med / 1000.0);
    println!("median_ms={med:.3}  best_ms={best:.3}  ~{tflops:.3} TFLOP/s");

    // Readback + sanity.
    let h_r = d_out_r.to_host_vec(&stream).unwrap();
    let h_g = d_out_g.to_host_vec(&stream).unwrap();
    let h_b = d_out_b.to_host_vec(&stream).unwrap();
    let center = (128 * (W as usize)) + 128;
    println!("pixel(128,128) = ({:.3}, {:.3}, {:.3})  [expect roughly red-dominant]",
             h_r[center], h_g[center], h_b[center]);
    let dominant = h_r[center] > h_g[center] && h_r[center] > h_b[center];
    println!("dominant-red sanity: {}", if dominant { "PASS" } else { "FAIL" });

    let ppm_path = concat!(env!("CARGO_MANIFEST_DIR"), "/output.ppm");
    save_ppm(ppm_path, &h_r, &h_g, &h_b, W, H);
    println!("wrote {}", ppm_path);
}
