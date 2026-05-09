// Wave 9: render a REAL 3D Gaussian Splatting scene by:
// 1. Parsing a public .ply file (SH degree 0 splat)
// 2. Projecting 3D->2D host-side (quaternion->R, scales->Sigma_3d, perspective
//    Jacobian -> Sigma_2d, invert for conic, SH DC -> RGB, sigmoid opacity, z-sort)
// 3. Feeding the 2D arrays to the SAME oxide-3dgs-mini kernel (unchanged).
//
// The kernel (copied from oxide-3dgs-mini) is GOOD AS-IS.

#![feature(core_intrinsics)]
#![allow(internal_features)]

use cuda_core::{CudaContext, CudaEvent, DeviceBuffer, LaunchConfig, sys};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::{cuda_launch, load_kernel_module};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::time::Instant;

const W: u32 = 800;
const H: u32 = 800;
const BS: u32 = 16;

// ---------- Kernel (identical to oxide-3dgs-mini) ----------

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
            let alpha = opacity[i] * unsafe { core::intrinsics::expf32(power) };
            if alpha >= 1.0 / 255.0 {
                let alpha_clamped = if alpha > 0.99 { 0.99 } else { alpha };
                let weight = alpha_clamped * transmittance;
                accum_r = accum_r + weight * color_r[i];
                accum_g = accum_g + weight * color_g[i];
                accum_b = accum_b + weight * color_b[i];
                transmittance = transmittance * (1.0 - alpha_clamped);
                if transmittance < 0.0001 {
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

// ---------- Host-side PLY parsing ----------

#[derive(Clone, Copy, Debug)]
struct RawGaussian {
    x: f32, y: f32, z: f32,
    f_dc: [f32; 3],
    opacity_logit: f32,
    scale: [f32; 3],
    rot: [f32; 4], // w, x, y, z (COLMAP/3DGS convention = rot_0..3)
}

fn parse_ply(path: &str) -> Vec<RawGaussian> {
    let mut f = File::open(path).expect("open ply");
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();

    // Locate end_header
    let needle = b"end_header\n";
    let mut header_end = 0;
    for i in 0..buf.len() - needle.len() {
        if &buf[i..i + needle.len()] == needle {
            header_end = i + needle.len();
            break;
        }
    }
    assert!(header_end > 0, "no end_header");
    let header = std::str::from_utf8(&buf[..header_end]).unwrap();
    // Parse vertex count + property order
    let mut n_vertex = 0usize;
    let mut props: Vec<String> = Vec::new();
    for line in header.lines() {
        if let Some(rest) = line.strip_prefix("element vertex ") {
            n_vertex = rest.trim().parse().unwrap();
        } else if let Some(rest) = line.strip_prefix("property float ") {
            props.push(rest.trim().to_string());
        }
    }
    let nprops = props.len();
    println!("PLY header: {} vertices, {} float props: {:?}",
             n_vertex, nprops, props);

    // Helper to find property index
    let idx = |name: &str| -> Option<usize> {
        props.iter().position(|p| p == name)
    };

    let ix = idx("x").unwrap();
    let iy = idx("y").unwrap();
    let iz = idx("z").unwrap();
    let ifdc0 = idx("f_dc_0").unwrap();
    let ifdc1 = idx("f_dc_1").unwrap();
    let ifdc2 = idx("f_dc_2").unwrap();
    let iop = idx("opacity").unwrap();
    let is0 = idx("scale_0").unwrap();
    let is1 = idx("scale_1").unwrap();
    let is2 = idx("scale_2").unwrap();
    let ir0 = idx("rot_0").unwrap();
    let ir1 = idx("rot_1").unwrap();
    let ir2 = idx("rot_2").unwrap();
    let ir3 = idx("rot_3").unwrap();

    let body = &buf[header_end..];
    let expected_len = n_vertex * nprops * 4;
    assert_eq!(body.len(), expected_len, "body size mismatch");

    let mut out = Vec::with_capacity(n_vertex);
    for i in 0..n_vertex {
        let base = i * nprops * 4;
        let read_f = |pi: usize| -> f32 {
            let off = base + pi * 4;
            f32::from_le_bytes([body[off], body[off + 1], body[off + 2], body[off + 3]])
        };
        out.push(RawGaussian {
            x: read_f(ix), y: read_f(iy), z: read_f(iz),
            f_dc: [read_f(ifdc0), read_f(ifdc1), read_f(ifdc2)],
            opacity_logit: read_f(iop),
            scale: [read_f(is0), read_f(is1), read_f(is2)],
            rot:   [read_f(ir0), read_f(ir1), read_f(ir2), read_f(ir3)],
        });
    }
    out
}

// ---------- Host-side projection ----------

#[derive(Clone, Copy)]
struct ProjGaussian {
    mx: f32, my: f32,
    conic_xx: f32, conic_xy: f32, conic_yy: f32,
    opacity: f32,
    r: f32, g: f32, b: f32,
    depth: f32,
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// Quaternion (w, x, y, z) -> 3x3 rotation, row-major stored as [9].
fn quat_to_mat3(q: [f32; 4]) -> [f32; 9] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt().max(1e-8);
    let w = q[0] / n;
    let x = q[1] / n;
    let y = q[2] / n;
    let z = q[3] / n;
    [
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),       2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),       2.0 * (y * z + w * x),       1.0 - 2.0 * (x * x + y * y),
    ]
}

// 3x3 M = A*B  (row-major 9-vec)
fn mat3_mul(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
    let mut r = [0.0_f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 { s += a[i * 3 + k] * b[k * 3 + j]; }
            r[i * 3 + j] = s;
        }
    }
    r
}
fn mat3_transpose(a: &[f32; 9]) -> [f32; 9] {
    [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]
}

// Return a 3x3 covariance. s = exp(scale).
fn cov3d(rot: [f32; 4], scale: [f32; 3]) -> [f32; 9] {
    let r = quat_to_mat3(rot);
    let rt = mat3_transpose(&r);
    let sx = scale[0].exp();
    let sy = scale[1].exp();
    let sz = scale[2].exp();
    // S^2 diag
    let s2 = [
        sx * sx, 0.0, 0.0,
        0.0, sy * sy, 0.0,
        0.0, 0.0, sz * sz,
    ];
    let rs2 = mat3_mul(&r, &s2);
    mat3_mul(&rs2, &rt)
}

#[derive(Clone, Copy)]
struct CamPose {
    // 3x3 row-major world-to-camera rotation W, translation t_wc.
    w: [f32; 9],
    t: [f32; 3],
    fx: f32, fy: f32, cx: f32, cy: f32,
}

fn project_all(raws: &[RawGaussian], cam: &CamPose) -> (Vec<ProjGaussian>, ProjStats) {
    let mut out = Vec::with_capacity(raws.len());
    let mut n_culled_behind = 0usize;
    let mut n_culled_far = 0usize;
    let mut n_culled_bad_cov = 0usize;
    let mut depth_min = f32::INFINITY;
    let mut depth_max = f32::NEG_INFINITY;
    let mut mean2d_min = (f32::INFINITY, f32::INFINITY);
    let mut mean2d_max = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    let mut conic_scales: Vec<f32> = Vec::new();

    for g in raws {
        // World -> camera: p_cam = W * p_world + t
        let p = [g.x, g.y, g.z];
        let pc = [
            cam.w[0] * p[0] + cam.w[1] * p[1] + cam.w[2] * p[2] + cam.t[0],
            cam.w[3] * p[0] + cam.w[4] * p[1] + cam.w[5] * p[2] + cam.t[1],
            cam.w[6] * p[0] + cam.w[7] * p[1] + cam.w[8] * p[2] + cam.t[2],
        ];
        if pc[2] < 0.1 { n_culled_behind += 1; continue; }
        if pc[2] > 100.0 { n_culled_far += 1; continue; }

        let mx = cam.fx * pc[0] / pc[2] + cam.cx;
        let my = cam.fy * pc[1] / pc[2] + cam.cy;

        // 3D cov in world frame
        let sigma_w = cov3d(g.rot, g.scale);
        // Rotate into camera frame: Sigma_cam = W * Sigma_w * W^T
        let wt = mat3_transpose(&cam.w);
        let tmp = mat3_mul(&cam.w, &sigma_w);
        let sigma_cam = mat3_mul(&tmp, &wt);

        // Jacobian of perspective proj at p_cam  (2x3):
        // J = [[fx/z, 0, -fx*x/z^2], [0, fy/z, -fy*y/z^2]]
        let z = pc[2];
        let z2 = z * z;
        let j00 = cam.fx / z;
        let j02 = -cam.fx * pc[0] / z2;
        let j11 = cam.fy / z;
        let j12 = -cam.fy * pc[1] / z2;
        // Sigma_2d = J * Sigma_cam * J^T  (2x2)
        // Let M = J * Sigma_cam  (2x3). Row 0: j00*row0 + j02*row2.  Row 1: j11*row1 + j12*row2
        let r0 = [sigma_cam[0], sigma_cam[1], sigma_cam[2]];
        let r1 = [sigma_cam[3], sigma_cam[4], sigma_cam[5]];
        let r2 = [sigma_cam[6], sigma_cam[7], sigma_cam[8]];
        let m0 = [j00 * r0[0] + j02 * r2[0], j00 * r0[1] + j02 * r2[1], j00 * r0[2] + j02 * r2[2]];
        let m1 = [j11 * r1[0] + j12 * r2[0], j11 * r1[1] + j12 * r2[1], j11 * r1[2] + j12 * r2[2]];
        // Sigma_2d = M * J^T; J^T column-j = (j00,0) for col0, (0,j11) col1, (j02,j12) col2? no
        // J is 2x3, J^T is 3x2. Rows of J^T: row0=[j00,0], row1=[0,j11], row2=[j02,j12].
        // Sigma_2d[i][j] = sum_k M[i][k] * JT[k][j]
        let a = m0[0] * j00 + m0[1] * 0.0   + m0[2] * j02;
        let b = m0[0] * 0.0  + m0[1] * j11  + m0[2] * j12;
        let c = m1[0] * 0.0  + m1[1] * j11  + m1[2] * j12;
        // (Sigma_2d row1 col0 should equal b by symmetry; skip)

        // Small anti-aliasing blur (standard 3DGS trick)
        let a_aa = a + 0.3;
        let c_aa = c + 0.3;
        let b_aa = b;

        let det = a_aa * c_aa - b_aa * b_aa;
        if det <= 0.0 || !det.is_finite() {
            n_culled_bad_cov += 1;
            continue;
        }
        let inv_det = 1.0 / det;
        let conic_xx = c_aa * inv_det;
        let conic_xy = -b_aa * inv_det;
        let conic_yy = a_aa * inv_det;

        // SH DC -> color.
        let c0 = 0.28209479_f32;
        let col_r = (c0 * g.f_dc[0] + 0.5).clamp(0.0, 1.0);
        let col_g = (c0 * g.f_dc[1] + 0.5).clamp(0.0, 1.0);
        let col_b = (c0 * g.f_dc[2] + 0.5).clamp(0.0, 1.0);
        let op = sigmoid(g.opacity_logit);

        depth_min = depth_min.min(pc[2]);
        depth_max = depth_max.max(pc[2]);
        mean2d_min.0 = mean2d_min.0.min(mx);
        mean2d_min.1 = mean2d_min.1.min(my);
        mean2d_max.0 = mean2d_max.0.max(mx);
        mean2d_max.1 = mean2d_max.1.max(my);
        conic_scales.push((a_aa + c_aa) * 0.5);

        out.push(ProjGaussian {
            mx, my,
            conic_xx, conic_xy, conic_yy,
            opacity: op,
            r: col_r, g: col_g, b: col_b,
            depth: pc[2],
        });
    }

    // Sort by depth ascending (closest first = drawn first, per kernel's front-to-back blend).
    out.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());

    conic_scales.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cs_med = if conic_scales.is_empty() { 0.0 } else { conic_scales[conic_scales.len() / 2] };

    let stats = ProjStats {
        n_total: raws.len(),
        n_projected: out.len(),
        n_culled_behind,
        n_culled_far,
        n_culled_bad_cov,
        depth_min, depth_max,
        mean2d_min, mean2d_max,
        conic_scale_median: cs_med,
    };
    (out, stats)
}

#[derive(Debug)]
struct ProjStats {
    n_total: usize,
    n_projected: usize,
    n_culled_behind: usize,
    n_culled_far: usize,
    n_culled_bad_cov: usize,
    depth_min: f32, depth_max: f32,
    mean2d_min: (f32, f32),
    mean2d_max: (f32, f32),
    conic_scale_median: f32,
}

// ---------- Output / I/O ----------

fn save_ppm(path: &str, pr: &[f32], pg: &[f32], pb: &[f32], w: u32, h: u32) {
    let f = File::create(path).expect("create ppm");
    let mut bw = BufWriter::new(f);
    writeln!(bw, "P6").unwrap();
    writeln!(bw, "{} {}", w, h).unwrap();
    writeln!(bw, "255").unwrap();
    let n = (w * h) as usize;
    let mut buf = Vec::with_capacity(n * 3);
    for i in 0..n {
        let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        buf.push(to_u8(pr[i]));
        buf.push(to_u8(pg[i]));
        buf.push(to_u8(pb[i]));
    }
    bw.write_all(&buf).unwrap();
}

// ---------- Render ----------

fn render_cam(
    ctx: &std::sync::Arc<CudaContext>,
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &std::sync::Arc<cuda_core::CudaModule>,
    raws: &[RawGaussian],
    cam: &CamPose,
    label: &str,
    out_ppm: &str,
) {
    let (proj, stats) = project_all(raws, cam);
    println!("[{label}] project stats:");
    println!("  total={}, projected={}, culled_behind={}, culled_far={}, culled_bad_cov={}",
             stats.n_total, stats.n_projected, stats.n_culled_behind, stats.n_culled_far, stats.n_culled_bad_cov);
    println!("  depth_range=[{:.3}, {:.3}]", stats.depth_min, stats.depth_max);
    println!("  mean2d_range=[({:.1},{:.1}) .. ({:.1},{:.1})]",
             stats.mean2d_min.0, stats.mean2d_min.1, stats.mean2d_max.0, stats.mean2d_max.1);
    println!("  conic_scale_median={:.4}", stats.conic_scale_median);

    if proj.is_empty() {
        println!("[{label}] no gaussians passed projection; skipping render");
        return;
    }

    let n = proj.len();
    let mut mx = Vec::with_capacity(n);
    let mut my = Vec::with_capacity(n);
    let mut cxx = Vec::with_capacity(n);
    let mut cxy = Vec::with_capacity(n);
    let mut cyy = Vec::with_capacity(n);
    let mut op = Vec::with_capacity(n);
    let mut cr = Vec::with_capacity(n);
    let mut cg = Vec::with_capacity(n);
    let mut cb = Vec::with_capacity(n);
    for g in &proj {
        mx.push(g.mx); my.push(g.my);
        cxx.push(g.conic_xx); cxy.push(g.conic_xy); cyy.push(g.conic_yy);
        op.push(g.opacity);
        cr.push(g.r); cg.push(g.g); cb.push(g.b);
    }

    let d_mx = DeviceBuffer::from_host(stream, &mx).unwrap();
    let d_my = DeviceBuffer::from_host(stream, &my).unwrap();
    let d_cxx = DeviceBuffer::from_host(stream, &cxx).unwrap();
    let d_cxy = DeviceBuffer::from_host(stream, &cxy).unwrap();
    let d_cyy = DeviceBuffer::from_host(stream, &cyy).unwrap();
    let d_op = DeviceBuffer::from_host(stream, &op).unwrap();
    let d_cr = DeviceBuffer::from_host(stream, &cr).unwrap();
    let d_cg = DeviceBuffer::from_host(stream, &cg).unwrap();
    let d_cb = DeviceBuffer::from_host(stream, &cb).unwrap();

    let pixels = (W * H) as usize;
    let mut d_out_r = DeviceBuffer::<f32>::zeroed(stream, pixels).unwrap();
    let mut d_out_g = DeviceBuffer::<f32>::zeroed(stream, pixels).unwrap();
    let mut d_out_b = DeviceBuffer::<f32>::zeroed(stream, pixels).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (W.div_ceil(BS), H.div_ceil(BS), 1),
        block_dim: (BS, BS, 1),
        shared_mem_bytes: 0,
    };
    let n_arg: u32 = n as u32;

    // Warmup
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

    // Timed
    let start: CudaEvent = ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT)).unwrap();
    let stop: CudaEvent = ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT)).unwrap();
    let s = stream.clone();
    let m = module.clone();
    let t0 = Instant::now();
    start.record(stream).unwrap();
    cuda_launch! {
        kernel: rasterize_2dgs, stream: s, module: m, config: cfg,
        args: [slice(&d_mx), slice(&d_my),
               slice(&d_cxx), slice(&d_cxy), slice(&d_cyy),
               slice(&d_op),
               slice(&d_cr), slice(&d_cg), slice(&d_cb),
               n_arg, W, H,
               slice_mut(&mut d_out_r), slice_mut(&mut d_out_g), slice_mut(&mut d_out_b)]
    }.unwrap();
    stop.record(stream).unwrap();
    stream.synchronize().unwrap();
    let gpu_ms = start.elapsed_ms(&stop).unwrap() as f64;
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("[{label}] render: N={} gpu_ms={:.3} cpu_wall_ms={:.3}", n, gpu_ms, cpu_ms);

    let h_r = d_out_r.to_host_vec(stream).unwrap();
    let h_g = d_out_g.to_host_vec(stream).unwrap();
    let h_b = d_out_b.to_host_vec(stream).unwrap();

    // Image sanity: non-zero pixel fraction.
    let mut nonzero = 0usize;
    for i in 0..h_r.len() {
        if h_r[i] > 0.01 || h_g[i] > 0.01 || h_b[i] > 0.01 { nonzero += 1; }
    }
    println!("[{label}] nonzero pixels: {}/{} = {:.1}%",
             nonzero, h_r.len(), 100.0 * nonzero as f32 / h_r.len() as f32);

    save_ppm(out_ppm, &h_r, &h_g, &h_b, W, H);
    println!("[{label}] wrote {out_ppm}");
}

fn main() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    // Wave 10: canonical 3DGS plush-toy scene (Utsuho, 53k gaussians, SH deg 3,
    // source: solaaaa/sample-gaussian-splats on HuggingFace).
    let scene_name = "utsuho_plush";
    let ply_path = format!("{}/scenes/{}.ply", manifest, scene_name);
    println!("Loading {}", ply_path);
    let raws = parse_ply(&ply_path);
    println!("Parsed {} gaussians", raws.len());

    // Inspect geometry: print centroid, extents.
    let mut cx = 0.0_f32; let mut cy = 0.0_f32; let mut cz = 0.0_f32;
    let mut mn = [f32::INFINITY; 3]; let mut mx = [f32::NEG_INFINITY; 3];
    for g in &raws {
        cx += g.x; cy += g.y; cz += g.z;
        mn[0] = mn[0].min(g.x); mn[1] = mn[1].min(g.y); mn[2] = mn[2].min(g.z);
        mx[0] = mx[0].max(g.x); mx[1] = mx[1].max(g.y); mx[2] = mx[2].max(g.z);
    }
    let n = raws.len() as f32;
    cx /= n; cy /= n; cz /= n;
    println!("Scene centroid: ({:.3}, {:.3}, {:.3})", cx, cy, cz);
    println!("Scene bbox   : min=({:.3},{:.3},{:.3}) max=({:.3},{:.3},{:.3})",
             mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]);
    let extent = ((mx[0] - mn[0]).powi(2) + (mx[1] - mn[1]).powi(2) + (mx[2] - mn[2]).powi(2)).sqrt();
    println!("Scene diag   : {:.3}", extent);

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = load_kernel_module(&ctx, "oxide_3dgs_real").expect("load module");

    // Camera intrinsics for 800x800 render.
    let fx = 800.0_f32;
    let fy = 800.0_f32;
    let cx_p = 400.0_f32;
    let cy_p = 400.0_f32;

    // Place camera so the object fits: camera at centroid + (0,0,-distance) looking +z.
    // In 3DGS/COLMAP convention the camera looks down +z with W=I, so the object must
    // have positive z in camera space. We construct p_cam = W*(p_world - cam_origin).
    // For W=I we just need camera origin such that cz_rel = g.z - origin_z > 0 for most gaussians.
    // Simpler: set t = -cam_origin (i.e. translate world so camera is at origin).
    //
    // Distance: extent * 1.5 from centroid.
    let dist = extent * 1.5;

    // Try multiple camera poses; pick the first one that produces a sane non-blank image.
    // Pose A: look at scene from -Z direction (camera at centroid - dist on Z axis).
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    // camera origin at (cx, cy, cz - dist); then t = -W * origin = -origin  (for W=I).
    let cam_a = CamPose {
        w: identity,
        t: [-cx, -cy, -(cz - dist)],
        fx, fy, cx: cx_p, cy: cy_p,
    };
    let cam_b = CamPose {
        w: identity,
        t: [-cx, -cy, -(cz + dist)],  // other side
        fx, fy, cx: cx_p, cy: cy_p,
    };
    // Flip y (3DGS world often has y-down; OpenGL/COLMAP have y-up)
    let flip_y = [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0];
    let cam_c = CamPose {
        w: flip_y,
        t: [-cx, cy, -(cz - dist)],
        fx, fy, cx: cx_p, cy: cy_p,
    };
    // COLMAP: y-down x-right z-forward. 3DGS stores world in COLMAP space. So pose A should work.
    // Rotate 180° around Y: camera looks from +Z side
    let rot_y180 = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
    let cam_d = CamPose {
        w: rot_y180,
        // camera origin at (cx, cy, cz + dist), so in cam-frame object is in front (+z cam)
        // p_cam = rot_y180 * (p_world - origin); let origin = (cx, cy, cz+dist)
        // t = -rot_y180 * origin = -(-cx, cy, -(cz+dist)) = (cx, -cy, cz+dist)
        t: [cx, -cy, cz + dist],
        fx, fy, cx: cx_p, cy: cy_p,
    };

    render_cam(&ctx, &stream, &module, &raws, &cam_a, "camA_minusZ", &format!("{}/output_{}_A.ppm", manifest, scene_name));
    render_cam(&ctx, &stream, &module, &raws, &cam_b, "camB_plusZ_noflip", &format!("{}/output_{}_B.ppm", manifest, scene_name));
    render_cam(&ctx, &stream, &module, &raws, &cam_c, "camC_flipY", &format!("{}/output_{}_C.ppm", manifest, scene_name));
    render_cam(&ctx, &stream, &module, &raws, &cam_d, "camD_roty180", &format!("{}/output_{}_D.ppm", manifest, scene_name));
}
