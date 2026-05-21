// Wave C2.4 — oxide-attn-gdn-tma:
// Rust-CUDA (cuda-oxide) Gated DeltaNet decode using TMA
// (cp.async.bulk.tensor.2d) for the dominant S_in slab load.
//
// Baselines on the same RTX 5090 / sm_120 box:
//   oxide-attn-gdn        ~276 GB/s   FFMA, no TC, plain LDG.E.128 over S_in.
//   cuda-attn-gdn-tma    ~1032 GB/s   nvcc + TMA, single UTMALDG per tile.
//
// Hypothesis: cuda-oxide already exposes both halves we need:
//   - Device side: `cuda_device::tma::cp_async_bulk_tensor_2d_g2s` (intrinsic
//     that lowers to `cp.async.bulk.tensor.2d.shared::cluster.global.tile…`).
//   - Host side : `cuda_core::sys::cuTensorMapEncodeTiled` (driver-API FFI,
//     same as the C++ path) plus `Barrier`/`mbarrier_*` helpers.
// Thanks to oxide-tma-copy already validating the round-trip, we can lift its
// pattern straight into the GDN decode.
//
// Algorithm shape — identical to oxide-attn-gdn (W1d):
//   gridDim  = (B*H, D_V / BV)
//   blockDim = (D_K, 1, 1)        -- one thread per d_k row, BV cols per thread
//
// Kernel changes vs oxide-attn-gdn:
//   1. Drop the `s_in: &[f32]` slice param. Add `tensor_map_s_in:
//      *const TmaDescriptor` (encoded host-side).
//   2. Add a shared-memory tile `TILE_S[D_K * BV]` (128B aligned) and a
//      single `Barrier`. Thread 0 issues the TMA + arrive_expect_tx; others
//      arrive normally; all spin on `mbarrier_try_wait`.
//   3. After the TMA lands, each thread reads its row of S_in from
//      `TILE_S[tid * BV + j]` for j in 0..BV — a shared-memory LDS
//      instead of a global LDG.E.128.
//
// Acceptance:
//   - Compiles via `cargo oxide build --arch sm_120`.
//   - Correctness on qwen3_next_decode (and the 64-dim correctness shape):
//     max_abs(o) ≤ 1e-3, max_abs(S_out) ≤ 1e-3 vs the f32 PyTorch reference.
//   - SASS shows UTMALDG.2D > 0.

#![feature(core_intrinsics)]
#![allow(internal_features)]

use cuda_core::{
    CudaContext, DeviceBuffer, LaunchConfig, sys,
    sys::{
        CUtensorMap, CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE, cuTensorMapEncodeTiled,
    },
};
use cuda_device::barrier::{
    Barrier, fence_proxy_async_shared_cta, mbarrier_arrive, mbarrier_arrive_expect_tx,
    mbarrier_init, mbarrier_try_wait,
};
use cuda_device::tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::{cuda_launch, load_kernel_module};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::mem::MaybeUninit;
use std::path::Path;
use std::time::Instant;

const BV: usize = 32;

// ─────────────────────────────────────────────────────────────────────────────
// Kernels
// ─────────────────────────────────────────────────────────────────────────────
//
// Two kernels: dk64 (correctness) and dk256 (qwen3-next decode). Same as the
// FFMA baseline. Only the S_in path is replaced with TMA.

/// GDN decode kernel for d_k = 64 (correctness shape). Block = 64 threads.
/// Grid: (B*H, d_v / BV).  d_k=64, d_v=64, BV=32 → grid_y = 2.
/// TMA tile = (D_K rows, BV cols) = (64, 32) = 2048 f32 = 8 KiB.
#[kernel]
pub fn gdn_decode_dk64_tma(
    q: &[f32],     // (B*H, 64)
    k: &[f32],     // (B*H, 64)
    v: &[f32],     // (B*H, 64)
    alpha: &[f32], // (B*H,)
    beta: &[f32],  // (B*H,)
    tensor_map_s_in: *const TmaDescriptor,
    mut s_out: DisjointSlice<f32>,
    mut o: DisjointSlice<f32>,
    _d_k: u32,
    _d_v: u32,
) {
    // Per-block shared mem.
    // TMA destination: must be 128-byte aligned.
    static mut TILE_S: SharedArray<f32, { 64 * BV }, 128> = SharedArray::UNINIT;
    static mut BAR: Barrier = Barrier::UNINIT;
    static mut K_SH: SharedArray<f32, 64> = SharedArray::UNINIT;
    static mut Q_SH: SharedArray<f32, 64> = SharedArray::UNINIT;
    static mut PROD: SharedArray<f32, 64> = SharedArray::UNINIT;
    static mut UVEC: SharedArray<f32, 32> = SharedArray::UNINIT;
    static mut OVEC: SharedArray<f32, 32> = SharedArray::UNINIT;
    static mut AB: SharedArray<f32, 2> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    let block_size = thread::blockDim_x();
    let bh = thread::blockIdx_x() as usize;
    let bv_idx = thread::blockIdx_y() as usize;

    // ── Initialize barrier ──
    if tid == 0 {
        unsafe {
            mbarrier_init(&raw mut BAR, block_size);
            // CRITICAL: fence so init is visible to TMA async proxy.
            fence_proxy_async_shared_cta();
        }
    }
    thread::sync_threads();

    // Load α, β.
    if tid == 0 {
        unsafe {
            AB[0] = alpha[bh];
        }
    }
    if tid == 1 {
        unsafe {
            AB[1] = beta[bh];
        }
    }

    // Broadcast q, k.
    let q_t = q[bh * 64 + tid];
    let k_t = k[bh * 64 + tid];
    unsafe {
        K_SH[tid] = k_t;
        Q_SH[tid] = q_t;
    }

    // ── Issue TMA bulk-tensor load: (D_K rows, BV cols) tile of S_in ──
    //
    // Descriptor encoded host-side as a 2D tensor with:
    //   innermost dim : D_V    (col stride = 1 elem)
    //   outer    dim  : B_H * D_K  (row stride = D_V elems)
    //   box dims      : (BV, D_K)  innermost first
    //
    // Tile origin in *elements*:
    //   coord0 = bv * BV       (column = d_v offset)
    //   coord1 = bh * D_K      (row    = which (bh, k) state row)
    let tile_bytes: u32 = (64 * BV * 4) as u32;
    if tid == 0 {
        unsafe {
            cp_async_bulk_tensor_2d_g2s(
                &raw mut TILE_S as *mut u8,
                tensor_map_s_in,
                (bv_idx * BV) as i32,
                (bh * 64) as i32,
                &raw mut BAR,
            );
        }
    }

    // ALL threads arrive at barrier — thread 0 with expect_tx (= TMA bytes).
    let token = unsafe {
        if tid == 0 {
            mbarrier_arrive_expect_tx(&raw const BAR, 1, tile_bytes)
        } else {
            mbarrier_arrive(&raw const BAR)
        }
    };
    // Spin until the TMA hardware has deposited the tile.
    unsafe {
        while !mbarrier_try_wait(&raw const BAR, token) {}
    }
    thread::sync_threads();

    let alpha_v = unsafe { AB[0] };
    let beta_v = unsafe { AB[1] };

    // ── Read this thread's row of the tile from shared, scale by α ──
    // TILE_S layout: row k_row owns BV contiguous floats at TILE_S[k_row * BV ..].
    let mut s_scaled: [f32; BV] = [0.0; BV];
    let mut j = 0usize;
    while j < BV {
        let s = unsafe { TILE_S[tid * BV + j] };
        s_scaled[j] = s * alpha_v;
        j += 1;
    }

    // ── Reduction 1: u[j] = sum_t k_t * s_scaled[t, j] ──
    let mut jj = 0usize;
    while jj < BV {
        unsafe {
            PROD[tid] = k_t * s_scaled[jj];
        }
        thread::sync_threads();
        let mut stride = 32usize;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PROD[tid] = PROD[tid] + PROD[tid + stride];
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
        if tid == 0 {
            unsafe {
                UVEC[jj] = PROD[0];
            }
        }
        thread::sync_threads();
        jj += 1;
    }

    // ── r = v - u; S_out_row = S_scaled + β·k_t·r ──
    let bv_col0 = bv_idx * BV;
    let mut s_out_row: [f32; BV] = [0.0; BV];
    let bk = beta_v * k_t;
    let mut jj2 = 0usize;
    while jj2 < BV {
        let v_j = v[bh * 64 + bv_col0 + jj2];
        let u_j = unsafe { UVEC[jj2] };
        let r_j = v_j - u_j;
        let so = unsafe { core::intrinsics::fmuladdf32(bk, r_j, s_scaled[jj2]) };
        s_out_row[jj2] = so;
        jj2 += 1;
    }

    // ── Reduction 2: o[j] = sum_t q_t * s_out_row[t, j] ──
    let mut jj3 = 0usize;
    while jj3 < BV {
        unsafe {
            PROD[tid] = q_t * s_out_row[jj3];
        }
        thread::sync_threads();
        let mut stride = 32usize;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PROD[tid] = PROD[tid] + PROD[tid + stride];
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
        if tid == 0 {
            unsafe {
                OVEC[jj3] = PROD[0];
            }
        }
        thread::sync_threads();
        jj3 += 1;
    }

    // ── Stores ──
    let s_row_base = (bh * 64 + tid) * 64;
    let so_ptr = s_out.as_mut_ptr();
    let mut wj = 0usize;
    while wj < BV {
        unsafe {
            *so_ptr.add(s_row_base + bv_col0 + wj) = s_out_row[wj];
        }
        wj += 1;
    }
    if tid < BV {
        let o_ptr = o.as_mut_ptr();
        unsafe {
            *o_ptr.add(bh * 64 + bv_col0 + tid) = OVEC[tid];
        }
    }
}

/// GDN decode kernel for d_k = 256 (Qwen3-Next decode shape).
/// Block = 256 threads, grid = (B*H, d_v / BV).
/// TMA tile = (256, 32) = 8192 f32 = 32 KiB.
#[kernel]
pub fn gdn_decode_dk256_tma(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: &[f32],
    beta: &[f32],
    tensor_map_s_in: *const TmaDescriptor,
    mut s_out: DisjointSlice<f32>,
    mut o: DisjointSlice<f32>,
    _d_k: u32,
    _d_v: u32,
) {
    static mut TILE_S: SharedArray<f32, { 256 * BV }, 128> = SharedArray::UNINIT;
    static mut BAR: Barrier = Barrier::UNINIT;
    static mut K_SH: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut Q_SH: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut PROD: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut UVEC: SharedArray<f32, 32> = SharedArray::UNINIT;
    static mut OVEC: SharedArray<f32, 32> = SharedArray::UNINIT;
    static mut AB: SharedArray<f32, 2> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    let block_size = thread::blockDim_x();
    let bh = thread::blockIdx_x() as usize;
    let bv_idx = thread::blockIdx_y() as usize;

    if tid == 0 {
        unsafe {
            mbarrier_init(&raw mut BAR, block_size);
            fence_proxy_async_shared_cta();
        }
    }
    thread::sync_threads();

    if tid == 0 {
        unsafe {
            AB[0] = alpha[bh];
        }
    }
    if tid == 1 {
        unsafe {
            AB[1] = beta[bh];
        }
    }

    let q_t = q[bh * 256 + tid];
    let k_t = k[bh * 256 + tid];
    unsafe {
        K_SH[tid] = k_t;
        Q_SH[tid] = q_t;
    }

    let tile_bytes: u32 = (256 * BV * 4) as u32;
    if tid == 0 {
        unsafe {
            cp_async_bulk_tensor_2d_g2s(
                &raw mut TILE_S as *mut u8,
                tensor_map_s_in,
                (bv_idx * BV) as i32,
                (bh * 256) as i32,
                &raw mut BAR,
            );
        }
    }
    let token = unsafe {
        if tid == 0 {
            mbarrier_arrive_expect_tx(&raw const BAR, 1, tile_bytes)
        } else {
            mbarrier_arrive(&raw const BAR)
        }
    };
    unsafe {
        while !mbarrier_try_wait(&raw const BAR, token) {}
    }
    thread::sync_threads();

    let alpha_v = unsafe { AB[0] };
    let beta_v = unsafe { AB[1] };

    let mut s_scaled: [f32; BV] = [0.0; BV];
    let mut j = 0usize;
    while j < BV {
        let s = unsafe { TILE_S[tid * BV + j] };
        s_scaled[j] = s * alpha_v;
        j += 1;
    }

    let mut jj = 0usize;
    while jj < BV {
        unsafe {
            PROD[tid] = k_t * s_scaled[jj];
        }
        thread::sync_threads();
        let mut stride = 128usize;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PROD[tid] = PROD[tid] + PROD[tid + stride];
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
        if tid == 0 {
            unsafe {
                UVEC[jj] = PROD[0];
            }
        }
        thread::sync_threads();
        jj += 1;
    }

    let bv_col0 = bv_idx * BV;
    let mut s_out_row: [f32; BV] = [0.0; BV];
    let bk = beta_v * k_t;
    let mut jj2 = 0usize;
    while jj2 < BV {
        let v_j = v[bh * 256 + bv_col0 + jj2];
        let u_j = unsafe { UVEC[jj2] };
        let r_j = v_j - u_j;
        let so = unsafe { core::intrinsics::fmuladdf32(bk, r_j, s_scaled[jj2]) };
        s_out_row[jj2] = so;
        jj2 += 1;
    }

    let mut jj3 = 0usize;
    while jj3 < BV {
        unsafe {
            PROD[tid] = q_t * s_out_row[jj3];
        }
        thread::sync_threads();
        let mut stride = 128usize;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    PROD[tid] = PROD[tid] + PROD[tid + stride];
                }
            }
            thread::sync_threads();
            stride >>= 1;
        }
        if tid == 0 {
            unsafe {
                OVEC[jj3] = PROD[0];
            }
        }
        thread::sync_threads();
        jj3 += 1;
    }

    let s_row_base = (bh * 256 + tid) * 256;
    let so_ptr = s_out.as_mut_ptr();
    let mut wj = 0usize;
    while wj < BV {
        unsafe {
            *so_ptr.add(s_row_base + bv_col0 + wj) = s_out_row[wj];
        }
        wj += 1;
    }
    if tid < BV {
        let o_ptr = o.as_mut_ptr();
        unsafe {
            *o_ptr.add(bh * 256 + bv_col0 + tid) = OVEC[tid];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Shape {
    name: &'static str,
    batch: usize,
    n_heads: usize,
    d_k: usize,
    d_v: usize,
}

const SHAPE_CORRECTNESS: Shape = Shape {
    name: "correctness",
    batch: 2,
    n_heads: 4,
    d_k: 64,
    d_v: 64,
};

const SHAPE_QWEN3_NEXT_DECODE: Shape = Shape {
    name: "qwen3_next_decode",
    batch: 1,
    n_heads: 16,
    d_k: 256,
    d_v: 256,
};

#[derive(Debug)]
struct Npy {
    shape: Vec<usize>,
    dtype: String,
    data: Vec<u8>,
}

impl Npy {
    fn num_elems(&self) -> usize {
        self.shape.iter().product()
    }
    fn as_f32(&self) -> Vec<f32> {
        let n = self.num_elems();
        let mut out = Vec::with_capacity(n);
        if self.dtype == "<f4" {
            for i in 0..n {
                let off = i * 4;
                let b = [
                    self.data[off],
                    self.data[off + 1],
                    self.data[off + 2],
                    self.data[off + 3],
                ];
                out.push(f32::from_le_bytes(b));
            }
        } else {
            panic!("as_f32 called on non-f4 npy: {}", self.dtype);
        }
        out
    }
}

fn load_npy(path: &str) -> Npy {
    let mut f = File::open(path).unwrap_or_else(|e| panic!("open {}: {}", path, e));
    let mut magic = [0u8; 6];
    f.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, b"\x93NUMPY", "not a .npy: {}", path);
    let mut ver = [0u8; 2];
    f.read_exact(&mut ver).unwrap();
    let header_len = if ver[0] == 1 {
        let mut h = [0u8; 2];
        f.read_exact(&mut h).unwrap();
        u16::from_le_bytes(h) as usize
    } else {
        let mut h = [0u8; 4];
        f.read_exact(&mut h).unwrap();
        u32::from_le_bytes(h) as usize
    };
    let mut header = vec![0u8; header_len];
    f.read_exact(&mut header).unwrap();
    let header = String::from_utf8(header).unwrap();

    let dp = header
        .find("'descr':")
        .or_else(|| header.find("\"descr\":"))
        .unwrap();
    let sq1 = header[dp + 8..].find('\'').unwrap() + dp + 8;
    let sq2 = header[sq1 + 1..].find('\'').unwrap() + sq1 + 1;
    let dtype = header[sq1 + 1..sq2].to_string();

    let spos = header
        .find("'shape':")
        .or_else(|| header.find("\"shape\":"))
        .unwrap();
    let lp = header[spos..].find('(').unwrap() + spos;
    let rp = header[lp..].find(')').unwrap() + lp;
    let shape_str = &header[lp + 1..rp];
    let shape: Vec<usize> = shape_str
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();

    let num_elems: usize = shape.iter().product();
    let elem_size = if dtype == "<f2" {
        2
    } else if dtype == "<f4" {
        4
    } else {
        panic!("unsupported dtype {}", dtype)
    };
    let mut data = vec![0u8; num_elems * elem_size];
    f.read_exact(&mut data).unwrap();
    Npy { shape, dtype, data }
}

fn upload_f32(stream: &cuda_core::CudaStream, dst: &DeviceBuffer<f32>, src: &[f32]) {
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

/// Encode the S_in TMA descriptor for a (B*H, D_K, D_V) row-major f32 tensor.
///
/// View it as 2D (innermost-first per TMA convention):
///   globalDim[0] = D_V          (innermost,  stride = 1 elem)
///   globalDim[1] = B_H * D_K    (outer,      stride = D_V elements in BYTES)
///   boxDim[0]    = BV
///   boxDim[1]    = D_K
fn encode_s_in_descriptor(
    d_s_in: u64,
    b_h: u64,
    d_k: u64,
    d_v: u64,
) -> Result<CUtensorMap, Box<dyn std::error::Error>> {
    let mut tensor_map = MaybeUninit::<CUtensorMap>::uninit();
    let global_dim: [u64; 2] = [d_v, b_h * d_k];
    let global_strides: [u64; 1] = [d_v * std::mem::size_of::<f32>() as u64];
    let box_dim: [u32; 2] = [BV as u32, d_k as u32];
    let element_strides: [u32; 2] = [1, 1];

    let result = unsafe {
        cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            2,
            d_s_in as *mut std::ffi::c_void,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            element_strides.as_ptr(),
            CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE,
            CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };
    if result != sys::cudaError_enum_CUDA_SUCCESS {
        return Err(format!("cuTensorMapEncodeTiled failed: {:?}", result).into());
    }
    Ok(unsafe { tensor_map.assume_init() })
}

fn run_shape(
    shape: &Shape,
    _ctx: &std::sync::Arc<CudaContext>,
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &std::sync::Arc<cuda_core::CudaModule>,
    csv: &mut BufWriter<File>,
) {
    let inputs_dir =
        "/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs";
    let q_path = format!("{}/gdn_{}_q_f32.npy", inputs_dir, shape.name);
    let k_path = format!("{}/gdn_{}_k_f32.npy", inputs_dir, shape.name);
    let v_path = format!("{}/gdn_{}_v_f32.npy", inputs_dir, shape.name);
    let alpha_path = format!("{}/gdn_{}_alpha_f32.npy", inputs_dir, shape.name);
    let beta_path = format!("{}/gdn_{}_beta_f32.npy", inputs_dir, shape.name);
    let s_in_path = format!("{}/gdn_{}_S_in_f32.npy", inputs_dir, shape.name);
    let s_out_path = format!("{}/gdn_{}_S_out_expected_f32.npy", inputs_dir, shape.name);

    if !Path::new(&q_path).exists() {
        panic!(
            "missing input {}. Run pytorch_reference_gdn.py to regenerate.",
            q_path
        );
    }

    let q_npy = load_npy(&q_path);
    let k_npy = load_npy(&k_path);
    let v_npy = load_npy(&v_path);
    let alpha_npy = load_npy(&alpha_path);
    let beta_npy = load_npy(&beta_path);
    let s_in_npy = load_npy(&s_in_path);
    let s_out_e_npy = load_npy(&s_out_path);
    let o_exp_f16_path = format!("{}/gdn_{}_o_expected_f16.npy", inputs_dir, shape.name);
    let o_exp_npy = load_npy(&o_exp_f16_path);
    assert_eq!(o_exp_npy.dtype, "<f2");

    let q_host = q_npy.as_f32();
    let k_host = k_npy.as_f32();
    let v_host = v_npy.as_f32();
    let alpha_host = alpha_npy.as_f32();
    let beta_host = beta_npy.as_f32();
    let s_in_host = s_in_npy.as_f32();
    let s_out_e_host = s_out_e_npy.as_f32();

    let o_exp_host: Vec<f32> = {
        let n = o_exp_npy.num_elems();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * 2;
            let bits = u16::from_le_bytes([o_exp_npy.data[off], o_exp_npy.data[off + 1]]);
            let sign = (bits >> 15) & 0x1;
            let exp = (bits >> 10) & 0x1f;
            let mant = bits & 0x3ff;
            let f = if exp == 0 {
                if mant == 0 {
                    0.0_f32
                } else {
                    let m = mant as f32 / 1024.0;
                    m * 2f32.powi(-14)
                }
            } else if exp == 31 {
                if mant == 0 {
                    f32::INFINITY
                } else {
                    f32::NAN
                }
            } else {
                let m = 1.0_f32 + (mant as f32) / 1024.0;
                m * 2f32.powi(exp as i32 - 15)
            };
            out.push(if sign == 1 { -f } else { f });
        }
        out
    };

    let bh = shape.batch * shape.n_heads;
    let q_elems = bh * shape.d_k;
    let k_elems = bh * shape.d_k;
    let v_elems = bh * shape.d_v;
    let s_elems = bh * shape.d_k * shape.d_v;
    let o_elems = bh * shape.d_v;
    let ab_elems = bh;

    assert_eq!(q_host.len(), q_elems);
    assert_eq!(k_host.len(), k_elems);
    assert_eq!(v_host.len(), v_elems);
    assert_eq!(s_in_host.len(), s_elems);
    assert_eq!(s_out_e_host.len(), s_elems);
    assert_eq!(o_exp_host.len(), o_elems);
    assert_eq!(alpha_host.len(), ab_elems);
    assert_eq!(beta_host.len(), ab_elems);

    let q_dev = DeviceBuffer::from_host(stream, &q_host).unwrap();
    let k_dev = DeviceBuffer::from_host(stream, &k_host).unwrap();
    let v_dev = DeviceBuffer::from_host(stream, &v_host).unwrap();
    let alpha_dev = DeviceBuffer::from_host(stream, &alpha_host).unwrap();
    let beta_dev = DeviceBuffer::from_host(stream, &beta_host).unwrap();
    let s_in_dev = DeviceBuffer::from_host(stream, &s_in_host).unwrap();
    let mut s_out_dev = DeviceBuffer::<f32>::zeroed(stream, s_elems).unwrap();
    let mut o_dev = DeviceBuffer::<f32>::zeroed(stream, o_elems).unwrap();

    upload_f32(stream, &q_dev, &q_host);
    upload_f32(stream, &k_dev, &k_host);
    upload_f32(stream, &v_dev, &v_host);
    upload_f32(stream, &alpha_dev, &alpha_host);
    upload_f32(stream, &beta_dev, &beta_host);
    upload_f32(stream, &s_in_dev, &s_in_host);

    // Build TMA descriptor host-side and upload to device buffer (so kernel can
    // dereference it via *const TmaDescriptor).
    let tensor_map = encode_s_in_descriptor(
        s_in_dev.cu_deviceptr(),
        bh as u64,
        shape.d_k as u64,
        shape.d_v as u64,
    )
    .expect("encode S_in TMA descriptor");
    // CUtensorMap is repr(C); upload its 128 raw bytes by reading the field
    // that bindgen generates for the inner array.
    let tmap_bytes: [u8; 128] = unsafe {
        std::mem::transmute_copy::<CUtensorMap, [u8; 128]>(&tensor_map)
    };
    let dev_tmap = DeviceBuffer::from_host(stream, &tmap_bytes).unwrap();

    println!(
        "[oxide-attn-gdn-tma] shape={} (B={} H={} d_k={} d_v={}) state_MB={:.2}",
        shape.name,
        shape.batch,
        shape.n_heads,
        shape.d_k,
        shape.d_v,
        (s_elems * 4) as f64 / 1e6
    );

    let cfg = LaunchConfig {
        grid_dim: (bh as u32, (shape.d_v / BV) as u32, 1),
        block_dim: (shape.d_k as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let d_k_u = shape.d_k as u32;
    let d_v_u = shape.d_v as u32;

    let mut s_out_dev_mut = &mut s_out_dev;
    let mut o_dev_mut = &mut o_dev;
    let tmap_ptr = dev_tmap.cu_deviceptr() as *const TmaDescriptor;

    {
        let s = stream.clone();
        let m = module.clone();
        if shape.d_k == 64 {
            cuda_launch! {
                kernel: gdn_decode_dk64_tma,
                stream: s,
                module: m,
                config: cfg,
                args: [
                    slice(&q_dev),
                    slice(&k_dev),
                    slice(&v_dev),
                    slice(&alpha_dev),
                    slice(&beta_dev),
                    tmap_ptr,
                    slice_mut(s_out_dev_mut),
                    slice_mut(o_dev_mut),
                    d_k_u,
                    d_v_u
                ]
            }
            .unwrap();
        } else {
            cuda_launch! {
                kernel: gdn_decode_dk256_tma,
                stream: s,
                module: m,
                config: cfg,
                args: [
                    slice(&q_dev),
                    slice(&k_dev),
                    slice(&v_dev),
                    slice(&alpha_dev),
                    slice(&beta_dev),
                    tmap_ptr,
                    slice_mut(s_out_dev_mut),
                    slice_mut(o_dev_mut),
                    d_k_u,
                    d_v_u
                ]
            }
            .unwrap();
        }
    }
    stream.synchronize().unwrap();

    let o_got = o_dev.to_host_vec(stream).unwrap();
    let s_out_got = s_out_dev.to_host_vec(stream).unwrap();

    let mut max_abs_o: f32 = 0.0;
    for i in 0..o_elems {
        let d = (o_got[i] - o_exp_host[i]).abs();
        if d > max_abs_o {
            max_abs_o = d;
        }
    }
    let mut max_abs_s: f32 = 0.0;
    for i in 0..s_elems {
        let d = (s_out_got[i] - s_out_e_host[i]).abs();
        if d > max_abs_s {
            max_abs_s = d;
        }
    }

    let tol_o: f32 = 1e-3;
    let tol_s: f32 = 1e-3;
    let ok_o = max_abs_o <= tol_o;
    let ok_s = max_abs_s <= tol_s;

    println!(
        "[oxide-attn-gdn-tma] {} correctness: max_abs(o)={:.3e} max_abs(S_out)={:.3e} (tol={}) -> {}",
        shape.name,
        max_abs_o,
        max_abs_s,
        tol_o,
        if ok_o && ok_s { "OK" } else { "FAIL" }
    );
    writeln!(
        csv,
        "oxide-attn-gdn-tma,{},correctness,max_abs_o,{:.6e}",
        shape.name, max_abs_o
    )
    .unwrap();
    writeln!(
        csv,
        "oxide-attn-gdn-tma,{},correctness,max_abs_S_out,{:.6e}",
        shape.name, max_abs_s
    )
    .unwrap();
}

// ─────────────────────────────────────────────────────────────────────────────
// Bench harness
// ─────────────────────────────────────────────────────────────────────────────

fn run_bench(
    shape: &Shape,
    ctx: &std::sync::Arc<CudaContext>,
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &std::sync::Arc<cuda_core::CudaModule>,
    csv: &mut BufWriter<File>,
    iters: usize,
    warmup: usize,
) {
    let inputs_dir =
        "/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs";
    let q_path = format!("{}/gdn_{}_q_f32.npy", inputs_dir, shape.name);
    let k_path = format!("{}/gdn_{}_k_f32.npy", inputs_dir, shape.name);
    let v_path = format!("{}/gdn_{}_v_f32.npy", inputs_dir, shape.name);
    let alpha_path = format!("{}/gdn_{}_alpha_f32.npy", inputs_dir, shape.name);
    let beta_path = format!("{}/gdn_{}_beta_f32.npy", inputs_dir, shape.name);
    let s_in_path = format!("{}/gdn_{}_S_in_f32.npy", inputs_dir, shape.name);

    if !Path::new(&q_path).exists() {
        panic!("missing input {}", q_path);
    }

    let q_host = load_npy(&q_path).as_f32();
    let k_host = load_npy(&k_path).as_f32();
    let v_host = load_npy(&v_path).as_f32();
    let alpha_host = load_npy(&alpha_path).as_f32();
    let beta_host = load_npy(&beta_path).as_f32();
    let s_in_host = load_npy(&s_in_path).as_f32();

    let bh = shape.batch * shape.n_heads;
    let s_elems = bh * shape.d_k * shape.d_v;
    let o_elems = bh * shape.d_v;

    let q_dev = DeviceBuffer::from_host(stream, &q_host).unwrap();
    let k_dev = DeviceBuffer::from_host(stream, &k_host).unwrap();
    let v_dev = DeviceBuffer::from_host(stream, &v_host).unwrap();
    let alpha_dev = DeviceBuffer::from_host(stream, &alpha_host).unwrap();
    let beta_dev = DeviceBuffer::from_host(stream, &beta_host).unwrap();
    let s_in_dev = DeviceBuffer::from_host(stream, &s_in_host).unwrap();
    let mut s_out_dev = DeviceBuffer::<f32>::zeroed(stream, s_elems).unwrap();
    let mut o_dev = DeviceBuffer::<f32>::zeroed(stream, o_elems).unwrap();

    upload_f32(stream, &q_dev, &q_host);
    upload_f32(stream, &k_dev, &k_host);
    upload_f32(stream, &v_dev, &v_host);
    upload_f32(stream, &alpha_dev, &alpha_host);
    upload_f32(stream, &beta_dev, &beta_host);
    upload_f32(stream, &s_in_dev, &s_in_host);

    let tensor_map = encode_s_in_descriptor(
        s_in_dev.cu_deviceptr(),
        bh as u64,
        shape.d_k as u64,
        shape.d_v as u64,
    )
    .expect("encode S_in TMA descriptor (bench)");
    let tmap_bytes: [u8; 128] = unsafe {
        std::mem::transmute_copy::<CUtensorMap, [u8; 128]>(&tensor_map)
    };
    let dev_tmap = DeviceBuffer::from_host(stream, &tmap_bytes).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (bh as u32, (shape.d_v / BV) as u32, 1),
        block_dim: (shape.d_k as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let d_k_u = shape.d_k as u32;
    let d_v_u = shape.d_v as u32;

    let state_bytes = 2.0_f64 * (shape.d_k as f64) * (shape.d_v as f64) * 4.0;
    let io_bytes = (2.0 * shape.d_k as f64 + 2.0 * shape.d_v as f64 + 2.0) * 2.0;
    let bytes_per_iter = (bh as f64) * (state_bytes + io_bytes);
    println!(
        "[oxide-attn-gdn-tma] [bench] shape={} bytes/iter={:.2}KB warmup={} iters={}",
        shape.name,
        bytes_per_iter / 1024.0,
        warmup,
        iters
    );

    let do_launch = |s_out_ref: &mut DeviceBuffer<f32>, o_ref: &mut DeviceBuffer<f32>| {
        let mut s_out_mut = s_out_ref;
        let mut o_mut = o_ref;
        let s = stream.clone();
        let m = module.clone();
        let tmap_ptr = dev_tmap.cu_deviceptr() as *const TmaDescriptor;
        if shape.d_k == 64 {
            cuda_launch! {
                kernel: gdn_decode_dk64_tma,
                stream: s,
                module: m,
                config: cfg,
                args: [
                    slice(&q_dev),
                    slice(&k_dev),
                    slice(&v_dev),
                    slice(&alpha_dev),
                    slice(&beta_dev),
                    tmap_ptr,
                    slice_mut(s_out_mut),
                    slice_mut(o_mut),
                    d_k_u,
                    d_v_u
                ]
            }
            .unwrap();
        } else {
            cuda_launch! {
                kernel: gdn_decode_dk256_tma,
                stream: s,
                module: m,
                config: cfg,
                args: [
                    slice(&q_dev),
                    slice(&k_dev),
                    slice(&v_dev),
                    slice(&alpha_dev),
                    slice(&beta_dev),
                    tmap_ptr,
                    slice_mut(s_out_mut),
                    slice_mut(o_mut),
                    d_k_u,
                    d_v_u
                ]
            }
            .unwrap();
        }
    };

    for _ in 0..warmup {
        do_launch(&mut s_out_dev, &mut o_dev);
    }
    stream.synchronize().unwrap();

    let mut times_ms: Vec<f64> = Vec::with_capacity(iters);
    for i in 0..iters {
        let ev_a = ctx
            .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
            .unwrap();
        let ev_b = ctx
            .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
            .unwrap();

        let cpu_t0 = Instant::now();
        ev_a.record(stream).unwrap();
        do_launch(&mut s_out_dev, &mut o_dev);
        ev_b.record(stream).unwrap();
        stream.synchronize().unwrap();
        let cpu_ms = cpu_t0.elapsed().as_secs_f64() * 1000.0;
        let gpu_ms = ev_a.elapsed_ms(&ev_b).unwrap() as f64;
        let gbps = bytes_per_iter / (gpu_ms * 1e-3) / 1e9;
        times_ms.push(gpu_ms);
        if i < 3 || i == iters - 1 {
            println!(
                "[oxide-attn-gdn-tma] [bench] iter={} gpu_ms={:.4} cpu_ms={:.4} gbps={:.1}",
                i, gpu_ms, cpu_ms, gbps
            );
        }
        writeln!(
            csv,
            "oxide-attn-gdn-tma,{},bench,{},{:.6},{:.3}",
            shape.name, i, gpu_ms, gbps
        )
        .unwrap();
    }

    if !times_ms.is_empty() {
        let mut sorted = times_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let best = sorted[0];
        let med = sorted[sorted.len() / 2];
        let mean = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
        let best_gbps = bytes_per_iter / (best * 1e-3) / 1e9;
        let med_gbps = bytes_per_iter / (med * 1e-3) / 1e9;
        let mean_gbps = bytes_per_iter / (mean * 1e-3) / 1e9;
        println!("[oxide-attn-gdn-tma] [bench] best={:.4}ms ({:.1}GB/s) med={:.4}ms ({:.1}GB/s) mean={:.4}ms ({:.1}GB/s)",
                 best, best_gbps, med, med_gbps, mean, mean_gbps);
        writeln!(csv, "oxide-attn-gdn-tma,{},bench_summary,best_ms,{:.6}", shape.name, best).unwrap();
        writeln!(csv, "oxide-attn-gdn-tma,{},bench_summary,best_gbps,{:.3}", shape.name, best_gbps).unwrap();
        writeln!(csv, "oxide-attn-gdn-tma,{},bench_summary,median_gbps,{:.3}", shape.name, med_gbps).unwrap();
        writeln!(csv, "oxide-attn-gdn-tma,{},bench_summary,mean_gbps,{:.3}", shape.name, mean_gbps).unwrap();
    }
}

fn main() {
    let bench_mode = std::env::args().any(|a| a == "--bench")
        || std::env::var("BENCH")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = load_kernel_module(&ctx, "oxide_attn_gdn_tma").expect("load module");

    let csv_path = "/home/codeseys/cuda-exploration/oxide-attn-gdn-tma/results.csv";
    let csv_file = File::create(csv_path).expect("create csv");
    let mut csv = BufWriter::new(csv_file);
    writeln!(&mut csv, "impl,shape,kind,metric,value").unwrap();

    println!("[oxide-attn-gdn-tma] Wave C2.4 — cuda-oxide GDN decode with TMA");
    println!("[oxide-attn-gdn-tma] GPU: RTX 5090 sm_120, bench_mode={}", bench_mode);

    run_shape(&SHAPE_CORRECTNESS, &ctx, &stream, &module, &mut csv);
    run_shape(&SHAPE_QWEN3_NEXT_DECODE, &ctx, &stream, &module, &mut csv);

    if bench_mode {
        run_bench(&SHAPE_QWEN3_NEXT_DECODE, &ctx, &stream, &module, &mut csv, 50, 5);
    }

    csv.flush().unwrap();
    println!("[oxide-attn-gdn-tma] results.csv written");
}
