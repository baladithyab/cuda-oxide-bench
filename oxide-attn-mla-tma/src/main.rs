// Wave C3.3 — oxide-attn-mla-tma:
// Rust-CUDA (cuda-oxide) MLA attention with TMA-loaded Q/K/V tiles.
//
// Apply the C2.4 oxide-attn-gdn-tma recipe (host-side cuTensorMapEncodeTiled,
// device-side cp_async_bulk_tensor_2d_g2s + Barrier) to the 3-kernel MLA
// decomposition. Algorithm and FFMA microtile structure are identical to
// oxide-attn-mla — only the Q/K (in qkt_kernel) and V (in pv_kernel) global
// tile loads are replaced with TMA. P (probs) stays as ordinary LDG.E in
// pv_kernel; softmax_kernel is byte-for-byte unchanged.
//
// Three TMA descriptors, each encoded host-side ONCE before the kernel-launch
// loop:
//   Q_tmap  : view Q as 2D (B*n_h*S rows, qk cols), box=(16, 64) innermost-first.
//   K_tmap  : view K as 2D (B*n_h*S rows, qk cols), box=(16, 64) innermost-first.
//   V_tmap  : view V as 2D (B*n_h*S rows, d_v cols), box=(64, 16) innermost-first.
//
// Tile shape rationale (mirroring the FFMA baseline):
//   qkt: per block (64 rows × 64 cols of S), K-tile = 16 along qk.
//        Q tile in smem: 64 rows × 16 cols (innermost qk axis). Natural
//        row-major load into TILE_Q[r*16 + kk] — matches the FFMA baseline
//        layout exactly. NO smem index changes for Q.
//        K tile in smem: ALSO 64 rows × 16 cols (K[col0+r, k_off+kk]).
//        TMA cannot transpose during load, so we load K's rows naturally
//        into TILE_K[r*16 + kk] and adjust the FFMA inner loop's K read
//        from `TILE_K[kk*64 + cc]` to `TILE_K[cc*16 + kk]`. Same memory
//        contents, swapped indexing — no layout transform needed.
//   pv : per block (64 rows × 64 cols of out), K-tile = 16 along seq.
//        V tile in smem: 16 rows × 64 cols (innermost d_v axis). Natural
//        row-major load into TILE_V[kk*64 + cc] — matches the FFMA baseline
//        layout exactly. NO smem index changes for V.
//
// Acceptance:
//   - Compiles via `cargo oxide build --arch sm_120`.
//   - Correctness on the SHAPE_CORRECTNESS shape (B=1, n_h=4, S=128,
//     qk=96, d_v=64): max_abs_err <= 1e-2 vs the f32 PyTorch reference.
//   - SASS shows UTMALDG > 0 across the qkt and pv kernels
//     (qkt: 2 UTMALDG for Q+K; pv: 1 UTMALDG for V).

#![feature(core_intrinsics)]
#![allow(internal_features)]

use cuda_core::{
    CudaContext, CudaEvent, DeviceBuffer, LaunchConfig, sys,
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
use cuda_device::{DisjointSlice, SharedArray, kernel, thread, warp};
use cuda_host::{cuda_launch, load_kernel_module};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::mem::MaybeUninit;
use std::path::Path;
use std::time::Instant;

// ---------- kernels ----------

/// Q[b, h, row, k] x K[b, h, col, k]^T = S[b, h, row, col] * scale
///
/// Same launch grid and FFMA microtile as oxide-attn-mla. The *only* changes
/// vs the FFMA baseline are:
///   1. Replace the cooperative LDG of Q's (64,16) tile with a single
///      cp_async_bulk_tensor_2d_g2s issued by tid=0.
///   2. Same for K's tile, BUT we load K's rows naturally (64 rows × 16 cols)
///      instead of building a transposed (16,64) layout in smem. The FFMA
///      inner loop's K read is rewritten to index the natural layout.
///   3. Add an mbarrier; thread 0 arrive_expect_tx, others arrive, all wait.
///
/// Launch: grid=(S/64, S/64, B*n_h), block=(16,16,1). 256 threads per block.
#[kernel]
pub fn mla_qkt_kernel(
    mut scores: DisjointSlice<f32>,
    tensor_map_q: *const TmaDescriptor,
    tensor_map_k: *const TmaDescriptor,
    seq: u32,
    qk: u32,
    n_h: u32,
    scale: f32,
) {
    // TMA destinations: must be 128-byte aligned.
    static mut TILE_Q: SharedArray<f32, 1024, 128> = SharedArray::UNINIT; // 64 rows × 16 cols
    static mut TILE_K: SharedArray<f32, 1024, 128> = SharedArray::UNINIT; // 64 rows × 16 cols
    static mut BAR: Barrier = Barrier::UNINIT;

    let tx = thread::threadIdx_x() as usize;
    let ty = thread::threadIdx_y() as usize;
    let bx = thread::blockIdx_x() as usize;
    let by = thread::blockIdx_y() as usize;
    let bz = thread::blockIdx_z() as usize;

    let seq_us = seq as usize;
    let qk_us = qk as usize;
    let n_h_us = n_h as usize;
    let block_size = thread::blockDim_x() * thread::blockDim_y() * thread::blockDim_z();

    // (batch, head) from bz; n_kv == n_h, no broadcasting
    let b = bz / n_h_us;
    let h = bz % n_h_us;

    let s_base = ((b * n_h_us) + h) * seq_us * seq_us;

    let row0 = by * 64 + ty * 4;
    let col0 = bx * 64 + tx * 4;
    let tid = ty * 16 + tx;
    let num_tiles = qk_us / 16;

    // Init barrier with arrive_count = block_size (256). All threads will
    // arrive once per K-tile-iteration.
    if tid == 0 {
        unsafe {
            mbarrier_init(&raw mut BAR, block_size);
            fence_proxy_async_shared_cta();
        }
    }
    thread::sync_threads();

    // 4x4 scalar accumulators.
    let mut c00: f32 = 0.0; let mut c01: f32 = 0.0; let mut c02: f32 = 0.0; let mut c03: f32 = 0.0;
    let mut c10: f32 = 0.0; let mut c11: f32 = 0.0; let mut c12: f32 = 0.0; let mut c13: f32 = 0.0;
    let mut c20: f32 = 0.0; let mut c21: f32 = 0.0; let mut c22: f32 = 0.0; let mut c23: f32 = 0.0;
    let mut c30: f32 = 0.0; let mut c31: f32 = 0.0; let mut c32: f32 = 0.0; let mut c33: f32 = 0.0;

    let tile_bytes_each: u32 = (64 * 16 * 4) as u32; // 4 KiB per tile (Q, K)
    let tile_bytes_total: u32 = tile_bytes_each * 2;

    let mut t: usize = 0;
    while t < num_tiles {
        let k_off = t * 16;

        // Issue both TMAs from tid 0; they share one barrier (one wait per iter).
        // Q descriptor view: (B*n_h*S rows, qk cols) row-major, box=(16, 64).
        //   coord_x = k_off (col),  coord_y = bh*S + by*64  (row)
        // K descriptor view: same shape, but coord_y = bh*S + bx*64.
        //   K tile in smem holds rows [bx*64 .. bx*64+63], cols [k_off..k_off+15].
        let bh_row = ((b * n_h_us) + h) * seq_us;
        if tid == 0 {
            unsafe {
                cp_async_bulk_tensor_2d_g2s(
                    &raw mut TILE_Q as *mut u8,
                    tensor_map_q,
                    k_off as i32,
                    (bh_row + by * 64) as i32,
                    &raw mut BAR,
                );
                cp_async_bulk_tensor_2d_g2s(
                    &raw mut TILE_K as *mut u8,
                    tensor_map_k,
                    k_off as i32,
                    (bh_row + bx * 64) as i32,
                    &raw mut BAR,
                );
            }
        }
        let token = unsafe {
            if tid == 0 {
                mbarrier_arrive_expect_tx(&raw const BAR, 1, tile_bytes_total)
            } else {
                mbarrier_arrive(&raw const BAR)
            }
        };
        unsafe {
            while !mbarrier_try_wait(&raw const BAR, token) {}
        }
        thread::sync_threads();

        let ty4 = ty * 4;
        let tx4 = tx * 4;
        let mut kk: usize = 0;
        while kk < 16 {
            // Q layout in smem: TILE_Q[r * 16 + kk] for r in 0..64.
            // K layout in smem (NATURAL): TILE_K[r * 16 + kk] where r is the
            // seq-column index inside the output tile (0..64). That is, the
            // FFMA "b" values come from different ROWS of TILE_K, not different
            // cols of a transposed tile. Correctness preserved — same data.
            let a0: f32; let a1: f32; let a2: f32; let a3: f32;
            let b0: f32; let b1: f32; let b2: f32; let b3: f32;
            unsafe {
                a0 = TILE_Q[(ty4 + 0) * 16 + kk];
                a1 = TILE_Q[(ty4 + 1) * 16 + kk];
                a2 = TILE_Q[(ty4 + 2) * 16 + kk];
                a3 = TILE_Q[(ty4 + 3) * 16 + kk];
                b0 = TILE_K[(tx4 + 0) * 16 + kk];
                b1 = TILE_K[(tx4 + 1) * 16 + kk];
                b2 = TILE_K[(tx4 + 2) * 16 + kk];
                b3 = TILE_K[(tx4 + 3) * 16 + kk];
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
            kk += 1;
        }
        thread::sync_threads();
        t += 1;
    }

    let s_ptr = scores.as_mut_ptr();
    unsafe {
        // Scale fused into the store.
        *s_ptr.add(s_base + (row0 + 0) * seq_us + col0 + 0) = c00 * scale;
        *s_ptr.add(s_base + (row0 + 0) * seq_us + col0 + 1) = c01 * scale;
        *s_ptr.add(s_base + (row0 + 0) * seq_us + col0 + 2) = c02 * scale;
        *s_ptr.add(s_base + (row0 + 0) * seq_us + col0 + 3) = c03 * scale;
        *s_ptr.add(s_base + (row0 + 1) * seq_us + col0 + 0) = c10 * scale;
        *s_ptr.add(s_base + (row0 + 1) * seq_us + col0 + 1) = c11 * scale;
        *s_ptr.add(s_base + (row0 + 1) * seq_us + col0 + 2) = c12 * scale;
        *s_ptr.add(s_base + (row0 + 1) * seq_us + col0 + 3) = c13 * scale;
        *s_ptr.add(s_base + (row0 + 2) * seq_us + col0 + 0) = c20 * scale;
        *s_ptr.add(s_base + (row0 + 2) * seq_us + col0 + 1) = c21 * scale;
        *s_ptr.add(s_base + (row0 + 2) * seq_us + col0 + 2) = c22 * scale;
        *s_ptr.add(s_base + (row0 + 2) * seq_us + col0 + 3) = c23 * scale;
        *s_ptr.add(s_base + (row0 + 3) * seq_us + col0 + 0) = c30 * scale;
        *s_ptr.add(s_base + (row0 + 3) * seq_us + col0 + 1) = c31 * scale;
        *s_ptr.add(s_base + (row0 + 3) * seq_us + col0 + 2) = c32 * scale;
        *s_ptr.add(s_base + (row0 + 3) * seq_us + col0 + 3) = c33 * scale;
    }
}

/// Row-wise softmax — BYTE-FOR-BYTE COPY of oxide-attn-mla's softmax.
/// No TMA here (per-row reduction; no large gmem tile worth hoisting).
#[kernel]
pub fn softmax_kernel(
    scores: &[f32],
    mut probs: DisjointSlice<f32>,
    seq: u32,
) {
    static mut WMAX: SharedArray<f32, 4> = SharedArray::UNINIT;
    static mut WSUM: SharedArray<f32, 4> = SharedArray::UNINIT;
    static mut ROW_MAX: SharedArray<f32, 1> = SharedArray::UNINIT;
    static mut ROW_SUM: SharedArray<f32, 1> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    let bid = thread::blockIdx_x() as usize;
    let seq_us = seq as usize;
    let lane = warp::lane_id() as usize;
    let warp_id = tid >> 5;
    let bdim = thread::blockDim_x() as usize;

    let row_base = bid * seq_us;
    let s_ptr = scores.as_ptr();
    let p_ptr = probs.as_mut_ptr();

    let mut m: f32 = f32::NEG_INFINITY;
    let mut i = tid;
    while i < seq_us {
        let v = unsafe { *s_ptr.add(row_base + i) };
        if v > m { m = v; }
        i += bdim;
    }
    let mut d = 16;
    while d > 0 {
        let other = warp::shuffle_xor_f32(m, d);
        if other > m { m = other; }
        d >>= 1;
    }
    if lane == 0 { unsafe { WMAX[warp_id] = m; } }
    thread::sync_threads();
    if warp_id == 0 {
        let mut v: f32 = if lane < 4 { unsafe { WMAX[lane] } } else { f32::NEG_INFINITY };
        let mut dd = 2;
        while dd > 0 {
            let other = warp::shuffle_xor_f32(v, dd);
            if other > v { v = other; }
            dd >>= 1;
        }
        if lane == 0 { unsafe { ROW_MAX[0] = v; } }
    }
    thread::sync_threads();
    let row_max = unsafe { ROW_MAX[0] };

    let mut s: f32 = 0.0;
    let mut j = tid;
    while j < seq_us {
        let v = unsafe { *s_ptr.add(row_base + j) };
        let e = unsafe { core::intrinsics::expf32(v - row_max) };
        unsafe { *p_ptr.add(row_base + j) = e; }
        s += e;
        j += bdim;
    }
    s += warp::shuffle_xor_f32(s, 16);
    s += warp::shuffle_xor_f32(s, 8);
    s += warp::shuffle_xor_f32(s, 4);
    s += warp::shuffle_xor_f32(s, 2);
    s += warp::shuffle_xor_f32(s, 1);
    if lane == 0 { unsafe { WSUM[warp_id] = s; } }
    thread::sync_threads();
    if warp_id == 0 {
        let mut v: f32 = if lane < 4 { unsafe { WSUM[lane] } } else { 0.0 };
        v += warp::shuffle_xor_f32(v, 2);
        v += warp::shuffle_xor_f32(v, 1);
        if lane == 0 { unsafe { ROW_SUM[0] = v; } }
    }
    thread::sync_threads();
    let row_sum = unsafe { ROW_SUM[0] };
    let inv = 1.0_f32 / row_sum;

    let mut l = tid;
    while l < seq_us {
        let e = unsafe { *p_ptr.add(row_base + l) };
        unsafe { *p_ptr.add(row_base + l) = e * inv; }
        l += bdim;
    }
}

/// probs[b,h,row,k] x V[b,h,k,col] = out[b,h,row,col]
/// Same FFMA microtile as the FFMA baseline. The *only* change is V's tile
/// load: TMA-load 16 rows × 64 cols (innermost d_v) into the smem layout
/// the FFMA baseline already uses (TILE_V[kk*64+cc]) — natural row-major
/// match, no index changes needed for V.
///
/// P (probs) tile stays as cooperative LDG.E (64 rows × 16 cols of probs).
/// Per Wave C3.3 task spec: TMA on V only in pv_kernel (1 UTMALDG), Q+K in
/// qkt_kernel (2 UTMALDG). Total 3 UTMALDG across kernels.
#[kernel]
pub fn mla_pv_kernel(
    probs: &[f32],
    mut out: DisjointSlice<f32>,
    tensor_map_v: *const TmaDescriptor,
    seq: u32,
    d_v: u32,
    n_h: u32,
) {
    static mut TILE_P: SharedArray<f32, 1024> = SharedArray::UNINIT; // 64 × 16
    // V tile: 16 rows × 64 cols, row-major (innermost d_v). 128B-aligned for TMA.
    static mut TILE_V: SharedArray<f32, 1024, 128> = SharedArray::UNINIT;
    static mut BAR: Barrier = Barrier::UNINIT;

    let tx = thread::threadIdx_x() as usize;
    let ty = thread::threadIdx_y() as usize;
    let bx = thread::blockIdx_x() as usize;
    let by = thread::blockIdx_y() as usize;
    let bz = thread::blockIdx_z() as usize;

    let seq_us = seq as usize;
    let d_v_us = d_v as usize;
    let n_h_us = n_h as usize;
    let block_size = thread::blockDim_x() * thread::blockDim_y() * thread::blockDim_z();

    let b = bz / n_h_us;
    let h = bz % n_h_us;

    let p_base = ((b * n_h_us) + h) * seq_us * seq_us;
    let o_base = ((b * n_h_us) + h) * seq_us * d_v_us;

    let row0 = by * 64 + ty * 4;
    let col0 = bx * 64 + tx * 4;
    let tid = ty * 16 + tx;
    let num_tiles = seq_us / 16;

    if tid == 0 {
        unsafe {
            mbarrier_init(&raw mut BAR, block_size);
            fence_proxy_async_shared_cta();
        }
    }
    thread::sync_threads();

    let mut c00: f32 = 0.0; let mut c01: f32 = 0.0; let mut c02: f32 = 0.0; let mut c03: f32 = 0.0;
    let mut c10: f32 = 0.0; let mut c11: f32 = 0.0; let mut c12: f32 = 0.0; let mut c13: f32 = 0.0;
    let mut c20: f32 = 0.0; let mut c21: f32 = 0.0; let mut c22: f32 = 0.0; let mut c23: f32 = 0.0;
    let mut c30: f32 = 0.0; let mut c31: f32 = 0.0; let mut c32: f32 = 0.0; let mut c33: f32 = 0.0;

    let p_ptr = probs.as_ptr();

    let v_tile_bytes: u32 = (16 * 64 * 4) as u32; // 4 KiB

    let mut t: usize = 0;
    while t < num_tiles {
        let k_off = t * 16;

        // Load TILE_P[r, kk] = probs[row0_block+r, k_off+kk]  -- cooperative LDG.
        let mut li: usize = 0;
        while li < 4 {
            let idx = tid + li * 256;
            let r = idx / 16;
            let kk = idx & 15;
            let gr = by * 64 + r;
            let gk = k_off + kk;
            unsafe {
                let vv = *p_ptr.add(p_base + gr * seq_us + gk);
                TILE_P[idx] = vv;
            }
            li += 1;
        }

        // Issue V TMA: V viewed as (B*n_h*S rows, d_v cols), box=(64, 16).
        //   coord_x = bx*64 (innermost = d_v col),
        //   coord_y = bh*S + k_off  (outer = which seq-row).
        let bh_row = ((b * n_h_us) + h) * seq_us;
        if tid == 0 {
            unsafe {
                cp_async_bulk_tensor_2d_g2s(
                    &raw mut TILE_V as *mut u8,
                    tensor_map_v,
                    (bx * 64) as i32,
                    (bh_row + k_off) as i32,
                    &raw mut BAR,
                );
            }
        }
        let token = unsafe {
            if tid == 0 {
                mbarrier_arrive_expect_tx(&raw const BAR, 1, v_tile_bytes)
            } else {
                mbarrier_arrive(&raw const BAR)
            }
        };
        unsafe {
            while !mbarrier_try_wait(&raw const BAR, token) {}
        }
        thread::sync_threads();

        let ty4 = ty * 4;
        let tx4 = tx * 4;
        let mut kk: usize = 0;
        while kk < 16 {
            let a0: f32; let a1: f32; let a2: f32; let a3: f32;
            let b0: f32; let b1: f32; let b2: f32; let b3: f32;
            unsafe {
                a0 = TILE_P[(ty4 + 0) * 16 + kk];
                a1 = TILE_P[(ty4 + 1) * 16 + kk];
                a2 = TILE_P[(ty4 + 2) * 16 + kk];
                a3 = TILE_P[(ty4 + 3) * 16 + kk];
                b0 = TILE_V[kk * 64 + tx4 + 0];
                b1 = TILE_V[kk * 64 + tx4 + 1];
                b2 = TILE_V[kk * 64 + tx4 + 2];
                b3 = TILE_V[kk * 64 + tx4 + 3];
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
            kk += 1;
        }
        thread::sync_threads();
        t += 1;
    }

    let o_ptr = out.as_mut_ptr();
    unsafe {
        *o_ptr.add(o_base + (row0 + 0) * d_v_us + col0 + 0) = c00;
        *o_ptr.add(o_base + (row0 + 0) * d_v_us + col0 + 1) = c01;
        *o_ptr.add(o_base + (row0 + 0) * d_v_us + col0 + 2) = c02;
        *o_ptr.add(o_base + (row0 + 0) * d_v_us + col0 + 3) = c03;
        *o_ptr.add(o_base + (row0 + 1) * d_v_us + col0 + 0) = c10;
        *o_ptr.add(o_base + (row0 + 1) * d_v_us + col0 + 1) = c11;
        *o_ptr.add(o_base + (row0 + 1) * d_v_us + col0 + 2) = c12;
        *o_ptr.add(o_base + (row0 + 1) * d_v_us + col0 + 3) = c13;
        *o_ptr.add(o_base + (row0 + 2) * d_v_us + col0 + 0) = c20;
        *o_ptr.add(o_base + (row0 + 2) * d_v_us + col0 + 1) = c21;
        *o_ptr.add(o_base + (row0 + 2) * d_v_us + col0 + 2) = c22;
        *o_ptr.add(o_base + (row0 + 2) * d_v_us + col0 + 3) = c23;
        *o_ptr.add(o_base + (row0 + 3) * d_v_us + col0 + 0) = c30;
        *o_ptr.add(o_base + (row0 + 3) * d_v_us + col0 + 1) = c31;
        *o_ptr.add(o_base + (row0 + 3) * d_v_us + col0 + 2) = c32;
        *o_ptr.add(o_base + (row0 + 3) * d_v_us + col0 + 3) = c33;
    }
}

// ---------- host code ----------

#[derive(Debug, Clone)]
struct Shape {
    name: &'static str,
    batch: usize,
    seq: usize,
    n_h: usize,
    qk: usize,
    d_v: usize,
}

// Wave C3.3 acceptance shape (per task spec):
// B=1 n_h=4 S=128 qk=96 d_v=64. Same as oxide-attn-mla SHAPE_CORRECTNESS.
const SHAPE_CORRECTNESS: Shape = Shape {
    name: "correctness_mla",
    batch: 1,
    seq: 128,
    n_h: 4,
    qk: 96,
    d_v: 64,
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
                let b = [self.data[off], self.data[off + 1], self.data[off + 2], self.data[off + 3]];
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

    let dp = header.find("'descr':").or_else(|| header.find("\"descr\":")).unwrap();
    let sq1 = header[dp + 8..].find('\'').unwrap() + dp + 8;
    let sq2 = header[sq1 + 1..].find('\'').unwrap() + sq1 + 1;
    let dtype = header[sq1 + 1..sq2].to_string();

    let spos = header.find("'shape':").or_else(|| header.find("\"shape\":")).unwrap();
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
    let elem_size = if dtype == "<f2" { 2 } else if dtype == "<f4" { 4 } else {
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

/// Encode a 2D TMA descriptor for a row-major f32 tensor.
///   globalDim    = { inner_size (innermost), outer_size }
///   globalStrides[0] = inner_size * 4 bytes
///   boxDim       = { box_inner, box_outer }
fn encode_tma_2d_f32(
    d_base: u64,
    outer_size: u64,
    inner_size: u64,
    box_inner: u32,
    box_outer: u32,
) -> Result<CUtensorMap, Box<dyn std::error::Error>> {
    let mut tensor_map = MaybeUninit::<CUtensorMap>::uninit();
    let global_dim: [u64; 2] = [inner_size, outer_size];
    let global_strides: [u64; 1] = [inner_size * std::mem::size_of::<f32>() as u64];
    let box_dim: [u32; 2] = [box_inner, box_outer];
    let element_strides: [u32; 2] = [1, 1];

    let result = unsafe {
        cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            2,
            d_base as *mut std::ffi::c_void,
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

fn upload_tmap(
    stream: &cuda_core::CudaStream,
    tmap: &CUtensorMap,
) -> DeviceBuffer<u8> {
    let bytes: [u8; 128] = unsafe {
        std::mem::transmute_copy::<CUtensorMap, [u8; 128]>(tmap)
    };
    DeviceBuffer::from_host(stream, &bytes).unwrap()
}

fn run_shape(
    shape: &Shape,
    ctx: &std::sync::Arc<CudaContext>,
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    module: &std::sync::Arc<cuda_core::CudaModule>,
    csv: &mut BufWriter<File>,
) {
    let inputs_dir = "/home/codeseys/cuda-exploration/analysis/wave15-attention-architecture/inputs";
    let q_path = format!("{}/mla_{}_q_f32.npy", inputs_dir, shape.name);
    let k_path = format!("{}/mla_{}_k_f32.npy", inputs_dir, shape.name);
    let v_path = format!("{}/mla_{}_v_f32.npy", inputs_dir, shape.name);
    let e_path = format!("{}/mla_{}_expected_f32.npy", inputs_dir, shape.name);

    if !Path::new(&q_path).exists() {
        panic!("missing input {}. Run pytorch_reference_mla.py to regenerate.", q_path);
    }

    let q_npy = load_npy(&q_path);
    let k_npy = load_npy(&k_path);
    let v_npy = load_npy(&v_path);
    let e_npy = load_npy(&e_path);

    let q_host = q_npy.as_f32();
    let k_host = k_npy.as_f32();
    let v_host = v_npy.as_f32();
    let e_host = e_npy.as_f32();

    let q_elems  = shape.batch * shape.n_h * shape.seq * shape.qk;
    let k_elems  = q_elems;
    let v_elems  = shape.batch * shape.n_h * shape.seq * shape.d_v;
    let scores_elems = shape.batch * shape.n_h * shape.seq * shape.seq;
    let out_elems = v_elems;

    assert_eq!(q_host.len(), q_elems);
    assert_eq!(k_host.len(), k_elems);
    assert_eq!(v_host.len(), v_elems);
    assert_eq!(e_host.len(), out_elems);

    let q_dev = DeviceBuffer::from_host(stream, &q_host).unwrap();
    let k_dev = DeviceBuffer::from_host(stream, &k_host).unwrap();
    let v_dev = DeviceBuffer::from_host(stream, &v_host).unwrap();
    let mut scores_dev = DeviceBuffer::<f32>::zeroed(stream, scores_elems).unwrap();
    let mut probs_dev = DeviceBuffer::<f32>::zeroed(stream, scores_elems).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(stream, out_elems).unwrap();

    upload_f32(stream, &q_dev, &q_host);
    upload_f32(stream, &k_dev, &k_host);
    upload_f32(stream, &v_dev, &v_host);

    // ---- Build the THREE TMA descriptors host-side, ONCE ----
    // Q view: 2D (B*n_h*S rows, qk cols), box = (16 inner cols, 64 outer rows).
    // Same for K. V view: 2D (B*n_h*S rows, d_v cols), box = (64 inner cols, 16 outer rows).
    let bh_s = (shape.batch * shape.n_h * shape.seq) as u64;
    let q_tmap = encode_tma_2d_f32(
        q_dev.cu_deviceptr(),
        bh_s,
        shape.qk as u64,
        16,
        64,
    ).expect("encode Q TMA descriptor");
    let k_tmap = encode_tma_2d_f32(
        k_dev.cu_deviceptr(),
        bh_s,
        shape.qk as u64,
        16,
        64,
    ).expect("encode K TMA descriptor");
    let v_tmap = encode_tma_2d_f32(
        v_dev.cu_deviceptr(),
        bh_s,
        shape.d_v as u64,
        64,
        16,
    ).expect("encode V TMA descriptor");

    let dev_q_tmap = upload_tmap(stream, &q_tmap);
    let dev_k_tmap = upload_tmap(stream, &k_tmap);
    let dev_v_tmap = upload_tmap(stream, &v_tmap);

    println!("[oxide-attn-mla-tma] shape={} (B={} S={} n_h={} qk={} d_v={}) scores_MB={:.1}",
        shape.name, shape.batch, shape.seq, shape.n_h, shape.qk, shape.d_v,
        (scores_elems * 4) as f64 / 1e6);

    let seq = shape.seq as u32;
    let qk = shape.qk as u32;
    let d_v = shape.d_v as u32;
    let n_h = shape.n_h as u32;
    let scale = 1.0_f32 / (shape.qk as f32).sqrt();
    let b_nh = shape.batch * shape.n_h;

    let cfg_qkt = LaunchConfig {
        grid_dim: ((shape.seq / 64) as u32, (shape.seq / 64) as u32, b_nh as u32),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };
    let cfg_softmax = LaunchConfig {
        grid_dim: ((shape.batch * shape.n_h * shape.seq) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let cfg_pv = LaunchConfig {
        grid_dim: ((shape.d_v / 64) as u32, (shape.seq / 64) as u32, b_nh as u32),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    let q_tmap_ptr = dev_q_tmap.cu_deviceptr() as *const TmaDescriptor;
    let k_tmap_ptr = dev_k_tmap.cu_deviceptr() as *const TmaDescriptor;
    let v_tmap_ptr = dev_v_tmap.cu_deviceptr() as *const TmaDescriptor;

    // warmup + correctness run.
    {
        let mut scores_mut = &mut scores_dev;
        let mut probs_mut = &mut probs_dev;
        let mut out_mut = &mut out_dev;
        let s = stream.clone(); let m = module.clone();
        cuda_launch! {
            kernel: mla_qkt_kernel, stream: s.clone(), module: m.clone(), config: cfg_qkt,
            args: [slice_mut(scores_mut), q_tmap_ptr, k_tmap_ptr, seq, qk, n_h, scale]
        }.unwrap();
        cuda_launch! {
            kernel: softmax_kernel, stream: s.clone(), module: m.clone(), config: cfg_softmax,
            args: [slice(&scores_dev), slice_mut(probs_mut), seq]
        }.unwrap();
        cuda_launch! {
            kernel: mla_pv_kernel, stream: s, module: m, config: cfg_pv,
            args: [slice(&probs_dev), slice_mut(out_mut), v_tmap_ptr, seq, d_v, n_h]
        }.unwrap();
        stream.synchronize().unwrap();
    }

    let got = out_dev.to_host_vec(stream).unwrap();
    let mut max_abs: f32 = 0.0;
    let mut max_rel: f32 = 0.0;
    for i in 0..out_elems {
        let d = (got[i] - e_host[i]).abs();
        if d > max_abs { max_abs = d; }
        let e_abs = e_host[i].abs().max(1e-6);
        let r = d / e_abs;
        if r > max_rel { max_rel = r; }
    }
    let loose_atol = 1e-2_f32;
    let ok = max_abs <= loose_atol;
    println!(
        "[oxide-attn-mla-tma] {} correctness: max_abs={:.3e} max_rel={:.3e} (atol={}) -> {}",
        shape.name, max_abs, max_rel, loose_atol, if ok {"PASS"} else {"FAIL"}
    );
    writeln!(csv, "oxide-attn-mla-tma,{},correctness,max_abs,{:.6e}", shape.name, max_abs).unwrap();
    writeln!(csv, "oxide-attn-mla-tma,{},correctness,max_rel,{:.6e}", shape.name, max_rel).unwrap();
    writeln!(csv, "oxide-attn-mla-tma,{},correctness,result,{}", shape.name, if ok {"PASS"} else {"FAIL"}).unwrap();

    // optional warm timing (single iter, not a real bench).
    let ev_a = ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT)).unwrap();
    let ev_b = ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT)).unwrap();
    let cpu_t0 = Instant::now();
    ev_a.record(stream).unwrap();
    {
        let mut scores_mut = &mut scores_dev;
        let mut probs_mut = &mut probs_dev;
        let mut out_mut = &mut out_dev;
        let s = stream.clone(); let m = module.clone();
        cuda_launch! {
            kernel: mla_qkt_kernel, stream: s.clone(), module: m.clone(), config: cfg_qkt,
            args: [slice_mut(scores_mut), q_tmap_ptr, k_tmap_ptr, seq, qk, n_h, scale]
        }.unwrap();
        cuda_launch! {
            kernel: softmax_kernel, stream: s.clone(), module: m.clone(), config: cfg_softmax,
            args: [slice(&scores_dev), slice_mut(probs_mut), seq]
        }.unwrap();
        cuda_launch! {
            kernel: mla_pv_kernel, stream: s, module: m, config: cfg_pv,
            args: [slice(&probs_dev), slice_mut(out_mut), v_tmap_ptr, seq, d_v, n_h]
        }.unwrap();
    }
    ev_b.record(stream).unwrap();
    stream.synchronize().unwrap();
    let cpu_ms = cpu_t0.elapsed().as_secs_f64() * 1000.0;
    let gpu_ms = ev_a.elapsed_ms(&ev_b).unwrap() as f64;
    println!("[oxide-attn-mla-tma] {} timing: gpu_ms={:.4} cpu_ms={:.4} (1 iter, not a bench)",
        shape.name, gpu_ms, cpu_ms);
    writeln!(csv, "oxide-attn-mla-tma,{},timing,gpu_ms_1iter,{:.6}", shape.name, gpu_ms).unwrap();
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = load_kernel_module(&ctx, "oxide_attn_mla_tma").expect("load module");

    let csv_path = "/home/codeseys/cuda-exploration/oxide-attn-mla-tma/results.csv";
    let csv_file = File::create(csv_path).expect("create csv");
    let mut csv = BufWriter::new(csv_file);
    writeln!(&mut csv, "impl,shape,kind,metric,value").unwrap();

    println!("[oxide-attn-mla-tma] Wave C3.3 — cuda-oxide MLA with TMA Q/K/V tiles");
    println!("[oxide-attn-mla-tma] GPU: RTX 5090 sm_120");

    run_shape(&SHAPE_CORRECTNESS, &ctx, &stream, &module, &mut csv);

    csv.flush().unwrap();
    println!("[oxide-attn-mla-tma] results.csv written");
}
