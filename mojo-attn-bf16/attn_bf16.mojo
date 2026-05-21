# Wave 22.5b -- mojo-attn-bf16: 3-kernel attention with bf16 matmul stages,
# scaled to DeepSeek-V3 decode shape and timed.
#
# Pipeline (mirrors cublas-attn-mla / cuda-attn-mla):
#   Stage 1 (qkt_kernel):     scores = Q @ K^T              [B, n_h, S, S]   f32
#   Stage 2 (softmax_kernel): probs  = softmax(scores * 1/sqrt(qk))          bf16
#   Stage 3 (pv_kernel):      out    = probs @ V            [B, n_h, S, d_v] f32
#
# All three stages stage to HBM between them (HBM round-trip is the cuda-attn-mla
# 24 TF ceiling per Wave 17 W1a). The matmul stages reuse the Wave 21
# mojo-matmul-bf16 pattern: TensorCore[bf16, bf16] for load_a/load_b fragments,
# raw mma() with f32 accumulator, hand-rolled m16n8 epilogue.
#
# W22.5  shape (correctness only): B=1, n_h=4,   S=128,  qk=64,  d_v=64.
# W22.5b shape (this file, bench): B=1, n_h=128, S=2048, qk=192, d_v=128.
#
# Tile shape: BM=BN=64, BK=32, WM=WN=32, MMA=16x8x16. 4 warps/block, 128 threads.
#
# Shape-divisibility check (the orchestrator flagged qk=192 vs BK=32 as risky):
#   192 / BK=32 = 6  ✓   (qkt K-iters)
#   2048 / BM=64 = 32 ✓  (S → BM tile rows, both qkt and pv)
#   2048 / BN=64 = 32 ✓  (S → BN tile cols on qkt)
#   2048 / BK=32 = 64 ✓  (pv K-iters along S)
#   128 / BN=64  = 2  ✓  (DV → BN tile cols on pv)
# BK=32 divides everything cleanly; no tail-handler needed. Keep Wave 21 BK=32
# tile shape unchanged.
#
# Pitfall vs Wave 21 matmul: Q@K^T needs K loaded with row/col swapped so that
# MMA's K-dim (= matmul-K-dim = qk_head_dim) maps to the inner-row index of K.
# We do this with a strided thread-cooperative load (NOT copy_dram_to_sram_async,
# which is row-major-only). That costs us the cp.async path on the qkt kernel —
# expected: HMMA SASS still emits, just without UTMALDG.

from std.math import ceildiv, sqrt, exp
from std.sys import has_accelerator
from std.gpu import (
    WARP_SIZE,
    barrier,
    block_idx,
    thread_idx,
    warp_id,
    lane_id,
    block_dim,
)
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace, async_copy_wait_all
from std.gpu.compute.mma import mma
from layout.layout_tensor import Layout, LayoutTensor, copy_dram_to_sram_async
from layout.tensor_core import TensorCore
from std.utils.index import Index


# ============================================================================
# Stage 1: Q @ K^T kernel.  scores = Q @ K^T.
# Per-(b, h) head: Q[S, qk], K[S, qk] -> scores[S, S].
# scores[i, j] = sum_k Q[i, k] * K[j, k]
#
# Map onto matmul with M=S, N=S, K=qk:
#   A = Q   (M, K) row-major, normal load.
#   B-tile in smem must be K's qk-dim along smem-row, S-dim along smem-col, i.e.
#       B_smem[k_inner, n_inner] = K[block_x*BN + n_inner, k_iter*BK + k_inner].
#   This is a thread-cooperative strided gather (BK x BN tile, BLOCK_THREADS=128
#   threads, BK*BN=2048 elements -> 16 elems/thread).
# ============================================================================
def qkt_kernel[
    layout_q: Layout,           # [BH * S, qk]
    layout_k: Layout,           # [BH * S, qk]
    layout_s: Layout,           # [BH * S, S]
    BM: Int, BN: Int, BK: Int,
    WM: Int, WN: Int,
    MMA_M: Int, MMA_N: Int, MMA_K: Int,
    S: Int,
    QK: Int,
](
    Q: LayoutTensor[DType.bfloat16, layout_q, MutAnyOrigin],
    K: LayoutTensor[DType.bfloat16, layout_k, MutAnyOrigin],
    Sm: LayoutTensor[DType.float32, layout_s, MutAnyOrigin],
):
    var bh = Int(block_idx.z)
    var wid = Int(warp_id())
    var warp_y = wid // (BN // WN)
    var warp_x = wid % (BN // WN)

    var loader = TensorCore[DType.bfloat16, DType.bfloat16, Index(MMA_M, MMA_N, MMA_K)]()

    # Per-head row offset: each head occupies rows [bh*S, (bh+1)*S) of Q, K, Sm.
    # block_idx.y indexes BM-rows within that head; (bh*S/BM + block_idx.y) is the
    # global BM-tile index. Requires S % BM == 0 (true for S=128, BM=64; also for
    # S=2048, BM=64).
    comptime bm_per_head = S // BM
    var head_bm_off = bh * bm_per_head

    var S_warp_tile = Sm.tile[BM, BN](
        head_bm_off + Int(block_idx.y), Int(block_idx.x)
    ).tile[WM, WN](warp_y, warp_x)

    var A_smem = LayoutTensor[
        DType.bfloat16,
        Layout.row_major(BM, BK),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var B_smem = LayoutTensor[
        DType.bfloat16,
        Layout.row_major(BK, BN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var c_reg = (
        LayoutTensor[
            DType.float32,
            Layout.row_major(WM // MMA_M, (WN * 4) // MMA_N),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0.0)
    )

    var tid = Int(thread_idx.x)
    var nthreads = Int(block_dim.x)

    var num_k_iters = QK // BK
    for k_i in range(num_k_iters):
        barrier()

        # ---- Load A tile = Q[head_bm_off+block_y, k_i] (BM x BK), row-major. ----
        var Q_dram_tile = Q.tile[BM, BK](
            head_bm_off + Int(block_idx.y), k_i
        )
        copy_dram_to_sram_async[thread_layout=Layout.row_major(4, 8)](
            A_smem.vectorize[1, 4](), Q_dram_tile.vectorize[1, 4]()
        )

        # ---- Load B tile = K^T view ----
        # B_smem[k_inner, n_inner] = K[head_row_off + (block_x*BN + n_inner), k_i*BK + k_inner]
        # head_row_off = bh * S = head_bm_off * BM (rows-of-K within the global K matrix).
        # Thread-cooperative element gather: BK*BN = 32*64 = 2048 elems, 128 threads -> 16/thread.
        var head_row_off_K = head_bm_off * BM   # = bh * S
        var k_block_off_row = Int(block_idx.x) * BN
        var k_block_off_col = k_i * BK
        var n_per_thr = (BK * BN + nthreads - 1) // nthreads
        for it in range(n_per_thr):
            var lin = tid + it * nthreads
            if lin < BK * BN:
                var k_inner = lin // BN
                var n_inner = lin % BN
                B_smem[k_inner, n_inner] = K[
                    head_row_off_K + k_block_off_row + n_inner,
                    k_block_off_col + k_inner,
                ]

        async_copy_wait_all()
        barrier()

        var A_warp_tile = A_smem.tile[WM, BK](warp_y, 0)
        var B_warp_tile = B_smem.tile[BK, WN](0, warp_x)

        comptime for mma_k in range(BK // MMA_K):
            comptime for mma_m in range(WM // MMA_M):
                comptime for mma_n in range(WN // MMA_N):
                    var A_mma_tile = A_warp_tile.tile[MMA_M, MMA_K](mma_m, mma_k)
                    var B_mma_tile = B_warp_tile.tile[MMA_K, MMA_N](mma_k, mma_n)

                    var a_lt = loader.load_a(A_mma_tile)
                    var b_lt = loader.load_b(B_mma_tile)

                    var a_frag = SIMD[DType.bfloat16, 8](0)
                    a_frag[0] = a_lt[0, 0][0]
                    a_frag[1] = a_lt[0, 1][0]
                    a_frag[2] = a_lt[0, 2][0]
                    a_frag[3] = a_lt[0, 3][0]
                    a_frag[4] = a_lt[0, 4][0]
                    a_frag[5] = a_lt[0, 5][0]
                    a_frag[6] = a_lt[0, 6][0]
                    a_frag[7] = a_lt[0, 7][0]
                    var b_frag = SIMD[DType.bfloat16, 4](0)
                    b_frag[0] = b_lt[0, 0][0]
                    b_frag[1] = b_lt[0, 1][0]
                    b_frag[2] = b_lt[0, 2][0]
                    b_frag[3] = b_lt[0, 3][0]

                    var c_reg_tile = c_reg.tile[1, 4](mma_m, mma_n)
                    var c_frag = SIMD[DType.float32, 4](0)
                    c_frag[0] = c_reg_tile[0, 0][0]
                    c_frag[1] = c_reg_tile[0, 1][0]
                    c_frag[2] = c_reg_tile[0, 2][0]
                    c_frag[3] = c_reg_tile[0, 3][0]
                    var d_frag = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)

                    mma(d_frag, a_frag, b_frag, c_frag)

                    c_reg_tile[0, 0] = d_frag[0]
                    c_reg_tile[0, 1] = d_frag[1]
                    c_reg_tile[0, 2] = d_frag[2]
                    c_reg_tile[0, 3] = d_frag[3]

    # ---- Epilogue: m16n8 distribution -> S_warp_tile (f32). ----
    var lane = Int(lane_id())
    var group_id = lane >> 2
    var tid_in_grp = lane & 3

    comptime for mma_m in range(WM // MMA_M):
        comptime for mma_n in range(WN // MMA_N):
            var c_reg_tile = c_reg.tile[1, 4](mma_m, mma_n)
            var S_mma_tile = S_warp_tile.tile[MMA_M, MMA_N](mma_m, mma_n)
            comptime for i in range(4):
                var row = group_id + (i >> 1) * 8
                var col = (tid_in_grp << 1) + (i & 1)
                S_mma_tile[row, col] = c_reg_tile[0, i]


# ============================================================================
# Stage 2: row-wise softmax with scale.  P[bh*S + i, j] = softmax_j( Sm[bh*S+i, j] * scale ).
# One block per (bh, i) row (flattened to grid_dim.x = BH*S). SOFTMAX_TPB
# threads cooperate on max -> sum -> normalize. Output is bf16.
# ============================================================================
comptime SOFTMAX_TPB = 128

def softmax_kernel[
    layout_s: Layout,     # [BH*S, S]
    layout_p: Layout,     # [BH*S, S]
    S: Int,
](
    Sm: LayoutTensor[DType.float32, layout_s, MutAnyOrigin],
    P:  LayoutTensor[DType.bfloat16, layout_p, MutAnyOrigin],
    scale: Float32,
):
    # Flatten (bh, row) into a single 1-D grid coordinate.
    var global_row = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var sbuf = LayoutTensor[
        DType.float32,
        Layout.row_major(SOFTMAX_TPB),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # Pass 1: max across the row.
    var lmax: Float32 = -3.4e38
    var j = tid
    while j < S:
        var v = Sm[global_row, j][0] * scale
        if v > lmax:
            lmax = v
        j += SOFTMAX_TPB
    sbuf[tid] = lmax
    barrier()
    var stride = SOFTMAX_TPB // 2
    while stride > 0:
        if tid < stride:
            var a = sbuf[tid][0]
            var b = sbuf[tid + stride][0]
            sbuf[tid] = a if a > b else b
        barrier()
        stride = stride // 2
    var rmax = sbuf[0][0]

    # Pass 2: sum of exp.
    var lsum: Float32 = 0.0
    j = tid
    while j < S:
        var v = Sm[global_row, j][0] * scale
        lsum += exp(v - rmax)
        j += SOFTMAX_TPB
    sbuf[tid] = lsum
    barrier()
    stride = SOFTMAX_TPB // 2
    while stride > 0:
        if tid < stride:
            sbuf[tid] = sbuf[tid][0] + sbuf[tid + stride][0]
        barrier()
        stride = stride // 2
    var rsum = sbuf[0][0]
    var inv_sum: Float32 = 1.0 / rsum

    # Pass 3: write bf16 normalized probs.
    j = tid
    while j < S:
        var v = Sm[global_row, j][0] * scale
        var p = exp(v - rmax) * inv_sum
        P[global_row, j] = p.cast[DType.bfloat16]()
        j += SOFTMAX_TPB


# ============================================================================
# Stage 3: P @ V kernel.  out = probs @ V.
# Per-(b, h) head: P[S, S] @ V[S, d_v] -> O[S, d_v].
# Standard row-major matmul (no transpose). Reuses Wave 21's full pattern.
# ============================================================================
def pv_kernel[
    layout_p: Layout,           # [BH*S, S]
    layout_v: Layout,           # [BH*S, d_v]
    layout_o: Layout,           # [BH*S, d_v]
    BM: Int, BN: Int, BK: Int,
    WM: Int, WN: Int,
    MMA_M: Int, MMA_N: Int, MMA_K: Int,
    S: Int,
    DV: Int,
](
    P: LayoutTensor[DType.bfloat16, layout_p, MutAnyOrigin],
    V: LayoutTensor[DType.bfloat16, layout_v, MutAnyOrigin],
    O: LayoutTensor[DType.float32, layout_o, MutAnyOrigin],
):
    var bh = Int(block_idx.z)
    var wid = Int(warp_id())
    var warp_y = wid // (BN // WN)
    var warp_x = wid % (BN // WN)

    var loader = TensorCore[DType.bfloat16, DType.bfloat16, Index(MMA_M, MMA_N, MMA_K)]()

    # Per-head row offset (same scheme as qkt_kernel).
    comptime bm_per_head = S // BM
    var head_bm_off = bh * bm_per_head

    var O_warp_tile = O.tile[BM, BN](
        head_bm_off + Int(block_idx.y), Int(block_idx.x)
    ).tile[WM, WN](warp_y, warp_x)

    var A_smem = LayoutTensor[
        DType.bfloat16,
        Layout.row_major(BM, BK),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var B_smem = LayoutTensor[
        DType.bfloat16,
        Layout.row_major(BK, BN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var c_reg = (
        LayoutTensor[
            DType.float32,
            Layout.row_major(WM // MMA_M, (WN * 4) // MMA_N),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0.0)
    )

    # Number of K-tile iterations across S = K-dim of P @ V.
    # P is [BH*S, S]; P_head occupies rows [bh*S, (bh+1)*S). We tile across
    # the second dim (length S) in BK-sized slabs.
    comptime num_k_iters = S // BK
    # bk_per_head_row = S // BK = K-iters per row of P.
    # Within P, each head's row k-slab at iter k_i sits at column k_i*BK.
    # Within V, each head's k-slab at iter k_i sits at row head_row_off_V + k_i*BK.
    var head_row_off_V = head_bm_off * BM   # = bh * S, the row at which V's head starts.

    for k_i in range(num_k_iters):
        barrier()

        # P tile: rows [head_bm_off+block_y]*BM, cols [k_i*BK, (k_i+1)*BK)
        # P is shape (BH*S, S). tile[BM, BK]((head_bm_off+block_y), k_i) -> (BM, BK).
        var P_dram_tile = P.tile[BM, BK](
            head_bm_off + Int(block_idx.y), k_i
        )
        # V tile: rows [bh*S + k_i*BK, ...], cols [block_x*BN, ...].
        # V is shape (BH*S, DV). To get the right row block we need
        # tile[BK, BN](row_idx_in_BK_units, block_x).
        # Row offset (in BK-units): head_row_off_V/BK + k_i = (bh * S / BK) + k_i.
        # bk_per_head_v_row = S // BK; same as num_k_iters.
        comptime bk_per_head = S // BK
        var V_row_block_idx = bh * bk_per_head + k_i
        var V_dram_tile = V.tile[BK, BN](V_row_block_idx, Int(block_idx.x))

        copy_dram_to_sram_async[thread_layout=Layout.row_major(4, 8)](
            A_smem.vectorize[1, 4](), P_dram_tile.vectorize[1, 4]()
        )
        copy_dram_to_sram_async[thread_layout=Layout.row_major(4, 8)](
            B_smem.vectorize[1, 4](), V_dram_tile.vectorize[1, 4]()
        )

        async_copy_wait_all()
        barrier()

        var A_warp_tile = A_smem.tile[WM, BK](warp_y, 0)
        var B_warp_tile = B_smem.tile[BK, WN](0, warp_x)

        comptime for mma_k in range(BK // MMA_K):
            comptime for mma_m in range(WM // MMA_M):
                comptime for mma_n in range(WN // MMA_N):
                    var A_mma_tile = A_warp_tile.tile[MMA_M, MMA_K](mma_m, mma_k)
                    var B_mma_tile = B_warp_tile.tile[MMA_K, MMA_N](mma_k, mma_n)

                    var a_lt = loader.load_a(A_mma_tile)
                    var b_lt = loader.load_b(B_mma_tile)

                    var a_frag = SIMD[DType.bfloat16, 8](0)
                    a_frag[0] = a_lt[0, 0][0]
                    a_frag[1] = a_lt[0, 1][0]
                    a_frag[2] = a_lt[0, 2][0]
                    a_frag[3] = a_lt[0, 3][0]
                    a_frag[4] = a_lt[0, 4][0]
                    a_frag[5] = a_lt[0, 5][0]
                    a_frag[6] = a_lt[0, 6][0]
                    a_frag[7] = a_lt[0, 7][0]
                    var b_frag = SIMD[DType.bfloat16, 4](0)
                    b_frag[0] = b_lt[0, 0][0]
                    b_frag[1] = b_lt[0, 1][0]
                    b_frag[2] = b_lt[0, 2][0]
                    b_frag[3] = b_lt[0, 3][0]

                    var c_reg_tile = c_reg.tile[1, 4](mma_m, mma_n)
                    var c_frag = SIMD[DType.float32, 4](0)
                    c_frag[0] = c_reg_tile[0, 0][0]
                    c_frag[1] = c_reg_tile[0, 1][0]
                    c_frag[2] = c_reg_tile[0, 2][0]
                    c_frag[3] = c_reg_tile[0, 3][0]
                    var d_frag = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)

                    mma(d_frag, a_frag, b_frag, c_frag)

                    c_reg_tile[0, 0] = d_frag[0]
                    c_reg_tile[0, 1] = d_frag[1]
                    c_reg_tile[0, 2] = d_frag[2]
                    c_reg_tile[0, 3] = d_frag[3]

    var lane = Int(lane_id())
    var group_id = lane >> 2
    var tid_in_grp = lane & 3

    comptime for mma_m in range(WM // MMA_M):
        comptime for mma_n in range(WN // MMA_N):
            var c_reg_tile = c_reg.tile[1, 4](mma_m, mma_n)
            var O_mma_tile = O_warp_tile.tile[MMA_M, MMA_N](mma_m, mma_n)
            comptime for i in range(4):
                var row = group_id + (i >> 1) * 8
                var col = (tid_in_grp << 1) + (i & 1)
                O_mma_tile[row, col] = c_reg_tile[0, i]


# ============================================================================
# Main: end-to-end correctness + timed bench at DeepSeek-V3 decode shape.
# ============================================================================
def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
        return

    with DeviceContext() as ctx:
        print("GPU:", ctx.name())

        # ---- Tile shape (Wave 21 pattern, kept unchanged for W22.5b) ----
        comptime BM = 64
        comptime BN = 64
        comptime BK = 32
        comptime WM = 32
        comptime WN = 32
        comptime MMA_M = 16
        comptime MMA_N = 8
        comptime MMA_K = 16
        comptime NUM_WARPS = (BM // WM) * (BN // WN)
        comptime BLOCK_THREADS = NUM_WARPS * WARP_SIZE  # 128

        # ---- Problem shape: DeepSeek-V3 decode (B=1, n_h=128, S=2048, qk=192, d_v=128). ----
        # All multiples of BM/BN/BK so no boundary guards needed.
        # Memory budget on RTX 5090 (32 GiB):
        #   Q, K bf16:  BH*S*QK *2 = 128*2048*192*2 = 100 MiB each
        #   V    bf16:  BH*S*DV *2 = 128*2048*128*2 =  64 MiB
        #   S    f32 :  BH*S*S  *4 = 128*2048*2048*4 = 2.0 GiB
        #   P    bf16:  BH*S*S  *2 = 128*2048*2048*2 = 1.0 GiB
        #   O    f32 :  BH*S*DV *4 = 128*2048*128*4 = 128 MiB
        #   Plus host mirrors of Q,K,V,O (~390 MiB).
        # Total ≈ 3.5 GiB device + 0.4 GiB host. Well under 32 GiB.
        comptime B = 1
        comptime NH = 128
        comptime S = 2048
        comptime QK = 192
        comptime DV = 128
        comptime BH = B * NH
        comptime QK_ELEMS = BH * S * QK
        comptime V_ELEMS  = BH * S * DV
        comptime SCORE_ELEMS = BH * S * S

        # Flat 2-D layouts: heads concatenated along the row dim.
        comptime layout_q = Layout.row_major(BH * S, QK)
        comptime layout_k = Layout.row_major(BH * S, QK)
        comptime layout_v = Layout.row_major(BH * S, DV)
        comptime layout_s = Layout.row_major(BH * S, S)
        comptime layout_p = Layout.row_major(BH * S, S)
        comptime layout_o = Layout.row_major(BH * S, DV)

        # ---- Buffers ----
        var q_dev = ctx.enqueue_create_buffer[DType.bfloat16](QK_ELEMS)
        var k_dev = ctx.enqueue_create_buffer[DType.bfloat16](QK_ELEMS)
        var v_dev = ctx.enqueue_create_buffer[DType.bfloat16](V_ELEMS)
        var s_dev = ctx.enqueue_create_buffer[DType.float32](SCORE_ELEMS)
        var p_dev = ctx.enqueue_create_buffer[DType.bfloat16](SCORE_ELEMS)
        var o_dev = ctx.enqueue_create_buffer[DType.float32](V_ELEMS)

        var q_host = ctx.enqueue_create_host_buffer[DType.bfloat16](QK_ELEMS)
        var k_host = ctx.enqueue_create_host_buffer[DType.bfloat16](QK_ELEMS)
        var v_host = ctx.enqueue_create_host_buffer[DType.bfloat16](V_ELEMS)
        var o_host = ctx.enqueue_create_host_buffer[DType.float32](V_ELEMS)
        ctx.synchronize()

        # ---- Init Q, K, V with deterministic small-magnitude bf16 values. ----
        # Small magnitudes keep softmax numerically stable.
        for i in range(QK_ELEMS):
            q_host[i] = (Float32(((i * 2654435761) % 64)) * 0.01 - 0.32).cast[DType.bfloat16]()
        for i in range(QK_ELEMS):
            k_host[i] = (Float32((((i + 17) * 2654435761) % 64)) * 0.01 - 0.32).cast[DType.bfloat16]()
        for i in range(V_ELEMS):
            v_host[i] = (Float32((((i + 31) * 2654435761) % 64)) * 0.01 - 0.32).cast[DType.bfloat16]()
        ctx.enqueue_copy(dst_buf=q_dev, src_buf=q_host)
        ctx.enqueue_copy(dst_buf=k_dev, src_buf=k_host)
        ctx.enqueue_copy(dst_buf=v_dev, src_buf=v_host)
        ctx.synchronize()

        var Q_lt = LayoutTensor[DType.bfloat16, layout_q, MutAnyOrigin](q_dev.unsafe_ptr())
        var K_lt = LayoutTensor[DType.bfloat16, layout_k, MutAnyOrigin](k_dev.unsafe_ptr())
        var V_lt = LayoutTensor[DType.bfloat16, layout_v, MutAnyOrigin](v_dev.unsafe_ptr())
        var S_lt = LayoutTensor[DType.float32,  layout_s, MutAnyOrigin](s_dev.unsafe_ptr())
        var P_lt = LayoutTensor[DType.bfloat16, layout_p, MutAnyOrigin](p_dev.unsafe_ptr())
        var O_lt = LayoutTensor[DType.float32,  layout_o, MutAnyOrigin](o_dev.unsafe_ptr())

        # ---- Kernel handles ----
        comptime kernel_qkt = qkt_kernel[
            layout_q, layout_k, layout_s,
            BM, BN, BK, WM, WN, MMA_M, MMA_N, MMA_K,
            S, QK,
        ]
        comptime kernel_sm = softmax_kernel[layout_s, layout_p, S]
        comptime kernel_pv = pv_kernel[
            layout_p, layout_v, layout_o,
            BM, BN, BK, WM, WN, MMA_M, MMA_N, MMA_K,
            S, DV,
        ]
        var scale: Float32 = 1.0 / sqrt(Float32(QK))

        # ---- Warmup launch (also captures SASS for the matmul kernels) ----
        ctx.enqueue_function[kernel_qkt, kernel_qkt, _dump_sass=True](
            Q_lt, K_lt, S_lt,
            grid_dim=(ceildiv(S, BN), ceildiv(S, BM), BH),
            block_dim=(BLOCK_THREADS,),
        )
        ctx.enqueue_function[kernel_sm, kernel_sm](
            S_lt, P_lt, scale,
            grid_dim=(BH * S,),
            block_dim=(SOFTMAX_TPB,),
        )
        ctx.enqueue_function[kernel_pv, kernel_pv, _dump_sass=True](
            P_lt, V_lt, O_lt,
            grid_dim=(ceildiv(DV, BN), ceildiv(S, BM), BH),
            block_dim=(BLOCK_THREADS,),
        )
        ctx.synchronize()

        # ---- Timed run: 10 per-iter ctx.execution_time samples -> median. ----
        # @parameter def body re-enqueues all 3 kernels per iter; cudaEvent
        # bracket spans the full pipeline (qkt + softmax + pv) on the stream.
        @parameter
        def body(ctx: DeviceContext) raises -> None:
            ctx.enqueue_function[kernel_qkt, kernel_qkt](
                Q_lt, K_lt, S_lt,
                grid_dim=(ceildiv(S, BN), ceildiv(S, BM), BH),
                block_dim=(BLOCK_THREADS,),
            )
            ctx.enqueue_function[kernel_sm, kernel_sm](
                S_lt, P_lt, scale,
                grid_dim=(BH * S,),
                block_dim=(SOFTMAX_TPB,),
            )
            ctx.enqueue_function[kernel_pv, kernel_pv](
                P_lt, V_lt, O_lt,
                grid_dim=(ceildiv(DV, BN), ceildiv(S, BM), BH),
                block_dim=(BLOCK_THREADS,),
            )

        var num_iters = 10
        var iter_ms = SIMD[DType.float64, 16](0.0)
        for it in range(num_iters):
            var t = ctx.execution_time[body](1)
            iter_ms[it] = Float64(t) / 1e6  # ns -> ms
        ctx.synchronize()

        # Insertion sort iter_ms[0:num_iters] for median.
        for ii in range(1, num_iters):
            var key = iter_ms[ii]
            var jj = ii - 1
            while jj >= 0 and iter_ms[jj] > key:
                iter_ms[jj + 1] = iter_ms[jj]
                jj -= 1
            iter_ms[jj + 1] = key
        var median_ms = iter_ms[num_iters // 2]
        var min_ms = iter_ms[0]
        var max_ms_iter = iter_ms[num_iters - 1]

        # ---- Copy back final iteration's output for correctness check. ----
        ctx.enqueue_copy(dst_buf=o_host, src_buf=o_dev)
        ctx.synchronize()

        # =====================================================================
        # CPU reference: SDPA (no mask). Sample-based correctness check.
        # 1024 samples, tolerance atol=1e-2 + rtol=1e-3*|ref| (Wave 21 spec).
        # Per sample: full row Q@K^T (S*QK ≈ 393K mult-adds) + softmax (3*S) +
        #             reduce-with-V (S mult-adds). 1024 samples ≈ 400M ops in
        #             interpreted Mojo — runs in a few seconds.
        # =====================================================================
        var max_err: Float32 = 0.0
        var max_rel_err: Float32 = 0.0
        var fail_bh = -1
        var fail_i  = -1
        var fail_d  = -1
        var fail_got: Float32 = 0.0
        var fail_ref: Float32 = 0.0

        var num_samples = 1024
        for s_idx in range(num_samples):
            # Knuth golden-ratio hash. Use disjoint high-bit windows for bh, i, d.
            var seed = s_idx * 2654435761
            var bh   = (((seed >> 24) % BH) + BH) % BH
            var i    = (((seed >> 11) % S)  + S)  % S
            var d    = (((seed >> 4)  % DV) + DV) % DV

            # Compute full row scores[j] = Q[bh, i, :] . K[bh, j, :] * scale.
            var row_scores = InlineArray[Float32, S](fill=0.0)
            for j in range(S):
                var sj: Float32 = 0.0
                for kk in range(QK):
                    var qv = q_host[bh * S * QK + i * QK + kk].cast[DType.float32]()
                    var kv = k_host[bh * S * QK + j * QK + kk].cast[DType.float32]()
                    sj += qv * kv
                row_scores[j] = sj * scale

            # Softmax over j.
            var rmax: Float32 = -3.4e38
            for j in range(S):
                if row_scores[j] > rmax:
                    rmax = row_scores[j]
            var rsum: Float32 = 0.0
            var probs = InlineArray[Float32, S](fill=0.0)
            for j in range(S):
                var e = exp(row_scores[j] - rmax)
                probs[j] = e
                rsum += e
            var inv_sum: Float32 = 1.0 / rsum
            for j in range(S):
                probs[j] = probs[j] * inv_sum

            # Cast probs through bf16 (matches kernel's intermediate dtype) and
            # reduce against V[bh, :, d].
            var refv: Float32 = 0.0
            for j in range(S):
                var pj = probs[j].cast[DType.bfloat16]().cast[DType.float32]()
                var vd = v_host[bh * S * DV + j * DV + d].cast[DType.float32]()
                refv += pj * vd

            var got = o_host[bh * S * DV + i * DV + d]
            var abs_err = abs(got - refv)
            var ref_abs = abs(refv)
            var rel_err: Float32 = 0.0
            if ref_abs > 0.0:
                rel_err = abs_err / ref_abs
            if abs_err > max_err:
                max_err = abs_err
            if rel_err > max_rel_err:
                max_rel_err = rel_err
            # Tolerance: atol=1e-2 + rtol=1e-3*|ref| (Wave 21 / Phase-7 spec).
            if abs_err > 1e-2 + 1e-3 * ref_abs and fail_bh < 0:
                fail_bh = bh
                fail_i = i
                fail_d = d
                fail_got = got
                fail_ref = refv

        # ---- Report. ----
        # FLOPs per iter (per Wave 17 W1a / ADR-0005 useful_flops convention):
        #   2 * B * n_h * S^2 * (qk + d_v)
        # = 2 * 1 * 128 * 2048^2 * (192 + 128)
        # = 2 * 128 * 4194304 * 320
        # = 343,597,383,680 = 343.6 GFLOPS
        var flops_per_iter: Float64 = (
            2.0 * Float64(B) * Float64(NH) * Float64(S) * Float64(S)
            * (Float64(QK) + Float64(DV))
        )
        var median_s: Float64 = median_ms * 1e-3
        var min_s: Float64 = min_ms * 1e-3
        var tflops_median: Float64 = flops_per_iter / median_s / 1e12
        var tflops_best: Float64 = flops_per_iter / min_s / 1e12

        print("[mojo-attn-bf16] shape: B=", B, " n_h=", NH, " S=", S, " qk=", QK, " d_v=", DV)
        print("[mojo-attn-bf16] tile: BM=", BM, " BN=", BN, " BK=", BK,
              " MMA=", MMA_M, "x", MMA_N, "x", MMA_K)
        print("[mojo-attn-bf16] flops_per_iter=", flops_per_iter / 1e9, " GFLOPS")
        print("[mojo-attn-bf16] timing: min_ms=", min_ms,
              " median_ms=", median_ms,
              " max_ms=", max_ms_iter)
        print("[mojo-attn-bf16] TFLOPS_median=", tflops_median,
              " TFLOPS_best=", tflops_best)
        print("[mojo-attn-bf16] correctness: max_abs_err=", max_err,
              " max_rel_err=", max_rel_err, " (vs CPU SDPA ref, ", num_samples, " samples)")
        if fail_bh >= 0:
            print("[mojo-attn-bf16] FAIL at (bh=", fail_bh, " i=", fail_i, " d=", fail_d,
                  "): got=", fail_got, " ref=", fail_ref)
        else:
            print("[mojo-attn-bf16] correctness PASSED at deepseek_v3 shape (1024 samples)")

        # Cross-frontend MLA comparison (Wave 17 W1a):
        print("[mojo-attn-bf16] cross-frontend MLA at this shape:")
        print("[mojo-attn-bf16]   cuTile-MLA      = 112.00 TF (fused, FlashAttention-class)")
        print("[mojo-attn-bf16]   cublas-attn-mla =  47.00 TF (3-kernel, cuBLAS GEMMs)")
        print("[mojo-attn-bf16]   oxide-attn-mla  =  24.70 TF (3-kernel, hand-WMMA)")
        print("[mojo-attn-bf16]   cuda-attn-mla   =  24.17 TF (3-kernel, hand-WMMA)")
        print("[mojo-attn-bf16]   mojo-attn-bf16  =  ", tflops_median, " TF (3-kernel, hand-MMA, this run, median)")
