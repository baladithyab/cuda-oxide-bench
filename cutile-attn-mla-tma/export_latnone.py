"""Export cubins for latency=None case to compare."""
import sys, os
sys.path.insert(0, '/home/codeseys/cuda-exploration/cutile-attn-mla-tma')
import cuda.tile as ct
import math
from cuda.tile.compilation import (
    ArrayConstraint, CallingConvention, KernelSignature, export_kernel,
)
from main import _pad_to_pow2_ge, pick_blocks, SHAPE_BENCH_MLA

shape = SHAPE_BENCH_MLA
qk_pad = _pad_to_pow2_ge(shape.qk_head_dim)
bm, bn = pick_blocks(shape)
SEQ_TILES_M = shape.seq // bm; SEQ_TILES_N = shape.seq // bn
scale = 1.0 / math.sqrt(shape.qk_head_dim)
NEG_INF = -1.0e30
QK_PAD, D_V, SEQ = qk_pad, shape.d_v, shape.seq
BLOCK_M, BLOCK_N = bm, bn

def build_export(allow_tma, out_path):
    @ct.kernel
    def k_func(Q, K, V, O):
        bid0 = ct.bid(0); bid1 = ct.bid(1)
        qrow = bid0 * SEQ_TILES_M + bid1
        kbase = bid0 * SEQ_TILES_N
        qv = Q.tiled_view((BLOCK_M, QK_PAD), padding_mode=ct.PaddingMode.ZERO)
        kv_ = K.tiled_view((BLOCK_N, QK_PAD), padding_mode=ct.PaddingMode.ZERO)
        vv = V.tiled_view((BLOCK_N, D_V), padding_mode=ct.PaddingMode.ZERO)
        ov = O.tiled_view((BLOCK_M, D_V), padding_mode=ct.PaddingMode.ZERO)
        qt = qv.load((qrow, 0), allow_tma=allow_tma)
        mi = ct.full((BLOCK_M, 1), NEG_INF, ct.float32)
        li = ct.zeros((BLOCK_M, 1), ct.float32)
        oacc = ct.zeros((BLOCK_M, D_V), ct.float32)
        for kb in range(SEQ_TILES_N):
            kt = kv_.load((kbase+kb, 0), allow_tma=allow_tma)
            vt = vv.load((kbase+kb, 0), allow_tma=allow_tma)
            k_t = ct.transpose(kt)
            sa = ct.zeros((BLOCK_M, BLOCK_N), ct.float32)
            sa = ct.mma(qt, k_t, sa)
            ss = sa * scale
            mr = ct.max(ss, axis=1, keepdims=True)
            mn = ct.maximum(mi, mr)
            a = ct.exp(mi - mn)
            p = ct.exp(ss - mn)
            ps = ct.sum(p, axis=1, keepdims=True)
            li = a*li + ps
            oacc = oacc * a
            pf = p.astype(ct.float16)
            oacc = ct.mma(pf, vt, oacc)
            mi = mn
        ofi = oacc / li
        ov.store((qrow, 0), ofi.astype(O.dtype), allow_tma=allow_tma)
    def _ac(dt):
        return ArrayConstraint(
            dtype=dt, ndim=2, index_dtype=ct.int32,
            stride_lower_bound_incl=0, alias_groups=(), may_alias_internally=False,
            stride_constant=(None, 1), stride_divisible_by=1,
            shape_divisible_by=1, base_addr_divisible_by=1,
        )
    sig = KernelSignature(
        parameters=[_ac(ct.float16)]*4,
        calling_convention=CallingConvention.cutile_python_v1(),
    )
    export_kernel(k_func, [sig], out_path, gpu_code='sm_120', output_format='cubin')
    print(f'  wrote {out_path}  ({os.path.getsize(out_path)} bytes)')

build_export(True,  'cubin_latNone_on.cubin')
build_export(False, 'cubin_latNone_off.cubin')
