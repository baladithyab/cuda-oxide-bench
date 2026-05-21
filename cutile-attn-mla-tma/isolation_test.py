"""Isolation experiment: which kwarg drives the perf delta? allow_tma or latency?"""
import sys, math
sys.path.insert(0, '/home/codeseys/cuda-exploration/cutile-attn-mla-tma')
import cuda.tile as ct
import cupy
import numpy as np
from main import (load_inputs, prepare_device, _pad_to_pow2_ge,
                  pick_blocks, mla_attention_flops, SHAPE_BENCH_MLA, ITERS, WARMUP)

shape = SHAPE_BENCH_MLA
q_np, k_np, v_np, _ = load_inputs(shape)
qk_pad = _pad_to_pow2_ge(shape.qk_head_dim)
q_d, k_d, v_d = prepare_device(q_np, k_np, v_np, shape, qk_pad)
o_d = cupy.zeros((q_d.shape[0], shape.d_v), dtype=cupy.float16)
bm, bn = pick_blocks(shape)
SEQ_TILES_M = shape.seq // bm; SEQ_TILES_N = shape.seq // bn
scale = 1.0 / math.sqrt(shape.qk_head_dim)
NEG_INF = -1.0e30
QK_PAD, D_V, SEQ = qk_pad, shape.d_v, shape.seq
BLOCK_M, BLOCK_N = bm, bn
grid = (shape.batch * shape.n_h, shape.seq // bm)
stream = cupy.cuda.get_current_stream()
flops = mla_attention_flops(shape)


def make_kernel(allow_tma, latency):
    if latency is None:
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
        return k_func
    else:
        @ct.kernel
        def k_func(Q, K, V, O):
            bid0 = ct.bid(0); bid1 = ct.bid(1)
            qrow = bid0 * SEQ_TILES_M + bid1
            kbase = bid0 * SEQ_TILES_N
            qv = Q.tiled_view((BLOCK_M, QK_PAD), padding_mode=ct.PaddingMode.ZERO)
            kv_ = K.tiled_view((BLOCK_N, QK_PAD), padding_mode=ct.PaddingMode.ZERO)
            vv = V.tiled_view((BLOCK_N, D_V), padding_mode=ct.PaddingMode.ZERO)
            ov = O.tiled_view((BLOCK_M, D_V), padding_mode=ct.PaddingMode.ZERO)
            qt = qv.load((qrow, 0), latency=latency, allow_tma=allow_tma)
            mi = ct.full((BLOCK_M, 1), NEG_INF, ct.float32)
            li = ct.zeros((BLOCK_M, 1), ct.float32)
            oacc = ct.zeros((BLOCK_M, D_V), ct.float32)
            for kb in range(SEQ_TILES_N):
                kt = kv_.load((kbase+kb, 0), latency=latency, allow_tma=allow_tma)
                vt = vv.load((kbase+kb, 0), latency=latency, allow_tma=allow_tma)
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
            ov.store((qrow, 0), ofi.astype(O.dtype), latency=latency, allow_tma=allow_tma)
        return k_func


def bench_kernel(kernel, label):
    for _ in range(WARMUP):
        ct.launch(stream.ptr, grid, kernel, (q_d, k_d, v_d, o_d))
    cupy.cuda.runtime.deviceSynchronize()
    starts = [cupy.cuda.Event() for _ in range(ITERS)]
    ends   = [cupy.cuda.Event() for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record(stream); ct.launch(stream.ptr, grid, kernel, (q_d, k_d, v_d, o_d)); ends[i].record(stream)
    stream.synchronize()
    times = sorted(cupy.cuda.get_elapsed_time(starts[i], ends[i]) for i in range(ITERS))
    best = times[0]; med = times[ITERS//2]
    print(f'{label:36s}  best={best:.3f}ms ({flops/(best*1e-3)/1e12:6.2f} TF)  median={med:.3f}ms ({flops/(med*1e-3)/1e12:6.2f} TF)')

# Run all 6 configs in one process
configs = [
    (True,  None, 'allow_tma=True  latency=None'),
    (False, None, 'allow_tma=False latency=None'),
    (True,  10,   'allow_tma=True  latency=10  '),
    (False, 10,   'allow_tma=False latency=10  '),
    (True,  1,    'allow_tma=True  latency=1   '),
    (False, 1,    'allow_tma=False latency=1   '),
]
kernels = [(make_kernel(at, lt), label) for at, lt, label in configs]
print('=== single-process bench across 6 configs ===')
for k, label in kernels:
    bench_kernel(k, label)
