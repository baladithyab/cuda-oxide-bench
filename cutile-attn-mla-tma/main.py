"""Wave C3.2 — cuTile fused MLA with explicit-TMA-mode annotations.

Clone of `cutile-attn-mla/main.py` with `allow_tma=True` (or False, per
CLI flag) and `latency=10` plumbed through every `TiledView.load` and
`TiledView.store`. Mirrors the W-C2.5 transformation pattern in
`cutile-attn-gdn-tma/main.py`.

Hypothesis (going in):
  C2.5 falsified TMA emission for GDN's tile shapes (1×D_K row tiles +
  D_K×BLOCK_V state tiles → 0 UTMA insts, cubin byte-identical between
  allow_tma=True and allow_tma=False). MLA's tile shapes are different:
  matmul-friendly (BLOCK_M, QK_PAD) = (64, 256) f16, (BLOCK_N, QK_PAD) =
  (64, 256) f16, (BLOCK_N, D_V) = (64, 128) f16, (BLOCK_M, D_V) = (64,
  128) f16 — these are *exactly* the kind of canonical 2D rectangles
  that cuTile's matmul_tiled emits UTMA for (BM=BN=128, BK=16 confirmed
  in W13 SASS with 17 UTMA matches). MLA's tile shapes are larger and
  more rectangular than GDN's row/state tiles, so the heuristic *might*
  accept them.

Acceptance:
  - kernel runs, correctness within MLA tolerance
  - bench TF reported (best/median)
  - SASS UTMALDG count documented (0 = falsification, >0 = confirms
    cuTile DSL CAN emit TMA on attention-class kernels at right shapes)
  - byte-comparison of cubin vs allow_tma=False (md5sum)

Original MLA kernel docstring follows.
================================================================

Wave 16.3 — cuTile fused MLA (Multi-Head Latent Attention, DeepSeek-V3).

Single @ct.kernel implementing FlashAttention-2-style forward attention
over MLA-shaped inputs. See cutile-attn-mla/main.py for the full design
notes.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cuda.tile as ct
import cupy
import numpy as np
from cuda.tile.compilation import (
    ArrayConstraint,
    CallingConvention,
    KernelSignature,
    export_kernel,
)

# Hook the wave-15 shared infra (MLA shapes, FLOPS, tolerances).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(
    0, str(REPO_ROOT / "analysis" / "wave15-attention-architecture" / "reference")
)
from shapes_mla import MLAShape, SHAPE_CORRECTNESS_MLA, SHAPE_BENCH_MLA  # noqa: E402
from flops_mla import mla_attention_flops  # noqa: E402
from tolerances import get as get_tol  # noqa: E402

INPUTS_DIR = REPO_ROOT / "analysis" / "wave15-attention-architecture" / "inputs"

WARMUP = 2
ITERS = 10


def _pad_to_pow2_ge(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Kernel factory
# ─────────────────────────────────────────────────────────────────────────────

def make_mla_kernel(
    BLOCK_M: int, BLOCK_N: int,
    QK_PAD: int, D_V: int, SEQ: int, N_H: int,
    qk_head_dim_true: int,
    allow_tma: bool = True,
    latency: int = 10,
):
    """Build a @ct.kernel specialized to these shape constants.

    Wave C3.2: every load/store call site takes explicit ``allow_tma``
    and ``latency`` kwargs. ``allow_tma=True`` is the cuTile API default
    (TMA is opt-OUT, not opt-IN) — we pass it explicitly to document
    intent and to give us the dial to falsification-test against
    ``allow_tma=False``.
    """
    SEQ_TILES_M = SEQ // BLOCK_M
    SEQ_TILES_N = SEQ // BLOCK_N
    scale = 1.0 / math.sqrt(qk_head_dim_true)
    NEG_INF = -1.0e30

    @ct.kernel
    def mla_fwd(Q, K, V, O):
        bid0 = ct.bid(0)  # flattened (batch, head)
        bid1 = ct.bid(1)  # query block

        q_tile_row = bid0 * SEQ_TILES_M + bid1
        kv_tile_row_base = bid0 * SEQ_TILES_N

        q_view = Q.tiled_view((BLOCK_M, QK_PAD), padding_mode=ct.PaddingMode.ZERO)
        k_view = K.tiled_view((BLOCK_N, QK_PAD), padding_mode=ct.PaddingMode.ZERO)
        v_view = V.tiled_view((BLOCK_N, D_V), padding_mode=ct.PaddingMode.ZERO)
        o_view = O.tiled_view((BLOCK_M, D_V), padding_mode=ct.PaddingMode.ZERO)

        # Q tile: persists across the K/V loop.
        q_tile = q_view.load((q_tile_row, 0),
                             latency=latency, allow_tma=allow_tma)

        # Online-softmax state (all f32, in registers).
        m_i = ct.full((BLOCK_M, 1), NEG_INF, ct.float32)
        l_i = ct.zeros((BLOCK_M, 1), ct.float32)
        o_acc = ct.zeros((BLOCK_M, D_V), ct.float32)

        for kb in range(SEQ_TILES_N):
            k_tile = k_view.load((kv_tile_row_base + kb, 0),
                                 latency=latency, allow_tma=allow_tma)
            v_tile = v_view.load((kv_tile_row_base + kb, 0),
                                 latency=latency, allow_tma=allow_tma)

            # QK^T: (BLOCK_M, QK_PAD) × (QK_PAD, BLOCK_N) → (BLOCK_M, BLOCK_N) f32
            k_t = ct.transpose(k_tile)
            s_acc = ct.zeros((BLOCK_M, BLOCK_N), ct.float32)
            s_acc = ct.mma(q_tile, k_t, s_acc)

            s_scaled = s_acc * scale

            m_row = ct.max(s_scaled, axis=1, keepdims=True)
            m_new = ct.maximum(m_i, m_row)
            alpha = ct.exp(m_i - m_new)

            p = ct.exp(s_scaled - m_new)
            p_row_sum = ct.sum(p, axis=1, keepdims=True)

            l_i = alpha * l_i + p_row_sum
            o_acc = o_acc * alpha

            p_f16 = p.astype(ct.float16)
            o_acc = ct.mma(p_f16, v_tile, o_acc)

            m_i = m_new

        o_final = o_acc / l_i
        o_view.store((q_tile_row, 0), o_final.astype(O.dtype),
                     latency=latency, allow_tma=allow_tma)

    return mla_fwd


# ─────────────────────────────────────────────────────────────────────────────
# Input I/O + padding
# ─────────────────────────────────────────────────────────────────────────────

def load_inputs(shape: MLAShape):
    prefix = INPUTS_DIR / f"mla_{shape.name}"
    q_np = np.load(f"{prefix}_q_f16.npy")
    k_np = np.load(f"{prefix}_k_f16.npy")
    v_np = np.load(f"{prefix}_v_f16.npy")
    exp_np = np.load(f"{prefix}_expected_f32.npy")
    return q_np, k_np, v_np, exp_np


def prepare_device(q_np, k_np, v_np, shape: MLAShape, qk_pad: int):
    B, S, N = shape.batch, shape.seq, shape.n_h
    qk = shape.qk_head_dim
    dv = shape.d_v
    assert q_np.shape == (B, N, S, qk)
    assert k_np.shape == (B, N, S, qk)
    assert v_np.shape == (B, N, S, dv)

    if qk_pad == qk:
        q_2d = q_np.reshape(B * N * S, qk)
        k_2d = k_np.reshape(B * N * S, qk)
    else:
        q_2d = np.zeros((B * N * S, qk_pad), dtype=np.float16)
        k_2d = np.zeros((B * N * S, qk_pad), dtype=np.float16)
        q_2d[:, :qk] = q_np.reshape(B * N * S, qk)
        k_2d[:, :qk] = k_np.reshape(B * N * S, qk)
    v_2d = v_np.reshape(B * N * S, dv)
    return (
        cupy.asarray(q_2d, dtype=cupy.float16),
        cupy.asarray(k_2d, dtype=cupy.float16),
        cupy.asarray(v_2d, dtype=cupy.float16),
    )


def pick_blocks(shape: MLAShape) -> tuple[int, int]:
    if shape.seq >= 512:
        return 64, 64
    bm = min(32, shape.seq)
    bn = min(32, shape.seq)
    assert shape.seq % bm == 0 and shape.seq % bn == 0
    return bm, bn


# ─────────────────────────────────────────────────────────────────────────────
# Smoke
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke(shape: MLAShape, allow_tma: bool = True, latency: int = 10) -> bool:
    print(
        f"[smoke] shape={shape.name} B={shape.batch} S={shape.seq} "
        f"n_h={shape.n_h} qk={shape.qk_head_dim} d_v={shape.d_v}  "
        f"allow_tma={allow_tma} latency={latency}"
    )
    q_np, k_np, v_np, expected = load_inputs(shape)
    qk_pad = _pad_to_pow2_ge(shape.qk_head_dim)
    print(f"[smoke] qk_head_dim={shape.qk_head_dim} → qk_pad={qk_pad}")
    q_d, k_d, v_d = prepare_device(q_np, k_np, v_np, shape, qk_pad)
    o_d = cupy.zeros((q_d.shape[0], shape.d_v), dtype=cupy.float16)

    bm, bn = pick_blocks(shape)
    print(f"[smoke] BLOCK_M={bm} BLOCK_N={bn}")
    kernel = make_mla_kernel(
        bm, bn, qk_pad, shape.d_v, shape.seq, shape.n_h,
        qk_head_dim_true=shape.qk_head_dim,
        allow_tma=allow_tma, latency=latency,
    )

    grid = (shape.batch * shape.n_h, shape.seq // bm)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream.ptr, grid, kernel, (q_d, k_d, v_d, o_d))
    cupy.cuda.runtime.deviceSynchronize()

    o_np = o_d.get().reshape(shape.batch, shape.n_h, shape.seq, shape.d_v)
    o_f32 = o_np.astype(np.float32)

    tol = get_tol("f16")
    abs_err = np.abs(o_f32 - expected)
    ref_mag = np.abs(expected).max() + 1e-30
    max_abs = float(abs_err.max())
    rel_err = max_abs / ref_mag
    ok = np.allclose(o_f32, expected, atol=tol.atol, rtol=tol.rtol)
    status = "OK" if ok else "FAIL"
    print(
        f"[smoke] max_abs={max_abs:.3e} rel={rel_err:.3e}  "
        f"atol={tol.atol} rtol={tol.rtol}  {status}"
    )
    if not ok:
        worst = np.unravel_index(abs_err.argmax(), abs_err.shape)
        print(
            f"        worst at {worst}: got={o_f32[worst]:.4f} "
            f"expected={expected[worst]:.4f}"
        )
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Bench
# ─────────────────────────────────────────────────────────────────────────────

def run_bench(shape: MLAShape, csv_path: str,
              allow_tma: bool = True, latency: int = 10) -> None:
    print(
        f"[bench] shape={shape.name} B={shape.batch} S={shape.seq} "
        f"n_h={shape.n_h} qk={shape.qk_head_dim} d_v={shape.d_v}  "
        f"allow_tma={allow_tma} latency={latency}"
    )
    q_np, k_np, v_np, _expected = load_inputs(shape)
    qk_pad = _pad_to_pow2_ge(shape.qk_head_dim)
    print(f"[bench] qk_head_dim={shape.qk_head_dim} → qk_pad={qk_pad}")
    q_d, k_d, v_d = prepare_device(q_np, k_np, v_np, shape, qk_pad)
    o_d = cupy.zeros((q_d.shape[0], shape.d_v), dtype=cupy.float16)

    bm, bn = pick_blocks(shape)
    print(f"[bench] BLOCK_M={bm} BLOCK_N={bn}")
    kernel = make_mla_kernel(
        bm, bn, qk_pad, shape.d_v, shape.seq, shape.n_h,
        qk_head_dim_true=shape.qk_head_dim,
        allow_tma=allow_tma, latency=latency,
    )

    grid = (shape.batch * shape.n_h, shape.seq // bm)
    stream = cupy.cuda.get_current_stream()
    flops = mla_attention_flops(shape)

    for _ in range(WARMUP):
        ct.launch(stream.ptr, grid, kernel, (q_d, k_d, v_d, o_d))
    cupy.cuda.runtime.deviceSynchronize()

    starts = [cupy.cuda.Event() for _ in range(ITERS)]
    ends = [cupy.cuda.Event() for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record(stream)
        ct.launch(stream.ptr, grid, kernel, (q_d, k_d, v_d, o_d))
        ends[i].record(stream)
    stream.synchronize()

    rows = []
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "impl", "kernel", "batch", "seq", "n_h", "qk_head_dim", "qk_pad",
            "d_v", "block_m", "block_n", "iter", "gpu_ms", "tflops",
        ])
        for i in range(ITERS):
            gpu_ms = cupy.cuda.get_elapsed_time(starts[i], ends[i])
            tflops = flops / (gpu_ms * 1e-3) / 1e12
            print(f"[bench] iter={i} gpu_ms={gpu_ms:.3f} tflops={tflops:.3f}")
            w.writerow([
                "cutile", "mla_fwd_fused", shape.batch, shape.seq, shape.n_h,
                shape.qk_head_dim, qk_pad, shape.d_v, bm, bn, i,
                f"{gpu_ms:.6f}", f"{tflops:.6f}",
            ])
            rows.append((gpu_ms, tflops))

    ms_sorted = sorted(r[0] for r in rows)
    tf_sorted = sorted(r[1] for r in rows)
    median_ms = ms_sorted[ITERS // 2]
    median_tf = tf_sorted[ITERS // 2]
    best_ms = ms_sorted[0]
    best_tf = tf_sorted[-1]
    print()
    print("=" * 64)
    print(
        f" BENCH SUMMARY — {shape.name} (B={shape.batch} S={shape.seq} "
        f"n_h={shape.n_h} qk={shape.qk_head_dim} d_v={shape.d_v})"
    )
    print(f"   BLOCK_M={bm} BLOCK_N={bn} QK_PAD={qk_pad} grid={grid}")
    print(f"   allow_tma={allow_tma} latency={latency}")
    print(f"   median : {median_ms:.3f} ms   {median_tf:.3f} TFLOPS")
    print(f"   best   : {best_ms:.3f} ms   {best_tf:.3f} TFLOPS")
    print(f"   cuBLAS hgemm peak (Wave 14.1) : 218 TFLOPS  → ratio {best_tf/218:.2%}")
    print(f"   cutile-attn-gqa (Wave 15.1)   : 165 TFLOPS  → ratio {best_tf/165:.2%}")
    print("=" * 64)


# ─────────────────────────────────────────────────────────────────────────────
# Cubin export
# ─────────────────────────────────────────────────────────────────────────────

def _ac(dt):
    return ArrayConstraint(
        dtype=dt, ndim=2, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=(), may_alias_internally=False,
        stride_constant=(None, 1), stride_divisible_by=1,
        shape_divisible_by=1, base_addr_divisible_by=1,
    )


def export_cubin(out_path: str, shape: MLAShape,
                 allow_tma: bool = True, latency: int = 10) -> str | None:
    bm, bn = pick_blocks(shape)
    qk_pad = _pad_to_pow2_ge(shape.qk_head_dim)
    kernel = make_mla_kernel(
        bm, bn, qk_pad, shape.d_v, shape.seq, shape.n_h,
        qk_head_dim_true=shape.qk_head_dim,
        allow_tma=allow_tma, latency=latency,
    )
    sig = KernelSignature(
        parameters=[_ac(ct.float16), _ac(ct.float16), _ac(ct.float16), _ac(ct.float16)],
        calling_convention=CallingConvention.cutile_python_v1(),
    )
    try:
        export_kernel(kernel, [sig], out_path, gpu_code="sm_120", output_format="cubin")
        import os
        size = os.path.getsize(out_path)
        print(f"  wrote {out_path}  ({size} bytes)")
        return out_path
    except Exception as e:
        print(f"  FAILED cubin export: {type(e).__name__}: {str(e)[:400]}", file=sys.stderr)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", default=False)
    ap.add_argument("--bench", action="store_true", default=False)
    ap.add_argument("--export-cubin", action="store_true", default=False)
    ap.add_argument("--csv-out", default="results.csv")
    ap.add_argument("--cubin-out", default="mla_fwd_fused.cubin")
    # Wave C3.2 flags — explicit-TMA-mode toggle.
    ap.add_argument(
        "--no-tma", action="store_true", default=False,
        help="Pass allow_tma=False on every load/store (TMA disabled).",
    )
    ap.add_argument(
        "--latency", type=int, default=10,
        help="Latency hint 1..10 for tile loads/stores (10 = heavy DRAM).",
    )
    args = ap.parse_args()

    allow_tma = not args.no_tma
    latency = args.latency

    if not (args.smoke or args.bench or args.export_cubin):
        args.smoke = True

    print(f"cuda-tile version: {ct.__version__}")
    print(f"cupy: {cupy.__version__}")
    props = cupy.cuda.runtime.getDeviceProperties(0)
    print(f"device: {props['name'].decode()}")
    print(f"compute capability: sm_{props['major']}{props['minor']}")
    print(f"WAVE-C3.2 mode: allow_tma={allow_tma}  latency={latency}")
    print()

    rc = 0
    if args.smoke:
        ok = run_smoke(SHAPE_CORRECTNESS_MLA,
                       allow_tma=allow_tma, latency=latency)
        if not ok:
            rc = 1
        print()

    if args.bench:
        run_bench(SHAPE_BENCH_MLA, args.csv_out,
                  allow_tma=allow_tma, latency=latency)
        print()

    if args.export_cubin:
        print("Exporting cubin at bench shape for SASS inspection…")
        export_cubin(args.cubin_out, SHAPE_BENCH_MLA,
                     allow_tma=allow_tma, latency=latency)

    return rc


if __name__ == "__main__":
    sys.exit(main())
