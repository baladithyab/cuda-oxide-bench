"""Wave 15.1 — cuTile fused GQA attention (FlashAttention-2 style).

Single @ct.kernel implementing GQA forward attention. For each (batch, head_q,
query_block) grid cell, loads a Q-tile, iterates over K/V blocks of the
corresponding KV head, and accumulates the output with online softmax in
registers. No materialization of the (S, S) attention matrix; no intermediate
HBM traffic between QK^T, softmax, and PV.

Uses `ct.mma` at f16 × f16 → f32 accumulator (Wave 13.1 finding: this is the
TC-engagement sweet spot for cuTile 1.3.0; f32×f32 falls back to CUDA cores).

Layout trick for GQA: Q/K/V are passed as 2D arrays flattened over
(batch, head, seq) — so Q is (B*n_q*S, D), K is (B*n_kv*S, D), V same.
The kernel's (bid0, bid1) pair is (batch*n_q + head_q_flat, q_block).
Inside, h_kv_flat = bid0 // groups provides the correct KV-head base row.
This avoids dynamic slicing and keeps tiled_view at a static 2D shape.

CLI:
    --smoke   correctness at SHAPE_CORRECTNESS vs inputs/gqa_correctness_expected_f32.npy
    --bench   timed at SHAPE_BENCH (Llama-3-8B), 1 warmup + 10 timed iters
    --csv-out FILE   bench CSV path (default results.csv)
    --export-cubin   write cubin for SASS inspection

Pitfalls (carried from cutile-matmul-tiled-mixed and cutile-vecadd-bench):
  - ct.launch(stream.ptr, grid, kernel, args) — NOT kernel[grid](args).
  - Use Python-closure factory for tile constants; ct.Constant[int] BROKEN for tile shapes.
  - ct.mma(a, b, acc) returns new tile; re-assign acc = ct.mma(...).
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

# Hook the wave-15 shared infra (shapes, flops, tolerances).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis" / "wave15-attention-architecture" / "reference"))
from shapes import GQAShape, SHAPE_CORRECTNESS, SHAPE_BENCH  # noqa: E402
from flops import gqa_attention_flops  # noqa: E402
from tolerances import get as get_tol  # noqa: E402

INPUTS_DIR = REPO_ROOT / "analysis" / "wave15-attention-architecture" / "inputs"

WARMUP = 2
ITERS = 10


# ─────────────────────────────────────────────────────────────────────────────
# Kernel factory — all tile shape constants captured as Python ints via closure
# ─────────────────────────────────────────────────────────────────────────────

def make_gqa_kernel(BLOCK_M: int, BLOCK_N: int, D_HEAD: int, SEQ: int,
                    N_Q: int, N_KV: int):
    """Build a @ct.kernel specialized to these shape constants.

    All shape constants must be Python ints at decoration time — cuTile 1.3.0's
    ct.Constant[int] launch-arg path does NOT propagate to tiled_view shapes
    (confirmed Wave 13.1 pitfall).

    Grid layout:
        bid0 ∈ [0, B*N_Q)       — flattened (batch, head_q)
        bid1 ∈ [0, SEQ//BLOCK_M) — query block index

    Array layouts (passed in 2D, flattened over batch*head*seq):
        Q : (B * N_Q  * SEQ, D_HEAD)    f16
        K : (B * N_KV * SEQ, D_HEAD)    f16
        V : (B * N_KV * SEQ, D_HEAD)    f16
        O : (B * N_Q  * SEQ, D_HEAD)    f16  (output)
    """
    groups = N_Q // N_KV
    SEQ_TILES_M = SEQ // BLOCK_M
    SEQ_TILES_N = SEQ // BLOCK_N
    scale = 1.0 / math.sqrt(D_HEAD)
    # Use -1e30 as "negative infinity" for f32 row-max init; safe under exp().
    NEG_INF = -1.0e30

    @ct.kernel
    def gqa_fwd(Q, K, V, O):
        # Grid axes. bid0 = flattened (batch, head_q). bid1 = query block.
        bid0 = ct.bid(0)
        bid1 = ct.bid(1)

        # Compute the row-base within each flattened 2D array.
        # Q's row block row = bid0 * SEQ_TILES_M + bid1
        #   (because each flattened head occupies SEQ_TILES_M consecutive
        #   BLOCK_M-sized row-tiles in the tiled_view.)
        q_tile_row = bid0 * SEQ_TILES_M + bid1

        # h_kv_flat = bid0 // groups. Base tile row in K/V's tiled_view.
        h_kv_flat = bid0 // groups
        kv_tile_row_base = h_kv_flat * SEQ_TILES_N

        # Tiled views of each 2D array.
        q_view = Q.tiled_view((BLOCK_M, D_HEAD), padding_mode=ct.PaddingMode.ZERO)
        k_view = K.tiled_view((BLOCK_N, D_HEAD), padding_mode=ct.PaddingMode.ZERO)
        v_view = V.tiled_view((BLOCK_N, D_HEAD), padding_mode=ct.PaddingMode.ZERO)
        o_view = O.tiled_view((BLOCK_M, D_HEAD), padding_mode=ct.PaddingMode.ZERO)

        # Load the Q tile once (persists across the K/V loop).
        q_tile = q_view.load((q_tile_row, 0))  # (BLOCK_M, D_HEAD) f16

        # Online-softmax running state (all f32, in registers).
        # m_i : row-wise running max, shape (BLOCK_M,) kept as (BLOCK_M, 1) for broadcast
        # l_i : row-wise running sum-of-exp (post-rescale)
        # o_acc : output accumulator
        m_i = ct.full((BLOCK_M, 1), NEG_INF, ct.float32)
        l_i = ct.zeros((BLOCK_M, 1), ct.float32)
        o_acc = ct.zeros((BLOCK_M, D_HEAD), ct.float32)

        # Iterate over K/V blocks.
        for kb in range(SEQ_TILES_N):
            k_tile = k_view.load((kv_tile_row_base + kb, 0))  # (BLOCK_N, D_HEAD) f16
            v_tile = v_view.load((kv_tile_row_base + kb, 0))  # (BLOCK_N, D_HEAD) f16

            # QK^T: (BLOCK_M, D_HEAD) × (D_HEAD, BLOCK_N) → (BLOCK_M, BLOCK_N) f32
            # Need K transposed — build via ct.transpose of the 2D f16 tile.
            k_t = ct.transpose(k_tile)  # (D_HEAD, BLOCK_N) f16
            s_acc = ct.zeros((BLOCK_M, BLOCK_N), ct.float32)
            s_acc = ct.mma(q_tile, k_t, s_acc)

            # Scale.
            s_scaled = s_acc * scale

            # Row-wise max over axis=1 with keepdims → (BLOCK_M, 1).
            m_row = ct.max(s_scaled, axis=1, keepdims=True)
            m_new = ct.maximum(m_i, m_row)  # (BLOCK_M, 1)

            # Rescale factor for previous accumulators.
            alpha = ct.exp(m_i - m_new)  # (BLOCK_M, 1)

            # P = exp(S - m_new) — broadcast m_new across BLOCK_N.
            p = ct.exp(s_scaled - m_new)  # (BLOCK_M, BLOCK_N) f32

            # Row sum of P.
            p_row_sum = ct.sum(p, axis=1, keepdims=True)  # (BLOCK_M, 1) f32

            # Update l_i: alpha * l_i + p_row_sum.
            l_i = alpha * l_i + p_row_sum

            # Rescale o_acc by alpha broadcast across D_HEAD, then accumulate P @ V.
            o_acc = o_acc * alpha  # broadcast (BLOCK_M, 1) over (BLOCK_M, D_HEAD)

            # Cast P to f16 for the mma (ct.mma wants matching input dtypes).
            p_f16 = p.astype(ct.float16)
            o_acc = ct.mma(p_f16, v_tile, o_acc)

            m_i = m_new

        # Final divide by l_i.
        o_final = o_acc / l_i  # broadcast (BLOCK_M, 1) over (BLOCK_M, D_HEAD)

        # Store (cast to output dtype; Q/K/V/O are all f16 in the canonical path).
        # NB: ct.store takes the bare Array with tile-space index; alternately
        # o_view.store((row, 0), tile) works. Using the tiled_view keeps the
        # tile shape explicit at the call site.
        o_view.store((q_tile_row, 0), o_final.astype(O.dtype))

    return gqa_fwd


# ─────────────────────────────────────────────────────────────────────────────
# Input I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_inputs(shape: GQAShape):
    prefix = INPUTS_DIR / f"gqa_{shape.name}"
    q_np = np.load(f"{prefix}_q_f16.npy")
    k_np = np.load(f"{prefix}_k_f16.npy")
    v_np = np.load(f"{prefix}_v_f16.npy")
    exp_np = np.load(f"{prefix}_expected_f32.npy")
    return q_np, k_np, v_np, exp_np


def prepare_2d(q_np: np.ndarray, k_np: np.ndarray, v_np: np.ndarray,
               shape: GQAShape):
    """Reshape (B, H, S, D) → (B*H*S, D) contiguous and move to device."""
    B, S, D = shape.batch, shape.seq, shape.d_head
    q_2d = q_np.reshape(B * shape.n_q * S, D)
    k_2d = k_np.reshape(B * shape.n_kv * S, D)
    v_2d = v_np.reshape(B * shape.n_kv * S, D)
    return (cupy.asarray(q_2d, dtype=cupy.float16),
            cupy.asarray(k_2d, dtype=cupy.float16),
            cupy.asarray(v_2d, dtype=cupy.float16))


# ─────────────────────────────────────────────────────────────────────────────
# Tile-size picker
# ─────────────────────────────────────────────────────────────────────────────

def pick_blocks(shape: GQAShape) -> tuple[int, int]:
    """Pick (BLOCK_M, BLOCK_N) that divide shape.seq evenly.

    For SHAPE_BENCH (seq=2048), 64×64 is the default — 32 Q-blocks, 32 K-blocks.
    For SHAPE_CORRECTNESS (seq=128), use 32×32 — 4 Q-blocks, 4 K-blocks.
    """
    if shape.seq >= 512:
        return 64, 64
    # small shape
    # want block dims that divide seq and give at least a few blocks.
    bm = min(32, shape.seq)
    bn = min(32, shape.seq)
    assert shape.seq % bm == 0 and shape.seq % bn == 0
    return bm, bn


# ─────────────────────────────────────────────────────────────────────────────
# Correctness
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke(shape: GQAShape) -> bool:
    print(f"[smoke] shape={shape.name} B={shape.batch} S={shape.seq} "
          f"n_q={shape.n_q} n_kv={shape.n_kv} d={shape.d_head}")
    q_np, k_np, v_np, expected = load_inputs(shape)
    q_d, k_d, v_d = prepare_2d(q_np, k_np, v_np, shape)
    o_d = cupy.zeros_like(q_d)

    bm, bn = pick_blocks(shape)
    print(f"[smoke] BLOCK_M={bm} BLOCK_N={bn}")
    kernel = make_gqa_kernel(bm, bn, shape.d_head, shape.seq, shape.n_q, shape.n_kv)

    grid = (shape.batch * shape.n_q, shape.seq // bm)
    stream = cupy.cuda.get_current_stream()
    ct.launch(stream.ptr, grid, kernel, (q_d, k_d, v_d, o_d))
    cupy.cuda.runtime.deviceSynchronize()

    # Reshape output back to (B, n_q, S, D) and compare in f32.
    o_np = o_d.get().reshape(shape.batch, shape.n_q, shape.seq, shape.d_head)
    o_f32 = o_np.astype(np.float32)

    tol = get_tol("f16")
    abs_err = np.abs(o_f32 - expected)
    ref_mag = np.abs(expected).max() + 1e-30
    max_abs = float(abs_err.max())
    rel_err = max_abs / ref_mag

    # Use numpy allclose with the shared tolerance table.
    ok = np.allclose(o_f32, expected, atol=tol.atol, rtol=tol.rtol)
    status = "OK" if ok else "FAIL"
    print(f"[smoke] max_abs={max_abs:.3e} rel={rel_err:.3e}  "
          f"atol={tol.atol} rtol={tol.rtol}  {status}")
    if not ok:
        # Report where the worst offenders are.
        worst = np.unravel_index(abs_err.argmax(), abs_err.shape)
        print(f"        worst offender at {worst}: "
              f"got={o_f32[worst]:.4f} expected={expected[worst]:.4f}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Bench
# ─────────────────────────────────────────────────────────────────────────────

def run_bench(shape: GQAShape, csv_path: str) -> None:
    print(f"[bench] shape={shape.name} B={shape.batch} S={shape.seq} "
          f"n_q={shape.n_q} n_kv={shape.n_kv} d={shape.d_head}")
    q_np, k_np, v_np, _expected = load_inputs(shape)
    q_d, k_d, v_d = prepare_2d(q_np, k_np, v_np, shape)
    o_d = cupy.zeros_like(q_d)

    bm, bn = pick_blocks(shape)
    print(f"[bench] BLOCK_M={bm} BLOCK_N={bn}")
    kernel = make_gqa_kernel(bm, bn, shape.d_head, shape.seq, shape.n_q, shape.n_kv)

    grid = (shape.batch * shape.n_q, shape.seq // bm)
    stream = cupy.cuda.get_current_stream()

    flops = gqa_attention_flops(shape)

    # Warmup (drops JIT compile + first-launch cost).
    for _ in range(WARMUP):
        ct.launch(stream.ptr, grid, kernel, (q_d, k_d, v_d, o_d))
    cupy.cuda.runtime.deviceSynchronize()

    # Timed iters — tight block.
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
        w.writerow(["impl", "kernel", "batch", "seq", "n_q", "n_kv",
                    "d_head", "block_m", "block_n", "iter", "gpu_ms", "tflops"])
        for i in range(ITERS):
            gpu_ms = cupy.cuda.get_elapsed_time(starts[i], ends[i])
            tflops = flops / (gpu_ms * 1e-3) / 1e12
            print(f"[bench] iter={i} gpu_ms={gpu_ms:.3f} tflops={tflops:.3f}")
            w.writerow(["cutile", "gqa_fwd_fused", shape.batch, shape.seq,
                        shape.n_q, shape.n_kv, shape.d_head, bm, bn, i,
                        f"{gpu_ms:.6f}", f"{tflops:.6f}"])
            rows.append((gpu_ms, tflops))

    ms_sorted = sorted(r[0] for r in rows)
    tf_sorted = sorted(r[1] for r in rows)
    median_ms = ms_sorted[ITERS // 2]
    median_tf = tf_sorted[ITERS // 2]
    best_ms = ms_sorted[0]
    best_tf = tf_sorted[-1]
    print()
    print("=" * 64)
    print(f" BENCH SUMMARY — {shape.name} (B={shape.batch} S={shape.seq} "
          f"n_q={shape.n_q} n_kv={shape.n_kv} d={shape.d_head})")
    print(f"   BLOCK_M={bm} BLOCK_N={bn} grid={grid}")
    print(f"   median : {median_ms:.3f} ms   {median_tf:.3f} TFLOPS")
    print(f"   best   : {best_ms:.3f} ms   {best_tf:.3f} TFLOPS")
    print(f"   cuBLAS hgemm peak (Wave 13): 218 TFLOPS  → ratio {best_tf/218:.2%}")
    print("=" * 64)


# ─────────────────────────────────────────────────────────────────────────────
# Cubin export (for SASS inspection — HMMA count)
# ─────────────────────────────────────────────────────────────────────────────

def _ac(dt):
    return ArrayConstraint(
        dtype=dt, ndim=2, index_dtype=ct.int32,
        stride_lower_bound_incl=0, alias_groups=(), may_alias_internally=False,
        stride_constant=(None, 1), stride_divisible_by=1,
        shape_divisible_by=1, base_addr_divisible_by=1,
    )


def export_cubin(out_path: str, shape: GQAShape) -> str | None:
    bm, bn = pick_blocks(shape)
    kernel = make_gqa_kernel(bm, bn, shape.d_head, shape.seq, shape.n_q, shape.n_kv)
    sig = KernelSignature(
        parameters=[_ac(ct.float16), _ac(ct.float16), _ac(ct.float16), _ac(ct.float16)],
        calling_convention=CallingConvention.cutile_python_v1(),
    )
    try:
        export_kernel(kernel, [sig], out_path,
                      gpu_code="sm_120", output_format="cubin")
        import os
        size = os.path.getsize(out_path)
        print(f"  wrote {out_path}  ({size} bytes)")
        return out_path
    except Exception as e:
        print(f"  FAILED cubin export: {type(e).__name__}: {str(e)[:400]}",
              file=sys.stderr)
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
    ap.add_argument("--cubin-out", default="gqa_fwd_fused.cubin")
    args = ap.parse_args()

    # Default: --smoke
    if not (args.smoke or args.bench or args.export_cubin):
        args.smoke = True

    print(f"cuda-tile version: {ct.__version__}")
    print(f"cupy: {cupy.__version__}")
    props = cupy.cuda.runtime.getDeviceProperties(0)
    print(f"device: {props['name'].decode()}")
    print(f"compute capability: sm_{props['major']}{props['minor']}")
    print()

    rc = 0
    if args.smoke:
        ok = run_smoke(SHAPE_CORRECTNESS)
        if not ok:
            rc = 1
        print()

    if args.bench:
        # Run smoke first as sanity, but at bench shape that's slow;
        # trust --smoke already ran if we got here from run.sh.
        run_bench(SHAPE_BENCH, args.csv_out)
        print()

    if args.export_cubin:
        print("Exporting cubin at bench shape for SASS inspection…")
        export_cubin(args.cubin_out, SHAPE_BENCH)

    return rc


if __name__ == "__main__":
    sys.exit(main())
