"""Wave C2.6 — PyTorch (numpy actually; torch optional) reference for KDA decode.

KDA = Kimi Delta Attention. Per-channel-gate variant of GDN. Single-timestep
recurrence (verbatim from cutile-attn-kda/main.py and docs/research/wave17-kda-spec.md):

    S_t = ( I − β_t · k_t · k_t^T ) · Diag(α_t) · S_{t−1}  +  β_t · k_t · v_t^T
    o_t = S_t^T · q_t        with α_t ∈ ℝ^{d_k}  (per-channel, decay = exp(g))

The only mathematical change vs GDN is line 1: the state is rescaled by the
per-channel decay vector exp(g) along the d_k axis instead of by a scalar α.

This reference computes the oracle in float64 and returns f16/f32 to match
what the GPU kernel will compute.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KDAShape:
    name: str
    batch: int
    n_heads: int
    d_k: int
    d_v: int


# Shape registry mirrors cutile-attn-kda/main.py (KDAShape registry)
SHAPE_CORRECTNESS = KDAShape("correctness", batch=1, n_heads=2, d_k=64,  d_v=64)
SHAPE_KIMI_LINEAR_DECODE = KDAShape(
    "kimi_linear_decode", batch=1, n_heads=32, d_k=128, d_v=128
)
SHAPE_QWEN3_NEXT_GDN_PARITY = KDAShape(
    "qwen3_next_gdn_parity", batch=1, n_heads=16, d_k=256, d_v=256
)
SHAPE_LARGE = KDAShape("large", batch=4, n_heads=64, d_k=256, d_v=256)


def naive_recurrent_kda_step(q, k, v, g, beta, S_in, scale=1.0):
    """One-step KDA recurrence in float64.

    Args (numpy arrays):
        q, k:   (B, H, d_k)        f32
        v:      (B, H, d_v)        f32
        g:      (B, H, d_k)        f32   — log-gate; decay = exp(g)
        beta:   (B, H)             f32
        S_in:   (B, H, d_k, d_v)   f32

    Returns:
        o:      (B, H, d_v)        f64
        S_out:  (B, H, d_k, d_v)   f64
    """
    q = q.astype(np.float64) * scale
    k = k.astype(np.float64)
    v = v.astype(np.float64)
    g = g.astype(np.float64)
    beta = beta.astype(np.float64)
    S = S_in.astype(np.float64)
    # Decay: S = exp(g)[..., None] * S
    S = S * np.exp(g)[..., None]
    # Residual u = (k * S).sum(K) → (B,H,V)
    u = (k[..., None] * S).sum(axis=-2)
    residual = v - u
    # Outer product β · k ⊗ residual → (B,H,K,V)
    S = S + (beta[..., None] * k)[..., None] * residual[..., None, :]
    # Output o = S^T q → (B,H,V)
    o = np.einsum("bhk,bhkv->bhv", q, S)
    return o, S


def make_inputs(shape: KDAShape, seed: int = 17):
    """Construct inputs for a KDA shape using the same recipe as
    cutile-attn-kda/main.py::load_inputs.

    Returns dict with f16/f32 tensors plus reference oracle outputs.
    """
    rng = np.random.default_rng(seed)
    B, H, d_k, d_v = shape.batch, shape.n_heads, shape.d_k, shape.d_v
    scale_k = 1.0 / np.sqrt(d_k)
    q_f32 = (rng.standard_normal((B, H, d_k)) * scale_k).astype(np.float32)
    k_f32 = (rng.standard_normal((B, H, d_k)) * scale_k).astype(np.float32)
    v_f32 = rng.standard_normal((B, H, d_v)).astype(np.float32)
    # Log-gate: small negative magnitudes → exp(g) ∈ (~0.6, 1].
    g_f32 = (-np.abs(rng.standard_normal((B, H, d_k))) * 0.5).astype(np.float32)
    beta_f32 = rng.uniform(0.1, 0.9, size=(B, H)).astype(np.float32)
    S_in_f32 = (rng.standard_normal((B, H, d_k, d_v)) * 0.1).astype(np.float32)

    q_f16 = q_f32.astype(np.float16)
    k_f16 = k_f32.astype(np.float16)
    v_f16 = v_f32.astype(np.float16)
    g_f16 = g_f32.astype(np.float16)
    beta_f16 = beta_f32.astype(np.float16)

    # Run the oracle on the *f16-roundtripped* inputs to match what the GPU sees.
    o_f64, S_out_f64 = naive_recurrent_kda_step(
        q_f16.astype(np.float32),
        k_f16.astype(np.float32),
        v_f16.astype(np.float32),
        g_f16.astype(np.float32),
        beta_f16.astype(np.float32),
        S_in_f32,
        scale=1.0,
    )
    return {
        "q_f16": q_f16, "k_f16": k_f16, "v_f16": v_f16,
        "g_f16": g_f16, "beta_f16": beta_f16,
        "q_f32": q_f32, "k_f32": k_f32, "v_f32": v_f32,
        "g_f32": g_f32, "beta_f32": beta_f32,
        "S_in_f32": S_in_f32,
        "o_expected_f16": o_f64.astype(np.float16),
        "S_out_expected_f32": S_out_f64.astype(np.float32),
    }
