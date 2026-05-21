"""Wave C2.6 — generate disk NPY inputs for KDA decode shapes.

Mirrors gen_w22_15_inputs.py (GDN). Writes the same set of tensors (q, k, v,
g [the per-channel log-gate], beta, S_in, plus oracle o_expected/S_out_expected)
to inputs/ for every KDA shape used by cuda-attn-kda and cutile-attn-kda.

Idempotent: skips shapes that already have the full file set.

Run:
    python3 gen_kda_inputs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pytorch_reference_kda import (  # noqa: E402
    KDAShape, make_inputs,
    SHAPE_CORRECTNESS, SHAPE_KIMI_LINEAR_DECODE,
    SHAPE_QWEN3_NEXT_GDN_PARITY, SHAPE_LARGE,
)

OUT_DIR = Path(__file__).parent.parent / "inputs"

KDA_SHAPES = [
    SHAPE_CORRECTNESS,
    SHAPE_KIMI_LINEAR_DECODE,
    SHAPE_QWEN3_NEXT_GDN_PARITY,
    SHAPE_LARGE,
]


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    print(f"[gen-kda] outputs → {OUT_DIR}")
    for shape in KDA_SHAPES:
        prefix = OUT_DIR / f"kda_{shape.name}"
        if (Path(f"{prefix}_q_f16.npy").exists()
                and Path(f"{prefix}_o_expected_f16.npy").exists()
                and Path(f"{prefix}_g_f16.npy").exists()):
            print(f"[gen-kda] {shape.name}: already exists, skipping")
            continue
        # Per-shape seed (mirrors gen_w22_15_inputs convention).
        seed = 0xCAFE + (hash(shape.name) & 0xFFFF)
        inp = make_inputs(shape, seed=seed)

        np.save(f"{prefix}_q_f16.npy",         inp["q_f16"])
        np.save(f"{prefix}_k_f16.npy",         inp["k_f16"])
        np.save(f"{prefix}_v_f16.npy",         inp["v_f16"])
        np.save(f"{prefix}_g_f16.npy",         inp["g_f16"])
        np.save(f"{prefix}_beta_f16.npy",      inp["beta_f16"])
        np.save(f"{prefix}_q_f32.npy",         inp["q_f32"])
        np.save(f"{prefix}_k_f32.npy",         inp["k_f32"])
        np.save(f"{prefix}_v_f32.npy",         inp["v_f32"])
        np.save(f"{prefix}_g_f32.npy",         inp["g_f32"])
        np.save(f"{prefix}_beta_f32.npy",      inp["beta_f32"])
        np.save(f"{prefix}_S_in_f32.npy",      inp["S_in_f32"])
        np.save(f"{prefix}_o_expected_f16.npy",   inp["o_expected_f16"])
        np.save(f"{prefix}_S_out_expected_f32.npy", inp["S_out_expected_f32"])
        state_mb = inp["S_in_f32"].nbytes / 1e6
        print(
            f"[gen-kda] {shape.name}: B={shape.batch} H={shape.n_heads} "
            f"d_k={shape.d_k} d_v={shape.d_v}  state={state_mb:.2f} MB  seed=0x{seed:x}"
        )
    print("[gen-kda] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
