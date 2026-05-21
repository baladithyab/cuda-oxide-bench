"""Wave 22.15 — generate disk NPY inputs for the new sweep shapes.

Adds tiny, small, large, wide GDN-decode shapes to the inputs/ tree. Uses
the SAME input-construction recipe as pytorch_reference_gdn.py::make_inputs
+ gdn_decode_reference so all kernels (cuda-attn-gdn-tma, cutile-attn-gdn)
share inputs and oracle outputs verbatim.

The qwen3_next_decode and correctness shapes already exist on disk and are
NOT regenerated here.

Run:
    python3 gen_w22_15_inputs.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Reuse the exact reference implementation + input recipe.
sys.path.insert(0, str(Path(__file__).parent))
from pytorch_reference_gdn import make_inputs, gdn_decode_reference  # noqa: E402
from shapes_gdn import GDNShape  # noqa: E402

OUT_DIR = Path(__file__).parent.parent / "inputs"

# Wave 22.15 sweep shapes. Names match the cuTile/CUDA shape-registry below.
W22_15_SHAPES = [
    GDNShape(name="tiny",  batch=1, n_heads=4,  d_k=64,  d_v=64),    # 32 blocks (BV=16)
    GDNShape(name="small", batch=1, n_heads=8,  d_k=128, d_v=128),   # 32 blocks (BV=32)
    GDNShape(name="large", batch=4, n_heads=64, d_k=256, d_v=256),   # 1024 blocks (BV=64)
    GDNShape(name="wide",  batch=1, n_heads=16, d_k=512, d_v=512),   # 64 blocks (BV=128)
]


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    print(f"[gen] outputs → {OUT_DIR}")
    print(f"[gen] torch={torch.__version__}  cuda_avail={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[gen] device: {torch.cuda.get_device_name(0)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for shape in W22_15_SHAPES:
        shape.assert_valid()
        prefix = OUT_DIR / f"gdn_{shape.name}"
        # Skip if already generated (idempotent).
        if (Path(f"{prefix}_q_f16.npy").exists()
                and Path(f"{prefix}_o_expected_f16.npy").exists()):
            print(f"[gen] {shape.name}: already exists, skipping")
            continue

        # Use a per-shape seed derived from the name to keep regenerations
        # deterministic; do NOT collide with the qwen3 default 0xC0FFEE.
        seed = 0xBAD_F00D + (hash(shape.name) & 0xFFFF)
        inp = make_inputs(shape, seed=seed)

        q16, k16, v16 = inp["q_f16"], inp["k_f16"], inp["v_f16"]
        a16, b16 = inp["alpha_f16"], inp["beta_f16"]
        S_in = inp["S_in_f32"]

        q_d = q16.to(device); k_d = k16.to(device); v_d = v16.to(device)
        a_d = a16.to(device); b_d = b16.to(device); S_d = S_in.to(device)

        with torch.no_grad():
            o_fused, S_fused = gdn_decode_reference(q_d, k_d, v_d, a_d, b_d, S_d)
        o_fused = o_fused.cpu(); S_fused = S_fused.cpu()

        np.save(f"{prefix}_q_f16.npy", q16.numpy())
        np.save(f"{prefix}_k_f16.npy", k16.numpy())
        np.save(f"{prefix}_v_f16.npy", v16.numpy())
        np.save(f"{prefix}_alpha_f16.npy", a16.numpy())
        np.save(f"{prefix}_beta_f16.npy", b16.numpy())
        np.save(f"{prefix}_S_in_f32.npy", S_in.numpy())
        np.save(f"{prefix}_o_expected_f16.npy", o_fused.numpy())
        np.save(f"{prefix}_S_out_expected_f32.npy", S_fused.numpy())
        # Also write f32 for parity with the existing files.
        np.save(f"{prefix}_q_f32.npy", inp["q_f32"].numpy())
        np.save(f"{prefix}_k_f32.npy", inp["k_f32"].numpy())
        np.save(f"{prefix}_v_f32.npy", inp["v_f32"].numpy())
        np.save(f"{prefix}_alpha_f32.npy", inp["alpha_f32"].numpy())
        np.save(f"{prefix}_beta_f32.npy", inp["beta_f32"].numpy())
        state_mb = S_in.numel() * 4 / 1e6
        print(
            f"[gen] {shape.name}: B={shape.batch} H={shape.n_heads} "
            f"d_k={shape.d_k} d_v={shape.d_v}  state={state_mb:.2f} MB"
        )

    print("[gen] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
