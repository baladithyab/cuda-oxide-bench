"""Host-side preprocessor for mojo-3dgs (Wave 23.1).

Parses utsuho_plush.ply, projects gaussians for cam A, evaluates SH3 to RGB,
depth-sorts ascending, and writes a flat binary blob the Mojo kernel reads.

Blob layout (little-endian, RTX 5090 host = LE so no swap):
    [uint32 n_proj]
    [n_proj * f32 mx]
    [n_proj * f32 my]
    [n_proj * f32 cxx]
    [n_proj * f32 cxy]
    [n_proj * f32 cyy]
    [n_proj * f32 opacity]
    [n_proj * f32 r]
    [n_proj * f32 g]
    [n_proj * f32 b]

This file mirrors cutile-3dgs-real/rasterize.py's host-side functions
(parse_ply, sh_eval_full, project_all, make_cameras) so the projected
blob is bit-equivalent to cutile's d_mx..d_cb. The Mojo kernel then has
no preprocessing responsibility — pure rasterization only.

Why this split (vs full PLY parsing in Mojo): the cutile reference already
proved correctness through this exact host-side pipeline; replicating
~400 LOC of PLY+SH3+projection in Mojo adds correctness risk without
informational value about Mojo's per-pixel iteration path, which is what
this Wave 23.1 cell is characterizing. If full-Mojo PLY/proj is desired,
that's a follow-up wave (W23.1c).
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

W = 800
H = 800

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
         -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
         0.3731763325901154, -0.4570457994644658, 1.445305721320277,
         -0.5900435899266435)


def parse_ply(path: str):
    with open(path, "rb") as fp:
        buf = fp.read()
    needle = b"end_header\n"
    idx = buf.find(needle)
    if idx < 0:
        raise RuntimeError("no end_header in PLY")
    header_end = idx + len(needle)
    header = buf[:header_end].decode("ascii", errors="replace")
    n_vertex = 0
    props: list[str] = []
    for line in header.splitlines():
        if line.startswith("element vertex "):
            n_vertex = int(line[len("element vertex "):].strip())
        elif line.startswith("property float "):
            props.append(line[len("property float "):].strip())
    nprops = len(props)
    print(f"PLY header: {n_vertex} vertices, {nprops} float props")

    body = buf[header_end:]
    expected = n_vertex * nprops * 4
    if len(body) != expected:
        raise RuntimeError(f"body size mismatch: got {len(body)} expected {expected}")

    arr = np.frombuffer(body, dtype="<f4").reshape(n_vertex, nprops)

    def col(name: str) -> np.ndarray:
        if name not in props:
            raise RuntimeError(f"property '{name}' not found")
        return arr[:, props.index(name)].astype(np.float32, copy=False)

    x = col("x"); y = col("y"); z = col("z")
    f_dc = np.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], axis=1)
    opacity_logit = col("opacity")
    scale = np.stack([col("scale_0"), col("scale_1"), col("scale_2")], axis=1)
    rot = np.stack([col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")], axis=1)

    have_rest = all(f"f_rest_{k}" in props for k in range(45))
    if have_rest:
        f_rest = np.stack([col(f"f_rest_{k}") for k in range(45)], axis=1)
        sh = "degree 3 (16 coefs/channel)"
    else:
        f_rest = None
        sh = "degree 0 only (DC)"
    print(f"SH support: {sh}")

    return {
        "x": x, "y": y, "z": z,
        "f_dc": f_dc, "f_rest": f_rest,
        "opacity": opacity_logit, "scale": scale, "rot": rot,
    }, sh


def sh_eval_full(f_dc: np.ndarray, f_rest: np.ndarray | None,
                 vd: np.ndarray) -> np.ndarray:
    n = f_dc.shape[0]
    out = np.empty((n, 3), dtype=np.float32)
    if f_rest is None:
        out[:, 0] = SH_C0 * f_dc[:, 0] + 0.5
        out[:, 1] = SH_C0 * f_dc[:, 1] + 0.5
        out[:, 2] = SH_C0 * f_dc[:, 2] + 0.5
        return out

    x = vd[:, 0]; y = vd[:, 1]; z = vd[:, 2]
    xx = x * x; yy = y * y; zz = z * z
    xy = x * y; yz = y * z; xz = x * z

    for ch in range(3):
        rest = f_rest[:, ch * 15:(ch + 1) * 15]
        dc = f_dc[:, ch]
        r = SH_C0 * dc
        r = r + SH_C1 * (-y * rest[:, 0] + z * rest[:, 1] - x * rest[:, 2])
        r = r + SH_C2[0] * xy * rest[:, 3] \
              + SH_C2[1] * yz * rest[:, 4] \
              + SH_C2[2] * (2.0 * zz - xx - yy) * rest[:, 5] \
              + SH_C2[3] * xz * rest[:, 6] \
              + SH_C2[4] * (xx - yy) * rest[:, 7]
        r = r + SH_C3[0] * y * (3.0 * xx - yy) * rest[:, 8] \
              + SH_C3[1] * xy * z * rest[:, 9] \
              + SH_C3[2] * y * (4.0 * zz - xx - yy) * rest[:, 10] \
              + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * rest[:, 11] \
              + SH_C3[4] * x * (4.0 * zz - xx - yy) * rest[:, 12] \
              + SH_C3[5] * z * (xx - yy) * rest[:, 13] \
              + SH_C3[6] * x * (xx - 3.0 * yy) * rest[:, 14]
        out[:, ch] = r + 0.5
    return out


def quat_to_mat3_batch(rot: np.ndarray) -> np.ndarray:
    n = rot.shape[0]
    norm = np.linalg.norm(rot, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    q = rot / norm
    w = q[:, 0]; x = q[:, 1]; y = q[:, 2]; z = q[:, 3]
    R = np.empty((n, 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - w * z)
    R[:, 0, 2] = 2.0 * (x * z + w * y)
    R[:, 1, 0] = 2.0 * (x * y + w * z)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - w * x)
    R[:, 2, 0] = 2.0 * (x * z - w * y)
    R[:, 2, 1] = 2.0 * (y * z + w * x)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return R


def make_cam_A(raws):
    x = raws["x"]; y = raws["y"]; z = raws["z"]
    cx = float(x.mean()); cy = float(y.mean()); cz = float(z.mean())
    extent = float(np.linalg.norm(
        [x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]))
    fx = 800.0; fy = 800.0; cx_p = 400.0; cy_p = 400.0
    dist = extent * 1.5
    I = np.eye(3, dtype=np.float32)
    return {"label": "camA_minusZ",
            "W": I.copy(), "t": np.array([-cx, -cy, -(cz - dist)], np.float32),
            "fx": fx, "fy": fy, "cx": cx_p, "cy": cy_p}


def project_all(raws: dict, cam: dict):
    x = raws["x"]; y = raws["y"]; z = raws["z"]
    n = x.size
    pos = np.stack([x, y, z], axis=1).astype(np.float32)

    W_mat = cam["W"].astype(np.float32)
    t = cam["t"].astype(np.float32)
    fx = cam["fx"]; fy = cam["fy"]; cx = cam["cx"]; cy = cam["cy"]

    pc = pos @ W_mat.T + t[None, :]
    z_cam = pc[:, 2]
    valid = (z_cam >= 0.1) & (z_cam <= 100.0)

    cam_origin = -W_mat.T @ t

    mx = fx * pc[:, 0] / np.maximum(z_cam, 1e-12) + cx
    my = fy * pc[:, 1] / np.maximum(z_cam, 1e-12) + cy

    R = quat_to_mat3_batch(raws["rot"])
    s_exp = np.exp(raws["scale"]).astype(np.float32)
    s2 = np.zeros((n, 3, 3), dtype=np.float32)
    s2[:, 0, 0] = s_exp[:, 0] ** 2
    s2[:, 1, 1] = s_exp[:, 1] ** 2
    s2[:, 2, 2] = s_exp[:, 2] ** 2
    sigma_w = R @ s2 @ R.transpose(0, 2, 1)
    sigma_cam = W_mat @ sigma_w @ W_mat.T

    z_safe = np.maximum(z_cam, 1e-12)
    z2 = z_safe * z_safe
    j00 = fx / z_safe
    j02 = -fx * pc[:, 0] / z2
    j11 = fy / z_safe
    j12 = -fy * pc[:, 1] / z2

    r0 = sigma_cam[:, 0, :]
    r1 = sigma_cam[:, 1, :]
    r2 = sigma_cam[:, 2, :]
    m0 = j00[:, None] * r0 + j02[:, None] * r2
    m1 = j11[:, None] * r1 + j12[:, None] * r2
    a = m0[:, 0] * j00 + m0[:, 2] * j02
    b = m0[:, 1] * j11 + m0[:, 2] * j12
    c = m1[:, 1] * j11 + m1[:, 2] * j12

    a_aa = a + 0.3
    c_aa = c + 0.3
    b_aa = b
    det = a_aa * c_aa - b_aa * b_aa
    valid &= (det > 0.0) & np.isfinite(det)

    inv_det = np.where(valid, 1.0 / np.where(det == 0, 1.0, det), 0.0)
    cxx = c_aa * inv_det
    cxy_ = -b_aa * inv_det
    cyy = a_aa * inv_det

    vd = pos - cam_origin[None, :]
    vdn = np.linalg.norm(vd, axis=1, keepdims=True)
    vdn = np.maximum(vdn, 1e-8)
    vd = vd / vdn
    rgb = sh_eval_full(raws["f_dc"], raws["f_rest"], vd)
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    op = 1.0 / (1.0 + np.exp(-raws["opacity"]))
    op = op.astype(np.float32)

    keep = np.where(valid)[0]
    n_proj = keep.size
    if n_proj == 0:
        return None, 0

    mx_k = mx[keep].astype(np.float32)
    my_k = my[keep].astype(np.float32)
    cxx_k = cxx[keep].astype(np.float32)
    cxy_k = cxy_[keep].astype(np.float32)
    cyy_k = cyy[keep].astype(np.float32)
    op_k = op[keep]
    cr_k = rgb[keep, 0]
    cg_k = rgb[keep, 1]
    cb_k = rgb[keep, 2]
    depth_k = z_cam[keep].astype(np.float32)

    order = np.argsort(depth_k, kind="stable")
    return {
        "mx": np.ascontiguousarray(mx_k[order]),
        "my": np.ascontiguousarray(my_k[order]),
        "cxx": np.ascontiguousarray(cxx_k[order]),
        "cxy": np.ascontiguousarray(cxy_k[order]),
        "cyy": np.ascontiguousarray(cyy_k[order]),
        "opacity": np.ascontiguousarray(op_k[order]),
        "r": np.ascontiguousarray(cr_k[order]),
        "g": np.ascontiguousarray(cg_k[order]),
        "b": np.ascontiguousarray(cb_k[order]),
    }, n_proj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply",
                    default="../oxide-3dgs-real/scenes/utsuho_plush.ply")
    ap.add_argument("--out", default="cam_A.bin")
    args = ap.parse_args()

    ply = str(Path(args.ply).resolve())
    print(f"Loading {ply}")
    raws, sh_status = parse_ply(ply)
    print(f"Parsed {raws['x'].size} gaussians; {sh_status}")

    cam = make_cam_A(raws)
    proj, n = project_all(raws, cam)
    if proj is None or n == 0:
        print("No projected gaussians")
        sys.exit(1)
    print(f"Projected {n} gaussians for cam A")

    out = Path(args.out)
    with open(out, "wb") as f:
        f.write(struct.pack("<I", n))
        for k in ["mx", "my", "cxx", "cxy", "cyy", "opacity", "r", "g", "b"]:
            f.write(proj[k].astype("<f4", copy=False).tobytes())
    print(f"Wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
