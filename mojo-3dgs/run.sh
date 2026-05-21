#!/usr/bin/env bash
# Wave 23.1 -- mojo-3dgs (5th frontend port of 3DGS rasterizer).
# Reproducible: from repo root, `bash mojo-3dgs/run.sh | tee mojo-3dgs/run.log`.
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$REPO_ROOT/mojo-workspace"
CELL="$REPO_ROOT/mojo-3dgs"

PY="$REPO_ROOT/cutile-vecadd-bench/.venv/bin/python"
PLY="$REPO_ROOT/oxide-3dgs-real/scenes/utsuho_plush.ply"

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv

echo ""
echo "=== mojo --version ==="
( cd "$WORKSPACE" && pixi run mojo --version )

echo ""
echo "=== prep.py: PLY -> binary blob (cam A) ==="
( cd "$CELL" && "$PY" prep.py --ply "$PLY" --out cam_A.bin )

echo ""
echo "=== rasterize.mojo: GPU rasterize cam A ==="
( cd "$CELL" && pixi run --manifest-path "$WORKSPACE/pixi.toml" mojo rasterize.mojo )

echo ""
echo "=== diff vs cuda-3dgs-real cam A ==="
( cd "$CELL" && "$PY" - <<'PY'
import numpy as np, sys
def read_ppm(p):
    with open(p,'rb') as f: d=f.read()
    assert d[:2]==b'P6'
    i=3
    def tok(j):
        while j<len(d) and d[j:j+1] in (b' ',b'\t',b'\n',b'\r'): j+=1
        if d[j:j+1]==b'#':
            while j<len(d) and d[j:j+1]!=b'\n': j+=1
            return tok(j)
        k=j
        while k<len(d) and d[k:k+1] not in (b' ',b'\t',b'\n',b'\r'): k+=1
        return d[j:k].decode(), k
    w,i=tok(i); h,i=tok(i); m,i=tok(i); i+=1
    w=int(w); h=int(h); assert int(m)==255
    return np.frombuffer(d[i:i+w*h*3],dtype=np.uint8).reshape(h,w,3)
ours=read_ppm('output_utsuho_plush_A.ppm')
ref=read_ppm('../cuda-3dgs-real/output_utsuho_plush_A.ppm')
assert ours.shape==ref.shape, f'shape mismatch {ours.shape} vs {ref.shape}'
diff=np.abs(ours.astype(np.int16)-ref.astype(np.int16)).astype(np.uint16)
n_pix=ours.shape[0]*ours.shape[1]
n_neq=int(diff.any(axis=2).sum())
n_gt2=int((diff.max(axis=2)>2).sum())
n_gt5=int((diff.max(axis=2)>5).sum())
print(f'  pixels         : {n_pix}')
print(f'  pixels w/ diff : {n_neq} ({100.0*n_neq/n_pix:.4f}%)')
print(f'  pixels w/ diff>2: {n_gt2} ({100.0*n_gt2/n_pix:.4f}%)')
print(f'  pixels w/ diff>5: {n_gt5} ({100.0*n_gt5/n_pix:.4f}%)')
print(f'  max u8 diff    : {int(diff.max())}')
print(f'  mean u8 diff   : {float(diff.mean()):.4f}')
print('  RESULT:', 'PASS (max diff <= 2)' if int(diff.max())<=2 else 'FAIL')
PY
)
