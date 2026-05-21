#!/usr/bin/env bash
# Wave 22.5 -- mojo-attn-bf16 reproduce script
# 3-kernel attention (Q@K^T + softmax + P@V) with bf16 matmul stages.
# Builds on Wave 21 mojo-matmul-bf16's hand-rolled bf16-in/f32-acc pattern.
set -e
cd "$(dirname "$0")"
HERE="$(pwd)"

export PATH="$HOME/.pixi/bin:$PATH"
WORKSPACE=/home/codeseys/cuda-exploration/mojo-workspace

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap,temperature.gpu,power.draw --format=csv

echo ""
echo "=== mojo --version ==="
(cd "$WORKSPACE" && pixi run mojo --version)

echo ""
echo "=== Run mojo-attn-bf16 (correctness at small shape) ==="
(cd "$WORKSPACE" && pixi run mojo "$HERE/attn_bf16.mojo")
