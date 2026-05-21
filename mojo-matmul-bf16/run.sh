#!/usr/bin/env bash
# Wave 21 -- mojo-matmul-bf16 reproduce script
# Hand-rolled bf16-in/f32-acc tiled matmul via raw mma() (closes Wave 20 harness gap).
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
echo "=== Run mojo-matmul-bf16 ==="
(cd "$WORKSPACE" && pixi run mojo "$HERE/matmul_bf16.mojo")
