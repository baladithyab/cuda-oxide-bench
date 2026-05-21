#!/usr/bin/env bash
# Wave 20 W1 -- mojo-mma-probe reproduce script
set -e
cd "$(dirname "$0")"

export PATH="$HOME/.pixi/bin:$PATH"
WORKSPACE=/home/codeseys/cuda-exploration/mojo-workspace

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap,temperature.gpu,power.draw --format=csv

echo ""
echo "=== mojo --version ==="
(cd "$WORKSPACE" && pixi run mojo --version)

echo ""
echo "=== Run mojo-mma-probe (with SASS dump) ==="
(cd "$WORKSPACE" && pixi run mojo "$(pwd)/mma_probe.mojo")
