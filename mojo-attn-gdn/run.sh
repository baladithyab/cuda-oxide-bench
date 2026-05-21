#!/usr/bin/env bash
# Wave C2.3 -- mojo-attn-gdn reproduce script.
# 5th frontend port of GDN. FFMA-class baseline (no TMA per W22.1).
# Mirrors W1c CUDA C++ FFMA-LDG.E.128 pattern.
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
echo "=== Run mojo-attn-gdn (qwen3_next_decode shape, 50-iter timed bench + sampled correctness) ==="
(cd "$WORKSPACE" && pixi run mojo "$HERE/attn_gdn.mojo")
