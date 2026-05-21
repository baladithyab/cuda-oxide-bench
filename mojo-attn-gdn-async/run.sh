#!/usr/bin/env bash
# Wave C3.4 -- mojo-attn-gdn-async reproduce script.
# Tests if Mojo's `copy_dram_to_sram_async` helps or hurts GDN
# (cuda::pipeline regressed -25% in C++; W22.9).
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
echo "=== Run mojo-attn-gdn-async (qwen3_next_decode shape, 50-iter timed bench + sampled correctness) ==="
(cd "$WORKSPACE" && pixi run mojo "$HERE/attn_gdn_async.mojo")
