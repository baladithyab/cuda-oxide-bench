#!/usr/bin/env bash
# Wave 18 Phase B.1: mojo-vecadd-bench.
# Reproducible: from repo root, `bash mojo-vecadd-bench/run.sh | tee mojo-vecadd-bench/run.log`.

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$REPO_ROOT/mojo-workspace"
SOURCE="$REPO_ROOT/mojo-vecadd-bench/vecadd_bench.mojo"

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap,temperature.gpu,power.draw \
    --format=csv

echo ""
echo "=== mojo --version ==="
( cd "$WORKSPACE" && pixi run mojo --version )

echo ""
echo "=== Run vecadd-bench ==="
( cd "$WORKSPACE" && pixi run mojo "$SOURCE" )
