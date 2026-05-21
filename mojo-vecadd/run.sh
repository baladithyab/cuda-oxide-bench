#!/usr/bin/env bash
# Wave 18 Phase A smoke test.
# Reproducible: from repo root, `bash mojo-vecadd/run.sh | tee mojo-vecadd/run.log`.

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$REPO_ROOT/mojo-workspace"
SOURCE="$REPO_ROOT/mojo-vecadd/vecadd.mojo"

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv

echo ""
echo "=== mojo --version ==="
( cd "$WORKSPACE" && pixi run mojo --version )

echo ""
echo "=== pixi info ==="
( cd "$WORKSPACE" && pixi info 2>&1 | head -20 )

echo ""
echo "=== Run vecadd ==="
( cd "$WORKSPACE" && pixi run mojo "$SOURCE" )
