#!/usr/bin/env bash
# Wave 18 Phase B.2: Mojo reduction.
# Reproducible: from repo root, `bash mojo-reduction/run.sh | tee mojo-reduction/run.log`.

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$REPO_ROOT/mojo-workspace"
SOURCE="$REPO_ROOT/mojo-reduction/reduction.mojo"

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap,temperature.gpu,power.draw \
    --format=csv

echo ""
echo "=== mojo --version ==="
( cd "$WORKSPACE" && pixi run mojo --version )

echo ""
echo "=== Run reduction (also dumps SASS to stdout) ==="
( cd "$WORKSPACE" && pixi run mojo "$SOURCE" )
