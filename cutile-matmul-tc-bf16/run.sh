#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/codeseys/cuda-exploration/cutile-vecadd-bench/.venv/bin/python

echo "=== smoke test (correctness, N=512) ==="
$PY main.py --smoke --n 512 2>&1 | tee smoke.log

echo
echo "=== bench (N=4096, 50 iters) ==="
$PY main.py --bench --n 4096 --csv-out results.csv 2>&1 | tee bench.log
