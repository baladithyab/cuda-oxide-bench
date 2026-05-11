#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/codeseys/cuda-exploration/cutile-vecadd-bench/.venv/bin/python

echo "=== smoke test (correctness shape) ==="
$PY main.py --smoke

echo
echo "=== bench (Llama-3-8B shape) ==="
$PY main.py --bench --csv-out results.csv

echo
echo "=== cubin export for SASS inspection ==="
$PY main.py --export-cubin --cubin-out gqa_fwd_fused.cubin

if [ -x /usr/local/cuda/bin/cuobjdump ]; then
  CUOBJ=/usr/local/cuda/bin/cuobjdump
elif command -v cuobjdump >/dev/null 2>&1; then
  CUOBJ=cuobjdump
else
  echo "cuobjdump not found; skipping SASS dump"
  CUOBJ=""
fi
if [ -n "$CUOBJ" ]; then
  $CUOBJ --dump-sass gqa_fwd_fused.cubin > gqa_fwd_fused.sass 2>&1 || true
  HMMA=$(grep -c "HMMA" gqa_fwd_fused.sass || true)
  MUFU=$(grep -c "MUFU.EX2" gqa_fwd_fused.sass || true)
  echo "SASS lines : $(wc -l < gqa_fwd_fused.sass)"
  echo "HMMA insts : ${HMMA}"
  echo "MUFU.EX2   : ${MUFU} (softmax exp)"
fi
