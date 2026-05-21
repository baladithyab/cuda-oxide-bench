#!/bin/bash
# Wave C2.6 — cuda-attn-kda build + correctness + bench driver.
set -euo pipefail
cd "$(dirname "$0")"

NVCC=/usr/local/cuda/bin/nvcc
CUOBJ=/usr/local/cuda/bin/cuobjdump

echo "=== build attn_kda (correctness binary) ==="
make clean >/dev/null
make attn_kda 2>&1 | tee build.log

echo
echo "=== build bench ==="
make bench 2>&1 | tee -a build.log

echo
echo "=== correctness run ==="
./attn_kda 2>&1 | tee run.log

echo
echo "=== timed bench (50 iters, kimi_linear_decode + large) ==="
./bench 2>&1 | tee bench.log

echo
echo "=== SASS dump ==="
$CUOBJ --dump-sass attn_kda > attn_kda.sass 2>&1 || true
HMMA=$(grep -c "HMMA" attn_kda.sass || true)
FFMA=$(grep -c "FFMA" attn_kda.sass || true)
LDGE=$(grep -c "LDG.E"  attn_kda.sass || true)
LDG128=$(grep -c "LDG.E.128" attn_kda.sass || true)
LDG64=$(grep -c "LDG.E.64"  attn_kda.sass || true)
STG128=$(grep -c "STG.E.128" attn_kda.sass || true)
MUFU=$(grep -c "MUFU" attn_kda.sass || true)
TOTAL=$(wc -l < attn_kda.sass)
echo "SASS lines : $TOTAL"
echo "HMMA       : $HMMA      (expect 0; KDA is memory-bound, no TC)"
echo "FFMA       : $FFMA      (expect > 0)"
echo "LDG.E      : $LDGE"
echo "LDG.E.128  : $LDG128    (expect > 0; vectorized state reads)"
echo "LDG.E.64   : $LDG64"
echo "STG.E.128  : $STG128    (expect > 0; vectorized state writes)"
echo "MUFU       : $MUFU      (expect > 0; exp() for per-channel decay)"
