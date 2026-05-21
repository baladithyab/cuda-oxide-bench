#!/usr/bin/env bash
# Wave C2.1 — cuda-matmul-tc-bf16 driver.
# Build, run bench (50 iters), dump SASS, count HMMA.16816.F32.BF16 instructions.
set -e
set -o pipefail
cd "$(dirname "$0")"

echo "[run.sh] === build ==="
make 2>&1 | tee build.log

echo "[run.sh] === run ==="
./matmul_tc_bf16 2>&1 | tee run.log || true

echo "[run.sh] === SASS dump ==="
/usr/local/cuda/bin/cuobjdump --dump-sass matmul_tc_bf16 > matmul_tc_bf16.sass

echo "[run.sh] HMMA count:               $(grep -c HMMA matmul_tc_bf16.sass || true)"
echo "[run.sh] HMMA.16816.F32.BF16 count: $(grep -c 'HMMA.16816.F32.BF16' matmul_tc_bf16.sass || true)"
echo "[run.sh] FFMA count:               $(grep -c FFMA matmul_tc_bf16.sass || true)"
echo "[run.sh] LDGSTS count (cp.async):  $(grep -c LDGSTS matmul_tc_bf16.sass || true)"
