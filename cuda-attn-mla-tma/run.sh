#!/usr/bin/env bash
# Wave 23.2 — cuda-attn-mla-tma driver script.
#
# Per W23.2 task discipline: author + correctness only. NO timed bench.
#
# Per AGENTS.md run-discipline, ALWAYS pipe through `tee run.log`.

set -e
set -o pipefail

cd "$(dirname "$0")"

echo "[run.sh] === build ==="
make 2>&1 | tee build.log

echo "[run.sh] === correctness ==="
./attn_mla_tma 2>&1 | tee run.log

echo "[run.sh] === SASS dump (UTMALDG / HMMA / LDG sanity) ==="
make sass 2>&1 | tee sass.log
