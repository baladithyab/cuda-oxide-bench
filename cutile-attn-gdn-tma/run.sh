#!/bin/bash
# Wave C2.5 — explicit-TMA-mode test of cuTile DSL on GDN.
#
# Three legs:
#   (A) explicit allow_tma=True + latency=10  (the "TMA-on" config)
#   (B) explicit allow_tma=False              (the "TMA-off" falsification leg)
#   (C) explicit allow_tma=True on the larger `large` shape
#
# For each: smoke, bench, cubin, SASS-grep for UTMALDG / UTMASTG counts.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/codeseys/cuda-exploration/cutile-vecadd-bench/.venv/bin/python

if [ -x /usr/local/cuda/bin/cuobjdump ]; then
  CUOBJ=/usr/local/cuda/bin/cuobjdump
elif command -v cuobjdump >/dev/null 2>&1; then
  CUOBJ=cuobjdump
else
  echo "cuobjdump not found; SASS dump will be skipped" >&2
  CUOBJ=""
fi

run_leg() {
  local label="$1"
  local extra_flags="$2"
  local shape="$3"
  local cubin="cubin_${label}.cubin"
  local sass="sass_${label}.sass"
  local csv="results_${label}.csv"
  local log="bench_${label}.log"

  echo
  echo "================================================================"
  echo " LEG: ${label}  flags=[${extra_flags}]  shape=${shape}"
  echo "================================================================"

  # smoke (always uses the SHAPE_CORRECTNESS shape, sanity).
  echo "--- smoke ---"
  $PY main.py --smoke ${extra_flags} 2>&1 | tee "smoke_${label}.log"

  # bench at requested shape.
  echo
  echo "--- bench shape=${shape} ---"
  $PY main.py --bench --csv-out "${csv}" --shape "${shape}" ${extra_flags} \
      2>&1 | tee "${log}"

  # cubin export at that shape.
  echo
  echo "--- cubin export shape=${shape} ---"
  $PY main.py --export-cubin --cubin-out "${cubin}" --shape "${shape}" \
      ${extra_flags} 2>&1 | tee -a "${log}"

  # SASS analysis.
  if [ -n "$CUOBJ" ] && [ -f "${cubin}" ]; then
    $CUOBJ --dump-sass "${cubin}" > "${sass}" 2>&1 || true
    HMMA=$(grep -c "HMMA"     "${sass}" || true)
    FFMA=$(grep -c "FFMA"     "${sass}" || true)
    LDG=$( grep -c "LDG.E"    "${sass}" || true)
    STG=$( grep -c "STG.E"    "${sass}" || true)
    UTMALDG=$(grep -c "UTMALDG" "${sass}" || true)
    UTMASTG=$(grep -c "UTMASTG" "${sass}" || true)
    UTMA_ANY=$(grep -c "UTMA"   "${sass}" || true)
    LINES=$(wc -l < "${sass}")
    echo
    echo "--- SASS summary (${label}) ---"
    echo "  SASS lines : ${LINES}"
    echo "  HMMA insts : ${HMMA}"
    echo "  FFMA insts : ${FFMA}"
    echo "  LDG insts  : ${LDG}"
    echo "  STG insts  : ${STG}"
    echo "  UTMALDG    : ${UTMALDG}"
    echo "  UTMASTG    : ${UTMASTG}"
    echo "  any UTMA   : ${UTMA_ANY}"
  fi
}

# Leg A — explicit TMA-on at qwen3_next_decode.
run_leg "tma_on_qwen3_next_decode" "" "qwen3_next_decode"

# Leg B — explicit TMA-off (falsification leg).
run_leg "tma_off_qwen3_next_decode" "--no-tma" "qwen3_next_decode"

# Leg C — explicit TMA-on at large shape (B=4 H=64 d_k=256 d_v=256).
run_leg "tma_on_large" "" "large"
