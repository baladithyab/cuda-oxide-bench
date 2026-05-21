#!/bin/bash
# Wave C3.2 — explicit-TMA-mode test of cuTile DSL on MLA-shaped kernel.
#
# Three legs:
#   (A) explicit allow_tma=True  + latency=10  ("TMA-on")
#   (B) explicit allow_tma=False               ("TMA-off" falsification)
#
# For each: smoke, bench, cubin, SASS-grep for UTMALDG / UTMASTG counts,
# and md5sum of the cubins (compare across legs).
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
  local cubin="cubin_${label}.cubin"
  local sass="sass_${label}.sass"
  local csv="results_${label}.csv"
  local log="bench_${label}.log"

  echo
  echo "================================================================"
  echo " LEG: ${label}  flags=[${extra_flags}]"
  echo "================================================================"

  echo "--- smoke ---"
  $PY main.py --smoke ${extra_flags} 2>&1 | tee "smoke_${label}.log"

  echo
  echo "--- bench ---"
  $PY main.py --bench --csv-out "${csv}" ${extra_flags} \
      2>&1 | tee "${log}"

  echo
  echo "--- cubin export ---"
  $PY main.py --export-cubin --cubin-out "${cubin}" \
      ${extra_flags} 2>&1 | tee -a "${log}"

  if [ -n "$CUOBJ" ] && [ -f "${cubin}" ]; then
    $CUOBJ --dump-sass "${cubin}" > "${sass}" 2>&1 || true
    HMMA=$(grep -c "HMMA"     "${sass}" || true)
    FFMA=$(grep -c "FFMA"     "${sass}" || true)
    LDG=$( grep -c "LDG.E"    "${sass}" || true)
    STG=$( grep -c "STG.E"    "${sass}" || true)
    UTMALDG=$(grep -c "UTMALDG" "${sass}" || true)
    UTMASTG=$(grep -c "UTMASTG" "${sass}" || true)
    UTMA_ANY=$(grep -c "UTMA"   "${sass}" || true)
    MUFU=$(grep -c "MUFU.EX2"   "${sass}" || true)
    LINES=$(wc -l < "${sass}")
    MD5=$(md5sum "${cubin}" | awk '{print $1}')
    SIZE=$(stat -c%s "${cubin}")
    echo
    echo "--- SASS summary (${label}) ---"
    echo "  cubin size : ${SIZE} bytes  md5=${MD5}"
    echo "  SASS lines : ${LINES}"
    echo "  HMMA insts : ${HMMA}"
    echo "  FFMA insts : ${FFMA}"
    echo "  LDG insts  : ${LDG}"
    echo "  STG insts  : ${STG}"
    echo "  UTMALDG    : ${UTMALDG}"
    echo "  UTMASTG    : ${UTMASTG}"
    echo "  any UTMA   : ${UTMA_ANY}"
    echo "  MUFU.EX2   : ${MUFU} (softmax exp)"
  fi
}

run_leg "tma_on"  ""
run_leg "tma_off" "--no-tma"

echo
echo "================================================================"
echo " CROSS-LEG COMPARISON"
echo "================================================================"
md5sum cubin_tma_on.cubin cubin_tma_off.cubin || true
echo
echo "If the two md5sums match, allow_tma kwarg is a no-op for these"
echo "MLA tile shapes (compiler refuses TMA regardless)."
echo
echo "Compare against baseline cutile-attn-mla cubin:"
md5sum /home/codeseys/cuda-exploration/cutile-attn-mla/mla_fwd_fused.cubin || true
