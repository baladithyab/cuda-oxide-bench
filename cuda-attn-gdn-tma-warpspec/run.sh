#!/bin/bash
# Wave 22.13 — cuda-attn-gdn-tma-warpspec build + correctness driver.
# Per task: COMPILE + CORRECTNESS only (no timed bench in author cell).
# Orchestrator runs `./bench` separately on idle GPU.
set -euo pipefail
cd "$(dirname "$0")"

NVCC=/usr/local/cuda/bin/nvcc
CXX=clang-14
CUOBJ=/usr/local/cuda/bin/cuobjdump

echo "=== build attn_gdn_tma_warpspec (correctness binary) ==="
make clean >/dev/null
make attn_gdn_tma_warpspec 2>&1 | tee build.log

echo
echo "=== build bench (smoke-only at this stage) ==="
make bench 2>&1 | tee -a build.log

echo
echo "=== correctness run ==="
./attn_gdn_tma_warpspec 2>&1 | tee run.log

echo
echo "=== SASS dump (W22.13 must show BOTH UTMALDG > 0 AND BSSY/BSYNC + SYNCS > 0) ==="
$CUOBJ --dump-sass attn_gdn_tma_warpspec > attn_gdn_tma_warpspec.sass 2>&1 || true
HMMA=$(grep -c "HMMA"        attn_gdn_tma_warpspec.sass || true)
FFMA=$(grep -c "FFMA"        attn_gdn_tma_warpspec.sass || true)
FMUL=$(grep -c "FMUL"        attn_gdn_tma_warpspec.sass || true)
FADD=$(grep -c "FADD"        attn_gdn_tma_warpspec.sass || true)
LDGE=$(grep -c "LDG.E"       attn_gdn_tma_warpspec.sass || true)
LDG128=$(grep -c "LDG.E.128" attn_gdn_tma_warpspec.sass || true)
STG128=$(grep -c "STG.E.128" attn_gdn_tma_warpspec.sass || true)
LDS128=$(grep -c "LDS.128"   attn_gdn_tma_warpspec.sass || true)
STS128=$(grep -c "STS.128"   attn_gdn_tma_warpspec.sass || true)
MUFU=$(grep -c "MUFU"        attn_gdn_tma_warpspec.sass || true)
UTMALDG=$(grep -c "UTMALDG"  attn_gdn_tma_warpspec.sass || true)
UTMASTG=$(grep -c "UTMASTG"  attn_gdn_tma_warpspec.sass || true)
LDGSTS=$(grep -c "LDGSTS"    attn_gdn_tma_warpspec.sass || true)
BSSY=$(grep -c "BSSY"        attn_gdn_tma_warpspec.sass || true)
BSYNC=$(grep -c "BSYNC"      attn_gdn_tma_warpspec.sass || true)
SYNCS=$(grep -c "SYNCS"      attn_gdn_tma_warpspec.sass || true)
BARSYNC=$(grep -cE "BAR\.SYNC" attn_gdn_tma_warpspec.sass || true)
TOTAL=$(wc -l < attn_gdn_tma_warpspec.sass)

echo "SASS lines : $TOTAL"
echo "HMMA       : $HMMA"
echo "FFMA       : $FFMA"
echo "FMUL       : $FMUL"
echo "FADD       : $FADD"
echo "LDG.E      : $LDGE"
echo "LDG.E.128  : $LDG128"
echo "STG.E.128  : $STG128"
echo "LDS.128    : $LDS128"
echo "STS.128    : $STS128"
echo "MUFU       : $MUFU"
echo "UTMALDG    : $UTMALDG    (TMA SIGNAL: must be > 0)"
echo "UTMASTG    : $UTMASTG"
echo "LDGSTS     : $LDGSTS"
echo "BSSY       : $BSSY      (WARP-SPEC SIGNAL)"
echo "BSYNC      : $BSYNC      (WARP-SPEC SIGNAL)"
echo "SYNCS      : $SYNCS      (WARP-SPEC SIGNAL)"
echo "BAR.SYNC   : $BARSYNC    (named-barrier count)"

echo
echo "=== resource usage ==="
$CUOBJ --dump-resource-usage attn_gdn_tma_warpspec 2>&1 | head -40

echo
echo "=== EIATTR (REQNTID, NUM_BARRIERS, MAXREG_COUNT) ==="
$CUOBJ --dump-elf attn_gdn_tma_warpspec 2>&1 | grep -E "EIATTR_REQNTID|EIATTR_NUM_BARRIERS|EIATTR_MAXREG_COUNT|EIATTR_REGCOUNT" | head -10 || true
