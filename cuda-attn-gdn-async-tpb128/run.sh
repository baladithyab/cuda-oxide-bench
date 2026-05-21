#!/bin/bash
# Wave 22.11 — cuda-attn-gdn-async-tpb128 build + correctness driver.
# Per task: COMPILE + CORRECTNESS only (no timed bench in W22.11 authoring).
# Orchestrator runs `./bench` separately on idle GPU.
set -euo pipefail
cd "$(dirname "$0")"

NVCC=/usr/local/cuda/bin/nvcc
CXX=clang-14
CUOBJ=/usr/local/cuda/bin/cuobjdump

echo "=== build attn_gdn_async_tpb128 (correctness binary) ==="
make clean >/dev/null
make attn_gdn_async_tpb128 2>&1 | tee build.log

echo
echo "=== build bench (smoke-only at this stage) ==="
make bench 2>&1 | tee -a build.log

echo
echo "=== correctness run ==="
./attn_gdn_async_tpb128 2>&1 | tee run.log

echo
echo "=== SASS dump (cp.async + warp-spec barriers per ADR-0004) ==="
$CUOBJ --dump-sass attn_gdn_async_tpb128 > attn_gdn_async_tpb128.sass 2>&1 || true
SASS=attn_gdn_async_tpb128.sass
HMMA=$(grep -c "HMMA"        $SASS || true)
FFMA=$(grep -c "FFMA"        $SASS || true)
LDGE=$(grep -c "LDG.E"       $SASS || true)
LDG128=$(grep -c "LDG.E.128" $SASS || true)
LDG64=$(grep -c "LDG.E.64"   $SASS || true)
STG128=$(grep -c "STG.E.128" $SASS || true)
MUFU=$(grep -c "MUFU"        $SASS || true)
LDGSTS=$(grep -c "LDGSTS"    $SASS || true)
BSSY=$(grep -c "BSSY"        $SASS || true)
BSYNC=$(grep -c "BSYNC"      $SASS || true)
RECON=$(grep -c "RECONVERGENT" $SASS || true)
BAR=$(grep -c "BAR.SYNC"     $SASS || true)
SYNCS=$(grep -c "SYNCS"      $SASS || true)
MBAR=$(grep -cE "ARRIVES|BAR.ARV|MBAR" $SASS || true)
TOTAL=$(wc -l < $SASS)
echo "SASS lines    : $TOTAL"
echo "HMMA          : $HMMA      (expect 0; GDN is memory-bound)"
echo "FFMA          : $FFMA      (expect > 0)"
echo "LDG.E         : $LDGE      (sync gmem loads)"
echo "LDG.E.128     : $LDG128    (W1c=16, W22.9=0; cp.async should keep this 0)"
echo "LDG.E.64      : $LDG64"
echo "STG.E.128     : $STG128    (state writes)"
echo "MUFU          : $MUFU      (expect 0)"
echo "LDGSTS        : $LDGSTS    (cp.async — preserved from W22.9)"
echo "BSSY          : $BSSY      (warp-spec barrier setup; cuTile pattern signal)"
echo "BSYNC         : $BSYNC     (warp-spec barrier wait)"
echo "RECONVERGENT  : $RECON     (warp divergence reconvergence — expected with 1P+3C)"
echo "BAR.SYNC      : $BAR       (block-wide barrier — __syncthreads)"
echo "SYNCS         : $SYNCS     (Blackwell async-tx; cuTile parity signal)"
echo "MBAR.*        : $MBAR      (mbarrier infrastructure)"
