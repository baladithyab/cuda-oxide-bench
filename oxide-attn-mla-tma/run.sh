#!/bin/bash
# Wave C3.3 — build + run cuda-oxide MLA with TMA.

set -euo pipefail
cd "$(dirname "$0")"

export CUDA_HOME=/usr/local/cuda
export LIBNVVM_PATH=/usr/local/cuda/nvvm/lib64/libnvvm.so
export PATH=/usr/lib/llvm-21/bin:$PATH:$HOME/.cargo/bin

echo "=== build ==="
cargo oxide build --arch sm_120 2>&1 | tee build.log

echo
echo "=== run ==="
cargo oxide run --arch sm_120 2>&1 | tee run.log

echo
echo "=== SASS analysis ==="
if [ -x /usr/local/cuda/bin/cuobjdump ]; then
  CUOBJ=/usr/local/cuda/bin/cuobjdump
else
  echo "ERROR: /usr/local/cuda/bin/cuobjdump not found"
  exit 1
fi
CUBIN=""
for c in oxide_attn_mla_tma.cubin target/oxide_attn_mla_tma.cubin \
         target/sm_120/release/deps/oxide_attn_mla_tma.cubin; do
  if [ -f "$c" ]; then CUBIN="$c"; break; fi
done
if [ -z "$CUBIN" ]; then
  CUBIN=$(find . -name 'oxide_attn_mla_tma*.cubin' 2>/dev/null | head -n1 || true)
fi
if [ -z "$CUBIN" ] || [ ! -f "$CUBIN" ]; then
  echo "WARN: no cubin found; SASS analysis skipped"
else
  echo "cubin: $CUBIN"
  $CUOBJ --dump-sass "$CUBIN" > oxide_attn_mla_tma.sass 2>&1 || true
  HMMA=$(grep -c "HMMA" oxide_attn_mla_tma.sass || true)
  FFMA=$(grep -c "FFMA" oxide_attn_mla_tma.sass || true)
  UTMALDG=$(grep -c "UTMALDG" oxide_attn_mla_tma.sass || true)
  LDG=$(grep -c "LDG.E" oxide_attn_mla_tma.sass || true)
  STG=$(grep -c "STG.E" oxide_attn_mla_tma.sass || true)
  LDS=$(grep -c "LDS" oxide_attn_mla_tma.sass || true)
  echo "SASS lines  : $(wc -l < oxide_attn_mla_tma.sass)"
  echo "HMMA insts  : ${HMMA}"
  echo "FFMA insts  : ${FFMA}"
  echo "UTMALDG ins : ${UTMALDG}"
  echo "LDG.E insts : ${LDG}"
  echo "STG.E insts : ${STG}"
  echo "LDS  insts  : ${LDS}"
  echo
  echo "--- per-kernel UTMALDG counts ---"
  awk '
    /Function[[:space:]]*:/    { sub(/.*: /, ""); fname=$0; if (!(fname in counts)) counts[fname]=0; next }
    /UTMALDG/                   { if (fname != "") counts[fname]++ }
    END { for (f in counts) printf("  %-40s %d\n", f, counts[f]) }
  ' oxide_attn_mla_tma.sass
fi
