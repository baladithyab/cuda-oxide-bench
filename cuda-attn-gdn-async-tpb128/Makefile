NVCC = /usr/local/cuda/bin/nvcc
CXX  = clang-14

# -O3 + sm_120 native (RTX 5090 Blackwell). lineinfo helps SASS-level inspection.
NVCCFLAGS = -O3 -arch=sm_120 -ccbin $(CXX) -std=c++17 -lineinfo -lstdc++ -lm

all: attn_gdn_async_tpb128 bench

attn_gdn_async_tpb128: attn_gdn_async_tpb128.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

bench: bench.cu attn_gdn_async_tpb128.cu
	$(NVCC) $(NVCCFLAGS) -o $@ bench.cu

# SASS dump — extract instruction-mix evidence per ADR-0004.
sass: attn_gdn_async_tpb128
	/usr/local/cuda/bin/cuobjdump --dump-sass attn_gdn_async_tpb128 > attn_gdn_async_tpb128.sass 2>&1 || true
	@echo "----- SASS instruction mix -----"
	@HMMA=$$(grep -c "HMMA"        attn_gdn_async_tpb128.sass || true); echo "HMMA          : $$HMMA"
	@FFMA=$$(grep -c "FFMA"        attn_gdn_async_tpb128.sass || true); echo "FFMA          : $$FFMA"
	@LDGE=$$(grep -c "LDG.E"       attn_gdn_async_tpb128.sass || true); echo "LDG.E         : $$LDGE"
	@LDG128=$$(grep -c "LDG.E.128" attn_gdn_async_tpb128.sass || true); echo "LDG.E.128     : $$LDG128"
	@LDG64=$$(grep -c "LDG.E.64"   attn_gdn_async_tpb128.sass || true); echo "LDG.E.64      : $$LDG64"
	@STG128=$$(grep -c "STG.E.128" attn_gdn_async_tpb128.sass || true); echo "STG.E.128     : $$STG128"
	@MUFU=$$(grep -c "MUFU"        attn_gdn_async_tpb128.sass || true); echo "MUFU          : $$MUFU"
	@LDGSTS=$$(grep -c "LDGSTS"    attn_gdn_async_tpb128.sass || true); echo "LDGSTS        : $$LDGSTS  (cp.async — preserved from W22.9)"
	@BSSY=$$(grep -c "BSSY"        attn_gdn_async_tpb128.sass || true); echo "BSSY          : $$BSSY  (warp-spec barrier setup)"
	@BSYNC=$$(grep -c "BSYNC"      attn_gdn_async_tpb128.sass || true); echo "BSYNC         : $$BSYNC"
	@RECON=$$(grep -c "RECONVERGENT" attn_gdn_async_tpb128.sass || true); echo "RECONVERGENT  : $$RECON"
	@BAR=$$(grep -c "BAR.SYNC"     attn_gdn_async_tpb128.sass || true); echo "BAR.SYNC      : $$BAR"
	@SYNCS=$$(grep -c "SYNCS"      attn_gdn_async_tpb128.sass || true); echo "SYNCS         : $$SYNCS  (Blackwell async-tx; cuTile parity signal)"
	@MBAR=$$(grep -cE "ARRIVES|BAR.ARV|MBAR" attn_gdn_async_tpb128.sass || true); echo "MBAR.*        : $$MBAR"

clean:
	rm -f attn_gdn_async_tpb128 bench attn_gdn_async_tpb128.sass results.csv run.log build.log

.PHONY: all clean sass
