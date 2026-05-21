NVCC = /usr/local/cuda/bin/nvcc
CXX  = clang-14

# -O3 + sm_120 native (RTX 5090 Blackwell). lineinfo helps SASS-level inspection.
# -lcuda needed for cuTensorMapEncodeTiled (driver API).
NVCCFLAGS = -O3 -arch=sm_120 -ccbin $(CXX) -std=c++17 -lineinfo -lstdc++ -lm
LDLIBS    = -lcuda

all: attn_gdn_tma_warpspec bench

attn_gdn_tma_warpspec: attn_gdn_tma_warpspec.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDLIBS)

bench: bench.cu attn_gdn_tma_warpspec.cu
	$(NVCC) $(NVCCFLAGS) -o $@ bench.cu $(LDLIBS)

# SASS dump — extract instruction-mix evidence per ADR-0004.
# Targets: BOTH UTMALDG.2D > 0 (TMA preserved) AND BSSY/BSYNC + SYNCS > 0 (warp-spec engaged).
sass: attn_gdn_tma_warpspec
	/usr/local/cuda/bin/cuobjdump --dump-sass attn_gdn_tma_warpspec > attn_gdn_tma_warpspec.sass 2>&1 || true
	@echo "----- SASS instruction mix (W22.13 cuda-attn-gdn-tma-warpspec) -----"
	@HMMA=$$(grep -c "HMMA"        attn_gdn_tma_warpspec.sass || true); echo "HMMA       : $$HMMA"
	@FFMA=$$(grep -c "FFMA"        attn_gdn_tma_warpspec.sass || true); echo "FFMA       : $$FFMA"
	@FMUL=$$(grep -c "FMUL"        attn_gdn_tma_warpspec.sass || true); echo "FMUL       : $$FMUL"
	@FADD=$$(grep -c "FADD"        attn_gdn_tma_warpspec.sass || true); echo "FADD       : $$FADD"
	@LDGE=$$(grep -c "LDG.E"       attn_gdn_tma_warpspec.sass || true); echo "LDG.E      : $$LDGE"
	@LDG128=$$(grep -c "LDG.E.128" attn_gdn_tma_warpspec.sass || true); echo "LDG.E.128  : $$LDG128"
	@STG128=$$(grep -c "STG.E.128" attn_gdn_tma_warpspec.sass || true); echo "STG.E.128  : $$STG128"
	@LDS128=$$(grep -c "LDS.128"   attn_gdn_tma_warpspec.sass || true); echo "LDS.128    : $$LDS128"
	@STS128=$$(grep -c "STS.128"   attn_gdn_tma_warpspec.sass || true); echo "STS.128    : $$STS128"
	@MUFU=$$(grep -c "MUFU"        attn_gdn_tma_warpspec.sass || true); echo "MUFU       : $$MUFU"
	@UTMALDG=$$(grep -c "UTMALDG"  attn_gdn_tma_warpspec.sass || true); echo "UTMALDG    : $$UTMALDG  (TMA SIGNAL: must be > 0)"
	@UTMASTG=$$(grep -c "UTMASTG"  attn_gdn_tma_warpspec.sass || true); echo "UTMASTG    : $$UTMASTG"
	@LDGSTS=$$(grep -c "LDGSTS"    attn_gdn_tma_warpspec.sass || true); echo "LDGSTS     : $$LDGSTS"
	@BSSY=$$(grep -c "BSSY"        attn_gdn_tma_warpspec.sass || true); echo "BSSY       : $$BSSY  (WARP-SPEC SIGNAL)"
	@BSYNC=$$(grep -c "BSYNC"      attn_gdn_tma_warpspec.sass || true); echo "BSYNC      : $$BSYNC  (WARP-SPEC SIGNAL)"
	@SYNCS=$$(grep -c "SYNCS"      attn_gdn_tma_warpspec.sass || true); echo "SYNCS      : $$SYNCS  (WARP-SPEC SIGNAL)"
	@BARSYNC=$$(grep -cE "BAR\.SYNC" attn_gdn_tma_warpspec.sass || true); echo "BAR.SYNC   : $$BARSYNC"

clean:
	rm -f attn_gdn_tma_warpspec bench attn_gdn_tma_warpspec.sass results.csv run.log build.log

.PHONY: all clean sass
