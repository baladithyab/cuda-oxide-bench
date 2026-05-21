NVCC = /usr/local/cuda/bin/nvcc
CXX  = clang-14

# -O3 + sm_120 native (RTX 5090 Blackwell). lineinfo helps SASS-level inspection.
# We need cuda driver lib for cuTensorMapEncodeTiled. Link with libcuda.
NVCCFLAGS = -O3 -arch=sm_120 -ccbin $(CXX) -std=c++17 -lineinfo -lstdc++ -lm
LDLIBS    = -lcuda

all: attn_gdn_tma bench bench_sweep

attn_gdn_tma: attn_gdn_tma.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDLIBS)

bench: bench.cu attn_gdn_tma.cu
	$(NVCC) $(NVCCFLAGS) -o $@ bench.cu $(LDLIBS)

bench_sweep: bench_sweep.cu attn_gdn_tma.cu
	$(NVCC) $(NVCCFLAGS) -o $@ bench_sweep.cu $(LDLIBS)

# SASS dump — extract instruction-mix evidence per ADR-0004.
# UTMALDG is the target signal (W22.10 hypothesis: nvcc-generated UTMALDG path).
sass: attn_gdn_tma
	/usr/local/cuda/bin/cuobjdump --dump-sass attn_gdn_tma > attn_gdn_tma.sass 2>&1 || true
	@echo "----- SASS instruction mix (W22.10 cuda-attn-gdn-tma) -----"
	@HMMA=$$(grep -c "HMMA"        attn_gdn_tma.sass || true); echo "HMMA       : $$HMMA"
	@FFMA=$$(grep -c "FFMA"        attn_gdn_tma.sass || true); echo "FFMA       : $$FFMA"
	@LDGE=$$(grep -c "LDG.E"       attn_gdn_tma.sass || true); echo "LDG.E      : $$LDGE"
	@LDG128=$$(grep -c "LDG.E.128" attn_gdn_tma.sass || true); echo "LDG.E.128  : $$LDG128  (W1c had 16; expect ~0 here)"
	@LDG64=$$(grep -c "LDG.E.64"   attn_gdn_tma.sass || true); echo "LDG.E.64   : $$LDG64"
	@STG128=$$(grep -c "STG.E.128" attn_gdn_tma.sass || true); echo "STG.E.128  : $$STG128"
	@LDS128=$$(grep -c "LDS.128"   attn_gdn_tma.sass || true); echo "LDS.128    : $$LDS128"
	@MUFU=$$(grep -c "MUFU"        attn_gdn_tma.sass || true); echo "MUFU       : $$MUFU"
	@UTMALDG=$$(grep -c "UTMALDG"  attn_gdn_tma.sass || true); echo "UTMALDG    : $$UTMALDG  (target signal: W1c had 0; expect > 0)"
	@UTMASTG=$$(grep -c "UTMASTG"  attn_gdn_tma.sass || true); echo "UTMASTG    : $$UTMASTG"
	@LDGSTS=$$(grep -c "LDGSTS"    attn_gdn_tma.sass || true); echo "LDGSTS     : $$LDGSTS  (cp.async, NOT cp.async.bulk.tensor)"
	@ARRIVES=$$(grep -cE "ARRIVES|MBAR" attn_gdn_tma.sass || true); echo "ARRIVES/MBAR : $$ARRIVES"
	@SYNCS=$$(grep -c "SYNCS"      attn_gdn_tma.sass || true); echo "SYNCS      : $$SYNCS"

clean:
	rm -f attn_gdn_tma bench attn_gdn_tma.sass results.csv run.log build.log

.PHONY: all clean sass
