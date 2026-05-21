NVCC = /usr/local/cuda/bin/nvcc
CXX  = clang-14

# ADR-0002: native sm_120, no PTX-JIT fallback.
# ADR-0001: cudaEvent timing in attn_mla_tma.cu.
# Need -lcuda for cuTensorMapEncodeTiled (driver API).
NVCC_FLAGS = -ccbin $(CXX) -O3 -arch=sm_120 -std=c++17 -lineinfo -lstdc++ -lm
LDLIBS     = -lcuda

attn_mla_tma: attn_mla_tma.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LDLIBS)

sass: attn_mla_tma
	/usr/local/cuda/bin/cuobjdump --dump-sass attn_mla_tma > attn_mla_tma.sass
	@echo "----- SASS instruction mix (W23.2 cuda-attn-mla-tma) -----"
	@HMMA=$$(grep -c "HMMA"        attn_mla_tma.sass || true); echo "HMMA       : $$HMMA  (target: > 0)"
	@FFMA=$$(grep -c "FFMA"        attn_mla_tma.sass || true); echo "FFMA       : $$FFMA"
	@LDGE=$$(grep -c "LDG.E"       attn_mla_tma.sass || true); echo "LDG.E      : $$LDGE  (W17 W1a baseline: 129; expect << that)"
	@LDG128=$$(grep -c "LDG.E.128" attn_mla_tma.sass || true); echo "LDG.E.128  : $$LDG128"
	@LDG64=$$(grep -c "LDG.E.64"   attn_mla_tma.sass || true); echo "LDG.E.64   : $$LDG64"
	@STG128=$$(grep -c "STG.E.128" attn_mla_tma.sass || true); echo "STG.E.128  : $$STG128"
	@LDS128=$$(grep -c "LDS.128"   attn_mla_tma.sass || true); echo "LDS.128    : $$LDS128"
	@MUFU=$$(grep -c "MUFU"        attn_mla_tma.sass || true); echo "MUFU       : $$MUFU"
	@UTMALDG=$$(grep -c "UTMALDG"  attn_mla_tma.sass || true); echo "UTMALDG    : $$UTMALDG  (target signal: W17 W1a had 0; expect > 0)"
	@UTMASTG=$$(grep -c "UTMASTG"  attn_mla_tma.sass || true); echo "UTMASTG    : $$UTMASTG"

clean:
	rm -f attn_mla_tma attn_mla_tma.sass results.csv run.log build.log

.PHONY: clean sass
