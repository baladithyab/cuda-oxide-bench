NVCC = /usr/local/cuda/bin/nvcc
CXX = clang-14

# ADR-0002: native sm_120, no PTX-JIT fallback
# ADR-0001: cudaEvent timing in attn_mla.cu
NVCC_FLAGS = -ccbin $(CXX) -O3 -arch=sm_120 -lstdc++ -lm

attn_mla: attn_mla.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Bench harness mirrors cublas-attn-gqa NPY harness. For Wave 17 W1a we
# fold bench + correctness into the same binary so a single `./attn_mla`
# does both — same pattern as cuda-attn-gqa.
bench.cu: attn_mla.cu
	@echo "bench.cu is intentionally a stub — bench logic lives in attn_mla.cu (single-binary pattern)."

sass: attn_mla
	/usr/local/cuda/bin/cuobjdump --dump-sass attn_mla > attn_mla.sass

clean:
	rm -f attn_mla attn_mla.sass results.csv run.log

.PHONY: clean sass
