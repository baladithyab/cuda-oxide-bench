NVCC = /usr/local/cuda/bin/nvcc
CXX = clang-14

# ADR-0002: native sm_120, no PTX-JIT fallback.
# ADR-0001: cudaEvent timing in matmul_tc_bf16.cu.
NVCC_FLAGS = -ccbin $(CXX) -O3 -arch=sm_120 -lstdc++ -lm

matmul_tc_bf16: matmul_tc_bf16.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

sass: matmul_tc_bf16
	/usr/local/cuda/bin/cuobjdump --dump-sass matmul_tc_bf16 > matmul_tc_bf16.sass

clean:
	rm -f matmul_tc_bf16 matmul_tc_bf16.sass results.csv run.log build.log

.PHONY: clean sass
