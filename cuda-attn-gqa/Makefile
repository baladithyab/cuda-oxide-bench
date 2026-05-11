NVCC = /usr/local/cuda/bin/nvcc
CXX = clang-14

attn_gqa: attn_gqa.cu
	$(NVCC) -ccbin $(CXX) -O3 -arch=sm_120 -lstdc++ -lm -o $@ $<

clean:
	rm -f attn_gqa results.csv run.log

.PHONY: clean
