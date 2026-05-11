NVCC := /usr/local/cuda/bin/nvcc
CXX  := clang-14

NVCCFLAGS := -ccbin $(CXX) -O3 -arch=sm_120 -std=c++17 -lstdc++ -lcublas -lm

attn_gqa: attn_gqa.cu softmax.cu
	$(NVCC) $(NVCCFLAGS) -o $@ attn_gqa.cu softmax.cu

clean:
	rm -f attn_gqa

.PHONY: clean
