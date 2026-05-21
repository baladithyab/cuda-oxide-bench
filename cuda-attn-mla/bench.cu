// Wave 17 W1a — bench.cu is intentionally a thin stub.
//
// The single-binary pattern from cuda-attn-gqa is preferred: attn_mla.cu
// hosts both correctness and bench drivers in one main(). This file exists
// for plan-row-symmetry with cublas-attn-gqa's split layout but compiles
// nothing; the Makefile does not build it. See attn_mla.cu for the harness.
