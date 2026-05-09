# ADR 0002: Use native `-arch=sm_120` for nvcc baselines, drop PTX-JIT path

**Status:** accepted (2026-05-08)

**Context.** v0 compiled `cuda-matmul/matmul.cu` with `nvcc -arch=sm_89` and relied on the driver's PTX JIT to retarget Blackwell at load time. This was because the system `nvcc` from `/usr/bin/nvcc` is a CUDA 12.0 shim that does not recognize `sm_120`. Phase 3 research discovered the actual CUDA toolkit at `/usr/local/cuda/bin/nvcc` is **13.2** (released March 2026), which natively supports `sm_120`.

**Decision.**

1. All `nvcc` invocations in this repo use `/usr/local/cuda/bin/nvcc` (CUDA 13.2) with `-arch=sm_120`.
2. SETUP.md documents the PATH gotcha and instructs users to either `export PATH=/usr/local/cuda/bin:$PATH` or invoke nvcc by absolute path.
3. The `Makefile` / build commands in each `cuda-*` and `cublas-*` folder reference the absolute path to be unambiguous.

**Why.**

- Native `sm_120` builds emit Blackwell-specific instructions (e.g. tcgen05, TMA-aware scheduling) where applicable. PTX-JIT from `sm_89` is forward-compatible but conservative.
- The methodology should report **what nvcc on this Blackwell SoC actually emits**, not the lowest-common-denominator Ampere-era PTX.
- For naive matmul specifically the difference is small (no Tensor Cores in the algorithm), but for any future kernel that does use Blackwell features, only native compilation will exercise them. Setting the convention now avoids retesting later.

**Alternatives considered.**

- *Keep `sm_89` for "lowest common denominator" portability.* Misleading: this isn't a portability test, it's a perf test. Use the best path each compiler can produce.
- *Compile `sm_120` with both 12.0 and 13.2 to compare PTX-JIT vs native.* Interesting but out of scope; would require resolving the apt-package collision, since the 12.0 nvcc shim doesn't know `sm_120`.

**Consequences.**

- Baseline TFLOPS may shift; document the v0 → v1 change in `SUMMARY.md`.
- Still single-driver (595.58.03), so we report what one driver/compiler combo does. Future versions may differ.
- The `cuda-matmul/run.log` from v0 is preserved (in commit `5065f3e`); v1's run.log overwrites it with a header noting `-arch=sm_120 -ccbin clang-14`.
