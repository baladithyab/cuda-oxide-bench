# mojo-vecadd analysis

Phase A of Wave 18 — toolchain smoke test of Mojo 1.0.0b1 on RTX 5090 sm_120.

## TL;DR

Mojo 1.0.0b1 compiles and runs a 1024-element f32 vector-add kernel on
RTX 5090 sm_120 with **zero environment configuration beyond `pixi add mojo`**,
and **first-try numerical correctness** against an analytical reference.

This phase is purely a "does the toolchain work?" gate — perf numbers come
in Phase B. The thing this phase rules out is a "Mojo is structurally
unreachable on consumer Blackwell" story analogous to wgpu→NVIDIA on WSL2.
That story is now ruled out: Mojo works.

## What ran

```
pixi run mojo /home/codeseys/cuda-exploration/mojo-vecadd/vecadd.mojo
```

Kernel: 1024 elements, 1 grid block × 256 threads × 4 thread-groups (`ceildiv(1024,256)=4`),
output `c[i] = a[i] + b[i]` where `a[i]=i` and `b[i]=0.5*i`, expected `c[i]=1.5*i`.

## Output (run.log captures the full transcript)

```
GPU detected:  NVIDIA GeForce RTX 5090
N = 1024
max_abs_err = 0.0
first nonzero err idx = -1
out[0]   = 0.0   (expected 0.0)
out[1]   = 1.5   (expected 1.5)
out[100] = 150.0   (expected 150.0)
out[N-1] = 1534.5   (expected 1534.5 )
PASS: vecadd numerically exact
```

## Significance

| dimension | result | comparison vs sibling frontends |
|---|---|---|
| install ergonomics | one channel + `pixi add mojo`, ~16s | comparable to cuTile (`pip install cuda-tile`); much simpler than cuda-oxide (LLVM 21 + nightly Rust + cargo-oxide build) |
| env vars required | **none** | cuda-oxide requires `CUDA_HOME` + `LIBNVVM_PATH` (libNVVM shadow bug); cuTile requires nothing; nvcc requires explicit `/usr/local/cuda/bin/nvcc` (apt shim issue). Mojo wins on this axis. |
| GPU detection | `DeviceContext().name()` returns full marketing name | cuda-oxide doesn't expose this directly; CUDA C++ requires `cudaGetDeviceProperties(&p, 0); p.name` boilerplate. |
| first-run JIT cost | sub-second visible (subsumed in compilation) | cuTile first launch is ~639ms JIT; cuda-oxide AOT-builds; nvcc AOT-builds. Mojo seems to be JIT but fast. Phase B with bigger N will reveal whether the JIT is per-kernel-source or per-launch-shape. |
| numerical correctness | exact (max_abs_err = 0.0) | parity with all frontends on this trivial example |

## Discovered API characteristics

1. **JIT vs AOT**: `pixi run mojo file.mojo` JIT-compiles + runs in one shot.
   `pixi run mojo build file.mojo` should AOT-compile to a binary
   (untested in this phase, will exercise in Phase B for stable timing).
2. **Single-source**: host code (allocations, copies, kernel launch,
   verification) and device code (the `vector_addition` def) live in
   the same `.mojo` file. Same shape as cuda-oxide single-source.
3. **TileTensor wrapping**: device buffers are wrapped in `TileTensor`
   before passing to the kernel. The kernel signature receives
   `TileTensor[float_dtype, type_of(layout), MutAnyOrigin]`. This is
   Mojo's safe typed-array indirection layer. Comparable to cuda-oxide's
   `&[T]` / `DisjointSlice<T>` wrappers, but type-richer.
4. **Launch syntax**: `ctx.enqueue_function[kernel, kernel](args, grid_dim=..., block_dim=...)`.
   The double `[kernel, kernel]` is the Mojo type-parameter syntax for
   kernel handle + kernel function. Comparable to cuda-oxide's
   `cuda_launch! { ... }` macro and cuTile's `ct.launch(stream, grid, kernel, args)`.
5. **HostBuffer / DeviceBuffer asymmetry**: Mojo distinguishes host-pinned
   buffers from device buffers explicitly. `enqueue_create_host_buffer`
   vs `enqueue_create_buffer`. `enqueue_copy` between them.

## What was NOT exercised (Phase B / C work)

- Performance measurement (this is N=1024, single-launch, no timing).
- AOT compilation via `mojo build --target-accelerator=sm_120`.
- `.cubin` extraction for SASS analysis.
- Larger N to push memory bandwidth; tile-size sweep.
- Reduction (the most informative kernel for compiler-character analysis).
- Tensor cores (mma.sync) — this is a scalar-arithmetic kernel.

## Time spent in Phase A

- Pixi install: ~5s
- Workspace init + `pixi add mojo`: ~21s wall clock
- Drafting + first run of vecadd.mojo: ~60s (no compile errors, ran first try)
- Writing SETUP.md + ANALYSIS.md: ~10 minutes
- **Total Phase A wall clock: ~15 minutes** (vs the 1-2hr budget). Well
  under budget — the cuTile-style multi-pitfall-discovery scenario did not
  materialize.

Phase B can begin immediately.
