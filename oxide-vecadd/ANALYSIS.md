# oxide-vecadd — Analysis

## Test description

End-to-end smoke test for the cuda-oxide compiler pipeline. Adds two `f32` vectors of length 1024:

```rust
#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = a[idx.get()] + b[idx.get()];
    }
}
```

This exists not as a performance benchmark — vector addition at length 1024 is dominated by kernel launch overhead — but as a **toolchain smoke test** that validates the entire Rust → MIR → Pliron IR → LLVM IR → PTX → driver-load pipeline. If this passes, every more interesting test in this repo can run.

## Methodology

Single launch, single iteration. The host code:

1. Builds two host vectors `[0, 1, 2, ..., 1023]` and `[0, 2, 4, ..., 2046]`.
2. Copies them to device via `DeviceBuffer::from_host`.
3. Launches `vecadd` with `LaunchConfig::for_num_elems(1024)` (block size 256, grid size 4).
4. Copies result back, verifies element-by-element against `a[i] + b[i]` with tolerance 1e-5.
5. Reports PASS / FAIL.

No timing collected. The pass/fail is the test.

## Result

```
PASSED: all 1024 elements correct
```

This confirms:

- `cargo oxide doctor` → all green
- `cargo oxide build` produces `oxide_vecadd.ptx` and `oxide_vecadd.ll`
- `load_kernel_module(&ctx, "oxide_vecadd")` finds and loads the PTX (note: artifact name uses underscore, not hyphen — see [Pitfalls](#pitfalls))
- The PTX kernel launches, executes, and produces correct results on the RTX 5090 under the WSL2 NVIDIA driver passthrough
- `DisjointSlice<T>` + `ThreadIndex` API works as designed

## Pitfalls discovered while writing this test

1. **Artifact naming**: cuda-oxide writes the PTX as `oxide_vecadd.ptx` (Rust crate name with `-` → `_`), but `load_kernel_module(&ctx, "oxide-vecadd")` (using the original project name with the hyphen) fails with `NoArtifact`. Always pass the underscored name.
2. **CARGO_MANIFEST_DIR is read at runtime**, not compile time, by `load_kernel_module`. So you must run from the project's manifest directory, or set the env var explicitly.
3. **rust-toolchain.toml is auto-generated** by `cargo oxide new` and pins a specific nightly (e.g., `nightly-2026-04-03`). If your installed nightly is older or the pinned one isn't downloaded yet, the build fails opaquely. `rustup install nightly-<date>` fixes it.

## Reproducing

```bash
cd oxide-vecadd
cargo oxide run oxide-vecadd
# Expected: PASSED: all 1024 elements correct
```

Prereqs (see top-level `SETUP.md`):
- LLVM 21 with NVPTX target
- Rust nightly with `rust-src`, `rustc-dev`, `llvm-tools-preview`
- CUDA 12.x toolkit + driver
- `cargo-oxide` installed via `cargo +nightly install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide`
- `cargo oxide doctor` from a project directory shows all green

## What this test does NOT cover

- Performance (vec add is launch-overhead dominated at this size)
- Multi-kernel modules
- Shared memory, atomics, warp ops
- Device functions / cross-crate kernels
- Complex argument types (only `&[f32]` and `DisjointSlice<f32>` exercised)
- Error paths (we panic on first failure)

For real perf comparison see `../oxide-matmul/ANALYSIS.md`.

## Files

- `src/main.rs` — kernel + host code (~50 lines, mostly boilerplate)
- `Cargo.toml` — pulls in `cuda-device`, `cuda-host`, `cuda-core` from NVlabs git
- `rust-toolchain.toml` — pins nightly + components
- `oxide_vecadd.ptx` — generated PTX (committed for inspection)
- `oxide_vecadd.ll` — generated LLVM IR (committed for inspection)
- `run.log` — last successful run output
