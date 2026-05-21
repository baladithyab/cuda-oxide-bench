# mojo-vecadd setup

Phase A of Wave 18: Mojo toolchain smoke test on RTX 5090 sm_120.

## Verified working configuration (2026-05-20)

- **OS:** WSL2 / Ubuntu 24.04 (glibc 2.39)
- **GPU:** NVIDIA GeForce RTX 5090 (Blackwell sm_120, 32607 MiB)
- **NVIDIA driver:** 596.21 (Windows host; ≥ 580 satisfies Mojo requirement)
- **CUDA toolkit:** 13.2 (`/usr/local/cuda`)
- **pixi:** 0.68.1
- **Mojo:** 1.0.0b1 (build a9591de6) — first stable beta
- **Conda channels:** `https://conda.modular.com/max` + `conda-forge`
- **Pixi env size on disk:** 890 MB
- **Total install wall-clock:** pixi install ≈ 5s, `pixi add mojo` ≈ 16s

## Install steps (verbatim, no environment surprises)

```bash
# 1. Install pixi if not present
sh /tmp/pixi-installer.sh         # downloads from https://pixi.sh/install.sh
# Adds ~/.pixi/bin to ~/.bashrc; source it or restart shell.
export PATH="$HOME/.pixi/bin:$PATH"

# 2. Create Mojo workspace with the correct channels.
#    This is the spot where pitfall #1 below lives.
mkdir -p /home/codeseys/cuda-exploration/mojo-workspace
cd /home/codeseys/cuda-exploration/mojo-workspace
pixi init . -c https://conda.modular.com/max/ -c conda-forge

# 3. Add Mojo
pixi add mojo

# 4. Sanity check
pixi run mojo --version       # → Mojo 1.0.0b1 (a9591de6)
```

## Run the vecadd smoke test

```bash
cd /home/codeseys/cuda-exploration/mojo-workspace
pixi run mojo /home/codeseys/cuda-exploration/mojo-vecadd/vecadd.mojo
```

Expected output: `PASS: vecadd numerically exact`, with `max_abs_err = 0.0`.

## Pitfalls discovered during install

### Pitfall #1: `pixi init --format mojoproject` does NOT add the Modular channel

The `--format mojoproject` flag generates a `mojoproject.toml` with **only**
`channels = ["conda-forge"]`. Subsequent `pixi add mojo` then fails because
`mojo` is not on conda-forge — it's on `https://conda.modular.com/max`.

**Use the explicit form from the [official quickstart][quickstart]:**

```bash
pixi init . -c https://conda.modular.com/max/ -c conda-forge
```

This generates a regular `pixi.toml` (not `mojoproject.toml`) but with the
correct channels list. The `mojoproject.toml` format option appears to be
either out of date or for a different workflow.

[quickstart]: https://docs.modular.com/mojo/manual/quickstart/

### Pitfall #2 (potential, NOT hit): driver < 580

For systems with NVIDIA driver older than 580, Mojo refuses to compile GPU
code. The workaround is `export MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas`
which bypasses the bundled compiler's driver check. Our driver 596.21 was
fine, this was not exercised. Recipe documented in the wave-18 research doc
in case a future runner hits this.

## What we verified

1. **GPU detection:** `DeviceContext().name()` returns `"NVIDIA GeForce RTX 5090"`.
2. **Code compiles:** `mojo` JIT-compiled the kernel without explicit
   `--target-accelerator=sm_120`. The compiler auto-detected the device.
3. **Kernel ran:** all 1024 elements correct, `max_abs_err = 0.0` against
   the analytical reference `i + 0.5*i = 1.5*i`.
4. **No env vars needed:** no `CUDA_HOME`, no `LIBNVVM_PATH`, no
   `MODULAR_NVPTX_COMPILER_PATH`.

This is in stark contrast to cuda-oxide, which requires `CUDA_HOME` and
`LIBNVVM_PATH` to be exported before any `cargo oxide` invocation (the
libNVVM shadow bug). Mojo's first-run UX is clearly better here.

## Mojo 1.0.0b1 GPU API surface (relevant imports)

The canonical example uses:

```mojo
from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from layout import TileTensor, row_major
```

`std.gpu.host.DeviceContext` is the host-side handle. `std.gpu.{block_dim,
block_idx, thread_idx}` are the kernel-side built-ins (cf. CUDA C++'s
`blockDim`, `blockIdx`, `threadIdx`). `layout.TileTensor` is Mojo's
typed-array wrapper exposed to kernels.

Older docs reference `from gpu.host import DeviceContext` (no `std.`).
That's an older path — 1.0.0b1 prefers `std.gpu.host`.

## Open questions Phase B will answer

1. Is there an event-based timing API on `DeviceContext` analogous to
   `cudaEventRecord` for accurate per-kernel timing?
2. Does `mojo build --target-accelerator=sm_120` produce a `.cubin` we can
   disassemble with `cuobjdump --dump-sass`?
3. Does Mojo's reduction lower to TMA bulk loads (`UTMALDG.1D`) on sm_120
   like cuTile does (+11% vs warp-shuffle path), or does it stay on the
   `SHFL.BFLY` warp-shuffle path like nvcc and cuda-oxide?

These are the Phase B exit criteria — if Phase B answers them, we have a
clear story for whether Mojo joins cuTile in the "TMA-on-sm_120 club" or
stays parity with the older paths.
