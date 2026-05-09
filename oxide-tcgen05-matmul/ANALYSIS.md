# Wave 6 W6C — cuda-oxide `tcgen05_matmul` on RTX 5090 (sm_120)

## What is `tcgen05`?
`tcgen05` is NVIDIA's 5th-generation Tensor Core instruction family, introduced
with Blackwell. It operates on a new on-chip **Tensor Memory (TMEM)** — a
dedicated memory space distinct from registers/SMEM — and exposes
warp-uniform issue of large fixed-shape MMAs (128×128×16 here, plus larger
variants) via the `tcgen05.mma` family. It is the hardware path for
low-precision training/inference formats on Blackwell: **FP4 (e2m1/e3m0),
FP6 (e2m3/e3m2), FP8 (e4m3/e5m2), plus BF16/FP16/TF32/INT8**. Supporting
ops (`tcgen05.alloc/dealloc`, `tcgen05.ld/st`, `tcgen05.commit`,
`tcgen05.mma.ws`, cluster-wide async completion) all live in the same class.

## Blackwell SM split
Blackwell ships in two SKU families:
- **sm_100 / sm_100a** — data-center (B100/B200/GB200/B300). Full `tcgen05`
  instruction set, TMEM, CTA-pair MMA, 2-SM cluster MMA.
- **sm_120** — consumer (RTX 5090/5080/5070 Ti, this box). Blackwell
  ISA minus `tcgen05`. No TMEM. Standard 4th-gen `wgmma`-style Tensor
  Cores plus FP4/FP8 throughput gains, but the `tcgen05.*` PTX opcodes
  are **not encodable** on sm_120.

## What we ran
- Imported example verbatim; swapped path-deps to git-deps; pinned the
  `nightly-2026-04-03` toolchain used by the sibling oxide crates.
- `cargo oxide build oxide-tcgen05-matmul` → **✓ succeeded in 58s**, emitted
  `oxide_tcgen05_matmul.ptx` with `.target sm_100a` (the codegen hard-selects
  sm_100a regardless of our CLI arch — the `tcgen05` intrinsics force it).
- Symlinked `tcgen05_matmul.ptx` to match the expected filename.
- `cargo oxide run` → runs, calls `cuModuleLoad`, driver returns
  `CUDA_ERROR_INVALID_PTX`. The example catches this and prints:
  > ⚠️ tcgen05 (5th gen tensor cores) requires sm_100 (datacenter Blackwell only).
  > Your GPU is sm_120 (consumer Blackwell has no tcgen05).
- `cargo oxide run --arch=sm_100a` — same outcome; host-arch flag does not
  change runtime device.

## Verdict
PTX codegen works end-to-end on sm_120 (good news for the toolchain — the
`rustc-codegen-cuda` backend emits correct sm_100a tcgen05 PTX). **Kernel
execution is impossible on RTX 5090** — this is a silicon limit, not a
software gap. To run this example we would need a B100/B200/GB200 class
device. Inspecting `tcgen05_matmul.ptx` (64 `tcgen05` instruction strings)
is the maximal extractable value on consumer Blackwell.
