# Upstream issue drafts — NVIDIA cutile-python

Drafts prepared for submission to <https://github.com/nvidia/cutile-python/issues>.
All findings are from **Wave 12–13** of cuda-exploration, an independent
third-party benchmark comparing cuda-oxide (rust-cuda), CUDA C++ (nvcc),
and cuTile on RTX 5090 (Blackwell consumer, `sm_120`) with cuda-tile 1.3.0.

Each issue is evidence-first: every headline claim has a corresponding
SASS dump, per-iter CSV, or reproducer script committed in this repo.

## Drafts

| # | file | title | status |
|---|---|---|---|
| 1 | [`01-ctmma-f32-no-ffma-fusion.md`](01-ctmma-f32-no-ffma-fusion.md) | `[bug] ct.mma(f32, f32, f32)` generates unfused scalar FMUL+FADD; missing FFMA contraction (RTX 5090 sm_120, v1.3.0) | **ready to submit** |
| 2 | [`02-constant-int-launch-arg-tiled-view.md`](02-constant-int-launch-arg-tiled-view.md) | `[bug/docs] ct.Constant[int]` launch arg cannot be used as `tiled_view` shape (TileTypeError) in v1.3.0 | **ready to submit** |

## Context for maintainers

**cuda-exploration** (<https://github.com/baladithyab/cuda-exploration>)
is a public benchmark repository evaluating GPU programming frontends for
Blackwell consumer hardware. Wave 12 added cuTile benchmarks; Wave 13
disassembled the resulting cubins with `cuobjdump` (CUDA 13.2) for
SASS-level cross-stack comparison. These two issues are what survived
that SASS-level audit with cross-stack evidence that the behavior is
cuTile-specific (i.e. not a hardware ceiling and not a general
compile-pipeline issue). The repo's overall findings for cuTile are
positive — `ct.mma` at f16/bf16/tf32 engages Blackwell tensor cores
correctly (172.5 TFLOPS at f16 N=4096) and `ct.reduce_sum` lowers to
`UTMALDG.1D` bulk-TMA loads (10-12% faster than hand-written reductions
on the same hardware) — so these two filings are narrowly scoped.
