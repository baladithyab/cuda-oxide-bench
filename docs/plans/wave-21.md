# Wave 21 plan — Mojo bf16-in/f32-acc tiled matmul (hand-rolled)

> **For Hermes:** Use subagent-driven-development for execution; orchestrator owns smoke-test + benchmark passes since the kernel is one tightly-coupled artifact.

**Status:** PLANNED, awaiting greenlight to execute.
**Created:** 2026-05-21
**Predecessor:** Wave 20 (proved `HMMA.16816.F32.BF16 × 1` reachable via raw `mma()` on `.target sm_120a`; full harness deferred). See [results/wave20-summary.md](../../results/wave20-summary.md).
**Skill:** `mlops/rust-gpu-compute` — `references/mojo-mma-shapes.md` is the per-shape SIMD-width + PTX-distribution reference; `references/mojo-tensor-core.md` is the Wave 19 kernel companion.

## Goal

Ship `mojo-matmul-bf16/` — a hand-rolled bf16-in/f32-acc tiled matmul on RTX 5090 sm_120, **closing the API-coverage gap** proven in Wave 20. The path exists; this wave builds the harness around it.

**Performance target:** 100-130 TFLOPS at M=N=K=4096 (60-80% of cuTile's 159 TF; 1.8-2.3× the Wave 19 TF32 path's 55.5 TF). Expected gap to cuTile is `cp.async` vs TMA loads, NOT compiler quality.

**Numerical correctness target:** combined `|a-b| <= atol + rtol*|b|` check vs `vendor_blas.matmul`, with `atol=1.0, rtol=1e-2` (loose because bf16 inputs lose ~7 mantissa bits — same tolerance cuTile bf16 matmul uses).

## Architecture

The Wave 19 `mojo-matmul-tc/matmul_tc.mojo` kernel structure (BM=64, BN=64, BK=32, WM=32, WN=32, 4 warps/block) is the template. Wave 21 keeps the smem layout, `cp.async` loads, and outer tile iteration **identical** but replaces the four `TensorCore`-wrapper calls with hand-rolled equivalents:

| Wave 19 (TF32, same-dtype) | Wave 21 (bf16-in/f32-acc) |
|---|---|
| `mma_op = TensorCore[f32, f32, Index(16,8,4)]()` | (delete — direct `mma()` call instead) |
| `a_reg = mma_op.load_a(A_mma_tile)` | hand-rolled per-lane bf16 fragment load |
| `b_reg = mma_op.load_b(B_mma_tile)` | hand-rolled per-lane bf16 fragment load |
| `d_reg = mma_op.mma_op(a_reg, b_reg, c_reg)` | `mma(d_frag, a_frag, b_frag, c_frag)` — m16n8k16 |
| `mma_op.store_d(C_mma_tile, c_reg)` | hand-rolled epilogue per PTX 9.7.13.4.8 |

**Per-warp SIMD widths (m16n8k16 bf16 dispatch lane):**
- A: `SIMD[bf16, 8]` per lane × 32 lanes = 256 elements (16×16 ✓)
- B: `SIMD[bf16, 4]` per lane × 32 lanes = 128 elements (16×8 ✓)
- C/D: `SIMD[f32, 4]` per lane × 32 lanes = 128 elements (16×8 ✓)

**MMA shape change:** `MMA_K=4` (TF32) → `MMA_K=16` (bf16). With BK=32, the inner k-loop runs `BK/MMA_K = 32/16 = 2` MMA calls per tile pass instead of 8. **Per-warp work per tile pass: same (`WM/MMA_M × WN/MMA_N = 2 × 4 = 8` MMAs), just at higher arithmetic intensity per call.**

## Tech Stack

- **Mojo 1.0.0b1** (a9591de6), pixi-managed at `mojo-workspace/`
- `from std.gpu.compute.mma import mma` — the raw m16n8k16 dispatch
- `from layout.layout_tensor import LayoutTensor, copy_dram_to_sram_async` — keep the shared-memory plumbing
- `from layout.tensor_core import _ldmatrix_x4` (if exposed) OR hand-rolled lane-stripe loads from smem
- **NO** `from layout.tensor_core import TensorCore` — that's the wrapper we're bypassing

## Risks (acknowledged up front)

1. **`ld_matrix` may not be reachable from user code** in 1.0.0b1. If `_ldmatrix_x4` is private (underscore-prefixed) or not exposed, we fall back to manual lane-striped reads from smem with the per-lane row/col formula. Both work; manual is more verbose but unblocks the wave.
2. **Per-lane fragment layout off-by-one** is the dominant numerical-correctness failure mode. The probe in Wave 20 sidestepped this by not checking output values. Mitigation: numerical check is a hard gate; if `max_err > tolerance`, we debug fragment indexing, not perf.
3. **Bank conflicts in smem load** — the bf16 path may need a different smem layout (8-element vector loads vs 4-element TF32 loads) than Wave 19 used. Mitigation: start with Wave 19's `Layout.row_major(BM, BK)` and adjust ONLY if SASS shows `LDS.U.128` conflicts.
4. **Compile time may explode** with deeper `comptime for` unrolls at MMA_K=16. Mitigation: if compile takes >5 min, drop to BK=16 (one MMA call per pass) and re-evaluate.
5. **Scaffolding complexity could exceed wave budget.** If Tasks 4-7 collectively exceed ~3 hrs of orchestrator wall time without a working numerically-correct kernel, **stop, document the partial state in `BLOCKED.md`, and re-plan as Wave 21.5 with smaller scope** (e.g. single-warp tile to validate epilogue first, then scale up).

## Phases

- **Phase 1 (orchestrator, ~30 min):** scaffold + minimal compile target. Tasks 1-3.
- **Phase 2 (orchestrator, ~60-90 min):** kernel body, fragment loads, MMA loop, epilogue. Tasks 4-7. Test compile-and-run early; numerical correctness later.
- **Phase 3 (orchestrator, ~30 min):** numerical correctness, perf measurement, SASS verification. Tasks 8-10.
- **Phase 4 (cross-model review, async):** spawn 2-3 reviewers in parallel via `delegate_task` for spec compliance + code quality. Phase 8 of deep-work-loop.
- **Phase 5 (orchestrator, ~15 min):** docs (ANALYSIS.md, results/wave21-summary.md, BACKLOG.md update, skill updates). Tasks 11-13.
- **Commit cadence:** one commit per task that produces a runnable artifact (Tasks 3, 7, 8, 9 as separate commits). Final wave-summary commit closes.

---

## Phase 1: Scaffold

### Task 1: Create `mojo-matmul-bf16/` skeleton

**Objective:** Empty cell with the standard 5-file shape (`.mojo`, `run.sh`, `.gitignore`, placeholders for `ANALYSIS.md` + `run.log`).

**Files:**
- Create: `mojo-matmul-bf16/.gitignore` (mirror `mojo-matmul-tc/.gitignore`)
- Create: `mojo-matmul-bf16/run.sh` (mirror `mojo-matmul-tc/run.sh`, change kernel filename)
- Create: `mojo-matmul-bf16/matmul_bf16.mojo` (skeleton: imports + `def main()` with `print("hello")` only)
- Create: `mojo-matmul-bf16/ANALYSIS.md` (one-line "Wave 21 in progress")

**Step 1: Read the reference cell layout**

```bash
ls -la /home/codeseys/cuda-exploration/mojo-matmul-tc/
cat /home/codeseys/cuda-exploration/mojo-matmul-tc/.gitignore
cat /home/codeseys/cuda-exploration/mojo-matmul-tc/run.sh
```

**Step 2: Verify Mojo can find pixi**

```bash
cd /home/codeseys/cuda-exploration/mojo-matmul-bf16
PATH="$HOME/.pixi/bin:$PATH" pixi run --manifest-path ../mojo-workspace/pixi.toml mojo --version
```

Expected: `Mojo 1.0.0b1 (a9591de6)` or compatible.

**Step 3: Compile-and-run the skeleton**

`matmul_bf16.mojo` minimum:
```mojo
from std.sys import has_accelerator
from std.gpu.host import DeviceContext

def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
        return
    with DeviceContext() as ctx:
        print("GPU:", ctx.name())
```

```bash
bash run.sh 2>&1 | tee run.log
```

Expected: `GPU: NVIDIA GeForce RTX 5090`.

**Step 4: Commit**

```bash
git add mojo-matmul-bf16/
git commit -m "Wave 21 Task 1: scaffold mojo-matmul-bf16 cell"
```

### Task 2: Verify probe replication on bigger dispatch

**Objective:** Before scaling to 4096³, replicate the Wave 20 mma_probe finding inside the new cell to confirm no regressions or env-drift since 2026-05-20.

**Files:**
- Modify: `mojo-matmul-bf16/matmul_bf16.mojo` — paste in the Wave 20 probe kernel verbatim, run with `_dump_sass=True`.

**Step 1: Run the probe**

```bash
bash run.sh 2>&1 | tee run.log
grep -c 'HMMA.16816.F32.BF16' run.log
```

Expected: `≥ 1` (Wave 20 produced exactly 1 in the single-warp probe).

**Step 2: If probe fails, STOP** and report. Most likely cause: pixi env drift, mojo version bump, driver change. Do not proceed to Task 3 until the path is confirmed alive.

**No commit** — this is a sanity check, not a deliverable. Move on to Task 3.

### Task 3: Define matmul harness shell (no kernel body yet)

**Objective:** Wire up `DeviceContext`, host-side init, layout tensors, timing harness, and numerical-check stub. Empty kernel body that just zeroes C. Confirms the host-side plumbing for bf16 inputs works before we touch fragments.

**Files:**
- Modify: `mojo-matmul-bf16/matmul_bf16.mojo`

**Algorithmic shape:**
- M=N=K=4096
- a_type=bf16, c_type=f32
- BM=64, BN=64, BK=32, WM=32, WN=32, MMA_M=16, MMA_N=8, MMA_K=16
- NUM_WARPS = (BM/WM) × (BN/WN) = 2×2 = 4 → 128 threads/block

**Critical bf16 init pitfall (from skill #12):**
```mojo
# WRONG -- implicit cast errors:
# a_host[i] = Float32(((i * 2654435761) % 256)) * 0.001
# RIGHT:
a_host[i] = (Float32(((i * 2654435761) % 256)) * 0.001).cast[a_type]()
```

**Empty kernel:**
```mojo
def matmul_bf16_kernel[
    layout_a: Layout, layout_b: Layout, layout_c: Layout,
    BM: Int, BN: Int, BK: Int, WM: Int, WN: Int,
    MMA_M: Int, MMA_N: Int, MMA_K: Int,
](
    A: LayoutTensor[DType.bfloat16, layout_a, MutAnyOrigin],
    B: LayoutTensor[DType.bfloat16, layout_b, MutAnyOrigin],
    C: LayoutTensor[DType.float32, layout_c, MutAnyOrigin],
):
    # Stub: zero my output tile. Real kernel in Task 4-7.
    var C_warp_tile = C.tile[BM, BN](Int(block_idx.y), Int(block_idx.x))
    if Int(thread_idx.x) == 0:
        for i in range(BM):
            for j in range(BN):
                C_warp_tile[i, j] = Float32(0.0)
```

**Step 1: Compile-and-run**

```bash
bash run.sh 2>&1 | tee run.log
```

Expected: runs to completion, prints `[mojo-matmul-bf16] M=N=K=4096 ...` with TFLOPS≈0 (kernel writes zeros).

**Step 2: Commit**

```bash
git add mojo-matmul-bf16/
git commit -m "Wave 21 Task 3: matmul harness shell with bf16 inputs"
```

---

## Phase 2: Hand-rolled MMA kernel

### Task 4: Replace stub with `mma()` call using Wave-19 frag layout

**Objective:** Get a kernel that *runs* with `mma()` calls, even if numerically wrong. Use the simplest possible per-lane fragment indexing — read raw smem with the per-lane formula, no `ld_matrix`. Correctness in Task 5/6.

**Reference: PTX 9.7.13.4.8 m16n8k16 bf16 input distribution**

For matrix A (16 rows × 16 cols), per-lane elements (a_frag[0..7]):
```
groupID = laneid >> 2          # 0..7
tid_in_grp = laneid & 3        # 0..3
For elem i in {0..7}:
    row = groupID + (i & 1) * 8 + (i >> 2) * 0   # rows 0/8 alternating
    # actually for k16: we need to look up the m16n8k16 A-distribution
```

**ACTION:** Before writing this task, retrieve `from std.gpu.compute.arch.mma_nvidia import _mma_nvidia` source via `pixi run mojo doc` or grep the pixi env, OR consult `references/mojo-mma-shapes.md` for the exact per-lane formula. The skill currently documents the C/D distribution in detail; A and B for k=16 require looking at the LLVM intrinsic doc (`llvm.nvvm.mma.m16n8k16.row.col.bf16`) which references PTX ISA 9.7.13.4.6.

**If `_ldmatrix_x4` (or equivalent) is reachable as a public-ish API**, use it — it does the per-lane stripe load in one PTX instruction (`ldmatrix.sync.aligned.m8n8.x4`) and is what `TensorCore.load_a` uses internally. Quick check:
```bash
grep -rn "ldmatrix\|ld_matrix" "$HOME/.pixi/envs"/*/lib/mojo/std/ 2>/dev/null | head -20
```

**Files:**
- Modify: `mojo-matmul-bf16/matmul_bf16.mojo`

**Acceptance for Task 4:** kernel runs without compile errors and without runtime CUDA errors. Numerical output may be wrong; that's fine for now.

**No commit yet** — Task 5 is the partner.

### Task 5: Numerical correctness vs CPU reference at small N

**Objective:** Before scaling to 4096³, validate fragment layout at M=N=K=64 (one block, four warps, one tile pass). CPU reference is cheap and unambiguous at that size.

**Files:**
- Modify: `mojo-matmul-bf16/matmul_bf16.mojo` — add a `--small` mode (or `comptime SMALL_TEST = True` flag) that uses M=N=K=64 with deterministic init (e.g. `A[i,j] = (i+j) % 7 * 0.1`, `B[i,j] = (i*j+1) % 5 * 0.1`).

**CPU reference:**
```mojo
# After running the kernel, copy C back to host. Compute CPU reference
# in f32, compare with combined-tolerance:
var max_err: Float32 = 0.0
var max_rel_err: Float32 = 0.0
for i in range(M):
    for j in range(N):
        var ref: Float32 = 0.0
        for k in range(K):
            ref += a_host[i*K+k].cast[DType.float32]() * b_host[k*N+j].cast[DType.float32]()
        var got = c_host[i*N+j]
        var abs_err = abs(got - ref)
        # Combined tolerance per skill pitfall:
        # |a-b| <= atol + rtol*|b|
        if abs_err > 1.0 + 1e-2 * abs(ref):
            print("MISMATCH at (", i, ",", j, "): got=", got, " ref=", ref)
            return
print("[mojo-matmul-bf16] correctness PASSED at M=N=K=", M)
```

**Acceptance for Task 5:** all 64×64 = 4096 outputs match within tolerance. If they don't, **debug fragment indexing — do not proceed**. The off-by-one in per-lane row/col is the most likely culprit; double-check against the PTX ISA section for both A and B distributions.

**Step 1: Run small-mode**
```bash
COMPTIME_SMALL=1 bash run.sh 2>&1 | tee run.log
grep "correctness PASSED" run.log
```

**Step 2: Commit (only if PASSED)**
```bash
git add mojo-matmul-bf16/
git commit -m "Wave 21 Task 5: numerical correctness validated at M=N=K=64"
```

### Task 6: Scale to M=N=K=256 (multi-block + multi-tile-pass)

**Objective:** At M=N=K=64 we have 1 block × 1 tile pass — minimal stress. At 256, we have 4×4 = 16 blocks and 8 K-tile passes. Validates inter-block and inter-pass accumulation.

**Files:**
- Modify: `mojo-matmul-bf16/matmul_bf16.mojo` — bump SMALL_TEST size to 256.

**Acceptance:** correctness still passes with same tolerance. If it fails, the K-loop accumulation across tile passes is wrong.

**No commit** if 64 passed and 256 passes — single commit at end of Task 6.

### Task 7: Scale to M=N=K=4096 + numerical check vs cuBLAS hgemm baseline

**Objective:** Full size, full-input deterministic matrices. Numerical check at this size requires a vendored reference (CPU is too slow). Use `vendor_blas.matmul` or a sampled-correctness check (random subset of 1000 output cells).

**Files:**
- Modify: `mojo-matmul-bf16/matmul_bf16.mojo` — make M=N=K=4096 the default, gate `--small` behind a comptime flag.

**Sampled correctness:** compute CPU reference for 1000 random `(i,j)` pairs (1000 × K = 4M flops × 32-bit FMA, ~few seconds on CPU), compare to GPU output with combined tolerance.

**Acceptance:**
- All 1000 samples pass tolerance → numerical correctness CONFIRMED
- If `vendor_blas` is exposed in Mojo 1.0.0b1, use it instead for full-tensor check

**Step 1: Compile-and-run**
```bash
bash run.sh 2>&1 | tee run.log
grep "TFLOPS" run.log
grep "correctness" run.log
```

**Step 2: Commit**
```bash
git add mojo-matmul-bf16/
git commit -m "Wave 21 Task 7: full-size 4096x4096 bf16 matmul w/ correctness check"
```

---

## Phase 3: Performance + SASS evidence

### Task 8: Performance measurement (10 iters, median)

**Objective:** Capture canonical TFLOPS number at 4096³, mirroring Wave 19's harness.

**Files:** `mojo-matmul-bf16/matmul_bf16.mojo` — already has timing from Task 3 stub. Just rerun on the final kernel.

**Step 1: Run with idle GPU (close other apps)**
```bash
nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv > /tmp/gpu-pre.txt
bash run.sh 2>&1 | tee run.log
nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv > /tmp/gpu-post.txt
```

**Acceptance:** TFLOPS in the 100-130 range. If outside range:
- < 100 TF: investigate. Likely smem bank conflicts (check SASS for LDS.U.32 vs LDS.U.128) or insufficient unroll.
- > 130 TF: too good — sanity-check correctness wasn't compromised. Re-run Task 7 numerical check.
- < 60 TF: something is broken. The wrapper's TF32 path was 55.5 TF; bf16 should be at least 1.8× that (since bf16 m16n8k16 is k=16 vs TF32's k=4 → 4× the flops/MMA but fewer MMAs needed).

**Step 2: Commit**
```bash
git add mojo-matmul-bf16/run.log mojo-matmul-bf16/results.txt
git commit -m "Wave 21 Task 8: 4096x4096 bf16 matmul perf @ <TFLOPS>"
```

### Task 9: SASS extraction + verification

**Objective:** Confirm the kernel emits the expected `HMMA.16816.F32.BF16` instruction count and uses `cp.async` (NOT TMA). This is the Wave 21 evidence artifact analogous to Wave 19's `matmul_tc.sass`.

**Files:**
- Create: `mojo-matmul-bf16/matmul_bf16.sass`

**Step 1: Add `_dump_sass=True` to enqueue_function call (already enabled per Task 4 reference). Re-run, capture SASS to file:**

```bash
bash run.sh 2>&1 | tee run.log
# extract SASS section
sed -n '/Function : .*matmul_bf16_kernel/,/^Function :/p' run.log > matmul_bf16.sass || \
sed -n '/SASS/,/done/p' run.log > matmul_bf16.sass
```

**Step 2: Quantitative checks**

```bash
# Expected counts:
# - HMMA.16816.F32.BF16: count should match 8 MMAs/warp × 4 warps/block × 2 K-tile-passes × (BM/WM × BN/WN) blocks
#   ~= 8 × 4 × 2 = 64 per block-pass, but unrolled total is the metric
grep -c 'HMMA.16816.F32.BF16' matmul_bf16.sass
grep -c 'LDGSTS\|LDG.E\|LDS.U' matmul_bf16.sass
grep -c 'UTMALDG' matmul_bf16.sass  # Should be 0 — Mojo doesn't auto-emit TMA
grep -c '.target sm_120' matmul_bf16.sass  # Should be 1 (.target sm_120a)
```

**Acceptance:**
- `HMMA.16816.F32.BF16` count > 0 (proves bf16 TC path engaged at scale)
- `UTMALDG` count = 0 (confirms Mojo `cp.async` path, gap explanation)
- `.target sm_120a` present (correct arch)

**Step 3: Commit**
```bash
git add mojo-matmul-bf16/matmul_bf16.sass
git commit -m "Wave 21 Task 9: SASS evidence — N HMMA.16816.F32.BF16 instructions"
```

### Task 10: Re-bench cuTile bf16 + cuBLAS hgemm on the same idle GPU

**Objective:** Per the rust-gpu-compute skill: when adding a new frontend lane, **always re-run the comparison baselines on the same idle GPU in the same session** to prevent thermal/power-state drift from inflating or deflating apparent gaps.

**Files:**
- Re-run: `cutile-matmul-tiled-mixed/` bf16 cell + `cublas-half-precision/` hgemm cell
- Update: `cutile-matmul-tiled-mixed/results.csv` (or wherever its perf log lives) AND `cublas-half-precision/results.csv` IF either drifts >5% from the existing committed numbers
- Commit baseline updates **separately** so the comparison commit is reproducible.

**Step 1: Find existing baseline harnesses**
```bash
find /home/codeseys/cuda-exploration -name 'results.csv' -path '*cutile*' -o -name 'results.csv' -path '*cublas*' 2>/dev/null
```

**Step 2: Re-run each, compare to existing, update if drifted**

**Step 3: Commit baseline rerun (if updated)**
```bash
git add <files-that-drifted>
git commit -m "Wave 21 Task 10: re-bench bf16 baselines on same idle GPU"
```

---

## Phase 4: Cross-model review (deep-work-loop Phase 8)

### Task 11: Spawn 2-3 parallel reviewers

**Objective:** Independent verification per `requesting-code-review` and `parallel-critique` skills. Different models = different reading paths = orthogonal findings.

**Reviewers (via `delegate_task` tasks array, parallel):**

1. **R1 — Spec compliance reviewer** (mid-tier model). Reads the wave plan + the ANALYSIS.md draft + the kernel + run.log. Asks: do the claimed numbers match the run.log? Does the SASS evidence support the headline?
2. **R2 — Code quality reviewer** (mid-tier model, different family). Reads only the `.mojo` file. Asks: is the fragment math right? Are there obvious bank-conflict opportunities? Is the smem layout sound?
3. **R3 — Numerical-correctness reviewer** (small model OK). Reads the correctness-check code only. Asks: is the tolerance reasonable? Is the sample size adequate? Does the comparison handle edge cases (denormals, NaN propagation)?

**Each reviewer gets:**
- Goal: focused single-axis review (one of the three above)
- Context: the wave plan, file paths to read, what to look for, what NOT to look at
- Toolsets: ['file', 'terminal'] minimum

**Step 1: Dispatch in parallel**

(See `delegate_task` tasks array — single tool call with 3 entries.)

**Step 2: Synthesize findings**

Each reviewer writes a numbered list of issues. Orchestrator looks for **intersection** (≥2 reviewers flagging same issue → high-priority) vs **union** (single-reviewer findings → log but lower-priority).

**Step 3: Address blocker findings before Task 12.** Non-blockers go in BACKLOG.md as Wave 22 candidates.

---

## Phase 5: Documentation + skill updates

### Task 12: Wave 21 summary + ANALYSIS.md

**Files:**
- Create: `results/wave21-summary.md` (mirror `wave19-summary.md` and `wave20-summary.md` shape)
- Update: `mojo-matmul-bf16/ANALYSIS.md` (full details)
- Update: `BACKLOG.md` (mark Wave 21 SHIPPED, add any Wave 22 candidates)

**Wave 21 summary content:**
1. Headline finding (TFLOPS at 4096³, % of cuTile, % of cuBLAS hgemm)
2. SASS evidence (HMMA count, no UTMALDG, sm_120a target)
3. Numerical correctness summary (tolerance, sample count, max-error observed)
4. Comparison table updated with Mojo bf16 row alongside Wave 19's Mojo TF32 row
5. Pitfalls discovered (any new ones — most likely fragment-layout bugs surfaced during Task 5)
6. Wave 22 candidates (TMA loads via `cp_async_bulk`, FP8 lane, attention follow-up, etc.)

### Task 13: Update `mlops/rust-gpu-compute` skill

**Files:**
- Update: `~/.hermes/skills/mlops/rust-gpu-compute/SKILL.md` — add Wave 21 row to the Mojo perf table; expand pitfall #11 if any new fragment-layout gotchas surfaced.
- Update: `references/mojo-mma-shapes.md` — fill in any A/B distribution detail that was missing for k=16 (the skill currently only documents C/D distribution in detail).
- Update: `references/mojo-tensor-core.md` — note the bf16 path is now harness-proven, not just probe-proven; cross-reference Wave 21.

**Step 1: Use `skill_manage(action='patch')`** for surgical edits, not `action='edit'` (full rewrite).

**Step 2: Final commit**
```bash
git add results/wave21-summary.md mojo-matmul-bf16/ANALYSIS.md BACKLOG.md
git commit -m "Wave 21: Mojo bf16-in/f32-acc tiled matmul shipped — <TFLOPS> at 4096^3"
```

---

## Acceptance criteria (whole wave)

The wave SHIPS only if all of these are true:

- [ ] `mojo-matmul-bf16/matmul_bf16.mojo` runs end-to-end on RTX 5090 sm_120
- [ ] Numerical correctness passes at M=N=K=64 (full CPU ref) AND M=N=K=4096 (sampled)
- [ ] Performance reported as median of 10 iters, in 60-130 TFLOPS range
- [ ] `matmul_bf16.sass` shows `HMMA.16816.F32.BF16 > 0` and `UTMALDG = 0`
- [ ] Comparison row added to `results/wave21-summary.md`
- [ ] cuTile + cuBLAS bf16 baselines re-run on same session, drift <5% (or updated)
- [ ] At least 2 of 3 cross-model reviewers approve (no blocker findings open)
- [ ] BACKLOG.md updated with Wave 22 candidates
- [ ] Skill updates committed to `~/.hermes/skills/mlops/rust-gpu-compute/`

## Out of scope (deferred to Wave 22+)

- TMA loads (`cp_async_bulk`) — would close the cp.async gap to cuTile
- FP8 lane (e4m3 / e5m2) at m16n8k32
- f16-in/f32-acc lane (separate dispatch but same scaffolding)
- Numerical correctness vs full `vendor_blas.matmul` (vs sampled CPU)
- Bank-conflict-free smem layout
- Attention column with bf16 (Phase D of Wave 18 lineage)

## Budget

- **Token budget:** ~150-200k for orchestrator (mostly skill loading + reading reference kernels). Reviewers: ~30k each × 3 = 90k.
- **Wall clock:** 2-3 hrs realistic if Task 5 numerical correctness lands first try; 4-6 hrs if fragment-layout debug burns a cycle.
- **Stop condition:** if Phase 2 exceeds 3 hrs without numerical correctness, declare blocked, write BLOCKED.md, defer.
