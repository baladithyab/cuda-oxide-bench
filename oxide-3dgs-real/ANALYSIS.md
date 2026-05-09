# oxide-3dgs-real — Wave 9

## What this is

Renders a **real, public 3D Gaussian Splatting scene** through the existing
`oxide-3dgs-mini` kernel (a 2D forward rasterizer in cuda-oxide). The kernel
itself is *unchanged*. All 3D→2D work (quaternion-to-rotation, covariance
construction, perspective Jacobian, conic inversion, SH-DC-to-RGB, sigmoid
opacity, depth sort) happens host-side in Rust before the kernel launch.

## Scene

- **URL**: `https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/luigi/luigi.ply`
- **File size**: 988,183 bytes (≈ 966 KB)
- **Gaussian count**: 14,526
- **Properties**: `x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3` (17 floats/vertex = **SH degree 0 only**, no `f_rest_*`)
- **Scene bbox**: `min=(-0.55,-0.66,-0.27)  max=(0.54,0.65,0.27)`, diag ≈ 1.79
- **Centroid**: `(-0.002, -0.005, -0.017)` — already normalised to origin

This is an object-scale asset (not a room scan), which is ideal: no floor,
no distractors, tight frustum.

## Camera pose used (winning render: `camC_flipY`)

- Intrinsics: `fx = fy = 800`, `cx = cy = 400` (for 800×800 render)
- World→camera rotation `W = diag(1, -1, 1)` (flip Y axis — PLY appears to
  be y-down while our projection expects y-up in image coords)
- Translation `t = (-cx_centroid, +cy_centroid, -(cz_centroid - 1.5·diag))`
  ⇒ camera placed at ≈ 2.7 world units from centroid along world -Z.
- Depth range in camera frame: **`[2.43, 2.97]`** — entire scene at a nearly
  constant depth, confirming the camera is far enough that the object is
  fully enclosed in a narrow depth slab.
- Projected 2D means range: `x∈[233, 564]`, `y∈[211, 602]` — well inside the
  800×800 image, centred-ish, with plenty of headroom on all sides.

## Sanity checks (all passed)

- Gaussians total: 14,526; projected: **14,526** (0 culled). Expected for an
  object scene entirely in front of the camera.
- Conic scale median ≈ 53 → implies a typical σ² of ≈ 1/53 ≈ 0.019 in pixel
  units squared, i.e. splats of radius ≈ 0.14 px before the 0.3-anti-alias
  blur we add. The anti-alias blur dominates visible footprint — this is
  normal for a sub-megapixel distilled splat.
- Non-zero-pixel fraction: **9.6 %** (61,431 / 640,000). A figurine occupies
  a compact bounding box; this is in the expected range.
- Bounding box of non-zero output: x∈[213, 582], y∈[193, 619] — the rendered
  subject is ~370×420 px centered in the image. Consistent with projected
  2D-means range above.
- No NaNs, no all-black, no all-one-color: SH +0.5 shift and sigmoid-opacity
  are both visibly correct (colour cast matches a green/overalls-red Luigi).

## Timing

Measured with `cuEventRecord` / `elapsed_ms` on RTX 5090 (sm_120), one timed
launch per camera after a warmup launch, 14,526 gaussians, 800×800 grid:

| Camera               | gpu_ms | cpu_wall_ms |
|---------------------:|-------:|------------:|
| A (identity, -Z)     | 9.60   | 10.31       |
| C (flipY, -Z)        | 11.11  | 11.43       |
| D (Y-rot 180°, +Z)   | 9.58   | 9.87        |

Render time is dominated by the kernel's O(pixels × N) inner loop. 800×800
pixels × 14,526 gaussians ≈ 9.3 G inner iterations; 10 ms ⇒ ~930 Giter/s,
which is in the right ballpark for this naive untiled rasterizer.

## Final visual verdict

**Recognizable.** An ASCII-downsampled view of the output shows a clearly
humanoid silhouette: a vertical "cap + head" structure at top (two blobs
side-by-side: the hat crown + a small cranial highlight), spread-out arms
across the middle, torso, and two legs with feet at the bottom. Camera A
rendered Luigi upside-down (confirming a y-axis convention mismatch);
camera C with the Y-flip produced a right-side-up figurine.

## Failure modes observed and diagnosed

- **Camera B (`+Z no flip`)**: all 14,526 gaussians were culled as
  "behind camera" — expected, because it places the camera on the opposite
  side of the object; object ends up with negative camera-z. Not a bug,
  the cull worked.
- No conic-degeneracy failures (`culled_bad_cov=0`) — the 0.3 anti-alias
  blur is doing its job keeping Σ_2d positive-definite.

## Files

- `scenes/luigi.ply` — *gitignored* (988 KB is small but the policy is to
  not commit binary scene data; document the URL instead).
- `src/main.rs` — parser + projection + render driver (420 lines).
- `output_real_{A,C,D}.ppm` — raw PPM outputs (ignored).
- `/tmp/real_3dgs.png` — canonical PNG for handoff = `output_real_C.ppm`.

## Unchanged pieces (validates kernel reuse)

The kernel `rasterize_2dgs` is byte-identical to the one in
`oxide-3dgs-mini` — same function body, same signature (2D means, 3-float
conic, opacity, per-gaussian RGB). Only the host marshalling changed. This
confirms the Wave-8 kernel is general enough to render real projected
3DGS data with no kernel-side modification.
