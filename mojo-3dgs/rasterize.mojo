# Wave 23.1 -- mojo-3dgs (5th frontend port of 3DGS rasterizer).
#
# Naive per-pixel-iter-over-all-gaussians 3DGS rasterizer in Mojo. Mirrors
# cuda-3dgs-real / oxide-3dgs-real / cutile-3dgs-real algorithm exactly:
#
#   For each pixel (px, py):
#     T = 1.0; r = g = b = 0.0
#     For each gaussian i (depth-ascending):
#       dx = px - mx[i]; dy = py - my[i]
#       power = -0.5 * (cxx*dx*dx + 2*cxy*dx*dy + cyy*dy*dy)
#       if power > 0: skip
#       alpha = min(opacity[i] * exp(power), 0.99)
#       if alpha < 1/255: skip
#       w = alpha * T
#       r += w * cr[i]; g += w * cg[i]; b += w * cb[i]
#       T *= (1 - alpha)
#       if T < 1e-4: break  -- early-out for perf
#
# Block: 16x16 threads (one per pixel of a 16x16 tile).
# Grid:  (W/16, H/16) = (50, 50).
#
# Host side: PLY parse + projection + SH3 evaluation + depth sort are
# pre-staged into a flat binary blob by prep.py. Mojo loads the blob,
# H2Ds it, runs the kernel, D2Hs the output, and writes a PPM.
#
# Blob layout (LE):
#   [u32 n_proj]
#   [n_proj * f32 mx]  [n_proj * f32 my]
#   [n_proj * f32 cxx] [n_proj * f32 cxy] [n_proj * f32 cyy]
#   [n_proj * f32 opacity]
#   [n_proj * f32 r]   [n_proj * f32 g]   [n_proj * f32 b]
#
# Output: output_utsuho_plush_A.ppm  (P6, 800x800, u8 RGB).

from std.math import ceildiv, exp
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.utils.numerics import bitcast


comptime float_dtype = DType.float32
comptime W: Int = 800
comptime H: Int = 800
comptime BS: Int = 16


# ============================================================================
# Kernel: naive per-pixel iter over ALL gaussians.
# ============================================================================
def rasterize_kernel(
    mx_ptr:  UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    my_ptr:  UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    cxx_ptr: UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    cxy_ptr: UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    cyy_ptr: UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    op_ptr:  UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    cr_ptr:  UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    cg_ptr:  UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    cb_ptr:  UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    out_r:   UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    out_g:   UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    out_b:   UnsafePointer[Scalar[float_dtype], MutAnyOrigin],
    n_gauss: Int,
):
    var px = Int(block_idx.x * block_dim.x + thread_idx.x)
    var py = Int(block_idx.y * block_dim.y + thread_idx.y)
    if px >= W or py >= H:
        return

    var pxf: Float32 = Float32(px)
    var pyf: Float32 = Float32(py)
    var transmittance: Float32 = 1.0
    var accum_r: Float32 = 0.0
    var accum_g: Float32 = 0.0
    var accum_b: Float32 = 0.0
    var ALPHA_FLOOR: Float32 = 1.0 / 255.0
    var ALPHA_CAP:   Float32 = 0.99
    var T_FLOOR:     Float32 = 1e-4

    var i: Int = 0
    while i < n_gauss:
        var dx: Float32 = pxf - mx_ptr[i]
        var dy: Float32 = pyf - my_ptr[i]
        var cxx: Float32 = cxx_ptr[i]
        var cxy: Float32 = cxy_ptr[i]
        var cyy: Float32 = cyy_ptr[i]
        var power: Float32 = -0.5 * (cxx * dx * dx + 2.0 * cxy * dx * dy + cyy * dy * dy)
        if power <= 0.0:
            var alpha_raw: Float32 = op_ptr[i] * exp(power)
            var alpha: Float32 = alpha_raw
            if alpha > ALPHA_CAP:
                alpha = ALPHA_CAP
            if alpha >= ALPHA_FLOOR:
                var w: Float32 = alpha * transmittance
                accum_r = accum_r + w * cr_ptr[i]
                accum_g = accum_g + w * cg_ptr[i]
                accum_b = accum_b + w * cb_ptr[i]
                transmittance = transmittance * (1.0 - alpha)
                if transmittance < T_FLOOR:
                    break
        i = i + 1

    var idx: Int = py * W + px
    out_r[idx] = accum_r
    out_g[idx] = accum_g
    out_b[idx] = accum_b


# ============================================================================
# Helpers: bytes -> u32 / f32 (little-endian).
# ============================================================================
fn read_u32_le(bytes: List[UInt8], off: Int) -> UInt32:
    return (UInt32(Int(bytes[off]))
          | (UInt32(Int(bytes[off + 1])) << 8)
          | (UInt32(Int(bytes[off + 2])) << 16)
          | (UInt32(Int(bytes[off + 3])) << 24))


fn read_f32_le(bytes: List[UInt8], off: Int) -> Float32:
    return bitcast[DType.float32](read_u32_le(bytes, off))


# ============================================================================
# u8 quantization for PPM (round-half-up; matches cuda-3dgs-real within
# the ≤2 u8 acceptance envelope).
# ============================================================================
fn quantize_u8(v: Float32) -> UInt8:
    var x: Float32 = v
    if x < 0.0:
        x = 0.0
    if x > 1.0:
        x = 1.0
    var s: Float32 = x * 255.0
    var r: Int = Int(s + 0.5)
    if r < 0:
        r = 0
    if r > 255:
        r = 255
    return UInt8(r)


# ============================================================================
# Main
# ============================================================================
def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
        return

    print("=== mojo-3dgs (Wave 23.1) ===")
    print("image: ", W, "x", H, "  BS=", BS)

    # ── Load blob ───────────────────────────────────────────────────────────
    var blob_path: String = "cam_A.bin"
    print("Loading blob:", blob_path)
    var blob_f = open(blob_path, "r")
    var bytes = blob_f.read_bytes()
    blob_f.close()

    var n: Int = Int(read_u32_le(bytes, 0))
    var expected_size: Int = 4 + n * 9 * 4
    if len(bytes) != expected_size:
        print("blob size mismatch: got", len(bytes), "expected", expected_size)
        raise Error("blob size mismatch")
    print("n_proj =", n)

    with DeviceContext() as ctx:
        print("GPU:", ctx.name())

        # Host buffers (one per channel)
        var mx_h  = ctx.enqueue_create_host_buffer[float_dtype](n)
        var my_h  = ctx.enqueue_create_host_buffer[float_dtype](n)
        var cxx_h = ctx.enqueue_create_host_buffer[float_dtype](n)
        var cxy_h = ctx.enqueue_create_host_buffer[float_dtype](n)
        var cyy_h = ctx.enqueue_create_host_buffer[float_dtype](n)
        var op_h  = ctx.enqueue_create_host_buffer[float_dtype](n)
        var cr_h  = ctx.enqueue_create_host_buffer[float_dtype](n)
        var cg_h  = ctx.enqueue_create_host_buffer[float_dtype](n)
        var cb_h  = ctx.enqueue_create_host_buffer[float_dtype](n)
        ctx.synchronize()

        # Decode 9 channels in order: mx, my, cxx, cxy, cyy, opacity, r, g, b.
        var ch_off: Int = 4   # past the n_proj u32
        var k: Int = 0
        while k < n:
            mx_h[k]  = read_f32_le(bytes, ch_off + 0 * n * 4 + k * 4)
            my_h[k]  = read_f32_le(bytes, ch_off + 1 * n * 4 + k * 4)
            cxx_h[k] = read_f32_le(bytes, ch_off + 2 * n * 4 + k * 4)
            cxy_h[k] = read_f32_le(bytes, ch_off + 3 * n * 4 + k * 4)
            cyy_h[k] = read_f32_le(bytes, ch_off + 4 * n * 4 + k * 4)
            op_h[k]  = read_f32_le(bytes, ch_off + 5 * n * 4 + k * 4)
            cr_h[k]  = read_f32_le(bytes, ch_off + 6 * n * 4 + k * 4)
            cg_h[k]  = read_f32_le(bytes, ch_off + 7 * n * 4 + k * 4)
            cb_h[k]  = read_f32_le(bytes, ch_off + 8 * n * 4 + k * 4)
            k = k + 1

        # Device buffers (gauss inputs + 3 output planes).
        var mx_d  = ctx.enqueue_create_buffer[float_dtype](n)
        var my_d  = ctx.enqueue_create_buffer[float_dtype](n)
        var cxx_d = ctx.enqueue_create_buffer[float_dtype](n)
        var cxy_d = ctx.enqueue_create_buffer[float_dtype](n)
        var cyy_d = ctx.enqueue_create_buffer[float_dtype](n)
        var op_d  = ctx.enqueue_create_buffer[float_dtype](n)
        var cr_d  = ctx.enqueue_create_buffer[float_dtype](n)
        var cg_d  = ctx.enqueue_create_buffer[float_dtype](n)
        var cb_d  = ctx.enqueue_create_buffer[float_dtype](n)
        var or_d  = ctx.enqueue_create_buffer[float_dtype](W * H)
        var og_d  = ctx.enqueue_create_buffer[float_dtype](W * H)
        var ob_d  = ctx.enqueue_create_buffer[float_dtype](W * H)

        ctx.enqueue_copy(dst_buf=mx_d,  src_buf=mx_h)
        ctx.enqueue_copy(dst_buf=my_d,  src_buf=my_h)
        ctx.enqueue_copy(dst_buf=cxx_d, src_buf=cxx_h)
        ctx.enqueue_copy(dst_buf=cxy_d, src_buf=cxy_h)
        ctx.enqueue_copy(dst_buf=cyy_d, src_buf=cyy_h)
        ctx.enqueue_copy(dst_buf=op_d,  src_buf=op_h)
        ctx.enqueue_copy(dst_buf=cr_d,  src_buf=cr_h)
        ctx.enqueue_copy(dst_buf=cg_d,  src_buf=cg_h)
        ctx.enqueue_copy(dst_buf=cb_d,  src_buf=cb_h)
        ctx.synchronize()

        var grid_x = ceildiv(W, BS)
        var grid_y = ceildiv(H, BS)

        # Capture pointers for closure use.
        var p_mx = mx_d.unsafe_ptr()
        var p_my = my_d.unsafe_ptr()
        var p_cxx = cxx_d.unsafe_ptr()
        var p_cxy = cxy_d.unsafe_ptr()
        var p_cyy = cyy_d.unsafe_ptr()
        var p_op = op_d.unsafe_ptr()
        var p_cr = cr_d.unsafe_ptr()
        var p_cg = cg_d.unsafe_ptr()
        var p_cb = cb_d.unsafe_ptr()
        var p_or = or_d.unsafe_ptr()
        var p_og = og_d.unsafe_ptr()
        var p_ob = ob_d.unsafe_ptr()

        # Warmup
        ctx.enqueue_function[rasterize_kernel, rasterize_kernel](
            p_mx, p_my, p_cxx, p_cxy, p_cyy, p_op, p_cr, p_cg, p_cb,
            p_or, p_og, p_ob, n,
            grid_dim=(grid_x, grid_y), block_dim=(BS, BS),
        )
        ctx.synchronize()

        # Single timed iter (correctness path).
        @parameter
        def body(ctx: DeviceContext) raises -> None:
            ctx.enqueue_function[rasterize_kernel, rasterize_kernel](
                p_mx, p_my, p_cxx, p_cxy, p_cyy, p_op, p_cr, p_cg, p_cb,
                p_or, p_og, p_ob, n,
                grid_dim=(grid_x, grid_y), block_dim=(BS, BS),
            )

        var elapsed_ns = ctx.execution_time[body](1)
        var ms: Float64 = Float64(elapsed_ns) / 1e6
        print("kernel time (cam A, 1 iter):", ms, "ms")

        # D2H output planes.
        var or_h = ctx.enqueue_create_host_buffer[float_dtype](W * H)
        var og_h = ctx.enqueue_create_host_buffer[float_dtype](W * H)
        var ob_h = ctx.enqueue_create_host_buffer[float_dtype](W * H)
        ctx.enqueue_copy(dst_buf=or_h, src_buf=or_d)
        ctx.enqueue_copy(dst_buf=og_h, src_buf=og_d)
        ctx.enqueue_copy(dst_buf=ob_h, src_buf=ob_d)
        ctx.synchronize()

        # Sanity stats.
        var nz: Int = 0
        var p: Int = 0
        var npx: Int = W * H
        while p < npx:
            var rr = Float32(or_h[p])
            var gg = Float32(og_h[p])
            var bb = Float32(ob_h[p])
            if rr > 0.01 or gg > 0.01 or bb > 0.01:
                nz = nz + 1
            p = p + 1
        print("nonzero pixels (>0.01):", nz, "/", npx)

        # ── Save PPM ────────────────────────────────────────────────────────
        var ppm_path: String = "output_utsuho_plush_A.ppm"
        var ppm_bytes = List[UInt8]()
        ppm_bytes.reserve(64 + npx * 3)

        # Header: "P6\n800 800\n255\n" (raw bytes; image is fixed-size 800x800).
        ppm_bytes.append(UInt8(0x50))   # P
        ppm_bytes.append(UInt8(0x36))   # 6
        ppm_bytes.append(UInt8(0x0A))   # \n
        ppm_bytes.append(UInt8(0x38))   # 8
        ppm_bytes.append(UInt8(0x30))   # 0
        ppm_bytes.append(UInt8(0x30))   # 0
        ppm_bytes.append(UInt8(0x20))   # space
        ppm_bytes.append(UInt8(0x38))   # 8
        ppm_bytes.append(UInt8(0x30))   # 0
        ppm_bytes.append(UInt8(0x30))   # 0
        ppm_bytes.append(UInt8(0x0A))   # \n
        ppm_bytes.append(UInt8(0x32))   # 2
        ppm_bytes.append(UInt8(0x35))   # 5
        ppm_bytes.append(UInt8(0x35))   # 5
        ppm_bytes.append(UInt8(0x0A))   # \n

        var q: Int = 0
        while q < npx:
            ppm_bytes.append(quantize_u8(Float32(or_h[q])))
            ppm_bytes.append(quantize_u8(Float32(og_h[q])))
            ppm_bytes.append(quantize_u8(Float32(ob_h[q])))
            q = q + 1

        var ppm_f = open(ppm_path, "w")
        ppm_f.write_bytes(ppm_bytes)
        ppm_f.close()
        print("wrote", ppm_path, "bytes=", len(ppm_bytes))
