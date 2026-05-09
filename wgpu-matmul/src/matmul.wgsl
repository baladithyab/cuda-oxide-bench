// Naive matmul: each thread computes one output element C[r,c] = sum_k A[r,k] * B[k,c]
@group(0) @binding(0) var<storage, read>       a : array<f32>;
@group(0) @binding(1) var<storage, read>       b : array<f32>;
@group(0) @binding(2) var<storage, read_write> c : array<f32>;
@group(0) @binding(3) var<uniform>             dim : u32;

@compute @workgroup_size(16, 16, 1)
fn matmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= dim || col >= dim) { return; }
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < dim; k = k + 1u) {
        acc = acc + a[row * dim + k] * b[k * dim + col];
    }
    c[row * dim + col] = acc;
}
