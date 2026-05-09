// wgpu naive matmul benchmark: C = A * B for f32 NxN matrices
// Compares against cuda-oxide and raw CUDA on the same shape.
//
// Methodology:
//  - 1 warmup, 5 timed iterations, report best + median
//  - Uses timestamp queries for GPU-side timing (excludes CPU dispatch overhead)
//  - Naive O(N^3) algorithm: each thread computes one output element

use std::time::Instant;
use wgpu::util::DeviceExt;

const N: u32 = 4096; // 4096x4096 matmul -> 2 * N^3 flops = 137.4 GFLOPs/iter

fn main() {
    pollster::block_on(run());
}

async fn run() {
    // On WSL the NVIDIA GPU is reachable only via DX12 (libd3d12 + /dev/dxg).
    // Vulkan ICDs only see Mesa llvmpipe (CPU). DX12 backend on Linux/WSL needs
    // a libd3d12core/dxcore wiring that wgpu doesn't ship out of the box, so in
    // practice on this machine the only working backend is Vulkan -> llvmpipe.
    //
    // We keep the run going on CPU so we have *some* number for the wgpu/WGSL
    // path, and so we can compare what cuda-oxide buys you (real GPU access)
    // vs the cross-vendor stack on a WSL host.
    let backends = wgpu::Backends::all();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        ..Default::default()
    });

    let mut chosen: Option<wgpu::Adapter> = None;
    let mut fallback: Option<wgpu::Adapter> = None;
    for ad in instance.enumerate_adapters(backends) {
        let info = ad.get_info();
        println!(
            "[wgpu] candidate: {} ({:?}, type={:?})",
            info.name, info.backend, info.device_type
        );
        if info.device_type != wgpu::DeviceType::Cpu && chosen.is_none() {
            chosen = Some(ad);
        } else if fallback.is_none() {
            fallback = Some(ad);
        }
    }
    let adapter = chosen.or(fallback).expect("no adapter at all");
    let info = adapter.get_info();
    println!("[wgpu] using: {} ({:?}, type={:?})", info.name, info.backend, info.device_type);
    if info.device_type == wgpu::DeviceType::Cpu {
        println!("[wgpu] !! WARNING: only CPU adapter available (WSL Vulkan limitation). Numbers below are CPU-side, not real GPU.");
    }

    let limits = adapter.limits();
    println!(
        "[wgpu] max_storage_buffer_binding_size = {} MiB",
        limits.max_storage_buffer_binding_size / (1024 * 1024)
    );
    let need = (4096u64 * 4096 * 4) as u32;
    let mut limits = limits;
    if limits.max_storage_buffer_binding_size < need {
        // Try requesting the larger limit; some adapters allow it even if the
        // reported default is small.
        limits.max_storage_buffer_binding_size = need;
        limits.max_buffer_size = limits.max_buffer_size.max(need as u64);
        println!("[wgpu] requesting max_storage_buffer_binding_size={} MiB", need / (1024*1024));
    }
    let features_avail = adapter.features();
    let want = wgpu::Features::TIMESTAMP_QUERY;
    let req_features = if features_avail.contains(want) { want } else { wgpu::Features::empty() };
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: req_features,
                required_limits: limits,
                ..Default::default()
            },
            None,
        )
        .await
        .expect("device");
    let has_ts = req_features.contains(wgpu::Features::TIMESTAMP_QUERY);
    println!("[wgpu] timestamp_query feature: {}", has_ts);

    let n = N as usize;
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.01).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.01).collect();

    let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("A"),
        contents: bytemuck::cast_slice(&a),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("B"),
        contents: bytemuck::cast_slice(&b),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_c = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("C"),
        size: (n * n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buf_n = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dim"),
        contents: bytemuck::cast_slice(&[N]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("matmul"),
        source: wgpu::ShaderSource::Wgsl(include_str!("matmul.wgsl").into()),
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "matmul",
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_c.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_n.as_entire_binding() },
        ],
    });

    // Timestamp query setup
    let ts_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("ts"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let ts_resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ts_resolve"),
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let ts_read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ts_read"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Workgroup is 16x16 -> dispatch (N/16, N/16, 1)
    let wg = (N + 15) / 16;
    let total_flops = 2.0_f64 * (n as f64).powi(3);
    println!("[wgpu] matmul {N}x{N} f32, {:.2} GFLOP/iter", total_flops / 1e9);

    let mut times_ms: Vec<f64> = Vec::new();
    let period = if has_ts { queue.get_timestamp_period() as f64 } else { 0.0 };

    for iter in 0..6 {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let timestamp_writes = if has_ts {
                Some(wgpu::ComputePassTimestampWrites {
                    query_set: &ts_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                })
            } else { None };
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg, wg, 1);
        }
        if has_ts {
            enc.resolve_query_set(&ts_set, 0..2, &ts_resolve, 0);
            enc.copy_buffer_to_buffer(&ts_resolve, 0, &ts_read, 0, 16);
        }
        let cpu_start = Instant::now();
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);
        let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

        let gpu_ms = if has_ts {
            let slice = ts_read.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let ts: &[u64] = bytemuck::cast_slice(&data);
            let gpu_ns = (ts[1].wrapping_sub(ts[0])) as f64 * period;
            drop(data);
            ts_read.unmap();
            gpu_ns / 1e6
        } else {
            cpu_ms
        };

        let label = if iter == 0 { "warmup" } else { "iter" };
        let tflops = (total_flops / 1e12) / (gpu_ms / 1000.0);
        let ts_label = if has_ts { "gpu_ts" } else { "cpu_wall" };
        println!("[wgpu] {label} {iter}: {ts_label}={gpu_ms:.2} ms ({tflops:.3} TFLOPS)  cpu_wall={cpu_ms:.2} ms");
        if iter > 0 { times_ms.push(gpu_ms); }
    }

    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let best = times_ms[0];
    let median = times_ms[times_ms.len() / 2];
    let best_tf = (total_flops / 1e12) / (best / 1000.0);
    let med_tf = (total_flops / 1e12) / (median / 1000.0);
    println!("\n[wgpu] BEST   {best:.2} ms  {best_tf:.3} TFLOPS");
    println!("[wgpu] MEDIAN {median:.2} ms  {med_tf:.3} TFLOPS");
}
