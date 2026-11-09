// Benchmark comparing CubeCL radix sort vs wgpu_sort
//
// Both implementations use the same underlying GPU but through different wgpu versions.

use std::num::NonZeroU32;
use std::time::{Duration, Instant};

use cubecl::client::ComputeClient;
use cubecl::prelude::*;

use wgpu_sort::utils::guess_workgroup_size;
use wgpu_sort::GPUSorter;

// Shared CubeCL radix sort implementation (decoupled fallback)
#[path = "../07_radix_sort/radix_sort.rs"]
mod radix_sort;

/// Context for wgpu_sort benchmarking
struct WgpuSortContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    sorter: GPUSorter,
}

impl WgpuSortContext {
    async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
            .await
            .expect("Failed to find adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_buffer_size: 1 << 30,
                        max_storage_buffer_binding_size: 1 << 30,
                        ..Default::default()
                    },
                    label: None,
                },
                None,
            )
            .await
            .expect("Failed to create device");

        // Try to guess subgroup size, fallback to 32 (common for Apple Silicon)
        let subgroup_size = match guess_workgroup_size(&device, &queue).await {
            Some(size) => {
                println!("wgpu_sort detected subgroup size: {}", size);
                size
            }
            None => {
                println!("wgpu_sort: could not detect subgroup size, using 32");
                32
            }
        };

        let sorter = GPUSorter::new(&device, subgroup_size);

        Self {
            device,
            queue,
            sorter,
        }
    }

    /// Benchmark sort - batch multiple iterations into single command buffer
    fn bench_sort(&self, n: u32, warmup_iters: usize, bench_iters: usize) -> Duration {
        let sort_buffers = self
            .sorter
            .create_sort_buffers(&self.device, NonZeroU32::new(n).unwrap());

        // Warmup - batch all warmup iterations
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            for _ in 0..warmup_iters {
                self.sorter
                    .sort(&mut encoder, &self.queue, &sort_buffers, None);
            }
            let idx = self.queue.submit([encoder.finish()]);
            self.device
                .poll(wgpu::Maintain::WaitForSubmissionIndex(idx));
        }

        // Benchmark - batch all iterations into one command buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let start = Instant::now();
        for _ in 0..bench_iters {
            self.sorter
                .sort(&mut encoder, &self.queue, &sort_buffers, None);
        }
        let idx = self.queue.submit([encoder.finish()]);
        self.device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(idx));
        let elapsed = start.elapsed();

        elapsed / bench_iters as u32
    }
}

/// Benchmark CubeCL radix sort
fn bench_cubecl_sort<R: Runtime>(
    client: &ComputeClient<R::Server>,
    n: usize,
    warmup_iters: usize,
    bench_iters: usize,
) -> Duration {
    // Generate test data once
    let data: Vec<u32> = (0..n as u32).rev().collect();

    // Warmup
    for _ in 0..warmup_iters {
        let _ = radix_sort::radix_sort_u32::<R>(client, &data);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..bench_iters {
        let _ = radix_sort::radix_sort_u32::<R>(client, &data);
    }
    let elapsed = start.elapsed();

    elapsed / bench_iters as u32
}

#[pollster::main]
async fn main() {
    println!("=== Radix Sort Benchmark: CubeCL vs wgpu_sort ===\n");

    // Initialize both contexts
    let wgpu_ctx = WgpuSortContext::new().await;

    let device = Default::default();
    let cubecl_client = cubecl::wgpu::WgpuRuntime::client(&device);
    type R = cubecl::wgpu::WgpuRuntime;

    println!();

    // Test sizes
    let sizes = [
        10_000, 50_000, 100_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000,
    ];

    let warmup = 3;
    let iters = 10;

    println!(
        "{:>12} {:>15} {:>15} {:>10}",
        "Size", "wgpu_sort", "CubeCL", "Speedup"
    );
    println!("{}", "-".repeat(55));

    for &size in &sizes {
        // Benchmark wgpu_sort
        let wgpu_time = wgpu_ctx.bench_sort(size as u32, warmup, iters);

        // Benchmark CubeCL
        let cubecl_time = bench_cubecl_sort::<R>(&cubecl_client, size, warmup, iters);

        // Calculate speedup (positive = CubeCL faster)
        let speedup = wgpu_time.as_secs_f64() / cubecl_time.as_secs_f64();

        println!(
            "{:>12} {:>12.3}ms {:>12.3}ms {:>9.2}x",
            size,
            wgpu_time.as_secs_f64() * 1000.0,
            cubecl_time.as_secs_f64() * 1000.0,
            speedup
        );
    }

    println!("\nSpeedup > 1.0 means CubeCL is faster");
}
