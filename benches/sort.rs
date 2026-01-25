// Benchmark comparing CubeCL GPU radix sort vs CPU sort
// Excludes buffer transfer overhead for GPU (GPU-only workflow scenario)
//
// IMPORTANT: Run with CUBECL_WGPU_MAX_TASKS=1000 to avoid hangs
// Due to GPU forward progress limits, max ~27K elements (7 blocks) works reliably

use std::time::{Duration, Instant};

// CubeCL radix sort implementation
#[allow(dead_code)]
mod radix_sort;

use cubecl::client::ComputeClient;
use cubecl::prelude::*;

/// Benchmark GPU sort - excludes initial data upload, measures kernel time only
async fn bench_gpu_sort<R: Runtime>(
    client: &ComputeClient<R::Server>,
    buffers: &radix_sort::RadixSortBuffers,
    runs: usize,
) -> Duration {
    // Warmup (skip for now - may cause issues with repeated sorts)

    // Benchmark - just one run to test
    let start = Instant::now();
    radix_sort::radix_sort_u32_gpu::<R>(client, buffers);
    client.flush();
    // Read to force completion (sync may have issues)
    let _ = client.read_one(buffers.keys_a.clone());
    let elapsed = start.elapsed();

    elapsed
}

/// Benchmark CPU sort
fn bench_cpu_sort(data: &[u32], runs: usize) -> Duration {
    // Warmup
    for _ in 0..3 {
        let mut copy = data.to_vec();
        copy.sort_unstable();
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..runs {
        let mut copy = data.to_vec();
        copy.sort_unstable();
    }
    let elapsed = start.elapsed();

    elapsed / runs as u32
}

#[pollster::main]
async fn main() {
    println!("=== Radix Sort Benchmark: CubeCL GPU vs CPU ===");
    println!("(GPU times exclude buffer transfer - GPU-only workflow)\n");

    // Setup CubeCL
    let device = Default::default();
    let client = cubecl::wgpu::WgpuRuntime::client(&device);
    type R = cubecl::wgpu::WgpuRuntime;

    println!(
        "{:>12} {:>15} {:>15} {:>10}",
        "Size", "GPU", "CPU", "Speedup"
    );
    println!("{}", "-".repeat(55));

    // Note: Max 7 blocks (~27K elements) due to GPU forward progress limits
    let sizes = [
        3_840usize,  // 1 block
        7_680,       // 2 blocks
        15_360,      // 4 blocks
        26_880,      // 7 blocks (max reliable)
    ];
    let runs = 5;

    for n in sizes {
        // Generate test data (reversed for worst-case)
        let data: Vec<u32> = (0..n as u32).rev().collect();

        // Create GPU buffers and upload data once (outside timing)
        let buffers = radix_sort::create_sort_buffers::<R>(&client, n);

        // Upload initial data to GPU (not timed)
        let padded_size = buffers.padded_size;
        let mut padded_data = data.clone();
        padded_data.resize(padded_size, 0xFFFFFFFF);
        let initial_keys = client.create(u32::as_bytes(&padded_data));
        // Copy to buffers.keys_a - we need to work around the API limitation
        // For now, just create fresh buffers with data for GPU benchmark
        drop(buffers);

        // Create buffers with data already loaded
        let buffers = {
            let padded_size = n.next_multiple_of(radix_sort::SCATTER_BLOCK_KVS);
            let num_blocks = padded_size / radix_sort::SCATTER_BLOCK_KVS;
            let num_planes = radix_sort::PREFIX_BLOCK_SIZE as u32 / cubecl::prelude::PLANE_DIM;

            let mut padded = data.clone();
            padded.resize(padded_size, 0xFFFFFFFF);

            radix_sort::RadixSortBuffers {
                keys_a: client.create(u32::as_bytes(&padded)),
                keys_b: client.empty(padded_size * 4),
                padded_size,
                original_len: n,
                num_blocks,
                num_planes,
            }
        };
        drop(initial_keys);

        // Benchmark GPU
        let gpu_time = bench_gpu_sort::<R>(&client, &buffers, runs).await;

        // Benchmark CPU
        let cpu_time = bench_cpu_sort(&data, runs);

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        println!(
            "{:>12} {:>12.3}ms {:>12.3}ms {:>9.2}x",
            n,
            gpu_time.as_secs_f64() * 1000.0,
            cpu_time.as_secs_f64() * 1000.0,
            speedup
        );
    }

    println!("\nSpeedup > 1.0 means GPU is faster than CPU");
}
