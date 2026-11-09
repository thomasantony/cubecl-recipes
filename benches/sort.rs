// Benchmark comparing CubeCL GPU radix sort (decoupled fallback) vs CPU sort.
// GPU times exclude host<->device transfers (GPU-only workflow scenario).

use std::time::{Duration, Instant};

#[path = "../src/bin/07_radix_sort/radix_sort.rs"]
mod radix_sort;

use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use radix_sort::{RadixSortBuffers, SortConfig};

fn random_keys(n: usize) -> Vec<u32> {
    let mut state = 0x2545F4914F6CDD1Du64;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 16) as u32
        })
        .collect()
}

/// Benchmark GPU sort: upload is untimed, the timed region covers all kernels until
/// the GPU is idle. Returns the best of `runs`.
fn bench_gpu_sort<R: Runtime>(
    client: &ComputeClient<R::Server>,
    buffers: &mut RadixSortBuffers,
    data: &[u32],
    runs: usize,
) -> Duration {
    let mut best = Duration::MAX;
    // One warmup run (pipeline compilation) + timed runs
    for run in 0..=runs {
        radix_sort::upload_keys::<R>(client, buffers, data);
        cubecl::future::block_on(client.sync());

        let start = Instant::now();
        radix_sort::radix_sort_u32_gpu::<R>(client, buffers, SortConfig::default());
        cubecl::future::block_on(client.sync());
        let elapsed = start.elapsed();

        if run > 0 {
            best = best.min(elapsed);
        }
    }
    best
}

/// Benchmark CPU sort (best of `runs`)
fn bench_cpu_sort(data: &[u32], runs: usize) -> Duration {
    let mut best = Duration::MAX;
    for _ in 0..runs {
        let mut copy = data.to_vec();
        let start = Instant::now();
        copy.sort_unstable();
        best = best.min(start.elapsed());
    }
    best
}

fn main() {
    println!("=== Radix Sort Benchmark: CubeCL GPU vs CPU ===");
    println!("(GPU times exclude buffer transfer - GPU-only workflow)\n");

    let device = Default::default();
    let client = cubecl::wgpu::WgpuRuntime::client(&device);
    type R = cubecl::wgpu::WgpuRuntime;

    println!("{:>12} {:>15} {:>15} {:>10}", "Size", "GPU", "CPU", "Speedup");
    println!("{}", "-".repeat(55));

    let sizes = [
        3_840usize, 26_880, 100_000, 1_000_000, 4_000_000, 16_000_000, 33_554_432,
    ];
    let runs = 5;

    for n in sizes {
        let data = random_keys(n);
        let mut buffers = radix_sort::create_sort_buffers::<R>(&client, n);

        let gpu_time = bench_gpu_sort::<R>(&client, &mut buffers, &data, runs);

        // Verify the last GPU result
        let sorted = client.read_one(buffers.keys_a.clone());
        let mut expected = data.clone();
        expected.sort_unstable();
        assert_eq!(&u32::from_bytes(&sorted)[..n], &expected[..], "GPU sort wrong for n={n}");

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
