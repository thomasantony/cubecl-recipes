// Benchmark comparing CubeCL radix sort (decoupled fallback) vs wgpu_sort and CPU std sorts
// (sort_unstable and the stable sort), all on the same inputs in one run.
//
// Methodology (identical for both sorters):
// - Same random u32 input for both, regenerated per size.
// - Before every timed run the input is re-uploaded and the GPU is idle (untimed),
//   so each run sorts the same unsorted data from scratch.
// - The timed region is: record/launch one complete sort + wait for the GPU to finish.
// - Report best and median over RUNS timed runs (after WARMUP untimed runs).
// - After timing, both results are downloaded and checked against std's sort and
//   against each other.
// - CPU sorts: each run sorts a fresh copy of the input (copy untimed).
//
// Caveat: wgpu_sort always sorts key/value pairs (it has no keys-only mode), so it
// also moves a 4-byte payload per key in each scatter pass; CubeCL sorts keys only.
// The two also use different wgpu versions (wgpu_sort: wgpu 0.20). On Metal, wgpu 0.20's
// submit+wait round trip is ~1.3 ms and wgpu_sort timings are quantised to multiples of
// it, so small sizes mostly measure that overhead; the round trip is printed for reference.

use std::num::NonZeroU32;
use std::time::{Duration, Instant};

use cubecl::client::ComputeClient;
use cubecl::prelude::*;

use wgpu_sort::GPUSorter;
use wgpu_sort::utils::{download_buffer, guess_workgroup_size};

// Shared CubeCL radix sort implementation (decoupled fallback)
#[path = "../07_radix_sort/radix_sort.rs"]
mod radix_sort;

const WARMUP: usize = 3;
const RUNS: usize = 15;

struct Timing {
    best: Duration,
    median: Duration,
}

impl Timing {
    fn from_samples(mut samples: Vec<Duration>) -> Self {
        samples.sort();
        Self {
            best: samples[0],
            median: samples[samples.len() / 2],
        }
    }
}

fn random_keys(n: usize, seed: u64) -> Vec<u32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 16) as u32
        })
        .collect()
}

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

        let subgroup_size = guess_workgroup_size(&device, &queue).await.unwrap_or_else(|| {
            println!("wgpu_sort: could not detect subgroup size, using 32");
            32
        });
        println!("wgpu_sort subgroup size: {subgroup_size}");

        let sorter = GPUSorter::new(&device, subgroup_size);
        Self {
            device,
            queue,
            sorter,
        }
    }

    /// Wall-clock time of submitting an empty command buffer and waiting for it.
    fn round_trip(&self) -> Timing {
        let samples = (0..WARMUP + RUNS)
            .map(|_| {
                let start = Instant::now();
                let encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                let idx = self.queue.submit([encoder.finish()]);
                self.device
                    .poll(wgpu::Maintain::WaitForSubmissionIndex(idx));
                start.elapsed()
            })
            .skip(WARMUP)
            .collect();
        Timing::from_samples(samples)
    }

    fn wait_idle(&self) {
        let idx = self.queue.submit([]);
        self.device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(idx));
    }

    /// Returns timing and the sorted keys of the last run.
    fn bench(&self, data: &[u32]) -> (Timing, Vec<u32>) {
        let n = data.len();
        let buffers = self
            .sorter
            .create_sort_buffers(&self.device, NonZeroU32::new(n as u32).unwrap());
        // Payload is the original index (wgpu_sort always sorts key/value pairs)
        let payload: Vec<u32> = (0..n as u32).collect();

        let mut samples = Vec::with_capacity(RUNS);
        for run in 0..WARMUP + RUNS {
            self.queue
                .write_buffer(buffers.keys(), 0, u32::as_bytes(data));
            self.queue
                .write_buffer(buffers.values(), 0, u32::as_bytes(&payload));
            self.wait_idle();

            let start = Instant::now();
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            self.sorter.sort(&mut encoder, &self.queue, &buffers, None);
            let idx = self.queue.submit([encoder.finish()]);
            self.device
                .poll(wgpu::Maintain::WaitForSubmissionIndex(idx));
            let elapsed = start.elapsed();

            if run >= WARMUP {
                samples.push(elapsed);
            }
        }

        let sorted: Vec<u32> = pollster::block_on(download_buffer(
            buffers.keys(),
            &self.device,
            &self.queue,
            0..buffers.keys_valid_size(),
        ));
        (Timing::from_samples(samples), sorted)
    }
}

/// Returns timing and the sorted keys of the last run.
fn bench_cubecl<R: Runtime>(client: &ComputeClient<R::Server>, data: &[u32]) -> (Timing, Vec<u32>) {
    let n = data.len();
    let mut buffers = radix_sort::create_sort_buffers::<R>(client, n);
    let config = radix_sort::SortConfig::default();

    let mut samples = Vec::with_capacity(RUNS);
    for run in 0..WARMUP + RUNS {
        radix_sort::upload_keys::<R>(client, &mut buffers, data);
        cubecl::future::block_on(client.sync());

        let start = Instant::now();
        radix_sort::radix_sort_u32_gpu::<R>(client, &buffers, config);
        cubecl::future::block_on(client.sync());
        let elapsed = start.elapsed();

        if run >= WARMUP {
            samples.push(elapsed);
        }
    }

    let bytes = client.read_one(buffers.keys_a.clone());
    let sorted = u32::from_bytes(&bytes)[..n].to_vec();
    (Timing::from_samples(samples), sorted)
}

/// Times a CPU std sort. The copy of the input is made before the timer starts.
fn bench_cpu(data: &[u32], sort: fn(&mut [u32])) -> Timing {
    let samples = (0..WARMUP + RUNS)
        .map(|_| {
            let mut v = data.to_vec();
            let start = Instant::now();
            sort(&mut v);
            let elapsed = start.elapsed();
            std::hint::black_box(&v);
            elapsed
        })
        .skip(WARMUP)
        .collect();
    Timing::from_samples(samples)
}

/// Wall-clock time of a sync with no work queued (CubeCL runtime overhead floor).
fn cubecl_round_trip<R: Runtime>(client: &ComputeClient<R::Server>) -> Timing {
    let samples = (0..WARMUP + RUNS)
        .map(|_| {
            let start = Instant::now();
            cubecl::future::block_on(client.sync());
            start.elapsed()
        })
        .skip(WARMUP)
        .collect();
    Timing::from_samples(samples)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

#[pollster::main]
async fn main() {
    println!("=== Radix Sort Benchmark: CubeCL (keys) vs wgpu_sort (key/value) vs CPU std sort ===\n");

    let wgpu_ctx = WgpuSortContext::new().await;

    type R = cubecl::wgpu::WgpuRuntime;
    let device = Default::default();
    let cubecl_client = R::client(&device);

    let sizes = [
        10_000, 100_000, 1_000_000, 2_000_000, 4_000_000, 5_000_000, 10_000_000, 16_000_000,
        33_554_432,
    ];

    let wgpu_rt = wgpu_ctx.round_trip();
    let cube_rt = cubecl_round_trip::<R>(&cubecl_client);
    println!(
        "\nEmpty submit+wait round trip (best / median ms): wgpu_sort {:.3} / {:.3}, CubeCL {:.3} / {:.3}",
        ms(wgpu_rt.best),
        ms(wgpu_rt.median),
        ms(cube_rt.best),
        ms(cube_rt.median)
    );
    println!(
        "\n{} timed runs per size (after {} warmup), times in ms: best / median\n",
        RUNS, WARMUP
    );
    println!(
        "{:>10}  {:>19}  {:>19}  {:>19}  {:>19}  {:>13}  {}",
        "Size",
        "CPU sort_unstable",
        "CPU sort (stable)",
        "wgpu_sort",
        "CubeCL",
        "vs wgpu_sort",
        "Check"
    );
    println!("{}", "-".repeat(124));

    let mut all_ok = true;
    for (i, &size) in sizes.iter().enumerate() {
        let data = random_keys(size, i as u64 + 1);
        let mut expected = data.clone();
        expected.sort_unstable();

        let unstable_t = bench_cpu(&data, <[u32]>::sort_unstable);
        let stable_t = bench_cpu(&data, <[u32]>::sort);
        let (wgpu_t, wgpu_out) = wgpu_ctx.bench(&data);
        let (cube_t, cube_out) = bench_cubecl::<R>(&cubecl_client, &data);

        let wgpu_ok = wgpu_out == expected;
        let cube_ok = cube_out == expected;
        let agree = wgpu_out == cube_out;
        all_ok &= wgpu_ok && cube_ok && agree;
        let check = match (wgpu_ok, cube_ok, agree) {
            (true, true, true) => "ok".to_string(),
            _ => format!("MISMATCH (wgpu_sort {wgpu_ok}, cubecl {cube_ok}, agree {agree})"),
        };

        println!(
            "{:>10}  {:>8.3} / {:>8.3}  {:>8.3} / {:>8.3}  {:>8.3} / {:>8.3}  {:>8.3} / {:>8.3}  {:>12.2}x  {}",
            size,
            ms(unstable_t.best),
            ms(unstable_t.median),
            ms(stable_t.best),
            ms(stable_t.median),
            ms(wgpu_t.best),
            ms(wgpu_t.median),
            ms(cube_t.best),
            ms(cube_t.median),
            wgpu_t.best.as_secs_f64() / cube_t.best.as_secs_f64(),
            check
        );
    }

    println!("\n\"vs wgpu_sort\" is the best-time speedup; > 1.0 means CubeCL is faster.");
    println!("Note: wgpu_sort also moves a payload per key (key/value sort).");
    if !all_ok {
        println!("SOME RESULTS DID NOT MATCH std sort");
        std::process::exit(1);
    }
}
