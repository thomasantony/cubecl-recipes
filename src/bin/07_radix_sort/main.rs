// GPU Radix Sort using CubeCL - stress test harness.
// The algorithm lives in radix_sort.rs (also used by benches/sort.rs and benchmark_sort).

mod radix_sort;

use cubecl::prelude::*;
use radix_sort::{SortConfig, SortStats, radix_sort_u32_with_stats};

fn xorshift(state: &mut u64) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 16) as u32
}

fn gen_input(kind: usize, n: usize, seed: u64) -> Vec<u32> {
    let mut st = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    match kind % 5 {
        0 => (0..n).map(|_| xorshift(&mut st)).collect(),
        1 => (0..n).map(|_| xorshift(&mut st) & 0xFF).collect(), // many duplicates
        2 => (0..n as u32).rev().collect(),
        3 => vec![0xFFFF_FFFFu32; n], // all equal to the padding value
        _ => (0..n).map(|_| xorshift(&mut st) | 0xFFFF_FF00).collect(), // near-max keys
    }
}

/// Usage: 07_radix_sort [--force-fallback] [--spins N] [ITERS] [SIZES...]
///
/// Sorts random/adversarial inputs of each size ITERS times and compares the
/// result with std's sort. `--force-fallback` withholds tile reductions and uses
/// zero spins, so nearly every look-back step goes through the fallback path.
fn main() {
    type R = cubecl::wgpu::WgpuRuntime;
    let device = Default::default();
    let client = R::client(&device);

    let mut config = SortConfig::default();
    let mut positional = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--force-fallback" => {
                config.post_reduction = false;
                config.max_spins = 0;
            }
            "--spins" => config.max_spins = args.next().expect("--spins N").parse().unwrap(),
            _ => positional.push(arg),
        }
    }
    let iters: usize = positional.first().map(|a| a.parse().unwrap()).unwrap_or(3);
    let sizes: Vec<usize> = if positional.len() > 1 {
        positional[1..].iter().map(|a| a.parse().unwrap()).collect()
    } else {
        vec![
            1, 100, 3840, 3841, 26_880, 65_550, 100_000, 1_000_000, 4_000_000, 16_000_000,
        ]
    };
    println!("config: {config:?}");

    let mut failures = 0usize;
    for &n in &sizes {
        let mut total_stats = SortStats::default();
        let mut best = std::time::Duration::MAX;
        for it in 0..iters {
            let kind = it % 5;
            let input = gen_input(kind, n, ((n as u64) << 20) | it as u64);
            let mut expected = input.clone();
            expected.sort_unstable();

            let start = std::time::Instant::now();
            let (got, stats) = radix_sort_u32_with_stats::<R>(&client, &input, config);
            best = best.min(start.elapsed());
            total_stats.fallbacks += stats.fallbacks;
            total_stats.lookback_steps += stats.lookback_steps;

            if got != expected {
                failures += 1;
                let bad = got.iter().zip(&expected).position(|(a, b)| a != b);
                println!("FAIL n={n} iter={it} kind={kind} first_mismatch={bad:?}");
            }
        }
        println!(
            "n={n:>9}  best {best:>10.3?}  fallbacks {:>8}  lookback steps {:>9}",
            total_stats.fallbacks, total_stats.lookback_steps
        );
    }

    if failures == 0 {
        println!("ALL PASSED");
    } else {
        println!("{failures} FAILURES");
        std::process::exit(1);
    }
}
