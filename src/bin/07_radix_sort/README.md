# GPU Radix Sort in CubeCL

A GPU radix sort for `u32` keys written in CubeCL. It is based on the
"onesweep"-style algorithm from [wgpu_sort](https://github.com/KeKsBoTer/wgpu_sort) and
Google's Fuchsia radix sort. The inter-workgroup prefix uses **Decoupled Fallback**, so the
sort runs correctly on GPUs that don't guarantee forward progress between workgroups, such
as Apple Silicon and ARM Mali.

## Files

| File | Contents |
|------|----------|
| [`radix_sort.rs`](radix_sort.rs) | The algorithm: kernels and host API. Shared with the benchmarks. |
| [`main.rs`](main.rs) | Stress-test harness that compares results with `std` sort. |

The benchmarks ([`benches/sort.rs`](../../../benches/sort.rs) and
[`benchmark_sort`](../benchmark_sort/main.rs)) include `radix_sort.rs` with
`#[path = ...] mod radix_sort;`.

## Concepts Demonstrated

- Plane (subgroup) operations: `plane_ballot` to rank keys that share a digit,
  `plane_shuffle` to broadcast a value from one lane, and `plane_exclusive_sum` /
  `plane_sum` for cube-wide scans
- Shared-memory atomics (`SharedMemory<Atomic<u32>>`) for tile histograms
- Global atomics (`Atomic::add`, `Atomic::max`, `Atomic::load`) for communication
  between workgroups
- Cube-uniform control flow: loops with `sync_cube()` inside, whose exit condition
  comes from shared memory read after a barrier
- `#[comptime]` kernel parameters to specialise a kernel (spin limit, test mode)
- `launch_unchecked` kernels and ping-pong buffers across passes

## Algorithm

Keys are sorted 8 bits at a time, least significant digit first, so 4 passes cover 32-bit
keys. Each pass is stable, which makes the whole LSD sort correct.

### Kernels

Each sort launches 7 kernels:

1. **`kernel_clear`**: zeroes the global histograms and the stats counters.
2. **`kernel_calc_histogram`**: one workgroup per tile. Counts the digits of all 4 passes
   in one read of the keys, accumulated in shared memory and then flushed to global memory
   with atomics. It also zeroes the partition status words for all passes.
3. **`kernel_prefix_histogram`**: a single workgroup turns each pass's histogram into an
   exclusive prefix sum. The result is where each digit's bucket starts in the output.
4. **`kernel_scatter`** × 4 (one launch per pass): one workgroup per tile of
   256 threads × 15 keys = 3840 keys.
   1. Load the tile. Each plane reads a contiguous chunk, with lanes strided by `PLANE_DIM`.
   2. Rank each key among the lanes of its plane with the same digit. Eight `plane_ballot`
      calls (one per digit bit) build a "same digit" lane mask.
   3. Build the tile histogram and each key's rank within the tile. Planes take turns in
      order, separated by barriers, so ranks follow input order and the pass is stable.
   4. **Decoupled fallback** (described below): find, for every digit, how many keys with
      that digit are in earlier tiles.
   5. Exclusive scan of the tile histogram gives each digit's start within the tile.
   6. Reorder the tile in shared memory so that the global writes are mostly contiguous.
   7. Write each key to bucket start + count from earlier tiles + rank within tile.

### Decoupled Fallback

Each tile needs, for every digit, the number of keys with that digit in all earlier tiles.
Tiles publish per-digit status words in a global buffer:

```
status word = status (2 bits) | count (30 bits)

NOT_READY (0)  nothing published yet
READY     (1)  count = this tile's own per-digit total (its reduction)
PREFIX    (2)  count = total over this tile and all earlier tiles (inclusive prefix)
```

A tile publishes `READY` with its own totals, then looks back through its predecessors.
Each `READY` count is added and the look-back continues to the next earlier tile. The
look-back stops at the first `PREFIX`, or at tile 0. Finally the tile publishes its own
`PREFIX`.

Plain **decoupled look-back** spins while a predecessor is `NOT_READY`. If that
predecessor's workgroup hasn't been scheduled, and the hardware has no forward-progress
guarantee, the spinning workgroup can wait forever. **Decoupled fallback** bounds the
wait:

- After `max_spins` polls (default 4) of a `NOT_READY` predecessor, the whole workgroup
  reads that tile's keys from the pass input and computes its digit totals itself.
- It publishes that result as `READY` with `atomicMax` and continues the look-back.
  Other workgroups waiting on the same tile can then use it.

Because every workgroup can always make progress by itself, the sort finishes however the
GPU schedules workgroups.

Implementation details:

- **Only moves forward:** the encoding is ordered `NOT_READY < READY < PREFIX`, and every
  publication uses `atomicMax`. A fallback `READY` can never overwrite a `PREFIX`.
  Duplicate `READY` publications are identical, because a tile's totals are deterministic.
- **One region per pass:** each pass uses its own region of the status buffer, so no
  clearing is needed between passes.
- **Lockstep look-back:** thread `t` handles digit `t`, but all 256 threads step through
  predecessors together, so the whole workgroup can work on a fallback. Three rotating
  shared-memory counter slots keep each look-back iteration to one barrier.
- **Bucket starts kept separate:** counts in status words are relative to the start of
  each digit's bucket. The global bucket start from `kernel_prefix_histogram` is added at
  the end, so tile 0's `READY` count is also its `PREFIX` count.

### Padding

The input is padded with `0xFFFFFFFF` up to a whole number of tiles. The padding keys sort
to the end, and the output is truncated back to the input length.

## API

```rust
use cubecl::wgpu::WgpuRuntime;

let device = Default::default();
let client = WgpuRuntime::client(&device);

// One-shot: upload, sort, download
let input: Vec<u32> = (1..=1_000_000).rev().collect();
let sorted = radix_sort_u32::<WgpuRuntime>(&client, &input);

// With GPU-side counters (fallbacks, look-back steps)
let (sorted, stats) =
    radix_sort_u32_with_stats::<WgpuRuntime>(&client, &input, SortConfig::default());

// GPU-only: reusable buffers, keys stay on the GPU
let mut buffers = create_sort_buffers::<WgpuRuntime>(&client, input.len());
upload_keys::<WgpuRuntime>(&client, &mut buffers, &input);
radix_sort_u32_gpu::<WgpuRuntime>(&client, &buffers, SortConfig::default());
// The result is in buffers.keys_a (first `input.len()` keys); this call does not sync.
```

| Item | Purpose |
|------|---------|
| `SortConfig::max_spins` | Polls of a `NOT_READY` predecessor before falling back (default 4) |
| `SortConfig::post_reduction` | Test-only. `false` stops tiles publishing `READY`, which forces fallbacks. |
| `SortStats` | `fallbacks` and `lookback_steps`, summed over all passes |

Fixed parameters at the top of `radix_sort.rs`: 8 bits per pass, 256 threads per
workgroup, 15 keys per thread, so 3840 keys per tile.

## Testing

```bash
# Default config: random + adversarial inputs, compared with std's sort
cargo run --release --bin 07_radix_sort -- 20
# Withhold tile reductions and use 0 spins: nearly every look-back step falls back
cargo run --release --bin 07_radix_sort -- --force-fallback 20 3841 65550 1000000
# Custom spin limit / sizes
cargo run --release --bin 07_radix_sort -- --spins 16 10 4000000 16000000
```

Usage: `07_radix_sort [--force-fallback] [--spins N] [ITERS] [SIZES...]`. Inputs cycle
through random keys, keys with many duplicates, reverse-sorted keys, all keys equal to the
padding value, and near-maximum keys. Each output is compared exactly with `sort_unstable`.
Each size prints the best time and the GPU-side fallback and look-back counters.

## Benchmarks

```bash
cargo run --release --bin benchmark_sort    # vs wgpu_sort and CPU std sorts (main benchmark)
cargo bench --bench sort                    # quick GPU vs CPU sort_unstable
```

Both benchmarks upload the input before each timed run, without timing it. A timed run is
one full sort plus waiting for the GPU to finish, and each benchmark checks its results
against `std` sort. `benchmark_sort` gives every sorter the same input, reports best and
median over 15 runs, and checks that the GPU outputs agree. The CPU sorts each get a fresh
copy of the input per run, made outside the timed region.

Results from one `benchmark_sort` run on an Apple M5 Max (wgpu / Metal), best of 15:

| Keys | CubeCL | wgpu_sort | CPU `sort_unstable` | CPU `sort` (stable) | CubeCL vs wgpu_sort | CubeCL vs `sort_unstable` |
|-----:|-------:|----------:|--------------------:|--------------------:|----:|----:|
| 10K   | 0.67 ms | 3.86 ms | 0.09 ms | 0.09 ms | 5.7× | 0.13× |
| 100K  | 0.63 ms | 3.79 ms | 0.77 ms | 0.88 ms | 6.0× | 1.2× |
| 1M    | 0.71 ms | 3.84 ms | 7.77 ms | 9.17 ms | 5.4× | 11× |
| 2M    | 0.72 ms | 4.98 ms | 16.3 ms | 20.2 ms | 6.9× | 22× |
| 4M    | 1.00 ms | 7.75 ms | 33.8 ms | 45.3 ms | 7.8× | 34× |
| 5M    | 1.24 ms | 9.93 ms | 43.2 ms | 61.7 ms | 8.0× | 35× |
| 10M   | 2.08 ms | 17.8 ms | 90.7 ms | 126 ms  | 8.5× | 44× |
| 16M   | 3.11 ms | 28.7 ms | 147 ms  | 206 ms  | 9.2× | 47× |
| 33.5M | 6.15 ms | 65.8 ms | 323 ms  | 441 ms  | 10.7× | 52× |

Up to about 2M keys the CubeCL time is dominated by its fixed cost of 7 kernel launches
plus a sync (about 0.6–0.7 ms), so the GPU only beats the CPU from roughly 100K keys.

On the CPU baselines: for plain `u32` keys a stable sort can't give a different result,
because equal keys are indistinguishable. `sort_unstable` (pattern-defeating quicksort) is
the faster of the two, so it is the main baseline; the stable `sort` (driftsort) is shown
for reference. The GPU radix sort is itself stable, which will matter once it sorts
key/value pairs.

Caveats on the wgpu_sort comparison:

- wgpu_sort always sorts key/value pairs, so it also moves a 4-byte payload per key.
- wgpu_sort uses wgpu 0.20. On Metal, its submit-and-wait round trip is about 1.3 ms, and
  its timings land on multiples of that, so small sizes mostly measure that overhead.
  `benchmark_sort` prints each runtime's empty round trip for reference.

## Limitations

- Keys-only sorting (no key-value pairs yet)
- Up to 2^30 - 1 keys (30-bit counts in the partition status words)
- Requires a plane (subgroup) size between 8 and 32. This is checked when buffers are
  created.
- Tested with the wgpu backend (Metal)

## Gotchas

- On the host, CubeCL's `PLANE_DIM` is a placeholder constant (`2`), not the hardware
  plane size. Compute plane-dependent values such as `CUBE_DIM / PLANE_DIM` inside the
  kernel, or read `client.properties().hardware.plane_size_*` on the host.
- `sync_cube()` must be reached by every unit of the cube. Loops containing barriers must
  exit on cube-uniform values, for example a shared-memory value read after a barrier.

## References

- Merrill, D., & Garland, M. (2016). *Single-pass Parallel Prefix Scan with Decoupled Look-back*. NVIDIA Technical Report NVR-2016-002.
- Smith, T., Levien, R., & Owens, J. D. (2025). *Decoupled Fallback: A Portable Single-Pass GPU Scan*. SPAA '25. [doi:10.1145/3694906.3743326](https://doi.org/10.1145/3694906.3743326)
- Adinets, A., & Merrill, D. (2022). *Onesweep: A Faster Least Significant Digit Radix Sort for GPUs*. arXiv:2206.01784.
- [wgpu_sort](https://github.com/KeKsBoTer/wgpu_sort), the WebGPU implementation this port is based on
- [Fuchsia radix sort](https://fuchsia.googlesource.com/fuchsia/+/refs/heads/main/src/graphics/lib/compute/radix_sort/)
