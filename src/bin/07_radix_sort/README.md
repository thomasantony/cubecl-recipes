# GPU Radix Sort in CubeCL

A GPU-accelerated radix sort implementation using CubeCL, based on the algorithm from wgpu_sort and Google's Fuchsia project.

## Algorithm

The inter-workgroup prefix uses **Decoupled Fallback**, a portable variant of
decoupled look-back:

- Merrill, D., & Garland, M. (2016). *Single-pass Parallel Prefix Scan with Decoupled Look-back*.
- Smith, T., Levien, R., & Owens, J. D. (2025). *Decoupled Fallback: A Portable Single-Pass GPU Scan*. SPAA '25.

The algorithm performs 4 passes (8 bits per pass for 32-bit keys):

1. **Histogram**: Count occurrences of each digit (0-255) for all 4 passes at once
   (this kernel also zeroes the per-pass partition status regions)
2. **Prefix Sum**: Exclusive prefix sum of each pass's global histogram
3. **Scatter** (per pass, one workgroup per tile of 3840 keys):
   - Rank each key within its plane (ballot) and within its tile (shared-memory histogram)
   - Publish the tile's per-digit reduction (`READY`), then look back over predecessor
     tiles until a `PREFIX` is found. If a predecessor is still `NOT_READY` after
     `max_spins` polls, the whole workgroup recomputes that tile's reduction from the
     input keys and inserts it with `atomicMax` (**fallback**), so no workgroup ever
     waits on one that has not been scheduled.
   - Publish the tile's inclusive prefix (`PREFIX`), reorder the tile in shared memory,
     and scatter to the output

Partition status words are `status (2 bits) | count (30 bits)` with the monotonic
encoding `NOT_READY (0) < READY (1) < PREFIX (2)`, so all publications are `atomicMax`
and can never move a tile backwards. Each pass has its own status region.

## Testing

```bash
# Default config: random + adversarial inputs, compared against std's sort
cargo run --release --bin 07_radix_sort -- 20
# Withhold tile reductions and use 0 spins: nearly every look-back step falls back
cargo run --release --bin 07_radix_sort -- --force-fallback 20 3841 65550 1000000
# Custom spin limit / sizes
cargo run --release --bin 07_radix_sort -- --spins 16 10 4000000 16000000
```

Each line reports the best time and the GPU-side fallback / look-back step counters.

## Limitations

- Keys-only sorting (no key-value pairs yet)
- Up to 2^30 - 1 keys (30-bit counts in the partition status words)
- Requires a plane (subgroup) size between 8 and 32
- Tested with the wgpu backend (Metal)

## Usage

```rust
use cubecl::wgpu::WgpuRuntime;

let device = Default::default();
let client = WgpuRuntime::client(&device);

let input: Vec<u32> = (1..=1_000_000).rev().collect();
let sorted = radix_sort_u32::<WgpuRuntime>(&client, &input);
```
