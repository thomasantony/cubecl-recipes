# Block Exclusive Sum

Demonstrates a sophisticated algorithm for computing prefix sums across an entire block using shared memory and plane intrinsics.

## Concepts Demonstrated

- Using `SharedMemory` for inter-plane communication
- Synchronizing threads with `sync_cube()`
- Multi-phase algorithms: local scan -> aggregate -> distribute
- Understanding `UNIT_POS_PLANE` - thread index within a plane

## Algorithm Overview

The block-level exclusive sum works in 4 phases:

1. **Local Scan**: Each plane computes its own exclusive prefix sum using `plane_exclusive_sum()`

2. **Aggregate**: The total from each plane is written to shared memory by thread 0 of each plane

3. **Scan Totals**: A single plane scans the totals to compute offsets for each plane

4. **Apply Offsets**: Each thread adds its plane's offset to its local scan result

## Key Code

```rust
#[cube(launch)]
fn kernel_block_exclusive_sum(input_data: &Array<u32>, output_data: &mut Array<u32>) {
    // Shared memory for inter-plane communication
    let mut shared_totals = SharedMemory::<u32>::new(2);

    // 1. Local scan within plane
    let original = input_data[thread_idx];
    let local_scan = plane_exclusive_sum(original);

    // 2. Plane totals -> shared memory
    let plane_total = plane_shuffle(local_scan, plane_size - 1)
                    + plane_shuffle(original, plane_size - 1);
    if plane_thread_idx == 0 {
        shared_totals[plane_idx] = plane_total;
    }
    sync_cube();

    // 3. Scan totals
    if plane_idx == 0 && plane_thread_idx < num_planes {
        shared_totals[plane_thread_idx] = plane_exclusive_sum(shared_totals[plane_thread_idx]);
    }
    sync_cube();

    // 4. Apply offset
    output_data[thread_idx] = local_scan + shared_totals[plane_idx];
}
```

## Why This Pattern?

Plane intrinsics only work within a single plane (32-64 threads). To compute prefix sums across larger blocks (e.g., 256 or 1024 threads), we need shared memory to communicate between planes.

This pattern is fundamental to many GPU algorithms including:
- Radix sort
- Stream compaction
- Histogram computation

## Running

```bash
cargo run --bin 04_block_sum
```
