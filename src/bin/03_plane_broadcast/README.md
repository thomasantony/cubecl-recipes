# Plane Broadcast

Demonstrates using `plane_broadcast` to broadcast a value from one specific thread to all other threads in the same plane/warp.

## Concepts Demonstrated

- Using `plane_broadcast()` to read a value from a specific thread index
- Broadcasting values across all threads in a plane

## Key Code

```rust
#[cube(launch)]
fn kernel_plane_broadcast(input_data: &Array<u32>, output_data: &mut Array<u32>) {
    let block_id = CUBE_POS;
    let thread_id = UNIT_POS;

    let local_data = input_data[block_id * CUBE_DIM + thread_id];

    // Get the value of local_data from thread #7 (zero-indexed) in the plane
    let value_from_thread_7 = plane_broadcast(local_data, 7);

    // Add it to the current value
    output_data[block_id * CUBE_DIM + thread_id] = local_data + value_from_thread_7;
}
```

## Plane Broadcast vs Plane Shuffle

- `plane_broadcast(value, idx)` - Returns the value of `value` from thread `idx`, optimized for when all threads read from the same source
- `plane_shuffle(value, idx)` - Similar, but each thread can read from a different source index

Both allow efficient intra-plane communication without shared memory.

## Example Output

For input `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` with block size 8:

- Block 1: Thread #7 has value 7, output is `[7, 8, 9, 10, 11, 12, 13, 14]`
- Block 2: Thread #7 has value 15, output is `[23, 24, 25, 26, 27, 28, 29, 30]`

## Running

```bash
cargo run --bin 03_plane_broadcast
```
