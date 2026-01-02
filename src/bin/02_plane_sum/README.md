# Plane Exclusive Sum

Demonstrates using plane intrinsics to compute prefix sums within thread groups (planes/warps).

## Concepts Demonstrated

- Using `plane_exclusive_sum()` for intra-plane communication
- Understanding planes/warps as groups of threads that can communicate directly
- Prefix sum (scan) operations on the GPU

## What is a Plane?

A "plane" in CubeCL corresponds to a "warp" (NVIDIA) or "wavefront" (AMD). It's a group of threads (typically 32 or 64) that execute in lockstep and can efficiently share data using special intrinsic functions.

## Key Code

```rust
#[cube(launch)]
fn kernel_plane_exclusive_sum(input_data: &Array<u32>, output_data: &mut Array<u32>) {
    // ABSOLUTE_POS is equivalent to CUBE_POS * CUBE_DIM + UNIT_POS
    let index = ABSOLUTE_POS;

    let local_data = input_data[index];
    let local_sum = plane_exclusive_sum(local_data);

    output_data[index] = local_sum;
}
```

## Example Output

For input `[1, 1, 1, 1, 1, 1, 1, 1]` (8 ones):

The exclusive prefix sum produces `[0, 1, 2, 3, 4, 5, 6, 7]`

Each element contains the sum of all preceding elements (exclusive means it doesn't include itself).

## Running

```bash
cargo run --bin 02_plane_sum
```
