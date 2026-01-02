# Hierarchical Exclusive Sum

Demonstrates a hierarchical prefix sum algorithm that works on arrays of arbitrary size by combining block-level scans across multiple levels.

## Concepts Demonstrated

- Pre-allocating GPU buffers with `client.empty()`
- Using `Handle::offset_start()` and `offset_end()` to create views into buffer regions
- Multi-level algorithms: up-sweep to collect totals, down-sweep to distribute offsets
- `#[comptime]` parameters for configurable `SharedMemory` sizes

## Algorithm Overview

For arrays larger than a single block, the algorithm processes multiple levels iteratively:

### Up-sweep (collect totals)
1. **Block Scan**: Each block computes its local prefix sum and outputs its total
2. **Repeat**: Continue scanning block totals at each level until a single block remains

### Down-sweep (distribute offsets)
3. **Add Offsets**: Working back down, each element adds its block's offset from the level above

For example, with 262,144 elements and block size 64:
```
Level 0: 262,144 elements -> 4,096 block totals
Level 1: 4,096 elements   -> 64 block totals
Level 2: 64 elements      -> 1 block (base case)
```

## Key Code

```rust
/// Describes a level in the hierarchical scan
struct LevelInfo {
    num_elements: usize,
    num_blocks: usize,
}

fn hierarchical_scan<R: Runtime>(
    client: &ComputeClient<R::Server>,
    input_gpu: &Handle,
    output_gpu: &Handle,
    num_elements: usize,
) {
    let levels = compute_level_sizes(num_elements);

    // Pre-allocate scratch buffers for all levels
    let totals_buffer = client.empty(total_scratch_elements * elem_size);
    let offsets_buffer = client.empty(total_scratch_elements * elem_size);

    // Create handles for each level's region using offsets
    for level in levels.iter() {
        totals_handles.push(
            totals_buffer.clone()
                .offset_start(byte_offset)
                .offset_end(totals_buffer.size() - byte_offset - level_bytes)
        );
        // ...
    }

    // Up-sweep: compute block totals at each level
    // Down-sweep: add offsets back at each level
}
```

## Buffer Management

Instead of allocating separate buffers for each level, this implementation:
1. Computes all level sizes upfront with `compute_level_sizes()`
2. Allocates two contiguous scratch buffers (totals and offsets)
3. Creates `Handle` views into regions using `offset_start()` and `offset_end()`

This reduces allocation overhead and is more predictable for GPU memory management.

## Running

```bash
cargo run --bin 05_hierarchical_scan
```
