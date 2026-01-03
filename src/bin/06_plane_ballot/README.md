# Plane Ballot

Demonstrates using `plane_ballot` to collect boolean values from all threads in a plane into a bitmask.

## Concepts Demonstrated

- Using `plane_ballot()` to aggregate boolean conditions across threads
- Understanding the `Line<u32>` return type for bitmask results
- How each thread contributes one bit to the result

## What is Plane Ballot?

`plane_ballot(condition)` collects a boolean value from each thread in the plane and packs them into a bitmask. Thread N's result is stored in bit N of the output.

```rust
let ballot_result = plane_ballot(UNIT_POS % 2 == 0);
// Thread 0: true  → bit 0 = 1
// Thread 1: false → bit 1 = 0
// Thread 2: true  → bit 2 = 1
// Thread 3: false → bit 3 = 0
// ... and so on
// Result: 0b...01010101 = 0x55555555
```

## The `Line<u32>` Return Type

The result is a `Line<u32>` because planes can have more than 32 threads:

| Plane Size | Meaningful Elements |
|------------|---------------------|
| 32 threads | `result[0]` only |
| 64 threads | `result[0]` (threads 0-31), `result[1]` (threads 32-63) |

For 64-thread wavefronts (AMD GPUs):
```rust
let ballot_result = plane_ballot(some_condition);
let low_bits = ballot_result[0];   // Threads 0-31
let high_bits = ballot_result[1];  // Threads 32-63
```

## Key Code

```rust
#[cube(launch)]
pub fn kernel_plane_ballot(output: &mut Array<u32>) {
    // Each thread votes true if its position is even
    let ballot_result = plane_ballot(UNIT_POS % 2 == 0);

    // Thread 0 writes the result
    if UNIT_POS == 0 {
        output[0] = ballot_result[0];
    }
}
```

## Example Output

```
=== Plane Ballot: Even threads ===
Condition: UNIT_POS % 2 == 0
Result: 0x55555555 (binary: 01010101010101010101010101010101)
Expected: 0x55555555 (alternating bits)

=== Plane Ballot: All true ===
Condition: true
Result: 0xffffffff (binary: 11111111111111111111111111111111)
Expected: 0xffffffff (all bits set)

=== Plane Ballot: First 8 threads ===
Condition: UNIT_POS < 8
Result: 0x000000ff (binary: 00000000000000000000000011111111)
Expected: 0x000000ff (first 8 bits set)
```

## Common Use Cases

- **Counting active threads**: `popcnt(plane_ballot(condition))` counts how many threads satisfy a condition
- **Finding first active thread**: `ctz(plane_ballot(condition))` finds the lowest thread index where condition is true
- **Divergence detection**: Check if all threads agree on a condition

## Running

```bash
cargo run --bin 06_plane_ballot
```
