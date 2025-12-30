# Double Numbers

The simplest CubeCL kernel - takes an array of numbers and doubles each element.

## Concepts Demonstrated

- Basic kernel structure with `#[cube(launch)]`
- Computing global index using `CUBE_POS` (block ID) and `UNIT_POS` (thread ID)
- Launching a kernel with `CubeCount` and `CubeDim`
- Reading results back from the GPU

## Key Code

```rust
#[cube(launch)]
fn kernel_double_numbers(input_data: &Array<u32>, output_data: &mut Array<u32>) {
    let block_id = CUBE_POS;
    let thread_id = UNIT_POS;

    let index = block_id * CUBE_DIM + thread_id;
    output_data[index] = input_data[index] * 2;
}
```

## Running

```bash
cargo run --bin 00_double_numbers
```
