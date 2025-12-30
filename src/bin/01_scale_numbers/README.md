# Scale Numbers

Demonstrates how to pass scalar values to a CubeCL kernel using `ScalarArg`.

## Concepts Demonstrated

- Passing scalar parameters to kernels with `ScalarArg::new()`
- Creating reusable helper functions with `#[cube]` (not `#[cube(launch)]`)
- Backend-agnostic kernel launching using generics with `Runtime` trait bounds

## Key Code

```rust
#[cube(launch)]
fn kernel_scale_numbers(input_data: &Array<u32>, scale: u32, output_data: &mut Array<u32>) {
    let index = CUBE_POS * CUBE_DIM + UNIT_POS;
    output_data[index] = input_data[index] * scale;
}

// Launch with ScalarArg for the scale parameter
kernel_scale_numbers::launch::<R>(
    &client,
    CubeCount::Static(1, 1, 1),
    CubeDim::new(num_elements as u32, 1, 1),
    ArrayArg::from_raw_parts::<u32>(input, num_elements, 1),
    ScalarArg::new(scale),  // <-- Scalar value passed here
    ArrayArg::from_raw_parts::<u32>(output, num_elements, 1),
)
```

## Backend-Agnostic Pattern

The `launch_scale_numbers_kernel` function uses a generic `R: Runtime` parameter,
allowing the same kernel launching logic to work across different GPU backends
(WGPU, CUDA) without code changes.

## Running

```bash
cargo run --bin 01_scale_numbers
```
