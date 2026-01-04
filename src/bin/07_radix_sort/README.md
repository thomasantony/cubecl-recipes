# GPU Radix Sort in CubeCL

A GPU-accelerated radix sort implementation using CubeCL, based on the algorithm from wgpu_sort and Google's Fuchsia project.

## Algorithm

This implementation uses a **decoupled look-back** approach for efficient parallel prefix sums across workgroups, as described in:

- Merrill, D., & Garland, M. (2016). *Single-pass Parallel Prefix Scan with Decoupled Look-back*. NVIDIA Technical Report NVR-2016-002.

The algorithm performs 4 passes (8 bits per pass for 32-bit keys):

1. **Histogram**: Count occurrences of each digit (0-255) across all blocks
2. **Prefix Sum**: Compute exclusive prefix sum of the global histogram
3. **Scatter**: For each pass, redistribute keys to their sorted positions using:
   - Local histogram accumulation within each block
   - Decoupled look-back to compute global prefixes
   - Local reordering in shared memory
   - Global scatter to output

## References

- [wgpu_sort](https://crates.io/crates/wgpu_sort) - The reference WebGPU implementation
- [Fuchsia Radix Sort](https://fuchsia.googlesource.com/fuchsia/+/refs/heads/main/src/graphics/lib/compute/radix_sort/) - Google's Vulkan implementation
- [CubeCL](https://github.com/tracel-ai/cubecl) - The GPU compute framework used

## Limitations

- Currently requires input size to be a multiple of 3840 (block size × tile size)
- Keys-only sorting (no key-value pairs yet)
- Tested with wgpu backend

## Usage

```rust
use cubecl::wgpu::WgpuRuntime;

let device = Default::default();
let client = WgpuRuntime::client(&device);

let input: Vec<u32> = (1..=7680).rev().collect();
let sorted = radix_sort_u32::<WgpuRuntime>(&client, &input);
```
