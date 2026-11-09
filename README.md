# CubeCL Recipes

A collection of examples demonstrating GPU-agnostic programming using [CubeCL](https://github.com/tracel-ai/cubecl).

## Running Examples

Make sure you have Rust installed, then run any example with:

```bash
cargo run --bin <example_name>
```

For example:

```bash
cargo run --bin 00_double_numbers
```

## Examples

| Example | Description |
|---------|-------------|
| [00_double_numbers](src/bin/00_double_numbers/) | The simplest CubeCL kernel - doubles each array element |
| [01_scale_numbers](src/bin/01_scale_numbers/) | Passing scalar parameters to kernels with `ScalarArg` |
| [02_plane_sum](src/bin/02_plane_sum/) | Plane-level exclusive prefix sum using `plane_exclusive_sum` |
| [03_plane_broadcast](src/bin/03_plane_broadcast/) | Broadcasting values across threads with `plane_broadcast` |
| [04_block_sum](src/bin/04_block_sum/) | Block-level prefix sum using shared memory |
| [05_hierarchical_scan](src/bin/05_hierarchical_scan/) | Hierarchical prefix sum for arbitrarily large arrays |
| [06_plane_ballot](src/bin/06_plane_ballot/) | Collecting per-thread booleans into a bitmask with `plane_ballot` |
| [07_radix_sort](src/bin/07_radix_sort/) | Single-pass 8-bit LSD radix sort of `u32` keys using decoupled fallback for the inter-workgroup prefix (portable to GPUs without forward-progress guarantees, e.g. Apple Silicon) |

## Radix Sort Benchmarks

The radix sort ([07_radix_sort](src/bin/07_radix_sort/)) comes with two benchmarks:

| Command | Compares |
|---------|----------|
| `cargo bench --bench sort` | CubeCL GPU radix sort vs Rust's `sort_unstable` on the CPU |
| `cargo run --release --bin benchmark_sort` | CubeCL radix sort vs [wgpu_sort](https://github.com/KeKsBoTer/wgpu_sort), with results cross-checked |

Both check the GPU output against `std` sort. See the
[radix sort README](src/bin/07_radix_sort/README.md#benchmarks) for methodology and results.

**Note:** `benchmark_sort` depends on `wgpu_sort` through a path dependency
(`wgpu_sort = { path = "../wgpu_sort" }` in `Cargo.toml`), so building the crate needs a
checkout of [wgpu_sort](https://github.com/KeKsBoTer/wgpu_sort) next to this repository.

## Resources

- [GPU-Agnostic Programming Using CubeCL](https://www.thomasantony.com/posts/202512281621-gpu-agnostic-programming-using-cubecl/) - Blog post explaining these examples
- [CubeCL Documentation](https://docs.rs/cubecl/latest/cubecl/)
- [CubeCL GitHub Repository](https://github.com/tracel-ai/cubecl)
