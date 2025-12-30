//! Plane Exclusive Sum - Prefix sums within a plane/warp
//!
//! This example demonstrates using plane intrinsics to compute prefix sums
//! within thread groups (planes/warps). The `plane_exclusive_sum` function
//! takes the value from every thread in the current plane and computes an
//! exclusive prefix sum.

use cubecl::prelude::*;

#[cube(launch)]
fn kernel_plane_exclusive_sum(input_data: &Array<u32>, output_data: &mut Array<u32>) {
    let block_id = CUBE_POS;
    let thread_id = UNIT_POS;

    let local_data = input_data[block_id * CUBE_DIM + thread_id];
    let local_sum = plane_exclusive_sum(local_data);

    output_data[block_id * CUBE_DIM + thread_id] = local_sum;
}

fn main() {
    let device = Default::default();
    let client = cubecl::wgpu::WgpuRuntime::client(&device);

    type R = cubecl::wgpu::WgpuRuntime;

    // Use data array that fits within a plane
    let input_data = vec![1u32; 16];
    println!("Input: {:?}", &input_data);

    let num_elements = input_data.len();
    let zeros = vec![0u32; num_elements];
    let input_data_gpu = client.create(u32::as_bytes(&input_data));
    let output_data_gpu = client.create(u32::as_bytes(&zeros));

    const BLOCK_SIZE: usize = 8;
    let num_blocks = num_elements / BLOCK_SIZE;

    unsafe {
        kernel_plane_exclusive_sum::launch::<R>(
            &client,
            CubeCount::Static(num_blocks as u32, 1, 1),
            CubeDim::new(BLOCK_SIZE as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&input_data_gpu, num_elements, 1),
            ArrayArg::from_raw_parts::<u32>(&output_data_gpu, num_elements, 1),
        )
    }

    let result = client.read_one(output_data_gpu.clone());
    let output = u32::from_bytes(&result).to_vec();
    println!("Plane Exclusive Sum: {:?}", output);
    println!("\nNote: Each group of {} elements shows an exclusive prefix sum", BLOCK_SIZE);
    println!("For input [1,1,1,...], output is [0,1,2,3,...] within each plane");
}
