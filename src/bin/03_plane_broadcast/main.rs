//! Plane Broadcast - Sharing values across threads in a plane
//!
//! This example demonstrates using `plane_broadcast` to broadcast a value
//! from one specific thread to all other threads in the same plane/warp.
//! This is useful for collective calculations where all threads need
//! access to a specific value.

use cubecl::prelude::*;

#[cube(launch)]
fn kernel_plane_broadcast(input_data: &Array<u32>, output_data: &mut Array<u32>) {
    // ABSOLUTE_POS is equivalent to CUBE_POS * CUBE_DIM + UNIT_POS
    let index = ABSOLUTE_POS;

    let local_data = input_data[index];

    // Get the value of local_data from thread #7 (zero-indexed) in the plane
    let value_from_thread_7 = plane_broadcast(local_data, 7);

    // Add it to the current value
    output_data[index] = local_data + value_from_thread_7;
}

fn main() {
    let device = Default::default();
    let client = cubecl::wgpu::WgpuRuntime::client(&device);

    type R = cubecl::wgpu::WgpuRuntime;

    let input_data = (0..16).collect::<Vec<u32>>();
    println!("Input: {:?}", &input_data);

    let num_elements = input_data.len();
    let zeros = vec![0u32; num_elements];
    let input_data_gpu = client.create(u32::as_bytes(&input_data));
    let output_data_gpu = client.create(u32::as_bytes(&zeros));

    const BLOCK_SIZE: usize = 8;
    let num_blocks = num_elements / BLOCK_SIZE;

    unsafe {
        kernel_plane_broadcast::launch::<R>(
            &client,
            CubeCount::Static(num_blocks as u32, 1, 1),
            CubeDim::new(BLOCK_SIZE as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&input_data_gpu, num_elements, 1),
            ArrayArg::from_raw_parts::<u32>(&output_data_gpu, num_elements, 1),
        )
    }

    let result = client.read_one(output_data_gpu.clone());
    let output = u32::from_bytes(&result).to_vec();
    println!("Plane Broadcast: {:?}", output);
    println!("\nNote: Each element has the value from thread #7 added to it");
    println!("Block 1: values 0-7, thread #7 has value 7, so output is [0+7, 1+7, ..., 7+7]");
    println!("Block 2: values 8-15, thread #7 has value 15, so output is [8+15, 9+15, ..., 15+15]");
}
