//! Double Numbers - The simplest CubeCL kernel
//!
//! This example demonstrates the basic structure of a CubeCL kernel that
//! takes an array of numbers and doubles each element.

use cubecl::prelude::*;

#[cube(launch)]
fn kernel_double_numbers(input_data: &Array<u32>, output_data: &mut Array<u32>) {
    let block_id = CUBE_POS;
    let thread_id = UNIT_POS;

    let index = block_id * CUBE_DIM + thread_id;
    output_data[index] = input_data[index] * 2;
}

fn main() {
    let device = Default::default();
    let client = cubecl::wgpu::WgpuRuntime::client(&device);

    type R = cubecl::wgpu::WgpuRuntime;

    let input_data = (1..=10).collect::<Vec<u32>>();
    println!("Input: {:?}", &input_data);

    let num_elements = input_data.len();
    let zeros = vec![0u32; num_elements];
    let input_data_gpu = client.create(u32::as_bytes(&input_data));
    let output_data_gpu = client.create(u32::as_bytes(&zeros));

    unsafe {
        kernel_double_numbers::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(num_elements as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&input_data_gpu, num_elements, 1),
            ArrayArg::from_raw_parts::<u32>(&output_data_gpu, num_elements, 1),
        )
    }

    let result = client.read_one(output_data_gpu.clone());
    let output = u32::from_bytes(&result).to_vec();
    println!("Output: {:?}", output);
}
