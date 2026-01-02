//! Scale Numbers - Passing scalar values to kernels
//!
//! This example demonstrates how to pass scalar values to a CubeCL kernel
//! using `ScalarArg`, enabling reusable code for multiplying array elements
//! by arbitrary values instead of hardcoded constants.

use cubecl::prelude::*;
use cubecl::server::Handle;

#[cube]
fn do_scale(input: &Array<u32>, scale: u32) -> u32 {
    // ABSOLUTE_POS is equivalent to CUBE_POS * CUBE_DIM + UNIT_POS
    let index = ABSOLUTE_POS;
    input[index] * scale
}

#[cube(launch)]
fn kernel_scale_numbers(input_data: &Array<u32>, scale: u32, output_data: &mut Array<u32>) {
    // ABSOLUTE_POS is equivalent to CUBE_POS * CUBE_DIM + UNIT_POS
    let index = ABSOLUTE_POS;
    output_data[index] = do_scale(input_data, scale);
}

fn launch_scale_numbers_kernel<R: Runtime>(
    client: &ComputeClient<R::Server>,
    input: &Handle,
    scale: u32,
    output: &Handle,
    num_elements: usize,
) {
    unsafe {
        kernel_scale_numbers::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(num_elements as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(input, num_elements, 1),
            ScalarArg::new(scale),
            ArrayArg::from_raw_parts::<u32>(output, num_elements, 1),
        )
    }
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

    let scale = 3;
    launch_scale_numbers_kernel::<R>(&client, &input_data_gpu, scale, &output_data_gpu, num_elements);

    let result = client.read_one(output_data_gpu.clone());
    let output = u32::from_bytes(&result).to_vec();
    println!("Scaled by {}: {:?}", scale, output);
}
