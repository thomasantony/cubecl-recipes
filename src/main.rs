use cubecl::prelude::*;
use cubecl::server::Handle;

#[cube]
pub fn exclusive_sum(input_data: &Array<u32>, output_data: &mut Array<u32>) {}

#[cube(launch)]
fn kernel_plane_exclusive_sum(input_data: &Array<u32>, output_data: &mut Array<u32>) {
    let block_id = CUBE_POS;
    let thread_id = UNIT_POS;

    let local_data = input_data[block_id * CUBE_DIM + thread_id];
    let local_sum = plane_exclusive_sum(local_data);

    output_data[block_id * CUBE_DIM + thread_id] = local_sum;
}

#[cube(launch)]
fn kernel_plane_broadcast(input_data: &Array<u32>, output_data: &mut Array<u32>) {
    let block_id = CUBE_POS;
    let thread_id = UNIT_POS;

    let local_data = input_data[block_id * CUBE_DIM + thread_id];
    let local_sum = plane_exclusive_sum(local_data);

    output_data[block_id * CUBE_DIM + thread_id] = local_sum;
}

#[cube(launch)]
fn kernel_double_numbers(input_data: &Array<u32>, scale: u32, output_data: &mut Array<u32>) {
    let block_id = CUBE_POS;
    let thread_id = UNIT_POS;

    let index = block_id * CUBE_DIM + thread_id;
    output_data[index] = input_data[index] * scale;
}

fn launch_double_numbers_kernel<R: Runtime>(
    client: &ComputeClient<R::Server>,
    input: &Handle,
    scale: u32,
    output: &Handle,
    num_elements: usize,
) {
    unsafe {
        kernel_double_numbers::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(num_elements as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(input, num_elements, 1),
            ScalarArg::new(scale),
            ArrayArg::from_raw_parts::<u32>(output, num_elements, 1),
        )
    }
}

fn launch_plane_exclusive_sum<R: Runtime>(
    client: &ComputeClient<R::Server>,
    input: &Handle,
    output: &Handle,
    num_elements: usize,
    block_size: usize,
) {
    let num_blocks = num_elements / block_size;
    unsafe {
        kernel_plane_exclusive_sum::launch::<R>(
            &client,
            CubeCount::Static(num_blocks as u32, 1, 1),
            CubeDim::new(block_size as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(input, num_elements, 1),
            ArrayArg::from_raw_parts::<u32>(output, num_elements, 1),
        )
    }
}

fn main() {
    let device = Default::default();
    let client = cubecl::wgpu::WgpuRuntime::client(&device);

    let input_data = (1..11).collect::<Vec<u32>>();
    println!("Input: {:?}", &input_data);
    let num_elements = input_data.len();
    let zeros = vec![0u32; num_elements];
    let input_data_gpu = client.create(u32::as_bytes(&input_data));
    let output_data_gpu = client.create(u32::as_bytes(&zeros));

    launch_double_numbers_kernel::<cubecl::wgpu::WgpuRuntime>(
        &client,
        &input_data_gpu,
        3,
        &output_data_gpu,
        num_elements,
    );
    let result = client.read_one(output_data_gpu.clone());
    let output = u32::from_bytes(&result).to_vec();
    println!("Scale numbers kernel: {:?}\n", output);

    // Use larger data array for exclusive sum
    let input_data = vec![1u32; 64];
    println!("Input: {:?}", &input_data);
    let num_elements = input_data.len();
    let zeros = vec![0u32; num_elements];
    let input_data_gpu = client.create(u32::as_bytes(&input_data));
    let output_data_gpu = client.create(u32::as_bytes(&zeros));
    launch_plane_exclusive_sum::<cubecl::wgpu::WgpuRuntime>(
        &client,
        &input_data_gpu,
        &output_data_gpu,
        num_elements,
        32,
    );
    let result = client.read_one(output_data_gpu.clone());
    let output = u32::from_bytes(&result).to_vec();
    println!("Plane Exclusive Sum: {:?}", output);
}
