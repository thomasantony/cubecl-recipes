use cubecl::prelude::*;
use cubecl::server::Handle;

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

    let plane_size = if CUBE_DIM < PLANE_DIM {
        CUBE_DIM
    } else {
        PLANE_DIM
    };

    let local_data = input_data[block_id * CUBE_DIM + thread_id];
    // Get the value of local_data from thread #7 (zero-indexed) in the plane/warp
    // let last_num_in_plane = plane_broadcast(local_data, 7);
    let last_num_in_plane = plane_shuffle(local_data, plane_size - 1);

    // Add it to the current value
    output_data[block_id * CUBE_DIM + thread_id] = local_data + last_num_in_plane;
}

/// Performs exclusive sum over all elements in a block, using plane primitives
#[cube(launch)]
fn kernel_block_exclusive_sum(input_data: &Array<u32>, output_data: &mut Array<u32>) {
    let num_planes: u32 = CUBE_DIM / PLANE_DIM;
    let block_id = CUBE_POS;
    let thread_id = UNIT_POS;
    let plane_thread_idx = UNIT_POS_PLANE;
    let plane_idx = thread_id / PLANE_DIM;

    let thread_idx = block_id * CUBE_DIM + thread_id;
    let plane_size = if CUBE_DIM < PLANE_DIM {
        CUBE_DIM
    } else {
        PLANE_DIM
    };

    // Define shared memory for inter-plane communication
    let mut shared_totals = SharedMemory::<u32>::new(2);

    // 1. local scan
    let original = input_data[thread_idx];
    let local_scan = plane_exclusive_sum(original);

    // 2. plane totals → shared mem
    let plane_total =
        plane_shuffle(local_scan, plane_size - 1) + plane_shuffle(original, plane_size - 1);
    if plane_thread_idx == 0 {
        shared_totals[plane_idx] = plane_total;
    }
    sync_cube();

    // 3. scan totals (single plane or serial)
    if plane_idx == 0 && plane_thread_idx < num_planes {
        let offset = plane_exclusive_sum(shared_totals[plane_thread_idx]);
        shared_totals[plane_thread_idx] = offset;
    }
    sync_cube();

    // 4. apply offset
    let result = local_scan + shared_totals[plane_idx];

    output_data[block_id * CUBE_DIM + thread_id] = result;
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

fn main() {
    let device = Default::default();
    let client = cubecl::wgpu::WgpuRuntime::client(&device);

    type R = cubecl::wgpu::WgpuRuntime;

    let input_data = (1..11).collect::<Vec<u32>>();
    println!("Input: {:?}", &input_data);
    let num_elements = input_data.len();
    let zeros = vec![0u32; num_elements];
    let input_data_gpu = client.create(u32::as_bytes(&input_data));
    let output_data_gpu = client.create(u32::as_bytes(&zeros));

    launch_double_numbers_kernel::<R>(&client, &input_data_gpu, 3, &output_data_gpu, num_elements);
    let result = client.read_one(output_data_gpu.clone());
    let output = u32::from_bytes(&result).to_vec();
    println!("Scale numbers kernel: {:?}\n", output);

    // Use larger data array for exclusive sum
    let input_data = vec![1u32; 16];
    println!("Input: {:?}", &input_data);
    let num_elements = input_data.len();
    let zeros = vec![0u32; num_elements];
    let input_data_gpu = client.create(u32::as_bytes(&input_data));
    let output_data_gpu = client.create(u32::as_bytes(&zeros));

    const SMALL_BLOCK_SIZE: usize = 8;
    let num_blocks = num_elements / SMALL_BLOCK_SIZE;
    unsafe {
        kernel_plane_exclusive_sum::launch::<R>(
            &client,
            CubeCount::Static(num_blocks as u32, 1, 1),
            CubeDim::new(SMALL_BLOCK_SIZE as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&input_data_gpu, num_elements, 1),
            ArrayArg::from_raw_parts::<u32>(&output_data_gpu, num_elements, 1),
        )
    }
    let result = client.read_one(output_data_gpu.clone());
    let output = u32::from_bytes(&result).to_vec();
    println!("Plane Exclusive Sum: {:?}\n", output);

    let input_data = vec![1u32; 64];
    println!("Input: {:?}", &input_data);
    let num_elements = input_data.len();
    let input_data_gpu = client.create(u32::as_bytes(&input_data));
    let zeros = vec![0u32; num_elements];
    let output_data_gpu = client.create(u32::as_bytes(&zeros));
    const BIG_BLOCK_SIZE: usize = 64;
    let num_blocks = num_elements / BIG_BLOCK_SIZE;
    unsafe {
        kernel_block_exclusive_sum::launch::<R>(
            &client,
            CubeCount::Static(num_blocks as u32, 1, 1),
            CubeDim::new(BIG_BLOCK_SIZE as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&input_data_gpu, num_elements, 1),
            ArrayArg::from_raw_parts::<u32>(&output_data_gpu, num_elements, 1),
        )
    }
    let result = client.read_one(output_data_gpu.clone());
    let output = u32::from_bytes(&result).to_vec();
    println!("Block Exclusive Sum: {:?}\n", output);

    let input_data = (0..16).collect::<Vec<u32>>();
    println!("Input: {:?}", &input_data);
    let num_elements = input_data.len();
    let input_data_gpu = client.create(u32::as_bytes(&input_data));
    let zeros = vec![0u32; num_elements];
    let output_data_gpu = client.create(u32::as_bytes(&zeros));
    let num_blocks = num_elements / SMALL_BLOCK_SIZE;
    unsafe {
        kernel_plane_broadcast::launch::<R>(
            &client,
            CubeCount::Static(num_blocks as u32, 1, 1),
            CubeDim::new(SMALL_BLOCK_SIZE as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&input_data_gpu, num_elements, 1),
            ArrayArg::from_raw_parts::<u32>(&output_data_gpu, num_elements, 1),
        )
    }
    let result = client.read_one(output_data_gpu.clone());
    let output = u32::from_bytes(&result).to_vec();
    println!("Plane Broadcast: {:?}", output);
}
