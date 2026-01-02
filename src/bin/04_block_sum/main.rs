//! Block Exclusive Sum - Prefix sums across an entire block
//!
//! This example demonstrates a more sophisticated algorithm that performs
//! prefix sums across entire blocks using shared memory. It combines:
//! 1. Local scan within each plane
//! 2. Accumulate plane totals in shared memory
//! 3. Perform exclusive scan on totals
//! 4. Apply offsets back to each plane's local scans

use cubecl::prelude::*;

#[cube(launch)]
fn kernel_block_exclusive_sum(
    input_data: &Array<u32>,
    output_data: &mut Array<u32>,
    #[comptime] num_planes: u32,
) {
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
    // Size is determined at kernel compile time via comptime parameter
    let mut shared_totals = SharedMemory::<u32>::new(num_planes);

    // 1. Local scan within plane
    let original = input_data[thread_idx];
    let local_scan = plane_exclusive_sum(original);

    // 2. Plane totals -> shared memory
    let plane_total =
        plane_shuffle(local_scan, plane_size - 1) + plane_shuffle(original, plane_size - 1);
    if plane_thread_idx == 0 {
        shared_totals[plane_idx] = plane_total;
    }
    sync_cube();

    // 3. Scan totals (single plane or serial)
    if plane_idx == 0 && plane_thread_idx < num_planes {
        let offset = plane_exclusive_sum(shared_totals[plane_thread_idx]);
        shared_totals[plane_thread_idx] = offset;
    }
    sync_cube();

    // 4. Apply offset from previous planes
    let result = local_scan + shared_totals[plane_idx];

    output_data[block_id * CUBE_DIM + thread_id] = result;
}

fn main() {
    let device = Default::default();
    let client = cubecl::wgpu::WgpuRuntime::client(&device);

    type R = cubecl::wgpu::WgpuRuntime;

    // Use larger data that spans multiple planes
    let input_data = vec![1u32; 64];
    println!("Input: {:?}", &input_data);

    let num_elements = input_data.len();
    let zeros = vec![0u32; num_elements];
    let input_data_gpu = client.create(u32::as_bytes(&input_data));
    let output_data_gpu = client.create(u32::as_bytes(&zeros));

    const BLOCK_SIZE: usize = 64;
    const NUM_PLANES: u32 = (BLOCK_SIZE / PLANE_DIM as usize) as u32;
    let num_blocks = num_elements / BLOCK_SIZE;

    unsafe {
        kernel_block_exclusive_sum::launch::<R>(
            &client,
            CubeCount::Static(num_blocks as u32, 1, 1),
            CubeDim::new(BLOCK_SIZE as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&input_data_gpu, num_elements, 1),
            ArrayArg::from_raw_parts::<u32>(&output_data_gpu, num_elements, 1),
            NUM_PLANES,
        )
    }

    let result = client.read_one(output_data_gpu.clone());
    let output = u32::from_bytes(&result).to_vec();
    println!("Block Exclusive Sum: {:?}", output);
    println!(
        "\nNote: This computes prefix sum across the entire block of {} elements",
        BLOCK_SIZE
    );
    println!("For input of all 1s, output is [0, 1, 2, 3, ..., 63]");
}
