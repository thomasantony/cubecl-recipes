//! Hierarchical Exclusive Sum - Prefix sums for arbitrarily large arrays
//!
//! This implements a recursive hierarchical scan that works on arrays of any size:
//! 1. Block-level scans using plane intrinsics + shared memory
//! 2. Recursively scan block totals to compute block offsets
//! 3. Add block offsets back to each element

use cubecl::prelude::*;
use cubecl::server::Handle;

const BLOCK_SIZE: usize = 64;
const NUM_PLANES: u32 = (BLOCK_SIZE as u32 + PLANE_DIM - 1) / PLANE_DIM; // div_ceil

/// Describes a level in the hierarchical scan
struct LevelInfo {
    /// Number of elements to scan at this level
    num_elements: usize,
    /// Number of blocks (and thus block totals) produced
    num_blocks: usize,
}

#[cube(launch)]
fn kernel_block_exclusive_sum(
    input_data: &Array<u32>,
    output_data: &mut Array<u32>,
    block_totals: &mut Array<u32>,
    #[comptime] num_planes: u32,
) {
    let block_id = CUBE_POS;
    let thread_id = UNIT_POS;
    let plane_thread_idx = UNIT_POS_PLANE;
    let plane_idx = thread_id / PLANE_DIM;

    let plane_size = if CUBE_DIM < PLANE_DIM {
        CUBE_DIM
    } else {
        PLANE_DIM
    };

    // Define shared memory for inter-plane communication
    // Size is determined at kernel compile time via comptime parameter
    let mut shared_totals = SharedMemory::<u32>::new(num_planes);

    // 1. Local scan within plane
    let original = input_data[ABSOLUTE_POS];
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
    output_data[ABSOLUTE_POS] = result;

    // Save the block total
    if thread_id == CUBE_DIM - 1 {
        block_totals[block_id] = result + original;
    }
}

#[cube(launch)]
fn kernel_add_block_offsets(block_scan_results: &mut Array<u32>, block_offsets: &Array<u32>) {
    let block_id = CUBE_POS;
    block_scan_results[ABSOLUTE_POS] += block_offsets[block_id];
}

fn launch_block_sum<R: Runtime>(
    client: &ComputeClient<R::Server>,
    input_gpu: &Handle,
    output_gpu: &Handle,
    block_totals_gpu: &Handle,
    num_elements: usize,
) {
    let num_blocks = num_elements.div_ceil(BLOCK_SIZE);
    unsafe {
        kernel_block_exclusive_sum::launch::<R>(
            client,
            CubeCount::Static(num_blocks as u32, 1, 1),
            CubeDim::new(BLOCK_SIZE as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(input_gpu, num_elements, 1),
            ArrayArg::from_raw_parts::<u32>(output_gpu, num_elements, 1),
            ArrayArg::from_raw_parts::<u32>(block_totals_gpu, num_blocks, 1),
            NUM_PLANES,
        )
    }
}

fn launch_add_offsets<R: Runtime>(
    client: &ComputeClient<R::Server>,
    output_gpu: &Handle,
    block_offsets_gpu: &Handle,
    num_elements: usize,
) {
    let num_blocks = num_elements.div_ceil(BLOCK_SIZE);
    unsafe {
        kernel_add_block_offsets::launch::<R>(
            client,
            CubeCount::Static(num_blocks as u32, 1, 1),
            CubeDim::new(BLOCK_SIZE as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(output_gpu, num_elements, 1),
            ArrayArg::from_raw_parts::<u32>(block_offsets_gpu, num_blocks, 1),
        );
    }
}

/// Compute buffer sizes needed for each level of the hierarchical scan.
fn compute_level_sizes(num_elements: usize) -> Vec<LevelInfo> {
    let mut levels = vec![];
    let mut n = num_elements;
    while n > BLOCK_SIZE {
        let num_blocks = n.div_ceil(BLOCK_SIZE);
        levels.push(LevelInfo { num_elements: n, num_blocks });
        n = num_blocks;
    }
    // Final level that fits in one block
    levels.push(LevelInfo { num_elements: n, num_blocks: 1 });
    levels
}

/// Performs a hierarchical exclusive scan on an array of any size.
/// Pre-allocates all intermediate buffers and processes iteratively.
fn hierarchical_scan<R: Runtime>(
    client: &ComputeClient<R::Server>,
    input_gpu: &Handle,
    output_gpu: &Handle,
    num_elements: usize,
) {
    let levels = compute_level_sizes(num_elements);
    let num_levels = levels.len();

    if num_levels == 1 {
        // Base case: fits in a single block
        let block_totals_gpu = client.empty(std::mem::size_of::<u32>());
        launch_block_sum::<R>(client, input_gpu, output_gpu, &block_totals_gpu, num_elements);
        return;
    }

    // Calculate total scratch space needed for block_totals and block_offsets
    // Each level (except last) needs space for its block totals
    let total_scratch_elements: usize = levels
        .iter()
        .take(num_levels - 1)
        .map(|level| level.num_blocks)
        .sum();
    let elem_size = std::mem::size_of::<u32>();

    // Allocate two scratch buffers: one for totals, one for offsets
    let totals_buffer = client.empty(total_scratch_elements * elem_size);
    let offsets_buffer = client.empty(total_scratch_elements * elem_size);

    // Create handles for each level's region within the scratch buffers
    let mut totals_handles = vec![];
    let mut offsets_handles = vec![];
    let mut byte_offset = 0u64;
    for level in levels.iter().take(num_levels - 1) {
        let level_bytes = (level.num_blocks * elem_size) as u64;
        totals_handles.push(
            totals_buffer
                .clone()
                .offset_start(byte_offset)
                .offset_end(totals_buffer.size() - byte_offset - level_bytes),
        );
        offsets_handles.push(
            offsets_buffer
                .clone()
                .offset_start(byte_offset)
                .offset_end(offsets_buffer.size() - byte_offset - level_bytes),
        );
        byte_offset += level_bytes;
    }

    // === Up-sweep: compute block totals at each level ===
    // Level 0: scan input -> output, save totals
    launch_block_sum::<R>(client, input_gpu, output_gpu, &totals_handles[0], levels[0].num_elements);

    // Remaining levels: scan previous totals -> current offsets (as temp), save new totals
    for i in 1..num_levels - 1 {
        launch_block_sum::<R>(
            client,
            &totals_handles[i - 1],
            &offsets_handles[i - 1], // Use as temp output for the scan
            &totals_handles[i],
            levels[i].num_elements,
        );
    }

    // Final level: scan into offsets (no more totals needed)
    let last_idx = num_levels - 2;
    let final_totals_gpu = client.empty(elem_size);
    launch_block_sum::<R>(
        client,
        &totals_handles[last_idx],
        &offsets_handles[last_idx],
        &final_totals_gpu,
        levels[num_levels - 1].num_elements,
    );

    // === Down-sweep: add offsets back at each level ===
    for i in (0..num_levels - 1).rev() {
        if i > 0 {
            // Add offsets to the previous level's offset buffer
            launch_add_offsets::<R>(client, &offsets_handles[i - 1], &offsets_handles[i], levels[i].num_elements);
        } else {
            // Final step: add offsets to the main output
            launch_add_offsets::<R>(client, output_gpu, &offsets_handles[0], levels[0].num_elements);
        }
    }
}

fn main() {
    let device = Default::default();
    let client = cubecl::wgpu::WgpuRuntime::client(&device);

    type R = cubecl::wgpu::WgpuRuntime;

    // Test with data that requires multiple levels of recursion
    // With BLOCK_SIZE=64: 262144 elements = 4096 blocks = 64 blocks = 1 top-level block (3 levels)
    let input_data = vec![1u32; 262144];
    println!("Input: [{}, {}, {}, ... {} elements total]", input_data[0], input_data[1], input_data[2], input_data.len());

    let num_elements = input_data.len();

    let input_data_gpu = client.create(u32::as_bytes(&input_data));
    let output_data_gpu = client.empty(num_elements * std::mem::size_of::<u32>());

    hierarchical_scan::<R>(&client, &input_data_gpu, &output_data_gpu, num_elements);

    let result = client.read_one(output_data_gpu.clone());
    let output = u32::from_bytes(&result).to_vec();

    println!("Global Exclusive Sum: [{}, {}, {}, ... {}]", output[0], output[1], output[2], output[output.len()-1]);

    // Verify correctness
    let expected: Vec<u32> = input_data
        .iter()
        .scan(0u32, |acc, &x| {
            let result = *acc;
            *acc += x;
            Some(result)
        })
        .collect();

    if output == expected {
        println!("\n✓ Result matches expected exclusive scan!");
    } else {
        println!("\n✗ Mismatch!");
        println!("Expected: {:?}", expected);
    }
}
