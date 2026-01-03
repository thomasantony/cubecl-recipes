//! Plane Ballot - Collecting boolean values across threads
//!
//! This example demonstrates using `plane_ballot` to collect boolean values
//! from all threads in a plane into a bitmask. Each thread contributes one bit
//! to the result based on whether its boolean condition is true or false.
//!
//! The result is a `Line<u32>` where each u32 holds up to 32 bits:
//! - For 32-thread planes: only `result[0]` is meaningful
//! - For 64-thread planes: `result[0]` has threads 0-31, `result[1]` has threads 32-63

use cubecl::prelude::*;

/// Demonstrates plane_ballot with an alternating pattern (even threads = true)
#[cube(launch)]
pub fn kernel_plane_ballot(output: &mut Array<u32>) {
    // Each thread votes true if its position is even
    let ballot_result = plane_ballot(UNIT_POS % 2 == 0);

    // Thread 0 writes the result
    // For 32 threads with even positions true: 0b01010101... = 0x55555555
    if UNIT_POS == 0 {
        output[0] = ballot_result[0];
    }
}

/// Demonstrates plane_ballot where all threads vote true
#[cube(launch)]
pub fn kernel_plane_ballot_all_true(output: &mut Array<u32>) {
    let ballot_result = plane_ballot(true);

    if UNIT_POS == 0 {
        output[0] = ballot_result[0];
    }
}

/// Demonstrates plane_ballot with a threshold condition
#[cube(launch)]
pub fn kernel_plane_ballot_threshold(output: &mut Array<u32>) {
    // First 8 threads vote true
    let ballot_result = plane_ballot(UNIT_POS < 8);

    if UNIT_POS == 0 {
        output[0] = ballot_result[0];
    }
}

fn main() {
    let device = Default::default();
    let client = cubecl::wgpu::WgpuRuntime::client(&device);

    type R = cubecl::wgpu::WgpuRuntime;

    const BLOCK_SIZE: usize = 32;
    let num_elements = BLOCK_SIZE;
    let zeros = vec![0u32; num_elements];
    let output_data_gpu = client.create(u32::as_bytes(&zeros));

    // Test 1: Alternating pattern (even threads = true)
    println!("=== Plane Ballot: Even threads ===");
    println!("Condition: UNIT_POS % 2 == 0");
    unsafe {
        kernel_plane_ballot::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(BLOCK_SIZE as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&output_data_gpu, num_elements, 1),
        )
    }
    let result = client.read_one(output_data_gpu.clone());
    let output = u32::from_bytes(&result).to_vec();
    println!("Result: {:#010x} (binary: {:032b})", output[0], output[0]);
    println!("Expected: 0x55555555 (alternating bits)\n");

    // Test 2: All threads true
    println!("=== Plane Ballot: All true ===");
    println!("Condition: true");
    let output_data_gpu = client.create(u32::as_bytes(&zeros));
    unsafe {
        kernel_plane_ballot_all_true::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(BLOCK_SIZE as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&output_data_gpu, num_elements, 1),
        )
    }
    let result = client.read_one(output_data_gpu.clone());
    let output = u32::from_bytes(&result).to_vec();
    println!("Result: {:#010x} (binary: {:032b})", output[0], output[0]);
    println!("Expected: 0xffffffff (all bits set)\n");

    // Test 3: First 8 threads true
    println!("=== Plane Ballot: First 8 threads ===");
    println!("Condition: UNIT_POS < 8");
    let output_data_gpu = client.create(u32::as_bytes(&zeros));
    unsafe {
        kernel_plane_ballot_threshold::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(BLOCK_SIZE as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&output_data_gpu, num_elements, 1),
        )
    }
    let result = client.read_one(output_data_gpu.clone());
    let output = u32::from_bytes(&result).to_vec();
    println!("Result: {:#010x} (binary: {:032b})", output[0], output[0]);
    println!("Expected: 0x000000ff (first 8 bits set)");
}
