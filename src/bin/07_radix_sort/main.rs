// GPU Radix Sort using CubeCL
// Based on wgpu_sort / Fuchsia implementation with decoupled look-back

const TILE_SIZE: usize = 15;
const TILE_SIZE_U32: u32 = TILE_SIZE as u32;

const BITS_PER_PASS: u32 = 8;
const NUM_DIGITS: usize = 32 / (BITS_PER_PASS as usize);
const HISTOGRAM_SIZE: usize = 1 << BITS_PER_PASS;
const HISTOGRAM_SIZE_U32: u32 = HISTOGRAM_SIZE as u32;

const HISTOGRAM_BLOCK_SIZE: usize = 256;
const PREFIX_BLOCK_SIZE: usize = HISTOGRAM_SIZE;
const SCATTER_BLOCK_SIZE: usize = 256;
const SCATTER_BLOCK_KVS: usize = SCATTER_BLOCK_SIZE * TILE_SIZE;

use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;

#[cube]
fn extract_digit(pass: u32, value: u32) -> u32 {
    (value >> (pass * BITS_PER_PASS)) & ((1u32 << BITS_PER_PASS) - 1u32)
}

#[derive(CubeType, CubeLaunch)]
struct PartitionMaskInfo {
    pub mask_invalid: u32,
    pub mask_reduction: u32,
    pub mask_prefix: u32,
}

// Status bits occupy top 2 bits (30-31)
// Even pass: after writing, bits are 0b00, 0b01, or 0b10
// Odd pass:  after writing, bits are 0b10, 0b11, or 0b00
// This way odd pass sees even's PREFIX (0b10) as INVALID, no zeroing needed

fn even_partition_mask_info<'a, R: Runtime>() -> PartitionMaskInfoLaunch<'a, R> {
    PartitionMaskInfoLaunch::<R>::new(
        ScalarArg::new(0b00 << 30), // mask_invalid = 0x00000000
        ScalarArg::new(0b01 << 30), // mask_reduction = 0x40000000
        ScalarArg::new(0b10 << 30), // mask_prefix = 0x80000000
    )
}

fn odd_partition_mask_info<'a, R: Runtime>() -> PartitionMaskInfoLaunch<'a, R> {
    // Odd's INVALID (0b10) equals Even's PREFIX, so odd passes see even's leftover as invalid
    PartitionMaskInfoLaunch::<R>::new(
        ScalarArg::new(0b10 << 30), // mask_invalid = 0x80000000
        ScalarArg::new(0b11 << 30), // mask_reduction = 0xC0000000
        ScalarArg::new(0b00 << 30), // mask_prefix = 0x00000000
    )
}

// Copy keys from global memory to local memory
// Uses plane-based loading for coalescing while maintaining stability
#[cube]
fn fill_kv(input_data: &Array<u32>, kv: &mut Array<u32>) {
    let block_keyvals = CUBE_DIM * TILE_SIZE_U32;
    let plane_keyvals = PLANE_DIM * TILE_SIZE_U32;

    let plane_id = UNIT_POS / PLANE_DIM;
    let plane_lane = UNIT_POS % PLANE_DIM;

    // Each plane gets a contiguous block of plane_keyvals elements
    // Within the plane, threads stride by PLANE_DIM for coalescing
    let kv_in_offset = CUBE_POS * block_keyvals + plane_id * plane_keyvals + plane_lane;

    for i in 0u32..comptime!(TILE_SIZE as u32) {
        let idx = kv_in_offset + i * PLANE_DIM;
        kv[i] = input_data[idx];
    }
}

#[cube]
fn histogram_pass(
    pass: u32,
    kv: &Array<u32>,
    smem_histogram: &SharedMemory<Atomic<u32>>,
    histograms: &Array<Atomic<u32>>,
) {
    // zero the memory first
    if UNIT_POS < comptime!(HISTOGRAM_SIZE_U32) {
        Atomic::store(&smem_histogram[UNIT_POS], 0u32);
    }
    sync_cube();

    // Use shared memory to accumulate histogram for current "tile"
    for j in 0..TILE_SIZE_U32 {
        let digit = extract_digit(pass, kv[j]);
        Atomic::add(&smem_histogram[digit], 1u32);
    }

    sync_cube();

    // Flush shared → global (one bin per thread)
    if UNIT_POS < HISTOGRAM_SIZE_U32 {
        Atomic::add(
            &histograms[HISTOGRAM_SIZE_U32 * pass + UNIT_POS],
            Atomic::load(&smem_histogram[UNIT_POS]),
        );
    }
}

#[cube(launch)]
fn kernel_calc_histogram(input_data: &Array<u32>, histograms: &mut Array<Atomic<u32>>) {
    // Allocate registers/local memory for the current "tile" of keys
    let mut kv = Array::<u32>::new(comptime!(TILE_SIZE as u32));

    fill_kv(input_data, &mut kv);

    let smem: SharedMemory<Atomic<u32>> = SharedMemory::<Atomic<u32>>::new(HISTOGRAM_SIZE_U32);

    histogram_pass(3u32, &kv, &smem, &histograms);
    histogram_pass(2u32, &kv, &smem, &histograms);
    histogram_pass(1u32, &kv, &smem, &histograms);
    histogram_pass(0u32, &kv, &smem, &histograms);
}

#[cube]
fn prefix_histogram_pass(pass: u32, histograms: &mut Array<u32>, #[comptime] num_planes: u32) {
    let plane_idx = UNIT_POS / PLANE_DIM;

    let digit_idx = HISTOGRAM_SIZE_U32 * pass + ABSOLUTE_POS;
    // Define shared memory for inter-plane communication
    // Size is determined at kernel compile time via comptime parameter
    let mut shared_totals = SharedMemory::<u32>::new(num_planes);

    // 1. Local scan within plane
    let original = histograms[digit_idx];
    let local_scan = plane_exclusive_sum(original);

    // 2. Plane totals -> shared memory
    let plane_total =
        plane_shuffle(local_scan, PLANE_DIM - 1) + plane_shuffle(original, PLANE_DIM - 1);
    if UNIT_POS_PLANE == 0 {
        shared_totals[plane_idx] = plane_total;
    }
    sync_cube();

    // 3. Scan totals (single plane or serial)
    if plane_idx == 0 && UNIT_POS_PLANE < num_planes {
        let offset = plane_exclusive_sum(shared_totals[UNIT_POS_PLANE]);
        shared_totals[UNIT_POS_PLANE] = offset;
    }
    sync_cube();

    // 4. Apply offset from previous planes
    let result = local_scan + shared_totals[plane_idx];

    histograms[digit_idx] = result;
}

// Assume prefix_histogram is launched with 256-thread blocks
#[cube(launch)]
fn kernel_prefix_histogram(histograms: &mut Array<u32>, #[comptime] num_planes: u32) {
    prefix_histogram_pass(0u32, histograms, num_planes);
    prefix_histogram_pass(1u32, histograms, num_planes);
    prefix_histogram_pass(2u32, histograms, num_planes);
    prefix_histogram_pass(3u32, histograms, num_planes);
}

const STATUS_MASK: u32 = 0xC0000000u32; // top 2 bits
const COUNT_MASK: u32 = 0x3FFFFFFFu32; // bottom 30 bits

#[cube]
fn compute_local_count_and_rank(kv: &Array<u32>, kr: &mut Array<u32>, pass: u32) {
    let lane_mask_lt = Line::new((1u32 << UNIT_POS_PLANE) - 1u32);

    for j in 0..TILE_SIZE_U32 {
        let digit = extract_digit(pass, kv[j]);

        // Build match_mask: bit j is set iff lane j has same digit as me

        let mut match_mask = Line::<u32>::new(0xFFFFFFFFu32);
        // Unrolled for 8-bit digit
        #[unroll]
        for bit in 0u32..8u32 {
            let my_bit = (digit >> bit) & 1u32;
            let predicate = my_bit == 1u32;
            let ballot = plane_ballot(predicate);

            let mask_for_bit = Line::new(ballot[0] ^ (0u32 - (1u32 - my_bit)));
            match_mask = match_mask & mask_for_bit;
        }

        // count = total lanes with same digit
        let count = match_mask.count_ones()[0];

        // rank = how many matching lanes have lower lane_id than me, plus 1
        // (1-indexed to match the WGSL implementation)
        match_mask = match_mask & lane_mask_lt;
        let rank = match_mask.count_ones()[0] + 1;

        kr[j] = (count << 16) | rank;
    }
}

#[cube]
fn accumulate_local_histogram(
    kv: &Array<u32>,
    kr: &mut Array<u32>,
    smem: &SharedMemory<Atomic<u32>>,
    pass: u32,
) {
    // First zero it out
    if UNIT_POS < HISTOGRAM_SIZE_U32 {
        Atomic::store(&smem[UNIT_POS], 0u32);
    }

    sync_cube();

    let plane_id = UNIT_POS / PLANE_DIM;
    let num_planes = CUBE_DIM / PLANE_DIM;

    // Sequential - each "plane" waits for the previous one to finish (hence the sync_cube())
    for i in 0..num_planes {
        if plane_id == i {
            for j in 0..TILE_SIZE_U32 {
                let digit = extract_digit(pass, kv[j]);
                let prev = Atomic::load(&smem[digit]);

                // Separate out rank and count from kr[j]
                let rank = kr[j] & 0xFFFF;
                let count = kr[j] >> 16;
                // Update kr to block-global rank
                kr[j] = prev + rank;

                // The rank will equal count in the last thread with this digit (since we used 1-based rank)
                // Only the last thread with this digit updates histogram
                //
                if rank == count {
                    Atomic::store(&smem[digit], prev + count);
                }
            }
        }
        sync_cube();
    }

    // Now:
    // - smem[digit] contains local histogram (total count per digit in this tile)
    // - kr[j] contains block-global rank (1-indexed) for each key
}

#[cube]
fn decoupled_lookback(
    histograms: &Array<Atomic<u32>>,            // global histogram buffer
    partition_status: &mut Array<Atomic<u32>>,  // storage for partition information
    smem_histogram: &SharedMemory<Atomic<u32>>, // local histogram from Step 3
    global_prefix_smem: &mut SharedMemory<u32>, // to cache global prefix
    pass: u32,
    partition_mask_info: &PartitionMaskInfo,
) {
    let partition_base = CUBE_POS * HISTOGRAM_SIZE_U32;

    if CUBE_POS == 0 {
        // First block: read directly from pre-computed global prefix
        if UNIT_POS < HISTOGRAM_SIZE_U32 {
            let hist_offset = pass * HISTOGRAM_SIZE_U32 + UNIT_POS;
            let global_prefix = Atomic::load(&histograms[hist_offset]);
            let local_prefix = Atomic::load(&smem_histogram[UNIT_POS]);

            // Cache global prefix for Step 8
            global_prefix_smem[UNIT_POS] = global_prefix;

            // Publish inclusive prefix with PREFIX status
            let inc = global_prefix + local_prefix;
            // partition_base is zero here since CUBE_POS is zero
            Atomic::store(
                &partition_status[UNIT_POS],
                inc | partition_mask_info.mask_prefix,
            );
        }
    } else {
        // All the later blocks

        // Publish local reduction first (so later blocks can see us)
        if UNIT_POS < HISTOGRAM_SIZE_U32 {
            let local_reduction = Atomic::load(&smem_histogram[UNIT_POS]);
            Atomic::store(
                &partition_status[partition_base + UNIT_POS],
                local_reduction | partition_mask_info.mask_reduction,
            );
        }

        // Look back to compute exclusive prefix
        if UNIT_POS < HISTOGRAM_SIZE_U32 {
            let mut global_prefix: u32 = 0;
            // We need partitions from the previous block
            let mut partition_base_prev = partition_base - HISTOGRAM_SIZE_U32;

            // Spin until we find a PREFIX
            loop {
                let prev = Atomic::load(&partition_status[partition_base_prev + UNIT_POS]);
                let status = prev & STATUS_MASK;

                if status != partition_mask_info.mask_invalid {
                    // Valid entry - accumulate count
                    global_prefix += prev & COUNT_MASK;

                    if status == partition_mask_info.mask_prefix {
                        // Found a prefix - we're done
                        break;
                    } else {
                        // It's a reduction - keep looking back
                        partition_base_prev -= HISTOGRAM_SIZE_U32;
                    }
                }
                // If INVALID, just loop again (spin-wait)
            }

            // Cache global prefix for Step 8
            global_prefix_smem[UNIT_POS] = global_prefix;

            // Upgrade our REDUCTION to PREFIX
            // Adding (exc | 0x40000000) flips REDUCTION (0b01) to PREFIX (0b10)
            // because 0b01 + 0b01 = 0b10

            Atomic::add(
                &partition_status[UNIT_POS + partition_base],
                global_prefix | (1u32 << 30),
            );
        }
    }

    sync_cube();

    // After this:
    // - global_prefix_smem[digit] = global exclusive prefix for this block
    // - smem[digit] = local histogram (unchanged, needed for Step 5)
}

#[cube]
fn local_prefix_sum(smem: &SharedMemory<Atomic<u32>>, #[comptime] num_planes: u32) {
    let lid = UNIT_POS;
    let plane_idx = lid / PLANE_DIM;

    // Inter-plane communication
    let mut shared_totals = SharedMemory::<u32>::new(num_planes);

    // 1. Local scan within plane
    let original = Atomic::load(&smem[lid]);
    let local_scan = plane_exclusive_sum(original);

    // 2. Plane totals -> shared memory
    let plane_total =
        plane_shuffle(local_scan, PLANE_DIM - 1) + plane_shuffle(original, PLANE_DIM - 1);

    if UNIT_POS_PLANE == 0 {
        shared_totals[plane_idx] = plane_total;
    }
    sync_cube();

    // 3. Scan totals (single plane or serial)
    if plane_idx == 0 && UNIT_POS_PLANE < num_planes {
        let offset = plane_exclusive_sum(shared_totals[UNIT_POS_PLANE]);
        shared_totals[UNIT_POS_PLANE] = offset;
    }
    sync_cube();

    // 4. Apply offset from previous planes
    let result = local_scan + shared_totals[plane_idx];
    Atomic::store(&smem[lid], result);

    sync_cube();
}

#[cube]
fn rank_to_local_index(
    kv: &Array<u32>,
    kr: &mut Array<u32>,
    histogram_smem: &SharedMemory<Atomic<u32>>, // now contains local prefix sums
    pass: u32,
) {
    for j in 0..TILE_SIZE_U32 {
        let digit = extract_digit(pass, kv[j]);
        let local_prefix = Atomic::load(&histogram_smem[digit]); // where this digit starts in tile
        let rank = kr[j]; // block-global rank (1-indexed)

        let local_idx = local_prefix + rank; // position within tile (1-indexed)

        kr[j] = rank | (local_idx << 16); // pack both
    }
}

/// Step 7: Reorder keys within tile using shared memory
///
/// Before: kv[j] = keys in original load order
///         kr[j] = (local_idx << 16) | rank
/// After:  kv[j] = keys in digit-sorted order within tile
///         kr[j] = rank only (lower 16 bits)
#[cube]
fn reorder_in_shared_memory(
    kv: &mut Array<u32>,
    kr: &mut Array<u32>,
    reorder_smem: &mut SharedMemory<u32>, // size = tile_size
) {
    // Gather pattern must match fill_kv's loading pattern (plane-based)
    let plane_keyvals = PLANE_DIM * TILE_SIZE_U32;
    let plane_id = UNIT_POS / PLANE_DIM;
    let plane_lane = UNIT_POS % PLANE_DIM;
    let base_idx = plane_id * plane_keyvals + plane_lane;

    // --- Scatter keys to sorted positions ---
    #[unroll]
    for j in 0..TILE_SIZE_U32 {
        let local_idx = kr[j] >> 16; // destination (1-indexed)
        reorder_smem[local_idx - 1] = kv[j]; // -1 for 0-indexed
    }

    sync_cube();

    // --- Gather keys back in plane-strided order ---
    #[unroll]
    for j in 0..TILE_SIZE_U32 {
        let smem_idx = base_idx + j * PLANE_DIM;
        kv[j] = reorder_smem[smem_idx];
    }

    sync_cube();

    // --- Scatter kr to sorted positions in reorder_smem (need rank for Step 8) ---
    #[unroll]
    for j in 0..TILE_SIZE_U32 {
        let local_idx = kr[j] >> 16;
        reorder_smem[local_idx - 1] = kr[j] & 0xFFFF; // store just rank
    }

    sync_cube();

    // --- Gather kr back in plane-strided order ---
    #[unroll]
    for j in 0..TILE_SIZE_U32 {
        let smem_idx = base_idx + j * PLANE_DIM;
        kr[j] = reorder_smem[smem_idx];
    }

    sync_cube();
}

/// Step 8: Convert local rank to global output index
///
/// Before: kv[j] = keys in digit-sorted order within tile
///         kr[j] = rank within digit group (1-indexed)
///         scatter_smem[digit] = global exclusive prefix for this block
/// After:  kr[j] = final global output index
#[cube]
fn local_to_global_index(
    kv: &Array<u32>,
    kr: &mut Array<u32>,
    scatter_smem: &SharedMemory<u32>, // global prefixes from Step 4
    pass: u32,
) {
    #[unroll]
    for j in 0..TILE_SIZE_U32 {
        let digit = extract_digit(pass, kv[j]);
        let global_prefix = scatter_smem[digit];
        let rank = kr[j]; // 1-indexed

        kr[j] = global_prefix + rank - 1; // 0-indexed global position
    }
}

/// Step 9: Store keys to global memory
#[cube]
fn store_to_global(kv: &Array<u32>, kr: &Array<u32>, keys_out: &mut Array<u32>) {
    #[unroll]
    for j in 0..TILE_SIZE_U32 {
        let global_idx = kr[j];
        keys_out[global_idx] = kv[j];
    }
}

#[cube(launch)]
fn kernel_scatter(
    keys_in: &Array<u32>,
    keys_out: &mut Array<u32>,
    histograms: &Array<Atomic<u32>>,
    partition_status: &mut Array<Atomic<u32>>,
    partition_mask_info: &PartitionMaskInfo,
    pass: u32, // Runtime parameter - changes per pass
    #[comptime] num_planes: u32,
) {
    // Allocate registers/local memory for the current "tile" of keys
    let mut kv = Array::<u32>::new(comptime!(TILE_SIZE_U32));

    // kr will hold (count | rank) for current tile's digits
    let mut kr = Array::<u32>::new(comptime!(TILE_SIZE_U32));

    // Step 1: Read keys from global memory
    fill_kv(keys_in, &mut kv);

    // Step 2: Compute local count + rank for each item in tile and store in kr
    compute_local_count_and_rank(&kv, &mut kr, pass);

    // Shared memory for block-level histogram

    // Histogram: atomic during accumulation, becomes local prefix after step 5
    let histogram_smem = SharedMemory::<Atomic<u32>>::new(HISTOGRAM_SIZE_U32);

    // Reorder buffer (step 7 only) - needs space for all keys in the tile
    let mut reorder_smem = SharedMemory::<u32>::new(comptime!(SCATTER_BLOCK_KVS as u32));

    // Step 3: Accumulate local histogram
    accumulate_local_histogram(&kv, &mut kr, &histogram_smem, pass);

    // Global prefixes from look-back (step 4), read in step 8
    let mut global_prefix_smem = SharedMemory::<u32>::new(HISTOGRAM_SIZE_U32);
    // Step 4. Decoupled Lookback
    decoupled_lookback(
        histograms,
        partition_status,
        &histogram_smem,
        &mut global_prefix_smem,
        pass,
        partition_mask_info,
    );

    // Step 5: Prefix scan of local histogram in shared memory
    // Restrict to first HISTOGRAM_SIZE_U32 threads
    if UNIT_POS < HISTOGRAM_SIZE_U32 {
        local_prefix_sum(&histogram_smem, num_planes);
    }
    sync_cube();

    // Step 6:
    rank_to_local_index(&kv, &mut kr, &histogram_smem, pass);

    // Step 7
    reorder_in_shared_memory(&mut kv, &mut kr, &mut reorder_smem);

    // Step 8
    local_to_global_index(&kv, &mut kr, &global_prefix_smem, pass);

    // Step 9
    store_to_global(&kv, &kr, keys_out);
}

/// Run one scatter pass
fn run_scatter_pass<R: Runtime>(
    client: &ComputeClient<R::Server>,
    keys_in: &Handle,
    keys_out: &Handle,
    histograms: &Handle,
    partition_status: &Handle,
    pass: u32,
    padded_size: usize,
    num_blocks: usize,
    num_planes: u32,
) {
    let partition_mask = if pass % 2 == 0 {
        even_partition_mask_info::<R>()
    } else {
        odd_partition_mask_info::<R>()
    };

    unsafe {
        kernel_scatter::launch::<R>(
            client,
            CubeCount::Static(num_blocks as u32, 1, 1),
            CubeDim::new(SCATTER_BLOCK_SIZE as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(keys_in, padded_size, 1),
            ArrayArg::from_raw_parts::<u32>(keys_out, padded_size, 1),
            ArrayArg::from_raw_parts::<u32>(histograms, NUM_DIGITS * HISTOGRAM_SIZE, 1),
            ArrayArg::from_raw_parts::<u32>(partition_status, num_blocks * HISTOGRAM_SIZE, 1),
            partition_mask,
            ScalarArg::new(pass),
            num_planes,
        );
    }
}

/// Sort u32 values using GPU radix sort.
/// Returns the sorted data.
/// Note: Input size must be a multiple of SCATTER_BLOCK_KVS (3840) for now.
pub fn radix_sort_u32<R: Runtime>(client: &ComputeClient<R::Server>, input: &[u32]) -> Vec<u32> {
    let num_elements = input.len();
    if num_elements == 0 {
        return vec![];
    }

    const ELEM_SIZE: usize = std::mem::size_of::<u32>();

    // Each block processes SCATTER_BLOCK_KVS keys (256 threads × 15 items = 3840)
    let num_blocks = num_elements.div_ceil(SCATTER_BLOCK_KVS);

    // Double-buffer for ping-pong between passes
    let keys_a = client.create(u32::as_bytes(input));
    let keys_b = client.empty(num_elements * ELEM_SIZE);

    // Zero-initialize histogram buffer (required for atomic adds)
    let histo_zeros = vec![0u32; NUM_DIGITS * HISTOGRAM_SIZE];
    let histograms = client.create(u32::as_bytes(&histo_zeros));

    // Step 1: Calculate histogram for all passes at once
    unsafe {
        kernel_calc_histogram::launch::<R>(
            client,
            CubeCount::Static(num_blocks as u32, 1, 1),
            CubeDim::new(HISTOGRAM_BLOCK_SIZE as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&keys_a, num_elements, 1),
            ArrayArg::from_raw_parts::<u32>(&histograms, NUM_DIGITS * HISTOGRAM_SIZE, 1),
        );
    }

    // Step 2: Prefix sum each pass's histogram
    let num_planes = PREFIX_BLOCK_SIZE as u32 / PLANE_DIM;
    unsafe {
        kernel_prefix_histogram::launch::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(PREFIX_BLOCK_SIZE as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&histograms, NUM_DIGITS * HISTOGRAM_SIZE, 1),
            num_planes,
        );
    }

    // Zero-initialize partition buffer (required for lookback)
    // Even passes see 0 as INVALID, odd passes see 0x80000000 as INVALID
    // The alternating mask trick means we don't need to re-zero between passes
    let partition_zeros = vec![0u32; num_blocks * HISTOGRAM_SIZE];
    let partition_status = client.create(u32::as_bytes(&partition_zeros));

    // Step 3: Run all 4 scatter passes
    // Pass 0 (even): keys_a → keys_b
    run_scatter_pass::<R>(
        client,
        &keys_a,
        &keys_b,
        &histograms,
        &partition_status,
        0,
        num_elements,
        num_blocks,
        num_planes,
    );

    // Pass 1 (odd): keys_b → keys_a
    run_scatter_pass::<R>(
        client,
        &keys_b,
        &keys_a,
        &histograms,
        &partition_status,
        1,
        num_elements,
        num_blocks,
        num_planes,
    );

    // Pass 2 (even): keys_a → keys_b
    run_scatter_pass::<R>(
        client,
        &keys_a,
        &keys_b,
        &histograms,
        &partition_status,
        2,
        num_elements,
        num_blocks,
        num_planes,
    );

    // Pass 3 (odd): keys_b → keys_a
    run_scatter_pass::<R>(
        client,
        &keys_b,
        &keys_a,
        &histograms,
        &partition_status,
        3,
        num_elements,
        num_blocks,
        num_planes,
    );

    // Final result is in keys_a
    let result = client.read_one(keys_a.clone());
    u32::from_bytes(&result).to_vec()
}

fn main() {
    let device = Default::default();
    let client = cubecl::wgpu::WgpuRuntime::client(&device);

    type R = cubecl::wgpu::WgpuRuntime;

    // Test with aligned sizes (multiples of 3840)
    let count = 7680u32; // 2 blocks
    let input_data: Vec<u32> = (1..=count).rev().collect();
    println!("Sorting {} elements...", input_data.len());

    let output = radix_sort_u32::<R>(&client, &input_data);

    // Verify
    let is_sorted = output.windows(2).all(|w| w[0] <= w[1]);

    if is_sorted {
        println!("FULLY SORTED! {} elements", output.len());
        println!("First 10: {:?}", &output[..10]);
        println!("Last 10: {:?}", &output[output.len() - 10..]);
    } else {
        println!("Sort FAILED");
        for i in 1..output.len() {
            if output[i] < output[i - 1] {
                println!("Error at {}: {} > {}", i, output[i - 1], output[i]);
                break;
            }
        }
    }
}
