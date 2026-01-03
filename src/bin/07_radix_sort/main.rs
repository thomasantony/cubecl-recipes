// tile size of 15? -> each thread will process 15 elements

const TILE_SIZE: usize = 15; // matching original impl
const TILE_SIZE_U32: u32 = TILE_SIZE as u32;

const BITS_PER_PASS: u32 = 8;
const NUM_DIGITS: usize = 32 / (BITS_PER_PASS as usize);
const HISTOGRAM_SIZE: usize = 1 << BITS_PER_PASS;
const HISTOGRAM_SIZE_U32: u32 = HISTOGRAM_SIZE as u32;

const HISTOGRAM_BLOCK_SIZE: usize = 256;
// const PREFIX_BLOCK_SIZE: usize = HISTOGRAM_SIZE / 2;
const PREFIX_BLOCK_SIZE: usize = HISTOGRAM_SIZE;
const SCATTER_BLOCK_SIZE: usize = 256;
const SCATTER_BLOCK_KVS: usize = SCATTER_BLOCK_SIZE * TILE_SIZE;

const HISTOGRAM_BLOCK_SIZE_U32: u32 = HISTOGRAM_BLOCK_SIZE as u32;

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
    PartitionMaskInfoLaunch::<R>::new(
        ScalarArg::new(0b01 << 30), // mask_invalid = 0x40000000
        ScalarArg::new(0b11 << 30), // mask_reduction = 0xC0000000
        ScalarArg::new(0b00 << 30), // mask_prefix = 0x00000000
    )
}

// Copy keys from global memory to local memory (use coalesced access for tiles)
#[cube]
fn fill_kv(input_data: &Array<u32>, kv: &mut Array<u32>, num_elements: u32) {
    let tile_start_idx: u32 = CUBE_POS * HISTOGRAM_BLOCK_SIZE_U32 * TILE_SIZE_U32;

    // Each thread will process TILE_SIZE elements
    for i in 0u32..comptime!(TILE_SIZE as u32) {
        // Stride to allow memory coalescing
        //
        // [ x x x ....  x     | y y  .........  ]
        //   ^th0        ^th255  ^th0
        //     ^th1                ^th1
        //  <---- tile 0 -----> <--- tile 1 ---->
        //
        let idx = tile_start_idx + i * TILE_SIZE_U32 + UNIT_POS;
        if idx < num_elements {
            kv[i] = input_data[idx];
        } else {
            kv[i] = 0xFFFFFFFFu32;
        }
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
fn kernel_calc_histogram(
    input_data: &Array<u32>,
    histograms: &mut Array<Atomic<u32>>,
    num_elements: u32,
) {
    // Allocate registers/local memory for the current "tile" of keys
    let mut kv = Array::<u32>::new(comptime!(TILE_SIZE as u32));

    fill_kv(input_data, &mut kv, num_elements);

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
const STATUS_BITS: u32 = 30;

#[cube]
fn popcount(x: u32) -> u32 {
    let mut v = x;
    v = v - ((v >> 1u32) & 0x55555555u32);
    v = (v & 0x33333333u32) + ((v >> 2u32) & 0x33333333u32);
    v = (v + (v >> 4u32)) & 0x0F0F0F0Fu32;
    v = v + (v >> 8u32);
    v = v + (v >> 16u32);
    v & 0x3Fu32
}

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
                // Update kr to workgroup-global rank
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
    // - kr[j] contains workgroup-global rank (1-indexed) for each key
}

#[cube]
fn decoupled_lookback(
    histograms: &Array<Atomic<u32>>,           // global histogram buffer
    partition_status: &mut Array<Atomic<u32>>, // storage for partition information
    smem: &SharedMemory<Atomic<u32>>,          // local histogram from Step 3
    scatter_smem: &mut SharedMemory<u32>,      // to cache global prefix
    pass: u32,
    partition_mask_info: &PartitionMaskInfo,
) {
    // Partition status lives after the global histograms
    let partition_offset = UNIT_POS;
    let partition_base = CUBE_POS * HISTOGRAM_SIZE_U32;

    if CUBE_POS == 0 {
        // First block: read directly from pre-computed global prefix
        if UNIT_POS < HISTOGRAM_SIZE_U32 {
            let hist_offset = pass * HISTOGRAM_SIZE_U32 + UNIT_POS;
            let global_prefix = Atomic::load(&histograms[hist_offset]);
            let local_prefix = Atomic::load(&smem[UNIT_POS]);

            // Cache global prefix for Step 8
            scatter_smem[UNIT_POS] = global_prefix;

            // Publish inclusive prefix with PREFIX status
            let inc = global_prefix + local_prefix;
            Atomic::store(
                &partition_status[partition_offset],
                inc | partition_mask_info.mask_prefix,
            );
        }
    } else {
        // All the later blocks

        // Publish local reduction first (so later workgroups can see us)
        if UNIT_POS < HISTOGRAM_SIZE_U32 {
            let local_reduction = Atomic::load(&smem[UNIT_POS]);
            Atomic::store(
                &partition_status[partition_base],
                local_reduction | partition_mask_info.mask_reduction,
            );
        }

        // Look back to compute exclusive prefix
        if UNIT_POS < HISTOGRAM_SIZE_U32 {
            let mut global_prefix: u32 = 0;
            let mut partition_base_prev = partition_base - HISTOGRAM_SIZE_U32;
            let mut done = false;

            // Spin until we find a PREFIX
            while !done {
                let prev = Atomic::load(&partition_status[partition_base_prev + partition_offset]);
                let status = prev & STATUS_MASK;

                if status != partition_mask_info.mask_invalid {
                    // Valid entry - accumulate count
                    global_prefix += prev & COUNT_MASK;

                    if status == partition_mask_info.mask_prefix {
                        // Found a prefix - we're done
                        done = true;
                    } else {
                        // It's a reduction - keep looking back
                        partition_base_prev -= HISTOGRAM_SIZE_U32;
                    }
                }
                // If INVALID, just loop again (spin-wait)
            }

            // Cache global prefix for Step 8
            scatter_smem[UNIT_POS] = global_prefix;

            // Upgrade our REDUCTION to PREFIX
            // Adding (exc | 0x40000000) flips REDUCTION (0b01) to PREFIX (0b10)
            // because 0b01 + 0b01 = 0b10

            Atomic::add(
                &partition_status[partition_offset + partition_base],
                global_prefix | (1u32 << 30),
            );
        }
    }

    sync_cube();

    // After this:
    // - scatter_smem[digit] = global exclusive prefix for this workgroup
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
    smem: &SharedMemory<Atomic<u32>>, // now contains local prefix sums
    pass: u32,
) {
    for j in 0..TILE_SIZE_U32 {
        let digit = extract_digit(kv[j], pass);
        let local_prefix = Atomic::load(&smem[digit]); // where this digit starts in tile
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
    let lid = UNIT_POS_X;

    // --- Scatter keys to sorted positions ---
    #[unroll]
    for j in 0..TILE_SIZE_U32 {
        let local_idx = kr[j] >> 16; // destination (1-indexed)
        reorder_smem[local_idx - 1] = kv[j]; // -1 for 0-indexed
    }

    sync_cube();

    // --- Gather keys back in linear order ---
    #[unroll]
    for j in 0..TILE_SIZE_U32 {
        let smem_idx = lid + j * CUBE_DIM;
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

    // --- Gather kr back in linear order ---
    #[unroll]
    for j in 0..TILE_SIZE_U32 {
        let smem_idx = lid + j * CUBE_DIM;
        kr[j] = reorder_smem[smem_idx];
    }

    sync_cube();
}

/// Step 8: Convert local rank to global output index
///
/// Before: kv[j] = keys in digit-sorted order within tile
///         kr[j] = rank within digit group (1-indexed)
///         scatter_smem[digit] = global exclusive prefix for this workgroup
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
        let digit = extract_digit(kv[j], pass);
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
    #[comptime] pass: u32,
    #[comptime] num_elements: u32,
    #[comptime] num_planes: u32,
) {
    // Allocate registers/local memory for the current "tile" of keys
    let mut kv = Array::<u32>::new(comptime!(TILE_SIZE_U32));

    // kr will hold (count | rank) for current tile's digits
    let mut kr = Array::<u32>::new(comptime!(TILE_SIZE_U32));

    // Step 1: Read keys from global memory
    fill_kv(keys_in, &mut kv, num_elements);

    // Step 2: Compute local count + rank for each item in tile and store in kr
    compute_local_count_and_rank(&kv, &mut kr, pass);

    // Shared memory for block-level histogram

    // Histogram: atomic during accumulation, becomes local prefix after step 5
    let histogram_smem = SharedMemory::<Atomic<u32>>::new(HISTOGRAM_SIZE_U32);

    // Global prefixes from look-back (step 4), read in step 8
    let mut global_prefix_smem = SharedMemory::<u32>::new(HISTOGRAM_SIZE_U32);

    // Reorder buffer (step 7 only) - could alias histogram_smem if you're clever
    let mut reorder_smem = SharedMemory::<u32>::new(TILE_SIZE_U32);

    // Step 3: Accumulate local histogram
    accumulate_local_histogram(&kv, &mut kr, &histogram_smem, pass);

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

    local_to_global_index(&kv, &mut kr, &global_prefix_smem, pass);

    store_to_global(&kv, &kr, keys_out);
}

fn main() {
    let device = Default::default();
    let client = cubecl::wgpu::WgpuRuntime::client(&device);

    type R = cubecl::wgpu::WgpuRuntime;

    let input_data = vec![1u32; 3840 * 2];
    let num_elements = input_data.len();

    const ELEM_SIZE: usize = std::mem::size_of::<u32>();
    let input_data_gpu = client.create(u32::as_bytes(&input_data));
    let output_data_gpu = client.empty(num_elements * ELEM_SIZE);
    let histo_buffer = client.empty(NUM_DIGITS * HISTOGRAM_SIZE * ELEM_SIZE);

    let num_blocks = num_elements.div_ceil(HISTOGRAM_BLOCK_SIZE);

    unsafe {
        kernel_calc_histogram::launch::<R>(
            &client,
            CubeCount::Static(num_blocks as u32, 1, 1),
            CubeDim::new(HISTOGRAM_BLOCK_SIZE as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&input_data_gpu, num_elements, 1),
            ArrayArg::from_raw_parts::<u32>(&histo_buffer, NUM_DIGITS * HISTOGRAM_SIZE, 1),
            ScalarArg::new(num_elements as u32),
        );
    }

    let num_planes = PREFIX_BLOCK_SIZE as u32 / PLANE_DIM;
    unsafe {
        kernel_prefix_histogram::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(PREFIX_BLOCK_SIZE as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&histo_buffer, NUM_DIGITS * HISTOGRAM_SIZE, 1),
            num_planes,
        );
    }
    println!("Calling scatter kernel");

    let partition_buffer = client.empty(num_blocks * HISTOGRAM_SIZE * ELEM_SIZE);
    unsafe {
        kernel_scatter::launch::<R>(
            &client,
            CubeCount::Static(num_blocks as u32, 1, 1),
            CubeDim::new(SCATTER_BLOCK_SIZE as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&input_data_gpu, num_elements, 1),
            ArrayArg::from_raw_parts::<u32>(&output_data_gpu, num_elements, 1),
            ArrayArg::from_raw_parts::<u32>(&histo_buffer, NUM_DIGITS * HISTOGRAM_SIZE, 1),
            ArrayArg::from_raw_parts::<u32>(&partition_buffer, num_blocks * HISTOGRAM_SIZE, 1),
            even_partition_mask_info::<R>(),
            0u32,
            num_elements as u32,
            num_planes,
        );
    }

    let result = client.read_one(output_data_gpu.clone());
    let output = u32::from_bytes(&result).to_vec();

    println!("Histo for pass 0: {:?}", &output[..256]);
    println!("Histo for pass 1: {:?}", &output[256..512]);
}
