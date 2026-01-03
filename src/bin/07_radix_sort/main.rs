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

const RS_PARTITION_MASK_STATUS: u32 = 0xC0000000u32; // top 2 bits
const RS_PARTITION_MASK_COUNT: u32 = 0x3FFFFFFFu32; // bottom 30 bits

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

#[cube(launch)]
fn kernel_scatter(
    keys_in: &Array<u32>,
    keys_out: &mut Array<u32>,
    histograms: &Array<u32>,
    #[comptime] num_elements: u32,
    #[comptime] pass: u32,
) {
    // Allocate registers/local memory for the current "tile" of keys
    let mut kv = Array::<u32>::new(comptime!(TILE_SIZE_U32));

    // kr will hold (count | rank) for current tile's digits
    let mut kr = Array::<u32>::new(comptime!(TILE_SIZE_U32));

    let lane_mask_lt = Line::new((1u32 << UNIT_POS_PLANE) - 1u32);

    fill_kv(keys_in, &mut kv, num_elements);
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

    // Shared memory for block-level histogram
    let smem: SharedMemory<Atomic<u32>> = SharedMemory::<Atomic<u32>>::new(HISTOGRAM_SIZE_U32);

    if UNIT_POS < HISTOGRAM_SIZE_U32 {
        Atomic::store(&smem[UNIT_POS], 0u32);
    }
    sync_cube();
}

fn main() {
    let device = Default::default();
    let client = cubecl::wgpu::WgpuRuntime::client(&device);

    type R = cubecl::wgpu::WgpuRuntime;

    let input_data = vec![1; 2 * 3840];
    let num_elements = input_data.len();

    const ELEM_SIZE: usize = std::mem::size_of::<u32>();
    let input_data_gpu = client.create(u32::as_bytes(&input_data));
    let output_data_gpu = client.empty(num_elements * ELEM_SIZE);
    let histo_buffer = client.empty(NUM_DIGITS * HISTOGRAM_SIZE * ELEM_SIZE);

    let num_blocks = num_elements.div_ceil(HISTOGRAM_BLOCK_SIZE);

    println!("num_blocks = {}", num_blocks);

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
    let result = client.read_one(histo_buffer.clone());
    let output = u32::from_bytes(&result).to_vec();

    println!("Histo for pass 0: {:?}", &output[..256]);
    println!("Histo for pass 1: {:?}", &output[256..512]);
}
