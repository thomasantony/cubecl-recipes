// GPU Radix Sort using CubeCL (shared by the 07_radix_sort binary and the benchmarks)
//
// Based on the wgpu_sort / Fuchsia "onesweep"-style implementation, with the
// inter-workgroup prefix computed by *Decoupled Fallback* (Smith, Levien, Owens,
// SPAA '25) instead of plain decoupled look-back.
//
// Plain decoupled look-back spins until a predecessor tile publishes its data, which
// can deadlock on GPUs without forward-progress guarantees (e.g. Apple, ARM Mali):
// a waiting workgroup may occupy the resources the predecessor needs in order to run.
// With decoupled fallback, a workgroup that has spun `max_spins` times on a predecessor
// recomputes that predecessor's per-digit reduction itself (from the pass's input
// keys) and publishes it with `atomicMax`. Every workgroup therefore finishes in a
// bounded number of steps no matter how the hardware schedules workgroups.

#![allow(dead_code)]

const TILE_SIZE: usize = 15;
const TILE_SIZE_U32: u32 = TILE_SIZE as u32;

const BITS_PER_PASS: u32 = 8;
const NUM_PASSES: usize = 32 / (BITS_PER_PASS as usize);
const HISTOGRAM_SIZE: usize = 1 << BITS_PER_PASS;
const HISTOGRAM_SIZE_U32: u32 = HISTOGRAM_SIZE as u32;

const HISTOGRAM_BLOCK_SIZE: usize = 256;
const PREFIX_BLOCK_SIZE: usize = HISTOGRAM_SIZE;
const SCATTER_BLOCK_SIZE: usize = 256;
const SCATTER_BLOCK_KVS: usize = SCATTER_BLOCK_SIZE * TILE_SIZE;
const SCATTER_BLOCK_KVS_U32: u32 = SCATTER_BLOCK_KVS as u32;

/// Upper bound on planes per cube (256 threads / minimum supported plane size of 8).
const MAX_PLANES: u32 = 32;

/// Default number of times a workgroup re-polls a NOT_READY predecessor before it
/// falls back to computing that predecessor's reduction itself.
pub const DEFAULT_MAX_SPINS: u32 = 4;

// Partition status words: top 2 bits = status, low 30 bits = count.
// The encoding is monotonic (INVALID < READY < PREFIX) so that every publication can be
// done with `atomicMax`: a fallback READY never overwrites a PREFIX, and duplicate
// READY publications of the same tile are identical (reductions are deterministic).
const STATUS_INVALID: u32 = 0u32;
const STATUS_READY: u32 = 1u32 << 30;
const STATUS_PREFIX: u32 = 2u32 << 30;
const STATUS_MASK: u32 = 0xC0000000u32;
const COUNT_MASK: u32 = 0x3FFFFFFFu32;

/// Indices into the stats buffer.
const STAT_FALLBACKS: u32 = 0;
const STAT_LOOKBACK_STEPS: u32 = 1;
const NUM_STATS: usize = 2;

use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;

#[cube]
fn extract_digit(pass: u32, value: u32) -> u32 {
    (value >> (pass * BITS_PER_PASS)) & ((1u32 << BITS_PER_PASS) - 1u32)
}

/// Exclusive scan of one u32 per unit across the whole cube.
/// `totals` must hold at least CUBE_DIM / PLANE_DIM elements, and
/// CUBE_DIM / PLANE_DIM must be <= PLANE_DIM.
#[cube]
fn cube_exclusive_scan(value: u32, totals: &mut SharedMemory<u32>) -> u32 {
    let plane_idx = UNIT_POS / PLANE_DIM;
    let num_planes = CUBE_DIM / PLANE_DIM;

    // 1. Scan within each plane, publish plane totals
    let local_scan = plane_exclusive_sum(value);
    let plane_total = plane_sum(value);
    if UNIT_POS_PLANE == 0 {
        totals[plane_idx] = plane_total;
    }
    sync_cube();

    // 2. First plane scans the plane totals (each lane only touches its own slot)
    if plane_idx == 0 {
        let mut v = 0u32;
        if UNIT_POS_PLANE < num_planes {
            v = totals[UNIT_POS_PLANE];
        }
        let offset = plane_exclusive_sum(v);
        if UNIT_POS_PLANE < num_planes {
            totals[UNIT_POS_PLANE] = offset;
        }
    }
    sync_cube();

    // 3. Add offset of preceding planes
    let result = local_scan + totals[plane_idx];
    // Make `totals` safe to reuse by a subsequent call
    sync_cube();
    result
}

// Copy keys from global memory to registers (cube-strided, order irrelevant: histogram only)
#[cube]
fn fill_kv(input_data: &Array<u32>, kv: &mut Array<u32>) {
    let kv_in_offset = CUBE_POS * SCATTER_BLOCK_KVS_U32 + UNIT_POS;
    for i in 0..TILE_SIZE_U32 {
        kv[i] = input_data[kv_in_offset + i * CUBE_DIM];
    }
}

// Copy keys from global memory to registers for the scatter pass.
// Each plane loads a contiguous chunk of TILE_SIZE * PLANE_DIM keys; within the plane,
// lanes stride by PLANE_DIM. The ranking (plane, j, lane) therefore follows input order,
// which is what makes the sort stable.
#[cube]
fn fill_kv_scatter(keys: &Array<u32>, kv: &mut Array<u32>) {
    let plane_id = UNIT_POS / PLANE_DIM;
    let plane_keyvals = TILE_SIZE_U32 * PLANE_DIM;
    let kv_in_offset =
        CUBE_POS * SCATTER_BLOCK_KVS_U32 + plane_id * plane_keyvals + UNIT_POS_PLANE;

    for i in 0..TILE_SIZE_U32 {
        kv[i] = keys[kv_in_offset + i * PLANE_DIM];
    }
}

#[cube]
fn histogram_pass(
    pass: u32,
    kv: &Array<u32>,
    smem_histogram: &SharedMemory<Atomic<u32>>,
    histograms: &Array<Atomic<u32>>,
) {
    Atomic::store(&smem_histogram[UNIT_POS], 0u32);
    sync_cube();

    for j in 0..TILE_SIZE_U32 {
        let digit = extract_digit(pass, kv[j]);
        Atomic::add(&smem_histogram[digit], 1u32);
    }
    sync_cube();

    // Flush shared -> global (one bin per thread)
    Atomic::add(
        &histograms[HISTOGRAM_SIZE_U32 * pass + UNIT_POS],
        Atomic::load(&smem_histogram[UNIT_POS]),
    );
    // Protect smem_histogram before the next pass zeroes it
    sync_cube();
}

/// Computes the global digit histograms for all passes, and zeroes the partition
/// status words (one region per pass) used by the scatter passes.
#[cube(launch_unchecked)]
fn kernel_calc_histogram(
    input_data: &Array<u32>,
    histograms: &mut Array<Atomic<u32>>,
    partition_status: &mut Array<Atomic<u32>>,
) {
    let region = CUBE_COUNT * HISTOGRAM_SIZE_U32;
    let slot = CUBE_POS * HISTOGRAM_SIZE_U32 + UNIT_POS;
    for p in 0..comptime!(NUM_PASSES as u32) {
        Atomic::store(&partition_status[p * region + slot], STATUS_INVALID);
    }

    let mut kv = Array::<u32>::new(TILE_SIZE_U32);
    fill_kv(input_data, &mut kv);

    let smem = SharedMemory::<Atomic<u32>>::new(HISTOGRAM_SIZE_U32);
    histogram_pass(3u32, &kv, &smem, histograms);
    histogram_pass(2u32, &kv, &smem, histograms);
    histogram_pass(1u32, &kv, &smem, histograms);
    histogram_pass(0u32, &kv, &smem, histograms);
}

/// Zeroes the global histograms (NUM_PASSES * 256) and stats. Single cube of 256 units.
#[cube(launch_unchecked)]
fn kernel_clear(histograms: &mut Array<u32>, stats: &mut Array<u32>) {
    for p in 0..comptime!(NUM_PASSES as u32) {
        histograms[p * HISTOGRAM_SIZE_U32 + UNIT_POS] = 0u32;
    }
    if UNIT_POS < comptime!(NUM_STATS as u32) {
        stats[UNIT_POS] = 0u32;
    }
}

/// Turns each pass's global histogram into its exclusive prefix sum (single cube of 256).
#[cube(launch_unchecked)]
fn kernel_prefix_histogram(histograms: &mut Array<u32>) {
    let mut totals = SharedMemory::<u32>::new(MAX_PLANES);
    for pass in 0..comptime!(NUM_PASSES as u32) {
        let idx = HISTOGRAM_SIZE_U32 * pass + UNIT_POS;
        let scanned = cube_exclusive_scan(histograms[idx], &mut totals);
        histograms[idx] = scanned;
    }
}

/// Step 2: per key, count/rank among lanes of the plane holding the same digit.
///
/// kr[j] = (last_lane << 16) | (count << 8) | rank
///   count     = number of lanes in the plane with the same digit
///   rank      = 1-based rank among those lanes (by lane id)
///   last_lane = lane id of the highest lane with the same digit
#[cube]
fn compute_local_count_and_rank(kv: &Array<u32>, kr: &mut Array<u32>, pass: u32) {
    let lane_mask_lt = Line::new((1u32 << UNIT_POS_PLANE) - 1u32);
    let mut active_lanes = 0xFFFFFFFFu32;
    if PLANE_DIM < 32u32 {
        active_lanes = (1u32 << PLANE_DIM) - 1u32;
    }

    for j in 0..TILE_SIZE_U32 {
        let digit = extract_digit(pass, kv[j]);

        // match_mask: bit l is set iff lane l has the same digit as me
        let mut match_mask = Line::<u32>::new(active_lanes);
        #[unroll]
        for bit in 0u32..BITS_PER_PASS {
            let my_bit = (digit >> bit) & 1u32;
            let ballot = plane_ballot(my_bit == 1u32);
            // keep lanes whose bit equals mine
            match_mask = match_mask & Line::new(ballot[0] ^ (0u32 - (1u32 - my_bit)));
        }

        let count = match_mask.count_ones()[0];
        let last_lane = 31u32 - match_mask.leading_zeros()[0];
        let rank = (match_mask & lane_mask_lt).count_ones()[0] + 1u32;

        kr[j] = (last_lane << 16) | (count << 8) | rank;
    }
}

/// Step 3: build the tile's histogram in shared memory and turn kr into the
/// 1-based rank of each key among all keys of the tile with the same digit.
///
/// Planes are processed in order (separated by cube barriers), and within a plane
/// rows j are processed in order, so ranks follow input order (stable).
#[cube]
fn accumulate_local_histogram(
    kv: &Array<u32>,
    kr: &mut Array<u32>,
    smem: &SharedMemory<Atomic<u32>>,
    pass: u32,
) {
    Atomic::store(&smem[UNIT_POS], 0u32);
    sync_cube();

    let plane_id = UNIT_POS / PLANE_DIM;
    let num_planes = CUBE_DIM / PLANE_DIM;

    for i in 0..num_planes {
        if plane_id == i {
            for j in 0..TILE_SIZE_U32 {
                let digit = extract_digit(pass, kv[j]);
                let packed = kr[j];
                let rank = packed & 0xFFu32;
                let count = (packed >> 8) & 0xFFu32;
                let last_lane = packed >> 16;

                // Exactly one lane per (row, digit) bumps the shared counter; the other
                // lanes with that digit receive the previous value via a shuffle. The
                // shuffle also orders this row's update before the next row's.
                let mut prev = 0u32;
                if rank == count {
                    prev = Atomic::add(&smem[digit], count);
                }
                let base = plane_shuffle(prev, last_lane);
                kr[j] = base + rank;
            }
        }
        sync_cube();
    }
    // smem[digit] = tile histogram, kr[j] = 1-based rank of key j within its digit
}

/// Step 4: Decoupled fallback.
///
/// Computes, for every digit, the number of keys with that digit in all preceding
/// tiles (exclusive prefix) and publishes this tile's inclusive prefix.
/// Thread `UNIT_POS` handles digit `UNIT_POS`, but all threads step back through the
/// predecessors together so that the whole cube can cooperate on a fallback.
#[cube]
fn decoupled_fallback(
    keys_in: &Array<u32>,
    histograms: &Array<Atomic<u32>>,
    partition_status: &mut Array<Atomic<u32>>,
    stats: &mut Array<Atomic<u32>>,
    smem_histogram: &SharedMemory<Atomic<u32>>,
    global_prefix_smem: &mut SharedMemory<u32>,
    pass: u32,
    #[comptime] max_spins: u32,
    #[comptime] post_reduction: bool,
) {
    let tid = UNIT_POS;
    let tile = CUBE_POS;
    let region = pass * CUBE_COUNT * HISTOGRAM_SIZE_U32;
    let local = Atomic::load(&smem_histogram[tid]);
    let mut exclusive = 0u32;

    if tile > 0u32 {
        // Publish our own reduction so successors can use it.
        // (`post_reduction = false` is a test mode that withholds it, forcing successors
        // into the fallback path.)
        if comptime!(post_reduction) {
            Atomic::max(
                &partition_status[region + tile * HISTOGRAM_SIZE_U32 + tid],
                STATUS_READY | local,
            );
        }

        // ctrl[2*s] = number of threads still looking back, ctrl[2*s+1] = number of threads
        // that saw INVALID. Three rotating slots allow a single barrier per iteration:
        // the slot reset in iteration k was last read in iteration k-2 (before barrier k-1),
        // and is next incremented in iteration k+1 (after barrier k).
        let ctrl = SharedMemory::<Atomic<u32>>::new(6u32);
        let fallback_hist = SharedMemory::<Atomic<u32>>::new(HISTOGRAM_SIZE_U32);
        if tid < 6u32 {
            Atomic::store(&ctrl[tid], 0u32);
        }
        sync_cube();

        let mut done = false;
        let mut lookback_tile = tile - 1u32;
        let mut spins: u32 = 0u32;
        let mut slot = 0u32;

        loop {
            let next_slot = (slot + 1u32) % 3u32;
            if tid == 0u32 {
                Atomic::store(&ctrl[2u32 * next_slot], 0u32);
                Atomic::store(&ctrl[2u32 * next_slot + 1u32], 0u32);
            }

            let status_idx = region + lookback_tile * HISTOGRAM_SIZE_U32 + tid;
            let mut value = 0u32;
            if !done {
                Atomic::add(&ctrl[2u32 * slot], 1u32);
                value = Atomic::load(&partition_status[status_idx]);
                if (value & STATUS_MASK) == STATUS_INVALID {
                    Atomic::add(&ctrl[2u32 * slot + 1u32], 1u32);
                }
            }
            sync_cube();

            // Both values are cube-uniform (read after the barrier)
            let num_active = Atomic::load(&ctrl[2u32 * slot]);
            let num_invalid = Atomic::load(&ctrl[2u32 * slot + 1u32]);
            slot = next_slot;

            if num_active == 0u32 {
                break;
            }

            let mut ready = num_invalid == 0u32;
            if !ready {
                spins += 1u32;
                if spins > comptime!(max_spins) {
                    // FALLBACK: recompute the blocking tile's reduction from its keys.
                    Atomic::store(&fallback_hist[tid], 0u32);
                    sync_cube();

                    let tile_offset = lookback_tile * SCATTER_BLOCK_KVS_U32 + tid;
                    for i in 0..TILE_SIZE_U32 {
                        let digit = extract_digit(pass, keys_in[tile_offset + i * CUBE_DIM]);
                        Atomic::add(&fallback_hist[digit], 1u32);
                    }
                    sync_cube();

                    if tid == 0u32 {
                        Atomic::add(&stats[STAT_FALLBACKS], 1u32);
                    }

                    if !done {
                        // Insert as READY; if the slot is already READY (identical value)
                        // or PREFIX (larger), atomicMax keeps the existing value.
                        let reduction = STATUS_READY | Atomic::load(&fallback_hist[tid]);
                        let previous = Atomic::max(&partition_status[status_idx], reduction);
                        value = reduction;
                        if previous > reduction {
                            value = previous;
                        }
                    }
                    ready = true;
                }
            }

            if ready {
                spins = 0u32;
                if tid == 0u32 {
                    Atomic::add(&stats[STAT_LOOKBACK_STEPS], 1u32);
                }
                if !done {
                    exclusive += value & COUNT_MASK;
                    // Tile 0's exclusive prefix is 0, so its reduction is also its
                    // inclusive prefix.
                    if (value & STATUS_MASK) == STATUS_PREFIX || lookback_tile == 0u32 {
                        done = true;
                    }
                }
                if lookback_tile > 0u32 {
                    lookback_tile -= 1u32;
                }
            }
        }
    }

    // Publish our inclusive prefix (counts relative to the start of each digit bucket)
    Atomic::max(
        &partition_status[region + tile * HISTOGRAM_SIZE_U32 + tid],
        STATUS_PREFIX | (exclusive + local),
    );

    // Global start of this tile's run of each digit
    global_prefix_smem[tid] =
        Atomic::load(&histograms[pass * HISTOGRAM_SIZE_U32 + tid]) + exclusive;
    sync_cube();
}

/// Step 5: exclusive prefix sum of the tile histogram (in place)
#[cube]
fn local_prefix_sum(smem: &SharedMemory<Atomic<u32>>, totals: &mut SharedMemory<u32>) {
    let scanned = cube_exclusive_scan(Atomic::load(&smem[UNIT_POS]), totals);
    Atomic::store(&smem[UNIT_POS], scanned);
    sync_cube();
}

/// Step 6: kr[j] = (local_idx << 16) | rank, local_idx = 1-based position within tile
#[cube]
fn rank_to_local_index(
    kv: &Array<u32>,
    kr: &mut Array<u32>,
    histogram_smem: &SharedMemory<Atomic<u32>>,
    pass: u32,
) {
    for j in 0..TILE_SIZE_U32 {
        let digit = extract_digit(pass, kv[j]);
        let digit_start = Atomic::load(&histogram_smem[digit]);
        let rank = kr[j];
        kr[j] = rank | ((digit_start + rank) << 16);
    }
}

/// Step 7: Reorder keys within the tile through shared memory, so that the global
/// stores in step 9 are (mostly) contiguous.
///
/// After: kv[j] = keys in digit-sorted order within tile, kr[j] = rank within digit
#[cube]
fn reorder_in_shared_memory(
    kv: &mut Array<u32>,
    kr: &mut Array<u32>,
    reorder_smem: &mut SharedMemory<u32>,
) {
    let plane_keyvals = PLANE_DIM * TILE_SIZE_U32;
    let plane_id = UNIT_POS / PLANE_DIM;
    let base_idx = plane_id * plane_keyvals + UNIT_POS_PLANE;

    #[unroll]
    for j in 0..TILE_SIZE_U32 {
        reorder_smem[(kr[j] >> 16) - 1u32] = kv[j];
    }
    sync_cube();

    #[unroll]
    for j in 0..TILE_SIZE_U32 {
        kv[j] = reorder_smem[base_idx + j * PLANE_DIM];
    }
    sync_cube();

    #[unroll]
    for j in 0..TILE_SIZE_U32 {
        reorder_smem[(kr[j] >> 16) - 1u32] = kr[j] & 0xFFFFu32;
    }
    sync_cube();

    #[unroll]
    for j in 0..TILE_SIZE_U32 {
        kr[j] = reorder_smem[base_idx + j * PLANE_DIM];
    }
    sync_cube();
}

/// Step 8 + 9: compute global index and store
#[cube]
fn store_to_global(
    kv: &Array<u32>,
    kr: &Array<u32>,
    global_prefix_smem: &SharedMemory<u32>,
    keys_out: &mut Array<u32>,
    pass: u32,
) {
    #[unroll]
    for j in 0..TILE_SIZE_U32 {
        let digit = extract_digit(pass, kv[j]);
        let global_idx = global_prefix_smem[digit] + kr[j] - 1u32;
        keys_out[global_idx] = kv[j];
    }
}

#[cube(launch_unchecked)]
fn kernel_scatter(
    keys_in: &Array<u32>,
    keys_out: &mut Array<u32>,
    histograms: &Array<Atomic<u32>>,
    partition_status: &mut Array<Atomic<u32>>,
    stats: &mut Array<Atomic<u32>>,
    pass: u32,
    #[comptime] max_spins: u32,
    #[comptime] post_reduction: bool,
) {
    let mut kv = Array::<u32>::new(TILE_SIZE_U32);
    let mut kr = Array::<u32>::new(TILE_SIZE_U32);

    let histogram_smem = SharedMemory::<Atomic<u32>>::new(HISTOGRAM_SIZE_U32);
    let mut global_prefix_smem = SharedMemory::<u32>::new(HISTOGRAM_SIZE_U32);
    let mut reorder_smem = SharedMemory::<u32>::new(SCATTER_BLOCK_KVS_U32);
    let mut scan_totals = SharedMemory::<u32>::new(MAX_PLANES);

    // Step 1: Read keys
    fill_kv_scatter(keys_in, &mut kv);

    // Step 2: Rank within plane
    compute_local_count_and_rank(&kv, &mut kr, pass);

    // Step 3: Tile histogram + rank within tile
    accumulate_local_histogram(&kv, &mut kr, &histogram_smem, pass);

    // Step 4: Inter-tile prefix via decoupled fallback
    decoupled_fallback(
        keys_in,
        histograms,
        partition_status,
        stats,
        &histogram_smem,
        &mut global_prefix_smem,
        pass,
        max_spins,
        post_reduction,
    );

    // Step 5: Tile histogram -> tile exclusive prefix
    local_prefix_sum(&histogram_smem, &mut scan_totals);

    // Step 6
    rank_to_local_index(&kv, &mut kr, &histogram_smem, pass);

    // Step 7
    reorder_in_shared_memory(&mut kv, &mut kr, &mut reorder_smem);

    // Steps 8 + 9
    store_to_global(&kv, &kr, &global_prefix_smem, keys_out, pass);
}

/// Tunables for the decoupled-fallback scatter.
#[derive(Clone, Copy, Debug)]
pub struct SortConfig {
    /// Re-polls of a NOT_READY predecessor before falling back.
    pub max_spins: u32,
    /// If false, tiles never publish their local reduction (only their final inclusive
    /// prefix). Test-only: it forces heavy use of the fallback path.
    pub post_reduction: bool,
}

impl Default for SortConfig {
    fn default() -> Self {
        Self {
            max_spins: DEFAULT_MAX_SPINS,
            post_reduction: true,
        }
    }
}

/// Counters collected on the GPU during a sort.
#[derive(Clone, Copy, Debug, Default)]
pub struct SortStats {
    /// Number of fallback operations performed (all passes).
    pub fallbacks: u32,
    /// Number of predecessor tiles consumed during look-back (all passes).
    pub lookback_steps: u32,
}

/// Sort buffers for GPU-only radix sort (no CPU transfers during sort)
pub struct RadixSortBuffers {
    pub keys_a: Handle,
    pub keys_b: Handle,
    pub histograms: Handle,
    pub partition_status: Handle,
    pub stats: Handle,
    pub padded_size: usize,
    pub original_len: usize,
    pub num_blocks: usize,
}

fn check_plane_size<R: Runtime>(client: &ComputeClient<R::Server>) {
    let hw = &client.properties().hardware;
    assert!(
        hw.plane_size_min >= 8 && hw.plane_size_max <= 32,
        "radix sort requires a plane (subgroup) size between 8 and 32, device reports {}..={}",
        hw.plane_size_min,
        hw.plane_size_max
    );
}

/// Create reusable sort buffers for a given size.
/// Fill `keys_a` with `padded_size` keys (pad with 0xFFFFFFFF), then call
/// `radix_sort_u32_gpu`.
pub fn create_sort_buffers<R: Runtime>(
    client: &ComputeClient<R::Server>,
    max_elements: usize,
) -> RadixSortBuffers {
    const ELEM_SIZE: usize = std::mem::size_of::<u32>();
    check_plane_size::<R>(client);

    let padded_size = max_elements.max(1).next_multiple_of(SCATTER_BLOCK_KVS);
    let num_blocks = padded_size / SCATTER_BLOCK_KVS;
    assert!(
        padded_size <= COUNT_MASK as usize,
        "radix sort supports at most {} keys",
        COUNT_MASK
    );

    RadixSortBuffers {
        keys_a: client.empty(padded_size * ELEM_SIZE),
        keys_b: client.empty(padded_size * ELEM_SIZE),
        histograms: client.empty(NUM_PASSES * HISTOGRAM_SIZE * ELEM_SIZE),
        partition_status: client.empty(NUM_PASSES * num_blocks * HISTOGRAM_SIZE * ELEM_SIZE),
        stats: client.empty(NUM_STATS * ELEM_SIZE),
        padded_size,
        original_len: max_elements,
        num_blocks,
    }
}

/// Upload `keys` (at most `original_len` of them) into `buffers.keys_a`, padding with
/// 0xFFFFFFFF, which sorts to the end.
pub fn upload_keys<R: Runtime>(
    client: &ComputeClient<R::Server>,
    buffers: &mut RadixSortBuffers,
    keys: &[u32],
) {
    assert!(keys.len() <= buffers.padded_size, "too many keys for these buffers");
    let mut padded = Vec::with_capacity(buffers.padded_size);
    padded.extend_from_slice(keys);
    padded.resize(buffers.padded_size, 0xFFFFFFFF);
    buffers.keys_a = client.create(u32::as_bytes(&padded));
}

/// GPU-only radix sort: sorts the keys in `buffers.keys_a` in place.
/// Does NOT sync - read the buffer or call client.sync() afterwards.
pub fn radix_sort_u32_gpu<R: Runtime>(
    client: &ComputeClient<R::Server>,
    buffers: &RadixSortBuffers,
    config: SortConfig,
) {
    let RadixSortBuffers {
        keys_a,
        keys_b,
        histograms,
        partition_status,
        stats,
        padded_size,
        num_blocks,
        ..
    } = buffers;
    let padded_size = *padded_size;
    let num_blocks = *num_blocks;
    let histogram_len = NUM_PASSES * HISTOGRAM_SIZE;
    let status_len = NUM_PASSES * num_blocks * HISTOGRAM_SIZE;

    // Histograms are accumulated with atomics, stats are counters: both need zeroing.
    // (Partition status is zeroed by the histogram kernel.)
    unsafe {
        kernel_clear::launch_unchecked::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(HISTOGRAM_SIZE_U32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(histograms, histogram_len, 1),
            ArrayArg::from_raw_parts::<u32>(stats, NUM_STATS, 1),
        );
    }

    // Step 1: Histograms for all passes (+ zero partition status)
    unsafe {
        kernel_calc_histogram::launch_unchecked::<R>(
            client,
            CubeCount::Static(num_blocks as u32, 1, 1),
            CubeDim::new(HISTOGRAM_BLOCK_SIZE as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(keys_a, padded_size, 1),
            ArrayArg::from_raw_parts::<u32>(histograms, histogram_len, 1),
            ArrayArg::from_raw_parts::<u32>(partition_status, status_len, 1),
        );
    }

    // Step 2: Exclusive prefix sum of each pass's histogram
    unsafe {
        kernel_prefix_histogram::launch_unchecked::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(PREFIX_BLOCK_SIZE as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(histograms, histogram_len, 1),
        );
    }

    // Step 3: Scatter passes, ping-ponging between keys_a and keys_b.
    // NUM_PASSES is even, so the result ends up back in keys_a.
    for pass in 0..NUM_PASSES as u32 {
        let (src, dst) = if pass % 2 == 0 {
            (keys_a, keys_b)
        } else {
            (keys_b, keys_a)
        };
        unsafe {
            kernel_scatter::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_blocks as u32, 1, 1),
                CubeDim::new(SCATTER_BLOCK_SIZE as u32, 1, 1),
                ArrayArg::from_raw_parts::<u32>(src, padded_size, 1),
                ArrayArg::from_raw_parts::<u32>(dst, padded_size, 1),
                ArrayArg::from_raw_parts::<u32>(histograms, histogram_len, 1),
                ArrayArg::from_raw_parts::<u32>(partition_status, status_len, 1),
                ArrayArg::from_raw_parts::<u32>(stats, NUM_STATS, 1),
                ScalarArg::new(pass),
                config.max_spins,
                config.post_reduction,
            );
        }
    }
}

/// Sort u32 values using GPU radix sort, also returning GPU-side statistics.
pub fn radix_sort_u32_with_stats<R: Runtime>(
    client: &ComputeClient<R::Server>,
    input: &[u32],
    config: SortConfig,
) -> (Vec<u32>, SortStats) {
    if input.is_empty() {
        return (vec![], SortStats::default());
    }
    let mut buffers = create_sort_buffers::<R>(client, input.len());
    upload_keys::<R>(client, &mut buffers, input);

    radix_sort_u32_gpu::<R>(client, &buffers, config);

    let mut out = client.read(vec![buffers.keys_a.clone(), buffers.stats.clone()]);
    let stats_bytes = out.pop().unwrap();
    let keys_bytes = out.pop().unwrap();
    let stats = u32::from_bytes(&stats_bytes);
    (
        u32::from_bytes(&keys_bytes)[..input.len()].to_vec(),
        SortStats {
            fallbacks: stats[STAT_FALLBACKS as usize],
            lookback_steps: stats[STAT_LOOKBACK_STEPS as usize],
        },
    )
}

/// Sort u32 values using GPU radix sort. Returns the sorted data.
pub fn radix_sort_u32<R: Runtime>(client: &ComputeClient<R::Server>, input: &[u32]) -> Vec<u32> {
    radix_sort_u32_with_stats::<R>(client, input, SortConfig::default()).0
}
