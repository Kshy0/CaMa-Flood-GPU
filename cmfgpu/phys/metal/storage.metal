struct FloodStageResult {
    float river_storage;
    float flood_storage;
    float river_depth;
    float flood_depth;
    float flood_fraction;
};

static inline FloodStageResult flood_stage_inline(
    float total_storage,
    float river_height,
    float catchment_area,
    float river_width,
    float river_length,
    device const float* flood_depth_table,
    int num_flood_levels
) {
    FloodStageResult result;
    float maximum_river_storage =
        river_length * river_width * river_height;
    if (total_storage <= maximum_river_storage) {
        result.river_storage = total_storage;
        result.flood_storage = 0.0f;
        result.river_depth = total_storage / (river_length * river_width);
        result.flood_depth = 0.0f;
        result.flood_fraction = 0.0f;
        return result;
    }

    float width_increment =
        (catchment_area / river_length) / (float)num_flood_levels;
    int level = 0;
    float accumulated_storage = maximum_river_storage;
    float previous_height = 0.0f;
    float previous_width = river_width;
    float previous_total_storage = maximum_river_storage;
    float previous_flood_depth = 0.0f;
    float next_flood_depth = 0.0f;

    for (int level_idx = 0; level_idx < num_flood_levels; ++level_idx) {
        float current_height = flood_depth_table[level_idx];
        float current_width = river_width
            + (float)(level_idx + 1) * width_increment;
        float storage_increment = river_length * 0.5f
            * (previous_width + current_width)
            * (current_height - previous_height);
        float current_storage = accumulated_storage + storage_increment;
        if (total_storage > current_storage) {
            ++level;
            previous_total_storage = current_storage;
            previous_flood_depth = current_height;
            accumulated_storage = current_storage;
            previous_height = current_height;
            previous_width = current_width;
        } else {
            next_flood_depth = current_height;
            break;
        }
    }

    float previous_total_width =
        river_width + (float)level * width_increment;
    float width_difference = 0.0f;
    if (level == num_flood_levels) {
        result.flood_depth = previous_flood_depth
            + (total_storage - previous_total_storage)
                / (previous_total_width * river_length);
    } else {
        float flood_gradient =
            (next_flood_depth - previous_flood_depth) / width_increment;
        width_difference = sqrt(
            previous_total_width * previous_total_width
            + 2.0f * (total_storage - previous_total_storage)
                / (flood_gradient * river_length)
        ) - previous_total_width;
        result.flood_depth =
            previous_flood_depth + width_difference * flood_gradient;
    }

    result.river_storage = min(
        maximum_river_storage
            + river_length * river_width * result.flood_depth,
        total_storage);
    result.flood_storage = max(
        total_storage - result.river_storage, 0.0f);
    result.river_depth =
        result.river_storage / (river_length * river_width);
    float middle_fraction = clamp(
        (previous_total_width + width_difference - river_width)
            * river_length / catchment_area,
        0.0f, 1.0f);
    result.flood_fraction = level == num_flood_levels
        ? 1.0f : middle_fraction;
    return result;
}

// Fold one per-catchment value across the threadgroup and issue a single
// atomic; same-address atomics otherwise serialize once per catchment.
// Every thread must call this (threadgroup barriers), so callers keep
// out-of-range lanes alive with 0.  ``scratch`` is reusable on return.
static inline void cmf_block_atomic_add(
    threadgroup float* scratch,
    uint lid,
    uint group_threads,
    float value,
    device atomic_float* destination
) {
    scratch[lid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // dispatchThreads permits a short final threadgroup; fold odd tail lanes.
    for (uint active_threads = group_threads; active_threads > 1;
            active_threads = (active_threads + 1) / 2) {
        uint next_active = (active_threads + 1) / 2;
        uint pair = lid + next_active;
        if (pair < active_threads) {
            scratch[lid] += scratch[pair];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0 && scratch[0] != 0.0f) {
        atomic_fetch_add_explicit(
            destination, scratch[0], memory_order_relaxed);
    }
    // Ensure the reduction is consumed before the caller reuses ``scratch``.
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// HYDROFORGE METAL KERNEL BODY: compute_flood_stage
long num_catchments = *args.num_catchments;
    long num_trials = *args.num_trials;
    long total = num_catchments * num_trials;
    if ((long)i >= total) return;

    long catchment = (long)i % num_catchments;
    long trial = (long)i / num_catchments;
    long trial_offset = trial * num_catchments;
    long cell = trial_offset + catchment;
    float time_step = args.time_step_ptr[0];

    float river_storage = args.river_storage_ptr[cell];
    float flood_storage = args.flood_storage_ptr[cell];
    float protected_storage = args.protected_storage_ptr[cell];
    float river_inflow = args.river_inflow_ptr[cell];
    float flood_inflow = args.flood_inflow_ptr[cell];
    float river_outflow = args.river_outflow_ptr[cell];
    float flood_outflow = args.flood_outflow_ptr[cell];
    float bifurcation_outflow = HAS_BIFURCATION
        ? args.global_bifurcation_outflow_ptr[cell] : 0.0f;
    long runoff_idx = batched_runoff ? cell : catchment;
    float runoff = args.runoff_ptr[runoff_idx];
    float prescribed_inflow = 0.0f;
    if (HAS_INFLOW) {
        int inflow_idx = args.catchment_inflow_idx_ptr[catchment];
        if (inflow_idx >= 0) {
            long inflow_trial_offset = batched_inflow
                ? trial * (long)(*args.num_inflow_gauges) : 0;
            prescribed_inflow = args.inflow_ptr[
                inflow_trial_offset + inflow_idx];
        }
    }

    float updated_river_storage = river_storage
        + (river_inflow - river_outflow) * time_step;
    float updated_flood_storage = flood_storage
        + (updated_river_storage < 0.0f ? updated_river_storage : 0.0f)
        + (flood_inflow - flood_outflow - bifurcation_outflow) * time_step;
    updated_river_storage = max(updated_river_storage, 0.0f);
    if (updated_flood_storage < 0.0f) {
        updated_river_storage = max(
            updated_river_storage + updated_flood_storage, 0.0f);
    }
    updated_flood_storage = max(updated_flood_storage, 0.0f);
    float total_storage = max(
        updated_river_storage + updated_flood_storage + protected_storage
            + (runoff + prescribed_inflow) * time_step,
        0.0f);

    long river_height_idx = batched_river_height ? cell : catchment;
    long catchment_area_idx = batched_catchment_area ? cell : catchment;
    long river_width_idx = batched_river_width ? cell : catchment;
    long river_length_idx = batched_river_length ? cell : catchment;
    float river_height = args.river_height_ptr[river_height_idx];
    float catchment_area = args.catchment_area_ptr[catchment_area_idx];
    float river_width = args.river_width_ptr[river_width_idx];
    float river_length = args.river_length_ptr[river_length_idx];
    long table_trial_offset = batched_flood_depth_table
        ? trial_offset * (long)num_flood_levels : 0;
    long table_cell_offset =
        table_trial_offset + catchment * (long)num_flood_levels;
    FloodStageResult stage = flood_stage_inline(
        total_storage, river_height, catchment_area,
        river_width, river_length,
        args.flood_depth_table_ptr + table_cell_offset,
        num_flood_levels);

    args.outgoing_storage_ptr[cell] = 0.0f;
    args.river_storage_ptr[cell] = stage.river_storage;
    args.flood_storage_ptr[cell] = stage.flood_storage;
    args.protected_storage_ptr[cell] = 0.0f;
    args.river_depth_ptr[cell] = stage.river_depth;
    args.flood_depth_ptr[cell] = stage.flood_depth;
    args.protected_depth_ptr[cell] = stage.flood_depth;
    args.flood_fraction_ptr[cell] = stage.flood_fraction;
// HYDROFORGE METAL KERNEL BODY: compute_flood_stage_log
long num_catchments = *args.num_catchments;
    threadgroup float log_scratch[HF_BLOCK_SIZE];

    // Out-of-range lanes contribute 0 so every thread reaches the barriers.
    bool active_lane = (long)i < num_catchments;
    int current_step = args.current_step_ptr[0];

    float log_storage_pre = 0.0f;
    float log_storage_next = 0.0f;
    float log_storage_new = 0.0f;
    float log_inflow = 0.0f;
    float log_outflow = 0.0f;
    float log_inflow_error = 0.0f;
    float log_storage_stage = 0.0f;
    float log_river_storage = 0.0f;
    float log_flood_storage = 0.0f;
    float log_flood_area = 0.0f;
    float log_stage_error = 0.0f;

    if (active_lane) {

    long cell = (long)i;
    float time_step = args.time_step_ptr[0];
    bool non_levee =
        !HAS_LEVEE || args.is_levee_ptr[cell] == 0;

    float river_storage = args.river_storage_ptr[cell];
    float flood_storage = args.flood_storage_ptr[cell];
    float protected_storage = args.protected_storage_ptr[cell];
    float river_inflow = args.river_inflow_ptr[cell];
    float flood_inflow = args.flood_inflow_ptr[cell];
    float river_outflow = args.river_outflow_ptr[cell];
    float flood_outflow = args.flood_outflow_ptr[cell];
    float bifurcation_outflow = HAS_BIFURCATION
        ? args.global_bifurcation_outflow_ptr[cell] : 0.0f;
    float runoff = args.runoff_ptr[cell];
    float prescribed_inflow = 0.0f;
    if (HAS_INFLOW) {
        int inflow_idx = args.catchment_inflow_idx_ptr[cell];
        if (inflow_idx >= 0) {
            prescribed_inflow = args.inflow_ptr[inflow_idx];
        }
    }

    float storage_before =
        river_storage + flood_storage + protected_storage;
    log_storage_pre = storage_before * 1e-9f;

    float updated_river_storage = river_storage
        + (river_inflow - river_outflow) * time_step;
    float updated_flood_storage = flood_storage
        + (updated_river_storage < 0.0f ? updated_river_storage : 0.0f)
        + (flood_inflow - flood_outflow - bifurcation_outflow) * time_step;
    updated_river_storage = max(updated_river_storage, 0.0f);
    if (updated_flood_storage < 0.0f) {
        updated_river_storage = max(
            updated_river_storage + updated_flood_storage, 0.0f);
    }
    updated_flood_storage = max(updated_flood_storage, 0.0f);
    float storage_after_routing =
        updated_river_storage + updated_flood_storage + protected_storage
        + (runoff + prescribed_inflow) * time_step;
    float total_storage = max(storage_after_routing, 0.0f);

    log_storage_next = storage_after_routing * 1e-9f;
    log_storage_new = total_storage * 1e-9f;
    log_inflow = (river_inflow + flood_inflow + prescribed_inflow)
        * time_step * 1e-9f;
    log_outflow = (river_outflow + flood_outflow) * time_step * 1e-9f;
    float inflow_error = storage_before - storage_after_routing
        + (river_inflow + flood_inflow + runoff + prescribed_inflow
            - river_outflow - flood_outflow - bifurcation_outflow)
            * time_step;
    log_inflow_error = inflow_error * 1e-9f;

    float river_height = args.river_height_ptr[cell];
    float catchment_area = args.catchment_area_ptr[cell];
    float river_width = args.river_width_ptr[cell];
    float river_length = args.river_length_ptr[cell];
    FloodStageResult stage = flood_stage_inline(
        total_storage, river_height, catchment_area,
        river_width, river_length,
        args.flood_depth_table_ptr + cell * (long)num_flood_levels,
        num_flood_levels);

    float stage_storage = stage.river_storage + stage.flood_storage;
    if (non_levee) {
        log_storage_stage = stage_storage * 1e-9f;
        log_river_storage = stage.river_storage * 1e-9f;
        log_flood_storage = stage.flood_storage * 1e-9f;
        log_flood_area = stage.flood_fraction * catchment_area * 1e-9f;
        log_stage_error = (stage_storage - total_storage) * 1e-9f;
    }

    args.outgoing_storage_ptr[cell] = 0.0f;
    args.river_storage_ptr[cell] = stage.river_storage;
    args.flood_storage_ptr[cell] = stage.flood_storage;
    args.protected_storage_ptr[cell] = 0.0f;
    args.river_depth_ptr[cell] = stage.river_depth;
    args.flood_depth_ptr[cell] = stage.flood_depth;
    args.protected_depth_ptr[cell] = stage.flood_depth;
    args.flood_fraction_ptr[cell] = stage.flood_fraction;

    }  // active_lane

    // One atomic per counter per threadgroup, reusing a single scratch buffer.
    long group_start = (long)i - (long)lid;
    uint group_threads = (uint)min((long)tpg, num_catchments - group_start);
    cmf_block_atomic_add(log_scratch, lid, group_threads, log_storage_pre,
        args.total_storage_pre_sum_ptr + current_step);
    cmf_block_atomic_add(log_scratch, lid, group_threads, log_storage_next,
        args.total_storage_next_sum_ptr + current_step);
    cmf_block_atomic_add(log_scratch, lid, group_threads, log_storage_new,
        args.total_storage_new_sum_ptr + current_step);
    cmf_block_atomic_add(log_scratch, lid, group_threads, log_inflow,
        args.total_inflow_sum_ptr + current_step);
    cmf_block_atomic_add(log_scratch, lid, group_threads, log_outflow,
        args.total_outflow_sum_ptr + current_step);
    cmf_block_atomic_add(log_scratch, lid, group_threads, log_inflow_error,
        args.total_inflow_error_sum_ptr + current_step);
    cmf_block_atomic_add(log_scratch, lid, group_threads, log_storage_stage,
        args.total_storage_stage_sum_ptr + current_step);
    cmf_block_atomic_add(log_scratch, lid, group_threads, log_river_storage,
        args.river_storage_sum_ptr + current_step);
    cmf_block_atomic_add(log_scratch, lid, group_threads, log_flood_storage,
        args.flood_storage_sum_ptr + current_step);
    cmf_block_atomic_add(log_scratch, lid, group_threads, log_flood_area,
        args.flood_area_sum_ptr + current_step);
    cmf_block_atomic_add(log_scratch, lid, group_threads, log_stage_error,
        args.total_stage_error_sum_ptr + current_step);
