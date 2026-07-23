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
            prescribed_inflow = args.inflow_ptr[
                trial * (long)(*args.num_inflow_gauges) + inflow_idx];
        }
    }

    // A completely dry, balanced cell is already in the exact state written
    // by the dry branch below.  Avoid geometry/table reads and eight redundant
    // global stores, which dominate the common no-water path.
    const bool dry_balanced = river_storage == 0.0f
        && flood_storage == 0.0f && protected_storage == 0.0f
        && river_inflow - river_outflow == 0.0f
        && flood_inflow - flood_outflow - bifurcation_outflow == 0.0f
        && runoff + prescribed_inflow == 0.0f;
    if (dry_balanced && args.outgoing_storage_ptr[cell] == 0.0f
        && args.river_depth_ptr[cell] == 0.0f
        && args.flood_depth_ptr[cell] == 0.0f
        && args.protected_depth_ptr[cell] == 0.0f
        && args.flood_fraction_ptr[cell] == 0.0f) {
        return;
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
    if ((long)i >= num_catchments) return;

    long cell = (long)i;
    int current_step = args.current_step_ptr[0];
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
    atomic_fetch_add_explicit(
        args.total_storage_pre_sum_ptr + current_step,
        storage_before * 1e-9f, memory_order_relaxed);

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

    if (non_levee) {
        atomic_fetch_add_explicit(
            args.total_storage_next_sum_ptr + current_step,
            storage_after_routing * 1e-9f, memory_order_relaxed);
        atomic_fetch_add_explicit(
            args.total_storage_new_sum_ptr + current_step,
            total_storage * 1e-9f, memory_order_relaxed);
        atomic_fetch_add_explicit(
            args.total_inflow_sum_ptr + current_step,
            (river_inflow + flood_inflow + prescribed_inflow)
                * time_step * 1e-9f,
            memory_order_relaxed);
        atomic_fetch_add_explicit(
            args.total_outflow_sum_ptr + current_step,
            (river_outflow + flood_outflow) * time_step * 1e-9f,
            memory_order_relaxed);
        float inflow_error = storage_before - storage_after_routing
            + (river_inflow + flood_inflow + runoff + prescribed_inflow
                - river_outflow - flood_outflow - bifurcation_outflow)
                * time_step;
        atomic_fetch_add_explicit(
            args.total_inflow_error_sum_ptr + current_step,
            inflow_error * 1e-9f, memory_order_relaxed);
    }

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
    atomic_fetch_add_explicit(
        args.total_storage_stage_sum_ptr + current_step,
        stage_storage * 1e-9f, memory_order_relaxed);
    if (non_levee) {
        atomic_fetch_add_explicit(
            args.river_storage_sum_ptr + current_step,
            stage.river_storage * 1e-9f, memory_order_relaxed);
        atomic_fetch_add_explicit(
            args.flood_storage_sum_ptr + current_step,
            stage.flood_storage * 1e-9f, memory_order_relaxed);
        atomic_fetch_add_explicit(
            args.flood_area_sum_ptr + current_step,
            stage.flood_fraction * catchment_area * 1e-9f,
            memory_order_relaxed);
        atomic_fetch_add_explicit(
            args.total_stage_error_sum_ptr + current_step,
            (stage_storage - total_storage) * 1e-9f,
            memory_order_relaxed);
    }

    args.outgoing_storage_ptr[cell] = 0.0f;
    args.river_storage_ptr[cell] = stage.river_storage;
    args.flood_storage_ptr[cell] = stage.flood_storage;
    args.protected_storage_ptr[cell] = 0.0f;
    args.river_depth_ptr[cell] = stage.river_depth;
    args.flood_depth_ptr[cell] = stage.flood_depth;
    args.protected_depth_ptr[cell] = stage.flood_depth;
    args.flood_fraction_ptr[cell] = stage.flood_fraction;
