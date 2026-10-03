#ifdef HF_HP_ENABLED
inline void routing_inflow_inline(long cell, long member, long n, int paths, bool has_bifurcation, bool has_reservoir,
    device const int* routing_edge_start_ptr,
    device const int* routing_edge_source_ptr,
    device const int* bifurcation_catchment_start_ptr,
    device const int* bifurcation_catchment_path_ptr,
    device const int* bifurcation_downstream_start_ptr,
    device const int* bifurcation_downstream_path_ptr,
    device const float* bifurcation_path_flow_ptr,
    device const float* river_outflow_ptr,
    device const float* flood_outflow_ptr,
    device cmf_storage* river_inflow_ptr,
    device cmf_storage* flood_inflow_ptr,
    device cmf_storage* global_bifurcation_outflow_ptr,
    device cmf_storage* reservoir_total_inflow_ptr,
    device const uchar* is_reservoir_ptr) {
    long offset = member * n;
    cmf_storage river = cmf_storage(0.0f), flood = cmf_storage(0.0f);
    int end = cell + 1 < n ? routing_edge_start_ptr[cell + 1] : int(n);
    for (int edge = routing_edge_start_ptr[cell]; edge < end; ++edge) {
        int source = routing_edge_source_ptr[edge];
        if (source == int(cell)) continue;
        river = river + cmf_storage(river_outflow_ptr[offset + source]);
        flood = flood + cmf_storage(flood_outflow_ptr[offset + source]);
    }
    river_inflow_ptr[offset + cell] = river;
    flood_inflow_ptr[offset + cell] = flood;
    if (has_reservoir && is_reservoir_ptr[cell] != 0) {
        // Only routed river/flood inflow enters this accumulator, as in CALC_INFLOW.
        reservoir_total_inflow_ptr[offset + cell] = reservoir_total_inflow_ptr[offset + cell]
            + river + flood;
    }
    if (has_bifurcation) {
        cmf_storage bifurcation = cmf_storage(0.0f);

        long path_offset = member * paths;
        int last = cell + 1 < n ? bifurcation_catchment_start_ptr[cell + 1] : paths;
        for (int edge = bifurcation_catchment_start_ptr[cell]; edge < last; ++edge) {
            int path = bifurcation_catchment_path_ptr[edge];
            bifurcation = bifurcation + cmf_storage(bifurcation_path_flow_ptr[path_offset + path]);
        }
        last = cell + 1 < n ? bifurcation_downstream_start_ptr[cell + 1] : paths;
        for (int edge = bifurcation_downstream_start_ptr[cell]; edge < last; ++edge) {
            int path = bifurcation_downstream_path_ptr[edge];
            bifurcation = bifurcation - cmf_storage(bifurcation_path_flow_ptr[path_offset + path]);
        }
        global_bifurcation_outflow_ptr[offset + cell] = bifurcation;
    }

}
#endif

struct FloodStageResult {
    cmf_storage river_storage;
    cmf_storage flood_storage;
    float river_depth;
    float flood_depth;
    float flood_fraction;
};

// CALC_STONXT; returns PSTOALL, the total storage the stage splits.
// Runoff splits by the previous stage's flood fraction; prescribed inflow
// (LUPSINF) joins the river part. Negative runoff may leave a negative total.
static inline cmf_storage update_storage_inline(
    cmf_storage river_storage, cmf_storage flood_storage, cmf_storage protected_storage,
    float river_inflow, float flood_inflow, float river_outflow,
    float flood_outflow, float bifurcation_outflow, float runoff,
    float prescribed_inflow, float flood_fraction, float time_step
) {
    cmf_storage river = river_storage + cmf_storage(river_inflow * time_step)
        - cmf_storage(river_outflow * time_step);
    cmf_storage flood = flood_storage;
    if (river < 0.0f) {
        flood = flood + river;
        river = 0.0f;
    }
    flood = flood + cmf_storage(flood_inflow * time_step) - cmf_storage(flood_outflow * time_step)
        - cmf_storage(bifurcation_outflow * time_step);
    if (flood < 0.0f) {
        river = max(river + flood, cmf_storage(0.0f));
        flood = 0.0f;
    }
    float river_runoff = runoff * (1.0f - flood_fraction) * time_step
        + prescribed_inflow * time_step;
    river = river + cmf_storage(river_runoff);
    flood = flood + cmf_storage(runoff * flood_fraction * time_step);
    return river + flood + protected_storage;
}

static inline FloodStageResult flood_stage_inline(
    cmf_storage total_storage_hp,
    float river_height,
    float catchment_area,
    float river_width,
    float river_length,
    device const float* flood_depth_table,
    int num_flood_levels,
    bool has_levee
) {
    FloodStageResult result;
    float total_storage = float(total_storage_hp);
    float maximum_river_storage =
        river_length * river_width * river_height;
    if (!(has_levee ? total_storage > maximum_river_storage
            : total_storage_hp > cmf_storage(maximum_river_storage))) {
        result.river_storage = total_storage_hp;
        result.flood_storage = 0.0f;
        result.river_depth =
            max(total_storage / river_length / river_width, 0.0f);
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
        if (has_levee ? total_storage > current_storage
                : total_storage_hp > cmf_storage(current_storage)) {
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

    result.river_storage = cmf_storage(maximum_river_storage
        + river_length * river_width * result.flood_depth);
    if (!has_levee) {
        result.river_storage = min(result.river_storage, total_storage_hp);
    }
    result.flood_storage = max(
        total_storage_hp - result.river_storage, cmf_storage(0.0f));
    result.river_depth =
        float(result.river_storage) / river_length / river_width;
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
    long ensemble_size = *args.ensemble_size;
    long total = num_catchments * ensemble_size;
    if ((long)i >= total) return;

    long catchment = (long)i % num_catchments;
    long member = (long)i / num_catchments;
    long member_offset = member * num_catchments;
    long cell = member_offset + catchment;

#ifdef HF_HP_ENABLED
    routing_inflow_inline(catchment, member, num_catchments,
        int(*args.num_bifurcation_paths), HAS_BIFURCATION, HAS_RESERVOIR,
        args.routing_edge_start_ptr,
        args.routing_edge_source_ptr,
        args.bifurcation_catchment_start_ptr,
        args.bifurcation_catchment_path_ptr,
        args.bifurcation_downstream_start_ptr,
        args.bifurcation_downstream_path_ptr,
        args.bifurcation_path_flow_ptr,
        args.river_outflow_ptr,
        args.flood_outflow_ptr,
        args.river_inflow_ptr,
        args.flood_inflow_ptr,
        args.global_bifurcation_outflow_ptr,
        args.reservoir_total_inflow_ptr,
        args.is_reservoir_ptr);
#endif
    float time_step = args.time_step_ptr[0];

    cmf_storage river_storage = args.river_storage_ptr[cell];
    cmf_storage flood_storage = args.flood_storage_ptr[cell];
    cmf_storage protected_storage = HAS_LEVEE
        ? args.protected_storage_ptr[cell] : cmf_storage(0.0f);
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
            long inflow_member_offset = batched_inflow
                ? member * (long)(*args.num_inflow_gauges) : 0;
            prescribed_inflow = args.inflow_ptr[
                inflow_member_offset + inflow_idx];
        }
    }

    cmf_storage total_storage = update_storage_inline(
        river_storage, flood_storage, protected_storage, river_inflow,
        flood_inflow, river_outflow, flood_outflow, bifurcation_outflow,
        runoff, prescribed_inflow, args.flood_fraction_ptr[cell], time_step);

    long river_height_idx = batched_river_height ? cell : catchment;
    long catchment_area_idx = batched_catchment_area ? cell : catchment;
    long river_width_idx = batched_river_width ? cell : catchment;
    long river_length_idx = batched_river_length ? cell : catchment;
    float river_height = args.river_height_ptr[river_height_idx];
    float catchment_area = args.catchment_area_ptr[catchment_area_idx];
    float river_width = args.river_width_ptr[river_width_idx];
    float river_length = args.river_length_ptr[river_length_idx];
    long table_member_offset = batched_flood_depth_table
        ? member_offset * (long)num_flood_levels : 0;
    long table_cell_offset =
        table_member_offset + catchment * (long)num_flood_levels;
    FloodStageResult stage = flood_stage_inline(
        total_storage, river_height, catchment_area,
        river_width, river_length,
        args.flood_depth_table_ptr + table_cell_offset,
        num_flood_levels, HAS_LEVEE);

    args.outgoing_storage_ptr[cell] = 0.0f;
    args.river_storage_ptr[cell] = stage.river_storage;
    args.flood_storage_ptr[cell] = stage.flood_storage;
    if (HAS_TOTAL_STORAGE_OUTPUT) {
        args.total_storage_ptr[cell] = total_storage;
    }
    if (HAS_LEVEE) {
        args.protected_storage_ptr[cell] = 0.0f;
    }
    args.river_depth_ptr[cell] = stage.river_depth;
    args.flood_depth_ptr[cell] = stage.flood_depth;
    if (HAS_LEVEE) {
        args.protected_depth_ptr[cell] = 0.0f;
    }
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

#ifdef HF_HP_ENABLED
    routing_inflow_inline(cell, 0, num_catchments,
        int(*args.num_bifurcation_paths), HAS_BIFURCATION, HAS_RESERVOIR,
        args.routing_edge_start_ptr,
        args.routing_edge_source_ptr,
        args.bifurcation_catchment_start_ptr,
        args.bifurcation_catchment_path_ptr,
        args.bifurcation_downstream_start_ptr,
        args.bifurcation_downstream_path_ptr,
        args.bifurcation_path_flow_ptr,
        args.river_outflow_ptr,
        args.flood_outflow_ptr,
        args.river_inflow_ptr,
        args.flood_inflow_ptr,
        args.global_bifurcation_outflow_ptr,
        args.reservoir_total_inflow_ptr,
        args.is_reservoir_ptr);
#endif
    float time_step = args.time_step_ptr[0];
    bool non_levee =
        !HAS_LEVEE || args.is_levee_ptr[cell] == 0;

    cmf_storage river_storage = args.river_storage_ptr[cell];
    cmf_storage flood_storage = args.flood_storage_ptr[cell];
    cmf_storage protected_storage = HAS_LEVEE
        ? args.protected_storage_ptr[cell] : cmf_storage(0.0f);
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

    cmf_storage total_storage = update_storage_inline(
        river_storage, flood_storage, protected_storage, river_inflow,
        flood_inflow, river_outflow, flood_outflow, bifurcation_outflow,
        runoff, prescribed_inflow, args.flood_fraction_ptr[cell], time_step);
    float storage_after_routing = total_storage;

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
        num_flood_levels, HAS_LEVEE);

    cmf_storage stage_storage = stage.river_storage + stage.flood_storage;
    if (non_levee) {
        log_storage_stage = float(stage_storage) * 1e-9f;
        log_river_storage = float(stage.river_storage) * 1e-9f;
        log_flood_storage = float(stage.flood_storage) * 1e-9f;
        log_flood_area = stage.flood_fraction * catchment_area * 1e-9f;
        log_stage_error = float(stage_storage - total_storage) * 1e-9f;
    }

    args.outgoing_storage_ptr[cell] = 0.0f;
    args.river_storage_ptr[cell] = stage.river_storage;
    args.flood_storage_ptr[cell] = stage.flood_storage;
    if (HAS_TOTAL_STORAGE_OUTPUT) {
        args.total_storage_ptr[cell] = total_storage;
    }
    if (HAS_LEVEE) {
        args.protected_storage_ptr[cell] = 0.0f;
    }
    args.river_depth_ptr[cell] = stage.river_depth;
    args.flood_depth_ptr[cell] = stage.flood_depth;
    if (HAS_LEVEE) {
        args.protected_depth_ptr[cell] = 0.0f;
    }
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
