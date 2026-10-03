struct LeveeStageResult {
    cmf_storage river_storage;
    cmf_storage flood_storage;
    cmf_storage protected_storage;
    float river_depth;
    float flood_depth;
    float protected_depth;
    float flood_fraction;
};

static inline LeveeStageResult levee_stage_inline(
    cmf_storage river_storage,
    cmf_storage flood_storage,
    float river_depth,
    float flood_depth,
    float flood_fraction,
    float river_height,
    float catchment_area,
    float river_width,
    float river_length,
    float levee_base_height,
    float levee_crown_height,
    float levee_fraction,
    device const float* flood_depth_table,
    int num_flood_levels,
    float maximum_river_storage,
    float levee_base_storage,
    float top_storage,
    float levee_fill_storage,
    float top_ilev
) {
    LeveeStageResult result;
    result.river_storage = river_storage;
    result.flood_storage = flood_storage;
    result.protected_storage = 0.0f;
    result.river_depth = river_depth;
    result.flood_depth = flood_depth;
    result.protected_depth = 0.0f;
    result.flood_fraction = flood_fraction;

    cmf_storage total_storage_hp = river_storage + flood_storage;
    float total_storage = float(total_storage_hp);
    // Case 0, water only in the river channel: the default stage stands.
    if (!(total_storage > maximum_river_storage)) return result;

    levee_crown_height = max(levee_crown_height, levee_base_height);
    float width_increment =
        (catchment_area / river_length) / (float)num_flood_levels;
    float levee_distance =
        levee_fraction * (catchment_area / river_length);
    float current_storage = maximum_river_storage;
    float previous_height = 0.0f;
    float previous_width = river_width;
    bool case3 = total_storage >= levee_base_storage && total_storage >= top_storage && total_storage < levee_fill_storage;
    bool case4 = total_storage >= levee_base_storage && total_storage >= top_storage && !(total_storage < levee_fill_storage);

    int levee_level = (int)(levee_fraction * (float)num_flood_levels);
    float case3_storage = 0.0f;
    float case3_width = 0.0f;
    float case3_depth = 0.0f;
    float case3_gradient = 0.0f;
    bool found_case3 = false;

    float case4_storage = 0.0f;
    float case4_width = 0.0f;
    float case4_gradient = 0.0f;
    bool found_case4 = false;

    // Dynamic partition search is needed only for cases 3 and 4.
    for (int level = 0; (case3 || case4) && level < num_flood_levels; ++level) {
        float depth = flood_depth_table[level];
        float height_increment = depth - previous_height;
        float middle_width = previous_width + 0.5f * width_increment;
        float storage_increment =
            river_length * middle_width * height_increment;
        float next_storage = current_storage + storage_increment;
        float gradient = height_increment / width_increment;

        if (case3 && level >= levee_level && !found_case3) {
            float wedge_storage = (levee_distance + river_width)
                * (levee_crown_height - depth) * river_length;
            float threshold = next_storage + wedge_storage;
            if (total_storage < threshold) {
                if (level == levee_level) case3_storage = top_ilev;
                case3_gradient = gradient;
                found_case3 = true;
            } else {
                case3_storage = threshold;
                case3_width =
                    width_increment * (float)(level + 1) - levee_distance;
                case3_depth = depth - levee_base_height;
            }
        }
        // Case 4 stops at the first layer the storage does not exceed.
        if (case4 && !found_case4 && !(total_storage > next_storage)) {
            case4_storage = current_storage;
            case4_width = previous_width;
            case4_gradient = gradient;
            found_case4 = true;
        }

        current_storage = next_storage;
        previous_height = depth;
        previous_width += width_increment;
        if ((!case3 || found_case3) && (!case4 || found_case4)) break;
    }

    if (total_storage < levee_base_storage) {
        // Case 1, below the levee base: the default stage stands.
    } else if (total_storage < top_storage) {
        // Case 2, river side below the crown, protected side dry.
        float added_storage = total_storage - levee_base_storage;
        result.flood_depth = levee_base_height
            + added_storage
                / (levee_distance + river_width) / river_length;
        result.river_storage = maximum_river_storage
            + river_length * river_width * result.flood_depth;
        result.river_depth = float(result.river_storage)
            / river_length / river_width;
        result.flood_storage = max(
            total_storage_hp - result.river_storage, cmf_storage(0.0f));
        result.flood_fraction = levee_fraction;
    } else if (total_storage < levee_fill_storage) {
        // Case 3, river side at the crown, protected side filling.
        float added_storage = total_storage - case3_storage;
        if (found_case3) {
            float added_width = -case3_width + sqrt(
                case3_width * case3_width
                + 2.0f * added_storage / river_length / case3_gradient);
            float added_depth = added_width * case3_gradient;
            result.protected_depth =
                levee_base_height + case3_depth + added_depth;
            result.flood_fraction = clamp(
                (case3_width + levee_distance)
                    / (width_increment * (float)num_flood_levels),
                0.0f, 1.0f);
        } else {
            float added_depth =
                added_storage / case3_width / river_length;
            result.protected_depth =
                levee_base_height + case3_depth + added_depth;
            result.flood_fraction = 1.0f;
        }
        result.flood_depth = levee_crown_height;
        result.river_storage = maximum_river_storage
            + river_length * river_width * result.flood_depth;
        result.river_depth = float(result.river_storage)
            / river_length / river_width;
        result.flood_storage = max(
            cmf_storage(top_storage) - result.river_storage, cmf_storage(0.0f));
        result.protected_storage = max(
            total_storage_hp - result.river_storage - result.flood_storage,
            cmf_storage(0.0f));
    } else {
        // Case 4, above the crown: the default river stage stands, with the
        // unclamped default-stage flood fraction.
        float added_width = 0.0f;
        if (found_case4) {
            added_width = -case4_width + sqrt(
                case4_width * case4_width
                + 2.0f * (total_storage - case4_storage) / river_length
                    / case4_gradient);
        } else {
            case4_width = previous_width;
        }
        result.flood_fraction = (-river_width + case4_width + added_width)
            / (width_increment * (float)num_flood_levels);
        float added_storage = (flood_depth - levee_crown_height)
            * (levee_distance + river_width) * river_length;
        result.flood_storage = max(
            cmf_storage(top_storage + added_storage) - river_storage, cmf_storage(0.0f));
        result.protected_storage = max(
            total_storage_hp - river_storage - result.flood_storage, cmf_storage(0.0f));
        result.protected_depth = flood_depth;
    }
    return result;
}

struct BifurcationLevelResult {
    float outflow;
    float cross_section_depth;
};

static inline BifurcationLevelResult bifurcation_level_inline(
    float previous_outflow,
    float previous_cross_section_depth,
    float maximum_water_surface,
    float elevation,
    float width,
    float manning,
    float slope,
    float gravity,
    float time_step,
    bool semi_implicit_depth
) {
    BifurcationLevelResult result;
    result.cross_section_depth = max(
        maximum_water_surface - elevation, 0.0f);
    float implicit_flow_depth = sqrt(
        result.cross_section_depth * previous_cross_section_depth);
    float flow_depth = semi_implicit_depth
        ? (implicit_flow_depth <= 0.0f
            ? result.cross_section_depth : implicit_flow_depth)
        : result.cross_section_depth;
    result.outflow = 0.0f;
    if (flow_depth > 1e-5f) {
        float unit_outflow = previous_outflow / width;
        float numerator = width * (
            unit_outflow + gravity * time_step * flow_depth * slope);
        float denominator = 1.0f
            + gravity * time_step * manning * manning
                * fabs(unit_outflow) * pow(flow_depth, -7.0f / 3.0f);
        result.outflow = numerator / denominator;
    }
    return result;
}

// Fold one per-levee value across the threadgroup and issue a single atomic;
// see the identical helper in storage.metal.  Every thread must call this
// (threadgroup barriers); ``scratch`` is reusable on return.
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

static inline float levee_bifurcation_outflow_inline(
    uint i,
    device const int* bifurcation_catchment_idx_ptr,
    device const int* bifurcation_downstream_idx_ptr,
    device const float* bifurcation_manning_ptr,
    device float* bifurcation_outflow_ptr,
    device const float* bifurcation_width_ptr,
    device const float* bifurcation_length_ptr,
    device const float* bifurcation_elevation_ptr,
    device float* bifurcation_cross_section_depth_ptr,
    device const float* river_depth_ptr,
    device const float* protected_depth_ptr,
    device const float* river_height_ptr,
    device const float* catchment_elevation_ptr,
    device const uchar* is_levee_ptr,
    device const cmf_storage* river_storage_ptr,
    device const cmf_storage* flood_storage_ptr,
    device const cmf_storage* protected_storage_ptr,
    float gravity,
    device const float* time_step_ptr,
    long num_bifurcation_paths,
    int num_bifurcation_levels,
    long ensemble_size,
    long num_catchments,
    bool batched_bifurcation_manning,
    bool batched_bifurcation_width,
    bool batched_bifurcation_length,
    bool batched_bifurcation_elevation,
    bool batched_river_height,
    bool batched_catchment_elevation
) {
    long num_paths = num_bifurcation_paths;
    long total = num_paths * ensemble_size;
    if ((long)i >= total) return 0.0f;

    long path = (long)i % num_paths;
    long member = (long)i / num_paths;
    long path_offset = member * num_paths;
    long catchment_offset = member * num_catchments;
    long level_offset = path_offset * (long)num_bifurcation_levels;
    long path_level = path * (long)num_bifurcation_levels;

    int catchment = bifurcation_catchment_idx_ptr[path];
    int downstream = bifurcation_downstream_idx_ptr[path];
    long catchment_cell = catchment_offset + catchment;
    long downstream_cell = catchment_offset + downstream;
    long length_idx = batched_bifurcation_length
        ? path_offset + path : path;
    float length = bifurcation_length_ptr[length_idx];
    long catchment_height_idx = batched_river_height
        ? catchment_cell : (long)catchment;
    long downstream_height_idx = batched_river_height
        ? downstream_cell : (long)downstream;
    long catchment_elevation_idx = batched_catchment_elevation
        ? catchment_cell : (long)catchment;
    long downstream_elevation_idx = batched_catchment_elevation
        ? downstream_cell : (long)downstream;
    float catchment_elevation =
        catchment_elevation_ptr[catchment_elevation_idx];
    float downstream_elevation =
        catchment_elevation_ptr[downstream_elevation_idx];
    // D2SFCELV = D2RIVELV + D2RIVDPH with D2RIVELV = D2ELEVTN - D2RIVHGT.
    float water_surface = river_depth_ptr[catchment_cell]
        + (catchment_elevation - river_height_ptr[catchment_height_idx]);
    float downstream_surface = river_depth_ptr[downstream_cell]
        + (downstream_elevation
            - river_height_ptr[downstream_height_idx]);
    float protected_surface = is_levee_ptr[catchment]
        ? min(
            catchment_elevation + protected_depth_ptr[catchment_cell],
            water_surface)
        : water_surface;
    float downstream_protected_surface = is_levee_ptr[downstream]
        ? min(
            downstream_elevation + protected_depth_ptr[downstream_cell],
            downstream_surface)
        : downstream_surface;
    float maximum_river_surface = max(water_surface, downstream_surface);
    float maximum_protected_surface = max(
        protected_surface, downstream_protected_surface);
    float slope = clamp(
        (water_surface - downstream_surface) / length, -CMF_ROUTING_SLOPE_LIMIT, CMF_ROUTING_SLOPE_LIMIT);
    float time_step = time_step_ptr[0];

    long manning_offset = batched_bifurcation_manning ? level_offset : 0;
    long width_offset = batched_bifurcation_width ? level_offset : 0;
    long elevation_offset = batched_bifurcation_elevation ? level_offset : 0;
    float total_outflow = 0.0f;
    for (int level = 0; level < num_bifurcation_levels; ++level) {
        long local_level = path_level + level;
        long state_level = level_offset + local_level;
        BifurcationLevelResult result = bifurcation_level_inline(
            bifurcation_outflow_ptr[state_level],
            bifurcation_cross_section_depth_ptr[state_level],
            level == 0 ? maximum_river_surface : maximum_protected_surface,
            bifurcation_elevation_ptr[elevation_offset + local_level],
            bifurcation_width_ptr[width_offset + local_level],
            bifurcation_manning_ptr[manning_offset + local_level],
            slope, gravity, time_step, level == 0);
        bifurcation_cross_section_depth_ptr[state_level] =
            result.cross_section_depth;
        bifurcation_outflow_ptr[state_level] = result.outflow;
        total_outflow += result.outflow;
    }

    float available_storage = min(
        river_storage_ptr[catchment_cell]
            + flood_storage_ptr[catchment_cell]
            + protected_storage_ptr[catchment_cell],
        river_storage_ptr[downstream_cell]
            + flood_storage_ptr[downstream_cell]
            + protected_storage_ptr[downstream_cell]);
    // CaMa-Flood LEVEE_OPT_PTHOUT limits a path only when its flow sum is non-zero.
    float limit = 1.0f;
    if (total_outflow != 0.0f) {
        limit = min(
            CMF_BACKFLOW_STORAGE_FRACTION * available_storage / (fabs(total_outflow) * time_step),
            1.0f);
    }
    total_outflow *= limit;
    for (int level = 0; level < num_bifurcation_levels; ++level) {
        long state_level = level_offset + path_level + level;
        bifurcation_outflow_ptr[state_level] *= limit;
    }

    return total_outflow;
}

// HYDROFORGE METAL KERNEL BODY: compute_levee_stage
long num_levees = *args.num_levees;
    long ensemble_size = *args.ensemble_size;
    long total = num_levees * ensemble_size;
    if ((long)i >= total) return;

    long levee = (long)i % num_levees;
    long member = (long)i / num_levees;
    long levee_offset = member * num_levees;
    long catchment_offset = member * *args.num_catchments;
    int local_catchment = args.levee_catchment_idx_ptr[levee];
    long catchment = catchment_offset + local_catchment;

    long river_length_idx = batched_river_length
        ? catchment : local_catchment;
    long river_width_idx = batched_river_width
        ? catchment : local_catchment;
    long river_height_idx = batched_river_height
        ? catchment : local_catchment;
    long catchment_area_idx = batched_catchment_area
        ? catchment : local_catchment;
    long levee_crown_idx = batched_levee_crown_height
        ? levee_offset + levee : levee;
    long levee_fraction_idx = batched_levee_fraction
        ? levee_offset + levee : levee;
    long levee_base_idx = batched_levee_base_height
        ? levee_offset + levee : levee;
    long table_member_offset = batched_flood_depth_table
        ? catchment_offset * (long)num_flood_levels : 0;
    long table_offset = table_member_offset
        + (long)local_catchment * (long)num_flood_levels;

    LeveeStageResult result = levee_stage_inline(
        args.river_storage_ptr[catchment],
        args.flood_storage_ptr[catchment],
        args.river_depth_ptr[catchment],
        args.flood_depth_ptr[catchment],
        args.flood_fraction_ptr[catchment],
        args.river_height_ptr[river_height_idx],
        args.catchment_area_ptr[catchment_area_idx],
        args.river_width_ptr[river_width_idx],
        args.river_length_ptr[river_length_idx],
        args.levee_base_height_ptr[levee_base_idx],
        args.levee_crown_height_ptr[levee_crown_idx],
        args.levee_fraction_ptr[levee_fraction_idx],
        args.flood_depth_table_ptr + table_offset,
        num_flood_levels, args.levee_river_max_storage_ptr[levee_offset + levee],
        args.levee_base_storage_ptr[levee_offset + levee],
        args.levee_top_storage_ptr[levee_offset + levee],
        args.levee_fill_storage_ptr[levee_offset + levee],
        args.levee_layer_top_storage_ptr[levee_offset + levee]);

    args.river_storage_ptr[catchment] = result.river_storage;
    args.flood_storage_ptr[catchment] = result.flood_storage;
    args.protected_storage_ptr[catchment] = result.protected_storage;
    args.river_depth_ptr[catchment] = result.river_depth;
    args.flood_depth_ptr[catchment] = result.flood_depth;
    args.protected_depth_ptr[catchment] = result.protected_depth;
    args.flood_fraction_ptr[catchment] = result.flood_fraction;

// HYDROFORGE METAL KERNEL BODY: compute_levee_stage_log
long num_levees = *args.num_levees;
    threadgroup float log_scratch[HF_BLOCK_SIZE];

    // Out-of-range lanes contribute 0 so every thread reaches the barriers.
    bool active_lane = (long)i < num_levees;
    int current_step = args.current_step_ptr[0];

    float log_storage_stage = 0.0f;
    float log_river_storage = 0.0f;
    float log_flood_storage = 0.0f;
    float log_flood_area = 0.0f;
    float log_stage_error = 0.0f;

    if (active_lane) {

    long levee = (long)i;
    int catchment = args.levee_catchment_idx_ptr[levee];
    cmf_storage total_storage = args.river_storage_ptr[catchment]
        + args.flood_storage_ptr[catchment];
    float catchment_area = args.catchment_area_ptr[catchment];
    LeveeStageResult result = levee_stage_inline(
        args.river_storage_ptr[catchment],
        args.flood_storage_ptr[catchment],
        args.river_depth_ptr[catchment],
        args.flood_depth_ptr[catchment],
        args.flood_fraction_ptr[catchment],
        args.river_height_ptr[catchment],
        catchment_area,
        args.river_width_ptr[catchment],
        args.river_length_ptr[catchment],
        args.levee_base_height_ptr[levee],
        args.levee_crown_height_ptr[levee],
        args.levee_fraction_ptr[levee],
        args.flood_depth_table_ptr
            + (long)catchment * (long)num_flood_levels,
        num_flood_levels, args.levee_river_max_storage_ptr[levee],
        args.levee_base_storage_ptr[levee],
        args.levee_top_storage_ptr[levee],
        args.levee_fill_storage_ptr[levee],
        args.levee_layer_top_storage_ptr[levee]);

    cmf_storage stage_storage = result.river_storage
        + result.flood_storage + result.protected_storage;
    log_storage_stage = float(stage_storage) * 1e-9f;
    log_river_storage = float(result.river_storage) * 1e-9f;
    log_flood_storage = float(result.flood_storage) * 1e-9f;
    log_flood_area = result.flood_fraction * catchment_area * 1e-9f;
    log_stage_error = float(stage_storage - total_storage) * 1e-9f;

    args.river_storage_ptr[catchment] = result.river_storage;
    args.flood_storage_ptr[catchment] = result.flood_storage;
    args.protected_storage_ptr[catchment] = result.protected_storage;
    args.river_depth_ptr[catchment] = result.river_depth;
    args.flood_depth_ptr[catchment] = result.flood_depth;
    args.protected_depth_ptr[catchment] = result.protected_depth;
    args.flood_fraction_ptr[catchment] = result.flood_fraction;

    }  // active_lane

    // One atomic per counter per threadgroup, reusing a single scratch buffer.
    long group_start = (long)i - (long)lid;
    uint group_threads = (uint)min((long)tpg, num_levees - group_start);
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

// HYDROFORGE METAL KERNEL BODY: compute_levee_bifurcation_outflow
    if ((long)i >= *args.num_bifurcation_paths * *args.ensemble_size) return;
    float total_outflow = levee_bifurcation_outflow_inline(
        i,
        args.bifurcation_catchment_idx_ptr,
        args.bifurcation_downstream_idx_ptr,
        args.bifurcation_manning_ptr,
        args.bifurcation_outflow_ptr,
        args.bifurcation_width_ptr,
        args.bifurcation_length_ptr,
        args.bifurcation_elevation_ptr,
        args.bifurcation_cross_section_depth_ptr,
        args.river_depth_ptr,
        args.protected_depth_ptr,
        args.river_height_ptr,
        args.catchment_elevation_ptr,
        args.is_levee_ptr,
        args.river_storage_ptr,
        args.flood_storage_ptr,
        args.protected_storage_ptr,
        *args.gravity,
        args.time_step_ptr,
        *args.num_bifurcation_paths,
        num_bifurcation_levels,
        *args.ensemble_size,
        *args.num_catchments,
        batched_bifurcation_manning,
        batched_bifurcation_width,
        batched_bifurcation_length,
        batched_bifurcation_elevation,
        batched_river_height,
        batched_catchment_elevation);
#ifdef HF_HP_ENABLED
    args.bifurcation_path_flow_ptr[i] = total_outflow;
#else
    long path = (long)i % *args.num_bifurcation_paths;
    long catchment_offset = ((long)i / *args.num_bifurcation_paths) * *args.num_catchments;
    long catchment_cell = catchment_offset + args.bifurcation_catchment_idx_ptr[path];
    long downstream_cell = catchment_offset + args.bifurcation_downstream_idx_ptr[path];
    // P2STOOUT flows, multiplied by the step in compute_inflow.
    atomic_fetch_add_explicit(
        args.outgoing_storage_ptr + catchment_cell,
        max(total_outflow, 0.0f), memory_order_relaxed);
    atomic_fetch_add_explicit(
        args.outgoing_storage_ptr + downstream_cell,
        -min(total_outflow, 0.0f), memory_order_relaxed);
#endif
