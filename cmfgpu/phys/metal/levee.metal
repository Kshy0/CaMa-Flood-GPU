struct LeveeStageResult {
    float river_storage;
    float flood_storage;
    float protected_storage;
    float river_depth;
    float flood_depth;
    float protected_depth;
    float flood_fraction;
};

static inline LeveeStageResult levee_stage_inline(
    float river_storage,
    float flood_storage,
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
    int num_flood_levels
) {
    LeveeStageResult result;
    result.river_storage = river_storage;
    result.flood_storage = flood_storage;
    result.protected_storage = 0.0f;
    result.river_depth = river_depth;
    result.flood_depth = flood_depth;
    result.protected_depth = 0.0f;
    result.flood_fraction = flood_fraction;

    float total_storage = river_storage + flood_storage;
    float maximum_river_storage =
        river_length * river_width * river_height;
    if (total_storage <= maximum_river_storage) return result;

    float width_increment =
        (catchment_area / river_length) / (float)num_flood_levels;
    float levee_distance =
        levee_fraction * (catchment_area / river_length);
    float current_storage = maximum_river_storage;
    float previous_height = 0.0f;
    float previous_width = river_width;
    float levee_base_storage = maximum_river_storage;
    float levee_fill_storage = maximum_river_storage;
    bool found_base = false;
    bool found_fill = false;

    int levee_level = (int)(levee_fraction * (float)num_flood_levels);
    float case3_storage = 0.0f;
    float case3_width = 0.0f;
    float case3_depth = 0.0f;
    float case3_gradient = 0.0f;
    bool found_case3 = false;

    for (int level = 0; level < num_flood_levels; ++level) {
        float depth = flood_depth_table[level];
        float height_increment = max(depth - previous_height, 1e-6f);
        float middle_width = previous_width + 0.5f * width_increment;
        float storage_increment =
            river_length * middle_width * height_increment;
        float next_storage = current_storage + storage_increment;
        float gradient = height_increment / width_increment;

        if (
            !found_base
            && levee_base_height > previous_height
            && levee_base_height <= depth
        ) {
            float ratio =
                (levee_base_height - previous_height) / height_increment;
            levee_base_storage = current_storage
                + river_length
                    * (previous_width + 0.5f * ratio * width_increment)
                    * (ratio * height_increment);
            found_base = true;
        }
        if (
            !found_fill
            && levee_crown_height > previous_height
            && levee_crown_height <= depth
        ) {
            float ratio =
                (levee_crown_height - previous_height) / height_increment;
            levee_fill_storage = current_storage
                + river_length
                    * (previous_width + 0.5f * ratio * width_increment)
                    * (ratio * height_increment);
            found_fill = true;
        }
        if (level >= levee_level && !found_case3) {
            float levee_height = levee_crown_height - levee_base_height;
            float top_storage = levee_base_storage
                + (levee_distance + river_width)
                    * levee_height * river_length;
            float wedge_storage = (levee_distance + river_width)
                * (levee_crown_height - depth) * river_length;
            float threshold = next_storage + wedge_storage;
            if (total_storage < threshold) {
                if (level == levee_level) case3_storage = top_storage;
                case3_gradient = gradient;
                found_case3 = true;
            } else {
                case3_storage = threshold;
                case3_width =
                    width_increment * (float)(level + 1) - levee_distance;
                case3_depth = depth - levee_base_height;
            }
        }

        current_storage = next_storage;
        previous_height = depth;
        previous_width += width_increment;
        if (found_base && found_fill && found_case3) break;
    }

    if (!found_base) {
        levee_base_storage = levee_base_height > previous_height
            ? current_storage + river_length * previous_width
                * (levee_base_height - previous_height)
            : maximum_river_storage;
    }
    if (!found_fill) {
        levee_fill_storage = levee_crown_height > previous_height
            ? current_storage + river_length * previous_width
                * (levee_crown_height - previous_height)
            : maximum_river_storage;
    }

    float levee_height = levee_crown_height - levee_base_height;
    float top_storage = levee_base_storage
        + (levee_distance + river_width) * levee_height * river_length;
    bool above_crown = total_storage >= levee_fill_storage;
    bool filling_protected =
        !above_crown && total_storage >= top_storage;
    bool below_crown =
        !above_crown && !filling_protected
        && total_storage >= levee_base_storage;

    if (below_crown) {
        float added_storage = total_storage - levee_base_storage;
        result.flood_depth = levee_base_height
            + added_storage
                / (levee_distance + river_width) / river_length;
        result.river_storage = maximum_river_storage
            + river_length * river_width * result.flood_depth;
        result.river_depth = result.river_storage
            / river_length / river_width;
        result.flood_storage = max(
            total_storage - result.river_storage, 0.0f);
        result.flood_fraction = levee_fraction;
    } else if (filling_protected) {
        float added_storage = total_storage - case3_storage;
        float width_term = case3_width * case3_width
            + 2.0f * added_storage / river_length
                / (case3_gradient + 1e-9f);
        float added_width =
            -case3_width + sqrt(max(width_term, 0.0f));
        float added_depth = added_width * case3_gradient;
        if (found_case3) {
            result.protected_depth =
                levee_base_height + case3_depth + added_depth;
            result.flood_fraction = clamp(
                (case3_width + levee_distance)
                    / (width_increment * (float)num_flood_levels),
                0.0f, 1.0f);
        } else {
            float extra_depth = added_storage
                / (case3_width * river_length + 1e-9f);
            result.protected_depth =
                levee_base_height + case3_depth + extra_depth;
            result.flood_fraction = 1.0f;
        }
        result.flood_depth = levee_crown_height;
        result.river_storage = maximum_river_storage
            + river_length * river_width * result.flood_depth;
        result.river_depth = result.river_storage
            / river_length / river_width;
        result.flood_storage = max(
            top_storage - result.river_storage, 0.0f);
        result.protected_storage = max(
            total_storage - result.river_storage - result.flood_storage,
            0.0f);
    } else if (above_crown) {
        float added_storage = (flood_depth - levee_crown_height)
            * (levee_distance + river_width) * river_length;
        result.flood_storage = max(
            top_storage + added_storage - river_storage, 0.0f);
        result.protected_storage = max(
            total_storage - river_storage - result.flood_storage, 0.0f);
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
    float flow_depth = semi_implicit_depth
        ? max(
            sqrt(result.cross_section_depth * previous_cross_section_depth),
            sqrt(result.cross_section_depth * 0.01f))
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

// HYDROFORGE METAL KERNEL BODY: compute_levee_stage
long num_levees = *args.num_levees;
    long num_trials = *args.num_trials;
    long total = num_levees * num_trials;
    if ((long)i >= total) return;

    long levee = (long)i % num_levees;
    long trial = (long)i / num_levees;
    long levee_offset = trial * num_levees;
    long catchment_offset = trial * *args.num_catchments;
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
    long table_trial_offset = batched_flood_depth_table
        ? catchment_offset * (long)num_flood_levels : 0;
    long table_offset = table_trial_offset
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
        num_flood_levels);

    args.river_storage_ptr[catchment] = result.river_storage;
    args.flood_storage_ptr[catchment] = result.flood_storage;
    args.protected_storage_ptr[catchment] = result.protected_storage;
    args.river_depth_ptr[catchment] = result.river_depth;
    args.flood_depth_ptr[catchment] = result.flood_depth;
    args.protected_depth_ptr[catchment] = result.protected_depth;
    args.flood_fraction_ptr[catchment] = result.flood_fraction;

// HYDROFORGE METAL KERNEL BODY: compute_levee_stage_log
long num_levees = *args.num_levees;
    if ((long)i >= num_levees) return;

    long levee = (long)i;
    int catchment = args.levee_catchment_idx_ptr[levee];
    float total_storage = args.river_storage_ptr[catchment]
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
        num_flood_levels);

    int current_step = args.current_step_ptr[0];
    float stage_storage = result.river_storage
        + result.flood_storage + result.protected_storage;
    atomic_fetch_add_explicit(
        args.total_storage_stage_sum_ptr + current_step,
        stage_storage * 1e-9f, memory_order_relaxed);
    atomic_fetch_add_explicit(
        args.river_storage_sum_ptr + current_step,
        result.river_storage * 1e-9f, memory_order_relaxed);
    atomic_fetch_add_explicit(
        args.flood_storage_sum_ptr + current_step,
        result.flood_storage * 1e-9f, memory_order_relaxed);
    atomic_fetch_add_explicit(
        args.flood_area_sum_ptr + current_step,
        result.flood_fraction * catchment_area * 1e-9f,
        memory_order_relaxed);
    atomic_fetch_add_explicit(
        args.total_stage_error_sum_ptr + current_step,
        (stage_storage - total_storage) * 1e-9f,
        memory_order_relaxed);

    args.river_storage_ptr[catchment] = result.river_storage;
    args.flood_storage_ptr[catchment] = result.flood_storage;
    args.protected_storage_ptr[catchment] = result.protected_storage;
    args.river_depth_ptr[catchment] = result.river_depth;
    args.flood_depth_ptr[catchment] = result.flood_depth;
    args.protected_depth_ptr[catchment] = result.protected_depth;
    args.flood_fraction_ptr[catchment] = result.flood_fraction;

// HYDROFORGE METAL KERNEL BODY: compute_levee_bifurcation_outflow
long num_paths = *args.num_bifurcation_paths;
    long num_trials = *args.num_trials;
    long total = num_paths * num_trials;
    if ((long)i >= total) return;

    long path = (long)i % num_paths;
    long trial = (long)i / num_paths;
    long path_offset = trial * num_paths;
    long catchment_offset = trial * *args.num_catchments;
    long level_offset = path_offset * (long)num_bifurcation_levels;
    long path_level = path * (long)num_bifurcation_levels;

    int catchment = args.bifurcation_catchment_idx_ptr[path];
    int downstream = args.bifurcation_downstream_idx_ptr[path];
    long catchment_cell = catchment_offset + catchment;
    long downstream_cell = catchment_offset + downstream;
    long length_idx = batched_bifurcation_length
        ? path_offset + path : path;
    float length = args.bifurcation_length_ptr[length_idx];
    float water_surface = args.water_surface_elevation_ptr[catchment_cell];
    float downstream_surface =
        args.water_surface_elevation_ptr[downstream_cell];
    float protected_surface =
        args.protected_water_surface_elevation_ptr[catchment_cell];
    float downstream_protected_surface =
        args.protected_water_surface_elevation_ptr[downstream_cell];
    float maximum_river_surface = max(water_surface, downstream_surface);
    float maximum_protected_surface = max(
        protected_surface, downstream_protected_surface);
    float slope = clamp(
        (water_surface - downstream_surface) / length, -0.005f, 0.005f);
    float gravity = *args.gravity;
    float time_step = args.time_step_ptr[0];

    long manning_offset = batched_bifurcation_manning ? level_offset : 0;
    long width_offset = batched_bifurcation_width ? level_offset : 0;
    long elevation_offset = batched_bifurcation_elevation ? level_offset : 0;
    float total_outflow = 0.0f;
    for (int level = 0; level < num_bifurcation_levels; ++level) {
        long local_level = path_level + level;
        long state_level = level_offset + local_level;
        BifurcationLevelResult result = bifurcation_level_inline(
            args.bifurcation_outflow_ptr[state_level],
            args.bifurcation_cross_section_depth_ptr[state_level],
            level == 0 ? maximum_river_surface : maximum_protected_surface,
            args.bifurcation_elevation_ptr[elevation_offset + local_level],
            args.bifurcation_width_ptr[width_offset + local_level],
            args.bifurcation_manning_ptr[manning_offset + local_level],
            slope, gravity, time_step, level == 0);
        args.bifurcation_cross_section_depth_ptr[state_level] =
            result.cross_section_depth;
        args.bifurcation_outflow_ptr[state_level] = result.outflow;
        total_outflow += result.outflow;
    }

    float available_storage = min(
        args.total_storage_ptr[catchment_cell],
        args.total_storage_ptr[downstream_cell]);
    float limit = min(
        0.05f * available_storage / (fabs(total_outflow) * time_step),
        1.0f);
    total_outflow *= limit;
    for (int level = 0; level < num_bifurcation_levels; ++level) {
        long state_level = level_offset + path_level + level;
        args.bifurcation_outflow_ptr[state_level] *= limit;
    }

    atomic_fetch_add_explicit(
        args.outgoing_storage_ptr + catchment_cell,
        max(total_outflow, 0.0f) * time_step, memory_order_relaxed);
    atomic_fetch_add_explicit(
        args.outgoing_storage_ptr + downstream_cell,
        -min(total_outflow, 0.0f) * time_step, memory_order_relaxed);
