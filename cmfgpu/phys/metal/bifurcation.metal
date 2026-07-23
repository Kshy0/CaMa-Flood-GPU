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

// HYDROFORGE METAL KERNEL BODY: compute_bifurcation_outflow
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
    float maximum_surface = max(water_surface, downstream_surface);
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
            maximum_surface,
            args.bifurcation_elevation_ptr[elevation_offset + local_level],
            args.bifurcation_width_ptr[width_offset + local_level],
            args.bifurcation_manning_ptr[manning_offset + local_level],
            slope, gravity, time_step, true);
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

// HYDROFORGE METAL KERNEL BODY: compute_bifurcation_inflow
long num_paths = *args.num_bifurcation_paths;
    long num_trials = *args.num_trials;
    long total = num_paths * num_trials;
    if ((long)i >= total) return;

    long path = (long)i % num_paths;
    long trial = (long)i / num_paths;
    long catchment_offset = trial * *args.num_catchments;
    long level_offset =
        trial * num_paths * (long)num_bifurcation_levels;

    int catchment = args.bifurcation_catchment_idx_ptr[path];
    int downstream = args.bifurcation_downstream_idx_ptr[path];
    float local_limit =
        args.limit_rate_ptr[catchment_offset + (long)catchment];
    float downstream_limit =
        args.limit_rate_ptr[catchment_offset + (long)downstream];
    float sum = 0.0f;

    for (int level = 0; level < num_bifurcation_levels; ++level) {
        long item = path * (long)num_bifurcation_levels + level;
        float outflow = args.bifurcation_outflow_ptr[level_offset + item];
        outflow *= outflow >= 0.0f ? local_limit : downstream_limit;
        sum += outflow;
        args.bifurcation_outflow_ptr[level_offset + item] = outflow;
    }

    atomic_fetch_add_explicit(
        args.global_bifurcation_outflow_ptr
            + catchment_offset + (long)catchment,
        sum, memory_order_relaxed);
    atomic_fetch_add_explicit(
        args.global_bifurcation_outflow_ptr
            + catchment_offset + (long)downstream,
        -sum, memory_order_relaxed);
