// HYDROFORGE METAL KERNEL BODY: compute_outflow
long num_catchments = *args.num_catchments;
    long num_trials = *args.num_trials;
    long total = num_catchments * num_trials;
    if ((long)i >= total) return;

    long catchment = (long)i % num_catchments;
    long trial = (long)i / num_catchments;
    long trial_offset = trial * num_catchments;
    long cell = trial_offset + catchment;
    int downstream = args.downstream_idx_ptr[catchment];
    long downstream_cell = trial_offset + (long)downstream;
    bool is_mouth = downstream == (int)catchment;
    float time_step = args.time_step_ptr[0];
    float gravity = *args.gravity;

    float river_outflow = args.river_outflow_ptr[cell];
    float river_depth = args.river_depth_ptr[cell];
    float river_storage = args.river_storage_ptr[cell];
    float flood_outflow = args.flood_outflow_ptr[cell];
    float flood_depth = args.flood_depth_ptr[cell];
    float protected_depth = args.protected_depth_ptr[cell];
    float flood_storage = args.flood_storage_ptr[cell];
    float protected_storage = args.protected_storage_ptr[cell];
    float river_xs_depth = args.river_cross_section_depth_ptr[cell];
    float flood_xs_depth = args.flood_cross_section_depth_ptr[cell];
    float flood_xs_area = args.flood_cross_section_area_ptr[cell];

    long river_manning_idx = batched_river_manning ? cell : catchment;
    long flood_manning_idx = batched_flood_manning ? cell : catchment;
    long river_width_idx = batched_river_width ? cell : catchment;
    long river_length_idx = batched_river_length ? cell : catchment;
    long river_height_idx = batched_river_height ? cell : catchment;
    long elevation_idx = batched_catchment_elevation ? cell : catchment;
    float river_manning = args.river_manning_ptr[river_manning_idx];
    float flood_manning = args.flood_manning_ptr[flood_manning_idx];
    float river_width = args.river_width_ptr[river_width_idx];
    float river_length = args.river_length_ptr[river_length_idx];
    float river_height = args.river_height_ptr[river_height_idx];
    float catchment_elevation = args.catchment_elevation_ptr[elevation_idx];
    long downstream_distance_idx = batched_downstream_distance
        ? cell : catchment;
    float downstream_distance =
        args.downstream_distance_ptr[downstream_distance_idx];

    float river_elevation = catchment_elevation - river_height;
    float water_surface = river_depth + river_elevation;
    float protected_surface = water_surface;
    if (HAS_LEVEE && args.is_levee_ptr[catchment]) {
        protected_surface =
            min(catchment_elevation + protected_depth, water_surface);
    }
    float total_storage = river_storage + flood_storage + protected_storage;

    long downstream_height_idx = batched_river_height
        ? downstream_cell : (long)downstream;
    long downstream_elevation_idx = batched_catchment_elevation
        ? downstream_cell : (long)downstream;
    float downstream_river_elevation =
        args.catchment_elevation_ptr[downstream_elevation_idx]
        - args.river_height_ptr[downstream_height_idx];
    float downstream_surface =
        args.river_depth_ptr[downstream_cell] + downstream_river_elevation;
    float effective_downstream_surface = is_mouth
        ? catchment_elevation : downstream_surface;
    if (HAS_SEA_LEVEL) {
        int sea_level = args.catchment_sea_level_idx_ptr[catchment];
        if (sea_level >= 0) {
            effective_downstream_surface = args.sea_surface_elevation_ptr[
                trial * *args.num_sea_level_boundaries + sea_level];
        }
    }
    float maximum_surface = max(water_surface, effective_downstream_surface);
    float river_slope =
        (water_surface - effective_downstream_surface) / downstream_distance;
    float flood_slope = clamp(river_slope, -0.005f, 0.005f);

    // The mouth boundary level controls slope, while CaMa-Flood uses the
    // local river depth itself for the mouth's hydraulic cross-section.
    float updated_river_xs = is_mouth
        ? river_depth : maximum_surface - river_elevation;
    float river_flow_depth =
        max(sqrt(updated_river_xs * river_xs_depth), 1e-6f);
    float flood_cross_section_surface = is_mouth
        ? water_surface : maximum_surface;
    float updated_flood_xs =
        max(flood_cross_section_surface - catchment_elevation, 0.0f);
    float flood_flow_depth =
        max(sqrt(updated_flood_xs * flood_xs_depth), 1e-6f);
    float updated_flood_area = max(
        flood_storage / river_length - flood_depth * river_width, 0.0f);

    float river_xs_area = updated_river_xs * river_width;
    bool river_active = river_flow_depth > 1e-5f && river_xs_area > 1e-5f;
    float unit_river_outflow = river_outflow / river_width;
    float river_numerator = river_width * (
        unit_river_outflow
        + gravity * time_step * river_flow_depth * river_slope);
    float river_denominator = 1.0f
        + gravity * time_step * river_manning * river_manning
        * abs(unit_river_outflow) * pow(river_flow_depth, -7.0f / 3.0f);
    float updated_river_outflow = river_active
        ? river_numerator / river_denominator : 0.0f;

    bool flood_active =
        flood_flow_depth > 1e-5f && updated_flood_area > 1e-5f;
    float updated_flood_outflow = 0.0f;
    if (flood_active) {
        float implicit_area = max(
            sqrt(updated_flood_area * max(flood_xs_area, 1e-6f)), 1e-6f);
        float flood_numerator = flood_outflow
            + gravity * time_step * implicit_area * flood_slope;
        float flood_denominator = 1.0f
            + gravity * time_step * flood_manning * flood_manning
            * abs(flood_outflow) * pow(flood_flow_depth, -4.0f / 3.0f)
            / implicit_area;
        updated_flood_outflow = flood_numerator / flood_denominator;
    }

    if (updated_river_outflow * updated_flood_outflow < 0.0f) {
        updated_flood_outflow = 0.0f;
    }
    if (updated_river_outflow < 0.0f && !is_mouth) {
        float negative_volume =
            (-updated_river_outflow - updated_flood_outflow) * time_step;
        float limit = min(0.05f * total_storage / negative_volume, 1.0f);
        updated_river_outflow *= limit;
        updated_flood_outflow *= limit;
    }

    if (HAS_RESERVOIR && args.is_dam_upstream_ptr[catchment] != 0) {
        float downstream_elevation =
            args.catchment_elevation_ptr[downstream_elevation_idx];
        float bed_slope = max(
            (catchment_elevation - downstream_elevation)
                / downstream_distance,
            MIN_KINEMATIC_SLOPE);
        float river_velocity = sqrt(bed_slope)
            * pow(river_depth * river_depth, 1.0f / 3.0f) / river_manning;
        updated_river_outflow = clamp(
            river_width * river_depth * river_velocity,
            0.0f, river_storage / time_step);

        float flood_velocity = sqrt(min(bed_slope, 0.005f))
            * pow(flood_depth * flood_depth, 1.0f / 3.0f) / flood_manning;
        float flood_area = max(
            flood_storage / river_length - flood_depth * river_width, 0.0f);
        updated_flood_outflow = clamp(
            flood_area * flood_velocity, 0.0f, flood_storage / time_step);
    }

    args.river_outflow_ptr[cell] = updated_river_outflow;
    args.flood_outflow_ptr[cell] = updated_flood_outflow;
    if (HAS_WATER_SURFACE) {
        args.water_surface_elevation_ptr[cell] = water_surface;
    }
    if (HAS_PROTECTED_WATER_SURFACE) {
        args.protected_water_surface_elevation_ptr[cell] = protected_surface;
    }
    args.river_cross_section_depth_ptr[cell] = updated_river_xs;
    args.flood_cross_section_depth_ptr[cell] = updated_flood_xs;
    args.flood_cross_section_area_ptr[cell] = updated_flood_area;
    if (HAS_TOTAL_STORAGE) {
        args.total_storage_ptr[cell] = total_storage;
    }
    args.river_inflow_ptr[cell] = 0.0f;
    args.flood_inflow_ptr[cell] = 0.0f;
    if (HAS_BIFURCATION) {
        args.global_bifurcation_outflow_ptr[cell] = 0.0f;
    }

    float positive_flow =
        max(updated_river_outflow, 0.0f) + max(updated_flood_outflow, 0.0f);
    float negative_flow =
        min(updated_river_outflow, 0.0f) + min(updated_flood_outflow, 0.0f);
    atomic_fetch_add_explicit(
        args.outgoing_storage_ptr + cell,
        positive_flow * time_step, memory_order_relaxed);
    if (!is_mouth) {
        atomic_fetch_add_explicit(
            args.outgoing_storage_ptr + downstream_cell,
            -negative_flow * time_step, memory_order_relaxed);
    }

// HYDROFORGE METAL KERNEL BODY: compute_inflow
long num_catchments = *args.num_catchments;
    long num_trials = *args.num_trials;
    long total = num_catchments * num_trials;
    if ((long)i >= total) return;

    long catchment = (long)i % num_catchments;
    long trial_offset = ((long)i / num_catchments) * num_catchments;
    long cell = trial_offset + catchment;

    float river_outflow = args.river_outflow_ptr[cell];
    float flood_outflow = args.flood_outflow_ptr[cell];
    float outgoing_storage = args.outgoing_storage_ptr[cell];
    float available_storage =
        args.river_storage_ptr[cell] + args.flood_storage_ptr[cell];
    float local_limit = outgoing_storage > 1e-8f
        ? min(available_storage / outgoing_storage, 1.0f)
        : 1.0f;

    int downstream = args.downstream_idx_ptr[catchment];
    long downstream_cell = trial_offset + (long)downstream;
    float downstream_outgoing = args.outgoing_storage_ptr[downstream_cell];
    float downstream_available =
        args.river_storage_ptr[downstream_cell]
        + args.flood_storage_ptr[downstream_cell];
    float downstream_limit = downstream_outgoing > 1e-8f
        ? min(downstream_available / downstream_outgoing, 1.0f)
        : 1.0f;

    float updated_river = river_outflow >= 0.0f
        ? river_outflow * local_limit
        : river_outflow * downstream_limit;
    float updated_flood = flood_outflow >= 0.0f
        ? flood_outflow * local_limit
        : flood_outflow * downstream_limit;

    args.river_outflow_ptr[cell] = updated_river;
    args.flood_outflow_ptr[cell] = updated_flood;
    if (HAS_BIFURCATION) {
        args.limit_rate_ptr[cell] = local_limit;
    }

    if (downstream != (int)catchment) {
        atomic_fetch_add_explicit(
            args.river_inflow_ptr + downstream_cell,
            updated_river, memory_order_relaxed);
        atomic_fetch_add_explicit(
            args.flood_inflow_ptr + downstream_cell,
            updated_flood, memory_order_relaxed);
        if (HAS_RESERVOIR && args.is_reservoir_ptr[downstream] != 0) {
            atomic_fetch_add_explicit(
                args.reservoir_total_inflow_ptr + downstream_cell,
                updated_river + updated_flood, memory_order_relaxed);
        }
    }
