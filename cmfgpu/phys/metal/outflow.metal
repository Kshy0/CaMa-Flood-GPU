#ifdef HF_HP_ENABLED
inline cmf_storage outgoing_demand_inline(long cell, long member, long n, int paths, bool has_bifurcation,
    device const int* routing_edge_start_ptr,
    device const int* routing_edge_source_ptr,
    device const int* bifurcation_catchment_start_ptr,
    device const int* bifurcation_catchment_path_ptr,
    device const int* bifurcation_downstream_start_ptr,
    device const int* bifurcation_downstream_path_ptr,
    device const float* bifurcation_path_flow_ptr,
    device const float* unlimited_outflow_ptr) {
    long offset = member * n;
    device const float2* uflow = reinterpret_cast<device const float2*>(unlimited_outflow_ptr);
    float2 own = uflow[offset + cell];
    cmf_storage outgoing = cmf_storage(max(own.x, 0.0f) + max(own.y, 0.0f));
    int end = cell + 1 < n ? routing_edge_start_ptr[cell + 1] : int(n);
    for (int edge = routing_edge_start_ptr[cell]; edge < end; ++edge) {
        int source = routing_edge_source_ptr[edge];
        if (source == int(cell)) continue;
        float2 src = uflow[offset + source];
        if (src.x < 0.0f) outgoing = outgoing + (-src.x);
        if (src.y < 0.0f) outgoing = outgoing + (-src.y);
    }
    if (has_bifurcation) {
        long path_offset = member * paths;
        int last = cell + 1 < n ? bifurcation_catchment_start_ptr[cell + 1] : paths;
        for (int edge = bifurcation_catchment_start_ptr[cell]; edge < last; ++edge) {
            int path = bifurcation_catchment_path_ptr[edge];
            float bf = bifurcation_path_flow_ptr[path_offset + path];
            if (bf > 0.0f) outgoing = outgoing + bf;
        }
        last = cell + 1 < n ? bifurcation_downstream_start_ptr[cell + 1] : paths;
        for (int edge = bifurcation_downstream_start_ptr[cell]; edge < last; ++edge) {
            int path = bifurcation_downstream_path_ptr[edge];
            float bf = bifurcation_path_flow_ptr[path_offset + path];
            if (bf < 0.0f) outgoing = outgoing + (-bf);
        }
    }
    return outgoing;
}
#endif

// HYDROFORGE METAL KERNEL BODY: compute_outflow
long num_catchments = *args.num_catchments;
    long ensemble_size = *args.ensemble_size;
    long total = num_catchments * ensemble_size;
    if ((long)i >= total) return;

    long catchment, member;
    cmf_decompose_index(i, num_catchments, ensemble_size, catchment, member);
    long member_offset = member * num_catchments;
    long cell = member_offset + catchment;
    int downstream = args.downstream_idx_ptr[catchment];
    long downstream_cell = member_offset + (long)downstream;
    bool is_mouth = downstream == (int)catchment;
    float time_step = args.time_step_ptr[0];
    float gravity = *args.gravity;

    float river_outflow = args.river_outflow_ptr[cell];
    float river_depth = args.river_depth_ptr[cell];
    cmf_storage river_storage_hp = args.river_storage_ptr[cell];
    float flood_outflow = args.flood_outflow_ptr[cell];
    float flood_depth = args.flood_depth_ptr[cell];
    cmf_storage flood_storage_hp = args.flood_storage_ptr[cell];
    float flood_storage = float(flood_storage_hp);
    cmf_storage total_storage_hp = (flood_storage == 0.0f)
        ? river_storage_hp : (river_storage_hp + flood_storage_hp);
    if (HAS_LEVEE) {
        cmf_storage protected_storage_hp = args.protected_storage_ptr[cell];
        if (float(protected_storage_hp) != 0.0f) {
            total_storage_hp = total_storage_hp + protected_storage_hp;
        }
    }
    float total_storage = float(total_storage_hp);
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
            long sea_member_offset = batched_sea_surface_elevation
                ? member * *args.num_sea_level_boundaries : 0;
            effective_downstream_surface = args.sea_surface_elevation_ptr[
                sea_member_offset + sea_level];
        }
    }
    float maximum_surface = max(water_surface, effective_downstream_surface);
    float river_slope =
        cmf_div(water_surface - effective_downstream_surface, downstream_distance);
    float flood_slope = clamp(river_slope, -CMF_ROUTING_SLOPE_LIMIT, CMF_ROUTING_SLOPE_LIMIT);

    // The mouth boundary controls slope; its cross-section uses local depth.
    float updated_river_xs = is_mouth
        ? river_depth : maximum_surface - river_elevation;
    float river_flow_depth =
        max(cmf_sqrt(updated_river_xs * river_xs_depth), 1e-6f);
    float flood_cross_section_surface = is_mouth
        ? water_surface : maximum_surface;
    float updated_flood_xs =
        max(flood_cross_section_surface - catchment_elevation, 0.0f);
    float flood_flow_depth = updated_flood_xs > 0.0f
        ? max(cmf_sqrt(updated_flood_xs * flood_xs_depth), 1e-6f) : 1e-6f;
    float flood_area_base = flood_storage != 0.0f
        ? cmf_div(flood_storage, river_length) : 0.0f;
    // HF-UPSTREAM-FIX[CMF-01]: Do not double-subtract above-channel area from floodplain cross-section
    //   upstream: CaMa-Flood v4.31 cmf_calc_outflw_mod.F90:95,152 (and damout:488) subtracts D2FLDDPH*D2RIVWTH from P2FLDSTO/L even though P2FLDSTO already excludes above-channel storage.
    //   fix: Use max(flood_storage / river_length, 0) directly as floodplain cross-sectional area.
    float updated_flood_area = max(flood_area_base, 0.0f);

    float river_xs_area = updated_river_xs * river_width;
    bool river_active = river_flow_depth > 1e-5f && river_xs_area > 1e-5f;
    float updated_river_outflow = 0.0f;
    if (river_active) {
        float unit_river_outflow = cmf_div(river_outflow, river_width);
        float river_numerator = river_width * (
            unit_river_outflow
            + gravity * time_step * river_flow_depth * river_slope);
        float river_denominator = 1.0f
            + gravity * time_step * river_manning * river_manning
            * fabs(unit_river_outflow) * cmf_inv_d73(river_flow_depth);
        updated_river_outflow = cmf_div(river_numerator, river_denominator);
    }

    bool flood_active =
        flood_flow_depth > 1e-5f && updated_flood_area > 1e-5f;
    float updated_flood_outflow = 0.0f;
    if (flood_active) {
        float implicit_area = max(
            cmf_sqrt(updated_flood_area * max(flood_xs_area, 1e-6f)), 1e-6f);
        float flood_numerator = flood_outflow
            + gravity * time_step * implicit_area * flood_slope;
        float flood_denominator = 1.0f
            + cmf_div(
                gravity * time_step * flood_manning * flood_manning
                * fabs(flood_outflow) * cmf_inv_d43(flood_flow_depth),
                implicit_area);
        updated_flood_outflow = cmf_div(flood_numerator, flood_denominator);
    }

    // Floodplain flow only moves with the river flow (rivout*fldout > 0).
    if (!(updated_river_outflow * updated_flood_outflow > 0.0f)) {
        updated_flood_outflow = 0.0f;
    }
    // v4.23 storage-change limiter on every non-mouth cell: reverse flow from
    // downstream into this cell is capped at 5% of this cell's storage per step.
    if (!is_mouth) {
        float backflow = max(
            (-updated_river_outflow - updated_flood_outflow) * time_step,
            CMF_OUTGOING_VOLUME_FLOOR);
        float limit = min(
            cmf_div(CMF_BACKFLOW_STORAGE_FRACTION * total_storage, backflow), 1.0f);
        updated_river_outflow *= limit;
        updated_flood_outflow *= limit;
    }

    if (HAS_RESERVOIR && args.is_dam_upstream_ptr[batched_is_dam_upstream ? (long)i : catchment] != 0) {
        float river_storage = float(river_storage_hp);
        float downstream_elevation =
            args.catchment_elevation_ptr[downstream_elevation_idx];
        float bed_slope = max(
            cmf_div(catchment_elevation - downstream_elevation, downstream_distance),
            min_kinematic_slope);
        // CaMa bounds the kinematic flows by the storage only (MIN), so a
        // negative storage (negative runoff) gives a negative flow.
        float river_velocity = cmf_div(1.0f, river_manning) * cmf_sqrt(bed_slope)
            * cmf_cbrt(river_depth * river_depth);
        updated_river_outflow = min(
            river_width * river_depth * river_velocity, cmf_div(river_storage, time_step));

        float flood_velocity = cmf_div(1.0f, flood_manning)
            * cmf_sqrt(min(bed_slope, CMF_ROUTING_SLOPE_LIMIT))
            * cmf_cbrt(flood_depth * flood_depth);
        updated_flood_outflow = min(
            updated_flood_area * flood_velocity, cmf_div(flood_storage, time_step));
    }

    args.river_outflow_ptr[cell] = updated_river_outflow;
    args.flood_outflow_ptr[cell] = updated_flood_outflow;
#ifdef HF_HP_ENABLED
    args.unlimited_outflow_ptr[2 * cell] = updated_river_outflow;
    args.unlimited_outflow_ptr[2 * cell + 1] = updated_flood_outflow;
#endif
    args.river_cross_section_depth_ptr[cell] = updated_river_xs;
    args.flood_cross_section_depth_ptr[cell] = updated_flood_xs;
    args.flood_cross_section_area_ptr[cell] = updated_flood_area;
    // Under HF_HP_ENABLED, compute_flood_stage overwrites river_inflow /
    // flood_inflow directly from CSR routing and never reads the workspace
    // global_bifurcation_outflow or outgoing_storage buffers.
#ifndef HF_HP_ENABLED
    args.river_inflow_ptr[cell] = 0.0f;
    args.flood_inflow_ptr[cell] = 0.0f;
    if (HAS_BIFURCATION) {
        args.global_bifurcation_outflow_ptr[cell] = 0.0f;
    }

    // P2STOOUT flows: the cell's own positive outflows, and the reversed flow
    // out of its downstream cell; compute_inflow multiplies by the step.
    float own_out = max(updated_river_outflow, 0.0f) + max(updated_flood_outflow, 0.0f);
    if (own_out != 0.0f) {
        atomic_fetch_add_explicit(
            args.outgoing_storage_ptr + cell, own_out, memory_order_relaxed);
    }
    if (!is_mouth) {
        float rev_out = max(-updated_river_outflow, 0.0f) + max(-updated_flood_outflow, 0.0f);
        if (rev_out != 0.0f) {
            atomic_fetch_add_explicit(
                args.outgoing_storage_ptr + downstream_cell,
                rev_out, memory_order_relaxed);
        }
    }
#endif

// HYDROFORGE METAL KERNEL BODY: compute_inflow
long num_catchments = *args.num_catchments;
    long ensemble_size = *args.ensemble_size;
    long total = num_catchments * ensemble_size;
    if ((long)i >= total) return;

    long catchment, member;
    cmf_decompose_index(i, num_catchments, ensemble_size, catchment, member);
    long member_offset = member * num_catchments;
    long cell = member_offset + catchment;

    float river_outflow = args.river_outflow_ptr[cell];
    float flood_outflow = args.flood_outflow_ptr[cell];
    // CaMa-Flood v4.23 supply-side limiter (CALC_INFLOW_LSPAMAT): a cell
    // releases at most its storage.  The rate of the cell a flow leaves sets
    // both of its flows, chosen by the river flow's direction.
    float time_step = args.time_step_ptr[0];
    int downstream = args.downstream_idx_ptr[catchment];
    long downstream_cell = member_offset + (long)downstream;
#ifdef HF_HP_ENABLED
    // Demand uses the preceding dispatch's immutable flow snapshot.
    cmf_storage local_outgoing = outgoing_demand_inline(
        catchment, member, num_catchments,
        int(*args.num_bifurcation_paths), HAS_BIFURCATION,
        args.routing_edge_start_ptr,
        args.routing_edge_source_ptr,
        args.bifurcation_catchment_start_ptr,
        args.bifurcation_catchment_path_ptr,
        args.bifurcation_downstream_start_ptr,
        args.bifurcation_downstream_path_ptr,
        args.bifurcation_path_flow_ptr,
        args.unlimited_outflow_ptr);
#else
    cmf_storage local_outgoing = args.outgoing_storage_ptr[cell];
#endif
    cmf_storage river_sto = args.river_storage_ptr[cell];
    cmf_storage flood_sto = args.flood_storage_ptr[cell];
    cmf_storage available_storage = (float(flood_sto) == 0.0f)
        ? river_sto : (river_sto + flood_sto);
    float local_out_vol = (float(local_outgoing) == 0.0f)
        ? 0.0f : float(local_outgoing * time_step);
    float local_limit = min(cmf_div(float(available_storage), max(
        local_out_vol, CMF_OUTGOING_VOLUME_FLOOR)), 1.0f);

    float rate = local_limit;
    if (river_outflow <= 0.0f && (river_outflow != 0.0f || flood_outflow != 0.0f) && downstream != (int)catchment) {
#ifdef HF_HP_ENABLED
        cmf_storage downstream_outgoing = outgoing_demand_inline(
            downstream, member, num_catchments,
            int(*args.num_bifurcation_paths), HAS_BIFURCATION,
            args.routing_edge_start_ptr,
            args.routing_edge_source_ptr,
            args.bifurcation_catchment_start_ptr,
            args.bifurcation_catchment_path_ptr,
            args.bifurcation_downstream_start_ptr,
            args.bifurcation_downstream_path_ptr,
            args.bifurcation_path_flow_ptr,
            args.unlimited_outflow_ptr);
#else
        cmf_storage downstream_outgoing = args.outgoing_storage_ptr[downstream_cell];
#endif
        cmf_storage dn_river_sto = args.river_storage_ptr[downstream_cell];
        cmf_storage dn_flood_sto = args.flood_storage_ptr[downstream_cell];
        cmf_storage downstream_available = (float(dn_flood_sto) == 0.0f)
            ? dn_river_sto : (dn_river_sto + dn_flood_sto);
        float dn_out_vol = (float(downstream_outgoing) == 0.0f)
            ? 0.0f : float(downstream_outgoing * time_step);
        rate = min(cmf_div(float(downstream_available), max(
            dn_out_vol, CMF_OUTGOING_VOLUME_FLOOR)), 1.0f);
    }

    float updated_river = river_outflow * rate;
    float updated_flood = flood_outflow * rate;

    args.river_outflow_ptr[cell] = updated_river;
    args.flood_outflow_ptr[cell] = updated_flood;
    if (HAS_BIFURCATION) {
        args.limit_rate_ptr[cell] = local_limit;
    }

#ifndef HF_HP_ENABLED
    if (downstream != (int)catchment) {
        if (updated_river != 0.0f) {
            atomic_fetch_add_explicit(
                args.river_inflow_ptr + downstream_cell,
                updated_river, memory_order_relaxed);
        }
        if (updated_flood != 0.0f) {
            atomic_fetch_add_explicit(
                args.flood_inflow_ptr + downstream_cell,
                updated_flood, memory_order_relaxed);
        }
        if (HAS_RESERVOIR && args.is_reservoir_ptr[downstream] != 0) {
            float res_in = updated_river + updated_flood;
            if (res_in != 0.0f) {
                atomic_fetch_add_explicit(
                    args.reservoir_total_inflow_ptr + downstream_cell,
                    res_in, memory_order_relaxed);
            }
        }
    }

#endif
