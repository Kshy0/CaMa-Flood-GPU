// HYDROFORGE METAL KERNEL BODY: compute_reservoir_outflow
    long num_reservoirs = *args.num_reservoirs;
    long num_catchments = *args.num_catchments;
    long reservoir_idx = (long)i % num_reservoirs;
    long member_index = (long)i / num_reservoirs;
    long member_offset = member_index * num_catchments;

    int local_catchment = args.reservoir_catchment_idx_ptr[reservoir_idx];
    int local_downstream = args.downstream_idx_ptr[local_catchment];
    bool is_river_mouth = local_downstream == local_catchment;
    long catchment = member_offset + local_catchment;
    long downstream = member_offset + local_downstream;
    float time_step = *args.time_step_ptr;

    float old_river_outflow = args.river_outflow_ptr[catchment];
    float old_flood_outflow = args.flood_outflow_ptr[catchment];
    float old_positive = max(old_river_outflow, 0.0f)
        + max(old_flood_outflow, 0.0f);
    float old_negative = min(old_river_outflow, 0.0f)
        + min(old_flood_outflow, 0.0f);

    // Undo exactly the outflow kernel's outgoing flows of this cell.
    atomic_fetch_add_explicit(
        &args.outgoing_storage_ptr[catchment],
        -old_positive, memory_order_relaxed);
    if (!is_river_mouth) {
        atomic_fetch_add_explicit(
            &args.outgoing_storage_ptr[downstream],
            old_negative, memory_order_relaxed);
    }

    float river_flood_storage = args.river_storage_ptr[catchment]
        + args.flood_storage_ptr[catchment];
    float dam_volume = river_flood_storage;
    if (HAS_LEVEE) {
        dam_volume += args.protected_storage_ptr[catchment];
    }
    long runoff_idx = batched_runoff ? catchment : local_catchment;
    float reservoir_inflow = args.reservoir_total_inflow_ptr[catchment]
        + args.runoff_ptr[runoff_idx];
    args.reservoir_total_inflow_ptr[catchment] = 0.0f;

    long member_reservoir = member_index * num_reservoirs + reservoir_idx;
    float conservation_volume = args.conservation_volume_ptr[
        batched_conservation_volume ? member_reservoir : reservoir_idx];
    float emergency_volume = args.emergency_volume_ptr[
        batched_emergency_volume ? member_reservoir : reservoir_idx];
    float adjustment_volume = args.adjustment_volume_ptr[
        batched_adjustment_volume ? member_reservoir : reservoir_idx];
    float normal_outflow = args.effective_normal_outflow_ptr[
        batched_effective_normal_outflow ? member_reservoir : reservoir_idx];
    float adjustment_outflow = args.adjustment_outflow_ptr[
        batched_adjustment_outflow ? member_reservoir : reservoir_idx];
    float flood_control_outflow = args.flood_control_outflow_ptr[
        batched_flood_control_outflow ? member_reservoir : reservoir_idx];

    float reservoir_outflow;
    if (dam_volume <= conservation_volume) {
        reservoir_outflow = normal_outflow
            * sqrt(dam_volume / conservation_volume);
    } else if (dam_volume <= adjustment_volume) {
        float fraction = (dam_volume - conservation_volume)
            / (adjustment_volume - conservation_volume);
        reservoir_outflow = normal_outflow
            + exp(3.0f * log(fraction))
            * (adjustment_outflow - normal_outflow);
    } else if (dam_volume <= emergency_volume) {
        float fraction = (dam_volume - adjustment_volume)
            / (emergency_volume - adjustment_volume);
        float controlled = adjustment_outflow
            + exp(CMF_RESERVOIR_RELEASE_EXPONENT * log(fraction))
            * (flood_control_outflow - adjustment_outflow);
        if (reservoir_inflow >= flood_control_outflow) {
            float flood = normal_outflow
                + (dam_volume - conservation_volume)
                / (emergency_volume - conservation_volume)
                * (reservoir_inflow - normal_outflow);
            reservoir_outflow = max(flood, controlled);
        } else {
            reservoir_outflow = controlled;
        }
    } else {
        reservoir_outflow = reservoir_inflow >= flood_control_outflow
            ? reservoir_inflow : flood_control_outflow;
    }

    // Flow limiter: the minimum first, so a negative storage releases nothing.
    reservoir_outflow = min(
        min(reservoir_outflow, dam_volume / time_step),
        river_flood_storage / time_step);
    reservoir_outflow = max(reservoir_outflow, 0.0f);
    args.river_outflow_ptr[catchment] = reservoir_outflow;
    args.flood_outflow_ptr[catchment] = 0.0f;

    atomic_fetch_add_explicit(
        &args.outgoing_storage_ptr[catchment],
        reservoir_outflow, memory_order_relaxed);
