// HYDROFORGE METAL KERNEL BODY: compute_reservoir_outflow
    long num_reservoirs = *args.num_reservoirs;
    long num_catchments = *args.num_catchments;
    long reservoir_idx = (long)i % num_reservoirs;
    long trial_idx = (long)i / num_reservoirs;
    long trial_offset = trial_idx * num_catchments;

    int local_catchment = args.reservoir_catchment_idx_ptr[reservoir_idx];
    int local_downstream = args.downstream_idx_ptr[local_catchment];
    bool is_river_mouth = local_downstream == local_catchment;
    long catchment = trial_offset + local_catchment;
    long downstream = trial_offset + local_downstream;
    float time_step = *args.time_step_ptr;

    float old_river_outflow = args.river_outflow_ptr[catchment];
    float old_flood_outflow = args.flood_outflow_ptr[catchment];
    float old_positive = max(old_river_outflow, 0.0f)
        + max(old_flood_outflow, 0.0f);
    float old_negative = min(old_river_outflow, 0.0f)
        + min(old_flood_outflow, 0.0f);

    atomic_fetch_add_explicit(
        &args.outgoing_storage_ptr[catchment],
        -(old_positive * time_step), memory_order_relaxed);
    if (!is_river_mouth) {
        atomic_fetch_add_explicit(
            &args.outgoing_storage_ptr[downstream],
            old_negative * time_step, memory_order_relaxed);
    }

    float river_storage = args.river_storage_ptr[catchment];
    float flood_storage = args.flood_storage_ptr[catchment];
    float total_storage = river_storage + flood_storage;
    float total_inflow = args.reservoir_total_inflow_ptr[catchment];
    args.reservoir_total_inflow_ptr[catchment] = 0.0f;
    float reservoir_inflow = total_inflow + args.runoff_ptr[catchment];

    float conservation_volume =
        args.conservation_volume_ptr[reservoir_idx];
    float emergency_volume = args.emergency_volume_ptr[reservoir_idx];
    float adjustment_volume = args.adjustment_volume_ptr[reservoir_idx];
    float normal_outflow =
        args.effective_normal_outflow_ptr[reservoir_idx];
    float adjustment_outflow = args.adjustment_outflow_ptr[reservoir_idx];
    float flood_control_outflow =
        args.flood_control_outflow_ptr[reservoir_idx];

    float reservoir_outflow;
    if (total_storage <= conservation_volume) {
        reservoir_outflow = normal_outflow
            * sqrt(total_storage / conservation_volume);
    } else if (total_storage <= adjustment_volume) {
        float fraction = (total_storage - conservation_volume)
            / (adjustment_volume - conservation_volume);
        reservoir_outflow = normal_outflow
            + exp(3.0f * log(fraction))
            * (adjustment_outflow - normal_outflow);
    } else if (total_storage <= emergency_volume) {
        float fraction = (total_storage - adjustment_volume)
            / (emergency_volume - adjustment_volume);
        float controlled = adjustment_outflow
            + exp(0.1f * log(fraction))
            * (flood_control_outflow - adjustment_outflow);
        if (reservoir_inflow >= flood_control_outflow) {
            float flood = normal_outflow
                + (total_storage - conservation_volume)
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

    reservoir_outflow = clamp(
        reservoir_outflow, 0.0f, total_storage / time_step);
    args.river_outflow_ptr[catchment] = reservoir_outflow;
    args.flood_outflow_ptr[catchment] = 0.0f;
    args.total_storage_ptr[catchment] = total_storage;

    atomic_fetch_add_explicit(
        &args.outgoing_storage_ptr[catchment],
        reservoir_outflow * time_step, memory_order_relaxed);
