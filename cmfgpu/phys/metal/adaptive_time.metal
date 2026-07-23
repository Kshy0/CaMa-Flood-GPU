// HYDROFORGE METAL KERNEL BODY
    threadgroup float shared_min[HF_BLOCK_SIZE];

    long num_catchments = *args.num_catchments;
    long num_trials = *args.num_trials;
    long total = num_catchments * num_trials;
    float outer_dt = *args.outer_time_step;

    if ((long)i >= total) {
        shared_min[lid] = outer_dt;
    } else {
        long catchment = (long)i % num_catchments;
        long trial_offset = ((long)i / num_catchments) * num_catchments;

        bool skip = false;
        if (HAS_RESERVOIR) {
            skip = args.is_dam_related_ptr[catchment] != 0;
        }

        long distance_offset = batched_downstream_distance
            ? trial_offset + catchment : catchment;
        float downstream_distance =
            args.downstream_distance_ptr[distance_offset];
        float river_depth = args.river_depth_ptr[trial_offset + catchment];
        float depth = max(river_depth, 0.01f);
        float candidate = *args.adaptive_time_factor
            * downstream_distance / sqrt(*args.gravity * depth);
        shared_min[lid] = skip ? outer_dt : min(candidate, outer_dt);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_min[lid] = min(
                shared_min[lid], shared_min[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        float minimum_dt = shared_min[0];
        int sub_steps = (int)floor(outer_dt / minimum_dt + 0.49f) + 1;
        atomic_fetch_max_explicit(
            args.max_sub_steps_ptr, sub_steps, memory_order_relaxed);
    }
