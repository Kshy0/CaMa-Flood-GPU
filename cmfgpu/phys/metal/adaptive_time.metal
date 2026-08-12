// HYDROFORGE METAL KERNEL BODY: compute_adaptive_time_step
    threadgroup int shared_steps[HF_BLOCK_SIZE];

    long num_catchments = *args.num_catchments;
    long num_trials = *args.num_trials;
    long total = num_catchments * num_trials;
    float outer_dt = *args.outer_time_step;

    if ((long)i >= total) {
        shared_steps[lid] = 0;
    } else {
        long catchment = (long)i % num_catchments;
        long trial_offset = ((long)i / num_catchments) * num_catchments;

        bool skip = false;
        if (HAS_RESERVOIR) {
            skip = args.is_dam_related_ptr[catchment] != 0;
        }

        long distance_offset = batched_downstream_distance
            ? trial_offset + catchment : catchment;
        if (skip) {
            shared_steps[lid] = 0;
        } else {
            float downstream_distance =
                args.downstream_distance_ptr[distance_offset];
            float river_depth = args.river_depth_ptr[trial_offset + catchment];
            float depth = max(river_depth, 0.01f);
            float candidate = *args.adaptive_time_factor
                * downstream_distance / sqrt(*args.gravity * depth);
            float minimum_dt = min(candidate, outer_dt);
            shared_steps[lid] = (int)(
                floor(outer_dt / minimum_dt - 0.01f) + 1.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dispatchThreads permits a short final threadgroup.  Derive its logical
    // active width from the grid extent, then fold every odd tail lane.
    long group_start = (long)i - (long)lid;
    uint group_threads = (uint)min((long)tpg, total - group_start);
    for (uint active_threads = group_threads; active_threads > 1;
            active_threads = (active_threads + 1) / 2) {
        uint next_active = (active_threads + 1) / 2;
        uint pair = lid + next_active;
        if (pair < active_threads) {
            shared_steps[lid] = max(
                shared_steps[lid], shared_steps[pair]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        int sub_steps = shared_steps[0];
        if (sub_steps > 0) {
            atomic_fetch_max_explicit(
                args.max_sub_steps_ptr, sub_steps, memory_order_relaxed);
        }
    }
