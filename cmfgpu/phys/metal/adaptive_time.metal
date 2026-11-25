// HYDROFORGE METAL KERNEL BODY: compute_adaptive_time_step
    threadgroup int simd_partials[HF_BLOCK_SIZE / 8];

    long num_catchments = *args.num_catchments;
    long ensemble_size = *args.ensemble_size;
    long total = num_catchments * ensemble_size;
    float outer_dt = args.outer_time_step_ptr[0];

    int local_steps = 1;
    if ((long)i < total) {
        long catchment, member;
        cmf_decompose_index(i, num_catchments, ensemble_size, catchment, member);
        long member_offset = member * num_catchments;

        bool skip = false;
        if (HAS_RESERVOIR) {
            skip = args.is_dam_related_ptr[batched_is_dam_related ? (long)i : catchment] != 0;
        }

        if (!skip) {
            long distance_offset = batched_downstream_distance
                ? member_offset + catchment : catchment;
            float downstream_distance =
                args.downstream_distance_ptr[distance_offset];
            float river_depth = args.river_depth_ptr[member_offset + catchment];
            float depth = max(river_depth, 0.01f);
            float candidate = cmf_div(*args.adaptive_time_factor
                * downstream_distance, cmf_sqrt(*args.gravity * depth));
            float minimum_dt = min(candidate, outer_dt);
            local_steps = (int)(
                floor(outer_dt / minimum_dt - 0.01f) + 1.0f);
        }
    }

    int sg_max = simd_max(local_steps);
    if (simd_lane == 0) {
        simd_partials[simd_group] = sg_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        // dispatchThreads permits a short final threadgroup; count only the
        // SIMD-groups that hold grid threads and wrote a partial.
        long group_start = (long)i - (long)lid;
        uint group_threads = (uint)min((long)tpg, total - group_start);
        uint num_simd = (group_threads + simd_width - 1) / simd_width;
        int val = (simd_lane < num_simd) ? simd_partials[simd_lane] : 1;
        int block_max = simd_max(val);
        if (simd_lane == 0) {
            atomic_fetch_max_explicit(
                args.max_sub_steps_ptr, block_max, memory_order_relaxed);
        }
    }
