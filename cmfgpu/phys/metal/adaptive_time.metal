// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0

#include <metal_stdlib>
using namespace metal;

kernel void compute_adaptive_time_step(
    device float*         river_depth_buf          [[buffer(0)]],
    device float*         downstream_distance_buf  [[buffer(1)]],
    device int*           is_dam_related_buf       [[buffer(2)]],
    device atomic_int*    max_sub_steps_buf        [[buffer(3)]],
    constant float&       time_step                [[buffer(4)]],
    constant float&       adaptive_time_factor     [[buffer(5)]],
    constant float&       gravity                  [[buffer(6)]],
    constant int&         num_catchments           [[buffer(7)]],
    constant int&         has_reservoir            [[buffer(8)]],
    uint idx [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
)
{
    // Shared memory for block-level min reduction
    threadgroup float shared_min[256];

    if ((int)idx >= num_catchments) {
        shared_min[lid] = time_step;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        // Skip dam-related cells
        bool skip = false;
        if (has_reservoir) {
            skip = (is_dam_related_buf[idx] != 0);
        }

        float ds_dist = downstream_distance_buf[idx];
        float riv_depth = river_depth_buf[idx];
        float depth = max(riv_depth, 0.01f);
        float dt = adaptive_time_factor * ds_dist / sqrt(gravity * depth);
        float dt_clamped = min(dt, time_step);

        shared_min[lid] = skip ? time_step : dt_clamped;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Tree reduction for min
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_min[lid] = min(shared_min[lid], shared_min[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // First thread in group computes n_steps and does atomic max
    if (lid == 0) {
        float min_dt = shared_min[0];
        float n_steps_f = floor(time_step / min_dt + 0.49f) + 1.0f;
        int n_steps = (int)n_steps_f;
        atomic_fetch_max_explicit(&max_sub_steps_buf[0], n_steps, memory_order_relaxed);
    }
}
