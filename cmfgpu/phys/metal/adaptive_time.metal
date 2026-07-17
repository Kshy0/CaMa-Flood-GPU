// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0

#include <metal_stdlib>
using namespace metal;

constant bool kHasReservoir [[function_constant(0)]];
constant bool kBatchedDownstreamDistance [[function_constant(1)]];

struct compute_adaptive_time_step_args {
    device float* river_depth_buf [[id(0)]];
    device float* downstream_distance_buf [[id(1)]];
    device int* is_dam_related_buf [[id(2)]];
    device atomic_int* max_sub_steps_buf [[id(3)]];
    constant float* time_step [[id(4)]];
    constant float* adaptive_time_factor [[id(5)]];
    constant float* gravity [[id(6)]];
    constant int* num_catchments [[id(7)]];
};

kernel void compute_adaptive_time_step(
    constant compute_adaptive_time_step_args& args [[buffer(0)]],
    uint idx [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
)
{
    device float* river_depth_buf = args.river_depth_buf;
    device float* downstream_distance_buf = args.downstream_distance_buf;
    device int* is_dam_related_buf = args.is_dam_related_buf;
    device atomic_int* max_sub_steps_buf = args.max_sub_steps_buf;
    const float time_step = *args.time_step;
    const float adaptive_time_factor = *args.adaptive_time_factor;
    const float gravity = *args.gravity;
    const int num_catchments = *args.num_catchments;
    // Shared memory for block-level min reduction
    threadgroup float shared_min[256];

    if ((int)idx >= num_catchments) {
        shared_min[lid] = time_step;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        // Skip dam-related cells
        bool skip = false;
        if (kHasReservoir) {
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


// =====================================================================
// Batched adaptive time step kernel
// Grid: num_catchments * num_trials
// =====================================================================
struct compute_adaptive_time_step_batched_args {
    device float* river_depth_buf [[id(0)]];
    device float* downstream_distance_buf [[id(1)]];
    device int* is_dam_related_buf [[id(2)]];
    device atomic_int* max_sub_steps_buf [[id(3)]];
    constant float* time_step [[id(4)]];
    constant float* adaptive_time_factor [[id(5)]];
    constant float* gravity [[id(6)]];
    constant int* num_catchments [[id(7)]];
    constant int* num_trials [[id(8)]];
};

kernel void compute_adaptive_time_step_batched(
    constant compute_adaptive_time_step_batched_args& args [[buffer(0)]],
    uint idx [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
)
{
    device float* river_depth_buf = args.river_depth_buf;
    device float* downstream_distance_buf = args.downstream_distance_buf;
    device int* is_dam_related_buf = args.is_dam_related_buf;
    device atomic_int* max_sub_steps_buf = args.max_sub_steps_buf;
    const float time_step = *args.time_step;
    const float adaptive_time_factor = *args.adaptive_time_factor;
    const float gravity = *args.gravity;
    const int num_catchments = *args.num_catchments;
    const int num_trials = *args.num_trials;
    threadgroup float shared_min[256];

    int total = num_catchments * num_trials;
    if ((int)idx >= total) {
        shared_min[lid] = time_step;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        int catchment_idx = (int)idx % num_catchments;
        int trial_offset = ((int)idx / num_catchments) * num_catchments;

        bool skip = false;
        if (kHasReservoir) {
            skip = (is_dam_related_buf[catchment_idx] != 0);
        }

        int ds_off = kBatchedDownstreamDistance ? (trial_offset + catchment_idx) : catchment_idx;
        float ds_dist = downstream_distance_buf[ds_off];
        float riv_depth = river_depth_buf[trial_offset + catchment_idx];
        float depth = max(riv_depth, 0.01f);
        float dt = adaptive_time_factor * ds_dist / sqrt(gravity * depth);
        float dt_clamped = min(dt, time_step);

        shared_min[lid] = skip ? time_step : dt_clamped;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_min[lid] = min(shared_min[lid], shared_min[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        float min_dt = shared_min[0];
        float n_steps_f = floor(time_step / min_dt + 0.49f) + 1.0f;
        int n_steps = (int)n_steps_f;
        // max_sub_steps is shared_state shape (1,) — always write offset 0
        atomic_fetch_max_explicit(&max_sub_steps_buf[0], n_steps, memory_order_relaxed);
    }
}
