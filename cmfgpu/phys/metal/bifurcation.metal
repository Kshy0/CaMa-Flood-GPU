// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0

#include <metal_stdlib>
using namespace metal;

kernel void compute_bifurcation_outflow(
    device int*           bif_catchment_idx_buf          [[buffer(0)]],
    device int*           bif_downstream_idx_buf         [[buffer(1)]],
    device float*         bif_manning_buf                [[buffer(2)]],
    device float*         bif_outflow_buf                [[buffer(3)]],
    device float*         bif_width_buf                  [[buffer(4)]],
    device float*         bif_length_buf                 [[buffer(5)]],
    device float*         bif_elevation_buf              [[buffer(6)]],
    device float*         bif_cross_section_depth_buf    [[buffer(7)]],
    device float*         water_surface_elevation_buf    [[buffer(8)]],
    device float*         total_storage_buf              [[buffer(9)]],
    device atomic_float*  outgoing_storage_buf           [[buffer(10)]],
    constant float&       gravity                        [[buffer(11)]],
    constant float&       time_step                      [[buffer(12)]],
    constant int&         num_bifurcation_paths          [[buffer(13)]],
    uint idx [[thread_position_in_grid]]
)
{
    if ((int)idx >= num_bifurcation_paths) return;

    int catch_idx = bif_catchment_idx_buf[idx];
    int ds_idx    = bif_downstream_idx_buf[idx];

    float bif_length = bif_length_buf[idx];

    float wse      = water_surface_elevation_buf[catch_idx];
    float wse_ds   = water_surface_elevation_buf[ds_idx];
    float max_wse  = max(wse, wse_ds);

    float bif_slope = (wse - wse_ds) / bif_length;
    bif_slope = clamp(bif_slope, -0.005f, 0.005f);

    float total_sto    = total_storage_buf[catch_idx];
    float total_sto_ds = total_storage_buf[ds_idx];

    float sum_outflow = 0.0f;

    // Unrolled loop over bifurcation levels
    for (int level = 0; level < __NUM_BIF_LEVELS__; level++) {
        int level_idx = (int)idx * __NUM_BIF_LEVELS__ + level;

        float manning   = bif_manning_buf[level_idx];
        float csd       = bif_cross_section_depth_buf[level_idx];
        float elevation = bif_elevation_buf[level_idx];

        float updated_csd = max(max_wse - elevation, 0.0f);

        float semi_d = max(
            sqrt(updated_csd * csd),
            sqrt(updated_csd * 0.01f)
        );

        float width   = bif_width_buf[level_idx];
        float outflow = bif_outflow_buf[level_idx];
        float unit_q  = outflow / width;

        float num = width * (
            unit_q + gravity * time_step * semi_d * bif_slope
        );
        float den = 1.0f + gravity * time_step * (manning * manning) * fabs(unit_q)
                    * pow(semi_d, -7.0f / 3.0f);

        float updated_outflow = num / den;
        updated_outflow = (semi_d > 1e-5f) ? updated_outflow : 0.0f;

        sum_outflow += updated_outflow;
        bif_cross_section_depth_buf[level_idx] = updated_csd;
        bif_outflow_buf[level_idx] = updated_outflow;
    }

    // Limit rate
    float limit_rate = min(
        0.05f * min(total_sto, total_sto_ds) / (fabs(sum_outflow) * time_step),
        1.0f
    );
    sum_outflow *= limit_rate;

    // Apply limit rate to per-level outflow
    for (int level = 0; level < __NUM_BIF_LEVELS__; level++) {
        int level_idx = (int)idx * __NUM_BIF_LEVELS__ + level;
        bif_outflow_buf[level_idx] *= limit_rate;
    }

    // Scatter to outgoing storage
    float pos_flow = max(sum_outflow, 0.0f);
    float neg_flow = min(sum_outflow, 0.0f);
    atomic_fetch_add_explicit(&outgoing_storage_buf[catch_idx],  pos_flow * time_step, memory_order_relaxed);
    atomic_fetch_add_explicit(&outgoing_storage_buf[ds_idx],    -neg_flow * time_step, memory_order_relaxed);
}

kernel void compute_bifurcation_inflow(
    device int*           bif_catchment_idx_buf          [[buffer(0)]],
    device int*           bif_downstream_idx_buf         [[buffer(1)]],
    device float*         limit_rate_buf                 [[buffer(2)]],
    device float*         bif_outflow_buf                [[buffer(3)]],
    device atomic_float*  global_bif_outflow_buf         [[buffer(4)]],
    constant int&         num_bifurcation_paths          [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
)
{
    if ((int)idx >= num_bifurcation_paths) return;

    int catch_idx = bif_catchment_idx_buf[idx];
    int ds_idx    = bif_downstream_idx_buf[idx];

    float lr    = limit_rate_buf[catch_idx];
    float lr_ds = limit_rate_buf[ds_idx];

    float sum_outflow = 0.0f;

    for (int level = 0; level < __NUM_BIF_LEVELS__; level++) {
        int level_idx = (int)idx * __NUM_BIF_LEVELS__ + level;
        float outflow = bif_outflow_buf[level_idx];
        outflow = (outflow >= 0.0f) ? outflow * lr : outflow * lr_ds;
        sum_outflow += outflow;
        bif_outflow_buf[level_idx] = outflow;
    }

    atomic_fetch_add_explicit(&global_bif_outflow_buf[catch_idx],  sum_outflow, memory_order_relaxed);
    atomic_fetch_add_explicit(&global_bif_outflow_buf[ds_idx],    -sum_outflow, memory_order_relaxed);
}


// =====================================================================
// Batched bifurcation outflow — flat grid num_bif_paths * num_trials
// =====================================================================
kernel void compute_bifurcation_outflow_batched(
    device int*           bif_catchment_idx_buf          [[buffer(0)]],
    device int*           bif_downstream_idx_buf         [[buffer(1)]],
    device float*         bif_manning_buf                [[buffer(2)]],
    device float*         bif_outflow_buf                [[buffer(3)]],
    device float*         bif_width_buf                  [[buffer(4)]],
    device float*         bif_length_buf                 [[buffer(5)]],
    device float*         bif_elevation_buf              [[buffer(6)]],
    device float*         bif_cross_section_depth_buf    [[buffer(7)]],
    device float*         water_surface_elevation_buf    [[buffer(8)]],
    device float*         total_storage_buf              [[buffer(9)]],
    device atomic_float*  outgoing_storage_buf           [[buffer(10)]],
    constant float&       gravity                        [[buffer(11)]],
    constant float&       time_step                      [[buffer(12)]],
    constant int&         num_bifurcation_paths          [[buffer(13)]],
    constant int&         num_catchments                 [[buffer(14)]],
    constant int&         num_trials                     [[buffer(15)]],
    constant int&         batched_bif_manning_flag       [[buffer(16)]],
    constant int&         batched_bif_width_flag         [[buffer(17)]],
    constant int&         batched_bif_length_flag        [[buffer(18)]],
    constant int&         batched_bif_elevation_flag     [[buffer(19)]],
    uint gid [[thread_position_in_grid]]
)
{
    int total = num_bifurcation_paths * num_trials;
    if ((int)gid >= total) return;

    int pi = (int)gid % num_bifurcation_paths;  // path index
    int trial_idx = (int)gid / num_bifurcation_paths;
    int to_paths = trial_idx * num_bifurcation_paths;
    int to_catch = trial_idx * num_catchments;
    int to_levels = trial_idx * num_bifurcation_paths * __NUM_BIF_LEVELS__;

    int catch_idx = bif_catchment_idx_buf[pi];
    int ds_idx    = bif_downstream_idx_buf[pi];

    float bif_length = bif_length_buf[(batched_bif_length_flag ? to_paths : 0) + pi];

    float wse      = water_surface_elevation_buf[to_catch + catch_idx];
    float wse_ds   = water_surface_elevation_buf[to_catch + ds_idx];
    float max_wse  = max(wse, wse_ds);

    float bif_slope = clamp((wse - wse_ds) / bif_length, -0.005f, 0.005f);

    float total_sto    = total_storage_buf[to_catch + catch_idx];
    float total_sto_ds = total_storage_buf[to_catch + ds_idx];

    int manning_base   = batched_bif_manning_flag   ? to_levels : 0;
    int width_base     = batched_bif_width_flag     ? to_levels : 0;
    int elevation_base = batched_bif_elevation_flag ? to_levels : 0;

    float sum_outflow = 0.0f;

    for (int level = 0; level < __NUM_BIF_LEVELS__; level++) {
        int level_idx = pi * __NUM_BIF_LEVELS__ + level;

        float manning   = bif_manning_buf[manning_base + level_idx];
        float csd       = bif_cross_section_depth_buf[to_levels + level_idx];
        float elevation = bif_elevation_buf[elevation_base + level_idx];

        float updated_csd = max(max_wse - elevation, 0.0f);
        float semi_d = max(sqrt(updated_csd * csd), sqrt(updated_csd * 0.01f));

        float width   = bif_width_buf[width_base + level_idx];
        float outflow = bif_outflow_buf[to_levels + level_idx];
        float unit_q  = outflow / width;

        float num = width * (unit_q + gravity * time_step * semi_d * bif_slope);
        float den = 1.0f + gravity * time_step * (manning * manning) * fabs(unit_q)
                    * pow(semi_d, -7.0f / 3.0f);

        float updated_outflow = (semi_d > 1e-5f) ? (num / den) : 0.0f;

        sum_outflow += updated_outflow;
        bif_cross_section_depth_buf[to_levels + level_idx] = updated_csd;
        bif_outflow_buf[to_levels + level_idx] = updated_outflow;
    }

    float limit_rate = min(
        0.05f * min(total_sto, total_sto_ds) / (fabs(sum_outflow) * time_step),
        1.0f
    );
    sum_outflow *= limit_rate;

    for (int level = 0; level < __NUM_BIF_LEVELS__; level++) {
        int level_idx = pi * __NUM_BIF_LEVELS__ + level;
        bif_outflow_buf[to_levels + level_idx] *= limit_rate;
    }

    float pos_flow = max(sum_outflow, 0.0f);
    float neg_flow = min(sum_outflow, 0.0f);
    atomic_fetch_add_explicit(&outgoing_storage_buf[to_catch + catch_idx],  pos_flow * time_step, memory_order_relaxed);
    atomic_fetch_add_explicit(&outgoing_storage_buf[to_catch + ds_idx],    -neg_flow * time_step, memory_order_relaxed);
}


// =====================================================================
// Batched bifurcation inflow — flat grid num_bif_paths * num_trials
// =====================================================================
kernel void compute_bifurcation_inflow_batched(
    device int*           bif_catchment_idx_buf          [[buffer(0)]],
    device int*           bif_downstream_idx_buf         [[buffer(1)]],
    device float*         limit_rate_buf                 [[buffer(2)]],
    device float*         bif_outflow_buf                [[buffer(3)]],
    device atomic_float*  global_bif_outflow_buf         [[buffer(4)]],
    constant int&         num_bifurcation_paths          [[buffer(5)]],
    constant int&         num_catchments                 [[buffer(6)]],
    constant int&         num_trials                     [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
)
{
    int total = num_bifurcation_paths * num_trials;
    if ((int)gid >= total) return;

    int pi = (int)gid % num_bifurcation_paths;
    int trial_idx = (int)gid / num_bifurcation_paths;
    int to_catch = trial_idx * num_catchments;
    int to_levels = trial_idx * num_bifurcation_paths * __NUM_BIF_LEVELS__;

    int catch_idx = bif_catchment_idx_buf[pi];
    int ds_idx    = bif_downstream_idx_buf[pi];

    float lr    = limit_rate_buf[to_catch + catch_idx];
    float lr_ds = limit_rate_buf[to_catch + ds_idx];

    float sum_outflow = 0.0f;

    for (int level = 0; level < __NUM_BIF_LEVELS__; level++) {
        int level_idx = pi * __NUM_BIF_LEVELS__ + level;
        float outflow = bif_outflow_buf[to_levels + level_idx];
        outflow = (outflow >= 0.0f) ? outflow * lr : outflow * lr_ds;
        sum_outflow += outflow;
        bif_outflow_buf[to_levels + level_idx] = outflow;
    }

    atomic_fetch_add_explicit(&global_bif_outflow_buf[to_catch + catch_idx],  sum_outflow, memory_order_relaxed);
    atomic_fetch_add_explicit(&global_bif_outflow_buf[to_catch + ds_idx],    -sum_outflow, memory_order_relaxed);
}
