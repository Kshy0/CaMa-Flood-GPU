// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0

#include <metal_stdlib>
using namespace metal;

kernel void compute_levee_bifurcation_outflow(
    device int*           bif_catchment_idx_buf          [[buffer(0)]],
    device int*           bif_downstream_idx_buf         [[buffer(1)]],
    device float*         bif_manning_buf                [[buffer(2)]],
    device float*         bif_outflow_buf                [[buffer(3)]],
    device float*         bif_width_buf                  [[buffer(4)]],
    device float*         bif_length_buf                 [[buffer(5)]],
    device float*         bif_elevation_buf              [[buffer(6)]],
    device float*         bif_cross_section_depth_buf    [[buffer(7)]],
    device float*         water_surface_elevation_buf    [[buffer(8)]],
    device float*         protected_wse_buf              [[buffer(9)]],
    device float*         total_storage_buf              [[buffer(10)]],
    device atomic_float*  outgoing_storage_buf           [[buffer(11)]],
    constant float&       gravity                        [[buffer(12)]],
    constant float&       time_step                      [[buffer(13)]],
    constant int&         num_bifurcation_paths          [[buffer(14)]],
    uint idx_g [[thread_position_in_grid]]
)
{
    const int NUM_BIF_LEVELS = __NUM_BIF_LEVELS__;

    if ((int)idx_g >= num_bifurcation_paths) return;

    int catch_idx = bif_catchment_idx_buf[idx_g];
    int ds_idx    = bif_downstream_idx_buf[idx_g];

    float bif_length = bif_length_buf[idx_g];

    // River WSE
    float wse    = water_surface_elevation_buf[catch_idx];
    float wse_ds = water_surface_elevation_buf[ds_idx];
    float max_wse = max(wse, wse_ds);

    // Protected WSE
    float p_wse    = protected_wse_buf[catch_idx];
    float p_wse_ds = protected_wse_buf[ds_idx];
    float max_p_wse = max(p_wse, p_wse_ds);

    float bif_slope = clamp((wse - wse_ds) / bif_length, -0.005f, 0.005f);

    float total_sto    = total_storage_buf[catch_idx];
    float total_sto_ds = total_storage_buf[ds_idx];

    float sum_outflow = 0.0f;

    for (int level = 0; level < NUM_BIF_LEVELS; level++) {
        int level_idx = (int)idx_g * NUM_BIF_LEVELS + level;

        float manning   = bif_manning_buf[level_idx];
        float csd       = bif_cross_section_depth_buf[level_idx];
        float elevation = bif_elevation_buf[level_idx];

        // Level 0: river WSE; Level > 0: protected WSE
        float current_max_wse = (level == 0) ? max_wse : max_p_wse;
        float updated_csd = max(current_max_wse - elevation, 0.0f);

        // Level 0: semi-implicit; Level > 0: explicit
        float semi_d;
        if (level == 0) {
            semi_d = max(sqrt(updated_csd * csd), sqrt(updated_csd * 0.01f));
        } else {
            semi_d = updated_csd;
        }

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

    for (int level = 0; level < NUM_BIF_LEVELS; level++) {
        int level_idx = (int)idx_g * NUM_BIF_LEVELS + level;
        bif_outflow_buf[level_idx] *= limit_rate;
    }

    float pos_flow = max(sum_outflow, 0.0f);
    float neg_flow = min(sum_outflow, 0.0f);
    atomic_fetch_add_explicit(&outgoing_storage_buf[catch_idx],  pos_flow * time_step, memory_order_relaxed);
    atomic_fetch_add_explicit(&outgoing_storage_buf[ds_idx],    -neg_flow * time_step, memory_order_relaxed);
}


// =====================================================================
// Batched levee bifurcation outflow — flat-grid pattern
// grid = num_bifurcation_paths * num_trials
// =====================================================================
kernel void compute_levee_bifurcation_outflow_batched(
    device int*           bif_catchment_idx_buf          [[buffer(0)]],
    device int*           bif_downstream_idx_buf         [[buffer(1)]],
    device float*         bif_manning_buf                [[buffer(2)]],
    device float*         bif_outflow_buf                [[buffer(3)]],
    device float*         bif_width_buf                  [[buffer(4)]],
    device float*         bif_length_buf                 [[buffer(5)]],
    device float*         bif_elevation_buf              [[buffer(6)]],
    device float*         bif_cross_section_depth_buf    [[buffer(7)]],
    device float*         water_surface_elevation_buf    [[buffer(8)]],
    device float*         protected_wse_buf              [[buffer(9)]],
    device float*         total_storage_buf              [[buffer(10)]],
    device atomic_float*  outgoing_storage_buf           [[buffer(11)]],
    constant float&       gravity                        [[buffer(12)]],
    constant float&       time_step                      [[buffer(13)]],
    constant int&         num_bifurcation_paths          [[buffer(14)]],
    constant int&         num_catchments                 [[buffer(15)]],
    constant int&         num_trials                     [[buffer(16)]],
    constant int&         batched_bif_manning_flag       [[buffer(17)]],
    constant int&         batched_bif_width_flag         [[buffer(18)]],
    constant int&         batched_bif_length_flag        [[buffer(19)]],
    constant int&         batched_bif_elevation_flag     [[buffer(20)]],
    uint idx_g [[thread_position_in_grid]]
)
{
    const int NUM_BIF_LEVELS = __NUM_BIF_LEVELS__;

    if ((int)idx_g >= num_bifurcation_paths * num_trials) return;

    int offs      = (int)idx_g % num_bifurcation_paths;
    int trial_idx = (int)idx_g / num_bifurcation_paths;

    int to_paths  = trial_idx * num_bifurcation_paths;
    int to_catch  = trial_idx * num_catchments;
    int to_levels = trial_idx * num_bifurcation_paths * NUM_BIF_LEVELS;

    // Topology is never batched
    int catch_idx = bif_catchment_idx_buf[offs];
    int ds_idx    = bif_downstream_idx_buf[offs];

    float bif_length = bif_length_buf[(batched_bif_length_flag ? to_paths : 0) + offs];

    // River WSE
    float wse    = water_surface_elevation_buf[to_catch + catch_idx];
    float wse_ds = water_surface_elevation_buf[to_catch + ds_idx];
    float max_wse = max(wse, wse_ds);

    // Protected WSE
    float p_wse    = protected_wse_buf[to_catch + catch_idx];
    float p_wse_ds = protected_wse_buf[to_catch + ds_idx];
    float max_p_wse = max(p_wse, p_wse_ds);

    float bif_slope = clamp((wse - wse_ds) / bif_length, -0.005f, 0.005f);

    float total_sto    = total_storage_buf[to_catch + catch_idx];
    float total_sto_ds = total_storage_buf[to_catch + ds_idx];

    int manning_base   = batched_bif_manning_flag   ? to_levels : 0;
    int width_base     = batched_bif_width_flag     ? to_levels : 0;
    int elevation_base = batched_bif_elevation_flag ? to_levels : 0;

    float sum_outflow = 0.0f;

    for (int level = 0; level < NUM_BIF_LEVELS; level++) {
        int level_idx = offs * NUM_BIF_LEVELS + level;

        float manning   = bif_manning_buf[manning_base + level_idx];
        float csd       = bif_cross_section_depth_buf[to_levels + level_idx];
        float elevation = bif_elevation_buf[elevation_base + level_idx];

        float current_max_wse = (level == 0) ? max_wse : max_p_wse;
        float updated_csd = max(current_max_wse - elevation, 0.0f);

        float semi_d;
        if (level == 0) {
            semi_d = max(sqrt(updated_csd * csd), sqrt(updated_csd * 0.01f));
        } else {
            semi_d = updated_csd;
        }

        float width   = bif_width_buf[width_base + level_idx];
        float outflow = bif_outflow_buf[to_levels + level_idx];
        float unit_q  = outflow / width;

        float num = width * (
            unit_q + gravity * time_step * semi_d * bif_slope
        );
        float den = 1.0f + gravity * time_step * (manning * manning) * fabs(unit_q)
                    * pow(semi_d, -7.0f / 3.0f);

        float updated_outflow = num / den;
        updated_outflow = (semi_d > 1e-5f) ? updated_outflow : 0.0f;

        sum_outflow += updated_outflow;
        bif_cross_section_depth_buf[to_levels + level_idx] = updated_csd;
        bif_outflow_buf[to_levels + level_idx] = updated_outflow;
    }

    // Limit rate
    float limit_rate = min(
        0.05f * min(total_sto, total_sto_ds) / (fabs(sum_outflow) * time_step),
        1.0f
    );
    sum_outflow *= limit_rate;

    for (int level = 0; level < NUM_BIF_LEVELS; level++) {
        int level_idx = offs * NUM_BIF_LEVELS + level;
        bif_outflow_buf[to_levels + level_idx] *= limit_rate;
    }

    float pos_flow = max(sum_outflow, 0.0f);
    float neg_flow = min(sum_outflow, 0.0f);
    atomic_fetch_add_explicit(&outgoing_storage_buf[to_catch + catch_idx],  pos_flow * time_step, memory_order_relaxed);
    atomic_fetch_add_explicit(&outgoing_storage_buf[to_catch + ds_idx],    -neg_flow * time_step, memory_order_relaxed);
}
