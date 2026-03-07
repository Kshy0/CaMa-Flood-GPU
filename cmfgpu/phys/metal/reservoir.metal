// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0

#include <metal_stdlib>
using namespace metal;

kernel void compute_reservoir_outflow(
    device int*           reservoir_catchment_idx_buf [[buffer(0)]],
    device int*           downstream_idx_buf          [[buffer(1)]],
    device atomic_float*  reservoir_total_inflow_buf  [[buffer(2)]],
    device float*         river_outflow_buf           [[buffer(3)]],
    device float*         flood_outflow_buf           [[buffer(4)]],
    device float*         river_storage_buf           [[buffer(5)]],
    device float*         flood_storage_buf           [[buffer(6)]],
    device float*         conservation_volume_buf     [[buffer(7)]],
    device float*         emergency_volume_buf        [[buffer(8)]],
    device float*         adjustment_volume_buf       [[buffer(9)]],
    device float*         normal_outflow_buf          [[buffer(10)]],
    device float*         adjustment_outflow_buf      [[buffer(11)]],
    device float*         flood_control_outflow_buf   [[buffer(12)]],
    device float*         runoff_buf                  [[buffer(13)]],
    device float*         total_storage_buf           [[buffer(14)]],
    device atomic_float*  outgoing_storage_buf        [[buffer(15)]],
    constant float&       time_step                   [[buffer(16)]],
    constant int&         num_reservoirs              [[buffer(17)]],
    uint idx [[thread_position_in_grid]]
)
{
    if ((int)idx >= num_reservoirs) return;

    // Index mapping
    int catch_idx = reservoir_catchment_idx_buf[idx];
    int ds_idx    = downstream_idx_buf[catch_idx];
    bool is_river_mouth = (ds_idx == catch_idx);

    // ================================================================
    // 1. Undo the main outflow kernel's outgoing_storage contribution
    // ================================================================
    float old_river_outflow = river_outflow_buf[catch_idx];
    float old_flood_outflow = flood_outflow_buf[catch_idx];

    float old_pos = max(old_river_outflow, 0.0f) + max(old_flood_outflow, 0.0f);
    float old_neg = min(old_river_outflow, 0.0f) + min(old_flood_outflow, 0.0f);

    // Subtract local positive contribution
    atomic_fetch_add_explicit(&outgoing_storage_buf[catch_idx], -(old_pos * time_step), memory_order_relaxed);

    // Undo downstream scatter of negative flow
    if (!is_river_mouth) {
        atomic_fetch_add_explicit(&outgoing_storage_buf[ds_idx], old_neg * time_step, memory_order_relaxed);
    }

    // ================================================================
    // 2. Compute reservoir outflow
    // ================================================================
    float river_storage = river_storage_buf[catch_idx];
    float flood_storage = flood_storage_buf[catch_idx];
    float total_storage = river_storage + flood_storage;

    // Read and zero the inflow accumulator
    float total_inflow = atomic_exchange_explicit(&reservoir_total_inflow_buf[catch_idx], 0.0f, memory_order_relaxed);
    float runoff = runoff_buf[catch_idx];
    float reservoir_inflow = total_inflow + runoff;

    // Reservoir parameters (reservoir-indexed)
    float conservation_volume   = conservation_volume_buf[idx];
    float emergency_volume      = emergency_volume_buf[idx];
    float adjustment_volume     = adjustment_volume_buf[idx];
    float normal_outflow        = normal_outflow_buf[idx];
    float adjustment_outflow    = adjustment_outflow_buf[idx];
    float flood_control_outflow = flood_control_outflow_buf[idx];

    float reservoir_outflow = 0.0f;

    // Case 1: below conservation volume
    if (total_storage <= conservation_volume) {
        reservoir_outflow = normal_outflow * sqrt(total_storage / conservation_volume);
    }
    // Case 2: above conservation, below adjustment volume
    else if (total_storage <= adjustment_volume) {
        float frac2 = (total_storage - conservation_volume) / (adjustment_volume - conservation_volume);
        reservoir_outflow = normal_outflow + exp(3.0f * log(frac2)) * (adjustment_outflow - normal_outflow);
    }
    // Case 3: above adjustment, below emergency volume
    else if (total_storage <= emergency_volume) {
        float frac3 = (total_storage - adjustment_volume) / (emergency_volume - adjustment_volume);
        bool flood_period = (reservoir_inflow >= flood_control_outflow);

        if (flood_period) {
            float outflow_flood = normal_outflow + (
                (total_storage - conservation_volume) / (emergency_volume - conservation_volume)
            ) * (reservoir_inflow - normal_outflow);
            float outflow_tmp = adjustment_outflow + exp(0.1f * log(frac3)) * (
                flood_control_outflow - adjustment_outflow
            );
            reservoir_outflow = max(outflow_flood, outflow_tmp);
        } else {
            reservoir_outflow = adjustment_outflow + exp(0.1f * log(frac3)) * (
                flood_control_outflow - adjustment_outflow
            );
        }
    }
    // Case 4: above emergency volume
    else {
        reservoir_outflow = (reservoir_inflow >= flood_control_outflow)
            ? reservoir_inflow : flood_control_outflow;
    }

    // Clamp to [0, total_storage / time_step]
    reservoir_outflow = clamp(reservoir_outflow, 0.0f, total_storage / time_step);

    // ================================================================
    // 3. Store results
    // ================================================================
    river_outflow_buf[catch_idx] = reservoir_outflow;
    flood_outflow_buf[catch_idx] = 0.0f;
    total_storage_buf[catch_idx] = total_storage;

    // Re-add corrected contribution
    float new_pos = max(reservoir_outflow, 0.0f);
    atomic_fetch_add_explicit(&outgoing_storage_buf[catch_idx], new_pos * time_step, memory_order_relaxed);

    float new_neg = min(reservoir_outflow, 0.0f);
    if (!is_river_mouth) {
        atomic_fetch_add_explicit(&outgoing_storage_buf[ds_idx], -(new_neg * time_step), memory_order_relaxed);
    }
}
