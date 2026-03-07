// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0

#include <metal_stdlib>
using namespace metal;

#define NUM_FLOOD_LEVELS __NUM_FLOOD_LEVELS__

// log_sums layout  (row major, stride = log_buffer_size):
//   row  0 = total_storage_pre_sum
//   row  1 = total_storage_next_sum
//   row  2 = total_storage_new_sum
//   row  3 = total_inflow_error_sum
//   row  4 = total_inflow_sum
//   row  5 = total_outflow_sum
//   row  6 = total_storage_stage_sum
//   row  7 = total_stage_error_sum
//   row  8 = river_storage_sum
//   row  9 = flood_storage_sum
//   row 10 = flood_area_sum

static inline void atomic_add_float(device atomic_float* addr, float val) {
    atomic_fetch_add_explicit(addr, val, memory_order_relaxed);
}

kernel void compute_flood_stage(
    // Storage update inputs
    device float*         river_inflow_buf         [[buffer(0)]],
    device float*         flood_inflow_buf         [[buffer(1)]],
    device float*         river_outflow_buf        [[buffer(2)]],
    device float*         flood_outflow_buf        [[buffer(3)]],
    device float*         bif_outflow_buf          [[buffer(4)]],
    device float*         runoff_buf               [[buffer(5)]],
    // Storage in/out
    device float*         outgoing_storage_buf     [[buffer(6)]],
    device float*         river_storage_buf        [[buffer(7)]],
    device float*         flood_storage_buf        [[buffer(8)]],
    device float*         protected_storage_buf    [[buffer(9)]],
    // Depth/fraction outputs
    device float*         river_depth_buf          [[buffer(10)]],
    device float*         flood_depth_buf          [[buffer(11)]],
    device float*         protected_depth_buf      [[buffer(12)]],
    device float*         flood_fraction_buf       [[buffer(13)]],
    // Reference tables
    device float*         river_height_buf         [[buffer(14)]],
    device float*         flood_depth_table_buf    [[buffer(15)]],
    device float*         catchment_area_buf       [[buffer(16)]],
    device float*         river_width_buf          [[buffer(17)]],
    device float*         river_length_buf         [[buffer(18)]],
    // Scalars
    constant float&       time_step                [[buffer(19)]],
    constant int&         num_catchments           [[buffer(20)]],
    constant int&         has_bifurcation          [[buffer(21)]],
    uint idx [[thread_position_in_grid]]
)
{
    if ((int)idx >= num_catchments) return;

    // ---- 1. Storage update ----
    float riv_sto  = river_storage_buf[idx];
    float fld_sto  = flood_storage_buf[idx];
    float prot_sto = protected_storage_buf[idx];
    float riv_in   = river_inflow_buf[idx];
    float fld_in   = flood_inflow_buf[idx];
    float riv_out  = river_outflow_buf[idx];
    float fld_out  = flood_outflow_buf[idx];
    float bif_out  = has_bifurcation ? bif_outflow_buf[idx] : 0.0f;
    float runoff   = runoff_buf[idx];

    float riv_sto_upd = riv_sto + (riv_in - riv_out) * time_step;
    float fld_sto_upd = fld_sto
        + (riv_sto_upd < 0.0f ? riv_sto_upd : 0.0f)
        + (fld_in - fld_out - bif_out) * time_step;
    riv_sto_upd = max(riv_sto_upd, 0.0f);
    if (fld_sto_upd < 0.0f) {
        riv_sto_upd = max(riv_sto_upd + fld_sto_upd, 0.0f);
    }
    fld_sto_upd = max(fld_sto_upd, 0.0f);
    float total_sto = max(riv_sto_upd + fld_sto_upd + prot_sto + runoff * time_step, 0.0f);

    // ---- 2. Flood stage via table scan ----
    float riv_height = river_height_buf[idx];
    float catch_area = catchment_area_buf[idx];
    float riv_width  = river_width_buf[idx];
    float riv_length = river_length_buf[idx];

    float riv_max_sto  = riv_length * riv_width * riv_height;
    float catch_width  = catch_area / riv_length;
    float width_inc    = catch_width / (float)NUM_FLOOD_LEVELS;

    int   level = (total_sto > riv_max_sto) ? 0 : -1;
    float S_accum = riv_max_sto;
    float prev_H = 0.0f;
    float prev_W = riv_width;
    float prev_total_sto = riv_max_sto;
    float prev_fld_depth = 0.0f;
    float next_fld_depth = 0.0f;

    for (int i = 0; i < NUM_FLOOD_LEVELS; i++) {
        float H_curr = flood_depth_table_buf[idx * NUM_FLOOD_LEVELS + i];
        float W_curr = riv_width + (float)(i + 1) * width_inc;
        float dS = riv_length * 0.5f * (prev_W + W_curr) * (H_curr - prev_H);
        float S_curr = S_accum + dS;

        if (level == i) next_fld_depth = H_curr;

        if (total_sto > S_curr) {
            level += 1;
            prev_total_sto = S_curr;
            prev_fld_depth = H_curr;
        }

        S_accum = S_curr;
        prev_H = H_curr;
        prev_W = W_curr;
    }

    bool no_flood = (level < 0);
    level = max(level, 0);

    float prev_total_W = riv_width + (float)level * width_inc;
    float flood_grad = (level == NUM_FLOOD_LEVELS)
        ? 0.0f
        : (next_fld_depth - prev_fld_depth) / width_inc;

    float diff_W = sqrt(
        prev_total_W * prev_total_W
        + 2.0f * (total_sto - prev_total_sto) / (flood_grad * riv_length)
    ) - prev_total_W;
    float fld_depth_mid = prev_fld_depth + diff_W * flood_grad;
    float fld_depth_top = prev_fld_depth + (total_sto - prev_total_sto) / (prev_total_W * riv_length);

    float fld_depth = no_flood ? 0.0f
        : (level == NUM_FLOOD_LEVELS ? fld_depth_top : fld_depth_mid);

    float riv_sto_final = no_flood
        ? total_sto
        : min(riv_max_sto + riv_length * riv_width * fld_depth, total_sto);
    float riv_depth = riv_sto_final / (riv_length * riv_width);

    float fld_frac_mid = clamp(
        (prev_total_W + diff_W - riv_width) * riv_length / catch_area, 0.0f, 1.0f);
    float fld_frac = no_flood ? 0.0f
        : (level == NUM_FLOOD_LEVELS ? 1.0f : fld_frac_mid);

    float fld_sto_final = max(total_sto - riv_sto_final, 0.0f);

    // ---- 3. Store ----
    outgoing_storage_buf[idx]   = 0.0f;
    river_storage_buf[idx]      = riv_sto_final;
    flood_storage_buf[idx]      = fld_sto_final;
    protected_storage_buf[idx]  = 0.0f;
    river_depth_buf[idx]        = riv_depth;
    flood_depth_buf[idx]        = fld_depth;
    protected_depth_buf[idx]    = fld_depth;
    flood_fraction_buf[idx]     = fld_frac;
}

kernel void compute_flood_stage_log(
    // Storage update inputs  (buffer binding order matches __init__.py args)
    device float*          river_inflow_buf              [[buffer(0)]],
    device float*          flood_inflow_buf              [[buffer(1)]],
    device float*          river_outflow_buf             [[buffer(2)]],
    device float*          flood_outflow_buf             [[buffer(3)]],
    device float*          bif_outflow_buf               [[buffer(4)]],
    device float*          runoff_buf                    [[buffer(5)]],
    // Storage in/out
    device float*          outgoing_storage_buf          [[buffer(6)]],
    device float*          river_storage_buf             [[buffer(7)]],
    device float*          flood_storage_buf             [[buffer(8)]],
    device float*          protected_storage_buf         [[buffer(9)]],
    // Depth/fraction outputs
    device float*          river_depth_buf               [[buffer(10)]],
    device float*          flood_depth_buf               [[buffer(11)]],
    device float*          protected_depth_buf           [[buffer(12)]],
    device float*          flood_fraction_buf            [[buffer(13)]],
    // Reference tables
    device float*          river_height_buf              [[buffer(14)]],
    device float*          flood_depth_table_buf         [[buffer(15)]],
    device float*          catchment_area_buf            [[buffer(16)]],
    device float*          river_width_buf               [[buffer(17)]],
    device float*          river_length_buf              [[buffer(18)]],
    // Levee mask
    device int*            is_levee_buf                  [[buffer(19)]],
    // Packed log sums — (11, log_buffer_size) contiguous
    device atomic_float*   log_sums_buf                  [[buffer(20)]],
    // Scalars
    constant float&        time_step                     [[buffer(21)]],
    constant int&          num_catchments                [[buffer(22)]],
    constant int&          has_bifurcation               [[buffer(23)]],
    device int*            current_step_buf              [[buffer(24)]],
    constant int&          log_buffer_size               [[buffer(25)]],
    uint idx [[thread_position_in_grid]]
)
{
    if ((int)idx >= num_catchments) return;

    int current_step = current_step_buf[0];
    int lbs = log_buffer_size;          // stride between rows
    bool non_levee = (is_levee_buf[idx] == 0);

    // ---- 1. Storage update ----
    float riv_sto  = river_storage_buf[idx];
    float fld_sto  = flood_storage_buf[idx];
    float prot_sto = protected_storage_buf[idx];
    float riv_in   = river_inflow_buf[idx];
    float fld_in   = flood_inflow_buf[idx];
    float riv_out  = river_outflow_buf[idx];
    float fld_out  = flood_outflow_buf[idx];
    float bif_out  = has_bifurcation ? bif_outflow_buf[idx] : 0.0f;
    float runoff   = runoff_buf[idx];

    // Pre-log: total storage before routing
    float total_stage_pre = riv_sto + fld_sto + prot_sto;
    atomic_add_float(&log_sums_buf[0 * lbs + current_step], total_stage_pre * 1e-9f);

    float riv_sto_upd = riv_sto + (riv_in - riv_out) * time_step;
    float fld_sto_upd = fld_sto
        + (riv_sto_upd < 0.0f ? riv_sto_upd : 0.0f)
        + (fld_in - fld_out - bif_out) * time_step;
    riv_sto_upd = max(riv_sto_upd, 0.0f);
    if (fld_sto_upd < 0.0f) {
        riv_sto_upd = max(riv_sto_upd + fld_sto_upd, 0.0f);
    }
    fld_sto_upd = max(fld_sto_upd, 0.0f);

    float total_sto_next = riv_sto_upd + fld_sto_upd + prot_sto + runoff * time_step;
    float total_sto = max(total_sto_next, 0.0f);

    // Mid-log: storage after routing, inflow/outflow, error  (non-levee only)
    if (non_levee) {
        atomic_add_float(&log_sums_buf[1 * lbs + current_step], total_sto_next * 1e-9f);
        atomic_add_float(&log_sums_buf[2 * lbs + current_step], total_sto * 1e-9f);
        atomic_add_float(&log_sums_buf[4 * lbs + current_step],
            (riv_in + fld_in) * time_step * 1e-9f);
        atomic_add_float(&log_sums_buf[5 * lbs + current_step],
            (riv_out + fld_out) * time_step * 1e-9f);
        float inflow_error = total_stage_pre - total_sto_next
            + (riv_in + fld_in + runoff - riv_out - fld_out - bif_out) * time_step;
        atomic_add_float(&log_sums_buf[3 * lbs + current_step], inflow_error * 1e-9f);
    }

    // ---- 2. Flood stage via table scan ----
    float riv_height = river_height_buf[idx];
    float catch_area = catchment_area_buf[idx];
    float riv_width  = river_width_buf[idx];
    float riv_length = river_length_buf[idx];

    float riv_max_sto  = riv_length * riv_width * riv_height;
    float catch_width  = catch_area / riv_length;
    float width_inc    = catch_width / (float)NUM_FLOOD_LEVELS;

    int   level = (total_sto > riv_max_sto) ? 0 : -1;
    float S_accum = riv_max_sto;
    float prev_H = 0.0f;
    float prev_W = riv_width;
    float prev_total_sto = riv_max_sto;
    float prev_fld_depth = 0.0f;
    float next_fld_depth = 0.0f;

    for (int i = 0; i < NUM_FLOOD_LEVELS; i++) {
        float H_curr = flood_depth_table_buf[idx * NUM_FLOOD_LEVELS + i];
        float W_curr = riv_width + (float)(i + 1) * width_inc;
        float dS = riv_length * 0.5f * (prev_W + W_curr) * (H_curr - prev_H);
        float S_curr = S_accum + dS;

        if (level == i) next_fld_depth = H_curr;

        if (total_sto > S_curr) {
            level += 1;
            prev_total_sto = S_curr;
            prev_fld_depth = H_curr;
        }

        S_accum = S_curr;
        prev_H = H_curr;
        prev_W = W_curr;
    }

    bool no_flood = (level < 0);
    level = max(level, 0);

    float prev_total_W = riv_width + (float)level * width_inc;
    float flood_grad = (level == NUM_FLOOD_LEVELS)
        ? 0.0f
        : (next_fld_depth - prev_fld_depth) / width_inc;

    float diff_W = sqrt(
        prev_total_W * prev_total_W
        + 2.0f * (total_sto - prev_total_sto) / (flood_grad * riv_length)
    ) - prev_total_W;
    float fld_depth_mid = prev_fld_depth + diff_W * flood_grad;
    float fld_depth_top = prev_fld_depth + (total_sto - prev_total_sto) / (prev_total_W * riv_length);

    float fld_depth = no_flood ? 0.0f
        : (level == NUM_FLOOD_LEVELS ? fld_depth_top : fld_depth_mid);

    float riv_sto_final = no_flood
        ? total_sto
        : min(riv_max_sto + riv_length * riv_width * fld_depth, total_sto);
    float riv_depth = riv_sto_final / (riv_length * riv_width);

    float fld_frac_mid = clamp(
        (prev_total_W + diff_W - riv_width) * riv_length / catch_area, 0.0f, 1.0f);
    float fld_frac = no_flood ? 0.0f
        : (level == NUM_FLOOD_LEVELS ? 1.0f : fld_frac_mid);

    float fld_sto_final = max(total_sto - riv_sto_final, 0.0f);

    // Post-log: stage results
    float total_storage_stage_new = riv_sto_final + fld_sto_final;
    atomic_add_float(&log_sums_buf[6 * lbs + current_step], total_storage_stage_new * 1e-9f);
    if (non_levee) {
        atomic_add_float(&log_sums_buf[8  * lbs + current_step], riv_sto_final * 1e-9f);
        atomic_add_float(&log_sums_buf[9  * lbs + current_step], fld_sto_final * 1e-9f);
        atomic_add_float(&log_sums_buf[10 * lbs + current_step], fld_frac * catch_area * 1e-9f);
        atomic_add_float(&log_sums_buf[7  * lbs + current_step],
            (total_storage_stage_new - total_sto) * 1e-9f);
    }

    // ---- 3. Store ----
    outgoing_storage_buf[idx]   = 0.0f;
    river_storage_buf[idx]      = riv_sto_final;
    flood_storage_buf[idx]      = fld_sto_final;
    protected_storage_buf[idx]  = 0.0f;
    river_depth_buf[idx]        = riv_depth;
    flood_depth_buf[idx]        = fld_depth;
    protected_depth_buf[idx]    = fld_depth;
    flood_fraction_buf[idx]     = fld_frac;
}
