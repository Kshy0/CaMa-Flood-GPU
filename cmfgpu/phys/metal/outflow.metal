// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0

#include <metal_stdlib>
using namespace metal;

constant bool kHasBifurcation [[function_constant(0)]];
constant bool kHasSeaLevel [[function_constant(1)]];
constant bool kHasReservoir [[function_constant(2)]];
constant bool kBatchedRiverManning [[function_constant(3)]];
constant bool kBatchedFloodManning [[function_constant(4)]];
constant bool kBatchedRiverWidth [[function_constant(5)]];
constant bool kBatchedRiverLength [[function_constant(6)]];
constant bool kBatchedRiverHeight [[function_constant(7)]];
constant bool kBatchedCatchmentElevation [[function_constant(8)]];

// =====================================================================
// Outflow kernel — computes river/flood outflow via semi-implicit method
// =====================================================================
struct compute_outflow_args {
    // Tensor buffers
    device int* downstream_idx_buf [[id(0)]];
    device float* river_inflow_buf [[id(1)]];
    device float* river_outflow_buf [[id(2)]];
    device float* river_manning_buf [[id(3)]];
    device float* river_depth_buf [[id(4)]];
    device float* river_width_buf [[id(5)]];
    device float* river_length_buf [[id(6)]];
    device float* river_height_buf [[id(7)]];
    device float* river_storage_buf [[id(8)]];
    device float* flood_inflow_buf [[id(9)]];
    device float* flood_outflow_buf [[id(10)]];
    device float* flood_manning_buf [[id(11)]];
    device float* flood_depth_buf [[id(12)]];
    device float* protected_depth_buf [[id(13)]];
    device float* catchment_elevation_buf [[id(14)]];
    device float* downstream_distance_buf [[id(15)]];
    device float* flood_storage_buf [[id(16)]];
    device float* protected_storage_buf [[id(17)]];
    device float* river_cross_section_depth_buf [[id(18)]];
    device float* flood_cross_section_depth_buf [[id(19)]];
    device float* flood_cross_section_area_buf [[id(20)]];
    device atomic_float* global_bifurcation_outflow_buf [[id(21)]];
    device float* total_storage_buf [[id(22)]];
    device atomic_float* outgoing_storage_buf [[id(23)]];
    device float* water_surface_elevation_buf [[id(24)]];
    device float* prot_wse_buf [[id(25)]];
    // Scalars packed in constant buffers
    constant float* gravity [[id(26)]];
    constant float* time_step [[id(27)]];
    constant int* num_catchments [[id(28)]];
    device float* sea_surface_elevation_buf [[id(29)]];
    device int* catchment_sea_level_idx_buf [[id(30)]];
};

kernel void compute_outflow(
    constant compute_outflow_args& args [[buffer(0)]],
    uint idx [[thread_position_in_grid]]
)
{
    // Tensor buffers
    device int* downstream_idx_buf = args.downstream_idx_buf;
    device float* river_inflow_buf = args.river_inflow_buf;
    device float* river_outflow_buf = args.river_outflow_buf;
    device float* river_manning_buf = args.river_manning_buf;
    device float* river_depth_buf = args.river_depth_buf;
    device float* river_width_buf = args.river_width_buf;
    device float* river_length_buf = args.river_length_buf;
    device float* river_height_buf = args.river_height_buf;
    device float* river_storage_buf = args.river_storage_buf;
    device float* flood_inflow_buf = args.flood_inflow_buf;
    device float* flood_outflow_buf = args.flood_outflow_buf;
    device float* flood_manning_buf = args.flood_manning_buf;
    device float* flood_depth_buf = args.flood_depth_buf;
    device float* protected_depth_buf = args.protected_depth_buf;
    device float* catchment_elevation_buf = args.catchment_elevation_buf;
    device float* downstream_distance_buf = args.downstream_distance_buf;
    device float* flood_storage_buf = args.flood_storage_buf;
    device float* protected_storage_buf = args.protected_storage_buf;
    device float* river_cross_section_depth_buf = args.river_cross_section_depth_buf;
    device float* flood_cross_section_depth_buf = args.flood_cross_section_depth_buf;
    device float* flood_cross_section_area_buf = args.flood_cross_section_area_buf;
    device atomic_float* global_bifurcation_outflow_buf = args.global_bifurcation_outflow_buf;
    device float* total_storage_buf = args.total_storage_buf;
    device atomic_float* outgoing_storage_buf = args.outgoing_storage_buf;
    device float* water_surface_elevation_buf = args.water_surface_elevation_buf;
    device float* prot_wse_buf = args.prot_wse_buf;
    // Scalars packed in constant buffers
    const float gravity = *args.gravity;
    const float time_step = *args.time_step;
    const int num_catchments = *args.num_catchments;
    device float* sea_surface_elevation_buf = args.sea_surface_elevation_buf;
    device int* catchment_sea_level_idx_buf = args.catchment_sea_level_idx_buf;
    if ((int)idx >= num_catchments) return;

    // (1) Load
    int ds_idx = downstream_idx_buf[idx];
    bool is_mouth = (ds_idx == (int)idx);

    float riv_outflow = river_outflow_buf[idx];
    float riv_manning = river_manning_buf[idx];
    float riv_depth   = river_depth_buf[idx];
    float riv_width   = river_width_buf[idx];
    float riv_length  = river_length_buf[idx];
    float riv_height  = river_height_buf[idx];
    float riv_storage = river_storage_buf[idx];

    float fld_outflow = flood_outflow_buf[idx];
    float fld_manning = flood_manning_buf[idx];
    float fld_depth   = flood_depth_buf[idx];
    float prot_depth  = protected_depth_buf[idx];
    float catch_elev  = catchment_elevation_buf[idx];
    float ds_dist     = downstream_distance_buf[idx];
    float fld_storage = flood_storage_buf[idx];
    float prot_storage= protected_storage_buf[idx];

    float riv_xs_depth = river_cross_section_depth_buf[idx];
    float fld_xs_depth = flood_cross_section_depth_buf[idx];
    float fld_xs_area  = flood_cross_section_area_buf[idx];

    // (2) Water surface elevation
    float riv_elev = catch_elev - riv_height;
    float wse      = riv_depth + riv_elev;
    float prot_wse = min(catch_elev + prot_depth, wse);
    float total_sto = riv_storage + fld_storage + prot_storage;

    // Downstream WSE
    float riv_depth_ds  = river_depth_buf[ds_idx];
    float riv_height_ds = river_height_buf[ds_idx];
    float catch_elev_ds = catchment_elevation_buf[ds_idx];
    float riv_elev_ds   = catch_elev_ds - riv_height_ds;
    float wse_ds        = riv_depth_ds + riv_elev_ds;

    float wse_ds_eff = is_mouth ? catch_elev : wse_ds;
    if (kHasSeaLevel) {
        int sea_idx = catchment_sea_level_idx_buf[idx];
        if (sea_idx >= 0) wse_ds_eff = sea_surface_elevation_buf[sea_idx];
    }
    float max_wse = max(wse, wse_ds_eff);

    // (4) Slopes
    float riv_slope = (wse - wse_ds_eff) / ds_dist;
    float fld_slope = clamp(riv_slope, -0.005f, 0.005f);

    // (5) Cross-section depths + semi-implicit flow depth
    float upd_riv_xs = max_wse - riv_elev;
    float riv_flow_d  = max(sqrt(upd_riv_xs * riv_xs_depth), 1e-6f);

    float upd_fld_xs = max(max_wse - catch_elev, 0.0f);
    float fld_flow_d  = max(sqrt(upd_fld_xs * fld_xs_depth), 1e-6f);

    // (6) Flood area
    float upd_fld_xs_area = max(fld_storage / riv_length - fld_depth * riv_width, 0.0f);

    // (7) River outflow
    float riv_xs_area = upd_riv_xs * riv_width;
    bool riv_cond = (riv_flow_d > 1e-5f) && (riv_xs_area > 1e-5f);
    float unit_riv_out = riv_outflow / riv_width;
    float num_riv = riv_width * (unit_riv_out + gravity * time_step * riv_flow_d * riv_slope);
    float den_riv = 1.0f + gravity * time_step * (riv_manning * riv_manning)
                    * abs(unit_riv_out) * pow(riv_flow_d, -7.0f / 3.0f);
    float upd_riv_out = riv_cond ? (num_riv / den_riv) : 0.0f;

    // (8) Flood outflow
    bool fld_cond = (fld_flow_d > 1e-5f) && (upd_fld_xs_area > 1e-5f);
    float upd_fld_out = 0.0f;
    if (fld_cond) {
        float fld_impl_area = max(sqrt(upd_fld_xs_area * max(fld_xs_area, 1e-6f)), 1e-6f);
        float num_fld = fld_outflow + gravity * time_step * fld_impl_area * fld_slope;
        float den_fld = 1.0f + gravity * time_step * (fld_manning * fld_manning)
                        * abs(fld_outflow) * pow(fld_flow_d, -4.0f / 3.0f) / fld_impl_area;
        upd_fld_out = num_fld / den_fld;
    }

    // (9) Prevent opposite directions + negative-flow limiting
    if (upd_riv_out * upd_fld_out < 0.0f) upd_fld_out = 0.0f;
    if (upd_riv_out < 0.0f && !is_mouth) {
        float tot_neg = (-upd_riv_out - upd_fld_out) * time_step;
        float lim = min(0.05f * total_sto / tot_neg, 1.0f);
        upd_riv_out *= lim;
        upd_fld_out *= lim;
    }

    // (10) Store
    river_outflow_buf[idx]             = upd_riv_out;
    flood_outflow_buf[idx]             = upd_fld_out;
    water_surface_elevation_buf[idx]   = wse;
    prot_wse_buf[idx]                  = prot_wse;
    river_cross_section_depth_buf[idx] = upd_riv_xs;
    flood_cross_section_depth_buf[idx] = upd_fld_xs;
    flood_cross_section_area_buf[idx]  = upd_fld_xs_area;
    total_storage_buf[idx]             = total_sto;

    river_inflow_buf[idx] = 0.0f;
    flood_inflow_buf[idx] = 0.0f;
    if (kHasBifurcation) {
        atomic_store_explicit(&global_bifurcation_outflow_buf[idx], 0.0f, memory_order_relaxed);
    }

    // (11) Outgoing storage (fused)
    float pos_flow = max(upd_riv_out, 0.0f) + max(upd_fld_out, 0.0f);
    float neg_flow = min(upd_riv_out, 0.0f) + min(upd_fld_out, 0.0f);
    atomic_fetch_add_explicit(&outgoing_storage_buf[idx], pos_flow * time_step, memory_order_relaxed);
    if (!is_mouth) {
        atomic_fetch_add_explicit(&outgoing_storage_buf[ds_idx], -neg_flow * time_step, memory_order_relaxed);
    }
}


// =====================================================================
// Inflow kernel — limits outflow, accumulates inflow to downstream
// =====================================================================
struct compute_inflow_args {
    device int* downstream_idx_buf [[id(0)]];
    device float* river_outflow_buf [[id(1)]];
    device float* flood_outflow_buf [[id(2)]];
    device float* river_storage_buf [[id(3)]];
    device float* flood_storage_buf [[id(4)]];
    device float* outgoing_storage_buf [[id(5)]];
    device atomic_float* river_inflow_buf [[id(6)]];
    device atomic_float* flood_inflow_buf [[id(7)]];
    device float* limit_rate_buf [[id(8)]];
    device atomic_float* res_total_inflow_buf [[id(9)]];
    device int* is_reservoir_buf [[id(10)]];
    constant int* num_catchments [[id(11)]];
};

kernel void compute_inflow(
    constant compute_inflow_args& args [[buffer(0)]],
    uint idx [[thread_position_in_grid]]
)
{
    device int* downstream_idx_buf = args.downstream_idx_buf;
    device float* river_outflow_buf = args.river_outflow_buf;
    device float* flood_outflow_buf = args.flood_outflow_buf;
    device float* river_storage_buf = args.river_storage_buf;
    device float* flood_storage_buf = args.flood_storage_buf;
    device float* outgoing_storage_buf = args.outgoing_storage_buf;
    device atomic_float* river_inflow_buf = args.river_inflow_buf;
    device atomic_float* flood_inflow_buf = args.flood_inflow_buf;
    device float* limit_rate_buf = args.limit_rate_buf;
    device atomic_float* res_total_inflow_buf = args.res_total_inflow_buf;
    device int* is_reservoir_buf = args.is_reservoir_buf;
    const int num_catchments = *args.num_catchments;
    if ((int)idx >= num_catchments) return;

    float riv_out = river_outflow_buf[idx];
    float fld_out = flood_outflow_buf[idx];
    float out_sto = outgoing_storage_buf[idx];
    float rate_sto = river_storage_buf[idx] + flood_storage_buf[idx];

    // Local limit
    float limit = (out_sto > 1e-8f) ? min(rate_sto / out_sto, 1.0f) : 1.0f;

    // Downstream limit
    int ds_idx = downstream_idx_buf[idx];
    float out_sto_ds = outgoing_storage_buf[ds_idx];
    float rate_sto_ds = river_storage_buf[ds_idx] + flood_storage_buf[ds_idx];
    float limit_ds = (out_sto_ds > 1e-8f) ? min(rate_sto_ds / out_sto_ds, 1.0f) : 1.0f;

    // Apply limits
    float upd_riv = (riv_out >= 0.0f) ? riv_out * limit : riv_out * limit_ds;
    float upd_fld = (fld_out >= 0.0f) ? fld_out * limit : fld_out * limit_ds;

    river_outflow_buf[idx] = upd_riv;
    flood_outflow_buf[idx] = upd_fld;
    limit_rate_buf[idx]    = limit;

    // Accumulate inflows to downstream
    bool is_mouth = (ds_idx == (int)idx);
    if (!is_mouth) {
        atomic_fetch_add_explicit(&river_inflow_buf[ds_idx], upd_riv, memory_order_relaxed);
        atomic_fetch_add_explicit(&flood_inflow_buf[ds_idx], upd_fld, memory_order_relaxed);

        if (kHasReservoir) {
            if (is_reservoir_buf[ds_idx] != 0) {
                float total_out = upd_riv + upd_fld;
                atomic_fetch_add_explicit(&res_total_inflow_buf[ds_idx], total_out, memory_order_relaxed);
            }
        }
    }
}


// =====================================================================
// Batched outflow kernel — flat grid num_catchments * num_trials
// =====================================================================

// Packed scalar/flag parameters — avoids exceeding the 31-buffer limit.
// time_step is kept as a separate buffer because it changes every sub-step.
// All members are 4-byte aligned; total 12 bytes, no padding. Feature and
// parameter-layout flags are compile-time-specialized above.
struct OutflowBatchedConfig {
    float gravity;
    int   num_catchments;
    int   num_trials;
};

struct compute_outflow_batched_args {
    device int* downstream_idx_buf [[id(0)]];
    device float* river_inflow_buf [[id(1)]];
    device float* river_outflow_buf [[id(2)]];
    device float* river_manning_buf [[id(3)]];
    device float* river_depth_buf [[id(4)]];
    device float* river_width_buf [[id(5)]];
    device float* river_length_buf [[id(6)]];
    device float* river_height_buf [[id(7)]];
    device float* river_storage_buf [[id(8)]];
    device float* flood_inflow_buf [[id(9)]];
    device float* flood_outflow_buf [[id(10)]];
    device float* flood_manning_buf [[id(11)]];
    device float* flood_depth_buf [[id(12)]];
    device float* protected_depth_buf [[id(13)]];
    device float* catchment_elevation_buf [[id(14)]];
    device float* downstream_distance_buf [[id(15)]];
    device float* flood_storage_buf [[id(16)]];
    device float* protected_storage_buf [[id(17)]];
    device float* river_cross_section_depth_buf [[id(18)]];
    device float* flood_cross_section_depth_buf [[id(19)]];
    device float* flood_cross_section_area_buf [[id(20)]];
    device atomic_float* global_bifurcation_outflow_buf [[id(21)]];
    device float* total_storage_buf [[id(22)]];
    device atomic_float* outgoing_storage_buf [[id(23)]];
    device float* water_surface_elevation_buf [[id(24)]];
    device float* prot_wse_buf [[id(25)]];
    constant float* time_step [[id(26)]];
    constant OutflowBatchedConfig* config [[id(27)]];
    device float* sea_surface_elevation_buf [[id(28)]];
    device int* catchment_sea_level_idx_buf [[id(29)]];
    constant int* num_sea_level_boundaries [[id(30)]];
};

kernel void compute_outflow_batched(
    constant compute_outflow_batched_args& args [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
)
{
    device int* downstream_idx_buf = args.downstream_idx_buf;
    device float* river_inflow_buf = args.river_inflow_buf;
    device float* river_outflow_buf = args.river_outflow_buf;
    device float* river_manning_buf = args.river_manning_buf;
    device float* river_depth_buf = args.river_depth_buf;
    device float* river_width_buf = args.river_width_buf;
    device float* river_length_buf = args.river_length_buf;
    device float* river_height_buf = args.river_height_buf;
    device float* river_storage_buf = args.river_storage_buf;
    device float* flood_inflow_buf = args.flood_inflow_buf;
    device float* flood_outflow_buf = args.flood_outflow_buf;
    device float* flood_manning_buf = args.flood_manning_buf;
    device float* flood_depth_buf = args.flood_depth_buf;
    device float* protected_depth_buf = args.protected_depth_buf;
    device float* catchment_elevation_buf = args.catchment_elevation_buf;
    device float* downstream_distance_buf = args.downstream_distance_buf;
    device float* flood_storage_buf = args.flood_storage_buf;
    device float* protected_storage_buf = args.protected_storage_buf;
    device float* river_cross_section_depth_buf = args.river_cross_section_depth_buf;
    device float* flood_cross_section_depth_buf = args.flood_cross_section_depth_buf;
    device float* flood_cross_section_area_buf = args.flood_cross_section_area_buf;
    device atomic_float* global_bifurcation_outflow_buf = args.global_bifurcation_outflow_buf;
    device float* total_storage_buf = args.total_storage_buf;
    device atomic_float* outgoing_storage_buf = args.outgoing_storage_buf;
    device float* water_surface_elevation_buf = args.water_surface_elevation_buf;
    device float* prot_wse_buf = args.prot_wse_buf;
    const float time_step = *args.time_step;
    const OutflowBatchedConfig config = *args.config;
    device float* sea_surface_elevation_buf = args.sea_surface_elevation_buf;
    device int* catchment_sea_level_idx_buf = args.catchment_sea_level_idx_buf;
    const int num_sea_level_boundaries = *args.num_sea_level_boundaries;
    int num_catchments = config.num_catchments;
    int total = num_catchments * config.num_trials;
    if ((int)gid >= total) return;

    float gravity   = config.gravity;
    int ci = (int)gid % num_catchments;           // catchment index
    int to = ((int)gid / num_catchments) * num_catchments; // trial offset

    // Topology — never batched
    int ds_idx = downstream_idx_buf[ci];
    bool is_mouth = (ds_idx == ci);

    // State — always per-trial (use to + ci)
    float riv_outflow = river_outflow_buf[to + ci];
    float riv_depth   = river_depth_buf[to + ci];
    float riv_storage = river_storage_buf[to + ci];
    float fld_outflow = flood_outflow_buf[to + ci];
    float fld_depth   = flood_depth_buf[to + ci];
    float prot_depth  = protected_depth_buf[to + ci];
    float fld_storage = flood_storage_buf[to + ci];
    float prot_storage= protected_storage_buf[to + ci];
    float riv_xs_depth = river_cross_section_depth_buf[to + ci];
    float fld_xs_depth = flood_cross_section_depth_buf[to + ci];
    float fld_xs_area  = flood_cross_section_area_buf[to + ci];

    // Params — conditionally batched
    float riv_manning = river_manning_buf[kBatchedRiverManning ? (to + ci) : ci];
    float riv_width   = river_width_buf[kBatchedRiverWidth ? (to + ci) : ci];
    float riv_length  = river_length_buf[kBatchedRiverLength ? (to + ci) : ci];
    float riv_height  = river_height_buf[kBatchedRiverHeight ? (to + ci) : ci];
    float fld_manning = flood_manning_buf[kBatchedFloodManning ? (to + ci) : ci];
    float catch_elev  = catchment_elevation_buf[kBatchedCatchmentElevation ? (to + ci) : ci];
    float ds_dist     = downstream_distance_buf[ci]; // never batched in outflow

    // (2) Water surface elevation
    float riv_elev = catch_elev - riv_height;
    float wse      = riv_depth + riv_elev;
    float prot_wse = min(catch_elev + prot_depth, wse);
    float total_sto = riv_storage + fld_storage + prot_storage;

    // Downstream WSE
    int ds_global = to + ds_idx;
    float riv_depth_ds  = river_depth_buf[ds_global];
    float riv_height_ds = river_height_buf[kBatchedRiverHeight ? ds_global : ds_idx];
    float catch_elev_ds = catchment_elevation_buf[kBatchedCatchmentElevation ? ds_global : ds_idx];
    float riv_elev_ds   = catch_elev_ds - riv_height_ds;
    float wse_ds        = riv_depth_ds + riv_elev_ds;

    float wse_ds_eff = is_mouth ? catch_elev : wse_ds;
    if (kHasSeaLevel) {
        int sea_idx = catchment_sea_level_idx_buf[ci];
        if (sea_idx >= 0)
            wse_ds_eff = sea_surface_elevation_buf[
                ((int)gid / num_catchments) * num_sea_level_boundaries + sea_idx
            ];
    }
    float max_wse = max(wse, wse_ds_eff);

    // (4) Slopes
    float riv_slope = (wse - wse_ds_eff) / ds_dist;
    float fld_slope = clamp(riv_slope, -0.005f, 0.005f);

    // (5) Cross-section depths + semi-implicit flow depth
    float upd_riv_xs = max_wse - riv_elev;
    float riv_flow_d  = max(sqrt(upd_riv_xs * riv_xs_depth), 1e-6f);

    float upd_fld_xs = max(max_wse - catch_elev, 0.0f);
    float fld_flow_d  = max(sqrt(upd_fld_xs * fld_xs_depth), 1e-6f);

    // (6) Flood area
    float upd_fld_xs_area = max(fld_storage / riv_length - fld_depth * riv_width, 0.0f);

    // (7) River outflow
    float riv_xs_area = upd_riv_xs * riv_width;
    bool riv_cond = (riv_flow_d > 1e-5f) && (riv_xs_area > 1e-5f);
    float unit_riv_out = riv_outflow / riv_width;
    float num_riv = riv_width * (unit_riv_out + gravity * time_step * riv_flow_d * riv_slope);
    float den_riv = 1.0f + gravity * time_step * (riv_manning * riv_manning)
                    * abs(unit_riv_out) * pow(riv_flow_d, -7.0f / 3.0f);
    float upd_riv_out = riv_cond ? (num_riv / den_riv) : 0.0f;

    // (8) Flood outflow
    bool fld_cond = (fld_flow_d > 1e-5f) && (upd_fld_xs_area > 1e-5f);
    float upd_fld_out = 0.0f;
    if (fld_cond) {
        float fld_impl_area = max(sqrt(upd_fld_xs_area * max(fld_xs_area, 1e-6f)), 1e-6f);
        float num_fld = fld_outflow + gravity * time_step * fld_impl_area * fld_slope;
        float den_fld = 1.0f + gravity * time_step * (fld_manning * fld_manning)
                        * abs(fld_outflow) * pow(fld_flow_d, -4.0f / 3.0f) / fld_impl_area;
        upd_fld_out = num_fld / den_fld;
    }

    // (9) Prevent opposite directions + negative-flow limiting
    if (upd_riv_out * upd_fld_out < 0.0f) upd_fld_out = 0.0f;
    if (upd_riv_out < 0.0f && !is_mouth) {
        float tot_neg = (-upd_riv_out - upd_fld_out) * time_step;
        float lim = min(0.05f * total_sto / tot_neg, 1.0f);
        upd_riv_out *= lim;
        upd_fld_out *= lim;
    }

    // (10) Store
    int s = to + ci;
    river_outflow_buf[s]             = upd_riv_out;
    flood_outflow_buf[s]             = upd_fld_out;
    water_surface_elevation_buf[s]   = wse;
    prot_wse_buf[s]                  = prot_wse;
    river_cross_section_depth_buf[s] = upd_riv_xs;
    flood_cross_section_depth_buf[s] = upd_fld_xs;
    flood_cross_section_area_buf[s]  = upd_fld_xs_area;
    total_storage_buf[s]             = total_sto;

    river_inflow_buf[s] = 0.0f;
    flood_inflow_buf[s] = 0.0f;
    if (kHasBifurcation) {
        atomic_store_explicit(&global_bifurcation_outflow_buf[s], 0.0f, memory_order_relaxed);
    }

    // (11) Outgoing storage (fused)
    float pos_flow = max(upd_riv_out, 0.0f) + max(upd_fld_out, 0.0f);
    float neg_flow = min(upd_riv_out, 0.0f) + min(upd_fld_out, 0.0f);
    atomic_fetch_add_explicit(&outgoing_storage_buf[s], pos_flow * time_step, memory_order_relaxed);
    if (!is_mouth) {
        atomic_fetch_add_explicit(&outgoing_storage_buf[to + ds_idx], -neg_flow * time_step, memory_order_relaxed);
    }
}


// =====================================================================
// Batched inflow kernel — flat grid num_catchments * num_trials
// =====================================================================
struct compute_inflow_batched_args {
    device int* downstream_idx_buf [[id(0)]];
    device float* river_outflow_buf [[id(1)]];
    device float* flood_outflow_buf [[id(2)]];
    device float* river_storage_buf [[id(3)]];
    device float* flood_storage_buf [[id(4)]];
    device float* outgoing_storage_buf [[id(5)]];
    device atomic_float* river_inflow_buf [[id(6)]];
    device atomic_float* flood_inflow_buf [[id(7)]];
    device float* limit_rate_buf [[id(8)]];
    device atomic_float* res_total_inflow_buf [[id(9)]];
    device int* is_reservoir_buf [[id(10)]];
    constant int* num_catchments [[id(11)]];
    constant int* num_trials [[id(12)]];
};

kernel void compute_inflow_batched(
    constant compute_inflow_batched_args& args [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
)
{
    device int* downstream_idx_buf = args.downstream_idx_buf;
    device float* river_outflow_buf = args.river_outflow_buf;
    device float* flood_outflow_buf = args.flood_outflow_buf;
    device float* river_storage_buf = args.river_storage_buf;
    device float* flood_storage_buf = args.flood_storage_buf;
    device float* outgoing_storage_buf = args.outgoing_storage_buf;
    device atomic_float* river_inflow_buf = args.river_inflow_buf;
    device atomic_float* flood_inflow_buf = args.flood_inflow_buf;
    device float* limit_rate_buf = args.limit_rate_buf;
    device atomic_float* res_total_inflow_buf = args.res_total_inflow_buf;
    device int* is_reservoir_buf = args.is_reservoir_buf;
    const int num_catchments = *args.num_catchments;
    const int num_trials = *args.num_trials;
    int total = num_catchments * num_trials;
    if ((int)gid >= total) return;

    int ci = (int)gid % num_catchments;
    int to = ((int)gid / num_catchments) * num_catchments;
    int s = to + ci;

    float riv_out = river_outflow_buf[s];
    float fld_out = flood_outflow_buf[s];
    float out_sto = outgoing_storage_buf[s];
    float rate_sto = river_storage_buf[s] + flood_storage_buf[s];

    float limit = (out_sto > 1e-8f) ? min(rate_sto / out_sto, 1.0f) : 1.0f;

    // Downstream limit — topology never batched
    int ds_idx = downstream_idx_buf[ci];
    int ds_global = to + ds_idx;
    float out_sto_ds = outgoing_storage_buf[ds_global];
    float rate_sto_ds = river_storage_buf[ds_global] + flood_storage_buf[ds_global];
    float limit_ds = (out_sto_ds > 1e-8f) ? min(rate_sto_ds / out_sto_ds, 1.0f) : 1.0f;

    float upd_riv = (riv_out >= 0.0f) ? riv_out * limit : riv_out * limit_ds;
    float upd_fld = (fld_out >= 0.0f) ? fld_out * limit : fld_out * limit_ds;

    river_outflow_buf[s] = upd_riv;
    flood_outflow_buf[s] = upd_fld;
    limit_rate_buf[s]    = limit;

    bool is_mouth = (ds_idx == ci);
    if (!is_mouth) {
        atomic_fetch_add_explicit(&river_inflow_buf[ds_global], upd_riv, memory_order_relaxed);
        atomic_fetch_add_explicit(&flood_inflow_buf[ds_global], upd_fld, memory_order_relaxed);

        if (kHasReservoir) {
            if (is_reservoir_buf[ds_idx] != 0) {
                float total_out = upd_riv + upd_fld;
                atomic_fetch_add_explicit(&res_total_inflow_buf[ds_global], total_out, memory_order_relaxed);
            }
        }
    }
}
