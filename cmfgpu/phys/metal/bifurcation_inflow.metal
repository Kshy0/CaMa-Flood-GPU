// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0

#include <metal_stdlib>
using namespace metal;

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
