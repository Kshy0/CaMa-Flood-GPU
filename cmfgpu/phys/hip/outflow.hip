#include "hip/hip_runtime.h"
// CaMa-Flood-GPU — Outflow kernel + launcher
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0

#include "common.cuh"
#include <torch/extension.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

// =========================================================================
// Kernel: compute_outflow
//   HAS_BIFURCATION_CONST, HAS_RESERVOIR_CONST, CMF_GRAVITY,
//   CMF_MIN_KINEMATIC_SLOPE are compile-time constants.
// =========================================================================
template <bool HAS_BIFURCATION, bool HAS_RESERVOIR>
__global__ void compute_outflow_kernel(
    const int*       __restrict__ downstream_idx,
    storage_t*       __restrict__ river_inflow,
    float*           __restrict__ river_outflow,
    const float*     __restrict__ river_manning,
    float*           __restrict__ river_depth,
    const float*     __restrict__ river_width,
    const float*     __restrict__ river_length,
    const float*     __restrict__ river_height,
    const storage_t* __restrict__ river_storage,
    storage_t*       __restrict__ flood_inflow,
    float*           __restrict__ flood_outflow,
    const float*     __restrict__ flood_manning,
    float*           __restrict__ flood_depth,
    const float*     __restrict__ protected_depth,
    const float*     __restrict__ catchment_elevation,
    const float*     __restrict__ downstream_distance,
    const storage_t* __restrict__ flood_storage,
    const storage_t* __restrict__ protected_storage,
    float*           __restrict__ river_cs_depth,
    float*           __restrict__ flood_cs_depth,
    float*           __restrict__ flood_cs_area,
    storage_t*       __restrict__ global_bifurcation_outflow,
    storage_t*       __restrict__ total_storage_out,
    storage_t*       __restrict__ outgoing_storage,
    float*           __restrict__ water_surface_elevation_out,
    float*           __restrict__ protected_wse_out,
    const bool*      __restrict__ is_dam_upstream,
    const float* __restrict__ time_step_ptr,
    int   num_catchments
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_catchments) return;
    float time_step = *time_step_ptr;

    int ds = downstream_idx[i];
    bool is_mouth = (ds == i);

    float r_outflow  = river_outflow[i];
    float r_manning  = river_manning[i];
    float r_depth    = river_depth[i];
    float r_width    = river_width[i];
    float r_length   = river_length[i];
    float r_height   = river_height[i];
    float r_storage  = (float)river_storage[i];

    float f_outflow  = flood_outflow[i];
    float f_manning  = flood_manning[i];
    float f_depth    = flood_depth[i];
    float p_depth    = protected_depth[i];
    float elev       = catchment_elevation[i];
    float ds_dist    = downstream_distance[i];
    float f_storage  = (float)flood_storage[i];
    float p_storage  = (float)protected_storage[i];

    float r_cs_depth = river_cs_depth[i];
    float f_cs_depth = flood_cs_depth[i];
    float f_cs_area  = flood_cs_area[i];

    float r_elev = elev - r_height;
    float wse = r_depth + r_elev;
    float p_wse = fminf(elev + p_depth, wse);
    float total_sto = r_storage + f_storage + p_storage;

    float r_depth_ds = river_depth[ds];
    float r_height_ds = river_height[ds];
    float elev_ds = catchment_elevation[ds];
    float r_elev_ds = elev_ds - r_height_ds;
    float wse_ds = r_depth_ds + r_elev_ds;

    float max_wse = fmaxf(wse, wse_ds);
    float wse_ds_eff = is_mouth ? elev : wse_ds;

    float r_slope = (wse - wse_ds_eff) / ds_dist;
    float f_slope = fminf(fmaxf(r_slope, -0.005f), 0.005f);

    float upd_r_cs = max_wse - r_elev;
    float r_semi = fmaxf(sqrtf(upd_r_cs * r_cs_depth), 1e-6f);
    float upd_f_cs = fmaxf(max_wse - elev, 0.0f);
    float f_semi = fmaxf(sqrtf(upd_f_cs * f_cs_depth), 1e-6f);

    float upd_f_area = fmaxf(f_storage / r_length - f_depth * r_width, 0.0f);
    float f_impl_area = fmaxf(sqrtf(upd_f_area * fmaxf(f_cs_area, 1e-6f)), 1e-6f);

    float r_cross = upd_r_cs * r_width;
    bool r_cond = (r_semi > 1e-5f) && (r_cross > 1e-5f);
    float unit_r = r_outflow / r_width;
    float num_r = r_width * (unit_r + CMF_GRAVITY * time_step * r_semi * r_slope);
    float den_r = 1.0f + CMF_GRAVITY * time_step * (r_manning * r_manning) * fabsf(unit_r) * powf(r_semi, -7.0f/3.0f);
    float upd_r_outflow = r_cond ? (num_r / den_r) : 0.0f;

    bool f_cond = (f_semi > 1e-5f) && (upd_f_area > 1e-5f);
    float num_f = f_outflow + CMF_GRAVITY * time_step * f_impl_area * f_slope;
    float den_f = 1.0f + CMF_GRAVITY * time_step * (f_manning * f_manning) * fabsf(f_outflow) * powf(f_semi, -4.0f/3.0f) / f_impl_area;
    float upd_f_outflow = f_cond ? (num_f / den_f) : 0.0f;

    if (upd_r_outflow * upd_f_outflow < 0.0f)
        upd_f_outflow = 0.0f;

    if (upd_r_outflow < 0.0f && !is_mouth) {
        float neg_total = (-upd_r_outflow - upd_f_outflow) * time_step;
        float lr = fminf(0.05f * total_sto / neg_total, 1.0f);
        upd_r_outflow *= lr;
        upd_f_outflow *= lr;
    }

    if (HAS_RESERVOIR && is_dam_upstream != nullptr && is_dam_upstream[i]) {
        float bed_slope = fmaxf((elev - elev_ds) / ds_dist, CMF_MIN_KINEMATIC_SLOPE);
        float kin_r_vel = (1.0f / r_manning) * sqrtf(bed_slope) * powf(r_depth, 2.0f/3.0f);
        float kin_r = fmaxf(fminf(r_width * r_depth * kin_r_vel, r_storage / time_step), 0.0f);
        float bed_slope_f = fminf(bed_slope, 0.005f);
        float kin_f_vel = (1.0f / f_manning) * sqrtf(bed_slope_f) * powf(f_depth, 2.0f/3.0f);
        float kin_f_area = fmaxf(f_storage / r_length - f_depth * r_width, 0.0f);
        float kin_f = fmaxf(fminf(kin_f_area * kin_f_vel, f_storage / time_step), 0.0f);
        upd_r_outflow = kin_r;
        upd_f_outflow = kin_f;
    }

    river_outflow[i] = upd_r_outflow;
    flood_outflow[i] = upd_f_outflow;
    water_surface_elevation_out[i] = wse;
    protected_wse_out[i] = p_wse;
    river_cs_depth[i] = upd_r_cs;
    flood_cs_depth[i] = upd_f_cs;
    flood_cs_area[i] = upd_f_area;
    total_storage_out[i] = STO_CAST(total_sto);

    river_inflow[i] = STO_ZERO;
    flood_inflow[i] = STO_ZERO;
    if (HAS_BIFURCATION) {
        global_bifurcation_outflow[i] = STO_ZERO;
    }

    float pos = fmaxf(upd_r_outflow, 0.0f) + fmaxf(upd_f_outflow, 0.0f);
    float neg = fminf(upd_r_outflow, 0.0f) + fminf(upd_f_outflow, 0.0f);
    atomicAdd(&outgoing_storage[i], STO_CAST(pos * time_step));
    if (!is_mouth) {
        atomicAdd(&outgoing_storage[ds], STO_CAST(-neg * time_step));
    }
}

// =========================================================================
// Launcher — single template instantiation via compile-time macros
// =========================================================================
void launch_outflow(
    torch::Tensor downstream_idx,
    torch::Tensor river_inflow, torch::Tensor river_outflow,
    torch::Tensor river_manning, torch::Tensor river_depth,
    torch::Tensor river_width, torch::Tensor river_length,
    torch::Tensor river_height, torch::Tensor river_storage,
    torch::Tensor flood_inflow, torch::Tensor flood_outflow,
    torch::Tensor flood_manning, torch::Tensor flood_depth,
    torch::Tensor protected_depth, torch::Tensor catchment_elevation,
    torch::Tensor downstream_distance, torch::Tensor flood_storage,
    torch::Tensor protected_storage,
    torch::Tensor river_cs_depth, torch::Tensor flood_cs_depth,
    torch::Tensor flood_cs_area,
    torch::Tensor global_bif_outflow, torch::Tensor total_storage,
    torch::Tensor outgoing_storage, torch::Tensor wse_out,
    torch::Tensor p_wse_out,
    torch::Tensor is_dam_upstream,
    torch::Tensor time_step_tensor,
    int num_catchments
) {
    const int bs = CMF_BLOCK_SIZE;
    const int grid = cdiv(num_catchments, bs);
    const auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    const bool* dam_ptr = (is_dam_upstream.defined() && is_dam_upstream.numel() > 0)
        ? is_dam_upstream.data_ptr<bool>() : nullptr;

   hipLaunchKernelGGL(( compute_outflow_kernel<(bool)HAS_BIFURCATION_CONST, (bool)HAS_RESERVOIR_CONST>)
        , dim3(grid), dim3(bs), 0, stream, 
            downstream_idx.data_ptr<int>(),
            river_inflow.data_ptr<storage_t>(), river_outflow.data_ptr<float>(),
            river_manning.data_ptr<float>(), river_depth.data_ptr<float>(),
            river_width.data_ptr<float>(), river_length.data_ptr<float>(),
            river_height.data_ptr<float>(), river_storage.data_ptr<storage_t>(),
            flood_inflow.data_ptr<storage_t>(), flood_outflow.data_ptr<float>(),
            flood_manning.data_ptr<float>(), flood_depth.data_ptr<float>(),
            protected_depth.data_ptr<float>(), catchment_elevation.data_ptr<float>(),
            downstream_distance.data_ptr<float>(), flood_storage.data_ptr<storage_t>(),
            protected_storage.data_ptr<storage_t>(),
            river_cs_depth.data_ptr<float>(), flood_cs_depth.data_ptr<float>(),
            flood_cs_area.data_ptr<float>(),
            global_bif_outflow.data_ptr<storage_t>(), total_storage.data_ptr<storage_t>(),
            outgoing_storage.data_ptr<storage_t>(), wse_out.data_ptr<float>(),
            p_wse_out.data_ptr<float>(),
            dam_ptr,
            time_step_tensor.data_ptr<float>(), num_catchments
        );
}
