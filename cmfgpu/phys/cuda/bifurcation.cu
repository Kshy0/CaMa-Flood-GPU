// CaMa-Flood-GPU — Bifurcation outflow & inflow kernels + launchers
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0

#include "common.cuh"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// =========================================================================
// Kernel: compute_bifurcation_outflow
//   NUM_BIF_LEVELS, CMF_GRAVITY are compile-time constants.
// =========================================================================
__global__ void compute_bifurcation_outflow_kernel(
    const int*       __restrict__ bif_catchment_idx,
    const int*       __restrict__ bif_downstream_idx,
    const float*     __restrict__ bif_manning,
    float*           __restrict__ bif_outflow,
    const float*     __restrict__ bif_width,
    const float*     __restrict__ bif_length,
    const float*     __restrict__ bif_elevation,
    float*           __restrict__ bif_cs_depth,
    const float*     __restrict__ water_surface_elevation,
    const storage_t* __restrict__ total_storage,
    storage_t*       __restrict__ outgoing_storage,
    const float* __restrict__ time_step_ptr,
    int num_paths
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_paths) return;
    float time_step = *time_step_ptr;

    int c_idx = bif_catchment_idx[p];
    int d_idx = bif_downstream_idx[p];
    float b_length = bif_length[p];

    float wse_c = water_surface_elevation[c_idx];
    float wse_d = water_surface_elevation[d_idx];
    float max_wse = fmaxf(wse_c, wse_d);
    float b_slope = fminf(fmaxf((wse_c - wse_d) / b_length, -0.005f), 0.005f);
    float sto_c = (float)total_storage[c_idx];
    float sto_d = (float)total_storage[d_idx];

    float sum_out = 0.0f;
    #pragma unroll
    for (int lv = 0; lv < NUM_BIF_LEVELS; lv++) {
        int li = p * NUM_BIF_LEVELS + lv;
        float manning = bif_manning[li];
        float cs_d = bif_cs_depth[li];
        float elev = bif_elevation[li];
        float upd_cs = fmaxf(max_wse - elev, 0.0f);
        float semi = fmaxf(sqrtf(upd_cs * cs_d), sqrtf(upd_cs * 0.01f));
        float width = bif_width[li];
        float outf = bif_outflow[li];
        float unit_out = outf / width;
        float num = width * (unit_out + CMF_GRAVITY * time_step * semi * b_slope);
        float den = 1.0f + CMF_GRAVITY * time_step * (manning * manning)
                    * fabsf(unit_out) * powf(semi, -7.0f / 3.0f);
        float upd = (semi > 1e-5f) ? (num / den) : 0.0f;
        sum_out += upd;
        bif_cs_depth[li] = upd_cs;
        bif_outflow[li] = upd;
    }

    float lr = fminf(0.05f * fminf(sto_c, sto_d)
                      / (fabsf(sum_out) * time_step), 1.0f);
    sum_out *= lr;
    #pragma unroll
    for (int lv = 0; lv < NUM_BIF_LEVELS; lv++) {
        bif_outflow[p * NUM_BIF_LEVELS + lv] *= lr;
    }

    float pos = fmaxf(sum_out, 0.0f);
    float neg = fminf(sum_out, 0.0f);
    atomicAdd(&outgoing_storage[c_idx], STO_CAST(pos * time_step));
    atomicAdd(&outgoing_storage[d_idx], STO_CAST(-neg * time_step));
}

// =========================================================================
// Launcher: bifurcation_outflow
// =========================================================================
void launch_bifurcation_outflow(
    torch::Tensor bif_catchment_idx, torch::Tensor bif_downstream_idx,
    torch::Tensor bif_manning, torch::Tensor bif_outflow,
    torch::Tensor bif_width, torch::Tensor bif_length,
    torch::Tensor bif_elevation, torch::Tensor bif_cs_depth,
    torch::Tensor wse, torch::Tensor total_storage,
    torch::Tensor outgoing_storage,
    torch::Tensor time_step_tensor, int num_paths
) {
    const int bs = CMF_BLOCK_SIZE;
    const int grid = cdiv(num_paths, bs);
    const auto stream = at::cuda::getCurrentCUDAStream();
    compute_bifurcation_outflow_kernel<<<grid, bs, 0, stream>>>(
        bif_catchment_idx.data_ptr<int>(), bif_downstream_idx.data_ptr<int>(),
        bif_manning.data_ptr<float>(), bif_outflow.data_ptr<float>(),
        bif_width.data_ptr<float>(), bif_length.data_ptr<float>(),
        bif_elevation.data_ptr<float>(), bif_cs_depth.data_ptr<float>(),
        wse.data_ptr<float>(), total_storage.data_ptr<storage_t>(),
        outgoing_storage.data_ptr<storage_t>(),
        time_step_tensor.data_ptr<float>(), num_paths);
}

// =========================================================================
// Kernel: compute_bifurcation_inflow
//   NUM_BIF_LEVELS is a compile-time constant.
// =========================================================================
__global__ void compute_bifurcation_inflow_kernel(
    const int*   __restrict__ bif_catchment_idx,
    const int*   __restrict__ bif_downstream_idx,
    const float* __restrict__ limit_rate,
    float*       __restrict__ bif_outflow,
    storage_t*   __restrict__ global_bif_outflow,
    int num_paths
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_paths) return;

    int c_idx = bif_catchment_idx[p];
    int d_idx = bif_downstream_idx[p];
    float lr_c = limit_rate[c_idx];
    float lr_d = limit_rate[d_idx];

    float sum = 0.0f;
    #pragma unroll
    for (int lv = 0; lv < NUM_BIF_LEVELS; lv++) {
        int li = p * NUM_BIF_LEVELS + lv;
        float out = bif_outflow[li];
        float upd = (out >= 0.0f) ? out * lr_c : out * lr_d;
        sum += upd;
        bif_outflow[li] = upd;
    }

    atomicAdd(&global_bif_outflow[c_idx], STO_CAST(sum));
    atomicAdd(&global_bif_outflow[d_idx], STO_CAST(-sum));
}

// =========================================================================
// Launcher: bifurcation_inflow
// =========================================================================
void launch_bifurcation_inflow(
    torch::Tensor bif_catchment_idx, torch::Tensor bif_downstream_idx,
    torch::Tensor limit_rate, torch::Tensor bif_outflow,
    torch::Tensor global_bif_outflow,
    int num_paths
) {
    const int bs = CMF_BLOCK_SIZE;
    const int grid = cdiv(num_paths, bs);
    const auto stream = at::cuda::getCurrentCUDAStream();
    compute_bifurcation_inflow_kernel<<<grid, bs, 0, stream>>>(
        bif_catchment_idx.data_ptr<int>(), bif_downstream_idx.data_ptr<int>(),
        limit_rate.data_ptr<float>(), bif_outflow.data_ptr<float>(),
        global_bif_outflow.data_ptr<storage_t>(),
        num_paths);
}
