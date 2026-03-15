// CaMa-Flood-GPU — Inflow kernel + launcher
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0

#include "common.cuh"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// =========================================================================
// Kernel: compute_inflow
//   HAS_RESERVOIR_CONST is a compile-time constant.
// =========================================================================
template <bool HAS_RESERVOIR>
__global__ void compute_inflow_kernel(
    const int*       __restrict__ downstream_idx,
    float*           __restrict__ river_outflow,
    float*           __restrict__ flood_outflow,
    const storage_t* __restrict__ river_storage,
    const storage_t* __restrict__ flood_storage,
    storage_t*       __restrict__ outgoing_storage,
    storage_t*       __restrict__ river_inflow,
    storage_t*       __restrict__ flood_inflow,
    float*           __restrict__ limit_rate_out,
    storage_t*       __restrict__ reservoir_total_inflow,
    const bool*      __restrict__ is_reservoir,
    int num_catchments
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_catchments) return;

    float r_out = river_outflow[i];
    float f_out = flood_outflow[i];
    float out_sto = (float)outgoing_storage[i];
    float rate_sto = (float)river_storage[i] + (float)flood_storage[i];

    float lr = (out_sto > 1e-8f) ? fminf(rate_sto / out_sto, 1.0f) : 1.0f;

    int ds = downstream_idx[i];
    float out_sto_ds = (float)outgoing_storage[ds];
    float rate_sto_ds = (float)river_storage[ds] + (float)flood_storage[ds];
    float lr_ds = (out_sto_ds > 1e-8f) ? fminf(rate_sto_ds / out_sto_ds, 1.0f) : 1.0f;

    float upd_r = (r_out >= 0.0f) ? r_out * lr : r_out * lr_ds;
    float upd_f = (f_out >= 0.0f) ? f_out * lr : f_out * lr_ds;

    river_outflow[i] = upd_r;
    flood_outflow[i] = upd_f;
    limit_rate_out[i] = lr;

    bool is_mouth = (ds == i);
    if (!is_mouth) {
        atomicAdd(&river_inflow[ds], STO_CAST(upd_r));
        atomicAdd(&flood_inflow[ds], STO_CAST(upd_f));
        if (HAS_RESERVOIR && is_reservoir != nullptr && is_reservoir[ds]) {
            atomicAdd(&reservoir_total_inflow[ds], STO_CAST(upd_r + upd_f));
        }
    }
}

// =========================================================================
// Launcher — single template via HAS_RESERVOIR_CONST macro
// =========================================================================
void launch_inflow(
    torch::Tensor downstream_idx,
    torch::Tensor river_outflow, torch::Tensor flood_outflow,
    torch::Tensor river_storage, torch::Tensor flood_storage,
    torch::Tensor outgoing_storage,
    torch::Tensor river_inflow, torch::Tensor flood_inflow,
    torch::Tensor limit_rate,
    torch::Tensor reservoir_total_inflow,
    torch::Tensor is_reservoir,
    int num_catchments
) {
    const int bs = CMF_BLOCK_SIZE;
    const int grid = cdiv(num_catchments, bs);
    const auto stream = at::cuda::getCurrentCUDAStream();

    storage_t* res_ptr = (HAS_RESERVOIR_CONST && reservoir_total_inflow.defined()
                          && reservoir_total_inflow.numel() > 0)
        ? reservoir_total_inflow.data_ptr<storage_t>() : nullptr;
    const bool* res_flag_ptr = (HAS_RESERVOIR_CONST && is_reservoir.defined()
                                && is_reservoir.numel() > 0)
        ? is_reservoir.data_ptr<bool>() : nullptr;

    compute_inflow_kernel<(bool)HAS_RESERVOIR_CONST><<<grid, bs, 0, stream>>>(
        downstream_idx.data_ptr<int>(),
        river_outflow.data_ptr<float>(), flood_outflow.data_ptr<float>(),
        river_storage.data_ptr<storage_t>(), flood_storage.data_ptr<storage_t>(),
        outgoing_storage.data_ptr<storage_t>(),
        river_inflow.data_ptr<storage_t>(), flood_inflow.data_ptr<storage_t>(),
        limit_rate.data_ptr<float>(),
        res_ptr, res_flag_ptr, num_catchments);
}
