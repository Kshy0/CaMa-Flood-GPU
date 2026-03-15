// CaMa-Flood-GPU — Reservoir outflow kernel + launcher
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0

#include "common.cuh"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// =========================================================================
// Kernel: compute_reservoir_outflow
// =========================================================================
__global__ void compute_reservoir_outflow_kernel(
    const int*       __restrict__ res_catchment_idx,
    const int*       __restrict__ downstream_idx,
    storage_t*       __restrict__ reservoir_total_inflow,
    float*           __restrict__ river_outflow,
    float*           __restrict__ flood_outflow,
    const storage_t* __restrict__ river_storage,
    const storage_t* __restrict__ flood_storage,
    const float*     __restrict__ conservation_volume,
    const float*     __restrict__ emergency_volume,
    const float*     __restrict__ adjustment_volume,
    const float*     __restrict__ normal_outflow,
    const float*     __restrict__ adjustment_outflow,
    const float*     __restrict__ flood_control_outflow,
    const float*     __restrict__ runoff,
    storage_t*       __restrict__ total_storage_out,
    storage_t*       __restrict__ outgoing_storage,
    const float* __restrict__ time_step_ptr, int num_reservoirs
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_reservoirs) return;
    float time_step = *time_step_ptr;

    int ci = res_catchment_idx[r];
    int ds = downstream_idx[ci];
    bool is_mouth = (ds == ci);

    // Undo old outflow contribution
    float old_r = river_outflow[ci];
    float old_f = flood_outflow[ci];
    float old_pos = fmaxf(old_r, 0.0f) + fmaxf(old_f, 0.0f);
    float old_neg = fminf(old_r, 0.0f) + fminf(old_f, 0.0f);
    atomicAdd(&outgoing_storage[ci], STO_CAST(-(old_pos * time_step)));
    if (!is_mouth) atomicAdd(&outgoing_storage[ds], STO_CAST(old_neg * time_step));

    // Downcast storage for physics computation
    float r_sto = (float)river_storage[ci];
    float f_sto = (float)flood_storage[ci];
    float total_sto = r_sto + f_sto;
    float total_inf = (float)reservoir_total_inflow[ci];
    reservoir_total_inflow[ci] = STO_ZERO;

    float roff = runoff[ci];
    float res_inflow = total_inf + roff;

    float cons_vol = conservation_volume[r];
    float emerg_vol = emergency_volume[r];
    float adj_vol = adjustment_volume[r];
    float norm_out = normal_outflow[r];
    float adj_out = adjustment_outflow[r];
    float fld_ctrl = flood_control_outflow[r];

    float res_out = 0.0f;
    if (total_sto <= cons_vol) {
        res_out = norm_out * sqrtf(total_sto / cons_vol);
    } else if (total_sto <= adj_vol) {
        float frac = (total_sto - cons_vol) / (adj_vol - cons_vol);
        res_out = norm_out + expf(3.0f * logf(frac)) * (adj_out - norm_out);
    } else if (total_sto <= emerg_vol) {
        bool flood = res_inflow >= fld_ctrl;
        float frac3 = (total_sto - adj_vol) / (emerg_vol - adj_vol);
        if (flood) {
            float out_flood = norm_out
                + ((total_sto - cons_vol) / (emerg_vol - cons_vol))
                  * (res_inflow - norm_out);
            float out_tmp = adj_out
                + expf(0.1f * logf(frac3)) * (fld_ctrl - adj_out);
            res_out = fmaxf(out_flood, out_tmp);
        } else {
            res_out = adj_out
                + expf(0.1f * logf(frac3)) * (fld_ctrl - adj_out);
        }
    } else {
        res_out = (res_inflow >= fld_ctrl) ? res_inflow : fld_ctrl;
    }

    res_out = fmaxf(fminf(res_out, total_sto / time_step), 0.0f);

    river_outflow[ci] = res_out;
    flood_outflow[ci] = 0.0f;
    total_storage_out[ci] = STO_CAST(total_sto);

    // Apply new outflow contribution
    float new_pos = fmaxf(res_out, 0.0f);
    atomicAdd(&outgoing_storage[ci], STO_CAST(new_pos * time_step));
    if (!is_mouth) {
        float new_neg = fminf(res_out, 0.0f);
        atomicAdd(&outgoing_storage[ds], STO_CAST(-(new_neg * time_step)));
    }
}

// =========================================================================
// Launcher
// =========================================================================
void launch_reservoir_outflow(
    torch::Tensor res_catchment_idx, torch::Tensor downstream_idx,
    torch::Tensor reservoir_total_inflow,
    torch::Tensor river_outflow, torch::Tensor flood_outflow,
    torch::Tensor river_storage, torch::Tensor flood_storage,
    torch::Tensor conservation_volume, torch::Tensor emergency_volume,
    torch::Tensor adjustment_volume, torch::Tensor normal_outflow,
    torch::Tensor adjustment_outflow, torch::Tensor flood_control_outflow,
    torch::Tensor runoff,
    torch::Tensor total_storage, torch::Tensor outgoing_storage,
    torch::Tensor time_step_tensor, int num_reservoirs
) {
    const int bs = CMF_BLOCK_SIZE;
    const int grid = cdiv(num_reservoirs, bs);
    const auto stream = at::cuda::getCurrentCUDAStream();
    compute_reservoir_outflow_kernel<<<grid, bs, 0, stream>>>(
        res_catchment_idx.data_ptr<int>(), downstream_idx.data_ptr<int>(),
        reservoir_total_inflow.data_ptr<storage_t>(),
        river_outflow.data_ptr<float>(), flood_outflow.data_ptr<float>(),
        river_storage.data_ptr<storage_t>(), flood_storage.data_ptr<storage_t>(),
        conservation_volume.data_ptr<float>(), emergency_volume.data_ptr<float>(),
        adjustment_volume.data_ptr<float>(), normal_outflow.data_ptr<float>(),
        adjustment_outflow.data_ptr<float>(), flood_control_outflow.data_ptr<float>(),
        runoff.data_ptr<float>(),
        total_storage.data_ptr<storage_t>(), outgoing_storage.data_ptr<storage_t>(),
        time_step_tensor.data_ptr<float>(), num_reservoirs);
}
