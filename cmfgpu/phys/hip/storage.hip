#include "hip/hip_runtime.h"
// CaMa-Flood-GPU — Flood stage kernel + launcher
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0

#include "common.cuh"
#include <torch/extension.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

// =========================================================================
// Kernel: compute_flood_stage
//   NUM_FLOOD_LEVELS, HAS_BIFURCATION_CONST are compile-time constants.
// =========================================================================
__global__ void compute_flood_stage_kernel(
    storage_t*       __restrict__ river_inflow,
    storage_t*       __restrict__ flood_inflow,
    const float*     __restrict__ river_outflow,
    const float*     __restrict__ flood_outflow,
    const storage_t* __restrict__ global_bifurcation_outflow,
    const float*     __restrict__ runoff,
    const float*     __restrict__ time_step_ptr,
    storage_t*       __restrict__ outgoing_storage,
    storage_t*       __restrict__ river_storage,
    storage_t*       __restrict__ flood_storage,
    storage_t*       __restrict__ protected_storage,
    float*           __restrict__ river_depth,
    float*           __restrict__ flood_depth,
    float*           __restrict__ protected_depth,
    float*           __restrict__ flood_fraction,
    const float*     __restrict__ river_height,
    const float*     __restrict__ flood_depth_table,
    const float*     __restrict__ catchment_area,
    const float*     __restrict__ river_width,
    const float*     __restrict__ river_length,
    int num_catchments
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_catchments) return;
    float time_step = *time_step_ptr;

    float r_inflow = (float)river_inflow[i];
    float f_inflow = (float)flood_inflow[i];
    float r_out = river_outflow[i];
    float f_out = flood_outflow[i];
#if HAS_BIFURCATION_CONST
    float bif_out = (float)global_bifurcation_outflow[i];
#else
    float bif_out = 0.0f;
#endif
    float roff = runoff[i];

    storage_t r_sto = river_storage[i] + STO_CAST((r_inflow - r_out) * time_step);
    storage_t f_sto_upd = flood_storage[i] + (r_sto < STO_ZERO ? r_sto : STO_ZERO)
                          + STO_CAST((f_inflow - f_out - bif_out) * time_step);
    if (r_sto < STO_ZERO) r_sto = STO_ZERO;
    if (f_sto_upd < STO_ZERO) {
        r_sto = sto_max(r_sto + f_sto_upd, STO_ZERO);
        f_sto_upd = STO_ZERO;
    }
    storage_t p_sto = protected_storage[i];
    float total_sto = (float)sto_max(r_sto + f_sto_upd + p_sto + STO_CAST(roff * time_step), STO_ZERO);

    float r_h = river_height[i];
    float area = catchment_area[i];
    float r_w = river_width[i];
    float r_l = river_length[i];

    float r_max_sto = r_l * r_w * r_h;
    float cw = area / r_l;
    constexpr float inv_nfl = 1.0f / NUM_FLOOD_LEVELS;
    float w_inc = cw * inv_nfl;

    int level = (total_sto > r_max_sto) ? 0 : -1;
    float S_accum = r_max_sto;
    float prev_H = 0.0f, prev_W = r_w;
    float prev_total_sto = r_max_sto;
    float prev_flood_depth = 0.0f;
    float next_flood_depth = 0.0f;

    #pragma unroll
    for (int fl = 0; fl < NUM_FLOOD_LEVELS; fl++) {
        float H_curr = flood_depth_table[i * NUM_FLOOD_LEVELS + fl];
        float W_curr = r_w + (fl + 1) * w_inc;
        float dS = r_l * 0.5f * (prev_W + W_curr) * (H_curr - prev_H);
        float S_curr = S_accum + dS;

        if (level == fl) next_flood_depth = H_curr;

        if (total_sto > S_curr) {
            level++;
            prev_total_sto = S_curr;
            prev_flood_depth = H_curr;
        }
        S_accum = S_curr;
        prev_H = H_curr;
        prev_W = W_curr;
    }

    bool no_flood = (level < 0);
    if (level < 0) level = 0;

    float prev_total_w = r_w + level * w_inc;
    float flood_grad = (level == NUM_FLOOD_LEVELS) ? 0.0f :
        (next_flood_depth - prev_flood_depth) / w_inc;

    float fld_depth, fld_frac;
    if (no_flood) {
        fld_depth = 0.0f;
        fld_frac = 0.0f;
    } else if (level == NUM_FLOOD_LEVELS) {
        fld_depth = prev_flood_depth + (total_sto - prev_total_sto) / (prev_total_w * r_l);
        fld_frac = 1.0f;
    } else {
        float diff_w = sqrtf(prev_total_w * prev_total_w +
            2.0f * (total_sto - prev_total_sto) / (flood_grad * r_l)) - prev_total_w;
        fld_depth = prev_flood_depth + diff_w * flood_grad;
        fld_frac = fminf(fmaxf((prev_total_w + diff_w - r_w) * r_l / area, 0.0f), 1.0f);
    }

    float r_sto_final = no_flood ? total_sto :
        fminf(r_max_sto + r_l * r_w * fld_depth, total_sto);
    float r_d = r_sto_final / (r_l * r_w);
    float f_sto_final = fmaxf(total_sto - r_sto_final, 0.0f);

    outgoing_storage[i] = STO_ZERO;
    river_storage[i] = STO_CAST(r_sto_final);
    flood_storage[i] = STO_CAST(f_sto_final);
    protected_storage[i] = STO_ZERO;
    river_depth[i] = r_d;
    flood_depth[i] = fld_depth;
    protected_depth[i] = fld_depth;
    flood_fraction[i] = fld_frac;
}

// =========================================================================
// Launcher
// =========================================================================
void launch_flood_stage(
    torch::Tensor river_inflow, torch::Tensor flood_inflow,
    torch::Tensor river_outflow, torch::Tensor flood_outflow,
    torch::Tensor global_bif_outflow,
    torch::Tensor runoff, torch::Tensor time_step_tensor,
    torch::Tensor outgoing_storage,
    torch::Tensor river_storage, torch::Tensor flood_storage,
    torch::Tensor protected_storage,
    torch::Tensor river_depth, torch::Tensor flood_depth,
    torch::Tensor protected_depth, torch::Tensor flood_fraction,
    torch::Tensor river_height, torch::Tensor flood_depth_table,
    torch::Tensor catchment_area, torch::Tensor river_width,
    torch::Tensor river_length,
    int num_catchments
) {
    const int bs = CMF_BLOCK_SIZE;
    const int grid = cdiv(num_catchments, bs);
    const auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
   hipLaunchKernelGGL(( compute_flood_stage_kernel), dim3(grid), dim3(bs), 0, stream, 
        river_inflow.data_ptr<storage_t>(), flood_inflow.data_ptr<storage_t>(),
        river_outflow.data_ptr<float>(), flood_outflow.data_ptr<float>(),
        global_bif_outflow.data_ptr<storage_t>(),
        runoff.data_ptr<float>(), time_step_tensor.data_ptr<float>(),
        outgoing_storage.data_ptr<storage_t>(),
        river_storage.data_ptr<storage_t>(), flood_storage.data_ptr<storage_t>(),
        protected_storage.data_ptr<storage_t>(),
        river_depth.data_ptr<float>(), flood_depth.data_ptr<float>(),
        protected_depth.data_ptr<float>(), flood_fraction.data_ptr<float>(),
        river_height.data_ptr<float>(), flood_depth_table.data_ptr<float>(),
        catchment_area.data_ptr<float>(), river_width.data_ptr<float>(),
        river_length.data_ptr<float>(),
        num_catchments);
}
