// CaMa-Flood-GPU — Levee stage + levee bifurcation outflow kernels + launchers
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0

#include "common.cuh"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// =========================================================================
// Kernel: compute_levee_stage
//   NUM_FLOOD_LEVELS is a compile-time constant.
// =========================================================================
__global__ void compute_levee_stage_kernel(
    const int*       __restrict__ levee_catchment_idx,
    storage_t*       __restrict__ river_storage,
    storage_t*       __restrict__ flood_storage,
    storage_t*       __restrict__ protected_storage,
    float*           __restrict__ river_depth,
    float*           __restrict__ flood_depth,
    float*           __restrict__ protected_depth,
    const float*     __restrict__ river_height,
    const float*     __restrict__ flood_depth_table,
    const float*     __restrict__ catchment_area,
    const float*     __restrict__ river_width,
    const float*     __restrict__ river_length,
    const float*     __restrict__ levee_base_height,
    const float*     __restrict__ levee_crown_height,
    const float*     __restrict__ levee_fraction,
    float*           __restrict__ flood_fraction,
    int num_levees
) {
    int lid = blockIdx.x * blockDim.x + threadIdx.x;
    if (lid >= num_levees) return;

    int ci = levee_catchment_idx[lid];

    float r_l   = river_length[ci];
    float r_w   = river_width[ci];
    float r_h   = river_height[ci];
    float area  = catchment_area[ci];
    float l_base_h  = levee_base_height[lid];
    float l_crown_h = levee_crown_height[lid];
    float l_frac    = levee_fraction[lid];

    float r_sto_curr = (float)river_storage[ci];
    float f_sto_curr = (float)flood_storage[ci];
    float f_dph_curr = flood_depth[ci];
    float total_sto  = r_sto_curr + f_sto_curr;

    float r_max_sto   = r_l * r_w * r_h;
    constexpr float inv_nfl = 1.0f / NUM_FLOOD_LEVELS;
    float dwth_inc    = (area / r_l) * inv_nfl;
    float l_distance  = l_frac * (area / r_l);

    float s_curr    = r_max_sto;
    float dhgt_pre  = 0.0f;
    float dwth_pre  = r_w;

    float l_base_sto = r_max_sto;
    float l_fill_sto = r_max_sto;
    bool found_base = false;
    bool found_fill = false;

    int ilev = (int)(l_frac * NUM_FLOOD_LEVELS);

    float dsto_fil_B = 0.0f;
    float dwth_fil_B = 0.0f;
    float ddph_fil_B = 0.0f;
    float gradient_B = 0.0f;
    bool  found_B    = false;

    #pragma unroll
    for (int i = 0; i < NUM_FLOOD_LEVELS; i++) {
        float depth_val = flood_depth_table[ci * NUM_FLOOD_LEVELS + i];
        float dhgt_seg  = fmaxf(depth_val - dhgt_pre, 1e-6f);
        float dwth_mid  = dwth_pre + 0.5f * dwth_inc;
        float dsto_seg  = r_l * dwth_mid * dhgt_seg;
        float s_next    = s_curr + dsto_seg;
        float gradient  = dhgt_seg / dwth_inc;

        if (!found_base && l_base_h > dhgt_pre && l_base_h <= depth_val) {
            float ratio = (l_base_h - dhgt_pre) / dhgt_seg;
            l_base_sto = s_curr + r_l * (dwth_pre + 0.5f * ratio * dwth_inc)
                         * (ratio * dhgt_seg);
            found_base = true;
        }

        if (!found_fill && l_crown_h > dhgt_pre && l_crown_h <= depth_val) {
            float ratio = (l_crown_h - dhgt_pre) / dhgt_seg;
            l_fill_sto = s_curr + r_l * (dwth_pre + 0.5f * ratio * dwth_inc)
                         * (ratio * dhgt_seg);
            found_fill = true;
        }

        if (i >= ilev && !found_B) {
            float dhgt_dif_loop = l_crown_h - l_base_h;
            float s_top_loop = l_base_sto
                + (l_distance + r_w) * dhgt_dif_loop * r_l;
            float dsto_add_wedge = (l_distance + r_w)
                * (l_crown_h - depth_val) * r_l;
            float threshold = s_next + dsto_add_wedge;

            float lower = (i == ilev) ? s_top_loop : dsto_fil_B;

            if (total_sto < threshold) {
                dsto_fil_B = lower;
                gradient_B = gradient;
                found_B = true;
            } else {
                dsto_fil_B = threshold;
                dwth_fil_B = dwth_inc * (i + 1) - l_distance;
                ddph_fil_B = depth_val - l_base_h;
            }
        }

        s_curr   = s_next;
        dhgt_pre = depth_val;
        dwth_pre += dwth_inc;
    }

    if (!found_base) {
        l_base_sto = (l_base_h > dhgt_pre)
            ? s_curr + r_l * dwth_pre * (l_base_h - dhgt_pre)
            : r_max_sto;
    }
    if (!found_fill) {
        l_fill_sto = (l_crown_h > dhgt_pre)
            ? s_curr + r_l * dwth_pre * (l_crown_h - dhgt_pre)
            : r_max_sto;
    }

    float dhgt_dif = l_crown_h - l_base_h;
    float s_top = l_base_sto + (l_distance + r_w) * dhgt_dif * r_l;

    bool is_case4 = (total_sto >= l_fill_sto);
    bool is_case3 = !is_case4 && (total_sto >= s_top);
    bool is_case2 = !is_case4 && !is_case3 && (total_sto >= l_base_sto);

    float r_sto, f_sto, p_sto, r_dph, f_dph, p_dph, f_frc;

    if (is_case2) {
        float dsto_add = total_sto - l_base_sto;
        float dwth_add = l_distance + r_w;
        f_dph = l_base_h + dsto_add / dwth_add / r_l;
        r_sto = r_max_sto + r_l * r_w * f_dph;
        r_dph = r_sto / (r_l * r_w);
        f_sto = fmaxf(total_sto - r_sto, 0.0f);
        p_sto = 0.0f;
        p_dph = 0.0f;
        f_frc = l_frac;
    } else if (is_case3) {
        float dsto_add = total_sto - dsto_fil_B;
        float term = dwth_fil_B * dwth_fil_B
            + 2.0f * dsto_add / r_l / (gradient_B + 1e-9f);
        float dwth_add = -dwth_fil_B + sqrtf(fmaxf(term, 0.0f));
        float ddph_add = dwth_add * gradient_B;
        if (!found_B) {
            ddph_add = dsto_add / (dwth_fil_B * r_l + 1e-9f);
        }
        p_dph = l_base_h + ddph_fil_B + ddph_add;
        f_frc = found_B
            ? fminf(fmaxf((dwth_fil_B + l_distance) / (dwth_inc * NUM_FLOOD_LEVELS), 0.0f), 1.0f)
            : 1.0f;

        f_dph = l_crown_h;
        r_sto = r_max_sto + r_l * r_w * f_dph;
        r_dph = r_sto / (r_l * r_w);
        f_sto = fmaxf(s_top - r_sto, 0.0f);
        p_sto = fmaxf(total_sto - r_sto - f_sto, 0.0f);
    } else if (is_case4) {
        f_dph = f_dph_curr;
        r_sto = r_sto_curr;
        float dsto_add_c4 = (f_dph - l_crown_h) * (l_distance + r_w) * r_l;
        f_sto = fmaxf(s_top + dsto_add_c4 - r_sto, 0.0f);
        p_sto = fmaxf(total_sto - r_sto - f_sto, 0.0f);
        r_dph = river_depth[ci];
        p_dph = f_dph;
        f_frc = flood_fraction[ci];
    } else {
        r_sto = r_sto_curr;
        f_sto = f_sto_curr;
        p_sto = 0.0f;
        r_dph = river_depth[ci];
        f_dph = f_dph_curr;
        p_dph = 0.0f;
        f_frc = flood_fraction[ci];
    }

    river_storage[ci]     = STO_CAST(r_sto);
    flood_storage[ci]     = STO_CAST(f_sto);
    protected_storage[ci] = STO_CAST(p_sto);
    river_depth[ci]       = r_dph;
    flood_depth[ci]       = f_dph;
    protected_depth[ci]   = p_dph;
    flood_fraction[ci]    = f_frc;
}

// =========================================================================
// Launcher: levee_stage
// =========================================================================
void launch_levee_stage(
    torch::Tensor levee_catchment_idx,
    torch::Tensor river_storage, torch::Tensor flood_storage,
    torch::Tensor protected_storage,
    torch::Tensor river_depth, torch::Tensor flood_depth,
    torch::Tensor protected_depth,
    torch::Tensor river_height, torch::Tensor flood_depth_table,
    torch::Tensor catchment_area, torch::Tensor river_width,
    torch::Tensor river_length,
    torch::Tensor levee_base_height, torch::Tensor levee_crown_height,
    torch::Tensor levee_fraction, torch::Tensor flood_fraction,
    int num_levees
) {
    const int bs = CMF_BLOCK_SIZE;
    const int grid = cdiv(num_levees, bs);
    const auto stream = at::cuda::getCurrentCUDAStream();
    compute_levee_stage_kernel<<<grid, bs, 0, stream>>>(
        levee_catchment_idx.data_ptr<int>(),
        river_storage.data_ptr<storage_t>(),
        flood_storage.data_ptr<storage_t>(),
        protected_storage.data_ptr<storage_t>(),
        river_depth.data_ptr<float>(), flood_depth.data_ptr<float>(),
        protected_depth.data_ptr<float>(),
        river_height.data_ptr<float>(), flood_depth_table.data_ptr<float>(),
        catchment_area.data_ptr<float>(), river_width.data_ptr<float>(),
        river_length.data_ptr<float>(),
        levee_base_height.data_ptr<float>(),
        levee_crown_height.data_ptr<float>(),
        levee_fraction.data_ptr<float>(),
        flood_fraction.data_ptr<float>(),
        num_levees);
}

// =========================================================================
// Kernel: compute_levee_bifurcation_outflow
//   NUM_BIF_LEVELS, CMF_GRAVITY are compile-time constants.
// =========================================================================
__global__ void compute_levee_bifurcation_outflow_kernel(
    const int*       __restrict__ bif_catchment_idx,
    const int*       __restrict__ bif_downstream_idx,
    const float*     __restrict__ bif_manning,
    float*           __restrict__ bif_outflow,
    const float*     __restrict__ bif_width,
    const float*     __restrict__ bif_length,
    const float*     __restrict__ bif_elevation,
    float*           __restrict__ bif_cs_depth,
    const float*     __restrict__ water_surface_elevation,
    const float*     __restrict__ protected_wse,
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

    float pwse_c = protected_wse[c_idx];
    float pwse_d = protected_wse[d_idx];
    float max_pwse = fmaxf(pwse_c, pwse_d);

    float b_slope = fminf(fmaxf((wse_c - wse_d) / b_length, -0.005f), 0.005f);
    float sto_c = (float)total_storage[c_idx];
    float sto_d = (float)total_storage[d_idx];

    float sum_out = 0.0f;
    #pragma unroll
    for (int lv = 0; lv < NUM_BIF_LEVELS; lv++) {
        int li = p * NUM_BIF_LEVELS + lv;
        float manning = bif_manning[li];
        float cs_d    = bif_cs_depth[li];
        float elev    = bif_elevation[li];
        float width   = bif_width[li];
        float outf    = bif_outflow[li];

        float current_max_wse = (lv == 0) ? max_wse : max_pwse;
        float upd_cs = fmaxf(current_max_wse - elev, 0.0f);

        float semi;
        if (lv == 0) {
            semi = fmaxf(sqrtf(upd_cs * cs_d), sqrtf(upd_cs * 0.01f));
        } else {
            semi = upd_cs;
        }

        float unit_out = outf / width;
        float num = width * (unit_out + CMF_GRAVITY * time_step * semi * b_slope);
        float den = 1.0f + CMF_GRAVITY * time_step * (manning * manning)
                    * fabsf(unit_out) * powf(semi, -7.0f / 3.0f);
        float upd = (semi > 1e-5f) ? (num / den) : 0.0f;
        sum_out += upd;
        bif_cs_depth[li] = upd_cs;
        bif_outflow[li]  = upd;
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
// Launcher: levee_bifurcation_outflow
// =========================================================================
void launch_levee_bifurcation_outflow(
    torch::Tensor bif_catchment_idx, torch::Tensor bif_downstream_idx,
    torch::Tensor bif_manning, torch::Tensor bif_outflow,
    torch::Tensor bif_width, torch::Tensor bif_length,
    torch::Tensor bif_elevation, torch::Tensor bif_cs_depth,
    torch::Tensor wse, torch::Tensor protected_wse,
    torch::Tensor total_storage, torch::Tensor outgoing_storage,
    torch::Tensor time_step_tensor, int num_paths
) {
    const int bs = CMF_BLOCK_SIZE;
    const int grid = cdiv(num_paths, bs);
    const auto stream = at::cuda::getCurrentCUDAStream();
    compute_levee_bifurcation_outflow_kernel<<<grid, bs, 0, stream>>>(
        bif_catchment_idx.data_ptr<int>(), bif_downstream_idx.data_ptr<int>(),
        bif_manning.data_ptr<float>(), bif_outflow.data_ptr<float>(),
        bif_width.data_ptr<float>(), bif_length.data_ptr<float>(),
        bif_elevation.data_ptr<float>(), bif_cs_depth.data_ptr<float>(),
        wse.data_ptr<float>(), protected_wse.data_ptr<float>(),
        total_storage.data_ptr<storage_t>(),
        outgoing_storage.data_ptr<storage_t>(),
        time_step_tensor.data_ptr<float>(), num_paths);
}
