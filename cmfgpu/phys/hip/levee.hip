// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// HIP backend for levee-aware flood-stage and levee bifurcation outflow.
//
// Mirrors cmfgpu/phys/triton/levee.py.  The stage kernel uses one thread per
// levee, writes back to the indexed catchment, and uses lane-local early exits
// for in-bank levee lanes and completed flood-level scans.  The levee
// bifurcation kernel uses the same lane-level skip style as outflow.cu.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

__device__ __forceinline__ float pow73(float x) { return x * x * cbrtf(x); }
__device__ __forceinline__ float clamp01(float x) { return fminf(fmaxf(x, 0.0f), 1.0f); }

template <typename STO>
__global__ void k_levee_stage(
    const int* __restrict__ levee_catchment_idx,
    STO* __restrict__ river_storage, STO* __restrict__ flood_storage,
    STO* __restrict__ protected_storage,
    float* __restrict__ river_depth, float* __restrict__ flood_depth,
    float* __restrict__ protected_depth,
    const float* __restrict__ river_height, const float* __restrict__ flood_depth_table,
    const float* __restrict__ catchment_area, const float* __restrict__ river_width,
    const float* __restrict__ river_length,
    const float* __restrict__ levee_base_height,
    const float* __restrict__ levee_crown_height,
    const float* __restrict__ levee_fraction,
    float* __restrict__ flood_fraction,
    long num_levees, int num_flood_levels)
{
    long li = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (li >= num_levees) return;

    int ci = __ldg(levee_catchment_idx + li);
    float rl = __ldg(river_length + ci);
    float rw = __ldg(river_width + ci);
    float rh = __ldg(river_height + ci);
    float ca = __ldg(catchment_area + ci);

    float l_crown = __ldg(levee_crown_height + li);
    float l_frac = __ldg(levee_fraction + li);
    float l_base_h = __ldg(levee_base_height + li);

    float riv_sto_curr = (float)river_storage[ci];
    float fld_sto_curr = (float)flood_storage[ci];
    float fld_depth_curr = __ldg(flood_depth + ci);
    float total_sto = riv_sto_curr + fld_sto_curr;

    float riv_max_sto = rl * rw * rh;

    // Below bank, no levee case can change river/flood state.  Still zero the
    // protected side to match the Triton false-case stores.
    bool possible_levee_case = total_sto >= riv_max_sto;
    if (!possible_levee_case) {
        river_storage[ci] = (STO)riv_sto_curr;
        flood_storage[ci] = (STO)fld_sto_curr;
        protected_storage[ci] = (STO)0;
        protected_depth[ci] = 0.0f;
        return;
    }

    float dwth_inc = (ca / rl) / (float)num_flood_levels;
    float levee_dist = l_frac * (ca / rl);

    float s_curr = riv_max_sto;
    float dhgt_pre = 0.0f;
    float dwth_pre = rw;
    float levee_base_sto = riv_max_sto;
    float levee_fill_sto = riv_max_sto;
    bool found_base = false;
    bool found_fill = false;

    int ilev = (int)(l_frac * (float)num_flood_levels);
    float dsto_fil_B = 0.0f;
    float dwth_fil_B = 0.0f;
    float ddph_fil_B = 0.0f;
    float gradient_B = 0.0f;
    bool found_B = false;

    bool scan_needed = true;
    for (int i = 0; i < num_flood_levels; ++i) {
        if (!scan_needed) break;

        float depth_val = __ldg(flood_depth_table + (long)ci * num_flood_levels + i);
        float dhgt_seg = fmaxf(depth_val - dhgt_pre, 1e-6f);
        float dwth_mid = dwth_pre + 0.5f * dwth_inc;
        float dsto_seg = rl * dwth_mid * dhgt_seg;
        float s_next = s_curr + dsto_seg;
        float gradient = dhgt_seg / dwth_inc;

        bool cond_base = (l_base_h > dhgt_pre) && (l_base_h <= depth_val);
        float ratio_base = (l_base_h - dhgt_pre) / dhgt_seg;
        float dsto_base_p = rl * (dwth_pre + 0.5f * ratio_base * dwth_inc) * (ratio_base * dhgt_seg);
        if (cond_base && !found_base) levee_base_sto = s_curr + dsto_base_p;
        found_base = found_base || cond_base;

        bool cond_fill = (l_crown > dhgt_pre) && (l_crown <= depth_val);
        float ratio_fill = (l_crown - dhgt_pre) / dhgt_seg;
        float dsto_fill_p = rl * (dwth_pre + 0.5f * ratio_fill * dwth_inc) * (ratio_fill * dhgt_seg);
        if (cond_fill && !found_fill) levee_fill_sto = s_curr + dsto_fill_p;
        found_fill = found_fill || cond_fill;

        float dhgt_dif_loop = l_crown - l_base_h;
        float s_top_loop = levee_base_sto + (levee_dist + rw) * dhgt_dif_loop * rl;
        float dsto_add_wedge = (levee_dist + rw) * (l_crown - depth_val) * rl;
        float threshold = s_next + dsto_add_wedge;

        bool cond_check = (i >= ilev) && !found_B;
        bool cond_found = cond_check && (total_sto < threshold);
        float current_lb = (i == ilev) ? s_top_loop : dsto_fil_B;
        if (cond_check && !cond_found) {
            dsto_fil_B = threshold;
            dwth_fil_B = dwth_inc * (float)(i + 1) - levee_dist;
            ddph_fil_B = depth_val - l_base_h;
        } else {
            dsto_fil_B = current_lb;
        }
        if (cond_found) gradient_B = gradient;
        found_B = found_B || cond_found;

        s_curr = s_next;
        dhgt_pre = depth_val;
        dwth_pre += dwth_inc;
        scan_needed = !(found_base && found_fill && found_B);
    }

    float s_base_extra = s_curr + rl * dwth_pre * (l_base_h - dhgt_pre);
    if (!found_base) levee_base_sto = (l_base_h > dhgt_pre) ? s_base_extra : riv_max_sto;
    float s_fill_extra = s_curr + rl * dwth_pre * (l_crown - dhgt_pre);
    if (!found_fill) levee_fill_sto = (l_crown > dhgt_pre) ? s_fill_extra : riv_max_sto;

    float dhgt_dif = l_crown - l_base_h;
    float s_top = levee_base_sto + (levee_dist + rw) * dhgt_dif * rl;

    bool is_case4 = total_sto >= levee_fill_sto;
    bool is_case3 = (!is_case4) && (total_sto >= s_top);
    bool is_case2 = (!is_case4) && (!is_case3) && (total_sto >= levee_base_sto);

    float dsto_add_c2 = total_sto - levee_base_sto;
    float dwth_add_c2 = levee_dist + rw;
    float f_dph_c2 = l_base_h + dsto_add_c2 / dwth_add_c2 / rl;
    float r_sto_c2 = riv_max_sto + rl * rw * f_dph_c2;
    float r_dph_c2 = r_sto_c2 / rl / rw;
    float f_sto_c2 = fmaxf(total_sto - r_sto_c2, 0.0f);
    float f_frc_c2 = l_frac;

    float dsto_add_B = total_sto - dsto_fil_B;
    float term_B = dwth_fil_B * dwth_fil_B + 2.0f * dsto_add_B / rl / (gradient_B + 1e-9f);
    float dwth_add_B = -dwth_fil_B + sqrtf(fmaxf(term_B, 0.0f));
    float ddph_add_B = dwth_add_B * gradient_B;
    float p_dph_B_found = l_base_h + ddph_fil_B + ddph_add_B;
    float f_frc_B_found = (dwth_fil_B + levee_dist) / (dwth_inc * (float)num_flood_levels);
    float ddph_add_B_extra = dsto_add_B / (dwth_fil_B * rl + 1e-9f);
    float p_dph_B_extra = l_base_h + ddph_fil_B + ddph_add_B_extra;
    float p_dph_B = found_B ? p_dph_B_found : p_dph_B_extra;
    float f_frc_B = found_B ? f_frc_B_found : 1.0f;

    float f_dph_c3 = l_crown;
    float r_sto_c3 = riv_max_sto + rl * rw * f_dph_c3;
    float r_dph_c3 = r_sto_c3 / rl / rw;
    float f_sto_c3 = fmaxf(s_top - r_sto_c3, 0.0f);
    float p_sto_c3 = fmaxf(total_sto - r_sto_c3 - f_sto_c3, 0.0f);
    float p_dph_c3 = p_dph_B;
    float f_frc_c3 = clamp01(f_frc_B);

    float f_dph_c4 = fld_depth_curr;
    float r_sto_c4 = riv_sto_curr;
    float dsto_add_c4 = (f_dph_c4 - l_crown) * (levee_dist + rw) * rl;
    float f_sto_c4 = fmaxf(s_top + dsto_add_c4 - r_sto_c4, 0.0f);
    float p_sto_c4 = fmaxf(total_sto - r_sto_c4 - f_sto_c4, 0.0f);
    float p_dph_c4 = f_dph_c4;

    float r_dph_curr = __ldg(river_depth + ci);
    float f_frc_curr = __ldg(flood_fraction + ci);

    float r_sto = is_case2 ? r_sto_c2 : (is_case3 ? r_sto_c3 : (is_case4 ? r_sto_c4 : riv_sto_curr));
    float f_sto = is_case2 ? f_sto_c2 : (is_case3 ? f_sto_c3 : (is_case4 ? f_sto_c4 : fld_sto_curr));
    float p_sto = is_case2 ? 0.0f : (is_case3 ? p_sto_c3 : (is_case4 ? p_sto_c4 : 0.0f));
    float r_dph = is_case2 ? r_dph_c2 : (is_case3 ? r_dph_c3 : r_dph_curr);
    float f_dph = is_case2 ? f_dph_c2 : (is_case3 ? f_dph_c3 : fld_depth_curr);
    float p_dph = is_case2 ? 0.0f : (is_case3 ? p_dph_c3 : (is_case4 ? p_dph_c4 : 0.0f));
    float f_frc = is_case2 ? f_frc_c2 : (is_case3 ? f_frc_c3 : f_frc_curr);

    river_storage[ci] = (STO)r_sto;
    flood_storage[ci] = (STO)f_sto;
    protected_storage[ci] = (STO)p_sto;
    river_depth[ci] = r_dph;
    flood_depth[ci] = f_dph;
    protected_depth[ci] = p_dph;
    flood_fraction[ci] = f_frc;
}

template <typename STO>
__global__ void k_levee_bif_outflow(
    const int* __restrict__ cat_idx, const int* __restrict__ dn_idx,
    const float* __restrict__ manning, float* __restrict__ outflow,
    const float* __restrict__ width, const float* __restrict__ length,
    const float* __restrict__ elevation, float* __restrict__ cs_depth,
    const float* __restrict__ wse, const float* __restrict__ protected_wse,
    const STO* __restrict__ total_storage, STO* __restrict__ outgoing_storage,
    float gravity, const float* __restrict__ time_step_ptr,
    long num_paths, int num_levels)
{
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t >= num_paths) return;

    float time_step = __ldg(time_step_ptr);

    int ci = __ldg(cat_idx + t);
    int di = __ldg(dn_idx + t);
    float blen = __ldg(length + t);

    float wse_c = __ldg(wse + ci);
    float wse_d = __ldg(wse + di);
    float max_wse = fmaxf(wse_c, wse_d);
    float pwse_c = __ldg(protected_wse + ci);
    float pwse_d = __ldg(protected_wse + di);
    float max_pwse = fmaxf(pwse_c, pwse_d);

    float slope = (wse_c - wse_d) / blen;
    slope = fminf(fmaxf(slope, -0.005f), 0.005f);

    float ts_c = (float)total_storage[ci];
    float ts_d = (float)total_storage[di];

    float sum_out = 0.0f;
    for (int lv = 0; lv < num_levels; ++lv) {
        long level_idx = t * (long)num_levels + lv;
        float current_max_wse = (lv == 0) ? max_wse : max_pwse;
        float elv = __ldg(elevation + level_idx);
        float upd_csd = fmaxf(current_max_wse - elv, 0.0f);
        float semi_depth;
        if (lv == 0) {
            float old_csd = __ldg(cs_depth + level_idx);
            semi_depth = fmaxf(sqrtf(upd_csd * old_csd), sqrtf(upd_csd * 0.01f));
        } else {
            semi_depth = upd_csd;
        }

        bool flow_condition = semi_depth > 1e-5f;
        float upd_out = 0.0f;
        if (flow_condition) {
            float man = __ldg(manning + level_idx);
            float w = __ldg(width + level_idx);
            float o = outflow[level_idx];
            float unit_o = o / w;
            float num = w * (unit_o + gravity * time_step * semi_depth * slope);
            float den = 1.0f + gravity * time_step * (man * man) * fabsf(unit_o)
                * (1.0f / pow73(semi_depth));
            upd_out = num / den;
        }

        sum_out += upd_out;
        cs_depth[level_idx] = upd_csd;
        outflow[level_idx] = upd_out;
    }

    float limit_rate = fminf(
        0.05f * fminf(ts_c, ts_d) / (fabsf(sum_out) * time_step), 1.0f);
    sum_out *= limit_rate;
    for (int lv = 0; lv < num_levels; ++lv) {
        long level_idx = t * (long)num_levels + lv;
        outflow[level_idx] = outflow[level_idx] * limit_rate;
    }

    float pos = fmaxf(sum_out, 0.0f);
    float neg = fminf(sum_out, 0.0f);
    atomicAdd(outgoing_storage + ci, (STO)(pos * time_step));
    atomicAdd(outgoing_storage + di, (STO)(-neg * time_step));
}

void launch_levee_stage(
    at::Tensor levee_catchment_idx,
    at::Tensor river_storage, at::Tensor flood_storage, at::Tensor protected_storage,
    at::Tensor river_depth, at::Tensor flood_depth, at::Tensor protected_depth,
    at::Tensor river_height, at::Tensor flood_depth_table,
    at::Tensor catchment_area, at::Tensor river_width, at::Tensor river_length,
    at::Tensor levee_base_height, at::Tensor levee_crown_height,
    at::Tensor levee_fraction, at::Tensor flood_fraction,
    long num_levees, long num_flood_levels, long block)
{
    int grid = (int)((num_levees + block - 1) / block);
    hipStream_t stream = c10::hip::getCurrentHIPStream();
    if (river_storage.scalar_type() == at::kDouble) {
        k_levee_stage<double><<<grid, (int)block, 0, stream>>>(
            levee_catchment_idx.data_ptr<int>(),
            river_storage.data_ptr<double>(), flood_storage.data_ptr<double>(),
            protected_storage.data_ptr<double>(),
            river_depth.data_ptr<float>(), flood_depth.data_ptr<float>(),
            protected_depth.data_ptr<float>(),
            river_height.data_ptr<float>(), flood_depth_table.data_ptr<float>(),
            catchment_area.data_ptr<float>(), river_width.data_ptr<float>(),
            river_length.data_ptr<float>(),
            levee_base_height.data_ptr<float>(), levee_crown_height.data_ptr<float>(),
            levee_fraction.data_ptr<float>(), flood_fraction.data_ptr<float>(),
            num_levees, (int)num_flood_levels);
    } else {
        k_levee_stage<float><<<grid, (int)block, 0, stream>>>(
            levee_catchment_idx.data_ptr<int>(),
            river_storage.data_ptr<float>(), flood_storage.data_ptr<float>(),
            protected_storage.data_ptr<float>(),
            river_depth.data_ptr<float>(), flood_depth.data_ptr<float>(),
            protected_depth.data_ptr<float>(),
            river_height.data_ptr<float>(), flood_depth_table.data_ptr<float>(),
            catchment_area.data_ptr<float>(), river_width.data_ptr<float>(),
            river_length.data_ptr<float>(),
            levee_base_height.data_ptr<float>(), levee_crown_height.data_ptr<float>(),
            levee_fraction.data_ptr<float>(), flood_fraction.data_ptr<float>(),
            num_levees, (int)num_flood_levels);
    }
}

void launch_levee_bif_outflow(
    at::Tensor cat_idx, at::Tensor dn_idx,
    at::Tensor manning, at::Tensor outflow, at::Tensor width,
    at::Tensor length, at::Tensor elevation, at::Tensor cs_depth,
    at::Tensor wse, at::Tensor protected_wse,
    at::Tensor total_storage, at::Tensor outgoing_storage,
    double gravity, at::Tensor time_step,
    long num_paths, long num_levels, long block)
{
    int grid = (int)((num_paths + block - 1) / block);
    hipStream_t stream = c10::hip::getCurrentHIPStream();
    if (total_storage.scalar_type() == at::kDouble) {
        k_levee_bif_outflow<double><<<grid, (int)block, 0, stream>>>(
            cat_idx.data_ptr<int>(), dn_idx.data_ptr<int>(),
            manning.data_ptr<float>(), outflow.data_ptr<float>(), width.data_ptr<float>(),
            length.data_ptr<float>(), elevation.data_ptr<float>(), cs_depth.data_ptr<float>(),
            wse.data_ptr<float>(), protected_wse.data_ptr<float>(),
            total_storage.data_ptr<double>(), outgoing_storage.data_ptr<double>(),
            (float)gravity, time_step.data_ptr<float>(), num_paths, (int)num_levels);
    } else {
        k_levee_bif_outflow<float><<<grid, (int)block, 0, stream>>>(
            cat_idx.data_ptr<int>(), dn_idx.data_ptr<int>(),
            manning.data_ptr<float>(), outflow.data_ptr<float>(), width.data_ptr<float>(),
            length.data_ptr<float>(), elevation.data_ptr<float>(), cs_depth.data_ptr<float>(),
            wse.data_ptr<float>(), protected_wse.data_ptr<float>(),
            total_storage.data_ptr<float>(), outgoing_storage.data_ptr<float>(),
            (float)gravity, time_step.data_ptr<float>(), num_paths, (int)num_levels);
    }
}
