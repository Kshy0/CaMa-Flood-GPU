// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// CUDA backend for levee-aware flood-stage and levee bifurcation outflow.
//
// The stage kernel uses one thread per levee, writes back to the indexed
// catchment, and uses lane-local early exits for in-bank levee lanes and
// completed flood-level scans.  The levee bifurcation kernel uses the same
// lane-level skip style as outflow.cu.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "block_reduce.cuh"

// Global water-balance counters the levee LOG path accumulates.
#define CMF_LEVEE_LOG_SUMS 5

template <typename REAL>
__device__ __forceinline__ REAL pow73(REAL x) { return x * x * cbrt(x); }
template <typename REAL>
__device__ __forceinline__ REAL clamp01(REAL x) {
    return fmin(fmax(x, (REAL)0), (REAL)1);
}

template <typename REAL, typename STO, bool LOG>
__global__ void k_levee_stage(
    const int* __restrict__ levee_catchment_idx,
    STO* __restrict__ river_storage, STO* __restrict__ flood_storage,
    STO* __restrict__ protected_storage,
    REAL* __restrict__ river_depth, REAL* __restrict__ flood_depth,
    REAL* __restrict__ protected_depth,
    const REAL* __restrict__ river_height, const REAL* __restrict__ flood_depth_table,
    const REAL* __restrict__ catchment_area, const REAL* __restrict__ river_width,
    const REAL* __restrict__ river_length,
    const REAL* __restrict__ levee_base_height,
    const REAL* __restrict__ levee_crown_height,
    const REAL* __restrict__ levee_fraction,
    REAL* __restrict__ flood_fraction,
    REAL* __restrict__ total_storage_stage_sum,
    REAL* __restrict__ river_storage_sum,
    REAL* __restrict__ flood_storage_sum,
    REAL* __restrict__ flood_area_sum,
    REAL* __restrict__ total_stage_error_sum,
    const int* __restrict__ current_step_ptr,
    long num_levees, int num_flood_levels)
{
    long li = blockIdx.x * (long)blockDim.x + threadIdx.x;
    REAL log_sum[CMF_LEVEE_LOG_SUMS];
    if constexpr (LOG) {
        // Out-of-range lanes stay alive with zero contributions so every
        // thread reaches the block reduction below.
#pragma unroll
        for (int i = 0; i < CMF_LEVEE_LOG_SUMS; ++i) log_sum[i] = (REAL)0;
    } else {
        if (li >= num_levees) return;
    }
    int step = LOG ? __ldg(current_step_ptr) : 0;

    if (li < num_levees) {

        int ci = __ldg(levee_catchment_idx + li);
        REAL rl = __ldg(river_length + ci);
        REAL rw = __ldg(river_width + ci);
        REAL rh = __ldg(river_height + ci);
        REAL ca = __ldg(catchment_area + ci);

        REAL l_crown = __ldg(levee_crown_height + li);
        REAL l_frac = __ldg(levee_fraction + li);
        REAL l_base_h = __ldg(levee_base_height + li);

        STO riv_sto_curr_hp = river_storage[ci];
        STO fld_sto_curr_hp = flood_storage[ci];
        STO total_sto_hp = riv_sto_curr_hp + fld_sto_curr_hp;
        REAL riv_sto_curr = (REAL)riv_sto_curr_hp;
        REAL fld_sto_curr = (REAL)fld_sto_curr_hp;
        REAL fld_depth_curr = __ldg(flood_depth + ci);
        REAL total_sto = (REAL)total_sto_hp;

        REAL riv_max_sto = rl * rw * rh;

        // Below bank, no levee case can change river/flood state.  Still zero the
        // protected side to keep outputs deterministic.
        bool possible_levee_case = total_sto >= riv_max_sto;
        if (!possible_levee_case) {
            if constexpr (LOG) {
                log_sum[0] += total_sto * (REAL)1e-9;
                log_sum[2] += riv_sto_curr * (REAL)1e-9;
                log_sum[3] += fld_sto_curr * (REAL)1e-9;
                log_sum[4] += __ldg(flood_fraction + ci) * ca * (REAL)1e-9;
            }
            river_storage[ci] = riv_sto_curr_hp;
            flood_storage[ci] = fld_sto_curr_hp;
            protected_storage[ci] = (STO)0;
            protected_depth[ci] = (REAL)0.0;
        } else {

        REAL dwth_inc = (ca / rl) / (REAL)num_flood_levels;
        REAL levee_dist = l_frac * (ca / rl);

        REAL s_curr = riv_max_sto;
        REAL dhgt_pre = (REAL)0.0;
        REAL dwth_pre = rw;
        REAL levee_base_sto = riv_max_sto;
        REAL levee_fill_sto = riv_max_sto;
        bool found_base = false;
        bool found_fill = false;

        int ilev = (int)(l_frac * (REAL)num_flood_levels);
        REAL dsto_fil_B = (REAL)0.0;
        REAL dwth_fil_B = (REAL)0.0;
        REAL ddph_fil_B = (REAL)0.0;
        REAL gradient_B = (REAL)0.0;
        bool found_B = false;

        bool scan_needed = true;
        for (int i = 0; i < num_flood_levels; ++i) {
            if (!scan_needed) break;

            REAL depth_val = __ldg(flood_depth_table + (long)ci * num_flood_levels + i);
            REAL dhgt_seg = fmax(depth_val - dhgt_pre, (REAL)1e-6);
            REAL dwth_mid = dwth_pre + (REAL)0.5 * dwth_inc;
            REAL dsto_seg = rl * dwth_mid * dhgt_seg;
            REAL s_next = s_curr + dsto_seg;
            REAL gradient = dhgt_seg / dwth_inc;

            bool cond_base = (l_base_h > dhgt_pre) && (l_base_h <= depth_val);
            REAL ratio_base = (l_base_h - dhgt_pre) / dhgt_seg;
            REAL dsto_base_p = rl * (dwth_pre + (REAL)0.5 * ratio_base * dwth_inc) * (ratio_base * dhgt_seg);
            if (cond_base && !found_base) levee_base_sto = s_curr + dsto_base_p;
            found_base = found_base || cond_base;

            bool cond_fill = (l_crown > dhgt_pre) && (l_crown <= depth_val);
            REAL ratio_fill = (l_crown - dhgt_pre) / dhgt_seg;
            REAL dsto_fill_p = rl * (dwth_pre + (REAL)0.5 * ratio_fill * dwth_inc) * (ratio_fill * dhgt_seg);
            if (cond_fill && !found_fill) levee_fill_sto = s_curr + dsto_fill_p;
            found_fill = found_fill || cond_fill;

            REAL dhgt_dif_loop = l_crown - l_base_h;
            REAL s_top_loop = levee_base_sto + (levee_dist + rw) * dhgt_dif_loop * rl;
            REAL dsto_add_wedge = (levee_dist + rw) * (l_crown - depth_val) * rl;
            REAL threshold = s_next + dsto_add_wedge;

            bool cond_check = (i >= ilev) && !found_B;
            bool cond_found = cond_check && (total_sto < threshold);
            REAL current_lb = (i == ilev) ? s_top_loop : dsto_fil_B;
            if (cond_check && !cond_found) {
                dsto_fil_B = threshold;
                dwth_fil_B = dwth_inc * (REAL)(i + 1) - levee_dist;
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

        REAL s_base_extra = s_curr + rl * dwth_pre * (l_base_h - dhgt_pre);
        if (!found_base) levee_base_sto = (l_base_h > dhgt_pre) ? s_base_extra : riv_max_sto;
        REAL s_fill_extra = s_curr + rl * dwth_pre * (l_crown - dhgt_pre);
        if (!found_fill) levee_fill_sto = (l_crown > dhgt_pre) ? s_fill_extra : riv_max_sto;

        REAL dhgt_dif = l_crown - l_base_h;
        REAL s_top = levee_base_sto + (levee_dist + rw) * dhgt_dif * rl;

        bool is_case4 = total_sto >= levee_fill_sto;
        bool is_case3 = (!is_case4) && (total_sto >= s_top);
        bool is_case2 = (!is_case4) && (!is_case3) && (total_sto >= levee_base_sto);

        REAL dsto_add_c2 = total_sto - levee_base_sto;
        REAL dwth_add_c2 = levee_dist + rw;
        REAL f_dph_c2 = l_base_h + dsto_add_c2 / dwth_add_c2 / rl;
        REAL r_sto_c2 = riv_max_sto + rl * rw * f_dph_c2;
        REAL r_dph_c2 = r_sto_c2 / rl / rw;
        REAL f_sto_c2 = fmax(total_sto - r_sto_c2, (REAL)0.0);
        REAL f_frc_c2 = l_frac;

        REAL dsto_add_B = total_sto - dsto_fil_B;
        REAL term_B = dwth_fil_B * dwth_fil_B + (REAL)2.0 * dsto_add_B / rl / (gradient_B + (REAL)1e-9);
        REAL dwth_add_B = -dwth_fil_B + sqrt(fmax(term_B, (REAL)0.0));
        REAL ddph_add_B = dwth_add_B * gradient_B;
        REAL p_dph_B_found = l_base_h + ddph_fil_B + ddph_add_B;
        REAL f_frc_B_found = (dwth_fil_B + levee_dist) / (dwth_inc * (REAL)num_flood_levels);
        REAL ddph_add_B_extra = dsto_add_B / (dwth_fil_B * rl + (REAL)1e-9);
        REAL p_dph_B_extra = l_base_h + ddph_fil_B + ddph_add_B_extra;
        REAL p_dph_B = found_B ? p_dph_B_found : p_dph_B_extra;
        REAL f_frc_B = found_B ? f_frc_B_found : (REAL)1.0;

        REAL f_dph_c3 = l_crown;
        REAL r_sto_c3 = riv_max_sto + rl * rw * f_dph_c3;
        REAL r_dph_c3 = r_sto_c3 / rl / rw;
        REAL f_sto_c3 = fmax(s_top - r_sto_c3, (REAL)0.0);
        REAL p_dph_c3 = p_dph_B;
        REAL f_frc_c3 = clamp01(f_frc_B);

        REAL f_dph_c4 = fld_depth_curr;
        REAL r_sto_c4 = riv_sto_curr;
        REAL dsto_add_c4 = (f_dph_c4 - l_crown) * (levee_dist + rw) * rl;
        REAL f_sto_c4 = fmax(s_top + dsto_add_c4 - r_sto_c4, (REAL)0.0);
        REAL p_dph_c4 = f_dph_c4;

        REAL r_dph_curr = __ldg(river_depth + ci);
        REAL f_frc_curr = __ldg(flood_fraction + ci);

        REAL r_sto_candidate = is_case2 ? r_sto_c2 : (is_case3 ? r_sto_c3 : (is_case4 ? r_sto_c4 : riv_sto_curr));
        REAL f_sto_candidate = is_case2 ? f_sto_c2 : (is_case3 ? f_sto_c3 : (is_case4 ? f_sto_c4 : fld_sto_curr));
        bool no_levee_partition = !is_case2 && !is_case3 && !is_case4;
        STO r_sto = (is_case4 || no_levee_partition)
            ? riv_sto_curr_hp : fmin((STO)r_sto_candidate, total_sto_hp);
        STO remaining_sto = total_sto_hp - r_sto;
        remaining_sto = remaining_sto > (STO)0 ? remaining_sto : (STO)0;
        STO f_sto_candidate_hp = (STO)f_sto_candidate;
        STO f_sto = fmin(
            f_sto_candidate_hp > (STO)0 ? f_sto_candidate_hp : (STO)0,
            remaining_sto);
        if (is_case2) f_sto = remaining_sto;
        if (no_levee_partition) f_sto = fld_sto_curr_hp;
        STO p_sto = remaining_sto - f_sto;
        p_sto = p_sto > (STO)0 ? p_sto : (STO)0;
        REAL r_dph = is_case2 ? r_dph_c2 : (is_case3 ? r_dph_c3 : r_dph_curr);
        REAL f_dph = is_case2 ? f_dph_c2 : (is_case3 ? f_dph_c3 : fld_depth_curr);
        REAL p_dph = is_case2 ? (REAL)0.0 : (is_case3 ? p_dph_c3 : (is_case4 ? p_dph_c4 : (REAL)0.0));
        REAL f_frc = is_case2 ? f_frc_c2 : (is_case3 ? f_frc_c3 : f_frc_curr);

        if constexpr (LOG) {
            STO total_new = r_sto + f_sto + p_sto;
            log_sum[0] += (REAL)total_new * (REAL)1e-9;
            log_sum[1] += (REAL)(total_new - total_sto_hp) * (REAL)1e-9;
            log_sum[2] += (REAL)r_sto * (REAL)1e-9;
            log_sum[3] += (REAL)f_sto * (REAL)1e-9;
            log_sum[4] += f_frc * ca * (REAL)1e-9;
        }

        river_storage[ci] = r_sto;
        flood_storage[ci] = f_sto;
        protected_storage[ci] = p_sto;
        river_depth[ci] = r_dph;
        flood_depth[ci] = f_dph;
        protected_depth[ci] = p_dph;
        flood_fraction[ci] = f_frc;

        }  // levee partition branch
    }  // active lane

    if constexpr (LOG) {
        REAL* const destination[CMF_LEVEE_LOG_SUMS] = {
            total_storage_stage_sum, total_stage_error_sum,
            river_storage_sum, flood_storage_sum, flood_area_sum,
        };
        cmf_block_atomic_add<REAL, CMF_LEVEE_LOG_SUMS>(
            log_sum, destination, step);
    }
}

template <typename REAL, typename STO>
__global__ void k_levee_bif_outflow(
    const int* __restrict__ cat_idx, const int* __restrict__ dn_idx,
    const REAL* __restrict__ manning, REAL* __restrict__ outflow,
    const REAL* __restrict__ width, const REAL* __restrict__ length,
    const REAL* __restrict__ elevation, REAL* __restrict__ cs_depth,
    const REAL* __restrict__ wse, const REAL* __restrict__ protected_wse,
    const STO* __restrict__ total_storage, STO* __restrict__ outgoing_storage,
    REAL gravity, const REAL* __restrict__ time_step_ptr,
    long num_paths, int num_levels)
{
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t >= num_paths) return;

    REAL time_step = __ldg(time_step_ptr);

    int ci = __ldg(cat_idx + t);
    int di = __ldg(dn_idx + t);
    REAL blen = __ldg(length + t);

    REAL wse_c = __ldg(wse + ci);
    REAL wse_d = __ldg(wse + di);
    REAL max_wse = fmax(wse_c, wse_d);
    REAL pwse_c = __ldg(protected_wse + ci);
    REAL pwse_d = __ldg(protected_wse + di);
    REAL max_pwse = fmax(pwse_c, pwse_d);

    REAL slope = (wse_c - wse_d) / blen;
    slope = fmin(fmax(slope, (REAL)-0.005), (REAL)0.005);

    REAL ts_c = (REAL)total_storage[ci];
    REAL ts_d = (REAL)total_storage[di];

    REAL sum_out = (REAL)0.0;
    for (int lv = 0; lv < num_levels; ++lv) {
        long level_idx = t * (long)num_levels + lv;
        REAL current_max_wse = (lv == 0) ? max_wse : max_pwse;
        REAL elv = __ldg(elevation + level_idx);
        REAL upd_csd = fmax(current_max_wse - elv, (REAL)0.0);
        REAL semi_depth;
        if (lv == 0) {
            REAL old_csd = __ldg(cs_depth + level_idx);
            semi_depth = sqrt(upd_csd * old_csd);
            if (semi_depth <= (REAL)0.0) semi_depth = upd_csd;
        } else {
            semi_depth = upd_csd;
        }

        bool flow_condition = semi_depth > (REAL)1e-5;
        REAL upd_out = (REAL)0.0;
        if (flow_condition) {
            REAL man = __ldg(manning + level_idx);
            REAL w = __ldg(width + level_idx);
            REAL o = outflow[level_idx];
            REAL unit_o = o / w;
            REAL num = w * (unit_o + gravity * time_step * semi_depth * slope);
            REAL den = (REAL)1.0 + gravity * time_step * (man * man) * fabs(unit_o)
                * ((REAL)1.0 / pow73(semi_depth));
            upd_out = num / den;
        }

        sum_out += upd_out;
        cs_depth[level_idx] = upd_csd;
        outflow[level_idx] = upd_out;
    }

    REAL limit_rate = fmin(
        (REAL)0.05 * fmin(ts_c, ts_d) / (fabs(sum_out) * time_step), (REAL)1.0);
    sum_out *= limit_rate;
    for (int lv = 0; lv < num_levels; ++lv) {
        long level_idx = t * (long)num_levels + lv;
        outflow[level_idx] = outflow[level_idx] * limit_rate;
    }

    REAL pos = fmax(sum_out, (REAL)0.0);
    REAL neg = fmin(sum_out, (REAL)0.0);
    atomicAdd(outgoing_storage + ci, (STO)(pos * time_step));
    atomicAdd(outgoing_storage + di, (STO)(-neg * time_step));
}

void launch_levee_stage(
    at::Tensor levee_catchment_idx_ptr,
    at::Tensor river_storage_ptr, at::Tensor flood_storage_ptr,
    at::Tensor protected_storage_ptr, at::Tensor river_depth_ptr,
    at::Tensor flood_depth_ptr, at::Tensor protected_depth_ptr,
    at::Tensor river_height_ptr, at::Tensor flood_depth_table_ptr,
    at::Tensor catchment_area_ptr, at::Tensor river_width_ptr,
    at::Tensor river_length_ptr, at::Tensor levee_base_height_ptr,
    at::Tensor levee_crown_height_ptr, at::Tensor levee_fraction_ptr,
    at::Tensor flood_fraction_ptr, long num_catchments, long num_levees,
    int num_flood_levels, long BLOCK_SIZE)
{
    (void)num_catchments;
    int grid = (int)((num_levees + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
#define LAUNCH_LEVEE(REAL_T, STO_T) \
        k_levee_stage<REAL_T, STO_T, false><<<grid, (int)BLOCK_SIZE, 0, stream>>>( \
            levee_catchment_idx_ptr.data_ptr<int>(), \
            river_storage_ptr.data_ptr<STO_T>(), flood_storage_ptr.data_ptr<STO_T>(), \
            protected_storage_ptr.data_ptr<STO_T>(), \
            river_depth_ptr.data_ptr<REAL_T>(), flood_depth_ptr.data_ptr<REAL_T>(), \
            protected_depth_ptr.data_ptr<REAL_T>(), \
            river_height_ptr.data_ptr<REAL_T>(), flood_depth_table_ptr.data_ptr<REAL_T>(), \
            catchment_area_ptr.data_ptr<REAL_T>(), river_width_ptr.data_ptr<REAL_T>(), \
            river_length_ptr.data_ptr<REAL_T>(), \
            levee_base_height_ptr.data_ptr<REAL_T>(), levee_crown_height_ptr.data_ptr<REAL_T>(), \
            levee_fraction_ptr.data_ptr<REAL_T>(), flood_fraction_ptr.data_ptr<REAL_T>(), \
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, \
            num_levees, num_flood_levels)
    if (river_depth_ptr.scalar_type() == at::kDouble) {
        LAUNCH_LEVEE(double, double);
    } else if (river_storage_ptr.scalar_type() == at::kDouble) {
        LAUNCH_LEVEE(float, double);
    } else {
        LAUNCH_LEVEE(float, float);
    }
#undef LAUNCH_LEVEE
}

void launch_levee_stage_log(
    at::Tensor levee_catchment_idx_ptr,
    at::Tensor river_storage_ptr, at::Tensor flood_storage_ptr,
    at::Tensor protected_storage_ptr, at::Tensor river_depth_ptr,
    at::Tensor flood_depth_ptr, at::Tensor protected_depth_ptr,
    at::Tensor river_height_ptr, at::Tensor flood_depth_table_ptr,
    at::Tensor catchment_area_ptr, at::Tensor river_width_ptr,
    at::Tensor river_length_ptr, at::Tensor levee_base_height_ptr,
    at::Tensor levee_crown_height_ptr, at::Tensor levee_fraction_ptr,
    at::Tensor flood_fraction_ptr, at::Tensor total_storage_stage_sum_ptr,
    at::Tensor river_storage_sum_ptr, at::Tensor flood_storage_sum_ptr,
    at::Tensor flood_area_sum_ptr, at::Tensor total_stage_error_sum_ptr,
    at::Tensor current_step_ptr, long num_levees,
    int num_flood_levels, long BLOCK_SIZE)
{
    int grid = (int)((num_levees + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
#define LAUNCH_LEVEE_LOG(REAL_TYPE, STO_TYPE) \
    k_levee_stage<REAL_TYPE, STO_TYPE, true><<<grid, (int)BLOCK_SIZE, 0, stream>>>( \
        levee_catchment_idx_ptr.data_ptr<int>(), \
        river_storage_ptr.data_ptr<STO_TYPE>(), flood_storage_ptr.data_ptr<STO_TYPE>(), \
        protected_storage_ptr.data_ptr<STO_TYPE>(), river_depth_ptr.data_ptr<REAL_TYPE>(), \
        flood_depth_ptr.data_ptr<REAL_TYPE>(), protected_depth_ptr.data_ptr<REAL_TYPE>(), \
        river_height_ptr.data_ptr<REAL_TYPE>(), flood_depth_table_ptr.data_ptr<REAL_TYPE>(), \
        catchment_area_ptr.data_ptr<REAL_TYPE>(), river_width_ptr.data_ptr<REAL_TYPE>(), \
        river_length_ptr.data_ptr<REAL_TYPE>(), levee_base_height_ptr.data_ptr<REAL_TYPE>(), \
        levee_crown_height_ptr.data_ptr<REAL_TYPE>(), levee_fraction_ptr.data_ptr<REAL_TYPE>(), \
        flood_fraction_ptr.data_ptr<REAL_TYPE>(), \
        total_storage_stage_sum_ptr.data_ptr<REAL_TYPE>(), \
        river_storage_sum_ptr.data_ptr<REAL_TYPE>(), flood_storage_sum_ptr.data_ptr<REAL_TYPE>(), \
        flood_area_sum_ptr.data_ptr<REAL_TYPE>(), total_stage_error_sum_ptr.data_ptr<REAL_TYPE>(), \
        current_step_ptr.data_ptr<int>(), num_levees, num_flood_levels)
    if (river_depth_ptr.scalar_type() == at::kDouble) {
        LAUNCH_LEVEE_LOG(double, double);
    } else if (river_storage_ptr.scalar_type() == at::kDouble) {
        LAUNCH_LEVEE_LOG(float, double);
    } else {
        LAUNCH_LEVEE_LOG(float, float);
    }
#undef LAUNCH_LEVEE_LOG
}

void launch_levee_bif_outflow(
    at::Tensor bifurcation_catchment_idx_ptr,
    at::Tensor bifurcation_downstream_idx_ptr,
    at::Tensor bifurcation_manning_ptr, at::Tensor bifurcation_outflow_ptr,
    at::Tensor bifurcation_width_ptr, at::Tensor bifurcation_length_ptr,
    at::Tensor bifurcation_elevation_ptr,
    at::Tensor bifurcation_cross_section_depth_ptr,
    at::Tensor water_surface_elevation_ptr,
    at::Tensor protected_water_surface_elevation_ptr,
    at::Tensor total_storage_ptr, at::Tensor outgoing_storage_ptr,
    double gravity, at::Tensor time_step_ptr,
    long num_catchments, long num_bifurcation_paths,
    int num_bifurcation_levels,
    long BLOCK_SIZE)
{
    (void)num_catchments;
    int grid = (int)((num_bifurcation_paths + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
#define LAUNCH_LEVEE_BIF(REAL_T, STO_T) \
        k_levee_bif_outflow<REAL_T, STO_T><<<grid, (int)BLOCK_SIZE, 0, stream>>>( \
            bifurcation_catchment_idx_ptr.data_ptr<int>(), \
            bifurcation_downstream_idx_ptr.data_ptr<int>(), \
            bifurcation_manning_ptr.data_ptr<REAL_T>(), \
            bifurcation_outflow_ptr.data_ptr<REAL_T>(), \
            bifurcation_width_ptr.data_ptr<REAL_T>(), \
            bifurcation_length_ptr.data_ptr<REAL_T>(), \
            bifurcation_elevation_ptr.data_ptr<REAL_T>(), \
            bifurcation_cross_section_depth_ptr.data_ptr<REAL_T>(), \
            water_surface_elevation_ptr.data_ptr<REAL_T>(), \
            protected_water_surface_elevation_ptr.data_ptr<REAL_T>(), \
            total_storage_ptr.data_ptr<STO_T>(), \
            outgoing_storage_ptr.data_ptr<STO_T>(), (REAL_T)gravity, \
            time_step_ptr.data_ptr<REAL_T>(), num_bifurcation_paths, \
            num_bifurcation_levels)
    if (water_surface_elevation_ptr.scalar_type() == at::kDouble) {
        LAUNCH_LEVEE_BIF(double, double);
    } else if (total_storage_ptr.scalar_type() == at::kDouble) {
        LAUNCH_LEVEE_BIF(float, double);
    } else {
        LAUNCH_LEVEE_BIF(float, float);
    }
#undef LAUNCH_LEVEE_BIF
}
