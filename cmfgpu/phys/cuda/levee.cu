// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
#include "block_reduce.cuh"
#include "canonical.cuh"

#define CMF_LEVEE_LOG_SUMS 5

template <typename REAL>
__device__ __forceinline__ REAL clamp01(REAL x) {
    return fmin(fmax(x, (REAL)0), (REAL)1);
}

template <typename REAL, typename STO, bool LOG>
__device__ __forceinline__ void k_levee_stage_cell(
    const int* __restrict__ levee_catchment_idx,
    const REAL* __restrict__ levee_river_max_storage,
    const REAL* __restrict__ levee_base_storage,
    const REAL* __restrict__ levee_top_storage,
    const REAL* __restrict__ levee_fill_storage,
    const REAL* __restrict__ levee_layer_top_storage,
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
    int num_levees, int num_flood_levels, int li)
{
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

        REAL l_frac = __ldg(levee_fraction + li);
        REAL l_base_h = __ldg(levee_base_height + li);
        REAL l_crown = fmax(__ldg(levee_crown_height + li), l_base_h);

        STO riv_sto_curr_hp = river_storage[ci];
        STO fld_sto_curr_hp = flood_storage[ci];
        STO total_sto_hp = riv_sto_curr_hp + fld_sto_curr_hp;
        REAL total_sto = (REAL)total_sto_hp;

        REAL riv_max_sto = levee_river_max_storage[li];

        // Case 0, water only in the river channel: the default stage stands and
        // the protected side is dry.
        if (!(total_sto > riv_max_sto)) {
            if constexpr (LOG) {
                log_sum[0] += total_sto * (REAL)1e-9;
                log_sum[2] += (REAL)riv_sto_curr_hp * (REAL)1e-9;
                log_sum[3] += (REAL)fld_sto_curr_hp * (REAL)1e-9;
                log_sum[4] += __ldg(flood_fraction + ci) * ca * (REAL)1e-9;
            }
            protected_storage[ci] = (STO)0;
            protected_depth[ci] = (REAL)0.0;
        } else {

        REAL dwth_inc = (ca / rl) / (REAL)num_flood_levels;
        REAL levee_dist = l_frac * (ca / rl);

        REAL s_curr = riv_max_sto;
        REAL dhgt_pre = (REAL)0;
        REAL dwth_pre = rw;
        const REAL levee_base_sto = levee_base_storage[li];
        const REAL s_top = levee_top_storage[li];
        const REAL levee_fill_sto = levee_fill_storage[li];
        const REAL top_ilev = levee_layer_top_storage[li];
        const bool case3 = total_sto >= levee_base_sto && total_sto >= s_top && total_sto < levee_fill_sto;
        const bool case4 = total_sto >= levee_base_sto && total_sto >= s_top && !(total_sto < levee_fill_sto);

        int ilev = (int)(l_frac * (REAL)num_flood_levels);
        REAL dsto_fil_B = (REAL)0.0;
        REAL dwth_fil_B = (REAL)0.0;
        REAL ddph_fil_B = (REAL)0.0;
        REAL gradient_B = (REAL)0.0;
        bool found_B = false;

        REAL dsto_fil_c4 = (REAL)0.0;
        REAL dwth_fil_c4 = (REAL)0.0;
        REAL gradient_c4 = (REAL)0.0;
        bool found_c4 = false;

        for (int i = 0; i < num_flood_levels; ++i) {
            if ((!case3 || found_B) && (!case4 || found_c4)) break;

            REAL depth_val = __ldg(flood_depth_table + (long)ci * num_flood_levels + i);
            REAL dhgt_seg = depth_val - dhgt_pre;
            REAL dwth_mid = dwth_pre + (REAL)0.5 * dwth_inc;
            REAL dsto_seg = rl * dwth_mid * dhgt_seg;
            REAL s_next = s_curr + dsto_seg;
            REAL gradient = dhgt_seg / dwth_inc;

            REAL dsto_add_wedge = (levee_dist + rw) * (l_crown - depth_val) * rl;
            REAL threshold = s_next + dsto_add_wedge;

            bool cond_check = case3 && (i >= ilev) && !found_B;
            bool cond_found = cond_check && (total_sto < threshold);
            REAL current_lb = (i == ilev) ? top_ilev : dsto_fil_B;
            if (cond_check && !cond_found) {
                dsto_fil_B = threshold;
                dwth_fil_B = dwth_inc * (REAL)(i + 1) - levee_dist;
                ddph_fil_B = depth_val - l_base_h;
            } else {
                dsto_fil_B = current_lb;
            }
            if (cond_found) gradient_B = gradient;
            found_B = found_B || cond_found;

            // Case 4 stops at the first layer the storage does not exceed.
            if (case4 && !found_c4 && !(total_sto > s_next)) {
                dsto_fil_c4 = s_curr;
                dwth_fil_c4 = dwth_pre;
                gradient_c4 = gradient;
                found_c4 = true;
            }

            s_curr = s_next;
            dhgt_pre = depth_val;
            dwth_pre += dwth_inc;
        }

        STO r_sto = riv_sto_curr_hp;
        STO f_sto = fld_sto_curr_hp;
        STO p_sto = (STO)0;
        REAL r_dph = __ldg(river_depth + ci);
        REAL f_dph = __ldg(flood_depth + ci);
        REAL p_dph = (REAL)0.0;
        REAL f_frc = __ldg(flood_fraction + ci);

        if (total_sto < levee_base_sto) {
            // Case 1, below the levee base: the default stage stands.
        } else if (total_sto < s_top) {
            // Case 2, river side below the crown, protected side dry.
            REAL dsto_add = total_sto - levee_base_sto;
            REAL dwth_add = levee_dist + rw;
            f_dph = l_base_h + dsto_add / dwth_add / rl;
            REAL r_sto_c = riv_max_sto + rl * rw * f_dph;
            r_sto = (STO)r_sto_c;
            r_dph = r_sto_c / rl / rw;
            f_sto = total_sto_hp - r_sto;
            f_sto = f_sto > (STO)0 ? f_sto : (STO)0;
            f_frc = l_frac;
        } else if (total_sto < levee_fill_sto) {
            // Case 3, river side at the crown, protected side filling.
            f_dph = l_crown;
            REAL r_sto_c = riv_max_sto + rl * rw * f_dph;
            r_sto = (STO)r_sto_c;
            r_dph = r_sto_c / rl / rw;
            f_sto = (STO)s_top - r_sto;
            f_sto = f_sto > (STO)0 ? f_sto : (STO)0;
            p_sto = total_sto_hp - r_sto - f_sto;
            p_sto = p_sto > (STO)0 ? p_sto : (STO)0;

            REAL dsto_add = total_sto - dsto_fil_B;
            if (found_B) {
                REAL dwth_add = -dwth_fil_B + sqrt(
                    dwth_fil_B * dwth_fil_B + (REAL)2.0 * dsto_add / rl / gradient_B);
                REAL ddph_add = dwth_add * gradient_B;
                p_dph = l_base_h + ddph_fil_B + ddph_add;
                f_frc = clamp01((dwth_fil_B + levee_dist) / (dwth_inc * (REAL)num_flood_levels));
            } else {
                REAL ddph_add = dsto_add / dwth_fil_B / rl;
                p_dph = l_base_h + ddph_fil_B + ddph_add;
                f_frc = (REAL)1.0;
            }
        } else {
            // Case 4, above the crown: the default river stage stands, with the
            // unclamped default-stage flood fraction.
            REAL dwth_add = (REAL)0.0;
            if (found_c4) {
                REAL dsto_add = total_sto - dsto_fil_c4;
                dwth_add = -dwth_fil_c4 + sqrt(
                    dwth_fil_c4 * dwth_fil_c4 + (REAL)2.0 * dsto_add / rl / gradient_c4);
            } else {
                dwth_fil_c4 = dwth_pre;
            }
            f_frc = (-rw + dwth_fil_c4 + dwth_add) / (dwth_inc * (REAL)num_flood_levels);

            REAL dsto_add = (f_dph - l_crown) * (levee_dist + rw) * rl;
            f_sto = (STO)(s_top + dsto_add) - r_sto;
            f_sto = f_sto > (STO)0 ? f_sto : (STO)0;
            p_sto = total_sto_hp - r_sto - f_sto;
            p_sto = p_sto > (STO)0 ? p_sto : (STO)0;
            p_dph = f_dph;
        }

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

// Generated-entry body over the canonical values ``a`` of compute_levee_stage,
// or of compute_levee_stage_log with LOG.
template <typename REAL, typename STO, bool LOG = false, class A>
__device__ __forceinline__ void levee_stage(const A& a, long t)
{
    const auto balance = cmf_balance<LOG>(a);
    k_levee_stage_cell<REAL, STO, LOG>(
        a.levee_catchment_idx_ptr, a.levee_river_max_storage_ptr, a.levee_base_storage_ptr, a.levee_top_storage_ptr, a.levee_fill_storage_ptr, a.levee_layer_top_storage_ptr, a.river_storage_ptr, a.flood_storage_ptr,
        a.protected_storage_ptr, a.river_depth_ptr, a.flood_depth_ptr, a.protected_depth_ptr,
        a.river_height_ptr, a.flood_depth_table_ptr, a.catchment_area_ptr, a.river_width_ptr,
        a.river_length_ptr, a.levee_base_height_ptr, a.levee_crown_height_ptr,
        a.levee_fraction_ptr, a.flood_fraction_ptr, balance.total_storage_stage_sum_ptr,
        balance.river_storage_sum_ptr, balance.flood_storage_sum_ptr,
        balance.flood_area_sum_ptr, balance.total_stage_error_sum_ptr,
        balance.current_step_ptr, a.num_levees, a.num_flood_levels, t);
}

template <typename REAL>
__device__ __forceinline__ REAL pow73(REAL x) { return x * x * cbrt(x); }

template <typename REAL, typename STO>
__device__ __forceinline__ void k_levee_bif_outflow_cell(
    const int* __restrict__ cat_idx, const int* __restrict__ dn_idx,
    const REAL* __restrict__ manning, REAL* __restrict__ outflow,
    const REAL* __restrict__ width, const REAL* __restrict__ length,
    const REAL* __restrict__ elevation, REAL* __restrict__ cs_depth,
    const REAL* __restrict__ river_depth,
    const REAL* __restrict__ protected_depth,
    const REAL* __restrict__ river_height,
    const REAL* __restrict__ catchment_elevation,
    const bool* __restrict__ is_levee,
    const STO* __restrict__ river_storage,
    const STO* __restrict__ flood_storage,
    const STO* __restrict__ protected_storage,
    STO* __restrict__ outgoing_storage,
    REAL gravity, const REAL* __restrict__ time_step_ptr, int num_levels, long t)
{
    REAL time_step = __ldg(time_step_ptr);

    int ci = __ldg(cat_idx + t);
    int di = __ldg(dn_idx + t);
    REAL blen = __ldg(length + t);

    REAL elevation_c = __ldg(catchment_elevation + ci);
    REAL elevation_d = __ldg(catchment_elevation + di);
    // D2SFCELV = D2RIVELV + D2RIVDPH with D2RIVELV = D2ELEVTN - D2RIVHGT.
    REAL wse_c = __ldg(river_depth + ci)
        + (elevation_c - __ldg(river_height + ci));
    REAL wse_d = __ldg(river_depth + di)
        + (elevation_d - __ldg(river_height + di));
    REAL max_wse = fmax(wse_c, wse_d);
    REAL pwse_c = is_levee[ci]
        ? fmin(elevation_c + __ldg(protected_depth + ci), wse_c) : wse_c;
    REAL pwse_d = is_levee[di]
        ? fmin(elevation_d + __ldg(protected_depth + di), wse_d) : wse_d;
    REAL max_pwse = fmax(pwse_c, pwse_d);

    REAL slope = (wse_c - wse_d) / blen;
    slope = fmin(fmax(slope, (REAL)-CMF_ROUTING_SLOPE_LIMIT), (REAL)CMF_ROUTING_SLOPE_LIMIT);

    REAL ts_c = (REAL)(
        river_storage[ci] + flood_storage[ci] + protected_storage[ci]);
    REAL ts_d = (REAL)(
        river_storage[di] + flood_storage[di] + protected_storage[di]);

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

    // CaMa-Flood LEVEE_OPT_PTHOUT limits a path only when its flow sum is non-zero.
    REAL limit_rate = (REAL)1.0;
    if (sum_out != (REAL)0.0) {
        limit_rate = fmin(
            (REAL)CMF_BACKFLOW_STORAGE_FRACTION * fmin(ts_c, ts_d) / (fabs(sum_out) * time_step), (REAL)1.0);
    }
    sum_out *= limit_rate;
    for (int lv = 0; lv < num_levels; ++lv) {
        long level_idx = t * (long)num_levels + lv;
        outflow[level_idx] = outflow[level_idx] * limit_rate;
    }

    REAL pos = fmax(sum_out, (REAL)0.0);
    REAL neg = fmin(sum_out, (REAL)0.0);
    // P2STOOUT flows, multiplied by the step in compute_inflow.
    atomicAdd(outgoing_storage + ci, (STO)pos);
    atomicAdd(outgoing_storage + di, (STO)(-neg));
}

// Generated-entry body over the canonical values ``a`` of
// compute_levee_bifurcation_outflow.
template <typename REAL, typename STO, class A>
__device__ __forceinline__ void levee_bif_outflow(const A& a, long t)
{
    k_levee_bif_outflow_cell<REAL, STO>(
        a.bifurcation_catchment_idx_ptr, a.bifurcation_downstream_idx_ptr,
        a.bifurcation_manning_ptr, a.bifurcation_outflow_ptr, a.bifurcation_width_ptr,
        a.bifurcation_length_ptr, a.bifurcation_elevation_ptr,
        a.bifurcation_cross_section_depth_ptr, a.river_depth_ptr, a.protected_depth_ptr,
        a.river_height_ptr, a.catchment_elevation_ptr, a.is_levee_ptr, a.river_storage_ptr,
        a.flood_storage_ptr, a.protected_storage_ptr, a.outgoing_storage_ptr, a.gravity,
        a.time_step_ptr, a.num_bifurcation_levels, t);
}

// Ensemble variants of levee_stage and levee_bif_outflow: offset member
// ``member``'s slices of the per-member buffers in their copy of ``a``, then
// run the single-member body.

template <typename REAL, typename STO, class A>
__device__ __forceinline__ void levee_stage_members(A a, long t, long member)
{
    const long member_offset = member * a.num_catchments;
    const long levee_offset = member * a.num_levees;
    a.levee_river_max_storage_ptr += levee_offset;
    a.levee_base_storage_ptr += levee_offset;
    a.levee_top_storage_ptr += levee_offset;
    a.levee_fill_storage_ptr += levee_offset;
    a.levee_layer_top_storage_ptr += levee_offset;
    a.river_storage_ptr += member_offset;
    a.flood_storage_ptr += member_offset;
    a.protected_storage_ptr += member_offset;
    a.river_depth_ptr += member_offset;
    a.flood_depth_ptr += member_offset;
    a.protected_depth_ptr += member_offset;
    a.flood_fraction_ptr += member_offset;
    if (a.batched_river_height) a.river_height_ptr += member_offset;
    if (a.batched_flood_depth_table)
        a.flood_depth_table_ptr += member_offset * a.num_flood_levels;
    if (a.batched_catchment_area) a.catchment_area_ptr += member_offset;
    if (a.batched_river_width) a.river_width_ptr += member_offset;
    if (a.batched_river_length) a.river_length_ptr += member_offset;
    if (a.batched_levee_base_height) a.levee_base_height_ptr += levee_offset;
    if (a.batched_levee_crown_height) a.levee_crown_height_ptr += levee_offset;
    if (a.batched_levee_fraction) a.levee_fraction_ptr += levee_offset;
    levee_stage<REAL, STO>(a, t);
}

template <typename REAL, typename STO, class A>
__device__ __forceinline__ void levee_bif_outflow_members(A a, long t, long member)
{
    const long member_offset = member * a.num_catchments;
    const long path_offset = member * a.num_bifurcation_paths;
    const long level_offset = path_offset * a.num_bifurcation_levels;
    a.bifurcation_outflow_ptr += level_offset;
    a.bifurcation_cross_section_depth_ptr += level_offset;
    a.river_depth_ptr += member_offset;
    a.protected_depth_ptr += member_offset;
    a.river_storage_ptr += member_offset;
    a.flood_storage_ptr += member_offset;
    a.protected_storage_ptr += member_offset;
    a.outgoing_storage_ptr += member_offset;
    if (a.batched_bifurcation_manning) a.bifurcation_manning_ptr += level_offset;
    if (a.batched_bifurcation_width) a.bifurcation_width_ptr += level_offset;
    if (a.batched_bifurcation_length) a.bifurcation_length_ptr += path_offset;
    if (a.batched_bifurcation_elevation) a.bifurcation_elevation_ptr += level_offset;
    if (a.batched_river_height) a.river_height_ptr += member_offset;
    if (a.batched_catchment_elevation) a.catchment_elevation_ptr += member_offset;
    levee_bif_outflow<REAL, STO>(a, t);
}
