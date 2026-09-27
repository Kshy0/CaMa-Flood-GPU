// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// CUDA backend for the fused storage-update + flood-stage kernel.
//
// CUDA storage-update + flood-stage implementation.  Lane-local early exits are
// used for dry cells and completed flood-level scans.  There is no warp/group
// vote path here; divergent lanes are left to the CUDA compiler and SIMT
// scheduler.

#include "block_reduce.cuh"
#include "canonical.cuh"

// Number of global water-balance counters the LOG path accumulates.
#define CMF_LOG_SUMS 11

template <typename REAL, typename STO, bool LOG>
__device__ __forceinline__ void k_flood_stage_cell(
    const STO* __restrict__ river_inflow, const STO* __restrict__ flood_inflow,
    const REAL* __restrict__ river_outflow, const REAL* __restrict__ flood_outflow,
    const STO* __restrict__ global_bif_outflow, const REAL* __restrict__ runoff,
    const REAL* __restrict__ inflow, const int* __restrict__ catchment_inflow_idx,
    const REAL* __restrict__ time_step_ptr,
    STO* __restrict__ outgoing_storage,
    STO* __restrict__ river_storage, STO* __restrict__ flood_storage,
    STO* __restrict__ protected_storage,
    STO* __restrict__ total_storage_output,
    REAL* __restrict__ river_depth, REAL* __restrict__ flood_depth,
    REAL* __restrict__ protected_depth, REAL* __restrict__ flood_fraction,
    const REAL* __restrict__ river_height, const REAL* __restrict__ flood_depth_table,
    const REAL* __restrict__ catchment_area, const REAL* __restrict__ river_width,
    const REAL* __restrict__ river_length,
    const bool* __restrict__ is_levee,
    REAL* __restrict__ total_storage_pre_sum,
    REAL* __restrict__ total_storage_next_sum,
    REAL* __restrict__ total_storage_new_sum,
    REAL* __restrict__ total_inflow_sum,
    REAL* __restrict__ total_outflow_sum,
    REAL* __restrict__ total_storage_stage_sum,
    REAL* __restrict__ river_storage_sum,
    REAL* __restrict__ flood_storage_sum,
    REAL* __restrict__ flood_area_sum,
    REAL* __restrict__ total_inflow_error_sum,
    REAL* __restrict__ total_stage_error_sum,
    const int* __restrict__ current_step_ptr,
    long num_catchments, int num_flood_levels,
    int has_bifurcation, int has_inflow, int has_levee,
    int has_total_storage_output, long t)
{
    REAL log_sum[CMF_LOG_SUMS];
    if constexpr (LOG) {
        // Every thread must reach the block reduction, so out-of-range lanes
        // stay alive with zero contributions instead of returning early.
#pragma unroll
        for (int i = 0; i < CMF_LOG_SUMS; ++i) log_sum[i] = (REAL)0;
    } else {
        if (t >= num_catchments) return;
    }
    int current_step = LOG ? __ldg(current_step_ptr) : 0;

    if (t < num_catchments) {

        REAL ts = __ldg(time_step_ptr);
        STO rsto = river_storage[t];
        STO fsto = flood_storage[t];
        STO prot = has_levee ? protected_storage[t] : (STO)0;
        REAL rinf = (REAL)river_inflow[t];
        REAL finf = (REAL)flood_inflow[t];
        REAL gbif = has_bifurcation ? (REAL)global_bif_outflow[t] : (REAL)0;
        REAL rout = __ldg(river_outflow + t);
        REAL fout = __ldg(flood_outflow + t);
        REAL ro = __ldg(runoff + t);
        REAL prescribed_inflow = (REAL)0;
        if (has_inflow) {
            int inflow_idx = __ldg(catchment_inflow_idx + t);
            if (inflow_idx >= 0) prescribed_inflow = __ldg(inflow + inflow_idx);
        }

        bool non_levee = true;
        if constexpr (LOG) {
            non_levee = !has_levee || !is_levee[t];
        }
        STO total_stage_pre = rsto + fsto + prot;
        if constexpr (LOG) {
            log_sum[0] += (REAL)total_stage_pre * (REAL)1e-9;
        }

        STO river_su = rsto + (STO)(rinf * ts) - (STO)(rout * ts);
        STO flood_su = fsto;
        if (river_su < (STO)0) {
            flood_su += river_su;
            river_su = (STO)0;
        }
        flood_su = flood_su + (STO)(finf * ts) - (STO)(fout * ts) - (STO)(gbif * ts);
        if (flood_su < (STO)0) {
            STO tt = river_su + flood_su;
            river_su = tt > (STO)0 ? tt : (STO)0;
            flood_su = (STO)0;
        }
        // Runoff splits by the previous stage's flood fraction; prescribed
        // inflow (LUPSINF) joins the river part.  Negative runoff may leave a
        // negative storage; only the depth is clamped.
        REAL frac = flood_fraction[t];
        REAL river_runoff = ro * ((REAL)1 - frac) * ts;
        if (has_inflow) river_runoff += prescribed_inflow * ts;
        river_su += (STO)river_runoff;
        flood_su += (STO)(ro * frac * ts);
        STO total_next = river_su + flood_su + prot;
        if constexpr (LOG) {
            log_sum[1] += (REAL)total_next * (REAL)1e-9;
            log_sum[2] += (rinf + finf + prescribed_inflow) * ts * (REAL)1e-9;
            log_sum[3] += (rout + fout) * ts * (REAL)1e-9;
            REAL balance = (REAL)total_stage_pre - (REAL)total_next
                + (rinf + finf + ro + prescribed_inflow - rout - fout - gbif) * ts;
            log_sum[4] += balance * (REAL)1e-9;
        }
        STO total_s = total_next;
        if (has_total_storage_output) total_storage_output[t] = total_s;
        REAL total_storage = (REAL)total_s;
        if constexpr (LOG) {
            log_sum[5] += total_storage * (REAL)1e-9;
        }

        REAL rh = __ldg(river_height + t);
        REAL rw = __ldg(river_width + t);
        REAL rl = __ldg(river_length + t);
        REAL river_max_storage = rl * rw * rh;

        bool wet = has_levee ? total_storage > river_max_storage
                             : total_s > (STO)river_max_storage;
        if (!wet) {
            REAL river_depth_dry = fmax(total_storage / rl / rw, (REAL)0);
            if constexpr (LOG) {
                if (non_levee) {
                    log_sum[6] += total_storage * (REAL)1e-9;
                    log_sum[7] += total_storage * (REAL)1e-9;
                }
            }
            outgoing_storage[t] = (STO)0;
            river_storage[t] = total_s;
            flood_storage[t] = (STO)0;
            if (has_levee) protected_storage[t] = (STO)0;
            river_depth[t] = river_depth_dry;
            flood_depth[t] = (REAL)0;
            if (has_levee) protected_depth[t] = (REAL)0;
            flood_fraction[t] = (REAL)0;
        } else {

        REAL ca = __ldg(catchment_area + t);
        REAL catchment_width = ca / rl;
        REAL width_increment = catchment_width / num_flood_levels;
        int level = 0;
        REAL S_accum = river_max_storage;
        REAL prev_H = (REAL)0;
        REAL prev_W = rw;
        REAL prev_total_storage = river_max_storage;
        REAL prev_flood_depth = (REAL)0;
        REAL next_flood_depth = (REAL)0;

        for (int i = 0; i < num_flood_levels; ++i) {
            REAL H_curr = __ldg(flood_depth_table + t * num_flood_levels + i);
            REAL W_curr = rw + (i + 1) * width_increment;
            REAL dS = rl * (REAL)0.5 * (prev_W + W_curr) * (H_curr - prev_H);
            REAL S_curr = S_accum + dS;
            next_flood_depth = H_curr;
            bool is_above = has_levee ? total_storage > S_curr : total_s > (STO)S_curr;
            if (is_above) {
                level += 1;
                prev_total_storage = S_curr;
                prev_flood_depth = H_curr;
                S_accum = S_curr;
                prev_H = H_curr;
                prev_W = W_curr;
            } else {
                break;
            }
        }

        REAL prev_total_width = rw + level * width_increment;
        REAL diff_width = (REAL)0;
        REAL fdep;
        if (level == num_flood_levels) {
            fdep = prev_flood_depth + (total_storage - prev_total_storage) / (prev_total_width * rl);
        } else {
            REAL flood_grad = (next_flood_depth - prev_flood_depth) / width_increment;
            diff_width = sqrt(prev_total_width * prev_total_width
                + (REAL)2 * (total_storage - prev_total_storage) / (flood_grad * rl)) - prev_total_width;
            fdep = prev_flood_depth + diff_width * flood_grad;
        }

        REAL cap = river_max_storage + rl * rw * fdep;
        STO cap_s = (STO)cap;
        STO river_storage_final = (has_levee || cap_s < total_s) ? cap_s : total_s;
        REAL rdep = (REAL)river_storage_final / rl / rw;
        REAL ff_mid = (prev_total_width + diff_width - rw) * rl / ca;
        ff_mid = ff_mid < (REAL)0 ? (REAL)0 : (ff_mid > (REAL)1 ? (REAL)1 : ff_mid);
        REAL ffr = (level == num_flood_levels) ? (REAL)1 : ff_mid;
        STO flood_storage_final = total_s - river_storage_final;
        if (flood_storage_final < (STO)0) flood_storage_final = (STO)0;

        if constexpr (LOG) {
            STO total_stage_new = river_storage_final + flood_storage_final;
            if (non_levee) {
                log_sum[6] += (REAL)total_stage_new * (REAL)1e-9;
                log_sum[10] += (REAL)(total_stage_new - total_s) * (REAL)1e-9;
                log_sum[7] += (REAL)river_storage_final * (REAL)1e-9;
                log_sum[8] += (REAL)flood_storage_final * (REAL)1e-9;
                log_sum[9] += ffr * ca * (REAL)1e-9;
            }
        }

        outgoing_storage[t] = (STO)0;
        river_storage[t] = river_storage_final;
        flood_storage[t] = flood_storage_final;
        if (has_levee) protected_storage[t] = (STO)0;
        river_depth[t] = rdep;
        flood_depth[t] = fdep;
        if (has_levee) protected_depth[t] = (REAL)0;
        flood_fraction[t] = ffr;

        }  // wet branch
    }  // active lane

    if constexpr (LOG) {
        REAL* const destination[CMF_LOG_SUMS] = {
            total_storage_pre_sum, total_storage_next_sum, total_inflow_sum,
            total_outflow_sum, total_inflow_error_sum, total_storage_new_sum,
            total_storage_stage_sum, river_storage_sum, flood_storage_sum,
            flood_area_sum, total_stage_error_sum,
        };
        cmf_block_atomic_add<REAL, CMF_LOG_SUMS>(
            log_sum, destination, current_step);
    }
}

// Generated-entry body over the canonical values ``a`` of compute_flood_stage,
// or of compute_flood_stage_log with LOG.
template <typename REAL, typename STO, bool HAS_BIFURCATION, bool HAS_INFLOW,
          bool HAS_LEVEE, bool HAS_TOTAL_STORAGE_OUTPUT, bool LOG = false, class A>
__device__ __forceinline__ void flood_stage(const A& a, long t)
{
    const auto balance = cmf_balance<LOG>(a);
    k_flood_stage_cell<REAL, STO, LOG>(
        a.river_inflow_ptr, a.flood_inflow_ptr, a.river_outflow_ptr, a.flood_outflow_ptr,
        cmf_optional<HAS_BIFURCATION>(a.global_bifurcation_outflow_ptr), a.runoff_ptr,
        cmf_optional<HAS_INFLOW>(a.inflow_ptr),
        cmf_optional<HAS_INFLOW>(a.catchment_inflow_idx_ptr), a.time_step_ptr,
        a.outgoing_storage_ptr, a.river_storage_ptr, a.flood_storage_ptr,
        cmf_optional<HAS_LEVEE>(a.protected_storage_ptr),
        cmf_optional<HAS_TOTAL_STORAGE_OUTPUT>(a.total_storage_ptr), a.river_depth_ptr,
        a.flood_depth_ptr, cmf_optional<HAS_LEVEE>(a.protected_depth_ptr),
        a.flood_fraction_ptr, a.river_height_ptr, a.flood_depth_table_ptr,
        a.catchment_area_ptr, a.river_width_ptr, a.river_length_ptr,
        cmf_optional<HAS_LEVEE>(balance.is_levee_ptr),
        balance.total_storage_pre_sum_ptr, balance.total_storage_next_sum_ptr,
        balance.total_storage_new_sum_ptr, balance.total_inflow_sum_ptr,
        balance.total_outflow_sum_ptr, balance.total_storage_stage_sum_ptr,
        balance.river_storage_sum_ptr, balance.flood_storage_sum_ptr,
        balance.flood_area_sum_ptr, balance.total_inflow_error_sum_ptr,
        balance.total_stage_error_sum_ptr, balance.current_step_ptr, a.num_catchments,
        a.num_flood_levels, HAS_BIFURCATION, HAS_INFLOW, HAS_LEVEE,
        HAS_TOTAL_STORAGE_OUTPUT, t);
}

// Ensemble members offset their slices, then run the body above.
template <typename REAL, typename STO, bool HAS_BIFURCATION, bool HAS_INFLOW,
          bool HAS_LEVEE, bool HAS_TOTAL_STORAGE_OUTPUT, class A>
__device__ __forceinline__ void flood_stage_members(A a, long t, long member)
{
    const long member_offset = member * a.num_catchments;
    a.river_inflow_ptr += member_offset;
    a.flood_inflow_ptr += member_offset;
    a.river_outflow_ptr += member_offset;
    a.flood_outflow_ptr += member_offset;
    a.outgoing_storage_ptr += member_offset;
    a.river_storage_ptr += member_offset;
    a.flood_storage_ptr += member_offset;
    a.river_depth_ptr += member_offset;
    a.flood_depth_ptr += member_offset;
    a.flood_fraction_ptr += member_offset;
    if constexpr (HAS_BIFURCATION) a.global_bifurcation_outflow_ptr += member_offset;
    if constexpr (HAS_LEVEE) {
        a.protected_storage_ptr += member_offset;
        a.protected_depth_ptr += member_offset;
    }
    if constexpr (HAS_TOTAL_STORAGE_OUTPUT) a.total_storage_ptr += member_offset;
    if (a.batched_runoff) a.runoff_ptr += member_offset;
    if constexpr (HAS_INFLOW) {
        if (a.batched_inflow) a.inflow_ptr += member * a.num_inflow_gauges;
    }
    if (a.batched_river_height) a.river_height_ptr += member_offset;
    if (a.batched_flood_depth_table)
        a.flood_depth_table_ptr += member_offset * a.num_flood_levels;
    if (a.batched_catchment_area) a.catchment_area_ptr += member_offset;
    if (a.batched_river_width) a.river_width_ptr += member_offset;
    if (a.batched_river_length) a.river_length_ptr += member_offset;
    flood_stage<REAL, STO, HAS_BIFURCATION, HAS_INFLOW, HAS_LEVEE,
                HAS_TOTAL_STORAGE_OUTPUT>(a, t);
}
