// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// CUDA backend for the reservoir outflow kernel.
//
// One thread per reservoir.  Undoes the main outflow kernel's outgoing-storage
// scatter, recomputes the regulated release (4 storage regimes of the
// reservoir's scheme), re-applies the corrected scatter.  if/else-if regime
// chains evaluate only the matching branch (avoids log() of a negative argument).

#include "canonical.cuh"

// Yamazaki & Funato (CaMa-Flood v4 default): water use up to the conservation
// volume, water excess up to the adjustment volume, flood control up to the
// emergency volume, then emergency.
template <typename REAL>
__device__ __forceinline__ REAL yamazaki_funato_release(
    REAL dam_volume, REAL inflow, REAL cons, REAL adj, REAL emerg,
    REAL n_out, REAL a_out, REAL fc_out, REAL release_exponent)
{
    if (dam_volume <= cons) return n_out * sqrt(dam_volume / cons);
    if (dam_volume <= adj) {
        REAL frac2 = (dam_volume - cons) / (adj - cons);
        return n_out + exp((REAL)3.0 * log(frac2)) * (a_out - n_out);
    }
    if (dam_volume <= emerg) {
        REAL frac3 = (dam_volume - adj) / (emerg - adj);
        REAL tmp = a_out
            + exp(release_exponent * log(frac3)) * (fc_out - a_out);
        if (inflow >= fc_out) {
            REAL flood = n_out + ((dam_volume - cons) / (emerg - cons)) * (inflow - n_out);
            return fmax(flood, tmp);
        }
        return tmp;
    }
    return (inflow >= fc_out) ? inflow : fc_out;
}

// Hanazaki et al. (2022), CaMa-Flood v4 LDAMH22: water use below the normal
// volume, water supply up to the conservation volume, flood control up to the
// emergency volume (passing a share of the inflow above Qf), then emergency.
template <typename REAL>
__device__ __forceinline__ REAL h22_release(
    REAL dam_volume, REAL inflow, REAL nor, REAL cons, REAL emerg,
    REAL qn, REAL qf, REAL share)
{
    if (dam_volume <= nor) return qn * (dam_volume / cons);
    if (dam_volume <= cons) {
        if (qf <= inflow)
            return qn * (REAL)0.5 + (dam_volume - nor) / (cons - nor) * (qf - qn);
        REAL frac = (dam_volume - nor) / (emerg - nor);
        return qn * (REAL)0.5 + frac * frac * (qf - qn);
    }
    if (dam_volume < emerg) {
        if (qf <= inflow)
            return qf + share * (dam_volume - cons) / (emerg - cons) * (inflow - qf);
        REAL frac = (dam_volume - nor) / (emerg - nor);
        return qn * (REAL)0.5 + frac * frac * (qf - qn);
    }
    return fmax(inflow, qf);
}

template <typename REAL, typename STO, bool HAS_LEVEE, bool HAS_H22>
__device__ __forceinline__ void k_reservoir_outflow_cell(
    const int* __restrict__ reservoir_catchment_idx,
    const int* __restrict__ downstream_idx,
    STO* __restrict__ reservoir_total_inflow,
    REAL* __restrict__ river_outflow, REAL* __restrict__ flood_outflow,
    const STO* __restrict__ river_storage, const STO* __restrict__ flood_storage,
    const STO* __restrict__ protected_storage,
    const REAL* __restrict__ conservation_volume, const REAL* __restrict__ emergency_volume,
    const REAL* __restrict__ adjustment_volume, const REAL* __restrict__ effective_normal_outflow,
    const REAL* __restrict__ adjustment_outflow, const REAL* __restrict__ flood_control_outflow,
    const REAL* __restrict__ normal_volume,
    const REAL* __restrict__ flood_inflow_share,
    const REAL* __restrict__ runoff,
    STO* __restrict__ outgoing_storage, const REAL* __restrict__ time_step_ptr, REAL release_exponent, long t)
{
    REAL time_step = __ldg(time_step_ptr);

    int ci = reservoir_catchment_idx[t];
    int di = downstream_idx[ci];
    bool is_river_mouth = (di == ci);

    // Undo exactly the outflow kernel's outgoing flows of this cell.
    REAL old_r = river_outflow[ci];
    REAL old_f = flood_outflow[ci];
    cmf_atomic_add(outgoing_storage + ci,
                   -(STO)(fmax(old_r, (REAL)0.0) + fmax(old_f, (REAL)0.0)));
    if (!is_river_mouth)
        cmf_atomic_add(outgoing_storage + di,
                       -((STO)fmax(-old_r, (REAL)0.0) + (STO)fmax(-old_f, (REAL)0.0)));

    STO storage = river_storage[ci] + flood_storage[ci];
    REAL river_flood_storage = (REAL)storage;
    if constexpr (HAS_LEVEE) storage += protected_storage[ci];
    REAL dam_volume = (REAL)storage;

    REAL inflow = (REAL)(reservoir_total_inflow[ci] + (STO)__ldg(runoff + ci));
    reservoir_total_inflow[ci] = (STO)0;

    REAL cons = __ldg(conservation_volume + t);
    REAL emerg = __ldg(emergency_volume + t);
    REAL fc_out = __ldg(flood_control_outflow + t);

    REAL ro;
    if constexpr (HAS_H22) {
        ro = h22_release<REAL>(dam_volume, inflow, __ldg(normal_volume + t), cons, emerg,
                               __ldg(effective_normal_outflow + t), fc_out,
                               __ldg(flood_inflow_share + t));
    } else {
        ro = yamazaki_funato_release<REAL>(
            dam_volume, inflow, cons, __ldg(adjustment_volume + t), emerg,
            __ldg(effective_normal_outflow + t), __ldg(adjustment_outflow + t), fc_out, release_exponent);
    }
    // Flow limiter: the minimum first, so a negative storage releases nothing.
    ro = fmin(fmin(ro, dam_volume / time_step), river_flood_storage / time_step);
    ro = fmax(ro, (REAL)0.0);

    river_outflow[ci] = ro;
    flood_outflow[ci] = (REAL)0.0;

    // The release is non-negative, so it only leaves this cell.
    cmf_atomic_add(outgoing_storage + ci, (STO)ro);
}

// Generated-entry body over the canonical values ``a`` of
// compute_reservoir_outflow.
template <typename REAL, typename STO, bool HAS_LEVEE, bool HAS_H22, class A>
__device__ __forceinline__ void reservoir_outflow(const A& a, long t)
{
    k_reservoir_outflow_cell<REAL, STO, HAS_LEVEE, HAS_H22>(
        a.reservoir_catchment_idx_ptr, a.downstream_idx_ptr, a.reservoir_total_inflow_ptr,
        a.river_outflow_ptr, a.flood_outflow_ptr, a.river_storage_ptr, a.flood_storage_ptr,
        cmf_optional<HAS_LEVEE>(a.protected_storage_ptr), a.conservation_volume_ptr,
        a.emergency_volume_ptr, cmf_optional<!HAS_H22>(a.adjustment_volume_ptr), a.effective_normal_outflow_ptr,
        cmf_optional<!HAS_H22>(a.adjustment_outflow_ptr), a.flood_control_outflow_ptr,
        cmf_optional<HAS_H22>(a.normal_volume_ptr),
        cmf_optional<HAS_H22>(a.flood_inflow_share_ptr), a.runoff_ptr,
        a.outgoing_storage_ptr, a.time_step_ptr, REAL(a.release_exponent), t);
}

// Ensemble members offset their state and any member-specific parameters,
// then run the same reservoir body.
template <typename REAL, typename STO, bool HAS_LEVEE, bool HAS_H22, class A>
__device__ __forceinline__ void reservoir_outflow_members(A a, long t, long member)
{
    const long member_offset = member * a.num_catchments;
    a.reservoir_total_inflow_ptr += member_offset;
    a.river_outflow_ptr += member_offset;
    a.flood_outflow_ptr += member_offset;
    a.river_storage_ptr += member_offset;
    a.flood_storage_ptr += member_offset;
    if constexpr (HAS_LEVEE) a.protected_storage_ptr += member_offset;
    a.outgoing_storage_ptr += member_offset;
    if (a.batched_runoff) a.runoff_ptr += member_offset;
    const long reservoir_offset = member * a.num_reservoirs;
    if (a.batched_conservation_volume) a.conservation_volume_ptr += reservoir_offset;
    if (a.batched_emergency_volume) a.emergency_volume_ptr += reservoir_offset;
    if (a.batched_effective_normal_outflow) a.effective_normal_outflow_ptr += reservoir_offset;
    if (a.batched_flood_control_outflow) a.flood_control_outflow_ptr += reservoir_offset;
    if constexpr (HAS_H22) {
        if (a.batched_normal_volume) a.normal_volume_ptr += reservoir_offset;
        if (a.batched_flood_inflow_share) a.flood_inflow_share_ptr += reservoir_offset;
    } else {
        if (a.batched_adjustment_volume) a.adjustment_volume_ptr += reservoir_offset;
        if (a.batched_adjustment_outflow) a.adjustment_outflow_ptr += reservoir_offset;
    }
    reservoir_outflow<REAL, STO, HAS_LEVEE, HAS_H22>(a, t);
}

// Generated-entry body over the canonical values ``a`` of fill_dam_storage
// (CMF_DAMOUT_INIT): a dam below its conservation volume starts full to it,
// with an empty floodplain.
template <typename REAL, typename STO, class A>
__device__ __forceinline__ void dam_fill(const A& a, long t)
{
    const int ci = a.reservoir_catchment_idx_ptr[t];
    const STO conservation = (STO)__ldg(a.conservation_volume_ptr + t);
    if (a.river_storage_ptr[ci] + a.flood_storage_ptr[ci] < conservation) {
        a.river_storage_ptr[ci] = conservation;
        a.flood_storage_ptr[ci] = (STO)0;
    }
}

template <typename REAL, typename STO, class A>
__device__ __forceinline__ void dam_fill_members(A a, long t, long member)
{
    a.river_storage_ptr += member * a.num_catchments;
    a.flood_storage_ptr += member * a.num_catchments;
    if (a.batched_conservation_volume)
        a.conservation_volume_ptr += member * a.num_reservoirs;
    dam_fill<REAL, STO>(a, t);
}
