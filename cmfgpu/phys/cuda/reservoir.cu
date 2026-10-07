// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// CUDA backend for the reservoir outflow kernel.
//
// One thread per reservoir.  Undoes the main outflow kernel's outgoing-storage
// scatter, recomputes the regulated release (4 storage regimes), re-applies the
// corrected scatter.  if/else-if regime chain evaluates only the matching branch
// (avoids log() of a negative argument).

#include "canonical.cuh"

template <typename REAL, typename STO, bool HAS_LEVEE>
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
    const REAL* __restrict__ runoff,
    STO* __restrict__ outgoing_storage, const REAL* __restrict__ time_step_ptr, long t)
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
    REAL adj = __ldg(adjustment_volume + t);
    REAL n_out = __ldg(effective_normal_outflow + t);
    REAL a_out = __ldg(adjustment_outflow + t);
    REAL fc_out = __ldg(flood_control_outflow + t);

    REAL ro;
    if (dam_volume <= cons) {
        ro = n_out * sqrt(dam_volume / cons);
    } else if (dam_volume <= adj) {
        REAL frac2 = (dam_volume - cons) / (adj - cons);
        ro = n_out + exp((REAL)3.0 * log(frac2)) * (a_out - n_out);
    } else if (dam_volume <= emerg) {
        REAL frac3 = (dam_volume - adj) / (emerg - adj);
        REAL tmp = a_out
            + exp((REAL)(float)CMF_RESERVOIR_RELEASE_EXPONENT * log(frac3)) * (fc_out - a_out);
        if (inflow >= fc_out) {
            REAL flood = n_out + ((dam_volume - cons) / (emerg - cons)) * (inflow - n_out);
            ro = fmax(flood, tmp);
        } else {
            ro = tmp;
        }
    } else {
        ro = (inflow >= fc_out) ? inflow : fc_out;
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
template <typename REAL, typename STO, bool HAS_LEVEE, class A>
__device__ __forceinline__ void reservoir_outflow(const A& a, long t)
{
    k_reservoir_outflow_cell<REAL, STO, HAS_LEVEE>(
        a.reservoir_catchment_idx_ptr, a.downstream_idx_ptr, a.reservoir_total_inflow_ptr,
        a.river_outflow_ptr, a.flood_outflow_ptr, a.river_storage_ptr, a.flood_storage_ptr,
        cmf_optional<HAS_LEVEE>(a.protected_storage_ptr), a.conservation_volume_ptr,
        a.emergency_volume_ptr, a.adjustment_volume_ptr, a.effective_normal_outflow_ptr,
        a.adjustment_outflow_ptr, a.flood_control_outflow_ptr, a.runoff_ptr,
        a.outgoing_storage_ptr, a.time_step_ptr, t);
}

// Ensemble members offset their slices, then run the body above.
template <typename REAL, typename STO, bool HAS_LEVEE, class A>
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
    if (a.batched_adjustment_volume) a.adjustment_volume_ptr += reservoir_offset;
    if (a.batched_effective_normal_outflow) a.effective_normal_outflow_ptr += reservoir_offset;
    if (a.batched_adjustment_outflow) a.adjustment_outflow_ptr += reservoir_offset;
    if (a.batched_flood_control_outflow) a.flood_control_outflow_ptr += reservoir_offset;
    reservoir_outflow<REAL, STO, HAS_LEVEE>(a, t);
}
