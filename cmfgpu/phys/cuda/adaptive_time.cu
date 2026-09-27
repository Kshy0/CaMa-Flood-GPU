// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// CUDA backend for the adaptive-time-step (CFL) kernel.
//
// The CFL sub-step count is monotonic with respect to the per-cell dt:
//   n(dt) = floor(outer_time_step/dt - 0.01) + 1
// decreases as dt increases, so a per-thread atomicMax over n_i gives the
// global maximum sub-step count without a separate reduction.

#include "block_reduce.cuh"
#include "canonical.cuh"

template <typename REAL>
__device__ __forceinline__ void k_adaptive_time_cell(
    const REAL* __restrict__ river_depth,
    const REAL* __restrict__ downstream_distance,
    const bool*  __restrict__ is_dam_related,
    int* __restrict__ max_sub_steps,
    const REAL* __restrict__ outer_time_step,
    REAL adaptive_time_factor, REAL gravity,
    long num_catchments, int has_reservoir, long t)
{
    int n_steps = 1;

    bool skip = (t >= num_catchments)
        || (has_reservoir && is_dam_related && is_dam_related[t]);
    if (!skip) {
        REAL dist = __ldg(downstream_distance + t);
        REAL raw_depth = __ldg(river_depth + t);
        REAL depth = fmax(raw_depth, (REAL)0.01);
        REAL dt = adaptive_time_factor * dist / sqrt(gravity * depth);
        REAL outer_dt = *outer_time_step;
        REAL dt_clamped = fmin(dt, outer_dt);
        n_steps = (int)(
            floor(outer_dt / dt_clamped - (REAL)0.01) + (REAL)1.0);
    }

    // Reduce inside the block so the single global scalar sees one atomic per
    // block instead of one per catchment.
    cmf_block_atomic_max(n_steps, max_sub_steps);
}

// Generated-entry body over the canonical values ``a`` of
// compute_adaptive_time_step; every thread reaches the block reduction.
template <typename REAL, bool HAS_RESERVOIR, class A>
__device__ __forceinline__ void adaptive_time(const A& a, long t)
{
    k_adaptive_time_cell<REAL>(
        a.river_depth_ptr, a.downstream_distance_ptr,
        cmf_optional<HAS_RESERVOIR>(a.is_dam_related_ptr),
        a.max_sub_steps_ptr, a.outer_time_step_ptr, a.adaptive_time_factor, a.gravity,
        a.num_catchments, HAS_RESERVOIR, t);
}

// Ensemble members offset their slices, then run the body above.
template <typename REAL, bool HAS_RESERVOIR, class A>
__device__ __forceinline__ void adaptive_time_members(A a, long t, long member)
{
    const long member_offset = member * a.num_catchments;
    a.river_depth_ptr += member_offset;
    if (a.batched_downstream_distance) a.downstream_distance_ptr += member_offset;
    adaptive_time<REAL, HAS_RESERVOIR>(a, t);
}
