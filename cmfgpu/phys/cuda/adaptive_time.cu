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

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "block_reduce.cuh"

template <typename REAL>
__global__ void k_adaptive_time(
    const REAL* __restrict__ river_depth,
    const REAL* __restrict__ downstream_distance,
    const bool*  __restrict__ is_dam_related,
    int* __restrict__ max_sub_steps,
    REAL outer_time_step, REAL adaptive_time_factor, REAL gravity,
    long num_catchments, int has_reservoir)
{
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    // 0 is the identity for this reduction: every contributing cell yields
    // n_steps >= 1, so non-contributing lanes cannot raise the block maximum.
    int n_steps = 0;

    bool skip = (t >= num_catchments)
        || (has_reservoir && is_dam_related && is_dam_related[t]);
    if (!skip) {
        REAL dist = __ldg(downstream_distance + t);
        REAL raw_depth = __ldg(river_depth + t);
        REAL depth = fmax(raw_depth, (REAL)0.01);
        REAL dt = adaptive_time_factor * dist / sqrt(gravity * depth);
        REAL dt_clamped = fmin(dt, outer_time_step);
        n_steps = (int)(
            floor(outer_time_step / dt_clamped - (REAL)0.01) + (REAL)1.0);
    }

    // Reduce inside the block so the single global scalar sees one atomic per
    // block instead of one per catchment.
    cmf_block_atomic_max(n_steps, max_sub_steps);
}

void launch_adaptive_time(
    at::Tensor river_depth_ptr, at::Tensor downstream_distance_ptr,
    c10::optional<at::Tensor> is_dam_related_ptr,
    at::Tensor max_sub_steps_ptr,
    float outer_time_step, float adaptive_time_factor, float gravity,
    long num_catchments, bool HAS_RESERVOIR, long BLOCK_SIZE)
{
    int grid = (int)((num_catchments + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const bool* dam = (
        is_dam_related_ptr ? is_dam_related_ptr->data_ptr<bool>() : nullptr);
    if (river_depth_ptr.scalar_type() == at::kDouble) {
        k_adaptive_time<double><<<grid, (int)BLOCK_SIZE, 0, stream>>>(
            river_depth_ptr.data_ptr<double>(),
            downstream_distance_ptr.data_ptr<double>(),
            dam, max_sub_steps_ptr.data_ptr<int>(),
            (double)outer_time_step, (double)adaptive_time_factor,
            (double)gravity, num_catchments, (int)HAS_RESERVOIR);
    } else {
        k_adaptive_time<float><<<grid, (int)BLOCK_SIZE, 0, stream>>>(
            river_depth_ptr.data_ptr<float>(),
            downstream_distance_ptr.data_ptr<float>(),
            dam, max_sub_steps_ptr.data_ptr<int>(),
            outer_time_step, adaptive_time_factor, gravity,
            num_catchments, (int)HAS_RESERVOIR);
    }
}
