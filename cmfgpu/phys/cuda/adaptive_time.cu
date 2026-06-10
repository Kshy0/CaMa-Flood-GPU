// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// CUDA backend for the adaptive-time-step (CFL) kernel.
//
// The CFL sub-step count is monotonic with respect to the per-cell dt:
//   n(dt) = floor(time_step/dt + 0.49) + 1
// decreases as dt increases, so a per-thread atomicMax over n_i gives the
// global maximum sub-step count without a separate reduction.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

__global__ void k_adaptive_time(
    const float* __restrict__ river_depth,
    const float* __restrict__ downstream_distance,
    const bool*  __restrict__ is_dam_related,
    int* __restrict__ max_sub_steps,
    float time_step, float adaptive_time_factor, float gravity,
    long num_catchments, int has_reservoir)
{
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t >= num_catchments) return;
    // Skip dam and upstream-of-dam cells from the CFL calculation.
    if (has_reservoir && is_dam_related && is_dam_related[t]) return;

    float dist = __ldg(downstream_distance + t);
    float depth = fmaxf(__ldg(river_depth + t), 0.01f);
    float dt = adaptive_time_factor * dist / sqrtf(gravity * depth);
    float dt_clamped = fminf(dt, time_step);
    float n_steps_f = floorf(time_step / dt_clamped + 0.49f) + 1.0f;
    int n_steps = (int)n_steps_f;
    atomicMax(max_sub_steps, n_steps);
}

void launch_adaptive_time(
    at::Tensor river_depth, at::Tensor downstream_distance,
    at::Tensor is_dam_related, at::Tensor max_sub_steps,
    double time_step, double adaptive_time_factor, double gravity,
    long num_catchments, long has_reservoir, long block)
{
    int grid = (int)((num_catchments + block - 1) / block);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const bool* dam = is_dam_related.numel() ? is_dam_related.data_ptr<bool>() : nullptr;
    k_adaptive_time<<<grid, (int)block, 0, stream>>>(
        river_depth.data_ptr<float>(), downstream_distance.data_ptr<float>(),
        dam, max_sub_steps.data_ptr<int>(),
        (float)time_step, (float)adaptive_time_factor, (float)gravity,
        num_catchments, (int)has_reservoir);
}
