// CaMa-Flood-GPU — Adaptive time step kernel + launcher
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0

#include "common.cuh"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// =========================================================================
// Kernel: compute_adaptive_time_step  (no storage_t — uses float only)
//   CMF_GRAVITY, HAS_RESERVOIR_CONST are compile-time constants.
// =========================================================================
__global__ void compute_adaptive_time_step_kernel(
    const float*  __restrict__ river_depth,
    const float*  __restrict__ downstream_distance,
    const bool*   __restrict__ is_dam_related,
    int*          __restrict__ max_sub_steps,
    float time_step, float adaptive_time_factor,
    int num_catchments
) {
    extern __shared__ float sdata[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float dt = time_step;
    if (i < num_catchments) {
#if HAS_RESERVOIR_CONST
        bool skip = is_dam_related != nullptr && is_dam_related[i];
#else
        bool skip = false;
#endif
        if (!skip) {
            float d = fmaxf(river_depth[i], 0.01f);
            float dd = downstream_distance[i];
            dt = fminf(adaptive_time_factor * dd / sqrtf(CMF_GRAVITY * d), time_step);
        }
    }
    sdata[tid] = dt;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) {
        float min_dt = sdata[0];
        int n = (int)floorf(time_step / min_dt + 0.49f) + 1;
        atomicMax(max_sub_steps, n);
    }
}

// =========================================================================
// Launcher
// =========================================================================
void launch_adaptive_time_step(
    torch::Tensor river_depth, torch::Tensor downstream_distance,
    torch::Tensor is_dam_related, torch::Tensor max_sub_steps,
    float time_step, float adaptive_time_factor,
    int num_catchments
) {
    const int bs = CMF_BLOCK_SIZE;
    const int grid = cdiv(num_catchments, bs);
    const auto stream = at::cuda::getCurrentCUDAStream();
    compute_adaptive_time_step_kernel<<<grid, bs, bs * sizeof(float), stream>>>(
        river_depth.data_ptr<float>(), downstream_distance.data_ptr<float>(),
        is_dam_related.data_ptr<bool>(), max_sub_steps.data_ptr<int>(),
        time_step, adaptive_time_factor, num_catchments);
}
