// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//

// Per-sub-step bookkeeping for the device-side uniform sub-step march: CaMa
// fixes num_sub_steps per interval, so the body advances current_step (log
// indexing), bumps the 1-based counter, and writes continue_flag (counter <
// num_sub_steps).  Runs at the START of the body, before do_one_sub_step.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void k_cama_march_step(const int* __restrict__ num_sub_steps,
        int* __restrict__ counter, int* __restrict__ current_step,
        int* __restrict__ continue_flag) {
    int c = *counter;            // 0-based index of the sub-step about to run
    *current_step = c;
    int next = c + 1;
    *counter = next;
    *continue_flag = (next < *num_sub_steps) ? 1 : 0;
}

void launch_march_step(at::Tensor num_sub_steps, at::Tensor counter,
                       at::Tensor current_step, at::Tensor continue_flag, long stream) {
    k_cama_march_step<<<1, 1, 0, (cudaStream_t)stream>>>(
        num_sub_steps.data_ptr<int>(), counter.data_ptr<int>(),
        current_step.data_ptr<int>(), continue_flag.data_ptr<int>());
}
