// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// HIP backend for the reservoir outflow kernel.
//
// Mirrors cmfgpu/phys/triton/reservoir.py::compute_reservoir_outflow_kernel.
// One thread per reservoir.  Undoes the main outflow kernel's outgoing-storage
// scatter, recomputes the regulated release (4 storage regimes), re-applies the
// corrected scatter.  if/else-if regime chain evaluates only the matching branch
// (avoids log() of a negative argument).  Templated on STO (hpfloat).  No
// -ffast-math (sqrt/exp/log parity).

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

template <typename STO>
__global__ void k_reservoir_outflow(
    const int* __restrict__ reservoir_catchment_idx,
    const int* __restrict__ downstream_idx,
    STO* __restrict__ reservoir_total_inflow,
    float* __restrict__ river_outflow, float* __restrict__ flood_outflow,
    const STO* __restrict__ river_storage, const STO* __restrict__ flood_storage,
    const float* __restrict__ conservation_volume, const float* __restrict__ emergency_volume,
    const float* __restrict__ adjustment_volume, const float* __restrict__ normal_outflow,
    const float* __restrict__ adjustment_outflow, const float* __restrict__ flood_control_outflow,
    const float* __restrict__ runoff, STO* __restrict__ total_storage,
    STO* __restrict__ outgoing_storage, const float* __restrict__ time_step_ptr,
    long num_reservoirs)
{
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t >= num_reservoirs) return;
    float time_step = __ldg(time_step_ptr);

    int ci = reservoir_catchment_idx[t];
    int di = downstream_idx[ci];
    bool is_river_mouth = (di == ci);

    float old_r = river_outflow[ci];
    float old_f = flood_outflow[ci];
    float old_pos = fmaxf(old_r, 0.0f) + fmaxf(old_f, 0.0f);
    float old_neg = fminf(old_r, 0.0f) + fminf(old_f, 0.0f);
    atomicAdd(outgoing_storage + ci, (STO)(-(old_pos * time_step)));
    STO undo_dn = is_river_mouth ? (STO)0 : (STO)(old_neg * time_step);
    atomicAdd(outgoing_storage + di, undo_dn);

    float rs = (float)river_storage[ci];
    float fs = (float)flood_storage[ci];
    float total = rs + fs;

    float total_inflow = (float)reservoir_total_inflow[ci];
    reservoir_total_inflow[ci] = (STO)0;
    float runoff_v = __ldg(runoff + ci);
    float inflow = total_inflow + runoff_v;

    float cons = __ldg(conservation_volume + t);
    float emerg = __ldg(emergency_volume + t);
    float adj = __ldg(adjustment_volume + t);
    float n_out = __ldg(normal_outflow + t);
    float a_out = __ldg(adjustment_outflow + t);
    float fc_out = __ldg(flood_control_outflow + t);

    float ro;
    if (total <= cons) {
        ro = n_out * sqrtf(total / cons);
    } else if (total <= adj) {
        float frac2 = (total - cons) / (adj - cons);
        ro = n_out + expf(3.0f * logf(frac2)) * (a_out - n_out);
    } else if (total <= emerg) {
        float frac3 = (total - adj) / (emerg - adj);
        float tmp = a_out + expf(0.1f * logf(frac3)) * (fc_out - a_out);
        if (inflow >= fc_out) {
            float flood = n_out + ((total - cons) / (emerg - cons)) * (inflow - n_out);
            ro = fmaxf(flood, tmp);
        } else {
            ro = tmp;
        }
    } else {
        ro = (inflow >= fc_out) ? inflow : fc_out;
    }
    ro = fmaxf(ro, 0.0f);
    ro = fminf(ro, total / time_step);

    river_outflow[ci] = ro;
    flood_outflow[ci] = 0.0f;
    total_storage[ci] = (STO)total;

    float new_pos = fmaxf(ro, 0.0f);
    atomicAdd(outgoing_storage + ci, (STO)(new_pos * time_step));
    float new_neg = fminf(ro, 0.0f);
    STO to_add = is_river_mouth ? (STO)0 : (STO)(-(new_neg * time_step));
    atomicAdd(outgoing_storage + di, to_add);
}

void launch_reservoir_outflow(
    at::Tensor reservoir_catchment_idx, at::Tensor downstream_idx,
    at::Tensor reservoir_total_inflow, at::Tensor river_outflow, at::Tensor flood_outflow,
    at::Tensor river_storage, at::Tensor flood_storage,
    at::Tensor conservation_volume, at::Tensor emergency_volume, at::Tensor adjustment_volume,
    at::Tensor normal_outflow, at::Tensor adjustment_outflow, at::Tensor flood_control_outflow,
    at::Tensor runoff, at::Tensor total_storage, at::Tensor outgoing_storage,
    at::Tensor time_step, long num_reservoirs, long block)
{
    int grid = (int)((num_reservoirs + block - 1) / block);
    hipStream_t stream = c10::hip::getCurrentHIPStream();
    bool dbl = (river_storage.scalar_type() == at::kDouble);
    if (dbl)
        k_reservoir_outflow<double><<<grid, (int)block, 0, stream>>>(
            reservoir_catchment_idx.data_ptr<int>(), downstream_idx.data_ptr<int>(),
            reservoir_total_inflow.data_ptr<double>(), river_outflow.data_ptr<float>(),
            flood_outflow.data_ptr<float>(), river_storage.data_ptr<double>(),
            flood_storage.data_ptr<double>(), conservation_volume.data_ptr<float>(),
            emergency_volume.data_ptr<float>(), adjustment_volume.data_ptr<float>(),
            normal_outflow.data_ptr<float>(), adjustment_outflow.data_ptr<float>(),
            flood_control_outflow.data_ptr<float>(), runoff.data_ptr<float>(),
            total_storage.data_ptr<double>(), outgoing_storage.data_ptr<double>(),
            time_step.data_ptr<float>(), num_reservoirs);
    else
        k_reservoir_outflow<float><<<grid, (int)block, 0, stream>>>(
            reservoir_catchment_idx.data_ptr<int>(), downstream_idx.data_ptr<int>(),
            reservoir_total_inflow.data_ptr<float>(), river_outflow.data_ptr<float>(),
            flood_outflow.data_ptr<float>(), river_storage.data_ptr<float>(),
            flood_storage.data_ptr<float>(), conservation_volume.data_ptr<float>(),
            emergency_volume.data_ptr<float>(), adjustment_volume.data_ptr<float>(),
            normal_outflow.data_ptr<float>(), adjustment_outflow.data_ptr<float>(),
            flood_control_outflow.data_ptr<float>(), runoff.data_ptr<float>(),
            total_storage.data_ptr<float>(), outgoing_storage.data_ptr<float>(),
            time_step.data_ptr<float>(), num_reservoirs);
}
