// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// HIP backend for the bifurcation outflow / inflow kernels.
//
// Mirrors cmfgpu/phys/triton/bifurcation.py.  One thread per bifurcation path;
// each path owns num_bifurcation_levels contiguous level entries.  Cross-path
// collisions on (catchment, downstream) cells use atomics.  Templated on STO
// (hpfloat) for total_storage / outgoing_storage / global_bif_outflow; all
// bifurcation geometry / outflow / manning are float.  No -ffast-math.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

__device__ __forceinline__ float pow73(float x) { return x * x * cbrtf(x); }

template <typename STO>
__global__ void k_bif_outflow(
    const int* __restrict__ cat_idx, const int* __restrict__ dn_idx,
    const float* __restrict__ manning, float* __restrict__ outflow,
    const float* __restrict__ width, const float* __restrict__ length,
    const float* __restrict__ elevation, float* __restrict__ cs_depth,
    const float* __restrict__ wse, const STO* __restrict__ total_storage,
    STO* __restrict__ outgoing_storage,
    float gravity, const float* __restrict__ time_step_ptr,
    long num_paths, int num_levels)
{
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t >= num_paths) return;
    float time_step = __ldg(time_step_ptr);

    int ci = cat_idx[t];
    int di = dn_idx[t];
    float blen = __ldg(length + t);

    float wse_c = __ldg(wse + ci);
    float wse_d = __ldg(wse + di);
    float max_wse = fmaxf(wse_c, wse_d);

    float slope = (wse_c - wse_d) / blen;
    slope = fminf(fmaxf(slope, -0.005f), 0.005f);

    float ts_c = (float)total_storage[ci];
    float ts_d = (float)total_storage[di];

    float sum_out = 0.0f;
    for (int lv = 0; lv < num_levels; ++lv) {
        long li = t * (long)num_levels + lv;
        float man = __ldg(manning + li);
        float csd = __ldg(cs_depth + li);
        float elv = __ldg(elevation + li);
        float upd_csd = fmaxf(max_wse - elv, 0.0f);
        float sifd = fmaxf(sqrtf(upd_csd * csd), sqrtf(upd_csd * 0.01f));
        bool flow_condition = sifd > 1e-5f;
        float upd_o = 0.0f;
        if (flow_condition) {
            float w = __ldg(width + li);
            float o = outflow[li];
            float unit_o = o / w;
            float num = w * (unit_o + gravity * time_step * sifd * slope);
            float den = 1.0f + gravity * time_step * (man * man) * fabsf(unit_o) * (1.0f / pow73(sifd));
            upd_o = num / den;
        }
        sum_out += upd_o;
        cs_depth[li] = upd_csd;
        outflow[li] = upd_o;
    }

    float limit_rate = fminf(0.05f * fminf(ts_c, ts_d) / (fabsf(sum_out) * time_step), 1.0f);
    sum_out *= limit_rate;
    for (int lv = 0; lv < num_levels; ++lv) {
        long li = t * (long)num_levels + lv;
        outflow[li] = outflow[li] * limit_rate;
    }

    float pos = fmaxf(sum_out, 0.0f);
    float neg = fminf(sum_out, 0.0f);
    atomicAdd(outgoing_storage + ci, (STO)(pos * time_step));
    atomicAdd(outgoing_storage + di, (STO)(-neg * time_step));
}

template <typename STO>
__global__ void k_bif_inflow(
    const int* __restrict__ cat_idx, const int* __restrict__ dn_idx,
    const float* __restrict__ limit_rate, float* __restrict__ outflow,
    STO* __restrict__ global_bif_outflow, long num_paths, int num_levels)
{
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t >= num_paths) return;

    int ci = cat_idx[t];
    int di = dn_idx[t];
    float lr_c = __ldg(limit_rate + ci);
    float lr_d = __ldg(limit_rate + di);

    float sum_out = 0.0f;
    for (int lv = 0; lv < num_levels; ++lv) {
        long li = t * (long)num_levels + lv;
        float o = outflow[li];
        float upd = (o >= 0.0f) ? o * lr_c : o * lr_d;
        sum_out += upd;
        outflow[li] = upd;
    }
    atomicAdd(global_bif_outflow + ci, (STO)sum_out);
    atomicAdd(global_bif_outflow + di, (STO)(-sum_out));
}

void launch_bif_outflow(
    at::Tensor cat_idx, at::Tensor dn_idx, at::Tensor manning, at::Tensor outflow,
    at::Tensor width, at::Tensor length, at::Tensor elevation, at::Tensor cs_depth,
    at::Tensor wse, at::Tensor total_storage, at::Tensor outgoing_storage,
    double gravity, at::Tensor time_step, long num_paths, long num_levels, long block)
{
    int grid = (int)((num_paths + block - 1) / block);
    hipStream_t stream = c10::hip::getCurrentHIPStream();
    bool dbl = (total_storage.scalar_type() == at::kDouble);
    if (dbl)
        k_bif_outflow<double><<<grid, (int)block, 0, stream>>>(
            cat_idx.data_ptr<int>(), dn_idx.data_ptr<int>(), manning.data_ptr<float>(),
            outflow.data_ptr<float>(), width.data_ptr<float>(), length.data_ptr<float>(),
            elevation.data_ptr<float>(), cs_depth.data_ptr<float>(), wse.data_ptr<float>(),
            total_storage.data_ptr<double>(), outgoing_storage.data_ptr<double>(),
            (float)gravity, time_step.data_ptr<float>(), num_paths, (int)num_levels);
    else
        k_bif_outflow<float><<<grid, (int)block, 0, stream>>>(
            cat_idx.data_ptr<int>(), dn_idx.data_ptr<int>(), manning.data_ptr<float>(),
            outflow.data_ptr<float>(), width.data_ptr<float>(), length.data_ptr<float>(),
            elevation.data_ptr<float>(), cs_depth.data_ptr<float>(), wse.data_ptr<float>(),
            total_storage.data_ptr<float>(), outgoing_storage.data_ptr<float>(),
            (float)gravity, time_step.data_ptr<float>(), num_paths, (int)num_levels);
}

void launch_bif_inflow(
    at::Tensor cat_idx, at::Tensor dn_idx, at::Tensor limit_rate, at::Tensor outflow,
    at::Tensor global_bif_outflow, long num_paths, long num_levels, long block)
{
    int grid = (int)((num_paths + block - 1) / block);
    hipStream_t stream = c10::hip::getCurrentHIPStream();
    bool dbl = (global_bif_outflow.scalar_type() == at::kDouble);
    if (dbl)
        k_bif_inflow<double><<<grid, (int)block, 0, stream>>>(
            cat_idx.data_ptr<int>(), dn_idx.data_ptr<int>(), limit_rate.data_ptr<float>(),
            outflow.data_ptr<float>(), global_bif_outflow.data_ptr<double>(),
            num_paths, (int)num_levels);
    else
        k_bif_inflow<float><<<grid, (int)block, 0, stream>>>(
            cat_idx.data_ptr<int>(), dn_idx.data_ptr<int>(), limit_rate.data_ptr<float>(),
            outflow.data_ptr<float>(), global_bif_outflow.data_ptr<float>(),
            num_paths, (int)num_levels);
}
