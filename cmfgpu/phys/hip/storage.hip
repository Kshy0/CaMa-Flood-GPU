// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// HIP backend for the fused storage-update + flood-stage kernel.
//
// Faithfully mirrors cmfgpu/phys/triton/storage.py::compute_flood_stage_kernel,
// with lane-local early exits for dry cells and per-cell scan completion.  There
// is no warp/group vote path here; divergent lanes are left to the HIP compiler
// and SIMT scheduler.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

template <typename STO>
__global__ void k_flood_stage(
    const STO* __restrict__ river_inflow, const STO* __restrict__ flood_inflow,
    const float* __restrict__ river_outflow, const float* __restrict__ flood_outflow,
    const STO* __restrict__ global_bif_outflow, const float* __restrict__ runoff,
    const float* __restrict__ time_step_ptr,
    STO* __restrict__ outgoing_storage,
    STO* __restrict__ river_storage, STO* __restrict__ flood_storage,
    STO* __restrict__ protected_storage,
    float* __restrict__ river_depth, float* __restrict__ flood_depth,
    float* __restrict__ protected_depth, float* __restrict__ flood_fraction,
    const float* __restrict__ river_height, const float* __restrict__ flood_depth_table,
    const float* __restrict__ catchment_area, const float* __restrict__ river_width,
    const float* __restrict__ river_length,
    long num_catchments, int num_flood_levels, int has_bifurcation)
{
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t >= num_catchments) return;

    float ts = __ldg(time_step_ptr);
    STO dt = (STO)ts;
    STO rsto = river_storage[t];
    STO fsto = flood_storage[t];
    STO prot = protected_storage[t];
    STO rinf = river_inflow[t];
    STO finf = flood_inflow[t];
    STO gbif = has_bifurcation ? global_bif_outflow[t] : (STO)0;
    float rout = __ldg(river_outflow + t);
    float fout = __ldg(flood_outflow + t);
    float ro = __ldg(runoff + t);

    STO river_su = rsto + (rinf - (STO)rout) * dt;
    STO flood_su = fsto
        + (river_su < (STO)0 ? river_su : (STO)0)
        + (finf - (STO)fout - gbif) * dt;
    river_su = river_su > (STO)0 ? river_su : (STO)0;
    if (flood_su < (STO)0) {
        STO tt = river_su + flood_su;
        river_su = tt > (STO)0 ? tt : (STO)0;
    }
    flood_su = flood_su > (STO)0 ? flood_su : (STO)0;
    STO total_s = river_su + flood_su + prot + (STO)ro * dt;
    total_s = total_s > (STO)0 ? total_s : (STO)0;
    float total_storage = (float)total_s;

    float rh = __ldg(river_height + t);
    float rw = __ldg(river_width + t);
    float rl = __ldg(river_length + t);
    float river_max_storage = rl * rw * rh;

    if (total_storage <= river_max_storage) {
        float river_depth_dry = total_storage / (rl * rw);
        outgoing_storage[t] = (STO)0;
        river_storage[t] = (STO)total_storage;
        flood_storage[t] = (STO)0;
        protected_storage[t] = (STO)0;
        river_depth[t] = river_depth_dry;
        flood_depth[t] = 0.f;
        protected_depth[t] = 0.f;
        flood_fraction[t] = 0.f;
        return;
    }

    float ca = __ldg(catchment_area + t);
    float catchment_width = ca / rl;
    float width_increment = catchment_width / num_flood_levels;
    int level = 0;
    float S_accum = river_max_storage;
    float prev_H = 0.f;
    float prev_W = rw;
    float prev_total_storage = river_max_storage;
    float prev_flood_depth = 0.f;
    float next_flood_depth = 0.f;

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        if (i >= num_flood_levels) break;
        float H_curr = __ldg(flood_depth_table + t * num_flood_levels + i);
        float W_curr = rw + (i + 1) * width_increment;
        float dS = rl * 0.5f * (prev_W + W_curr) * (H_curr - prev_H);
        float S_curr = S_accum + dS;
        next_flood_depth = H_curr;
        bool is_above = total_storage > S_curr;
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

    float prev_total_width = rw + level * width_increment;
    float diff_width = 0.f;
    float fdep;
    if (level == num_flood_levels) {
        fdep = prev_flood_depth + (total_storage - prev_total_storage) / (prev_total_width * rl);
    } else {
        float flood_grad = (next_flood_depth - prev_flood_depth) / width_increment;
        diff_width = sqrtf(prev_total_width * prev_total_width
            + 2.f * (total_storage - prev_total_storage) / (flood_grad * rl)) - prev_total_width;
        fdep = prev_flood_depth + diff_width * flood_grad;
    }

    float cap = river_max_storage + rl * rw * fdep;
    float river_storage_final = cap < total_storage ? cap : total_storage;
    float rdep = river_storage_final / (rl * rw);
    float ff_mid = (prev_total_width + diff_width - rw) * rl / ca;
    ff_mid = ff_mid < 0.f ? 0.f : (ff_mid > 1.f ? 1.f : ff_mid);
    float ffr = (level == num_flood_levels) ? 1.f : ff_mid;
    float flood_storage_final = total_storage - river_storage_final;
    if (flood_storage_final < 0.f) flood_storage_final = 0.f;

    outgoing_storage[t] = (STO)0;
    river_storage[t] = (STO)river_storage_final;
    flood_storage[t] = (STO)flood_storage_final;
    protected_storage[t] = (STO)0;
    river_depth[t] = rdep;
    flood_depth[t] = fdep;
    protected_depth[t] = fdep;
    flood_fraction[t] = ffr;
}

template <typename STO>
static void launch_t(const at::Tensor& ri, const at::Tensor& fi,
    const at::Tensor& ro, const at::Tensor& fo, const at::Tensor& gb,
    const at::Tensor& run, const at::Tensor& tsp, const at::Tensor& outs,
    const at::Tensor& rs, const at::Tensor& fs, const at::Tensor& ps,
    const at::Tensor& rd, const at::Tensor& fd, const at::Tensor& pd, const at::Tensor& ff,
    const at::Tensor& rh, const at::Tensor& tbl, const at::Tensor& ca,
    const at::Tensor& rw, const at::Tensor& rl,
    long n, int nl, int has_bif, int block)
{
    int grid = (int)((n + block - 1) / block);
    hipStream_t stream = c10::hip::getCurrentHIPStream();
    const STO* gbp = gb.numel() ? gb.data_ptr<STO>() : nullptr;
    k_flood_stage<STO><<<grid, block, 0, stream>>>(
        ri.data_ptr<STO>(), fi.data_ptr<STO>(),
        ro.data_ptr<float>(), fo.data_ptr<float>(), gbp, run.data_ptr<float>(), tsp.data_ptr<float>(),
        outs.data_ptr<STO>(), rs.data_ptr<STO>(), fs.data_ptr<STO>(), ps.data_ptr<STO>(),
        rd.data_ptr<float>(), fd.data_ptr<float>(), pd.data_ptr<float>(), ff.data_ptr<float>(),
        rh.data_ptr<float>(), tbl.data_ptr<float>(), ca.data_ptr<float>(), rw.data_ptr<float>(), rl.data_ptr<float>(),
        n, nl, has_bif);
}

static void launch(
    at::Tensor river_inflow, at::Tensor flood_inflow, at::Tensor river_outflow,
    at::Tensor flood_outflow, at::Tensor global_bif_outflow, at::Tensor runoff,
    at::Tensor time_step, at::Tensor outgoing_storage, at::Tensor river_storage,
    at::Tensor flood_storage, at::Tensor protected_storage, at::Tensor river_depth,
    at::Tensor flood_depth, at::Tensor protected_depth, at::Tensor flood_fraction,
    at::Tensor river_height, at::Tensor flood_depth_table, at::Tensor catchment_area,
    at::Tensor river_width, at::Tensor river_length,
    long n, int nl, int has_bif, int block)
{
    if (river_storage.scalar_type() == at::kDouble) {
        launch_t<double>(river_inflow, flood_inflow, river_outflow, flood_outflow,
            global_bif_outflow, runoff, time_step, outgoing_storage, river_storage,
            flood_storage, protected_storage, river_depth, flood_depth, protected_depth,
            flood_fraction, river_height, flood_depth_table, catchment_area, river_width,
            river_length, n, nl, has_bif, block);
    } else {
        launch_t<float>(river_inflow, flood_inflow, river_outflow, flood_outflow,
            global_bif_outflow, runoff, time_step, outgoing_storage, river_storage,
            flood_storage, protected_storage, river_depth, flood_depth, protected_depth,
            flood_fraction, river_height, flood_depth_table, catchment_area, river_width,
            river_length, n, nl, has_bif, block);
    }
}

void launch_flood_stage(
    at::Tensor river_inflow, at::Tensor flood_inflow, at::Tensor river_outflow,
    at::Tensor flood_outflow, at::Tensor global_bif_outflow, at::Tensor runoff,
    at::Tensor time_step, at::Tensor outgoing_storage, at::Tensor river_storage,
    at::Tensor flood_storage, at::Tensor protected_storage, at::Tensor river_depth,
    at::Tensor flood_depth, at::Tensor protected_depth, at::Tensor flood_fraction,
    at::Tensor river_height, at::Tensor flood_depth_table, at::Tensor catchment_area,
    at::Tensor river_width, at::Tensor river_length,
    long num_catchments, long num_flood_levels, long has_bifurcation, long block)
{
    launch(river_inflow, flood_inflow, river_outflow, flood_outflow,
        global_bif_outflow, runoff, time_step, outgoing_storage, river_storage,
        flood_storage, protected_storage, river_depth, flood_depth, protected_depth,
        flood_fraction, river_height, flood_depth_table, catchment_area, river_width,
        river_length, num_catchments, (int)num_flood_levels, (int)has_bifurcation, (int)block);
}
