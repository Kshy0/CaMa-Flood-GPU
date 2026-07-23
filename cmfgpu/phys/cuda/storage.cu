// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// CUDA backend for the fused storage-update + flood-stage kernel.
//
// CUDA storage-update + flood-stage implementation.  Lane-local early exits are
// used for dry cells and completed flood-level scans.  There is no warp/group
// vote path here; divergent lanes are left to the CUDA compiler and SIMT
// scheduler.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

template <typename REAL, typename STO, bool LOG>
__global__ void k_flood_stage(
    const STO* __restrict__ river_inflow, const STO* __restrict__ flood_inflow,
    const REAL* __restrict__ river_outflow, const REAL* __restrict__ flood_outflow,
    const STO* __restrict__ global_bif_outflow, const REAL* __restrict__ runoff,
    const REAL* __restrict__ inflow, const int* __restrict__ catchment_inflow_idx,
    const REAL* __restrict__ time_step_ptr,
    STO* __restrict__ outgoing_storage,
    STO* __restrict__ river_storage, STO* __restrict__ flood_storage,
    STO* __restrict__ protected_storage,
    REAL* __restrict__ river_depth, REAL* __restrict__ flood_depth,
    REAL* __restrict__ protected_depth, REAL* __restrict__ flood_fraction,
    const REAL* __restrict__ river_height, const REAL* __restrict__ flood_depth_table,
    const REAL* __restrict__ catchment_area, const REAL* __restrict__ river_width,
    const REAL* __restrict__ river_length,
    const bool* __restrict__ is_levee,
    REAL* __restrict__ total_storage_pre_sum,
    REAL* __restrict__ total_storage_next_sum,
    REAL* __restrict__ total_storage_new_sum,
    REAL* __restrict__ total_inflow_sum,
    REAL* __restrict__ total_outflow_sum,
    REAL* __restrict__ total_storage_stage_sum,
    REAL* __restrict__ river_storage_sum,
    REAL* __restrict__ flood_storage_sum,
    REAL* __restrict__ flood_area_sum,
    REAL* __restrict__ total_inflow_error_sum,
    REAL* __restrict__ total_stage_error_sum,
    const int* __restrict__ current_step_ptr,
    long num_catchments, int num_flood_levels,
    int has_bifurcation, int has_inflow, int has_levee)
{
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t >= num_catchments) return;

    REAL ts = __ldg(time_step_ptr);
    STO rsto = river_storage[t];
    STO fsto = flood_storage[t];
    STO prot = protected_storage[t];
    REAL rinf = (REAL)river_inflow[t];
    REAL finf = (REAL)flood_inflow[t];
    REAL gbif = has_bifurcation ? (REAL)global_bif_outflow[t] : (REAL)0;
    REAL rout = __ldg(river_outflow + t);
    REAL fout = __ldg(flood_outflow + t);
    REAL ro = __ldg(runoff + t);
    REAL prescribed_inflow = (REAL)0;
    if (has_inflow) {
        int inflow_idx = __ldg(catchment_inflow_idx + t);
        if (inflow_idx >= 0) prescribed_inflow = __ldg(inflow + inflow_idx);
    }

    bool non_levee = !has_levee || !is_levee[t];
    int current_step = LOG ? __ldg(current_step_ptr) : 0;
    STO total_stage_pre = rsto + fsto + prot;
    if constexpr (LOG) {
        atomicAdd(total_storage_pre_sum + current_step,
                  (REAL)total_stage_pre * (REAL)1e-9);
    }

    REAL river_delta = (rinf - rout) * ts;
    STO river_su = rsto + (STO)river_delta;
    REAL flood_delta = (finf - fout - gbif) * ts;
    STO flood_su = fsto
        + (river_su < (STO)0 ? river_su : (STO)0)
        + (STO)flood_delta;
    river_su = river_su > (STO)0 ? river_su : (STO)0;
    if (flood_su < (STO)0) {
        STO tt = river_su + flood_su;
        river_su = tt > (STO)0 ? tt : (STO)0;
    }
    flood_su = flood_su > (STO)0 ? flood_su : (STO)0;
    REAL external_delta = (ro + prescribed_inflow) * ts;
    STO total_next = river_su + flood_su + prot + (STO)external_delta;
    if constexpr (LOG) {
        if (non_levee) {
            atomicAdd(total_storage_next_sum + current_step,
                      (REAL)total_next * (REAL)1e-9);
            atomicAdd(total_inflow_sum + current_step,
                      (rinf + finf + prescribed_inflow) * ts * (REAL)1e-9);
            atomicAdd(total_outflow_sum + current_step,
                      (rout + fout) * ts * (REAL)1e-9);
            REAL balance = (REAL)total_stage_pre - (REAL)total_next
                + (rinf + finf + ro + prescribed_inflow - rout - fout - gbif) * ts;
            atomicAdd(total_inflow_error_sum + current_step,
                      balance * (REAL)1e-9);
        }
    }
    STO total_s = total_next;
    total_s = total_s > (STO)0 ? total_s : (STO)0;
    REAL total_storage = (REAL)total_s;
    if constexpr (LOG) {
        if (non_levee) {
            atomicAdd(total_storage_new_sum + current_step,
                      total_storage * (REAL)1e-9);
        }
    }

    REAL rh = __ldg(river_height + t);
    REAL rw = __ldg(river_width + t);
    REAL rl = __ldg(river_length + t);
    REAL river_max_storage = rl * rw * rh;

    if (total_storage <= river_max_storage) {
        REAL river_depth_dry = total_storage / (rl * rw);
        if constexpr (LOG) {
            atomicAdd(total_storage_stage_sum + current_step,
                      total_storage * (REAL)1e-9);
            if (non_levee) {
                atomicAdd(river_storage_sum + current_step,
                          total_storage * (REAL)1e-9);
            }
        }
        outgoing_storage[t] = (STO)0;
        river_storage[t] = (STO)total_storage;
        flood_storage[t] = (STO)0;
        protected_storage[t] = (STO)0;
        river_depth[t] = river_depth_dry;
        flood_depth[t] = (REAL)0;
        protected_depth[t] = (REAL)0;
        flood_fraction[t] = (REAL)0;
        return;
    }

    REAL ca = __ldg(catchment_area + t);
    REAL catchment_width = ca / rl;
    REAL width_increment = catchment_width / num_flood_levels;
    int level = 0;
    REAL S_accum = river_max_storage;
    REAL prev_H = (REAL)0;
    REAL prev_W = rw;
    REAL prev_total_storage = river_max_storage;
    REAL prev_flood_depth = (REAL)0;
    REAL next_flood_depth = (REAL)0;

    for (int i = 0; i < num_flood_levels; ++i) {
        REAL H_curr = __ldg(flood_depth_table + t * num_flood_levels + i);
        REAL W_curr = rw + (i + 1) * width_increment;
        REAL dS = rl * (REAL)0.5 * (prev_W + W_curr) * (H_curr - prev_H);
        REAL S_curr = S_accum + dS;
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

    REAL prev_total_width = rw + level * width_increment;
    REAL diff_width = (REAL)0;
    REAL fdep;
    if (level == num_flood_levels) {
        fdep = prev_flood_depth + (total_storage - prev_total_storage) / (prev_total_width * rl);
    } else {
        REAL flood_grad = (next_flood_depth - prev_flood_depth) / width_increment;
        diff_width = sqrt(prev_total_width * prev_total_width
            + (REAL)2 * (total_storage - prev_total_storage) / (flood_grad * rl)) - prev_total_width;
        fdep = prev_flood_depth + diff_width * flood_grad;
    }

    REAL cap = river_max_storage + rl * rw * fdep;
    REAL river_storage_final = cap < total_storage ? cap : total_storage;
    REAL rdep = river_storage_final / (rl * rw);
    REAL ff_mid = (prev_total_width + diff_width - rw) * rl / ca;
    ff_mid = ff_mid < (REAL)0 ? (REAL)0 : (ff_mid > (REAL)1 ? (REAL)1 : ff_mid);
    REAL ffr = (level == num_flood_levels) ? (REAL)1 : ff_mid;
    REAL flood_storage_final = total_storage - river_storage_final;
    if (flood_storage_final < (REAL)0) flood_storage_final = (REAL)0;

    if constexpr (LOG) {
        REAL total_stage_new = river_storage_final + flood_storage_final;
        atomicAdd(total_storage_stage_sum + current_step,
                  total_stage_new * (REAL)1e-9);
        if (non_levee) {
            atomicAdd(total_stage_error_sum + current_step,
                      (total_stage_new - total_storage) * (REAL)1e-9);
            atomicAdd(river_storage_sum + current_step,
                      river_storage_final * (REAL)1e-9);
            atomicAdd(flood_storage_sum + current_step,
                      flood_storage_final * (REAL)1e-9);
            atomicAdd(flood_area_sum + current_step,
                      ffr * ca * (REAL)1e-9);
        }
    }

    outgoing_storage[t] = (STO)0;
    river_storage[t] = (STO)river_storage_final;
    flood_storage[t] = (STO)flood_storage_final;
    protected_storage[t] = (STO)0;
    river_depth[t] = rdep;
    flood_depth[t] = fdep;
    protected_depth[t] = fdep;
    flood_fraction[t] = ffr;
}

template <typename REAL, typename STO, bool LOG>
static void launch_t(const at::Tensor& ri, const at::Tensor& fi,
    const at::Tensor& ro, const at::Tensor& fo,
    const c10::optional<at::Tensor>& gb, const at::Tensor& run,
    const c10::optional<at::Tensor>& inflow,
    const c10::optional<at::Tensor>& catchment_inflow_idx,
    const at::Tensor& tsp,
    const at::Tensor& outs,
    const at::Tensor& rs, const at::Tensor& fs, const at::Tensor& ps,
    const at::Tensor& rd, const at::Tensor& fd, const at::Tensor& pd, const at::Tensor& ff,
    const at::Tensor& rh, const at::Tensor& tbl, const at::Tensor& ca,
    const at::Tensor& rw, const at::Tensor& rl,
    const c10::optional<at::Tensor>& is_levee,
    REAL* total_storage_pre_sum, REAL* total_storage_next_sum,
    REAL* total_storage_new_sum, REAL* total_inflow_sum,
    REAL* total_outflow_sum, REAL* total_storage_stage_sum,
    REAL* river_storage_sum, REAL* flood_storage_sum,
    REAL* flood_area_sum, REAL* total_inflow_error_sum,
    REAL* total_stage_error_sum,
    const c10::optional<at::Tensor>& current_step,
    long n, int nl, int has_bif, int has_inflow,
    int has_levee, int block)
{
    int grid = (int)((n + block - 1) / block);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const STO* gbp = gb ? gb->data_ptr<STO>() : nullptr;
    const REAL* inflow_p = inflow ? inflow->data_ptr<REAL>() : nullptr;
    const int* inflow_idx_p = catchment_inflow_idx
        ? catchment_inflow_idx->data_ptr<int>() : nullptr;
    const bool* levee_p = is_levee ? is_levee->data_ptr<bool>() : nullptr;
    const int* step_p = current_step ? current_step->data_ptr<int>() : nullptr;
    k_flood_stage<REAL, STO, LOG><<<grid, block, 0, stream>>>(
        ri.data_ptr<STO>(), fi.data_ptr<STO>(),
        ro.data_ptr<REAL>(), fo.data_ptr<REAL>(), gbp, run.data_ptr<REAL>(),
        inflow_p, inflow_idx_p, tsp.data_ptr<REAL>(),
        outs.data_ptr<STO>(), rs.data_ptr<STO>(), fs.data_ptr<STO>(), ps.data_ptr<STO>(),
        rd.data_ptr<REAL>(), fd.data_ptr<REAL>(), pd.data_ptr<REAL>(), ff.data_ptr<REAL>(),
        rh.data_ptr<REAL>(), tbl.data_ptr<REAL>(), ca.data_ptr<REAL>(), rw.data_ptr<REAL>(), rl.data_ptr<REAL>(),
        levee_p, total_storage_pre_sum, total_storage_next_sum,
        total_storage_new_sum, total_inflow_sum, total_outflow_sum,
        total_storage_stage_sum, river_storage_sum, flood_storage_sum,
        flood_area_sum, total_inflow_error_sum, total_stage_error_sum, step_p,
        n, nl, has_bif, has_inflow, has_levee);
}

static void launch(
    at::Tensor river_inflow, at::Tensor flood_inflow, at::Tensor river_outflow,
    at::Tensor flood_outflow, c10::optional<at::Tensor> global_bif_outflow,
    at::Tensor runoff, c10::optional<at::Tensor> inflow,
    c10::optional<at::Tensor> catchment_inflow_idx, at::Tensor time_step,
    at::Tensor outgoing_storage, at::Tensor river_storage,
    at::Tensor flood_storage, at::Tensor protected_storage, at::Tensor river_depth,
    at::Tensor flood_depth, at::Tensor protected_depth, at::Tensor flood_fraction,
    at::Tensor river_height, at::Tensor flood_depth_table, at::Tensor catchment_area,
    at::Tensor river_width, at::Tensor river_length,
    long n, int nl, int has_bif, int has_inflow, int block)
{
    if (river_depth.scalar_type() == at::kDouble) {
        launch_t<double, double, false>(river_inflow, flood_inflow, river_outflow, flood_outflow,
            global_bif_outflow, runoff, inflow, catchment_inflow_idx, time_step,
            outgoing_storage, river_storage,
            flood_storage, protected_storage, river_depth, flood_depth, protected_depth,
            flood_fraction, river_height, flood_depth_table, catchment_area, river_width,
            river_length, c10::nullopt,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, c10::nullopt,
            n, nl, has_bif, has_inflow, 0, block);
    } else if (river_storage.scalar_type() == at::kDouble) {
        launch_t<float, double, false>(river_inflow, flood_inflow, river_outflow, flood_outflow,
            global_bif_outflow, runoff, inflow, catchment_inflow_idx, time_step,
            outgoing_storage, river_storage,
            flood_storage, protected_storage, river_depth, flood_depth, protected_depth,
            flood_fraction, river_height, flood_depth_table, catchment_area, river_width,
            river_length, c10::nullopt,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, c10::nullopt,
            n, nl, has_bif, has_inflow, 0, block);
    } else {
        launch_t<float, float, false>(river_inflow, flood_inflow, river_outflow, flood_outflow,
            global_bif_outflow, runoff, inflow, catchment_inflow_idx, time_step,
            outgoing_storage, river_storage,
            flood_storage, protected_storage, river_depth, flood_depth, protected_depth,
            flood_fraction, river_height, flood_depth_table, catchment_area, river_width,
            river_length, c10::nullopt,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, c10::nullopt,
            n, nl, has_bif, has_inflow, 0, block);
    }
}

void launch_flood_stage(
    at::Tensor river_inflow_ptr, at::Tensor flood_inflow_ptr,
    at::Tensor river_outflow_ptr, at::Tensor flood_outflow_ptr,
    c10::optional<at::Tensor> global_bifurcation_outflow_ptr,
    at::Tensor runoff_ptr, c10::optional<at::Tensor> inflow_ptr,
    c10::optional<at::Tensor> catchment_inflow_idx_ptr,
    at::Tensor time_step_ptr, at::Tensor outgoing_storage_ptr,
    at::Tensor river_storage_ptr, at::Tensor flood_storage_ptr,
    at::Tensor protected_storage_ptr, at::Tensor river_depth_ptr,
    at::Tensor flood_depth_ptr, at::Tensor protected_depth_ptr,
    at::Tensor flood_fraction_ptr, at::Tensor river_height_ptr,
    at::Tensor flood_depth_table_ptr, at::Tensor catchment_area_ptr,
    at::Tensor river_width_ptr, at::Tensor river_length_ptr,
    long num_catchments, int num_inflow_gauges,
    int num_flood_levels, bool HAS_BIFURCATION,
    bool HAS_INFLOW, long BLOCK_SIZE)
{
    (void)num_inflow_gauges;
    launch(
        river_inflow_ptr, flood_inflow_ptr, river_outflow_ptr,
        flood_outflow_ptr, global_bifurcation_outflow_ptr, runoff_ptr,
        inflow_ptr, catchment_inflow_idx_ptr, time_step_ptr,
        outgoing_storage_ptr, river_storage_ptr, flood_storage_ptr,
        protected_storage_ptr, river_depth_ptr, flood_depth_ptr,
        protected_depth_ptr, flood_fraction_ptr, river_height_ptr,
        flood_depth_table_ptr, catchment_area_ptr, river_width_ptr,
        river_length_ptr, num_catchments, num_flood_levels,
        (int)HAS_BIFURCATION, (int)HAS_INFLOW, (int)BLOCK_SIZE);
}

void launch_flood_stage_log(
    at::Tensor river_inflow_ptr, at::Tensor flood_inflow_ptr,
    at::Tensor river_outflow_ptr, at::Tensor flood_outflow_ptr,
    c10::optional<at::Tensor> global_bifurcation_outflow_ptr,
    at::Tensor runoff_ptr, c10::optional<at::Tensor> inflow_ptr,
    c10::optional<at::Tensor> catchment_inflow_idx_ptr,
    at::Tensor time_step_ptr, at::Tensor outgoing_storage_ptr,
    at::Tensor river_storage_ptr, at::Tensor flood_storage_ptr,
    at::Tensor protected_storage_ptr, at::Tensor river_depth_ptr,
    at::Tensor flood_depth_ptr, at::Tensor protected_depth_ptr,
    at::Tensor flood_fraction_ptr, at::Tensor river_height_ptr,
    at::Tensor flood_depth_table_ptr, at::Tensor catchment_area_ptr,
    at::Tensor river_width_ptr, at::Tensor river_length_ptr,
    c10::optional<at::Tensor> is_levee_ptr,
    at::Tensor total_storage_pre_sum_ptr,
    at::Tensor total_storage_next_sum_ptr,
    at::Tensor total_storage_new_sum_ptr,
    at::Tensor total_inflow_sum_ptr, at::Tensor total_outflow_sum_ptr,
    at::Tensor total_storage_stage_sum_ptr,
    at::Tensor river_storage_sum_ptr, at::Tensor flood_storage_sum_ptr,
    at::Tensor flood_area_sum_ptr, at::Tensor total_inflow_error_sum_ptr,
    at::Tensor total_stage_error_sum_ptr,
    at::Tensor current_step_ptr, long num_catchments,
    int num_flood_levels,
    bool HAS_BIFURCATION, bool HAS_INFLOW, bool HAS_LEVEE,
    long BLOCK_SIZE)
{
    if (river_depth_ptr.scalar_type() == at::kDouble) {
        launch_t<double, double, true>(
            river_inflow_ptr, flood_inflow_ptr, river_outflow_ptr,
            flood_outflow_ptr, global_bifurcation_outflow_ptr, runoff_ptr,
            inflow_ptr, catchment_inflow_idx_ptr, time_step_ptr,
            outgoing_storage_ptr, river_storage_ptr, flood_storage_ptr,
            protected_storage_ptr, river_depth_ptr, flood_depth_ptr,
            protected_depth_ptr, flood_fraction_ptr, river_height_ptr,
            flood_depth_table_ptr, catchment_area_ptr, river_width_ptr,
            river_length_ptr, is_levee_ptr,
            total_storage_pre_sum_ptr.data_ptr<double>(),
            total_storage_next_sum_ptr.data_ptr<double>(),
            total_storage_new_sum_ptr.data_ptr<double>(),
            total_inflow_sum_ptr.data_ptr<double>(),
            total_outflow_sum_ptr.data_ptr<double>(),
            total_storage_stage_sum_ptr.data_ptr<double>(),
            river_storage_sum_ptr.data_ptr<double>(),
            flood_storage_sum_ptr.data_ptr<double>(),
            flood_area_sum_ptr.data_ptr<double>(),
            total_inflow_error_sum_ptr.data_ptr<double>(),
            total_stage_error_sum_ptr.data_ptr<double>(), current_step_ptr,
            num_catchments, num_flood_levels,
            HAS_BIFURCATION, HAS_INFLOW, HAS_LEVEE, BLOCK_SIZE);
    } else if (river_storage_ptr.scalar_type() == at::kDouble) {
        launch_t<float, double, true>(
            river_inflow_ptr, flood_inflow_ptr, river_outflow_ptr,
            flood_outflow_ptr, global_bifurcation_outflow_ptr, runoff_ptr,
            inflow_ptr, catchment_inflow_idx_ptr, time_step_ptr,
            outgoing_storage_ptr, river_storage_ptr, flood_storage_ptr,
            protected_storage_ptr, river_depth_ptr, flood_depth_ptr,
            protected_depth_ptr, flood_fraction_ptr, river_height_ptr,
            flood_depth_table_ptr, catchment_area_ptr, river_width_ptr,
            river_length_ptr, is_levee_ptr,
            total_storage_pre_sum_ptr.data_ptr<float>(),
            total_storage_next_sum_ptr.data_ptr<float>(),
            total_storage_new_sum_ptr.data_ptr<float>(),
            total_inflow_sum_ptr.data_ptr<float>(),
            total_outflow_sum_ptr.data_ptr<float>(),
            total_storage_stage_sum_ptr.data_ptr<float>(),
            river_storage_sum_ptr.data_ptr<float>(),
            flood_storage_sum_ptr.data_ptr<float>(),
            flood_area_sum_ptr.data_ptr<float>(),
            total_inflow_error_sum_ptr.data_ptr<float>(),
            total_stage_error_sum_ptr.data_ptr<float>(), current_step_ptr,
            num_catchments, num_flood_levels,
            HAS_BIFURCATION, HAS_INFLOW, HAS_LEVEE, BLOCK_SIZE);
    } else {
        launch_t<float, float, true>(
            river_inflow_ptr, flood_inflow_ptr, river_outflow_ptr,
            flood_outflow_ptr, global_bifurcation_outflow_ptr, runoff_ptr,
            inflow_ptr, catchment_inflow_idx_ptr, time_step_ptr,
            outgoing_storage_ptr, river_storage_ptr, flood_storage_ptr,
            protected_storage_ptr, river_depth_ptr, flood_depth_ptr,
            protected_depth_ptr, flood_fraction_ptr, river_height_ptr,
            flood_depth_table_ptr, catchment_area_ptr, river_width_ptr,
            river_length_ptr, is_levee_ptr,
            total_storage_pre_sum_ptr.data_ptr<float>(),
            total_storage_next_sum_ptr.data_ptr<float>(),
            total_storage_new_sum_ptr.data_ptr<float>(),
            total_inflow_sum_ptr.data_ptr<float>(),
            total_outflow_sum_ptr.data_ptr<float>(),
            total_storage_stage_sum_ptr.data_ptr<float>(),
            river_storage_sum_ptr.data_ptr<float>(),
            flood_storage_sum_ptr.data_ptr<float>(),
            flood_area_sum_ptr.data_ptr<float>(),
            total_inflow_error_sum_ptr.data_ptr<float>(),
            total_stage_error_sum_ptr.data_ptr<float>(), current_step_ptr,
            num_catchments, num_flood_levels,
            HAS_BIFURCATION, HAS_INFLOW, HAS_LEVEE, BLOCK_SIZE);
    }
}
