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
// (avoids log() of a negative argument).  Templated on STO (hpfloat).  No
// --use_fast_math, keeping sqrt/exp/log behavior predictable.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

template <typename REAL, typename STO>
__device__ __forceinline__ void k_reservoir_outflow_cell(
    const int* __restrict__ reservoir_catchment_idx,
    const int* __restrict__ downstream_idx,
    STO* __restrict__ reservoir_total_inflow,
    REAL* __restrict__ river_outflow, REAL* __restrict__ flood_outflow,
    const STO* __restrict__ river_storage, const STO* __restrict__ flood_storage,
    const REAL* __restrict__ conservation_volume, const REAL* __restrict__ emergency_volume,
    const REAL* __restrict__ adjustment_volume, const REAL* __restrict__ effective_normal_outflow,
    const REAL* __restrict__ adjustment_outflow, const REAL* __restrict__ flood_control_outflow,
    const REAL* __restrict__ runoff,
    STO* __restrict__ outgoing_storage, const REAL* __restrict__ time_step_ptr,
    int num_reservoirs)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_reservoirs) return;
    REAL time_step = __ldg(time_step_ptr);

    int ci = reservoir_catchment_idx[t];
    int di = downstream_idx[ci];
    bool is_river_mouth = (di == ci);

    REAL old_r = river_outflow[ci];
    REAL old_f = flood_outflow[ci];
    REAL old_pos = fmax(old_r, (REAL)0.0) + fmax(old_f, (REAL)0.0);
    REAL old_neg = fmin(old_r, (REAL)0.0) + fmin(old_f, (REAL)0.0);
    atomicAdd(outgoing_storage + ci, (STO)(-(old_pos * time_step)));
    STO undo_dn = is_river_mouth ? (STO)0 : (STO)(old_neg * time_step);
    atomicAdd(outgoing_storage + di, undo_dn);

    REAL rs = (REAL)river_storage[ci];
    REAL fs = (REAL)flood_storage[ci];
    REAL total = rs + fs;

    REAL total_inflow = (REAL)reservoir_total_inflow[ci];
    reservoir_total_inflow[ci] = (STO)0;
    REAL runoff_v = __ldg(runoff + ci);
    REAL inflow = total_inflow + runoff_v;

    REAL cons = __ldg(conservation_volume + t);
    REAL emerg = __ldg(emergency_volume + t);
    REAL adj = __ldg(adjustment_volume + t);
    REAL n_out = __ldg(effective_normal_outflow + t);
    REAL a_out = __ldg(adjustment_outflow + t);
    REAL fc_out = __ldg(flood_control_outflow + t);

    REAL ro;
    if (total <= cons) {
        ro = n_out * sqrt(total / cons);
    } else if (total <= adj) {
        REAL frac2 = (total - cons) / (adj - cons);
        ro = n_out + exp((REAL)3.0 * log(frac2)) * (a_out - n_out);
    } else if (total <= emerg) {
        REAL frac3 = (total - adj) / (emerg - adj);
        REAL tmp = a_out + exp((REAL)CMF_RESERVOIR_RELEASE_EXPONENT * log(frac3)) * (fc_out - a_out);
        if (inflow >= fc_out) {
            REAL flood = n_out + ((total - cons) / (emerg - cons)) * (inflow - n_out);
            ro = fmax(flood, tmp);
        } else {
            ro = tmp;
        }
    } else {
        ro = (inflow >= fc_out) ? inflow : fc_out;
    }
    ro = fmax(ro, (REAL)0.0);
    ro = fmin(ro, total / time_step);

    river_outflow[ci] = ro;
    flood_outflow[ci] = (REAL)0.0;

    REAL new_pos = fmax(ro, (REAL)0.0);
    atomicAdd(outgoing_storage + ci, (STO)(new_pos * time_step));
    REAL new_neg = fmin(ro, (REAL)0.0);
    STO to_add = is_river_mouth ? (STO)0 : (STO)(-(new_neg * time_step));
    atomicAdd(outgoing_storage + di, to_add);
}

template <typename REAL, typename STO>
__global__ void k_reservoir_outflow(
    const int* __restrict__ reservoir_catchment_idx,
    const int* __restrict__ downstream_idx,
    STO* __restrict__ reservoir_total_inflow,
    REAL* __restrict__ river_outflow, REAL* __restrict__ flood_outflow,
    const STO* __restrict__ river_storage, const STO* __restrict__ flood_storage,
    const REAL* __restrict__ conservation_volume, const REAL* __restrict__ emergency_volume,
    const REAL* __restrict__ adjustment_volume, const REAL* __restrict__ effective_normal_outflow,
    const REAL* __restrict__ adjustment_outflow, const REAL* __restrict__ flood_control_outflow,
    const REAL* __restrict__ runoff,
    STO* __restrict__ outgoing_storage, const REAL* __restrict__ time_step_ptr,
    int num_reservoirs)
{
    k_reservoir_outflow_cell<REAL, STO>(
        reservoir_catchment_idx, downstream_idx, reservoir_total_inflow, river_outflow,
        flood_outflow, river_storage, flood_storage, conservation_volume, emergency_volume,
        adjustment_volume, effective_normal_outflow, adjustment_outflow, flood_control_outflow,
        runoff, outgoing_storage, time_step_ptr, num_reservoirs);
}

template <typename REAL, typename STO>
__global__ void k_reservoir_outflow_batched(
    const int* __restrict__ reservoir_catchment_idx,
    const int* __restrict__ downstream_idx,
    STO* __restrict__ reservoir_total_inflow,
    REAL* __restrict__ river_outflow, REAL* __restrict__ flood_outflow,
    const STO* __restrict__ river_storage, const STO* __restrict__ flood_storage,
    const REAL* __restrict__ conservation_volume, const REAL* __restrict__ emergency_volume,
    const REAL* __restrict__ adjustment_volume, const REAL* __restrict__ effective_normal_outflow,
    const REAL* __restrict__ adjustment_outflow, const REAL* __restrict__ flood_control_outflow,
    const REAL* __restrict__ runoff,
    STO* __restrict__ outgoing_storage, const REAL* __restrict__ time_step_ptr,
    int num_reservoirs, long num_catchments, bool batched_runoff,
    bool batched_conservation_volume, bool batched_emergency_volume,
    bool batched_adjustment_volume, bool batched_effective_normal_outflow,
    bool batched_adjustment_outflow, bool batched_flood_control_outflow)
{
    const long member_offset = (long)blockIdx.y * num_catchments;
    reservoir_total_inflow += member_offset;
    river_outflow += member_offset;
    flood_outflow += member_offset;
    river_storage += member_offset;
    flood_storage += member_offset;
    outgoing_storage += member_offset;
    if (batched_runoff) runoff += member_offset;
    const long reservoir_offset = (long)blockIdx.y * num_reservoirs;
    if (batched_conservation_volume) conservation_volume += reservoir_offset;
    if (batched_emergency_volume) emergency_volume += reservoir_offset;
    if (batched_adjustment_volume) adjustment_volume += reservoir_offset;
    if (batched_effective_normal_outflow) effective_normal_outflow += reservoir_offset;
    if (batched_adjustment_outflow) adjustment_outflow += reservoir_offset;
    if (batched_flood_control_outflow) flood_control_outflow += reservoir_offset;
    k_reservoir_outflow_cell<REAL, STO>(
        reservoir_catchment_idx, downstream_idx, reservoir_total_inflow, river_outflow,
        flood_outflow, river_storage, flood_storage, conservation_volume, emergency_volume,
        adjustment_volume, effective_normal_outflow, adjustment_outflow, flood_control_outflow,
        runoff, outgoing_storage, time_step_ptr, num_reservoirs);
}

void launch_reservoir_outflow(
    at::Tensor reservoir_catchment_idx_ptr, at::Tensor downstream_idx_ptr,
    at::Tensor reservoir_total_inflow_ptr, at::Tensor river_outflow_ptr,
    at::Tensor flood_outflow_ptr, at::Tensor river_storage_ptr,
    at::Tensor flood_storage_ptr, at::Tensor conservation_volume_ptr,
    at::Tensor emergency_volume_ptr, at::Tensor adjustment_volume_ptr,
    at::Tensor effective_normal_outflow_ptr, at::Tensor adjustment_outflow_ptr,
    at::Tensor flood_control_outflow_ptr, at::Tensor runoff_ptr,
    at::Tensor outgoing_storage_ptr,
    at::Tensor time_step_ptr, long num_catchments,
    int num_reservoirs, long ensemble_size, bool batched_runoff, long BLOCK_SIZE)
{
    const dim3 grid((num_reservoirs + BLOCK_SIZE - 1) / BLOCK_SIZE, ensemble_size);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    bool real64 = river_outflow_ptr.scalar_type() == at::kDouble;
    bool sto64 = river_storage_ptr.scalar_type() == at::kDouble;
#define LAUNCH_RESERVOIR(REAL_T, STO_T) \
        do { \
            if (ensemble_size > 1) { \
                k_reservoir_outflow_batched<REAL_T, STO_T><<<grid, (int)BLOCK_SIZE, 0, stream>>>( \
                    reservoir_catchment_idx_ptr.data_ptr<int>(), downstream_idx_ptr.data_ptr<int>(), \
                    reservoir_total_inflow_ptr.data_ptr<STO_T>(), \
                    river_outflow_ptr.data_ptr<REAL_T>(), flood_outflow_ptr.data_ptr<REAL_T>(), \
                    river_storage_ptr.data_ptr<STO_T>(), flood_storage_ptr.data_ptr<STO_T>(), \
                    conservation_volume_ptr.data_ptr<REAL_T>(), \
                    emergency_volume_ptr.data_ptr<REAL_T>(), \
                    adjustment_volume_ptr.data_ptr<REAL_T>(), \
                    effective_normal_outflow_ptr.data_ptr<REAL_T>(), \
                    adjustment_outflow_ptr.data_ptr<REAL_T>(), \
                    flood_control_outflow_ptr.data_ptr<REAL_T>(), runoff_ptr.data_ptr<REAL_T>(), \
                    outgoing_storage_ptr.data_ptr<STO_T>(), time_step_ptr.data_ptr<REAL_T>(), \
                    num_reservoirs, num_catchments, batched_runoff, \
                    conservation_volume_ptr.dim() == 2, emergency_volume_ptr.dim() == 2, \
                    adjustment_volume_ptr.dim() == 2, effective_normal_outflow_ptr.dim() == 2, \
                    adjustment_outflow_ptr.dim() == 2, flood_control_outflow_ptr.dim() == 2); \
            } else { \
                k_reservoir_outflow<REAL_T, STO_T><<<grid, (int)BLOCK_SIZE, 0, stream>>>( \
                    reservoir_catchment_idx_ptr.data_ptr<int>(), downstream_idx_ptr.data_ptr<int>(), \
                    reservoir_total_inflow_ptr.data_ptr<STO_T>(), \
                    river_outflow_ptr.data_ptr<REAL_T>(), flood_outflow_ptr.data_ptr<REAL_T>(), \
                    river_storage_ptr.data_ptr<STO_T>(), flood_storage_ptr.data_ptr<STO_T>(), \
                    conservation_volume_ptr.data_ptr<REAL_T>(), \
                    emergency_volume_ptr.data_ptr<REAL_T>(), \
                    adjustment_volume_ptr.data_ptr<REAL_T>(), \
                    effective_normal_outflow_ptr.data_ptr<REAL_T>(), \
                    adjustment_outflow_ptr.data_ptr<REAL_T>(), \
                    flood_control_outflow_ptr.data_ptr<REAL_T>(), runoff_ptr.data_ptr<REAL_T>(), \
                    outgoing_storage_ptr.data_ptr<STO_T>(), time_step_ptr.data_ptr<REAL_T>(), \
                    num_reservoirs); \
            } \
        } while (false)
    if (real64) {
        LAUNCH_RESERVOIR(double, double);
    } else if (sto64) {
        LAUNCH_RESERVOIR(float, double);
    } else {
        LAUNCH_RESERVOIR(float, float);
    }
#undef LAUNCH_RESERVOIR
}
