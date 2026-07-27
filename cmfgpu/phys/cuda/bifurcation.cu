// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// CUDA backend for the bifurcation outflow / inflow kernels.
//
// One thread per bifurcation path; each path owns num_bifurcation_levels
// contiguous level entries.  Cross-path collisions on (catchment, downstream)
// cells use atomics.  Templated on STO (hpfloat) for total_storage /
// outgoing_storage / global_bif_outflow and REAL for ordinary model fields.
// No --use_fast_math.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

template <typename REAL>
__device__ __forceinline__ REAL pow73(REAL x) { return x * x * cbrt(x); }

template <typename REAL, typename STO>
__global__ void k_bif_outflow(
    const int* __restrict__ cat_idx, const int* __restrict__ dn_idx,
    const REAL* __restrict__ manning, REAL* __restrict__ outflow,
    const REAL* __restrict__ width, const REAL* __restrict__ length,
    const REAL* __restrict__ elevation, REAL* __restrict__ cs_depth,
    const REAL* __restrict__ wse, const STO* __restrict__ total_storage,
    STO* __restrict__ outgoing_storage,
    REAL gravity, const REAL* __restrict__ time_step_ptr,
    long num_paths, int num_levels)
{
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t >= num_paths) return;
    REAL time_step = __ldg(time_step_ptr);

    int ci = cat_idx[t];
    int di = dn_idx[t];
    REAL blen = __ldg(length + t);

    REAL wse_c = __ldg(wse + ci);
    REAL wse_d = __ldg(wse + di);
    REAL max_wse = fmax(wse_c, wse_d);

    REAL slope = (wse_c - wse_d) / blen;
    slope = fmin(fmax(slope, (REAL)-0.005), (REAL)0.005);

    REAL ts_c = (REAL)total_storage[ci];
    REAL ts_d = (REAL)total_storage[di];

    // Carry the per-level updated flows in registers so each level is stored
    // exactly once, instead of storing then re-loading them for the limiter.
    // CaMa bifurcation maps use NPTHLEV <= 5; the generic path below keeps
    // correctness for any level count.
    constexpr int MAX_REGISTER_LEVELS = 8;
    REAL sum_out = (REAL)0;
    REAL upd[MAX_REGISTER_LEVELS];
    const bool fits = (num_levels <= MAX_REGISTER_LEVELS);

    if (fits) {
#pragma unroll
        for (int lv = 0; lv < MAX_REGISTER_LEVELS; ++lv) {
            REAL upd_o = (REAL)0;
            if (lv < num_levels) {
                long li = t * (long)num_levels + lv;
                REAL man = __ldg(manning + li);
                REAL csd = __ldg(cs_depth + li);
                REAL elv = __ldg(elevation + li);
                REAL upd_csd = fmax(max_wse - elv, (REAL)0);
                REAL sifd = fmax(sqrt(upd_csd * csd), sqrt(upd_csd * (REAL)0.01));
                if (sifd > (REAL)1e-5) {
                    REAL w = __ldg(width + li);
                    REAL o = outflow[li];
                    REAL unit_o = o / w;
                    REAL num = w * (unit_o + gravity * time_step * sifd * slope);
                    REAL den = (REAL)1 + gravity * time_step * (man * man)
                        * fabs(unit_o) * ((REAL)1 / pow73(sifd));
                    upd_o = num / den;
                }
                sum_out += upd_o;
                cs_depth[li] = upd_csd;
            }
            upd[lv] = upd_o;
        }
    } else {
        for (int lv = 0; lv < num_levels; ++lv) {
            long li = t * (long)num_levels + lv;
            REAL man = __ldg(manning + li);
            REAL csd = __ldg(cs_depth + li);
            REAL elv = __ldg(elevation + li);
            REAL upd_csd = fmax(max_wse - elv, (REAL)0);
            REAL sifd = fmax(sqrt(upd_csd * csd), sqrt(upd_csd * (REAL)0.01));
            bool flow_condition = sifd > (REAL)1e-5;
            REAL upd_o = (REAL)0;
            if (flow_condition) {
                REAL w = __ldg(width + li);
                REAL o = outflow[li];
                REAL unit_o = o / w;
                REAL num = w * (unit_o + gravity * time_step * sifd * slope);
                REAL den = (REAL)1 + gravity * time_step * (man * man)
                    * fabs(unit_o) * ((REAL)1 / pow73(sifd));
                upd_o = num / den;
            }
            sum_out += upd_o;
            cs_depth[li] = upd_csd;
            outflow[li] = upd_o;
        }
    }

    REAL limit_rate = fmin((REAL)0.05 * fmin(ts_c, ts_d) / (fabs(sum_out) * time_step), (REAL)1);
    sum_out *= limit_rate;
    if (fits) {
#pragma unroll
        for (int lv = 0; lv < MAX_REGISTER_LEVELS; ++lv) {
            if (lv < num_levels) {
                outflow[t * (long)num_levels + lv] = upd[lv] * limit_rate;
            }
        }
    } else {
        for (int lv = 0; lv < num_levels; ++lv) {
            long li = t * (long)num_levels + lv;
            outflow[li] = outflow[li] * limit_rate;
        }
    }

    REAL pos = fmax(sum_out, (REAL)0);
    REAL neg = fmin(sum_out, (REAL)0);
    atomicAdd(outgoing_storage + ci, (STO)(pos * time_step));
    atomicAdd(outgoing_storage + di, (STO)(-neg * time_step));
}

template <typename REAL, typename STO>
__global__ void k_bif_inflow(
    const int* __restrict__ cat_idx, const int* __restrict__ dn_idx,
    const REAL* __restrict__ limit_rate, REAL* __restrict__ outflow,
    STO* __restrict__ global_bif_outflow, long num_paths, int num_levels)
{
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t >= num_paths) return;

    int ci = cat_idx[t];
    int di = dn_idx[t];
    REAL lr_c = __ldg(limit_rate + ci);
    REAL lr_d = __ldg(limit_rate + di);

    REAL raw_sum = (REAL)0;
    for (int lv = 0; lv < num_levels; ++lv) {
        long li = t * (long)num_levels + lv;
        REAL o = outflow[li];
        raw_sum += o;
        REAL upd = (o >= (REAL)0) ? o * lr_c : o * lr_d;
        outflow[li] = upd;
    }
    REAL net = (raw_sum >= (REAL)0) ? raw_sum * lr_c : raw_sum * lr_d;
    atomicAdd(global_bif_outflow + ci, (STO)net);
    atomicAdd(global_bif_outflow + di, (STO)(-net));
}

void launch_bif_outflow(
    at::Tensor bifurcation_catchment_idx_ptr,
    at::Tensor bifurcation_downstream_idx_ptr,
    at::Tensor bifurcation_manning_ptr, at::Tensor bifurcation_outflow_ptr,
    at::Tensor bifurcation_width_ptr, at::Tensor bifurcation_length_ptr,
    at::Tensor bifurcation_elevation_ptr,
    at::Tensor bifurcation_cross_section_depth_ptr,
    at::Tensor water_surface_elevation_ptr, at::Tensor total_storage_ptr,
    at::Tensor outgoing_storage_ptr, float gravity,
    at::Tensor time_step_ptr, long num_catchments,
    long num_bifurcation_paths,
    int num_bifurcation_levels, long BLOCK_SIZE)
{
    (void)num_catchments;
    int grid = (int)((num_bifurcation_paths + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    bool real64 = (water_surface_elevation_ptr.scalar_type() == at::kDouble);
    bool sto64 = (total_storage_ptr.scalar_type() == at::kDouble);
#define LAUNCH_BIF_OUT(REAL_T, STO_T) \
        k_bif_outflow<REAL_T, STO_T><<<grid, (int)BLOCK_SIZE, 0, stream>>>( \
            bifurcation_catchment_idx_ptr.data_ptr<int>(), \
            bifurcation_downstream_idx_ptr.data_ptr<int>(), \
            bifurcation_manning_ptr.data_ptr<REAL_T>(), \
            bifurcation_outflow_ptr.data_ptr<REAL_T>(), \
            bifurcation_width_ptr.data_ptr<REAL_T>(), \
            bifurcation_length_ptr.data_ptr<REAL_T>(), \
            bifurcation_elevation_ptr.data_ptr<REAL_T>(), \
            bifurcation_cross_section_depth_ptr.data_ptr<REAL_T>(), \
            water_surface_elevation_ptr.data_ptr<REAL_T>(), \
            total_storage_ptr.data_ptr<STO_T>(), \
            outgoing_storage_ptr.data_ptr<STO_T>(), (REAL_T)gravity, \
            time_step_ptr.data_ptr<REAL_T>(), num_bifurcation_paths, \
            num_bifurcation_levels)
    if (real64) {
        LAUNCH_BIF_OUT(double, double);
    } else if (sto64) {
        LAUNCH_BIF_OUT(float, double);
    } else {
        LAUNCH_BIF_OUT(float, float);
    }
#undef LAUNCH_BIF_OUT
}

void launch_bif_inflow(
    at::Tensor bifurcation_catchment_idx_ptr,
    at::Tensor bifurcation_downstream_idx_ptr, at::Tensor limit_rate_ptr,
    at::Tensor bifurcation_outflow_ptr,
    at::Tensor global_bifurcation_outflow_ptr,
    long num_catchments, long num_bifurcation_paths,
    int num_bifurcation_levels,
    long BLOCK_SIZE)
{
    (void)num_catchments;
    int grid = (int)((num_bifurcation_paths + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    bool real64 = (bifurcation_outflow_ptr.scalar_type() == at::kDouble);
    bool sto64 = (global_bifurcation_outflow_ptr.scalar_type() == at::kDouble);
#define LAUNCH_BIF_IN(REAL_T, STO_T) \
        k_bif_inflow<REAL_T, STO_T><<<grid, (int)BLOCK_SIZE, 0, stream>>>( \
            bifurcation_catchment_idx_ptr.data_ptr<int>(), \
            bifurcation_downstream_idx_ptr.data_ptr<int>(), \
            limit_rate_ptr.data_ptr<REAL_T>(), \
            bifurcation_outflow_ptr.data_ptr<REAL_T>(), \
            global_bifurcation_outflow_ptr.data_ptr<STO_T>(), \
            num_bifurcation_paths, num_bifurcation_levels)
    if (real64) {
        LAUNCH_BIF_IN(double, double);
    } else if (sto64) {
        LAUNCH_BIF_IN(float, double);
    } else {
        LAUNCH_BIF_IN(float, float);
    }
#undef LAUNCH_BIF_IN
}
