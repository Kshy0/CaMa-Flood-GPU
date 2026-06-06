// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// CUDA backend for the outflow / inflow kernels.
//
// Mirrors cmfgpu/phys/triton/outflow.py::compute_outflow_kernel and
// compute_inflow_kernel.  Straight-line memory-bound kernels (downstream gather
// + atomic scatter), using lane-local skips for branch-heavy flood momentum.
// Templated on STO = hpfloat storage dtype (float/double): storage is always
// downcast to float for the physics, but pointer/atomic/store types follow STO.
// Compiled WITHOUT --use_fast_math to keep IEEE div/sqrt parity with Triton.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// ----------------------------------------------------------------------------
// (1) compute_outflow_kernel
// ----------------------------------------------------------------------------
template <typename STO>
__global__ void k_outflow(
    const int* __restrict__ downstream_idx,
    STO* __restrict__ river_inflow, float* __restrict__ river_outflow,
    const float* __restrict__ river_manning, const float* __restrict__ river_depth,
    const float* __restrict__ river_width, const float* __restrict__ river_length,
    const float* __restrict__ river_height, const STO* __restrict__ river_storage,
    STO* __restrict__ flood_inflow, float* __restrict__ flood_outflow,
    const float* __restrict__ flood_manning, const float* __restrict__ flood_depth,
    const float* __restrict__ protected_depth, const float* __restrict__ catchment_elevation,
    const float* __restrict__ downstream_distance, const STO* __restrict__ flood_storage,
    const STO* __restrict__ protected_storage,
    float* __restrict__ river_cross_section_depth, float* __restrict__ flood_cross_section_depth,
    float* __restrict__ flood_cross_section_area,
    STO* __restrict__ global_bifurcation_outflow, STO* __restrict__ total_storage,
    STO* __restrict__ outgoing_storage, float* __restrict__ water_surface_elevation,
    float* __restrict__ protected_water_surface_elevation,
    float gravity, const float* __restrict__ time_step_ptr,
    long num_catchments, int has_bifurcation,
    const bool* __restrict__ is_dam_upstream, int has_reservoir, float min_kinematic_slope)
{
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t >= num_catchments) return;
    float time_step = __ldg(time_step_ptr);

    int dn = downstream_idx[t];
    bool is_river_mouth = (dn == (int)t);

    float r_out = river_outflow[t];
    float r_man = __ldg(river_manning + t);
    float r_dep = __ldg(river_depth + t);
    float r_wid = __ldg(river_width + t);
    float r_len = __ldg(river_length + t);
    float r_hgt = __ldg(river_height + t);

    float f_out = flood_outflow[t];
    float f_man = __ldg(flood_manning + t);
    float f_dep = __ldg(flood_depth + t);
    float p_dep = __ldg(protected_depth + t);
    float c_elv = __ldg(catchment_elevation + t);
    float dn_dist = __ldg(downstream_distance + t);

    float r_cs_dep = __ldg(river_cross_section_depth + t);
    float f_cs_dep = __ldg(flood_cross_section_depth + t);
    float f_cs_area = __ldg(flood_cross_section_area + t);

    float rs = (float)river_storage[t];
    float fs = (float)flood_storage[t];
    float ps = (float)protected_storage[t];

    float river_elevation = c_elv - r_hgt;
    float wse = r_dep + river_elevation;
    float pwse = fminf(c_elv + p_dep, wse);
    float total_storage_f = rs + fs + ps;

    float r_dep_dn = __ldg(river_depth + dn);
    float r_hgt_dn = __ldg(river_height + dn);
    float c_elv_dn = __ldg(catchment_elevation + dn);
    float river_elevation_dn = c_elv_dn - r_hgt_dn;
    float wse_dn = r_dep_dn + river_elevation_dn;

    float max_wse = fmaxf(wse, wse_dn);
    if (is_river_mouth) wse_dn = c_elv;

    float river_slope = (wse - wse_dn) / dn_dist;
    float flood_slope = fminf(fmaxf(river_slope, -0.005f), 0.005f);

    float upd_r_cs_dep = max_wse - river_elevation;
    float r_sifd = fmaxf(sqrtf(upd_r_cs_dep * r_cs_dep), 1e-6f);
    float upd_f_cs_dep = fmaxf(max_wse - c_elv, 0.0f);
    float f_sifd = fmaxf(sqrtf(upd_f_cs_dep * f_cs_dep), 1e-6f);

    float upd_f_cs_area = fmaxf(fs / r_len - f_dep * r_wid, 0.0f);

    float r_cs_area = upd_r_cs_dep * r_wid;
    bool river_condition = (r_sifd > 1e-5f) && (r_cs_area > 1e-5f);
    float unit_r_out = r_out / r_wid;
    float num_r = r_wid * (unit_r_out + gravity * time_step * r_sifd * river_slope);
    float den_r = 1.0f + gravity * time_step * (r_man * r_man) * fabsf(unit_r_out)
                  * (1.0f / (r_sifd * r_sifd * cbrtf(r_sifd)));
    float upd_r_out = num_r / den_r;
    upd_r_out = river_condition ? upd_r_out : 0.0f;

    // Flood momentum is lane-local: dry lanes keep 0 and wet lanes pay the
    // expensive sqrt(f_imp_area) + cbrt momentum path.
    bool flood_condition = (f_sifd > 1e-5f) && (upd_f_cs_area > 1e-5f);
    float upd_f_out = 0.0f;
    if (flood_condition) {
        float f_imp_area = fmaxf(sqrtf(upd_f_cs_area * fmaxf(f_cs_area, 1e-6f)), 1e-6f);
        float num_f = f_out + gravity * time_step * f_imp_area * flood_slope;
        float den_f = 1.0f + gravity * time_step * (f_man * f_man) * fabsf(f_out)
                      * (1.0f / (f_sifd * cbrtf(f_sifd))) / f_imp_area;
        upd_f_out = num_f / den_f;
    }

    bool opposite_direction = (upd_r_out * upd_f_out) < 0.0f;
    if (opposite_direction) upd_f_out = 0.0f;
    bool is_negative_flow = (upd_r_out < 0.0f) && !is_river_mouth;
    float total_negative_flow = is_negative_flow ? (-upd_r_out - upd_f_out) * time_step : 1.0f;
    float limit_rate = fminf(
        is_negative_flow ? 0.05f * total_storage_f / total_negative_flow : 1.0f, 1.0f);
    if (is_negative_flow) { upd_r_out *= limit_rate; upd_f_out *= limit_rate; }

    if (has_reservoir) {
        bool is_dam_up = is_dam_upstream && is_dam_upstream[t];
        float bed_slope = (c_elv - c_elv_dn) / dn_dist;
        bed_slope = fmaxf(bed_slope, min_kinematic_slope);
        float kin_riv_vel = (1.0f / r_man) * sqrtf(bed_slope) * cbrtf(r_dep * r_dep);
        float kin_riv = r_wid * r_dep * kin_riv_vel;
        kin_riv = fminf(kin_riv, rs / time_step);
        kin_riv = fmaxf(kin_riv, 0.0f);
        float bed_slope_f = fminf(bed_slope, 0.005f);
        float kin_fld_vel = (1.0f / f_man) * sqrtf(bed_slope_f) * cbrtf(f_dep * f_dep);
        float kin_fld_area = fmaxf(fs / r_len - f_dep * r_wid, 0.0f);
        float kin_fld = kin_fld_area * kin_fld_vel;
        kin_fld = fminf(kin_fld, fs / time_step);
        kin_fld = fmaxf(kin_fld, 0.0f);
        if (is_dam_up) { upd_r_out = kin_riv; upd_f_out = kin_fld; }
    }

    river_outflow[t] = upd_r_out;
    flood_outflow[t] = upd_f_out;
    water_surface_elevation[t] = wse;
    protected_water_surface_elevation[t] = pwse;
    river_cross_section_depth[t] = upd_r_cs_dep;
    flood_cross_section_depth[t] = upd_f_cs_dep;
    flood_cross_section_area[t] = upd_f_cs_area;
    total_storage[t] = (STO)total_storage_f;

    river_inflow[t] = (STO)0;
    flood_inflow[t] = (STO)0;
    if (has_bifurcation) global_bifurcation_outflow[t] = (STO)0;

    float pos = fmaxf(upd_r_out, 0.0f) + fmaxf(upd_f_out, 0.0f);
    float neg = fminf(upd_r_out, 0.0f) + fminf(upd_f_out, 0.0f);
    atomicAdd(outgoing_storage + t, (STO)(pos * time_step));
    float to_add = is_river_mouth ? 0.0f : -neg * time_step;
    atomicAdd(outgoing_storage + dn, (STO)to_add);
}

// ----------------------------------------------------------------------------
// (2) compute_inflow_kernel
// ----------------------------------------------------------------------------
template <typename STO>
__global__ void k_inflow(
    const int* __restrict__ downstream_idx,
    float* __restrict__ river_outflow, float* __restrict__ flood_outflow,
    const STO* __restrict__ river_storage, const STO* __restrict__ flood_storage,
    const STO* __restrict__ outgoing_storage,
    STO* __restrict__ river_inflow, STO* __restrict__ flood_inflow,
    float* __restrict__ limit_rate_out, STO* __restrict__ reservoir_total_inflow,
    const bool* __restrict__ is_reservoir, long num_catchments, int has_reservoir)
{
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t >= num_catchments) return;

    float r_out = river_outflow[t];
    float f_out = flood_outflow[t];
    float outgoing = (float)outgoing_storage[t];
    float rate_storage = (float)(river_storage[t] + flood_storage[t]);
    float limit_rate = (outgoing > 1e-8f) ? fminf(rate_storage / outgoing, 1.0f) : 1.0f;

    int dn = downstream_idx[t];
    float outgoing_dn = (float)outgoing_storage[dn];
    float rate_storage_dn = (float)(river_storage[dn] + flood_storage[dn]);
    float limit_rate_dn = (outgoing_dn > 1e-8f) ? fminf(rate_storage_dn / outgoing_dn, 1.0f) : 1.0f;

    float upd_r_out = (r_out >= 0.0f) ? r_out * limit_rate : r_out * limit_rate_dn;
    float upd_f_out = (f_out >= 0.0f) ? f_out * limit_rate : f_out * limit_rate_dn;

    river_outflow[t] = upd_r_out;
    flood_outflow[t] = upd_f_out;
    limit_rate_out[t] = limit_rate;

    bool is_river_mouth = (dn == (int)t);
    if (!is_river_mouth) {
        atomicAdd(river_inflow + dn, (STO)upd_r_out);
        atomicAdd(flood_inflow + dn, (STO)upd_f_out);
        if (has_reservoir) {
            bool is_downstream_res = is_reservoir && (is_reservoir[dn] != 0);
            if (is_downstream_res)
                atomicAdd(reservoir_total_inflow + dn, (STO)(upd_r_out + upd_f_out));
        }
    }
}

// ----------------------------------------------------------------------------
// Host launchers (dispatch on storage dtype)
// ----------------------------------------------------------------------------
static inline float* PF(at::Tensor& x) { return x.numel() ? x.data_ptr<float>() : nullptr; }
static inline bool*  PB(at::Tensor& x) { return x.numel() ? x.data_ptr<bool>()  : nullptr; }
static inline int*   PI(at::Tensor& x) { return x.numel() ? x.data_ptr<int>()   : nullptr; }
template <typename STO> static inline STO* PS(at::Tensor& x) { return x.numel() ? x.data_ptr<STO>() : nullptr; }

template <typename STO>
static void launch_outflow_t(
    at::Tensor& di, at::Tensor& ri, at::Tensor& ro, at::Tensor& rman, at::Tensor& rd,
    at::Tensor& rw, at::Tensor& rl, at::Tensor& rh, at::Tensor& rs,
    at::Tensor& fi, at::Tensor& fo, at::Tensor& fman, at::Tensor& fd, at::Tensor& pd,
    at::Tensor& ce, at::Tensor& dd, at::Tensor& fsto, at::Tensor& psto,
    at::Tensor& rcsd, at::Tensor& fcsd, at::Tensor& fcsa,
    at::Tensor& gb, at::Tensor& ts_out, at::Tensor& outs, at::Tensor& wse, at::Tensor& pwse,
    float gravity, at::Tensor& tsp, long n, int has_bif,
    at::Tensor& dam, int has_res, float minslope, int block)
{
    int grid = (int)((n + block - 1) / block);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    k_outflow<STO><<<grid, block, 0, stream>>>(
        PI(di), PS<STO>(ri), PF(ro), PF(rman), PF(rd), PF(rw), PF(rl), PF(rh), PS<STO>(rs),
        PS<STO>(fi), PF(fo), PF(fman), PF(fd), PF(pd), PF(ce), PF(dd), PS<STO>(fsto), PS<STO>(psto),
        PF(rcsd), PF(fcsd), PF(fcsa),
        PS<STO>(gb), PS<STO>(ts_out), PS<STO>(outs), PF(wse), PF(pwse),
        gravity, PF(tsp), n, has_bif, PB(dam), has_res, minslope);
}

void launch_outflow(
    at::Tensor downstream_idx,
    at::Tensor river_inflow, at::Tensor river_outflow, at::Tensor river_manning,
    at::Tensor river_depth, at::Tensor river_width, at::Tensor river_length,
    at::Tensor river_height, at::Tensor river_storage,
    at::Tensor flood_inflow, at::Tensor flood_outflow, at::Tensor flood_manning,
    at::Tensor flood_depth, at::Tensor protected_depth, at::Tensor catchment_elevation,
    at::Tensor downstream_distance, at::Tensor flood_storage, at::Tensor protected_storage,
    at::Tensor river_cross_section_depth, at::Tensor flood_cross_section_depth,
    at::Tensor flood_cross_section_area,
    at::Tensor global_bifurcation_outflow, at::Tensor total_storage,
    at::Tensor outgoing_storage, at::Tensor water_surface_elevation,
    at::Tensor protected_water_surface_elevation,
    double gravity, at::Tensor time_step,
    long num_catchments, long has_bifurcation,
    at::Tensor is_dam_upstream, long has_reservoir, double min_kinematic_slope, long block)
{
    if (river_storage.scalar_type() == at::kDouble)
        launch_outflow_t<double>(downstream_idx, river_inflow, river_outflow, river_manning,
            river_depth, river_width, river_length, river_height, river_storage,
            flood_inflow, flood_outflow, flood_manning, flood_depth, protected_depth,
            catchment_elevation, downstream_distance, flood_storage, protected_storage,
            river_cross_section_depth, flood_cross_section_depth, flood_cross_section_area,
            global_bifurcation_outflow, total_storage, outgoing_storage,
            water_surface_elevation, protected_water_surface_elevation,
            (float)gravity, time_step, num_catchments, (int)has_bifurcation,
            is_dam_upstream, (int)has_reservoir, (float)min_kinematic_slope, (int)block);
    else
        launch_outflow_t<float>(downstream_idx, river_inflow, river_outflow, river_manning,
            river_depth, river_width, river_length, river_height, river_storage,
            flood_inflow, flood_outflow, flood_manning, flood_depth, protected_depth,
            catchment_elevation, downstream_distance, flood_storage, protected_storage,
            river_cross_section_depth, flood_cross_section_depth, flood_cross_section_area,
            global_bifurcation_outflow, total_storage, outgoing_storage,
            water_surface_elevation, protected_water_surface_elevation,
            (float)gravity, time_step, num_catchments, (int)has_bifurcation,
            is_dam_upstream, (int)has_reservoir, (float)min_kinematic_slope, (int)block);
}

template <typename STO>
static void launch_inflow_t(
    at::Tensor& di, at::Tensor& ro, at::Tensor& fo, at::Tensor& rs, at::Tensor& fs,
    at::Tensor& outs, at::Tensor& ri, at::Tensor& fi, at::Tensor& lr,
    at::Tensor& rti, at::Tensor& isres, long n, int has_res, int block)
{
    int grid = (int)((n + block - 1) / block);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    k_inflow<STO><<<grid, block, 0, stream>>>(
        PI(di), PF(ro), PF(fo), PS<STO>(rs), PS<STO>(fs), PS<STO>(outs),
        PS<STO>(ri), PS<STO>(fi), PF(lr), PS<STO>(rti), PB(isres), n, has_res);
}

void launch_inflow(
    at::Tensor downstream_idx,
    at::Tensor river_outflow, at::Tensor flood_outflow,
    at::Tensor river_storage, at::Tensor flood_storage, at::Tensor outgoing_storage,
    at::Tensor river_inflow, at::Tensor flood_inflow, at::Tensor limit_rate,
    at::Tensor reservoir_total_inflow, at::Tensor is_reservoir,
    long num_catchments, long has_reservoir, long block)
{
    if (river_storage.scalar_type() == at::kDouble)
        launch_inflow_t<double>(downstream_idx, river_outflow, flood_outflow, river_storage,
            flood_storage, outgoing_storage, river_inflow, flood_inflow, limit_rate,
            reservoir_total_inflow, is_reservoir, num_catchments, (int)has_reservoir, (int)block);
    else
        launch_inflow_t<float>(downstream_idx, river_outflow, flood_outflow, river_storage,
            flood_storage, outgoing_storage, river_inflow, flood_inflow, limit_rate,
            reservoir_total_inflow, is_reservoir, num_catchments, (int)has_reservoir, (int)block);
}
