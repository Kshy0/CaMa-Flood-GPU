#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>


template <bool HAS_BIFURCATION, typename REAL, typename STO>
__device__ __forceinline__ void k_inflow_cell(
    const int* __restrict__ downstream_idx,
    REAL* __restrict__ river_outflow, REAL* __restrict__ flood_outflow,
    const STO* __restrict__ river_storage, const STO* __restrict__ flood_storage,
    const STO* __restrict__ outgoing_storage,
    STO* __restrict__ river_inflow, STO* __restrict__ flood_inflow,
    REAL* __restrict__ limit_rate_out, STO* __restrict__ reservoir_total_inflow,
    const bool* __restrict__ is_reservoir, long num_catchments,
    int has_reservoir)
{
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t >= num_catchments) return;

    REAL r_out = river_outflow[t];
    REAL f_out = flood_outflow[t];
    REAL outgoing = (REAL)outgoing_storage[t];
    REAL rate_storage = (REAL)(river_storage[t] + flood_storage[t]);
    REAL limit_rate = (outgoing > (REAL)1e-8) ? fmin(rate_storage / outgoing, (REAL)1.0) : (REAL)1.0;

    int dn = downstream_idx[t];
    REAL outgoing_dn = (REAL)outgoing_storage[dn];
    REAL rate_storage_dn = (REAL)(river_storage[dn] + flood_storage[dn]);
    REAL limit_rate_dn = (outgoing_dn > (REAL)1e-8) ? fmin(rate_storage_dn / outgoing_dn, (REAL)1.0) : (REAL)1.0;

    REAL upd_r_out = (r_out >= (REAL)0.0) ? r_out * limit_rate : r_out * limit_rate_dn;
    REAL upd_f_out = (f_out >= (REAL)0.0) ? f_out * limit_rate : f_out * limit_rate_dn;

    river_outflow[t] = upd_r_out;
    flood_outflow[t] = upd_f_out;
    if constexpr (HAS_BIFURCATION) limit_rate_out[t] = limit_rate;

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

template <bool HAS_BIFURCATION, typename REAL, typename STO>
__global__ void k_inflow(
    const int* __restrict__ downstream_idx,
    REAL* __restrict__ river_outflow, REAL* __restrict__ flood_outflow,
    const STO* __restrict__ river_storage, const STO* __restrict__ flood_storage,
    const STO* __restrict__ outgoing_storage,
    STO* __restrict__ river_inflow, STO* __restrict__ flood_inflow,
    REAL* __restrict__ limit_rate_out, STO* __restrict__ reservoir_total_inflow,
    const bool* __restrict__ is_reservoir, long num_catchments,
    int has_reservoir)
{
    k_inflow_cell<HAS_BIFURCATION, REAL, STO>(
        downstream_idx, river_outflow, flood_outflow, river_storage, flood_storage,
        outgoing_storage, river_inflow, flood_inflow, limit_rate_out, reservoir_total_inflow,
        is_reservoir, num_catchments, has_reservoir);
}

template <bool HAS_BIFURCATION, typename REAL, typename STO>
__global__ void k_inflow_batched(
    const int* __restrict__ downstream_idx,
    REAL* __restrict__ river_outflow, REAL* __restrict__ flood_outflow,
    const STO* __restrict__ river_storage, const STO* __restrict__ flood_storage,
    const STO* __restrict__ outgoing_storage,
    STO* __restrict__ river_inflow, STO* __restrict__ flood_inflow,
    REAL* __restrict__ limit_rate_out, STO* __restrict__ reservoir_total_inflow,
    const bool* __restrict__ is_reservoir, long num_catchments,
    int has_reservoir)
{
    const long member_offset = (long)blockIdx.y * num_catchments;
    river_outflow += member_offset;
    flood_outflow += member_offset;
    river_storage += member_offset;
    flood_storage += member_offset;
    outgoing_storage += member_offset;
    river_inflow += member_offset;
    flood_inflow += member_offset;
    if constexpr (HAS_BIFURCATION) limit_rate_out += member_offset;
    if (has_reservoir) reservoir_total_inflow += member_offset;
    k_inflow_cell<HAS_BIFURCATION, REAL, STO>(
        downstream_idx, river_outflow, flood_outflow, river_storage, flood_storage,
        outgoing_storage, river_inflow, flood_inflow, limit_rate_out, reservoir_total_inflow,
        is_reservoir, num_catchments, has_reservoir);
}

template <typename REAL, typename STO>
static void launch_inflow_t(
    at::Tensor& di, at::Tensor& ro, at::Tensor& fo, at::Tensor& rs, at::Tensor& fs,
    at::Tensor& outs, at::Tensor& ri, at::Tensor& fi,
    c10::optional<at::Tensor>& lr,
    c10::optional<at::Tensor>& rti, c10::optional<at::Tensor>& isres,
    long n, int has_bif, int has_res, long ensemble_size, int block)
{
    const dim3 grid((n + block - 1) / block, ensemble_size);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
#define LAUNCH_INFLOW(HAS_BIFURCATION) \
    do { \
        if (ensemble_size > 1) { \
            k_inflow_batched<HAS_BIFURCATION, REAL, STO><<<grid, block, 0, stream>>>( \
                di.data_ptr<int>(), ro.data_ptr<REAL>(), fo.data_ptr<REAL>(), \
                rs.data_ptr<STO>(), fs.data_ptr<STO>(), outs.data_ptr<STO>(), \
                ri.data_ptr<STO>(), fi.data_ptr<STO>(), \
                lr ? lr->data_ptr<REAL>() : nullptr, \
                rti ? rti->data_ptr<STO>() : nullptr, \
                isres ? isres->data_ptr<bool>() : nullptr, n, has_res); \
        } else { \
            k_inflow<HAS_BIFURCATION, REAL, STO><<<grid, block, 0, stream>>>( \
                di.data_ptr<int>(), ro.data_ptr<REAL>(), fo.data_ptr<REAL>(), \
                rs.data_ptr<STO>(), fs.data_ptr<STO>(), outs.data_ptr<STO>(), \
                ri.data_ptr<STO>(), fi.data_ptr<STO>(), \
                lr ? lr->data_ptr<REAL>() : nullptr, \
                rti ? rti->data_ptr<STO>() : nullptr, \
                isres ? isres->data_ptr<bool>() : nullptr, n, has_res); \
        } \
    } while (false)
    if (has_bif) LAUNCH_INFLOW(true);
    else LAUNCH_INFLOW(false);
#undef LAUNCH_INFLOW
}

void launch_inflow(
    at::Tensor downstream_idx_ptr,
    at::Tensor river_outflow_ptr, at::Tensor flood_outflow_ptr,
    at::Tensor river_storage_ptr, at::Tensor flood_storage_ptr,
    at::Tensor outgoing_storage_ptr, at::Tensor river_inflow_ptr,
    at::Tensor flood_inflow_ptr,
    c10::optional<at::Tensor> limit_rate_ptr,
    c10::optional<at::Tensor> reservoir_total_inflow_ptr,
    c10::optional<at::Tensor> is_reservoir_ptr,
    long num_catchments, bool HAS_BIFURCATION, bool HAS_RESERVOIR,
    long ensemble_size, long BLOCK_SIZE)
{
    if (river_outflow_ptr.scalar_type() == at::kDouble)
        launch_inflow_t<double, double>(
            downstream_idx_ptr, river_outflow_ptr, flood_outflow_ptr,
            river_storage_ptr, flood_storage_ptr, outgoing_storage_ptr,
            river_inflow_ptr, flood_inflow_ptr, limit_rate_ptr,
            reservoir_total_inflow_ptr, is_reservoir_ptr, num_catchments,
            (int)HAS_BIFURCATION, (int)HAS_RESERVOIR, ensemble_size, (int)BLOCK_SIZE);
    else if (river_storage_ptr.scalar_type() == at::kDouble)
        launch_inflow_t<float, double>(
            downstream_idx_ptr, river_outflow_ptr, flood_outflow_ptr,
            river_storage_ptr, flood_storage_ptr, outgoing_storage_ptr,
            river_inflow_ptr, flood_inflow_ptr, limit_rate_ptr,
            reservoir_total_inflow_ptr, is_reservoir_ptr, num_catchments,
            (int)HAS_BIFURCATION, (int)HAS_RESERVOIR, ensemble_size, (int)BLOCK_SIZE);
    else
        launch_inflow_t<float, float>(
            downstream_idx_ptr, river_outflow_ptr, flood_outflow_ptr,
            river_storage_ptr, flood_storage_ptr, outgoing_storage_ptr,
            river_inflow_ptr, flood_inflow_ptr, limit_rate_ptr,
            reservoir_total_inflow_ptr, is_reservoir_ptr, num_catchments,
            (int)HAS_BIFURCATION, (int)HAS_RESERVOIR, ensemble_size, (int)BLOCK_SIZE);
}

// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>


template <bool BASE_ONLY, typename REAL, typename STO>
__device__ __forceinline__ void k_outflow_cell(
    const int* __restrict__ downstream_idx,
    STO* __restrict__ river_inflow, REAL* __restrict__ river_outflow,
    const REAL* __restrict__ river_manning, const REAL* __restrict__ river_depth,
    const REAL* __restrict__ river_width, const REAL* __restrict__ river_length,
    const REAL* __restrict__ river_height, const STO* __restrict__ river_storage,
    STO* __restrict__ flood_inflow, REAL* __restrict__ flood_outflow,
    const REAL* __restrict__ flood_manning, const REAL* __restrict__ flood_depth,
    const REAL* __restrict__ catchment_elevation,
    const REAL* __restrict__ downstream_distance, const STO* __restrict__ flood_storage,
    const STO* __restrict__ protected_storage,
    REAL* __restrict__ river_cross_section_depth, REAL* __restrict__ flood_cross_section_depth,
    REAL* __restrict__ flood_cross_section_area,
    STO* __restrict__ global_bifurcation_outflow,
    STO* __restrict__ outgoing_storage,
    REAL gravity, const REAL* __restrict__ time_step_ptr,
    long num_catchments, int has_bifurcation, int has_levee,
    const bool* __restrict__ is_dam_upstream, int has_reservoir, REAL min_kinematic_slope,
    const REAL* __restrict__ sea_surface_elevation,
    const int* __restrict__ catchment_sea_level_idx, int has_sea_level)
{
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t >= num_catchments) return;
    REAL time_step = __ldg(time_step_ptr);

    int dn = downstream_idx[t];
    bool is_river_mouth = (dn == (int)t);

    REAL r_out = river_outflow[t];
    REAL r_man = __ldg(river_manning + t);
    REAL r_dep = __ldg(river_depth + t);
    REAL r_wid = __ldg(river_width + t);
    REAL r_len = __ldg(river_length + t);
    REAL r_hgt = __ldg(river_height + t);

    REAL f_out = flood_outflow[t];
    REAL f_man = __ldg(flood_manning + t);
    REAL f_dep = __ldg(flood_depth + t);
    REAL c_elv = __ldg(catchment_elevation + t);
    REAL dn_dist = __ldg(downstream_distance + t);

    REAL r_cs_dep = __ldg(river_cross_section_depth + t);
    REAL f_cs_dep = __ldg(flood_cross_section_depth + t);
    REAL f_cs_area = __ldg(flood_cross_section_area + t);

    REAL rs = (REAL)river_storage[t];
    REAL fs = (REAL)flood_storage[t];
    REAL ps = has_levee ? (REAL)protected_storage[t] : (REAL)0;

    REAL river_elevation = c_elv - r_hgt;
    REAL wse = r_dep + river_elevation;
    REAL total_storage_f = rs + fs + ps;

    REAL r_dep_dn = __ldg(river_depth + dn);
    REAL r_hgt_dn = __ldg(river_height + dn);
    REAL c_elv_dn = __ldg(catchment_elevation + dn);
    REAL river_elevation_dn = c_elv_dn - r_hgt_dn;
    REAL wse_dn = r_dep_dn + river_elevation_dn;

    if (is_river_mouth) wse_dn = c_elv;
    if constexpr (!BASE_ONLY) {
        if (has_sea_level) {
            int sea_idx = __ldg(catchment_sea_level_idx + t);
            if (sea_idx >= 0) wse_dn = __ldg(sea_surface_elevation + sea_idx);
        }
    }
    REAL max_wse = fmax(wse, wse_dn);

    REAL river_slope = (wse - wse_dn) / dn_dist;
    REAL flood_slope = fmin(fmax(river_slope, (REAL)-CMF_ROUTING_SLOPE_LIMIT), (REAL)CMF_ROUTING_SLOPE_LIMIT);

    // The downstream boundary controls mouth slope, but not local flow depth.
    REAL upd_r_cs_dep = is_river_mouth ? r_dep : max_wse - river_elevation;
    REAL r_sifd = fmax(sqrt(upd_r_cs_dep * r_cs_dep), (REAL)1e-6);
    REAL flood_surface = is_river_mouth ? wse : max_wse;
    REAL upd_f_cs_dep = fmax(flood_surface - c_elv, (REAL)0.0);
    REAL f_sifd = fmax(sqrt(upd_f_cs_dep * f_cs_dep), (REAL)1e-6);

    REAL upd_f_cs_area = fmax(fs / r_len - f_dep * r_wid, (REAL)0.0);

    REAL r_cs_area = upd_r_cs_dep * r_wid;
    bool river_condition = (r_sifd > (REAL)1e-5) && (r_cs_area > (REAL)1e-5);
    REAL upd_r_out = (REAL)0.0;
    if (river_condition || sizeof(REAL) == sizeof(float)) {
        REAL unit_r_out = r_out / r_wid;
        REAL num_r = r_wid * (unit_r_out + gravity * time_step * r_sifd * river_slope);
        REAL den_r = (REAL)1.0 + gravity * time_step * (r_man * r_man) * fabs(unit_r_out)
                      * ((REAL)1.0 / (r_sifd * r_sifd * cbrt(r_sifd)));
        upd_r_out = num_r / den_r;
    }
    upd_r_out = river_condition ? upd_r_out : (REAL)0.0;

    // Flood momentum is lane-local: dry lanes keep 0 and wet lanes pay the
    // expensive sqrt(f_imp_area) + cbrt momentum path.
    bool flood_condition = (f_sifd > (REAL)1e-5) && (upd_f_cs_area > (REAL)1e-5);
    REAL upd_f_out = (REAL)0.0;
    if (flood_condition) {
        REAL f_imp_area = fmax(sqrt(upd_f_cs_area * fmax(f_cs_area, (REAL)1e-6)), (REAL)1e-6);
        REAL num_f = f_out + gravity * time_step * f_imp_area * flood_slope;
        REAL den_f = (REAL)1.0 + gravity * time_step * (f_man * f_man) * fabs(f_out)
                      * ((REAL)1.0 / (f_sifd * cbrt(f_sifd))) / f_imp_area;
        upd_f_out = num_f / den_f;
    }

    bool opposite_direction = (upd_r_out * upd_f_out) < (REAL)0.0;
    if (opposite_direction) upd_f_out = (REAL)0.0;
    bool is_negative_flow = (upd_r_out < (REAL)0.0) && !is_river_mouth;
    REAL total_negative_flow = is_negative_flow ? (-upd_r_out - upd_f_out) * time_step : (REAL)1.0;
    REAL limit_rate = fmin(
        is_negative_flow ? (REAL)CMF_BACKFLOW_STORAGE_FRACTION * total_storage_f / total_negative_flow : (REAL)1.0, (REAL)1.0);
    if (is_negative_flow) { upd_r_out *= limit_rate; upd_f_out *= limit_rate; }

    if constexpr (!BASE_ONLY) {
        if (has_reservoir &&
            (sizeof(REAL) == sizeof(float) || (is_dam_upstream && is_dam_upstream[t]))) {
            REAL bed_slope = (c_elv - c_elv_dn) / dn_dist;
            bed_slope = fmax(bed_slope, min_kinematic_slope);
            REAL kin_riv_vel = ((REAL)1.0 / r_man) * sqrt(bed_slope) * cbrt(r_dep * r_dep);
            REAL kin_riv = r_wid * r_dep * kin_riv_vel;
            kin_riv = fmin(kin_riv, rs / time_step);
            kin_riv = fmax(kin_riv, (REAL)0.0);
            REAL bed_slope_f = fmin(bed_slope, (REAL)CMF_ROUTING_SLOPE_LIMIT);
            REAL kin_fld_vel = ((REAL)1.0 / f_man) * sqrt(bed_slope_f) * cbrt(f_dep * f_dep);
            REAL kin_fld_area = fmax(fs / r_len - f_dep * r_wid, (REAL)0.0);
            REAL kin_fld = kin_fld_area * kin_fld_vel;
            kin_fld = fmin(kin_fld, fs / time_step);
            kin_fld = fmax(kin_fld, (REAL)0.0);
            if (is_dam_upstream && is_dam_upstream[t]) {
                upd_r_out = kin_riv;
                upd_f_out = kin_fld;
            }
        }
    }

    river_outflow[t] = upd_r_out;
    flood_outflow[t] = upd_f_out;
    river_cross_section_depth[t] = upd_r_cs_dep;
    flood_cross_section_depth[t] = upd_f_cs_dep;
    flood_cross_section_area[t] = upd_f_cs_area;

    river_inflow[t] = (STO)0;
    flood_inflow[t] = (STO)0;
    if (has_bifurcation) global_bifurcation_outflow[t] = (STO)0;

    REAL pos = fmax(upd_r_out, (REAL)0.0) + fmax(upd_f_out, (REAL)0.0);
    REAL neg = fmin(upd_r_out, (REAL)0.0) + fmin(upd_f_out, (REAL)0.0);
    atomicAdd(outgoing_storage + t, (STO)(pos * time_step));
    REAL to_add = is_river_mouth ? (REAL)0.0 : -neg * time_step;
    atomicAdd(outgoing_storage + dn, (STO)to_add);
}

template <bool BASE_ONLY, typename REAL, typename STO>
__global__ void k_outflow(
    const int* __restrict__ downstream_idx,
    STO* __restrict__ river_inflow, REAL* __restrict__ river_outflow,
    const REAL* __restrict__ river_manning, const REAL* __restrict__ river_depth,
    const REAL* __restrict__ river_width, const REAL* __restrict__ river_length,
    const REAL* __restrict__ river_height, const STO* __restrict__ river_storage,
    STO* __restrict__ flood_inflow, REAL* __restrict__ flood_outflow,
    const REAL* __restrict__ flood_manning, const REAL* __restrict__ flood_depth,
    const REAL* __restrict__ catchment_elevation,
    const REAL* __restrict__ downstream_distance, const STO* __restrict__ flood_storage,
    const STO* __restrict__ protected_storage,
    REAL* __restrict__ river_cross_section_depth, REAL* __restrict__ flood_cross_section_depth,
    REAL* __restrict__ flood_cross_section_area,
    STO* __restrict__ global_bifurcation_outflow,
    STO* __restrict__ outgoing_storage,
    REAL gravity, const REAL* __restrict__ time_step_ptr,
    long num_catchments, int has_bifurcation, int has_levee,
    const bool* __restrict__ is_dam_upstream, int has_reservoir, REAL min_kinematic_slope,
    const REAL* __restrict__ sea_surface_elevation,
    const int* __restrict__ catchment_sea_level_idx, int has_sea_level)
{
    k_outflow_cell<BASE_ONLY, REAL, STO>(
        downstream_idx, river_inflow, river_outflow, river_manning, river_depth, river_width,
        river_length, river_height, river_storage, flood_inflow, flood_outflow, flood_manning,
        flood_depth, catchment_elevation, downstream_distance, flood_storage, protected_storage,
        river_cross_section_depth, flood_cross_section_depth, flood_cross_section_area,
        global_bifurcation_outflow, outgoing_storage, gravity, time_step_ptr, num_catchments,
        has_bifurcation, has_levee, is_dam_upstream, has_reservoir, min_kinematic_slope,
        sea_surface_elevation, catchment_sea_level_idx, has_sea_level);
}

template <bool BASE_ONLY, typename REAL, typename STO>
__global__ void k_outflow_batched(
    const int* __restrict__ downstream_idx,
    STO* __restrict__ river_inflow, REAL* __restrict__ river_outflow,
    const REAL* __restrict__ river_manning, const REAL* __restrict__ river_depth,
    const REAL* __restrict__ river_width, const REAL* __restrict__ river_length,
    const REAL* __restrict__ river_height, const STO* __restrict__ river_storage,
    STO* __restrict__ flood_inflow, REAL* __restrict__ flood_outflow,
    const REAL* __restrict__ flood_manning, const REAL* __restrict__ flood_depth,
    const REAL* __restrict__ catchment_elevation,
    const REAL* __restrict__ downstream_distance, const STO* __restrict__ flood_storage,
    const STO* __restrict__ protected_storage,
    REAL* __restrict__ river_cross_section_depth, REAL* __restrict__ flood_cross_section_depth,
    REAL* __restrict__ flood_cross_section_area,
    STO* __restrict__ global_bifurcation_outflow,
    STO* __restrict__ outgoing_storage,
    REAL gravity, const REAL* __restrict__ time_step_ptr,
    long num_catchments, int has_bifurcation, int has_levee,
    const bool* __restrict__ is_dam_upstream, int has_reservoir, REAL min_kinematic_slope,
    const REAL* __restrict__ sea_surface_elevation,
    const int* __restrict__ catchment_sea_level_idx, int has_sea_level,
    long num_sea_level_boundaries,
    bool batched_river_manning, bool batched_river_width,
    bool batched_river_length, bool batched_river_height,
    bool batched_flood_manning, bool batched_catchment_elevation,
    bool batched_downstream_distance, bool batched_sea_surface_elevation)
{
    const long member_offset = (long)blockIdx.y * num_catchments;
    river_inflow += member_offset;
    river_outflow += member_offset;
    river_depth += member_offset;
    river_storage += member_offset;
    flood_inflow += member_offset;
    flood_outflow += member_offset;
    flood_depth += member_offset;
    flood_storage += member_offset;
    river_cross_section_depth += member_offset;
    flood_cross_section_depth += member_offset;
    flood_cross_section_area += member_offset;
    outgoing_storage += member_offset;
    if (has_levee) protected_storage += member_offset;
    if (has_bifurcation) global_bifurcation_outflow += member_offset;
    if (batched_river_manning) river_manning += member_offset;
    if (batched_river_width) river_width += member_offset;
    if (batched_river_length) river_length += member_offset;
    if (batched_river_height) river_height += member_offset;
    if (batched_flood_manning) flood_manning += member_offset;
    if (batched_catchment_elevation) catchment_elevation += member_offset;
    if (batched_downstream_distance) downstream_distance += member_offset;
    if (has_sea_level && batched_sea_surface_elevation)
        sea_surface_elevation += (long)blockIdx.y * num_sea_level_boundaries;
    k_outflow_cell<BASE_ONLY, REAL, STO>(
        downstream_idx, river_inflow, river_outflow, river_manning, river_depth, river_width,
        river_length, river_height, river_storage, flood_inflow, flood_outflow, flood_manning,
        flood_depth, catchment_elevation, downstream_distance, flood_storage, protected_storage,
        river_cross_section_depth, flood_cross_section_depth, flood_cross_section_area,
        global_bifurcation_outflow, outgoing_storage, gravity, time_step_ptr, num_catchments,
        has_bifurcation, has_levee, is_dam_upstream, has_reservoir, min_kinematic_slope,
        sea_surface_elevation, catchment_sea_level_idx, has_sea_level);
}

template <typename REAL, typename STO>
static void launch_outflow_t(
    at::Tensor& di, at::Tensor& ri, at::Tensor& ro, at::Tensor& rman, at::Tensor& rd,
    at::Tensor& rw, at::Tensor& rl, at::Tensor& rh, at::Tensor& rs,
    at::Tensor& fi, at::Tensor& fo, at::Tensor& fman, at::Tensor& fd,
    at::Tensor& ce, at::Tensor& dd, at::Tensor& fsto,
    c10::optional<at::Tensor>& psto,
    at::Tensor& rcsd, at::Tensor& fcsd, at::Tensor& fcsa,
    c10::optional<at::Tensor>& gb, at::Tensor& outs,
    REAL gravity, at::Tensor& tsp, long n, int has_bif, int has_levee,
    c10::optional<at::Tensor>& dam, int has_res, REAL minslope,
    c10::optional<at::Tensor>& sea, c10::optional<at::Tensor>& sea_idx,
    int has_sea, long num_sea_level_boundaries, long ensemble_size,
    bool batched_river_manning, bool batched_river_width,
    bool batched_river_length, bool batched_river_height,
    bool batched_flood_manning, bool batched_catchment_elevation,
    bool batched_downstream_distance, bool batched_sea_surface_elevation,
    int block)
{
    const dim3 grid((n + block - 1) / block, ensemble_size);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
#define LAUNCH_OUTFLOW(BASE_ONLY) \
    do { \
        if (ensemble_size > 1) { \
            k_outflow_batched<BASE_ONLY, REAL, STO><<<grid, block, 0, stream>>>( \
                di.data_ptr<int>(), ri.data_ptr<STO>(), ro.data_ptr<REAL>(), \
                rman.data_ptr<REAL>(), rd.data_ptr<REAL>(), rw.data_ptr<REAL>(), \
                rl.data_ptr<REAL>(), rh.data_ptr<REAL>(), rs.data_ptr<STO>(), \
                fi.data_ptr<STO>(), fo.data_ptr<REAL>(), fman.data_ptr<REAL>(), \
                fd.data_ptr<REAL>(), ce.data_ptr<REAL>(), dd.data_ptr<REAL>(), \
                fsto.data_ptr<STO>(), psto ? psto->data_ptr<STO>() : nullptr, \
                rcsd.data_ptr<REAL>(), fcsd.data_ptr<REAL>(), fcsa.data_ptr<REAL>(), \
                gb ? gb->data_ptr<STO>() : nullptr, outs.data_ptr<STO>(), \
                gravity, tsp.data_ptr<REAL>(), n, has_bif, has_levee, \
                dam ? dam->data_ptr<bool>() : nullptr, has_res, minslope, \
                sea ? sea->data_ptr<REAL>() : nullptr, \
                sea_idx ? sea_idx->data_ptr<int>() : nullptr, has_sea, num_sea_level_boundaries, \
                batched_river_manning, batched_river_width, batched_river_length, \
                batched_river_height, batched_flood_manning, batched_catchment_elevation, \
                batched_downstream_distance, batched_sea_surface_elevation); \
        } else { \
            k_outflow<BASE_ONLY, REAL, STO><<<grid, block, 0, stream>>>( \
                di.data_ptr<int>(), ri.data_ptr<STO>(), ro.data_ptr<REAL>(), \
                rman.data_ptr<REAL>(), rd.data_ptr<REAL>(), rw.data_ptr<REAL>(), \
                rl.data_ptr<REAL>(), rh.data_ptr<REAL>(), rs.data_ptr<STO>(), \
                fi.data_ptr<STO>(), fo.data_ptr<REAL>(), fman.data_ptr<REAL>(), \
                fd.data_ptr<REAL>(), ce.data_ptr<REAL>(), dd.data_ptr<REAL>(), \
                fsto.data_ptr<STO>(), psto ? psto->data_ptr<STO>() : nullptr, \
                rcsd.data_ptr<REAL>(), fcsd.data_ptr<REAL>(), fcsa.data_ptr<REAL>(), \
                gb ? gb->data_ptr<STO>() : nullptr, outs.data_ptr<STO>(), \
                gravity, tsp.data_ptr<REAL>(), n, has_bif, has_levee, \
                dam ? dam->data_ptr<bool>() : nullptr, has_res, minslope, \
                sea ? sea->data_ptr<REAL>() : nullptr, \
                sea_idx ? sea_idx->data_ptr<int>() : nullptr, has_sea); \
        } \
    } while (false)
    if (!has_res && !has_sea) LAUNCH_OUTFLOW(true);
    else LAUNCH_OUTFLOW(false);
#undef LAUNCH_OUTFLOW
}

void launch_outflow(
    at::Tensor downstream_idx_ptr, at::Tensor river_inflow_ptr,
    at::Tensor river_outflow_ptr, at::Tensor river_manning_ptr,
    at::Tensor river_depth_ptr, at::Tensor river_width_ptr,
    at::Tensor river_length_ptr, at::Tensor river_height_ptr,
    at::Tensor river_storage_ptr, at::Tensor flood_inflow_ptr,
    at::Tensor flood_outflow_ptr, at::Tensor flood_manning_ptr,
    at::Tensor flood_depth_ptr,
    at::Tensor catchment_elevation_ptr, at::Tensor downstream_distance_ptr,
    at::Tensor flood_storage_ptr,
    c10::optional<at::Tensor> protected_storage_ptr,
    at::Tensor river_cross_section_depth_ptr,
    at::Tensor flood_cross_section_depth_ptr,
    at::Tensor flood_cross_section_area_ptr,
    c10::optional<at::Tensor> global_bifurcation_outflow_ptr,
    at::Tensor outgoing_storage_ptr,
    double gravity, at::Tensor time_step_ptr, long num_catchments,
    bool HAS_BIFURCATION, bool HAS_LEVEE,
    c10::optional<at::Tensor> is_dam_upstream_ptr, bool HAS_RESERVOIR,
    double min_kinematic_slope,
    c10::optional<at::Tensor> sea_surface_elevation_ptr,
    c10::optional<at::Tensor> catchment_sea_level_idx_ptr,
    bool HAS_SEA_LEVEL, long num_sea_level_boundaries, long ensemble_size,
    bool batched_river_manning, bool batched_river_width,
    bool batched_river_length, bool batched_river_height,
    bool batched_flood_manning, bool batched_catchment_elevation,
    bool batched_downstream_distance, bool batched_sea_surface_elevation,
    long BLOCK_SIZE)
{
    if (river_outflow_ptr.scalar_type() == at::kDouble)
        launch_outflow_t<double, double>(
            downstream_idx_ptr, river_inflow_ptr, river_outflow_ptr,
            river_manning_ptr, river_depth_ptr, river_width_ptr,
            river_length_ptr, river_height_ptr, river_storage_ptr,
            flood_inflow_ptr, flood_outflow_ptr, flood_manning_ptr,
            flood_depth_ptr,
            catchment_elevation_ptr,
            downstream_distance_ptr, flood_storage_ptr, protected_storage_ptr,
            river_cross_section_depth_ptr, flood_cross_section_depth_ptr,
            flood_cross_section_area_ptr, global_bifurcation_outflow_ptr,
            outgoing_storage_ptr, (double)gravity,
            time_step_ptr, num_catchments, (int)HAS_BIFURCATION,
            (int)HAS_LEVEE,
            is_dam_upstream_ptr, (int)HAS_RESERVOIR,
            (double)min_kinematic_slope, sea_surface_elevation_ptr,
            catchment_sea_level_idx_ptr, (int)HAS_SEA_LEVEL,
            num_sea_level_boundaries, ensemble_size,
            batched_river_manning, batched_river_width,
            batched_river_length, batched_river_height,
            batched_flood_manning, batched_catchment_elevation,
            batched_downstream_distance, batched_sea_surface_elevation,
            (int)BLOCK_SIZE);
    else if (river_storage_ptr.scalar_type() == at::kDouble)
        launch_outflow_t<float, double>(
            downstream_idx_ptr, river_inflow_ptr, river_outflow_ptr,
            river_manning_ptr, river_depth_ptr, river_width_ptr,
            river_length_ptr, river_height_ptr, river_storage_ptr,
            flood_inflow_ptr, flood_outflow_ptr, flood_manning_ptr,
            flood_depth_ptr,
            catchment_elevation_ptr,
            downstream_distance_ptr, flood_storage_ptr, protected_storage_ptr,
            river_cross_section_depth_ptr, flood_cross_section_depth_ptr,
            flood_cross_section_area_ptr, global_bifurcation_outflow_ptr,
            outgoing_storage_ptr, gravity,
            time_step_ptr, num_catchments, (int)HAS_BIFURCATION,
            (int)HAS_LEVEE,
            is_dam_upstream_ptr, (int)HAS_RESERVOIR,
            min_kinematic_slope, sea_surface_elevation_ptr,
            catchment_sea_level_idx_ptr, (int)HAS_SEA_LEVEL,
            num_sea_level_boundaries, ensemble_size,
            batched_river_manning, batched_river_width,
            batched_river_length, batched_river_height,
            batched_flood_manning, batched_catchment_elevation,
            batched_downstream_distance, batched_sea_surface_elevation,
            (int)BLOCK_SIZE);
    else
        launch_outflow_t<float, float>(
            downstream_idx_ptr, river_inflow_ptr, river_outflow_ptr,
            river_manning_ptr, river_depth_ptr, river_width_ptr,
            river_length_ptr, river_height_ptr, river_storage_ptr,
            flood_inflow_ptr, flood_outflow_ptr, flood_manning_ptr,
            flood_depth_ptr,
            catchment_elevation_ptr,
            downstream_distance_ptr, flood_storage_ptr, protected_storage_ptr,
            river_cross_section_depth_ptr, flood_cross_section_depth_ptr,
            flood_cross_section_area_ptr, global_bifurcation_outflow_ptr,
            outgoing_storage_ptr, gravity,
            time_step_ptr, num_catchments, (int)HAS_BIFURCATION,
            (int)HAS_LEVEE,
            is_dam_upstream_ptr, (int)HAS_RESERVOIR,
            min_kinematic_slope, sea_surface_elevation_ptr,
            catchment_sea_level_idx_ptr, (int)HAS_SEA_LEVEL,
            num_sea_level_boundaries, ensemble_size,
            batched_river_manning, batched_river_width,
            batched_river_length, batched_river_height,
            batched_flood_manning, batched_catchment_elevation,
            batched_downstream_distance, batched_sea_surface_elevation,
            (int)BLOCK_SIZE);
}
