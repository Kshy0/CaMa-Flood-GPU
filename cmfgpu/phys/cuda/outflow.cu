// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// CUDA backend for the outflow / inflow kernels.
//
// Straight-line memory-bound kernels (downstream gather + atomic scatter), using
// lane-local skips for branch-heavy flood momentum.
// Templated on STO = hpfloat storage dtype (REAL/double): storage is always
// downcast to REAL for the physics, but pointer/atomic/store types follow STO.
// Compiled WITHOUT --use_fast_math for predictable IEEE div/sqrt behavior.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// ----------------------------------------------------------------------------
// (1) compute_outflow_kernel
// ----------------------------------------------------------------------------
template <bool BASE_ONLY, typename REAL, typename STO>
__global__ void k_outflow(
    const int* __restrict__ downstream_idx,
    STO* __restrict__ river_inflow, REAL* __restrict__ river_outflow,
    const REAL* __restrict__ river_manning, const REAL* __restrict__ river_depth,
    const REAL* __restrict__ river_width, const REAL* __restrict__ river_length,
    const REAL* __restrict__ river_height, const STO* __restrict__ river_storage,
    STO* __restrict__ flood_inflow, REAL* __restrict__ flood_outflow,
    const REAL* __restrict__ flood_manning, const REAL* __restrict__ flood_depth,
    const REAL* __restrict__ protected_depth, const REAL* __restrict__ catchment_elevation,
    const REAL* __restrict__ downstream_distance, const STO* __restrict__ flood_storage,
    const STO* __restrict__ protected_storage,
    REAL* __restrict__ river_cross_section_depth, REAL* __restrict__ flood_cross_section_depth,
    REAL* __restrict__ flood_cross_section_area,
    STO* __restrict__ global_bifurcation_outflow, STO* __restrict__ total_storage,
    STO* __restrict__ outgoing_storage, REAL* __restrict__ water_surface_elevation,
    REAL* __restrict__ protected_water_surface_elevation,
    REAL gravity, const REAL* __restrict__ time_step_ptr,
    long num_catchments, int has_bifurcation, int has_total_storage,
    int has_water_surface, int has_protected_water_surface,
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
    REAL p_dep = __ldg(protected_depth + t);
    REAL c_elv = __ldg(catchment_elevation + t);
    REAL dn_dist = __ldg(downstream_distance + t);

    REAL r_cs_dep = __ldg(river_cross_section_depth + t);
    REAL f_cs_dep = __ldg(flood_cross_section_depth + t);
    REAL f_cs_area = __ldg(flood_cross_section_area + t);

    REAL rs = (REAL)river_storage[t];
    REAL fs = (REAL)flood_storage[t];
    REAL ps = (REAL)protected_storage[t];

    REAL river_elevation = c_elv - r_hgt;
    REAL wse = r_dep + river_elevation;
    REAL pwse = fmin(c_elv + p_dep, wse);
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
    REAL flood_slope = fmin(fmax(river_slope, (REAL)-0.005), (REAL)0.005);

    REAL upd_r_cs_dep = max_wse - river_elevation;
    REAL r_sifd = fmax(sqrt(upd_r_cs_dep * r_cs_dep), (REAL)1e-6);
    REAL upd_f_cs_dep = fmax(max_wse - c_elv, (REAL)0.0);
    REAL f_sifd = fmax(sqrt(upd_f_cs_dep * f_cs_dep), (REAL)1e-6);

    REAL upd_f_cs_area = fmax(fs / r_len - f_dep * r_wid, (REAL)0.0);

    REAL r_cs_area = upd_r_cs_dep * r_wid;
    bool river_condition = (r_sifd > (REAL)1e-5) && (r_cs_area > (REAL)1e-5);
    REAL unit_r_out = r_out / r_wid;
    REAL num_r = r_wid * (unit_r_out + gravity * time_step * r_sifd * river_slope);
    REAL den_r = (REAL)1.0 + gravity * time_step * (r_man * r_man) * fabs(unit_r_out)
                  * ((REAL)1.0 / (r_sifd * r_sifd * cbrt(r_sifd)));
    REAL upd_r_out = num_r / den_r;
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
        is_negative_flow ? (REAL)0.05 * total_storage_f / total_negative_flow : (REAL)1.0, (REAL)1.0);
    if (is_negative_flow) { upd_r_out *= limit_rate; upd_f_out *= limit_rate; }

    if constexpr (!BASE_ONLY) {
        if (has_reservoir) {
            bool is_dam_up = is_dam_upstream && is_dam_upstream[t];
            REAL bed_slope = (c_elv - c_elv_dn) / dn_dist;
            bed_slope = fmax(bed_slope, min_kinematic_slope);
            REAL kin_riv_vel = ((REAL)1.0 / r_man) * sqrt(bed_slope) * cbrt(r_dep * r_dep);
            REAL kin_riv = r_wid * r_dep * kin_riv_vel;
            kin_riv = fmin(kin_riv, rs / time_step);
            kin_riv = fmax(kin_riv, (REAL)0.0);
            REAL bed_slope_f = fmin(bed_slope, (REAL)0.005);
            REAL kin_fld_vel = ((REAL)1.0 / f_man) * sqrt(bed_slope_f) * cbrt(f_dep * f_dep);
            REAL kin_fld_area = fmax(fs / r_len - f_dep * r_wid, (REAL)0.0);
            REAL kin_fld = kin_fld_area * kin_fld_vel;
            kin_fld = fmin(kin_fld, fs / time_step);
            kin_fld = fmax(kin_fld, (REAL)0.0);
            if (is_dam_up) { upd_r_out = kin_riv; upd_f_out = kin_fld; }
        }
    }

    river_outflow[t] = upd_r_out;
    flood_outflow[t] = upd_f_out;
    if (has_water_surface) water_surface_elevation[t] = wse;
    if (has_protected_water_surface)
        protected_water_surface_elevation[t] = pwse;
    river_cross_section_depth[t] = upd_r_cs_dep;
    flood_cross_section_depth[t] = upd_f_cs_dep;
    flood_cross_section_area[t] = upd_f_cs_area;
    if (has_total_storage) total_storage[t] = (STO)total_storage_f;

    river_inflow[t] = (STO)0;
    flood_inflow[t] = (STO)0;
    if (has_bifurcation) global_bifurcation_outflow[t] = (STO)0;

    REAL pos = fmax(upd_r_out, (REAL)0.0) + fmax(upd_f_out, (REAL)0.0);
    REAL neg = fmin(upd_r_out, (REAL)0.0) + fmin(upd_f_out, (REAL)0.0);
    atomicAdd(outgoing_storage + t, (STO)(pos * time_step));
    REAL to_add = is_river_mouth ? (REAL)0.0 : -neg * time_step;
    atomicAdd(outgoing_storage + dn, (STO)to_add);
}

// ----------------------------------------------------------------------------
// (2) compute_inflow_kernel
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
// Host launchers (dispatch on storage dtype)
// ----------------------------------------------------------------------------
static inline bool*  PB(at::Tensor& x) { return x.numel() ? x.data_ptr<bool>()  : nullptr; }
static inline int*   PI(at::Tensor& x) { return x.numel() ? x.data_ptr<int>()   : nullptr; }
template <typename REAL> static inline REAL* PR(at::Tensor& x) { return x.numel() ? x.data_ptr<REAL>() : nullptr; }
template <typename STO> static inline STO* PS(at::Tensor& x) { return x.numel() ? x.data_ptr<STO>() : nullptr; }
template <typename REAL> static inline REAL* PRO(c10::optional<at::Tensor>& x) { return x ? x->data_ptr<REAL>() : nullptr; }
static inline bool*  PBO(c10::optional<at::Tensor>& x) { return x ? x->data_ptr<bool>() : nullptr; }
static inline int*   PIO(c10::optional<at::Tensor>& x) { return x ? x->data_ptr<int>() : nullptr; }
template <typename STO> static inline STO* PSO(c10::optional<at::Tensor>& x) {
    return x ? x->data_ptr<STO>() : nullptr;
}

template <typename REAL, typename STO>
static void launch_outflow_t(
    at::Tensor& di, at::Tensor& ri, at::Tensor& ro, at::Tensor& rman, at::Tensor& rd,
    at::Tensor& rw, at::Tensor& rl, at::Tensor& rh, at::Tensor& rs,
    at::Tensor& fi, at::Tensor& fo, at::Tensor& fman, at::Tensor& fd, at::Tensor& pd,
    at::Tensor& ce, at::Tensor& dd, at::Tensor& fsto, at::Tensor& psto,
    at::Tensor& rcsd, at::Tensor& fcsd, at::Tensor& fcsa,
    c10::optional<at::Tensor>& gb, c10::optional<at::Tensor>& ts_out,
    at::Tensor& outs, c10::optional<at::Tensor>& wse,
    c10::optional<at::Tensor>& pwse,
    REAL gravity, at::Tensor& tsp, long n, int has_bif, int has_total,
    int has_water_surface, int has_protected_surface,
    c10::optional<at::Tensor>& dam, int has_res, REAL minslope,
    c10::optional<at::Tensor>& sea, c10::optional<at::Tensor>& sea_idx,
    int has_sea, int block)
{
    int grid = (int)((n + block - 1) / block);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
#define LAUNCH_OUTFLOW(BASE_ONLY) \
    k_outflow<BASE_ONLY, REAL, STO><<<grid, block, 0, stream>>>( \
        PI(di), PS<STO>(ri), PR<REAL>(ro), PR<REAL>(rman), PR<REAL>(rd), PR<REAL>(rw), PR<REAL>(rl), PR<REAL>(rh), PS<STO>(rs), \
        PS<STO>(fi), PR<REAL>(fo), PR<REAL>(fman), PR<REAL>(fd), PR<REAL>(pd), PR<REAL>(ce), PR<REAL>(dd), PS<STO>(fsto), PS<STO>(psto), \
        PR<REAL>(rcsd), PR<REAL>(fcsd), PR<REAL>(fcsa), \
        PSO<STO>(gb), PSO<STO>(ts_out), PS<STO>(outs), PRO<REAL>(wse), PRO<REAL>(pwse), \
        gravity, PR<REAL>(tsp), n, has_bif, has_total, has_water_surface, \
        has_protected_surface, \
        PBO(dam), has_res, minslope, \
        PRO<REAL>(sea), PIO(sea_idx), has_sea)
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
    at::Tensor flood_depth_ptr, at::Tensor protected_depth_ptr,
    at::Tensor catchment_elevation_ptr, at::Tensor downstream_distance_ptr,
    at::Tensor flood_storage_ptr, at::Tensor protected_storage_ptr,
    at::Tensor river_cross_section_depth_ptr,
    at::Tensor flood_cross_section_depth_ptr,
    at::Tensor flood_cross_section_area_ptr,
    c10::optional<at::Tensor> global_bifurcation_outflow_ptr,
    c10::optional<at::Tensor> total_storage_ptr,
    at::Tensor outgoing_storage_ptr,
    c10::optional<at::Tensor> water_surface_elevation_ptr,
    c10::optional<at::Tensor> protected_water_surface_elevation_ptr,
    float gravity, at::Tensor time_step_ptr, long num_catchments,
    bool HAS_BIFURCATION, bool HAS_TOTAL_STORAGE,
    bool HAS_WATER_SURFACE, bool HAS_PROTECTED_WATER_SURFACE,
    c10::optional<at::Tensor> is_dam_upstream_ptr, bool HAS_RESERVOIR,
    float MIN_KINEMATIC_SLOPE,
    c10::optional<at::Tensor> sea_surface_elevation_ptr,
    c10::optional<at::Tensor> catchment_sea_level_idx_ptr,
    bool HAS_SEA_LEVEL, long num_sea_level_boundaries, long BLOCK_SIZE)
{
    (void)num_sea_level_boundaries;
    if (river_outflow_ptr.scalar_type() == at::kDouble)
        launch_outflow_t<double, double>(
            downstream_idx_ptr, river_inflow_ptr, river_outflow_ptr,
            river_manning_ptr, river_depth_ptr, river_width_ptr,
            river_length_ptr, river_height_ptr, river_storage_ptr,
            flood_inflow_ptr, flood_outflow_ptr, flood_manning_ptr,
            flood_depth_ptr, protected_depth_ptr, catchment_elevation_ptr,
            downstream_distance_ptr, flood_storage_ptr, protected_storage_ptr,
            river_cross_section_depth_ptr, flood_cross_section_depth_ptr,
            flood_cross_section_area_ptr, global_bifurcation_outflow_ptr,
            total_storage_ptr, outgoing_storage_ptr,
            water_surface_elevation_ptr,
            protected_water_surface_elevation_ptr, (double)gravity,
            time_step_ptr, num_catchments, (int)HAS_BIFURCATION,
            (int)HAS_TOTAL_STORAGE,
            (int)HAS_WATER_SURFACE,
            (int)HAS_PROTECTED_WATER_SURFACE,
            is_dam_upstream_ptr, (int)HAS_RESERVOIR,
            (double)MIN_KINEMATIC_SLOPE, sea_surface_elevation_ptr,
            catchment_sea_level_idx_ptr, (int)HAS_SEA_LEVEL,
            (int)BLOCK_SIZE);
    else if (river_storage_ptr.scalar_type() == at::kDouble)
        launch_outflow_t<float, double>(
            downstream_idx_ptr, river_inflow_ptr, river_outflow_ptr,
            river_manning_ptr, river_depth_ptr, river_width_ptr,
            river_length_ptr, river_height_ptr, river_storage_ptr,
            flood_inflow_ptr, flood_outflow_ptr, flood_manning_ptr,
            flood_depth_ptr, protected_depth_ptr, catchment_elevation_ptr,
            downstream_distance_ptr, flood_storage_ptr, protected_storage_ptr,
            river_cross_section_depth_ptr, flood_cross_section_depth_ptr,
            flood_cross_section_area_ptr, global_bifurcation_outflow_ptr,
            total_storage_ptr, outgoing_storage_ptr,
            water_surface_elevation_ptr,
            protected_water_surface_elevation_ptr, gravity,
            time_step_ptr, num_catchments, (int)HAS_BIFURCATION,
            (int)HAS_TOTAL_STORAGE,
            (int)HAS_WATER_SURFACE,
            (int)HAS_PROTECTED_WATER_SURFACE,
            is_dam_upstream_ptr, (int)HAS_RESERVOIR,
            MIN_KINEMATIC_SLOPE, sea_surface_elevation_ptr,
            catchment_sea_level_idx_ptr, (int)HAS_SEA_LEVEL,
            (int)BLOCK_SIZE);
    else
        launch_outflow_t<float, float>(
            downstream_idx_ptr, river_inflow_ptr, river_outflow_ptr,
            river_manning_ptr, river_depth_ptr, river_width_ptr,
            river_length_ptr, river_height_ptr, river_storage_ptr,
            flood_inflow_ptr, flood_outflow_ptr, flood_manning_ptr,
            flood_depth_ptr, protected_depth_ptr, catchment_elevation_ptr,
            downstream_distance_ptr, flood_storage_ptr, protected_storage_ptr,
            river_cross_section_depth_ptr, flood_cross_section_depth_ptr,
            flood_cross_section_area_ptr, global_bifurcation_outflow_ptr,
            total_storage_ptr, outgoing_storage_ptr,
            water_surface_elevation_ptr,
            protected_water_surface_elevation_ptr, gravity,
            time_step_ptr, num_catchments, (int)HAS_BIFURCATION,
            (int)HAS_TOTAL_STORAGE,
            (int)HAS_WATER_SURFACE,
            (int)HAS_PROTECTED_WATER_SURFACE,
            is_dam_upstream_ptr, (int)HAS_RESERVOIR,
            MIN_KINEMATIC_SLOPE, sea_surface_elevation_ptr,
            catchment_sea_level_idx_ptr, (int)HAS_SEA_LEVEL,
            (int)BLOCK_SIZE);
}

template <typename REAL, typename STO>
static void launch_inflow_t(
    at::Tensor& di, at::Tensor& ro, at::Tensor& fo, at::Tensor& rs, at::Tensor& fs,
    at::Tensor& outs, at::Tensor& ri, at::Tensor& fi,
    c10::optional<at::Tensor>& lr,
    c10::optional<at::Tensor>& rti, c10::optional<at::Tensor>& isres,
    long n, int has_bif, int has_res, int block)
{
    int grid = (int)((n + block - 1) / block);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
#define LAUNCH_INFLOW(HAS_BIFURCATION) \
    k_inflow<HAS_BIFURCATION, REAL, STO><<<grid, block, 0, stream>>>( \
        PI(di), PR<REAL>(ro), PR<REAL>(fo), PS<STO>(rs), PS<STO>(fs), \
        PS<STO>(outs), PS<STO>(ri), PS<STO>(fi), PRO<REAL>(lr), \
        PSO<STO>(rti), PBO(isres), n, has_res)
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
    long BLOCK_SIZE)
{
    if (river_outflow_ptr.scalar_type() == at::kDouble)
        launch_inflow_t<double, double>(
            downstream_idx_ptr, river_outflow_ptr, flood_outflow_ptr,
            river_storage_ptr, flood_storage_ptr, outgoing_storage_ptr,
            river_inflow_ptr, flood_inflow_ptr, limit_rate_ptr,
            reservoir_total_inflow_ptr, is_reservoir_ptr, num_catchments,
            (int)HAS_BIFURCATION, (int)HAS_RESERVOIR, (int)BLOCK_SIZE);
    else if (river_storage_ptr.scalar_type() == at::kDouble)
        launch_inflow_t<float, double>(
            downstream_idx_ptr, river_outflow_ptr, flood_outflow_ptr,
            river_storage_ptr, flood_storage_ptr, outgoing_storage_ptr,
            river_inflow_ptr, flood_inflow_ptr, limit_rate_ptr,
            reservoir_total_inflow_ptr, is_reservoir_ptr, num_catchments,
            (int)HAS_BIFURCATION, (int)HAS_RESERVOIR, (int)BLOCK_SIZE);
    else
        launch_inflow_t<float, float>(
            downstream_idx_ptr, river_outflow_ptr, flood_outflow_ptr,
            river_storage_ptr, flood_storage_ptr, outgoing_storage_ptr,
            river_inflow_ptr, flood_inflow_ptr, limit_rate_ptr,
            reservoir_total_inflow_ptr, is_reservoir_ptr, num_catchments,
            (int)HAS_BIFURCATION, (int)HAS_RESERVOIR, (int)BLOCK_SIZE);
}
