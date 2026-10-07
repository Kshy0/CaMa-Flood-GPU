#include "canonical.cuh"

template <typename REAL, typename STO>
__device__ __forceinline__ REAL cmf_supply_rate(STO outgoing_flow, STO storage, STO step)
{
    REAL volume = fmax((REAL)(outgoing_flow * step), (REAL)CMF_OUTGOING_VOLUME_FLOOR);
    return fmin((REAL)storage / volume, (REAL)1.0);
}

template <bool HAS_BIFURCATION, typename REAL, typename STO>
__device__ __forceinline__ void k_inflow_cell(
    const int* __restrict__ downstream_idx,
    REAL* __restrict__ river_outflow, REAL* __restrict__ flood_outflow,
    const STO* __restrict__ river_storage, const STO* __restrict__ flood_storage,
    const STO* __restrict__ outgoing_storage, const REAL* __restrict__ time_step_ptr,
    STO* __restrict__ river_inflow, STO* __restrict__ flood_inflow,
    REAL* __restrict__ limit_rate_out, STO* __restrict__ reservoir_total_inflow,
    const bool* __restrict__ is_reservoir, int has_reservoir, long t)
{
    // CaMa-Flood v4.23 supply-side limiter (CALC_INFLOW_LSPAMAT): a cell
    // releases at most its storage.  The rate of the cell a flow leaves sets
    // both of its flows, chosen by the river flow's direction.
    REAL r_out = river_outflow[t];
    REAL f_out = flood_outflow[t];
    STO step = (STO)__ldg(time_step_ptr);
    REAL limit_rate = cmf_supply_rate<REAL>(
        outgoing_storage[t], river_storage[t] + flood_storage[t], step);

    int dn = downstream_idx[t];
    REAL limit_rate_dn = cmf_supply_rate<REAL>(
        outgoing_storage[dn], river_storage[dn] + flood_storage[dn], step);

    REAL rate = (r_out > (REAL)0.0) ? limit_rate : limit_rate_dn;
    REAL upd_r_out = r_out * rate;
    REAL upd_f_out = f_out * rate;

    river_outflow[t] = upd_r_out;
    flood_outflow[t] = upd_f_out;
    if constexpr (HAS_BIFURCATION) limit_rate_out[t] = limit_rate;

    bool is_river_mouth = (dn == (int)t);
    if (!is_river_mouth) {
        cmf_atomic_add(river_inflow + dn, (STO)upd_r_out);
        cmf_atomic_add(flood_inflow + dn, (STO)upd_f_out);
        if (has_reservoir) {
            bool is_downstream_res = is_reservoir && (is_reservoir[dn] != 0);
            if (is_downstream_res)
                cmf_atomic_add(reservoir_total_inflow + dn, (STO)upd_r_out + (STO)upd_f_out);
        }
    }
}

// Generated-entry body over the canonical values ``a`` of compute_inflow.
template <typename REAL, typename STO, bool HAS_BIFURCATION, bool HAS_RESERVOIR, class A>
__device__ __forceinline__ void inflow(const A& a, long t)
{
    k_inflow_cell<HAS_BIFURCATION, REAL, STO>(
        a.downstream_idx_ptr, a.river_outflow_ptr, a.flood_outflow_ptr, a.river_storage_ptr,
        a.flood_storage_ptr, a.outgoing_storage_ptr, a.time_step_ptr, a.river_inflow_ptr,
        a.flood_inflow_ptr,
        cmf_optional<HAS_BIFURCATION>(a.limit_rate_ptr),
        cmf_optional<HAS_RESERVOIR>(a.reservoir_total_inflow_ptr),
        cmf_optional<HAS_RESERVOIR>(a.is_reservoir_ptr), HAS_RESERVOIR, t);
}

// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//

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
    int has_bifurcation, int has_levee,
    const bool* __restrict__ is_dam_upstream, int has_reservoir, REAL min_kinematic_slope,
    const REAL* __restrict__ sea_surface_elevation,
    const int* __restrict__ catchment_sea_level_idx, int has_sea_level, long t)
{
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
    STO storage_sum = river_storage[t] + flood_storage[t];
    if (has_levee) storage_sum += protected_storage[t];
    REAL total_storage = (REAL)storage_sum;

    REAL river_elevation = c_elv - r_hgt;
    REAL wse = r_dep + river_elevation;

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

    // Floodplain flow only moves with the river flow (rivout*fldout > 0).
    if (!(upd_r_out * upd_f_out > (REAL)0.0)) upd_f_out = (REAL)0.0;
    // v4.23 storage-change limiter on every non-mouth cell: flow towards the
    // upstream cell removes at most 5% of the storage per step.
    if (!is_river_mouth) {
        REAL backflow = fmax((-upd_r_out - upd_f_out) * time_step,
                             (REAL)(float)CMF_OUTGOING_VOLUME_FLOOR);
        REAL limit_rate = fmin(
            (REAL)CMF_BACKFLOW_STORAGE_FRACTION * total_storage / backflow, (REAL)1.0);
        upd_r_out *= limit_rate;
        upd_f_out *= limit_rate;
    }

    if constexpr (!BASE_ONLY) {
        if (has_reservoir &&
            (sizeof(REAL) == sizeof(float) || (is_dam_upstream && is_dam_upstream[t]))) {
            REAL bed_slope = (c_elv - c_elv_dn) / dn_dist;
            bed_slope = fmax(bed_slope, min_kinematic_slope);
            REAL kin_riv_vel = ((REAL)1.0 / r_man) * sqrt(bed_slope) * cbrt(r_dep * r_dep);
            // CaMa bounds the kinematic flows by the storage only (MIN), so a
            // negative storage (negative runoff) gives a negative flow.
            REAL kin_riv = fmin(r_wid * r_dep * kin_riv_vel, rs / time_step);
            REAL bed_slope_f = fmin(bed_slope, (REAL)CMF_ROUTING_SLOPE_LIMIT);
            REAL kin_fld_vel = ((REAL)1.0 / f_man) * sqrt(bed_slope_f) * cbrt(f_dep * f_dep);
            REAL kin_fld_area = fmax(fs / r_len - f_dep * r_wid, (REAL)0.0);
            REAL kin_fld = fmin(kin_fld_area * kin_fld_vel, fs / time_step);
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
    // Next step's DARE_pr uses D2FLDDPH_PRE = max(D2RIVDPH_PRE - D2RIVHGT, 0).
    flood_cross_section_area[t] = fmax(fs / r_len - fmax(r_dep - r_hgt, (REAL)0.0) * r_wid, (REAL)0.0);

    river_inflow[t] = (STO)0;
    flood_inflow[t] = (STO)0;
    if (has_bifurcation) global_bifurcation_outflow[t] = (STO)0;

    // P2STOOUT includes positive flows from this cell and reversed flows
    // into its downstream cell.
    cmf_atomic_add(outgoing_storage + t,
                   (STO)(fmax(upd_r_out, (REAL)0.0) + fmax(upd_f_out, (REAL)0.0)));
    if (!is_river_mouth)
        cmf_atomic_add(outgoing_storage + dn,
                       (STO)fmax(-upd_r_out, (REAL)0.0) + (STO)fmax(-upd_f_out, (REAL)0.0));
}

// Generated-entry body over the canonical values ``a`` of compute_outflow.
template <typename REAL, typename STO, bool HAS_BIFURCATION, bool HAS_LEVEE,
          bool HAS_RESERVOIR, bool HAS_SEA_LEVEL, class A>
__device__ __forceinline__ void outflow(const A& a, long t)
{
    constexpr bool BASE_ONLY = !HAS_RESERVOIR && !HAS_SEA_LEVEL;
    k_outflow_cell<BASE_ONLY, REAL, STO>(
        a.downstream_idx_ptr, a.river_inflow_ptr, a.river_outflow_ptr, a.river_manning_ptr,
        a.river_depth_ptr, a.river_width_ptr, a.river_length_ptr, a.river_height_ptr,
        a.river_storage_ptr, a.flood_inflow_ptr, a.flood_outflow_ptr, a.flood_manning_ptr,
        a.flood_depth_ptr, a.catchment_elevation_ptr, a.downstream_distance_ptr,
        a.flood_storage_ptr, cmf_optional<HAS_LEVEE>(a.protected_storage_ptr),
        a.river_cross_section_depth_ptr, a.flood_cross_section_depth_ptr,
        a.flood_cross_section_area_ptr,
        cmf_optional<HAS_BIFURCATION>(a.global_bifurcation_outflow_ptr),
        a.outgoing_storage_ptr, a.gravity, a.time_step_ptr, HAS_BIFURCATION, HAS_LEVEE,
        cmf_optional<HAS_RESERVOIR>(a.is_dam_upstream_ptr), HAS_RESERVOIR,
        a.min_kinematic_slope, cmf_optional<HAS_SEA_LEVEL>(a.sea_surface_elevation_ptr),
        cmf_optional<HAS_SEA_LEVEL>(a.catchment_sea_level_idx_ptr), HAS_SEA_LEVEL, t);
}

// Ensemble variants of inflow and outflow: offset member ``member``'s slices of
// the per-member buffers in their copy of ``a``, then run the single-member body.

template <typename REAL, typename STO, bool HAS_BIFURCATION, bool HAS_RESERVOIR, class A>
__device__ __forceinline__ void inflow_members(A a, long t, long member)
{
    const long member_offset = member * a.num_catchments;
    a.river_outflow_ptr += member_offset;
    a.flood_outflow_ptr += member_offset;
    a.river_storage_ptr += member_offset;
    a.flood_storage_ptr += member_offset;
    a.outgoing_storage_ptr += member_offset;
    a.river_inflow_ptr += member_offset;
    a.flood_inflow_ptr += member_offset;
    if constexpr (HAS_BIFURCATION) a.limit_rate_ptr += member_offset;
    if constexpr (HAS_RESERVOIR) a.reservoir_total_inflow_ptr += member_offset;
    inflow<REAL, STO, HAS_BIFURCATION, HAS_RESERVOIR>(a, t);
}

template <typename REAL, typename STO, bool HAS_BIFURCATION, bool HAS_LEVEE,
          bool HAS_RESERVOIR, bool HAS_SEA_LEVEL, class A>
__device__ __forceinline__ void outflow_members(A a, long t, long member)
{
    const long member_offset = member * a.num_catchments;
    a.river_inflow_ptr += member_offset;
    a.river_outflow_ptr += member_offset;
    a.river_depth_ptr += member_offset;
    a.river_storage_ptr += member_offset;
    a.flood_inflow_ptr += member_offset;
    a.flood_outflow_ptr += member_offset;
    a.flood_depth_ptr += member_offset;
    a.flood_storage_ptr += member_offset;
    a.river_cross_section_depth_ptr += member_offset;
    a.flood_cross_section_depth_ptr += member_offset;
    a.flood_cross_section_area_ptr += member_offset;
    a.outgoing_storage_ptr += member_offset;
    if constexpr (HAS_LEVEE) a.protected_storage_ptr += member_offset;
    if constexpr (HAS_BIFURCATION) a.global_bifurcation_outflow_ptr += member_offset;
    if (a.batched_river_manning) a.river_manning_ptr += member_offset;
    if (a.batched_river_width) a.river_width_ptr += member_offset;
    if (a.batched_river_length) a.river_length_ptr += member_offset;
    if (a.batched_river_height) a.river_height_ptr += member_offset;
    if (a.batched_flood_manning) a.flood_manning_ptr += member_offset;
    if (a.batched_catchment_elevation) a.catchment_elevation_ptr += member_offset;
    if (a.batched_downstream_distance) a.downstream_distance_ptr += member_offset;
    if constexpr (HAS_SEA_LEVEL) {
        if (a.batched_sea_surface_elevation)
            a.sea_surface_elevation_ptr += member * a.num_sea_level_boundaries;
    }
    outflow<REAL, STO, HAS_BIFURCATION, HAS_LEVEE, HAS_RESERVOIR, HAS_SEA_LEVEL>(a, t);
}
