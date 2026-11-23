// LICENSE HEADER MANAGED BY add-license-header
// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// CUDA kernels for high-resolution flood-depth and inundation-duration downscaling.

#include "canonical.cuh"

template <typename REAL, bool HAS_LEVEE, bool HAS_TRIBUTARY, class A>
__device__ __forceinline__ void downscale_flood_depth(const A& a, long p)
{
    int ci = __ldg(a.pixel_catchment_idx_ptr + p);
    REAL flddif = __ldg(a.pixel_flood_diff_ptr + p);
    REAL depth = (REAL)0.0;

    if constexpr (HAS_LEVEE) {
        if (a.pixel_is_protected_ptr[p]) {
            REAL levdph = __ldg(a.protected_depth_ptr + ci);
            if (levdph > flddif) {
                depth = (levdph - flddif) + a.protected_depth_offset;
            }
        } else {
            REAL flddph = __ldg(a.flood_depth_ptr + ci);
            if (flddph > flddif) {
                depth = flddph - flddif;
            }
        }
    } else {
        REAL flddph = __ldg(a.flood_depth_ptr + ci);
        if (flddph > flddif) {
            depth = flddph - flddif;
        }
    }

    if constexpr (!HAS_TRIBUTARY) {
        if (a.permanent_water_depth > (REAL)0.0 && a.pixel_is_permanent_water_ptr[p]) {
            depth = fmax(a.permanent_water_depth, depth);
        }
    }

    a.pixel_flood_depth_ptr[p] = depth;
}

template <typename REAL, bool HAS_LEVEE, bool HAS_TRIBUTARY, class A>
__device__ __forceinline__ void downscale_flood_depth_members(A a, long p, long member)
{
    const long cat_off = member * a.num_catchments;
    const long pix_off = member * a.num_pixels;
    if (a.batched_flood_depth) a.flood_depth_ptr += cat_off;
    if constexpr (HAS_LEVEE) {
        if (a.batched_protected_depth) a.protected_depth_ptr += cat_off;
        if (a.batched_pixel_is_protected) a.pixel_is_protected_ptr += pix_off;
    }
    if (a.batched_pixel_flood_diff) a.pixel_flood_diff_ptr += pix_off;
    if (a.batched_pixel_is_permanent_water) a.pixel_is_permanent_water_ptr += pix_off;
    a.pixel_flood_depth_ptr += pix_off;
    downscale_flood_depth<REAL, HAS_LEVEE, HAS_TRIBUTARY>(a, p);
}

template <typename REAL, class A>
__device__ __forceinline__ void downscale_tributary_depth(const A& a, long t)
{
    int p = __ldg(a.tributary_pixel_idx_ptr + t);
    REAL old_dph = a.pixel_flood_depth_ptr[p];
    REAL best_dph = old_dph;

    int start = __ldg(a.tributary_edge_start_ptr + t);
    int end = (t + 1 < a.num_tributary_pixels)
        ? __ldg(a.tributary_edge_start_ptr + t + 1)
        : (int)a.num_tributary_links;

    for (int k = start; k < end; ++k) {
        int e = __ldg(a.tributary_edge_order_ptr + k);
        int up_p = __ldg(a.tributary_upstream_pixel_idx_ptr + e);
        int dn_p = __ldg(a.tributary_downstream_pixel_idx_ptr + e);
        REAL dph1 = __ldg(a.pixel_flood_depth_ptr + up_p);
        REAL dph2 = __ldg(a.pixel_flood_depth_ptr + dn_p);
        REAL step = __ldg(a.tributary_chain_step_ptr + e);
        REAL len = __ldg(a.tributary_chain_length_ptr + e);
        REAL ddph = fmax((dph1 - dph2) / len, (REAL)0.0);
        REAL cand = dph1 - ddph * step;
        if (cand > best_dph) {
            best_dph = cand;
        }
    }

    if (best_dph > old_dph) {
        a.pixel_flood_depth_ptr[p] = best_dph;
        int ci = __ldg(a.pixel_catchment_idx_ptr + p);
        atomicMax(a.catchment_tributary_modified_ptr + ci, 1);
    }
}

template <typename REAL, class A>
__device__ __forceinline__ void downscale_tributary_depth_members(A a, long t, long member)
{
    const long cat_off = member * a.num_catchments;
    const long pix_off = member * a.num_pixels;
    const long link_off = member * a.num_tributary_links;
    if (a.batched_tributary_chain_step) a.tributary_chain_step_ptr += link_off;
    if (a.batched_tributary_chain_length) a.tributary_chain_length_ptr += link_off;
    a.pixel_flood_depth_ptr += pix_off;
    a.catchment_tributary_modified_ptr += cat_off;
    downscale_tributary_depth<REAL>(a, t);
}

template <typename REAL, class A>
__device__ __forceinline__ void downscale_backwater_depth(const A& a, long b)
{
    int p = __ldg(a.backwater_pixel_idx_ptr + b);
    int ci = __ldg(a.pixel_catchment_idx_ptr + p);
    if (__ldg(a.catchment_tributary_modified_ptr + ci) <= 0) return;

    int q = __ldg(a.backwater_target_pixel_idx_ptr + b);
    REAL dph1 = fmax(
        __ldg(a.backwater_floor_elevation_ptr + b),
        __ldg(a.pixel_flood_diff_ptr + q) + __ldg(a.pixel_flood_depth_ptr + q));
    REAL flddif_p = __ldg(a.pixel_flood_diff_ptr + p);
    REAL cur_dph = a.pixel_flood_depth_ptr[p];
    if (flddif_p + cur_dph < dph1) {
        a.pixel_flood_depth_ptr[p] = dph1 - flddif_p;
    }
}

template <typename REAL, class A>
__device__ __forceinline__ void downscale_backwater_depth_members(A a, long b, long member)
{
    const long cat_off = member * a.num_catchments;
    const long pix_off = member * a.num_pixels;
    const long bw_off = member * a.num_backwater_pixels;
    if (a.batched_pixel_flood_diff) a.pixel_flood_diff_ptr += pix_off;
    if (a.batched_backwater_floor_elevation) a.backwater_floor_elevation_ptr += bw_off;
    a.catchment_tributary_modified_ptr += cat_off;
    a.pixel_flood_depth_ptr += pix_off;
    downscale_backwater_depth<REAL>(a, b);
}

template <typename REAL, class A>
__device__ __forceinline__ void downscale_permanent_water(const A& a, long p)
{
    if (a.permanent_water_depth > (REAL)0.0 && a.pixel_is_permanent_water_ptr[p]) {
        a.pixel_flood_depth_ptr[p] = fmax(
            a.permanent_water_depth, a.pixel_flood_depth_ptr[p]);
    }
}

template <typename REAL, class A>
__device__ __forceinline__ void downscale_permanent_water_members(A a, long p, long member)
{
    const long pix_off = member * a.num_pixels;
    if (a.batched_pixel_is_permanent_water) a.pixel_is_permanent_water_ptr += pix_off;
    a.pixel_flood_depth_ptr += pix_off;
    downscale_permanent_water<REAL>(a, p);
}

template <typename REAL, class A>
__device__ __forceinline__ void downscale_flood_duration(const A& a, long p)
{
    int ci = __ldg(a.pixel_catchment_idx_ptr + p);
    REAL diffdph = __ldg(a.pixel_flood_diff_ptr + p);
    REAL dur = (REAL)0.0;

    if (diffdph >= (REAL)0.0) {
        int bin_idx = (int)(diffdph / a.duration_depth_interval);
        if (bin_idx >= a.num_duration_bins) {
            bin_idx = a.num_duration_bins - 1;
        }
        dur = __ldg(
            a.flood_duration_table_ptr + (long)ci * a.num_duration_bins + bin_idx);
    }

    if (a.permanent_water_duration > (REAL)0.0 && a.pixel_is_permanent_water_ptr[p]) {
        dur = fmax(a.permanent_water_duration, dur);
    }

    a.pixel_flood_duration_ptr[p] = dur;
}

template <typename REAL, class A>
__device__ __forceinline__ void downscale_flood_duration_members(A a, long p, long member)
{
    const long cat_off = member * a.num_catchments;
    const long pix_off = member * a.num_pixels;
    if (a.batched_flood_duration_table) {
        a.flood_duration_table_ptr += cat_off * a.num_duration_bins;
    }
    if (a.batched_pixel_flood_diff) a.pixel_flood_diff_ptr += pix_off;
    if (a.batched_pixel_is_permanent_water) a.pixel_is_permanent_water_ptr += pix_off;
    a.pixel_flood_duration_ptr += pix_off;
    downscale_flood_duration<REAL>(a, p);
}
