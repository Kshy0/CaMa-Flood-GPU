# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Triton kernels for high-resolution flood-depth and inundation-duration downscaling."""

import triton
import triton.language as tl
from hydroforge.kernels import triton_math as hm


@triton.jit
def compute_downscale_flood_depth_kernel(
    pixel_catchment_idx_ptr,
    pixel_flood_diff_ptr,
    pixel_is_permanent_water_ptr,
    flood_depth_ptr,
    protected_depth_ptr,
    pixel_is_protected_ptr,
    pixel_flood_depth_ptr,
    permanent_water_depth,
    protected_depth_offset,
    num_pixels: tl.constexpr,
    num_catchments: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_LEVEE: tl.constexpr = False,
    HAS_TRIBUTARY: tl.constexpr = False,
):
    _ = num_catchments
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_pixels

    ci = tl.load(pixel_catchment_idx_ptr + offs, mask=mask, other=0)
    flddif = tl.load(pixel_flood_diff_ptr + offs, mask=mask, other=0.0)
    zero = tl.zeros_like(flddif)

    flddph = tl.load(flood_depth_ptr + ci, mask=mask, other=0.0)
    unprotected_dph = tl.where(flddph > flddif, flddph - flddif, zero)

    if HAS_LEVEE:
        is_prot = tl.load(pixel_is_protected_ptr + offs, mask=mask, other=False)
        levdph = tl.load(protected_depth_ptr + ci, mask=mask, other=0.0)
        prot_dph = tl.where(
            levdph > flddif, (levdph - flddif) + protected_depth_offset, zero
        )
        depth = tl.where(is_prot, prot_dph, unprotected_dph)
    else:
        depth = unprotected_dph

    if not HAS_TRIBUTARY:
        is_perm = tl.load(
            pixel_is_permanent_water_ptr + offs, mask=mask, other=False
        )
        depth = tl.where(
            (permanent_water_depth > 0.0) & is_perm,
            tl.maximum(hm.constant(permanent_water_depth, flddif), depth),
            depth,
        )

    tl.store(pixel_flood_depth_ptr + offs, depth, mask=mask)


@triton.jit
def compute_downscale_flood_depth_batched_kernel(
    pixel_catchment_idx_ptr,
    pixel_flood_diff_ptr,
    pixel_is_permanent_water_ptr,
    flood_depth_ptr,
    protected_depth_ptr,
    pixel_is_protected_ptr,
    pixel_flood_depth_ptr,
    permanent_water_depth,
    protected_depth_offset,
    num_pixels: tl.constexpr,
    num_catchments: tl.constexpr,
    ensemble_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_LEVEE: tl.constexpr,
    HAS_TRIBUTARY: tl.constexpr,
    batched_flood_depth: tl.constexpr,
    batched_protected_depth: tl.constexpr,
    batched_pixel_flood_diff: tl.constexpr,
    batched_pixel_is_permanent_water: tl.constexpr,
    batched_pixel_is_protected: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < (num_pixels * ensemble_size)

    member = idx // num_pixels
    offs = idx % num_pixels
    cat_off = member * num_catchments
    pix_off = member * num_pixels

    ci = tl.load(pixel_catchment_idx_ptr + offs, mask=mask, other=0)
    flddif = tl.load(
        pixel_flood_diff_ptr + (pix_off if batched_pixel_flood_diff else 0) + offs,
        mask=mask,
        other=0.0,
    )
    zero = tl.zeros_like(flddif)

    flddph = tl.load(
        flood_depth_ptr + (cat_off if batched_flood_depth else 0) + ci,
        mask=mask,
        other=0.0,
    )
    unprotected_dph = tl.where(flddph > flddif, flddph - flddif, zero)

    if HAS_LEVEE:
        is_prot = tl.load(
            pixel_is_protected_ptr
            + (pix_off if batched_pixel_is_protected else 0)
            + offs,
            mask=mask,
            other=False,
        )
        levdph = tl.load(
            protected_depth_ptr
            + (cat_off if batched_protected_depth else 0)
            + ci,
            mask=mask,
            other=0.0,
        )
        prot_dph = tl.where(
            levdph > flddif, (levdph - flddif) + protected_depth_offset, zero
        )
        depth = tl.where(is_prot, prot_dph, unprotected_dph)
    else:
        depth = unprotected_dph

    if not HAS_TRIBUTARY:
        is_perm = tl.load(
            pixel_is_permanent_water_ptr
            + (pix_off if batched_pixel_is_permanent_water else 0)
            + offs,
            mask=mask,
            other=False,
        )
        depth = tl.where(
            (permanent_water_depth > 0.0) & is_perm,
            tl.maximum(hm.constant(permanent_water_depth, flddif), depth),
            depth,
        )

    tl.store(pixel_flood_depth_ptr + pix_off + offs, depth, mask=mask)


@triton.jit
def compute_downscale_tributary_depth_kernel(
    tributary_pixel_idx_ptr,
    pixel_catchment_idx_ptr,
    tributary_edge_start_ptr,
    tributary_edge_order_ptr,
    tributary_upstream_pixel_idx_ptr,
    tributary_downstream_pixel_idx_ptr,
    tributary_chain_step_ptr,
    tributary_chain_length_ptr,
    pixel_flood_depth_ptr,
    catchment_tributary_modified_ptr,
    num_tributary_pixels: tl.constexpr,
    num_tributary_links: tl.constexpr,
    num_pixels: tl.constexpr,
    num_catchments: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    _ = num_pixels
    _ = num_catchments
    pid = tl.program_id(0)
    t = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = t < num_tributary_pixels

    p = tl.load(tributary_pixel_idx_ptr + t, mask=mask, other=0)
    old_dph = tl.load(pixel_flood_depth_ptr + p, mask=mask, other=0.0)
    best_dph = old_dph

    start = tl.load(tributary_edge_start_ptr + t, mask=mask, other=0)
    next_mask = mask & ((t + 1) < num_tributary_pixels)
    end = tl.where(
        (t + 1) < num_tributary_pixels,
        tl.load(tributary_edge_start_ptr + t + 1, mask=next_mask, other=num_tributary_links),
        num_tributary_links,
    )
    end = tl.where(mask, end, start)
    max_deg = tl.max(end - start, axis=0)

    for rel in range(max_deg):
        k = start + rel
        active = mask & (k < end)
        e = tl.load(tributary_edge_order_ptr + k, mask=active, other=0)
        up_p = tl.load(tributary_upstream_pixel_idx_ptr + e, mask=active, other=0)
        dn_p = tl.load(tributary_downstream_pixel_idx_ptr + e, mask=active, other=0)
        dph1 = tl.load(pixel_flood_depth_ptr + up_p, mask=active, other=0.0)
        dph2 = tl.load(pixel_flood_depth_ptr + dn_p, mask=active, other=0.0)
        step = tl.load(tributary_chain_step_ptr + e, mask=active, other=0.0)
        length = tl.load(tributary_chain_length_ptr + e, mask=active, other=1.0)
        ddph = tl.maximum(hm.divide(dph1 - dph2, length), 0.0)
        cand = dph1 - ddph * step
        best_dph = tl.where(active & (cand > best_dph), cand, best_dph)

    changed = mask & (best_dph > old_dph)
    tl.store(pixel_flood_depth_ptr + p, best_dph, mask=changed)
    ci = tl.load(pixel_catchment_idx_ptr + p, mask=changed, other=0)
    tl.atomic_max(
        catchment_tributary_modified_ptr + ci, 1, mask=changed, sem="relaxed"
    )


@triton.jit
def compute_downscale_tributary_depth_batched_kernel(
    tributary_pixel_idx_ptr,
    pixel_catchment_idx_ptr,
    tributary_edge_start_ptr,
    tributary_edge_order_ptr,
    tributary_upstream_pixel_idx_ptr,
    tributary_downstream_pixel_idx_ptr,
    tributary_chain_step_ptr,
    tributary_chain_length_ptr,
    pixel_flood_depth_ptr,
    catchment_tributary_modified_ptr,
    num_tributary_pixels: tl.constexpr,
    num_tributary_links: tl.constexpr,
    num_pixels: tl.constexpr,
    num_catchments: tl.constexpr,
    ensemble_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    batched_tributary_chain_step: tl.constexpr,
    batched_tributary_chain_length: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < (num_tributary_pixels * ensemble_size)

    member = idx // num_tributary_pixels
    t = idx % num_tributary_pixels
    cat_off = member * num_catchments
    pix_off = member * num_pixels
    link_off = member * num_tributary_links

    p = tl.load(tributary_pixel_idx_ptr + t, mask=mask, other=0)
    old_dph = tl.load(pixel_flood_depth_ptr + pix_off + p, mask=mask, other=0.0)
    best_dph = old_dph

    start = tl.load(tributary_edge_start_ptr + t, mask=mask, other=0)
    next_mask = mask & ((t + 1) < num_tributary_pixels)
    end = tl.where(
        (t + 1) < num_tributary_pixels,
        tl.load(tributary_edge_start_ptr + t + 1, mask=next_mask, other=num_tributary_links),
        num_tributary_links,
    )
    end = tl.where(mask, end, start)
    max_deg = tl.max(end - start, axis=0)

    for rel in range(max_deg):
        k = start + rel
        active = mask & (k < end)
        e = tl.load(tributary_edge_order_ptr + k, mask=active, other=0)
        up_p = tl.load(tributary_upstream_pixel_idx_ptr + e, mask=active, other=0)
        dn_p = tl.load(tributary_downstream_pixel_idx_ptr + e, mask=active, other=0)
        dph1 = tl.load(pixel_flood_depth_ptr + pix_off + up_p, mask=active, other=0.0)
        dph2 = tl.load(pixel_flood_depth_ptr + pix_off + dn_p, mask=active, other=0.0)
        step = tl.load(
            tributary_chain_step_ptr
            + (link_off if batched_tributary_chain_step else 0)
            + e,
            mask=active,
            other=0.0,
        )
        length = tl.load(
            tributary_chain_length_ptr
            + (link_off if batched_tributary_chain_length else 0)
            + e,
            mask=active,
            other=1.0,
        )
        ddph = tl.maximum(hm.divide(dph1 - dph2, length), 0.0)
        cand = dph1 - ddph * step
        best_dph = tl.where(active & (cand > best_dph), cand, best_dph)

    changed = mask & (best_dph > old_dph)
    tl.store(pixel_flood_depth_ptr + pix_off + p, best_dph, mask=changed)
    ci = tl.load(pixel_catchment_idx_ptr + p, mask=changed, other=0)
    tl.atomic_max(
        catchment_tributary_modified_ptr + cat_off + ci,
        1,
        mask=changed,
        sem="relaxed",
    )


@triton.jit
def compute_downscale_backwater_depth_kernel(
    backwater_pixel_idx_ptr,
    backwater_target_pixel_idx_ptr,
    pixel_catchment_idx_ptr,
    pixel_flood_diff_ptr,
    backwater_floor_elevation_ptr,
    catchment_tributary_modified_ptr,
    pixel_flood_depth_ptr,
    num_backwater_pixels: tl.constexpr,
    num_pixels: tl.constexpr,
    num_catchments: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    _ = num_pixels
    _ = num_catchments
    pid = tl.program_id(0)
    b = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = b < num_backwater_pixels

    p = tl.load(backwater_pixel_idx_ptr + b, mask=mask, other=0)
    ci = tl.load(pixel_catchment_idx_ptr + p, mask=mask, other=0)
    modified = tl.load(catchment_tributary_modified_ptr + ci, mask=mask, other=0) > 0
    active = mask & modified

    q = tl.load(backwater_target_pixel_idx_ptr + b, mask=active, other=0)
    floor_elv = tl.load(backwater_floor_elevation_ptr + b, mask=active, other=-1.0e20)
    flddif_q = tl.load(pixel_flood_diff_ptr + q, mask=active, other=0.0)
    dph_q = tl.load(pixel_flood_depth_ptr + q, mask=active, other=0.0)
    dph1 = tl.maximum(floor_elv, flddif_q + dph_q)

    flddif_p = tl.load(pixel_flood_diff_ptr + p, mask=active, other=0.0)
    cur_dph = tl.load(pixel_flood_depth_ptr + p, mask=active, other=0.0)
    update = active & ((flddif_p + cur_dph) < dph1)
    tl.store(pixel_flood_depth_ptr + p, dph1 - flddif_p, mask=update)


@triton.jit
def compute_downscale_backwater_depth_batched_kernel(
    backwater_pixel_idx_ptr,
    backwater_target_pixel_idx_ptr,
    pixel_catchment_idx_ptr,
    pixel_flood_diff_ptr,
    backwater_floor_elevation_ptr,
    catchment_tributary_modified_ptr,
    pixel_flood_depth_ptr,
    num_backwater_pixels: tl.constexpr,
    num_pixels: tl.constexpr,
    num_catchments: tl.constexpr,
    ensemble_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    batched_pixel_flood_diff: tl.constexpr,
    batched_backwater_floor_elevation: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < (num_backwater_pixels * ensemble_size)

    member = idx // num_backwater_pixels
    b = idx % num_backwater_pixels
    cat_off = member * num_catchments
    pix_off = member * num_pixels
    bw_off = member * num_backwater_pixels

    p = tl.load(backwater_pixel_idx_ptr + b, mask=mask, other=0)
    ci = tl.load(pixel_catchment_idx_ptr + p, mask=mask, other=0)
    modified = (
        tl.load(catchment_tributary_modified_ptr + cat_off + ci, mask=mask, other=0)
        > 0
    )
    active = mask & modified

    q = tl.load(backwater_target_pixel_idx_ptr + b, mask=active, other=0)
    floor_elv = tl.load(
        backwater_floor_elevation_ptr
        + (bw_off if batched_backwater_floor_elevation else 0)
        + b,
        mask=active,
        other=-1.0e20,
    )
    flddif_off = pix_off if batched_pixel_flood_diff else 0
    flddif_q = tl.load(pixel_flood_diff_ptr + flddif_off + q, mask=active, other=0.0)
    dph_q = tl.load(pixel_flood_depth_ptr + pix_off + q, mask=active, other=0.0)
    dph1 = tl.maximum(floor_elv, flddif_q + dph_q)

    flddif_p = tl.load(pixel_flood_diff_ptr + flddif_off + p, mask=active, other=0.0)
    cur_dph = tl.load(pixel_flood_depth_ptr + pix_off + p, mask=active, other=0.0)
    update = active & ((flddif_p + cur_dph) < dph1)
    tl.store(pixel_flood_depth_ptr + pix_off + p, dph1 - flddif_p, mask=update)


@triton.jit
def compute_downscale_permanent_water_kernel(
    pixel_is_permanent_water_ptr,
    pixel_flood_depth_ptr,
    permanent_water_depth,
    num_pixels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_pixels

    is_perm = tl.load(pixel_is_permanent_water_ptr + offs, mask=mask, other=False)
    active = mask & is_perm & (permanent_water_depth > 0.0)
    cur = tl.load(pixel_flood_depth_ptr + offs, mask=active, other=0.0)
    tl.store(
        pixel_flood_depth_ptr + offs,
        tl.maximum(hm.constant(permanent_water_depth, cur), cur),
        mask=active,
    )


@triton.jit
def compute_downscale_permanent_water_batched_kernel(
    pixel_is_permanent_water_ptr,
    pixel_flood_depth_ptr,
    permanent_water_depth,
    num_pixels: tl.constexpr,
    ensemble_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    batched_pixel_is_permanent_water: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < (num_pixels * ensemble_size)

    offs = idx % num_pixels
    perm_idx = idx if batched_pixel_is_permanent_water else offs
    is_perm = tl.load(
        pixel_is_permanent_water_ptr + perm_idx, mask=mask, other=False
    )
    active = mask & is_perm & (permanent_water_depth > 0.0)
    cur = tl.load(pixel_flood_depth_ptr + idx, mask=active, other=0.0)
    tl.store(
        pixel_flood_depth_ptr + idx,
        tl.maximum(hm.constant(permanent_water_depth, cur), cur),
        mask=active,
    )


@triton.jit
def compute_downscale_flood_duration_kernel(
    pixel_catchment_idx_ptr,
    pixel_flood_diff_ptr,
    pixel_is_permanent_water_ptr,
    flood_duration_table_ptr,
    pixel_flood_duration_ptr,
    duration_depth_interval,
    permanent_water_duration,
    num_pixels: tl.constexpr,
    num_catchments: tl.constexpr,
    num_duration_bins: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    _ = num_catchments
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_pixels

    ci = tl.load(pixel_catchment_idx_ptr + offs, mask=mask, other=0)
    diffdph = tl.load(pixel_flood_diff_ptr + offs, mask=mask, other=-1.0)
    non_neg = mask & (diffdph >= 0.0)
    raw_bin = hm.to_index(hm.divide(tl.maximum(diffdph, 0.0), duration_depth_interval))
    bin_idx = tl.minimum(raw_bin, num_duration_bins - 1)

    dur = tl.load(
        flood_duration_table_ptr + ci * num_duration_bins + bin_idx,
        mask=non_neg,
        other=0.0,
    )
    is_perm = tl.load(pixel_is_permanent_water_ptr + offs, mask=mask, other=False)
    dur = tl.where(
        (permanent_water_duration > 0.0) & is_perm,
        tl.maximum(hm.constant(permanent_water_duration, diffdph), dur),
        dur,
    )
    tl.store(pixel_flood_duration_ptr + offs, dur, mask=mask)


@triton.jit
def compute_downscale_flood_duration_batched_kernel(
    pixel_catchment_idx_ptr,
    pixel_flood_diff_ptr,
    pixel_is_permanent_water_ptr,
    flood_duration_table_ptr,
    pixel_flood_duration_ptr,
    duration_depth_interval,
    permanent_water_duration,
    num_pixels: tl.constexpr,
    num_catchments: tl.constexpr,
    num_duration_bins: tl.constexpr,
    ensemble_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    batched_flood_duration_table: tl.constexpr,
    batched_pixel_flood_diff: tl.constexpr,
    batched_pixel_is_permanent_water: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < (num_pixels * ensemble_size)

    member = idx // num_pixels
    offs = idx % num_pixels
    cat_off = member * num_catchments
    pix_off = member * num_pixels

    ci = tl.load(pixel_catchment_idx_ptr + offs, mask=mask, other=0)
    diffdph = tl.load(
        pixel_flood_diff_ptr + (pix_off if batched_pixel_flood_diff else 0) + offs,
        mask=mask,
        other=-1.0,
    )
    non_neg = mask & (diffdph >= 0.0)
    raw_bin = hm.to_index(hm.divide(tl.maximum(diffdph, 0.0), duration_depth_interval))
    bin_idx = tl.minimum(raw_bin, num_duration_bins - 1)

    table_off = (cat_off * num_duration_bins) if batched_flood_duration_table else 0
    dur = tl.load(
        flood_duration_table_ptr + table_off + ci * num_duration_bins + bin_idx,
        mask=non_neg,
        other=0.0,
    )
    is_perm = tl.load(
        pixel_is_permanent_water_ptr
        + (pix_off if batched_pixel_is_permanent_water else 0)
        + offs,
        mask=mask,
        other=False,
    )
    dur = tl.where(
        (permanent_water_duration > 0.0) & is_perm,
        tl.maximum(hm.constant(permanent_water_duration, diffdph), dur),
        dur,
    )
    tl.store(pixel_flood_duration_ptr + idx, dur, mask=mask)
