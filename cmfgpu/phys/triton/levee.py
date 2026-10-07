# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Triton kernels for levee-aware flood stage calculations."""

import triton
import triton.language as tl
from hydroforge.kernels import triton_math as hm


from cmfgpu import config as _constants

BACKFLOW_STORAGE_FRACTION = tl.constexpr(_constants.BACKFLOW_STORAGE_FRACTION)
ROUTING_SLOPE_LIMIT = tl.constexpr(_constants.ROUTING_SLOPE_LIMIT)


@triton.jit
def levee_stage_inline(
    flood_depth_row_ptr,
    mask,
    river_storage_hp,
    flood_storage_hp,
    river_depth,
    flood_depth,
    flood_fraction,
    river_length,
    river_width,
    river_max_storage,
    dwth_inc,
    levee_distance,
    ilev,
    levee_crown_height,
    levee_fraction,
    levee_base_height,
    levee_base_storage, s_top, levee_fill_storage, top_ilev,
    num_flood_levels: tl.constexpr,
):
    """Levee flood stage over the default stage of one block of levees.

    Returns the river, flood and protected storages and the river,
    flood and protected depths and flood fraction.
    """
    levee_crown_height = tl.maximum(levee_crown_height, levee_base_height)
    total_storage_hp = river_storage_hp + flood_storage_hp
    total_storage = hm.to_compute(total_storage_hp, river_length)
    zero = tl.zeros_like(river_length)
    not_found = tl.zeros_like(mask)

    above_base = (total_storage > river_max_storage) & ~(total_storage < levee_base_storage)
    is_case2 = above_base & (total_storage < s_top)
    above_top = above_base & ~(total_storage < s_top)
    is_case3 = above_top & (total_storage < levee_fill_storage)
    is_case4 = above_top & ~(total_storage < levee_fill_storage)

    dwth_pre = river_width
    dsto_fil_B = zero
    dwth_fil_B = zero
    ddph_fil_B = zero
    gradient_B = zero
    found_B = not_found

    dsto_fil_c4 = zero
    dwth_fil_c4 = zero
    gradient_c4 = zero
    found_c4 = not_found

    search_mask = mask & (is_case3 | is_case4)
    if tl.sum(hm.to_index(search_mask), axis=0) > 0:
        s_curr = river_max_storage
        dhgt_pre = zero
        for i in tl.static_range(num_flood_levels):
            depth_val = tl.load(flood_depth_row_ptr + i, mask=search_mask, other=0.0)
            dhgt_seg = depth_val - dhgt_pre
            dwth_mid = dwth_pre + 0.5 * dwth_inc
            s_next = s_curr + river_length * dwth_mid * dhgt_seg
            gradient = hm.divide(dhgt_seg, dwth_inc)

            # Case 3: the protected side fills layer by layer from the levee layer.
            dsto_add_wedge = (levee_distance + river_width) * (levee_crown_height - depth_val) * river_length
            threshold = s_next + dsto_add_wedge
            cond_check = (i >= ilev) & ~found_B
            cond_found = cond_check & (total_storage < threshold)
            cond_pass = cond_check & ~cond_found
            dsto_fil_B = tl.where(
                cond_pass, threshold, tl.where(i == ilev, top_ilev, dsto_fil_B),
            )
            dwth_fil_B = tl.where(cond_pass, dwth_inc * (i + 1) - levee_distance, dwth_fil_B)
            ddph_fil_B = tl.where(cond_pass, depth_val - levee_base_height, ddph_fil_B)
            gradient_B = tl.where(cond_found, gradient, gradient_B)
            found_B = found_B | cond_found

            # Case 4 stops at the first layer the storage does not exceed.
            cond_c4 = ~found_c4 & ~(total_storage > s_next)
            dsto_fil_c4 = tl.where(cond_c4, s_curr, dsto_fil_c4)
            dwth_fil_c4 = tl.where(cond_c4, dwth_pre, dwth_fil_c4)
            gradient_c4 = tl.where(cond_c4, gradient, gradient_c4)
            found_c4 = found_c4 | cond_c4

            s_curr = s_next
            dhgt_pre = depth_val
            dwth_pre += dwth_inc


    # Case 2: river side below the crown, protected side dry.
    f_dph_c2 = levee_base_height + hm.divide(
        hm.divide(total_storage - levee_base_storage, levee_distance + river_width),
        river_length,
    )
    r_sto_c2 = river_max_storage + river_length * river_width * f_dph_c2
    r_dph_c2 = hm.divide(hm.divide(r_sto_c2, river_length), river_width)

    # Case 3: river side at the crown, protected side filling.
    r_sto_c3 = river_max_storage + river_length * river_width * levee_crown_height
    r_dph_c3 = hm.divide(hm.divide(r_sto_c3, river_length), river_width)
    dsto_add_B = total_storage - dsto_fil_B
    dwth_add_B = -dwth_fil_B + hm.sqrt(
        dwth_fil_B * dwth_fil_B
        + hm.divide(hm.divide(2.0 * dsto_add_B, river_length), gradient_B)
    )
    p_dph_c3 = tl.where(
        found_B,
        levee_base_height + ddph_fil_B + dwth_add_B * gradient_B,
        levee_base_height + ddph_fil_B + hm.divide(hm.divide(dsto_add_B, dwth_fil_B), river_length),
    )
    f_frc_c3 = tl.where(
        found_B,
        hm.clamp(hm.divide(dwth_fil_B + levee_distance, dwth_inc * num_flood_levels), 0.0, 1.0),
        hm.constant(1.0, river_length),
    )

    # Case 4: above the crown; the default river stage stands, with the
    # unclamped default-stage flood fraction.
    dwth_add_c4 = tl.where(
        found_c4,
        -dwth_fil_c4 + hm.sqrt(
            dwth_fil_c4 * dwth_fil_c4
            + hm.divide(hm.divide(2.0 * (total_storage - dsto_fil_c4), river_length), gradient_c4)
        ),
        zero,
    )
    dwth_fil_c4 = tl.where(found_c4, dwth_fil_c4, dwth_pre)
    f_frc_c4 = hm.divide(-river_width + dwth_fil_c4 + dwth_add_c4, dwth_inc * num_flood_levels)
    dsto_add_c4 = (flood_depth - levee_crown_height) * (levee_distance + river_width) * river_length

    levee_partition = is_case2 | is_case3 | is_case4
    river_candidate = tl.where(is_case2, r_sto_c2, r_sto_c3)
    river_storage_new = tl.where(
        is_case2 | is_case3,
        river_candidate.to(total_storage_hp.dtype), river_storage_hp,
    )
    flood_top = tl.where(is_case3, s_top, s_top + dsto_add_c4).to(total_storage_hp.dtype)
    flood_storage_new = tl.where(
        levee_partition,
        hm.at_least(tl.where(is_case2, total_storage_hp, flood_top) - river_storage_new, 0.0),
        flood_storage_hp,
    )
    protected_storage_new = tl.where(
        is_case3 | is_case4,
        hm.at_least(total_storage_hp - river_storage_new - flood_storage_new, 0.0),
        tl.zeros_like(total_storage_hp),
    )

    river_depth_new = tl.where(is_case2, r_dph_c2, tl.where(is_case3, r_dph_c3, river_depth))
    flood_depth_new = tl.where(is_case2, f_dph_c2, tl.where(is_case3, levee_crown_height, flood_depth))
    protected_depth_new = tl.where(is_case3, p_dph_c3, tl.where(is_case4, flood_depth, zero))
    flood_fraction_new = tl.where(
        is_case2, levee_fraction,
        tl.where(is_case3, f_frc_c3, tl.where(is_case4, f_frc_c4, flood_fraction)),
    )
    return (
        river_storage_new, flood_storage_new, protected_storage_new,
        river_depth_new, flood_depth_new, protected_depth_new, flood_fraction_new,
    )


@triton.jit
def compute_levee_stage_kernel(
    levee_catchment_idx_ptr,
    levee_river_max_storage_ptr,
    levee_base_storage_ptr,
    levee_top_storage_ptr,
    levee_fill_storage_ptr,
    levee_layer_top_storage_ptr,
    river_storage_ptr,                      # *f64
    flood_storage_ptr,                      # *f64
    protected_storage_ptr,                  # *f64
    river_depth_ptr,
    flood_depth_ptr,
    protected_depth_ptr,
    river_height_ptr,
    flood_depth_table_ptr,
    catchment_area_ptr,
    river_width_ptr,
    river_length_ptr,
    levee_base_height_ptr,
    levee_crown_height_ptr,
    levee_fraction_ptr,
    flood_fraction_ptr,
    num_levees: tl.constexpr,
    num_flood_levels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_catchments: tl.constexpr,
):
    _ = num_catchments
    pid = tl.program_id(0)
    levee_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = levee_offs < num_levees
    levee_catchment_idx = tl.load(levee_catchment_idx_ptr + levee_offs, mask=mask, other=0)

    river_length = tl.load(river_length_ptr + levee_catchment_idx, mask=mask, other=1.0)
    river_width = tl.load(river_width_ptr + levee_catchment_idx, mask=mask, other=1.0)
    river_height = tl.load(river_height_ptr + levee_catchment_idx, mask=mask, other=0.0)
    catchment_area = tl.load(catchment_area_ptr + levee_catchment_idx, mask=mask, other=0.0)
    levee_crown_height = tl.load(levee_crown_height_ptr + levee_offs, mask=mask, other=0.0)
    levee_fraction = tl.load(levee_fraction_ptr + levee_offs, mask=mask, other=0.0)
    levee_base_height = tl.load(levee_base_height_ptr + levee_offs, mask=mask, other=0.0)

    river_max_storage = river_length * river_width * river_height
    dwth_inc = hm.divide(hm.divide(catchment_area, river_length), num_flood_levels)
    levee_distance = levee_fraction * hm.divide(catchment_area, river_length)
    ilev = hm.to_index(levee_fraction * num_flood_levels)

    river_max_storage = tl.load(levee_river_max_storage_ptr + levee_offs, mask=mask, other=0.0)
    levee_base_storage = tl.load(levee_base_storage_ptr + levee_offs, mask=mask, other=0.0)
    s_top = tl.load(levee_top_storage_ptr + levee_offs, mask=mask, other=0.0)
    levee_fill_storage = tl.load(levee_fill_storage_ptr + levee_offs, mask=mask, other=0.0)
    top_ilev = tl.load(levee_layer_top_storage_ptr + levee_offs, mask=mask, other=0.0)
    r_sto, f_sto, p_sto, r_dph, f_dph, p_dph, f_frc = levee_stage_inline(
        flood_depth_table_ptr + levee_catchment_idx * num_flood_levels,
        mask,
        tl.load(river_storage_ptr + levee_catchment_idx, mask=mask, other=0.0),
        tl.load(flood_storage_ptr + levee_catchment_idx, mask=mask, other=0.0),
        tl.load(river_depth_ptr + levee_catchment_idx, mask=mask, other=0.0),
        tl.load(flood_depth_ptr + levee_catchment_idx, mask=mask, other=0.0),
        tl.load(flood_fraction_ptr + levee_catchment_idx, mask=mask, other=0.0),
        river_length, river_width, river_max_storage, dwth_inc, levee_distance, ilev,
        levee_crown_height, levee_fraction, levee_base_height,
        levee_base_storage, s_top, levee_fill_storage, top_ilev, num_flood_levels,
    )

    tl.store(river_storage_ptr + levee_catchment_idx, r_sto, mask=mask)
    tl.store(flood_storage_ptr + levee_catchment_idx, f_sto, mask=mask)
    tl.store(protected_storage_ptr + levee_catchment_idx, p_sto, mask=mask)
    tl.store(river_depth_ptr + levee_catchment_idx, r_dph, mask=mask)
    tl.store(flood_depth_ptr + levee_catchment_idx, f_dph, mask=mask)
    tl.store(protected_depth_ptr + levee_catchment_idx, p_dph, mask=mask)
    tl.store(flood_fraction_ptr + levee_catchment_idx, f_frc, mask=mask)


@triton.jit
def compute_levee_stage_log_kernel(
    levee_catchment_idx_ptr,
    levee_river_max_storage_ptr,
    levee_base_storage_ptr,
    levee_top_storage_ptr,
    levee_fill_storage_ptr,
    levee_layer_top_storage_ptr,
    river_storage_ptr,                      # *f64
    flood_storage_ptr,                      # *f64
    protected_storage_ptr,                  # *f64
    river_depth_ptr,
    flood_depth_ptr,
    protected_depth_ptr,
    river_height_ptr,
    flood_depth_table_ptr,
    catchment_area_ptr,
    river_width_ptr,
    river_length_ptr,
    levee_base_height_ptr,
    levee_crown_height_ptr,
    levee_fraction_ptr,
    flood_fraction_ptr,
    total_storage_stage_sum_ptr,
    river_storage_sum_ptr,
    flood_storage_sum_ptr,
    flood_area_sum_ptr,
    total_stage_error_sum_ptr,
    current_step_ptr,
    num_levees: tl.constexpr,
    num_flood_levels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 128,
):
    pid = tl.program_id(0)
    levee_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = levee_offs < num_levees
    current_step = tl.load(current_step_ptr)
    levee_catchment_idx = tl.load(levee_catchment_idx_ptr + levee_offs, mask=mask, other=0)

    river_length = tl.load(river_length_ptr + levee_catchment_idx, mask=mask, other=1.0)
    river_width = tl.load(river_width_ptr + levee_catchment_idx, mask=mask, other=1.0)
    river_height = tl.load(river_height_ptr + levee_catchment_idx, mask=mask, other=0.0)
    catchment_area = tl.load(catchment_area_ptr + levee_catchment_idx, mask=mask, other=0.0)
    levee_crown_height = tl.load(levee_crown_height_ptr + levee_offs, mask=mask, other=0.0)
    levee_fraction = tl.load(levee_fraction_ptr + levee_offs, mask=mask, other=0.0)
    levee_base_height = tl.load(levee_base_height_ptr + levee_offs, mask=mask, other=0.0)

    river_max_storage = river_length * river_width * river_height
    dwth_inc = hm.divide(hm.divide(catchment_area, river_length), num_flood_levels)
    levee_distance = levee_fraction * hm.divide(catchment_area, river_length)
    ilev = hm.to_index(levee_fraction * num_flood_levels)

    river_storage_hp = tl.load(river_storage_ptr + levee_catchment_idx, mask=mask, other=0.0)
    flood_storage_hp = tl.load(flood_storage_ptr + levee_catchment_idx, mask=mask, other=0.0)
    river_max_storage = tl.load(levee_river_max_storage_ptr + levee_offs, mask=mask, other=0.0)
    levee_base_storage = tl.load(levee_base_storage_ptr + levee_offs, mask=mask, other=0.0)
    s_top = tl.load(levee_top_storage_ptr + levee_offs, mask=mask, other=0.0)
    levee_fill_storage = tl.load(levee_fill_storage_ptr + levee_offs, mask=mask, other=0.0)
    top_ilev = tl.load(levee_layer_top_storage_ptr + levee_offs, mask=mask, other=0.0)
    r_sto, f_sto, p_sto, r_dph, f_dph, p_dph, f_frc = levee_stage_inline(
        flood_depth_table_ptr + levee_catchment_idx * num_flood_levels,
        mask,
        river_storage_hp,
        flood_storage_hp,
        tl.load(river_depth_ptr + levee_catchment_idx, mask=mask, other=0.0),
        tl.load(flood_depth_ptr + levee_catchment_idx, mask=mask, other=0.0),
        tl.load(flood_fraction_ptr + levee_catchment_idx, mask=mask, other=0.0),
        river_length, river_width, river_max_storage, dwth_inc, levee_distance, ilev,
        levee_crown_height, levee_fraction, levee_base_height,
        levee_base_storage, s_top, levee_fill_storage, top_ilev, num_flood_levels,
    )

    # Log variables
    total_storage_stage_new = r_sto + f_sto + p_sto
    tl.atomic_add(total_storage_stage_sum_ptr + current_step, tl.sum(total_storage_stage_new) * 1e-9)
    tl.atomic_add(river_storage_sum_ptr + current_step, tl.sum(r_sto) * 1e-9)
    tl.atomic_add(flood_storage_sum_ptr + current_step, tl.sum(f_sto) * 1e-9)
    tl.atomic_add(flood_area_sum_ptr + current_step, tl.sum(f_frc * catchment_area) * 1e-9)
    tl.atomic_add(
        total_stage_error_sum_ptr + current_step,
        tl.sum(total_storage_stage_new - (river_storage_hp + flood_storage_hp)) * 1e-9,
    )

    tl.store(river_storage_ptr + levee_catchment_idx, r_sto, mask=mask)
    tl.store(flood_storage_ptr + levee_catchment_idx, f_sto, mask=mask)
    tl.store(protected_storage_ptr + levee_catchment_idx, p_sto, mask=mask)
    tl.store(river_depth_ptr + levee_catchment_idx, r_dph, mask=mask)
    tl.store(flood_depth_ptr + levee_catchment_idx, f_dph, mask=mask)
    tl.store(protected_depth_ptr + levee_catchment_idx, p_dph, mask=mask)
    tl.store(flood_fraction_ptr + levee_catchment_idx, f_frc, mask=mask)


@triton.jit
def compute_levee_bifurcation_outflow_kernel(
    # Indices and configuration
    bifurcation_catchment_idx_ptr,                          # *i32: Catchment indices
    bifurcation_downstream_idx_ptr,                         # *i32: Downstream indices
    bifurcation_manning_ptr,                    # *f32: Bifurcation Manning coefficient
    bifurcation_outflow_ptr,                    # *f32: Bifurcation outflow (in/out)
    bifurcation_width_ptr,                      # *f32: Bifurcation width
    bifurcation_length_ptr,                     # *f32: Bifurcation length
    bifurcation_elevation_ptr,                  # *f32: Bifurcation length
    bifurcation_cross_section_depth_ptr,   # *f32: Bifurcation cross-section depth
    river_depth_ptr,                            # *f32: River depth
    protected_depth_ptr,                        # *f32: Protected depth
    river_height_ptr,                           # *f32: River bank height
    catchment_elevation_ptr,                    # *f32: Catchment elevation
    is_levee_ptr,                               # *bool: Levee mask
    river_storage_ptr,                          # *f64: River storage
    flood_storage_ptr,                          # *f64: Flood storage
    protected_storage_ptr,                      # *f64: Protected storage
    outgoing_storage_ptr,                       # *f64: Outgoing storage (in/out)
    gravity: tl.constexpr,                      # f32: Gravity constant
    time_step_ptr,                                  # f32: Time step
    num_bifurcation_paths: tl.constexpr,        # Total number of bifurcation paths
    num_bifurcation_levels: tl.constexpr,       # int: Number of bifurcation levels    
    BLOCK_SIZE: tl.constexpr,                   # Block size
    num_catchments: tl.constexpr,
):
    _ = num_catchments
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_bifurcation_paths
    time_step = tl.load(time_step_ptr)
    
    # Load indices
    bifurcation_catchment_idx = tl.load(bifurcation_catchment_idx_ptr + offs, mask=mask, other=0)
    bifurcation_downstream_idx = tl.load(bifurcation_downstream_idx_ptr + offs, mask=mask, other=0)
    
    # Load bifurcation properties
    
    bifurcation_length = tl.load(bifurcation_length_ptr + offs, mask=mask, other=0.0)
    
    # Derive diagnostics from persistent state.
    catchment_elevation = tl.load(
        catchment_elevation_ptr + bifurcation_catchment_idx,
        mask=mask, other=0.0,
    )
    downstream_elevation = tl.load(
        catchment_elevation_ptr + bifurcation_downstream_idx,
        mask=mask, other=0.0,
    )
    # D2SFCELV = D2RIVELV + D2RIVDPH with D2RIVELV = D2ELEVTN - D2RIVHGT.
    bifurcation_water_surface_elevation = (
        tl.load(river_depth_ptr + bifurcation_catchment_idx, mask=mask, other=0.0)
        + (catchment_elevation
           - tl.load(river_height_ptr + bifurcation_catchment_idx, mask=mask, other=0.0))
    )
    bifurcation_water_surface_elevation_downstream = (
        tl.load(river_depth_ptr + bifurcation_downstream_idx, mask=mask, other=0.0)
        + (downstream_elevation
           - tl.load(river_height_ptr + bifurcation_downstream_idx, mask=mask, other=0.0))
    )
    max_bifurcation_water_surface_elevation = tl.maximum(bifurcation_water_surface_elevation, bifurcation_water_surface_elevation_downstream)

    # Protected-side WSE is only distinct at levee catchments.
    bifurcation_protected_water_surface_elevation = tl.where(
        tl.load(is_levee_ptr + bifurcation_catchment_idx, mask=mask, other=False),
        tl.minimum(
            catchment_elevation
            + tl.load(protected_depth_ptr + bifurcation_catchment_idx, mask=mask, other=0.0),
            bifurcation_water_surface_elevation,
        ),
        bifurcation_water_surface_elevation,
    )
    bifurcation_protected_water_surface_elevation_downstream = tl.where(
        tl.load(is_levee_ptr + bifurcation_downstream_idx, mask=mask, other=False),
        tl.minimum(
            downstream_elevation
            + tl.load(protected_depth_ptr + bifurcation_downstream_idx, mask=mask, other=0.0),
            bifurcation_water_surface_elevation_downstream,
        ),
        bifurcation_water_surface_elevation_downstream,
    )
    max_bifurcation_protected_water_surface_elevation = tl.maximum(bifurcation_protected_water_surface_elevation, bifurcation_protected_water_surface_elevation_downstream)

    # Bifurcation slope (clamped similarly to flood slope)
    bifurcation_slope = hm.divide(bifurcation_water_surface_elevation - bifurcation_water_surface_elevation_downstream, bifurcation_length)
    bifurcation_slope = hm.clamp(bifurcation_slope, -ROUTING_SLOPE_LIMIT, ROUTING_SLOPE_LIMIT)

    # Storage change limiter calculation
    bifurcation_total_storage = hm.to_compute(
        tl.load(river_storage_ptr + bifurcation_catchment_idx, mask=mask, other=0.0)
        + tl.load(flood_storage_ptr + bifurcation_catchment_idx, mask=mask, other=0.0)
        + tl.load(protected_storage_ptr + bifurcation_catchment_idx, mask=mask, other=0.0),
        bifurcation_length,
    )
    bifurcation_total_storage_downstream = hm.to_compute(
        tl.load(river_storage_ptr + bifurcation_downstream_idx, mask=mask, other=0.0)
        + tl.load(flood_storage_ptr + bifurcation_downstream_idx, mask=mask, other=0.0)
        + tl.load(protected_storage_ptr + bifurcation_downstream_idx, mask=mask, other=0.0),
        bifurcation_length,
    )
    sum_bifurcation_outflow = tl.zeros_like(bifurcation_length)

    for level in tl.static_range(num_bifurcation_levels):
        
        level_idx = offs * num_bifurcation_levels + level
        bifurcation_manning = tl.load(bifurcation_manning_ptr + level_idx, mask=mask, other=0.0)
        bifurcation_cross_section_depth = tl.load(bifurcation_cross_section_depth_ptr + level_idx, mask=mask, other=0.0)
        bifurcation_elevation = tl.load(bifurcation_elevation_ptr + level_idx, mask=mask, other=0.0)
        
        # Calculate bifurcation cross-section depth
        # Level 0: River channel (use river WSE)
        # Level > 0: Overland (use protected WSE)
        
        if level == 0:
            current_max_wse = max_bifurcation_water_surface_elevation
        else:
            current_max_wse = max_bifurcation_protected_water_surface_elevation

        updated_bifurcation_cross_section_depth = tl.maximum(current_max_wse - bifurcation_elevation, 0.0)
        
        # Calculate semi-implicit flow depth for bifurcation
        # Level 0: Semi-implicit
        # Level > 0: Explicit (no semi-implicit)
        
        if level == 0:
            semi_implicit_depth = hm.sqrt(
                updated_bifurcation_cross_section_depth
                * bifurcation_cross_section_depth,
            )
            bifurcation_semi_implicit_flow_depth = tl.where(
                semi_implicit_depth <= 0.0,
                updated_bifurcation_cross_section_depth,
                semi_implicit_depth,
            )
        else:
            bifurcation_semi_implicit_flow_depth = updated_bifurcation_cross_section_depth
        
        bifurcation_width = tl.load(bifurcation_width_ptr + level_idx, mask=mask, other=0.0)
        bifurcation_outflow = tl.load(bifurcation_outflow_ptr + level_idx, mask=mask, other=0.0)

        unit_bifurcation_outflow = hm.divide(bifurcation_outflow, bifurcation_width)

        numerator = bifurcation_width * (
            unit_bifurcation_outflow + gravity * time_step 
            * bifurcation_semi_implicit_flow_depth * bifurcation_slope
        )
        denominator = 1.0 + gravity * time_step * (bifurcation_manning * bifurcation_manning) * tl.abs(unit_bifurcation_outflow) \
                    * hm.divide(1.0, bifurcation_semi_implicit_flow_depth * bifurcation_semi_implicit_flow_depth * hm.cbrt(bifurcation_semi_implicit_flow_depth))
        
        updated_bifurcation_outflow = hm.divide(numerator, denominator)
        bifurcation_condition = bifurcation_semi_implicit_flow_depth > hm.constant(1e-5, bifurcation_semi_implicit_flow_depth)
        updated_bifurcation_outflow = tl.where(bifurcation_condition, updated_bifurcation_outflow, 0.0)
        sum_bifurcation_outflow += updated_bifurcation_outflow
        tl.store(bifurcation_cross_section_depth_ptr + level_idx, updated_bifurcation_cross_section_depth, mask=mask)
        tl.store(bifurcation_outflow_ptr + level_idx, updated_bifurcation_outflow, mask=mask)
    # CaMa-Flood LEVEE_OPT_PTHOUT limits a path only when its flow sum is non-zero.
    limit_rate = tl.where(
        sum_bifurcation_outflow != 0.0,
        hm.at_most(hm.divide(BACKFLOW_STORAGE_FRACTION * tl.minimum(bifurcation_total_storage, bifurcation_total_storage_downstream), tl.abs(sum_bifurcation_outflow) * time_step), 1.0),
        hm.constant(1.0, sum_bifurcation_outflow),
    )
    sum_bifurcation_outflow *= limit_rate
    for level in tl.static_range(num_bifurcation_levels):
        level_idx = offs * num_bifurcation_levels + level
        updated_bifurcation_outflow = tl.load(bifurcation_outflow_ptr + level_idx, mask=mask)
        updated_bifurcation_outflow *= limit_rate
        tl.store(bifurcation_outflow_ptr + level_idx, updated_bifurcation_outflow, mask=mask)

    # P2STOOUT flows, multiplied by the step in compute_inflow.
    pos_flow = tl.maximum(sum_bifurcation_outflow, 0.0)
    neg_flow = tl.minimum(sum_bifurcation_outflow, 0.0)
    tl.atomic_add(outgoing_storage_ptr + bifurcation_catchment_idx, pos_flow, mask=mask, sem="relaxed")
    tl.atomic_add(outgoing_storage_ptr + bifurcation_downstream_idx, -neg_flow, mask=mask, sem="relaxed")


@triton.jit
def compute_levee_stage_batched_kernel(
    levee_catchment_idx_ptr,
    levee_river_max_storage_ptr,
    levee_base_storage_ptr,
    levee_top_storage_ptr,
    levee_fill_storage_ptr,
    levee_layer_top_storage_ptr,
    river_storage_ptr,                      # *f64
    flood_storage_ptr,                      # *f64
    protected_storage_ptr,                  # *f64
    river_depth_ptr,
    flood_depth_ptr,
    protected_depth_ptr,
    river_height_ptr,
    flood_depth_table_ptr,
    catchment_area_ptr,
    river_width_ptr,
    river_length_ptr,
    levee_base_height_ptr,
    levee_crown_height_ptr,
    levee_fraction_ptr,
    flood_fraction_ptr,
    num_levees: tl.constexpr,
    num_flood_levels: tl.constexpr,
    ensemble_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_catchments: tl.constexpr,
    # Batch flags
    batched_river_length: tl.constexpr,
    batched_river_width: tl.constexpr,
    batched_river_height: tl.constexpr,
    batched_catchment_area: tl.constexpr,
    batched_levee_crown_height: tl.constexpr,
    batched_levee_fraction: tl.constexpr,
    batched_levee_base_height: tl.constexpr,
    batched_flood_depth_table: tl.constexpr
):
    # --- Loop-based batched kernel ---
    # Grid = cdiv(num_levees, BLOCK_SIZE), each block loops over members.
    # Shared (non-member) parameters are loaded once and reused across members.
    pid = tl.program_id(0)
    levee_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = levee_offs < num_levees

    # Topology is never batched
    levee_catchment_idx = tl.load(levee_catchment_idx_ptr + levee_offs, mask=mask, other=0)

    # ---- Load shared (non-member) parameters once ----
    if not batched_river_length:
        river_length_shared = tl.load(river_length_ptr + levee_catchment_idx, mask=mask, other=1.0)
    if not batched_river_width:
        river_width_shared = tl.load(river_width_ptr + levee_catchment_idx, mask=mask, other=1.0)
    if not batched_river_height:
        river_height_shared = tl.load(river_height_ptr + levee_catchment_idx, mask=mask, other=0.0)
    if not batched_catchment_area:
        catchment_area_shared = tl.load(catchment_area_ptr + levee_catchment_idx, mask=mask, other=0.0)
    if not batched_levee_crown_height:
        levee_crown_height_shared = tl.load(levee_crown_height_ptr + levee_offs, mask=mask, other=0.0)
    if not batched_levee_fraction:
        levee_fraction_shared = tl.load(levee_fraction_ptr + levee_offs, mask=mask, other=0.0)
    if not batched_levee_base_height:
        levee_base_height_shared = tl.load(levee_base_height_ptr + levee_offs, mask=mask, other=0.0)

    # Pre-compute derived constants that don't change across members
    if not batched_river_length and not batched_river_width and not batched_river_height:
        river_max_storage_shared = river_length_shared * river_width_shared * river_height_shared
    if not batched_catchment_area and not batched_river_length:
        dwth_inc_shared = hm.divide(hm.divide(catchment_area_shared, river_length_shared), num_flood_levels)
    if not batched_levee_fraction and not batched_catchment_area and not batched_river_length:
        levee_distance_shared = levee_fraction_shared * hm.divide(catchment_area_shared, river_length_shared)
    if not batched_levee_fraction:
        ilev_shared = hm.to_index(levee_fraction_shared * num_flood_levels)

    # ---- Loop over members ----
    for t in tl.static_range(ensemble_size):
        member_offset_catchments = t * num_catchments
        member_offset_levees = t * num_levees

        # Use pre-loaded shared values or load per-member batched values
        river_length = tl.load(river_length_ptr + member_offset_catchments + levee_catchment_idx, mask=mask, other=1.0) if batched_river_length else river_length_shared
        river_width = tl.load(river_width_ptr + member_offset_catchments + levee_catchment_idx, mask=mask, other=1.0) if batched_river_width else river_width_shared
        river_height = tl.load(river_height_ptr + member_offset_catchments + levee_catchment_idx, mask=mask, other=0.0) if batched_river_height else river_height_shared
        catchment_area = tl.load(catchment_area_ptr + member_offset_catchments + levee_catchment_idx, mask=mask, other=0.0) if batched_catchment_area else catchment_area_shared
        levee_crown_height = tl.load(levee_crown_height_ptr + member_offset_levees + levee_offs, mask=mask, other=0.0) if batched_levee_crown_height else levee_crown_height_shared
        levee_fraction = tl.load(levee_fraction_ptr + member_offset_levees + levee_offs, mask=mask, other=0.0) if batched_levee_fraction else levee_fraction_shared
        levee_base_height = tl.load(levee_base_height_ptr + member_offset_levees + levee_offs, mask=mask, other=0.0) if batched_levee_base_height else levee_base_height_shared

        # Use pre-computed derived constants when possible
        if batched_river_length or batched_river_width or batched_river_height:
            river_max_storage = river_length * river_width * river_height
        else:
            river_max_storage = river_max_storage_shared
        if batched_catchment_area or batched_river_length:
            dwth_inc = hm.divide(hm.divide(catchment_area, river_length), num_flood_levels)
        else:
            dwth_inc = dwth_inc_shared
        if batched_levee_fraction or batched_catchment_area or batched_river_length:
            levee_distance = levee_fraction * hm.divide(catchment_area, river_length)
        else:
            levee_distance = levee_distance_shared
        if batched_levee_fraction:
            ilev = hm.to_index(levee_fraction * num_flood_levels)
        else:
            ilev = ilev_shared

        if batched_flood_depth_table:
            table_base_offset = member_offset_catchments * num_flood_levels
        else:
            table_base_offset = 0

        cell = member_offset_catchments + levee_catchment_idx
        river_max_storage = tl.load(levee_river_max_storage_ptr + member_offset_levees + levee_offs, mask=mask, other=0.0)
        levee_base_storage = tl.load(levee_base_storage_ptr + member_offset_levees + levee_offs, mask=mask, other=0.0)
        s_top = tl.load(levee_top_storage_ptr + member_offset_levees + levee_offs, mask=mask, other=0.0)
        levee_fill_storage = tl.load(levee_fill_storage_ptr + member_offset_levees + levee_offs, mask=mask, other=0.0)
        top_ilev = tl.load(levee_layer_top_storage_ptr + member_offset_levees + levee_offs, mask=mask, other=0.0)
        r_sto, f_sto, p_sto, r_dph, f_dph, p_dph, f_frc = levee_stage_inline(
            flood_depth_table_ptr + table_base_offset + levee_catchment_idx * num_flood_levels,
            mask,
            tl.load(river_storage_ptr + cell, mask=mask, other=0.0),
            tl.load(flood_storage_ptr + cell, mask=mask, other=0.0),
            tl.load(river_depth_ptr + cell, mask=mask, other=0.0),
            tl.load(flood_depth_ptr + cell, mask=mask, other=0.0),
            tl.load(flood_fraction_ptr + cell, mask=mask, other=0.0),
            river_length, river_width, river_max_storage, dwth_inc, levee_distance, ilev,
            levee_crown_height, levee_fraction, levee_base_height,
            levee_base_storage, s_top, levee_fill_storage, top_ilev, num_flood_levels,
        )

        tl.store(river_storage_ptr + cell, r_sto, mask=mask)
        tl.store(flood_storage_ptr + cell, f_sto, mask=mask)
        tl.store(protected_storage_ptr + cell, p_sto, mask=mask)
        tl.store(river_depth_ptr + cell, r_dph, mask=mask)
        tl.store(flood_depth_ptr + cell, f_dph, mask=mask)
        tl.store(protected_depth_ptr + cell, p_dph, mask=mask)
        tl.store(flood_fraction_ptr + cell, f_frc, mask=mask)


@triton.jit
def compute_levee_bifurcation_outflow_batched_kernel(
    # Indices and configuration
    bifurcation_catchment_idx_ptr,                          # *i32: Catchment indices
    bifurcation_downstream_idx_ptr,                         # *i32: Downstream indices
    bifurcation_manning_ptr,                    # *f32: Bifurcation Manning coefficient
    bifurcation_outflow_ptr,                    # *f32: Bifurcation outflow (in/out)
    bifurcation_width_ptr,                      # *f32: Bifurcation width
    bifurcation_length_ptr,                     # *f32: Bifurcation length
    bifurcation_elevation_ptr,                  # *f32: Bifurcation length
    bifurcation_cross_section_depth_ptr,   # *f32: Bifurcation cross-section depth
    river_depth_ptr,                            # *f32: River depth
    protected_depth_ptr,                        # *f32: Protected depth
    river_height_ptr,                           # *f32: River bank height
    catchment_elevation_ptr,                    # *f32: Catchment elevation
    is_levee_ptr,                               # *bool: Levee mask
    river_storage_ptr,                          # *f64: River storage
    flood_storage_ptr,                          # *f64: Flood storage
    protected_storage_ptr,                      # *f64: Protected storage
    outgoing_storage_ptr,                       # *f64: Outgoing storage (in/out)
    gravity: tl.constexpr,                      # f32: Gravity constant
    time_step_ptr,                                  # f32: Time step
    num_bifurcation_paths: tl.constexpr,        # Total number of bifurcation paths
    num_bifurcation_levels: tl.constexpr,       # int: Number of bifurcation levels    
    ensemble_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,                    # Block size
    num_catchments: tl.constexpr,
    # Batch flags
    batched_bifurcation_manning: tl.constexpr,
    batched_bifurcation_width: tl.constexpr,
    batched_bifurcation_length: tl.constexpr,
    batched_bifurcation_elevation: tl.constexpr,
    batched_river_height: tl.constexpr,
    batched_catchment_elevation: tl.constexpr,
):
    pid_x = tl.program_id(0)
    idx = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate member and path indices
    member_index = idx // num_bifurcation_paths
    offs = idx % num_bifurcation_paths
    
    mask = idx < (num_bifurcation_paths * ensemble_size)
    time_step = tl.load(time_step_ptr)
    
    member_offset_paths = member_index * num_bifurcation_paths
    member_offset_catchments = member_index * num_catchments
    member_offset_levels = member_index * num_bifurcation_paths * num_bifurcation_levels
    
    # Load indices
    # Topology is never batched
    bifurcation_catchment_idx = tl.load(bifurcation_catchment_idx_ptr + offs, mask=mask, other=0)
    bifurcation_downstream_idx = tl.load(bifurcation_downstream_idx_ptr + offs, mask=mask, other=0)
    
    # Load bifurcation properties
    bifurcation_length = tl.load(bifurcation_length_ptr + (member_offset_paths if batched_bifurcation_length else 0) + offs, mask=mask, other=0.0)
    
    # Derive diagnostics from this member's source fields.
    catchment_cell = member_offset_catchments + bifurcation_catchment_idx
    downstream_cell = member_offset_catchments + bifurcation_downstream_idx
    catchment_height_idx = (
        catchment_cell if batched_river_height else bifurcation_catchment_idx
    )
    downstream_height_idx = (
        downstream_cell if batched_river_height else bifurcation_downstream_idx
    )
    catchment_elevation_idx = (
        catchment_cell
        if batched_catchment_elevation else bifurcation_catchment_idx
    )
    downstream_elevation_idx = (
        downstream_cell
        if batched_catchment_elevation else bifurcation_downstream_idx
    )
    catchment_elevation = tl.load(
        catchment_elevation_ptr + catchment_elevation_idx,
        mask=mask, other=0.0,
    )
    downstream_elevation = tl.load(
        catchment_elevation_ptr + downstream_elevation_idx,
        mask=mask, other=0.0,
    )
    # D2SFCELV = D2RIVELV + D2RIVDPH with D2RIVELV = D2ELEVTN - D2RIVHGT.
    bifurcation_water_surface_elevation = (
        tl.load(river_depth_ptr + catchment_cell, mask=mask, other=0.0)
        + (catchment_elevation
           - tl.load(river_height_ptr + catchment_height_idx, mask=mask, other=0.0))
    )
    bifurcation_water_surface_elevation_downstream = (
        tl.load(river_depth_ptr + downstream_cell, mask=mask, other=0.0)
        + (downstream_elevation
           - tl.load(river_height_ptr + downstream_height_idx, mask=mask, other=0.0))
    )
    max_bifurcation_water_surface_elevation = tl.maximum(bifurcation_water_surface_elevation, bifurcation_water_surface_elevation_downstream)

    bifurcation_protected_water_surface_elevation = tl.where(
        tl.load(is_levee_ptr + bifurcation_catchment_idx, mask=mask, other=False),
        tl.minimum(
            catchment_elevation
            + tl.load(protected_depth_ptr + catchment_cell, mask=mask, other=0.0),
            bifurcation_water_surface_elevation,
        ),
        bifurcation_water_surface_elevation,
    )
    bifurcation_protected_water_surface_elevation_downstream = tl.where(
        tl.load(is_levee_ptr + bifurcation_downstream_idx, mask=mask, other=False),
        tl.minimum(
            downstream_elevation
            + tl.load(protected_depth_ptr + downstream_cell, mask=mask, other=0.0),
            bifurcation_water_surface_elevation_downstream,
        ),
        bifurcation_water_surface_elevation_downstream,
    )
    max_bifurcation_protected_water_surface_elevation = tl.maximum(bifurcation_protected_water_surface_elevation, bifurcation_protected_water_surface_elevation_downstream)

    # Bifurcation slope (clamped similarly to flood slope)
    bifurcation_slope = hm.divide(bifurcation_water_surface_elevation - bifurcation_water_surface_elevation_downstream, bifurcation_length)
    bifurcation_slope = hm.clamp(bifurcation_slope, -ROUTING_SLOPE_LIMIT, ROUTING_SLOPE_LIMIT)

    # Storage change limiter calculation
    bifurcation_total_storage = hm.to_compute(
        tl.load(river_storage_ptr + catchment_cell, mask=mask, other=0.0)
        + tl.load(flood_storage_ptr + catchment_cell, mask=mask, other=0.0)
        + tl.load(protected_storage_ptr + catchment_cell, mask=mask, other=0.0),
        bifurcation_length,
    )
    bifurcation_total_storage_downstream = hm.to_compute(
        tl.load(river_storage_ptr + downstream_cell, mask=mask, other=0.0)
        + tl.load(flood_storage_ptr + downstream_cell, mask=mask, other=0.0)
        + tl.load(protected_storage_ptr + downstream_cell, mask=mask, other=0.0),
        bifurcation_length,
    )
    sum_bifurcation_outflow = tl.zeros_like(bifurcation_length)

    # Base offsets for level-dependent arrays
    manning_base = (member_offset_levels if batched_bifurcation_manning else 0)
    width_base = (member_offset_levels if batched_bifurcation_width else 0)
    elevation_base = (member_offset_levels if batched_bifurcation_elevation else 0)

    for level in tl.static_range(num_bifurcation_levels):
        
        level_idx = offs * num_bifurcation_levels + level
        bifurcation_manning = tl.load(bifurcation_manning_ptr + manning_base + level_idx, mask=mask, other=0.0)
        bifurcation_cross_section_depth = tl.load(bifurcation_cross_section_depth_ptr + member_offset_levels + level_idx, mask=mask, other=0.0)
        bifurcation_elevation = tl.load(bifurcation_elevation_ptr + elevation_base + level_idx, mask=mask, other=0.0)
        
        # Calculate bifurcation cross-section depth
        # Level 0: River channel (use river WSE)
        # Level > 0: Overland (use protected WSE)
        
        if level == 0:
            current_max_wse = max_bifurcation_water_surface_elevation
        else:
            current_max_wse = max_bifurcation_protected_water_surface_elevation

        updated_bifurcation_cross_section_depth = tl.maximum(current_max_wse - bifurcation_elevation, 0.0)
        
        # Calculate semi-implicit flow depth for bifurcation
        # Level 0: Semi-implicit
        # Level > 0: Explicit (no semi-implicit)
        
        if level == 0:
            semi_implicit_depth = hm.sqrt(
                updated_bifurcation_cross_section_depth
                * bifurcation_cross_section_depth,
            )
            bifurcation_semi_implicit_flow_depth = tl.where(
                semi_implicit_depth <= 0.0,
                updated_bifurcation_cross_section_depth,
                semi_implicit_depth,
            )
        else:
            bifurcation_semi_implicit_flow_depth = updated_bifurcation_cross_section_depth
        
        bifurcation_width = tl.load(bifurcation_width_ptr + width_base + level_idx, mask=mask, other=0.0)
        bifurcation_outflow = tl.load(bifurcation_outflow_ptr + member_offset_levels + level_idx, mask=mask, other=0.0)

        unit_bifurcation_outflow = hm.divide(bifurcation_outflow, bifurcation_width)

        numerator = bifurcation_width * (
            unit_bifurcation_outflow + gravity * time_step 
            * bifurcation_semi_implicit_flow_depth * bifurcation_slope
        )
        denominator = 1.0 + gravity * time_step * (bifurcation_manning * bifurcation_manning) * tl.abs(unit_bifurcation_outflow) \
                    * hm.divide(1.0, bifurcation_semi_implicit_flow_depth * bifurcation_semi_implicit_flow_depth * hm.cbrt(bifurcation_semi_implicit_flow_depth))
        
        updated_bifurcation_outflow = hm.divide(numerator, denominator)
        bifurcation_condition = bifurcation_semi_implicit_flow_depth > hm.constant(1e-5, bifurcation_semi_implicit_flow_depth)
        updated_bifurcation_outflow = tl.where(bifurcation_condition, updated_bifurcation_outflow, 0.0)
        sum_bifurcation_outflow += updated_bifurcation_outflow
        tl.store(bifurcation_cross_section_depth_ptr + member_offset_levels + level_idx, updated_bifurcation_cross_section_depth, mask=mask)
        tl.store(bifurcation_outflow_ptr + member_offset_levels + level_idx, updated_bifurcation_outflow, mask=mask)
    # CaMa-Flood LEVEE_OPT_PTHOUT limits a path only when its flow sum is non-zero.
    limit_rate = tl.where(
        sum_bifurcation_outflow != 0.0,
        hm.at_most(hm.divide(BACKFLOW_STORAGE_FRACTION * tl.minimum(bifurcation_total_storage, bifurcation_total_storage_downstream), tl.abs(sum_bifurcation_outflow) * time_step), 1.0),
        hm.constant(1.0, sum_bifurcation_outflow),
    )
    sum_bifurcation_outflow *= limit_rate
    for level in tl.static_range(num_bifurcation_levels):
        level_idx = offs * num_bifurcation_levels + level
        updated_bifurcation_outflow = tl.load(bifurcation_outflow_ptr + member_offset_levels + level_idx, mask=mask)
        updated_bifurcation_outflow *= limit_rate
        tl.store(bifurcation_outflow_ptr + member_offset_levels + level_idx, updated_bifurcation_outflow, mask=mask)

    # P2STOOUT flows, multiplied by the step in compute_inflow.
    pos_flow = tl.maximum(sum_bifurcation_outflow, 0.0)
    neg_flow = tl.minimum(sum_bifurcation_outflow, 0.0)
    tl.atomic_add(outgoing_storage_ptr + member_offset_catchments + bifurcation_catchment_idx, pos_flow, mask=mask, sem="relaxed")
    tl.atomic_add(outgoing_storage_ptr + member_offset_catchments + bifurcation_downstream_idx, -neg_flow, mask=mask, sem="relaxed")
