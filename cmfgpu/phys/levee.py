# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Triton kernels for levee-aware flood stage calculations."""

import triton
import triton.language as tl


@triton.jit
def compute_levee_stage_kernel(
    levee_catchment_idx_ptr,
    river_storage_ptr,
    flood_storage_ptr,
    protected_storage_ptr,
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
    flood_area_ptr,
    num_levees: tl.constexpr,
    num_flood_levels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    levee_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = levee_offs < num_levees
    
    # Load levee index
    levee_catchment_idx = tl.load(levee_catchment_idx_ptr + levee_offs, mask=mask, other=0)
    
    # Load basic parameters
    river_length = tl.load(river_length_ptr + levee_catchment_idx, mask=mask, other=1.0)
    river_width = tl.load(river_width_ptr + levee_catchment_idx, mask=mask, other=1.0)
    river_height = tl.load(river_height_ptr + levee_catchment_idx, mask=mask, other=0.0)
    catchment_area = tl.load(catchment_area_ptr + levee_catchment_idx, mask=mask, other=0.0)
    
    # Load levee parameters
    levee_crown_height = tl.load(levee_crown_height_ptr + levee_offs, mask=mask, other=0.0)
    levee_fraction = tl.load(levee_fraction_ptr + levee_offs, mask=mask, other=0.0)
    levee_base_height = tl.load(levee_base_height_ptr + levee_offs, mask=mask, other=0.0)
    # levee_base_storage calculated below
    
    # Load current state (computed by standard kernel)
    river_storage_curr = tl.load(river_storage_ptr + levee_catchment_idx, mask=mask, other=0.0)
    flood_storage_curr = tl.load(flood_storage_ptr + levee_catchment_idx, mask=mask, other=0.0)
    flood_depth_curr = tl.load(flood_depth_ptr + levee_catchment_idx, mask=mask, other=0.0)
    
    total_storage = river_storage_curr + flood_storage_curr
    
    # Derived parameters
    river_max_storage = river_length * river_width * river_height
    dwth_inc = (catchment_area / river_length) / num_flood_levels
    levee_distance = levee_fraction * (catchment_area / river_length)
    
    # Calculate levee_base_storage and levee_fill_storage
    s_curr = river_max_storage
    dhgt_pre = 0.0
    dwth_pre = river_width
    
    levee_base_storage = river_max_storage
    levee_fill_storage = river_max_storage
    
    found_base = 0
    found_fill = 0
    
    # --- Logic for Case 3 (Search B) ---
    ilev = (levee_fraction * num_flood_levels).to(tl.int32)
    
    dsto_fil_B = 0.0
    dwth_fil_B = 0.0
    ddph_fil_B = 0.0
    gradient_B = 0.0
    found_B = 0
    
    for i in tl.static_range(num_flood_levels):
        depth_val = tl.load(flood_depth_table_ptr + levee_catchment_idx * num_flood_levels + i, mask=mask, other=0.0)
        
        dhgt_seg = depth_val - dhgt_pre
        dhgt_seg = tl.maximum(dhgt_seg, 1e-6)
        
        dwth_mid = dwth_pre + 0.5 * dwth_inc
        dsto_seg = river_length * dwth_mid * dhgt_seg
        s_next = s_curr + dsto_seg
        gradient = dhgt_seg / dwth_inc
        
        # Check Base
        cond_base = (levee_base_height > dhgt_pre) & (levee_base_height <= depth_val)
        ratio_base = (levee_base_height - dhgt_pre) / dhgt_seg
        dsto_base_partial = river_length * (dwth_pre + 0.5 * ratio_base * dwth_inc) * (ratio_base * dhgt_seg)
        s_base_cand = s_curr + dsto_base_partial
        levee_base_storage = tl.where(cond_base, s_base_cand, levee_base_storage)
        found_base = found_base | cond_base
        
        # Check Fill
        cond_fill = (levee_crown_height > dhgt_pre) & (levee_crown_height <= depth_val)
        ratio_fill = (levee_crown_height - dhgt_pre) / dhgt_seg
        dsto_fill_partial = river_length * (dwth_pre + 0.5 * ratio_fill * dwth_inc) * (ratio_fill * dhgt_seg)
        s_fill_cand = s_curr + dsto_fill_partial
        levee_fill_storage = tl.where(cond_fill, s_fill_cand, levee_fill_storage)
        found_fill = found_fill | cond_fill
        
        # --- Case 3 Search Logic ---
        # Calculate temporary s_top for current iteration
        dhgt_dif_loop = levee_crown_height - levee_base_height
        s_top_loop = levee_base_storage + (levee_distance + river_width) * dhgt_dif_loop * river_length
        
        dsto_add_wedge = (levee_distance + river_width) * (levee_crown_height - depth_val) * river_length
        threshold = s_next + dsto_add_wedge
        
        cond_check = (i >= ilev) & (found_B == 0)
        cond_found = cond_check & (total_storage < threshold)
        
        # Determine lower bound for this step
        current_lower_bound = tl.where(i == ilev, s_top_loop, dsto_fil_B)
        
        # Update dsto_fil_B: if found, keep lower bound; if not found, update to current threshold (new lower bound)
        dsto_fil_B = tl.where(cond_check & (cond_found == 0), threshold, current_lower_bound)
        
        dwth_fil_B_next = dwth_inc * (i + 1) - levee_distance
        dwth_fil_B = tl.where(cond_check & (cond_found == 0), dwth_fil_B_next, dwth_fil_B)
        
        ddph_fil_B_next = depth_val - levee_base_height
        ddph_fil_B = tl.where(cond_check & (cond_found == 0), ddph_fil_B_next, ddph_fil_B)
        
        gradient_B = tl.where(cond_found != 0, gradient, gradient_B)
        found_B = found_B | cond_found
        
        s_curr = s_next
        dhgt_pre = depth_val
        dwth_pre += dwth_inc
        
    # Handle out of bounds
    s_base_extra = s_curr + river_length * dwth_pre * (levee_base_height - dhgt_pre)
    levee_base_storage = tl.where(found_base != 0, levee_base_storage, tl.where(levee_base_height > dhgt_pre, s_base_extra, river_max_storage))
    
    s_fill_extra = s_curr + river_length * dwth_pre * (levee_crown_height - dhgt_pre)
    levee_fill_storage = tl.where(found_fill != 0, levee_fill_storage, tl.where(levee_crown_height > dhgt_pre, s_fill_extra, river_max_storage))

    # Calculate s_top
    dhgt_dif = levee_crown_height - levee_base_height
    s_top = levee_base_storage + (levee_distance + river_width) * dhgt_dif * river_length
    
    # Determine Case
    is_case4 = total_storage >= levee_fill_storage
    is_case3 = (is_case4 == 0) & (total_storage >= s_top)
    is_case2 = (is_case4 == 0) & (is_case3 == 0) & (total_storage >= levee_base_storage)
    
    # --- Logic for Case 2 ---
    dsto_add_c2 = total_storage - levee_base_storage
    dwth_add_c2 = levee_distance + river_width
    f_dph_c2 = levee_base_height + dsto_add_c2 / dwth_add_c2 / river_length
    r_sto_c2 = river_max_storage + river_length * river_width * f_dph_c2
    r_dph_c2 = r_sto_c2 / river_length / river_width
    f_sto_c2 = tl.maximum(total_storage - r_sto_c2, 0.0)
    f_frc_c2 = levee_fraction
    
    # --- Logic for Case 3 (Search B Results) ---
    dsto_add_B = total_storage - dsto_fil_B
    term_B = dwth_fil_B * dwth_fil_B + 2.0 * dsto_add_B / river_length / (gradient_B + 1e-9)
    dwth_add_B = -dwth_fil_B + tl.sqrt(tl.maximum(term_B, 0.0))
    ddph_add_B = dwth_add_B * gradient_B
    p_dph_B_found = levee_base_height + ddph_fil_B + ddph_add_B
    f_frc_B_found = (dwth_fil_B + levee_distance) / (dwth_inc * num_flood_levels)
    
    # If not found (extrapolate)
    ddph_add_B_extra = dsto_add_B / (dwth_fil_B * river_length + 1e-9)
    p_dph_B_extra = levee_base_height + ddph_fil_B + ddph_add_B_extra
    f_frc_B_extra = 1.0
    
    p_dph_B = tl.where(found_B != 0, p_dph_B_found, p_dph_B_extra)
    f_frc_B = tl.where(found_B != 0, f_frc_B_found, f_frc_B_extra)
    
    f_dph_c3 = levee_crown_height
    r_sto_c3 = river_max_storage + river_length * river_width * f_dph_c3
    r_dph_c3 = r_sto_c3 / river_length / river_width
    f_sto_c3 = tl.maximum(s_top - r_sto_c3, 0.0)
    p_sto_c3 = tl.maximum(total_storage - r_sto_c3 - f_sto_c3, 0.0)
    p_dph_c3 = p_dph_B
    f_frc_c3 = tl.clamp(f_frc_B, 0.0, 1.0)
    
    # --- Logic for Case 4 ---
    f_dph_c4 = flood_depth_curr
    r_sto_c4 = river_storage_curr
    
    dsto_add_wedge_c4 = (f_dph_c4 - levee_crown_height) * (levee_distance + river_width) * river_length
    f_sto_c4 = tl.maximum(s_top + dsto_add_wedge_c4 - r_sto_c4, 0.0)
    p_sto_c4 = tl.maximum(total_storage - r_sto_c4 - f_sto_c4, 0.0)
    p_dph_c4 = f_dph_c4
    
    # --- Select Results ---
    r_dph_curr = tl.load(river_depth_ptr + levee_catchment_idx, mask=mask, other=0.0)
    
    r_sto = tl.where(is_case2, r_sto_c2,
             tl.where(is_case3, r_sto_c3,
              tl.where(is_case4, r_sto_c4, river_storage_curr)))
              
    f_sto = tl.where(is_case2, f_sto_c2,
             tl.where(is_case3, f_sto_c3,
              tl.where(is_case4, f_sto_c4, flood_storage_curr)))
              
    p_sto = tl.where(is_case2, 0.0,
             tl.where(is_case3, p_sto_c3,
              tl.where(is_case4, p_sto_c4, 0.0)))
              
    r_dph = tl.where(is_case2, r_dph_c2,
             tl.where(is_case3, r_dph_c3, r_dph_curr))
             
    f_dph = tl.where(is_case2, f_dph_c2,
             tl.where(is_case3, f_dph_c3, flood_depth_curr))
             
    p_dph = tl.where(is_case2, 0.0,
             tl.where(is_case3, p_dph_c3,
              tl.where(is_case4, p_dph_c4, 0.0)))
    
    # If levee_fraction == 1.0, protected_depth is flood_depth
    p_dph = tl.where(levee_fraction == 1.0, f_dph, p_dph)
              
    f_frc = tl.where(is_case2, f_frc_c2,
             tl.where(is_case3, f_frc_c3, 
              tl.load(flood_fraction_ptr + levee_catchment_idx, mask=mask, other=0.0)))

    # Store results
    tl.store(river_storage_ptr + levee_catchment_idx, r_sto, mask=mask)
    tl.store(flood_storage_ptr + levee_catchment_idx, f_sto, mask=mask)
    tl.store(protected_storage_ptr + levee_offs, p_sto, mask=mask)
    tl.store(river_depth_ptr + levee_catchment_idx, r_dph, mask=mask)
    tl.store(flood_depth_ptr + levee_catchment_idx, f_dph, mask=mask)
    tl.store(protected_depth_ptr + levee_catchment_idx, p_dph, mask=mask)
    tl.store(flood_fraction_ptr + levee_catchment_idx, f_frc, mask=mask)
    tl.store(flood_area_ptr + levee_catchment_idx, f_frc * catchment_area, mask=mask)


@triton.jit
def compute_levee_stage_log_kernel(
    levee_catchment_idx_ptr,
    river_storage_ptr,
    flood_storage_ptr,
    protected_storage_ptr,
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
    flood_area_ptr,
    total_storage_stage_sum_ptr,
    river_storage_sum_ptr,
    flood_storage_sum_ptr,
    flood_area_sum_ptr,
    total_stage_error_sum_ptr,
    current_step,
    num_levees: tl.constexpr,
    num_flood_levels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    levee_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = levee_offs < num_levees
    
    # Load levee index
    levee_catchment_idx = tl.load(levee_catchment_idx_ptr + levee_offs, mask=mask, other=0)
    
    # Load basic parameters
    river_length = tl.load(river_length_ptr + levee_catchment_idx, mask=mask, other=1.0)
    river_width = tl.load(river_width_ptr + levee_catchment_idx, mask=mask, other=1.0)
    river_height = tl.load(river_height_ptr + levee_catchment_idx, mask=mask, other=0.0)
    catchment_area = tl.load(catchment_area_ptr + levee_catchment_idx, mask=mask, other=0.0)
    
    # Load levee parameters
    levee_crown_height = tl.load(levee_crown_height_ptr + levee_offs, mask=mask, other=0.0)
    levee_fraction = tl.load(levee_fraction_ptr + levee_offs, mask=mask, other=0.0)
    levee_base_height = tl.load(levee_base_height_ptr + levee_offs, mask=mask, other=0.0)
    # levee_base_storage calculated below
    
    # Load current state
    river_storage_curr = tl.load(river_storage_ptr + levee_catchment_idx, mask=mask, other=0.0)
    flood_storage_curr = tl.load(flood_storage_ptr + levee_catchment_idx, mask=mask, other=0.0)
    flood_depth_curr = tl.load(flood_depth_ptr + levee_catchment_idx, mask=mask, other=0.0)
    
    total_storage = river_storage_curr + flood_storage_curr
    
    # Derived parameters
    river_max_storage = river_length * river_width * river_height
    dwth_inc = (catchment_area / river_length) / num_flood_levels
    levee_distance = levee_fraction * (catchment_area / river_length)
    
    # Calculate levee_base_storage and levee_fill_storage
    s_curr = river_max_storage
    dhgt_pre = 0.0
    dwth_pre = river_width
    
    levee_base_storage = river_max_storage
    levee_fill_storage = river_max_storage
    
    found_base = 0
    found_fill = 0
    
    # --- Logic for Case 3 (Search B) ---
    ilev = (levee_fraction * num_flood_levels).to(tl.int32)
    
    dsto_fil_B = 0.0
    dwth_fil_B = 0.0
    ddph_fil_B = 0.0
    gradient_B = 0.0
    found_B = 0
    
    for i in tl.static_range(num_flood_levels):
        depth_val = tl.load(flood_depth_table_ptr + levee_catchment_idx * num_flood_levels + i, mask=mask, other=0.0)
        
        dhgt_seg = depth_val - dhgt_pre
        dhgt_seg = tl.maximum(dhgt_seg, 1e-6)
        
        dwth_mid = dwth_pre + 0.5 * dwth_inc
        dsto_seg = river_length * dwth_mid * dhgt_seg
        s_next = s_curr + dsto_seg
        gradient = dhgt_seg / dwth_inc
        
        # Check Base
        cond_base = (levee_base_height > dhgt_pre) & (levee_base_height <= depth_val)
        ratio_base = (levee_base_height - dhgt_pre) / dhgt_seg
        dsto_base_partial = river_length * (dwth_pre + 0.5 * ratio_base * dwth_inc) * (ratio_base * dhgt_seg)
        s_base_cand = s_curr + dsto_base_partial
        levee_base_storage = tl.where(cond_base, s_base_cand, levee_base_storage)
        found_base = found_base | cond_base
        
        # Check Fill
        cond_fill = (levee_crown_height > dhgt_pre) & (levee_crown_height <= depth_val)
        ratio_fill = (levee_crown_height - dhgt_pre) / dhgt_seg
        dsto_fill_partial = river_length * (dwth_pre + 0.5 * ratio_fill * dwth_inc) * (ratio_fill * dhgt_seg)
        s_fill_cand = s_curr + dsto_fill_partial
        levee_fill_storage = tl.where(cond_fill, s_fill_cand, levee_fill_storage)
        found_fill = found_fill | cond_fill
        
        # --- Case 3 Search Logic ---
        # Calculate temporary s_top for current iteration
        dhgt_dif_loop = levee_crown_height - levee_base_height
        s_top_loop = levee_base_storage + (levee_distance + river_width) * dhgt_dif_loop * river_length
        
        dsto_add_wedge = (levee_distance + river_width) * (levee_crown_height - depth_val) * river_length
        threshold = s_next + dsto_add_wedge
        
        cond_check = (i >= ilev) & (found_B == 0)
        cond_found = cond_check & (total_storage < threshold)
        
        # Determine lower bound for this step
        current_lower_bound = tl.where(i == ilev, s_top_loop, dsto_fil_B)
        
        # Update dsto_fil_B: if found, keep lower bound; if not found, update to current threshold (new lower bound)
        dsto_fil_B = tl.where(cond_check & (cond_found == 0), threshold, current_lower_bound)
        
        dwth_fil_B_next = dwth_inc * (i + 1) - levee_distance
        dwth_fil_B = tl.where(cond_check & (cond_found == 0), dwth_fil_B_next, dwth_fil_B)
        
        ddph_fil_B_next = depth_val - levee_base_height
        ddph_fil_B = tl.where(cond_check & (cond_found == 0), ddph_fil_B_next, ddph_fil_B)
        
        gradient_B = tl.where(cond_found != 0, gradient, gradient_B)
        found_B = found_B | cond_found
        
        s_curr = s_next
        dhgt_pre = depth_val
        dwth_pre += dwth_inc
        
    # Handle out of bounds
    s_base_extra = s_curr + river_length * dwth_pre * (levee_base_height - dhgt_pre)
    levee_base_storage = tl.where(found_base != 0, levee_base_storage, tl.where(levee_base_height > dhgt_pre, s_base_extra, river_max_storage))
    
    s_fill_extra = s_curr + river_length * dwth_pre * (levee_crown_height - dhgt_pre)
    levee_fill_storage = tl.where(found_fill != 0, levee_fill_storage, tl.where(levee_crown_height > dhgt_pre, s_fill_extra, river_max_storage))

    # Calculate s_top
    dhgt_dif = levee_crown_height - levee_base_height
    s_top = levee_base_storage + (levee_distance + river_width) * dhgt_dif * river_length
    
    # Determine Case
    is_case4 = total_storage >= levee_fill_storage
    is_case3 = (is_case4 == 0) & (total_storage >= s_top)
    is_case2 = (is_case4 == 0) & (is_case3 == 0) & (total_storage >= levee_base_storage)
    
    # --- Logic for Case 2 ---
    dsto_add_c2 = total_storage - levee_base_storage
    dwth_add_c2 = levee_distance + river_width
    f_dph_c2 = levee_base_height + dsto_add_c2 / dwth_add_c2 / river_length
    r_sto_c2 = river_max_storage + river_length * river_width * f_dph_c2
    r_dph_c2 = r_sto_c2 / river_length / river_width
    f_sto_c2 = tl.maximum(total_storage - r_sto_c2, 0.0)
    f_frc_c2 = levee_fraction
    
    # --- Logic for Case 3 (Search B Results) ---
    dsto_add_B = total_storage - dsto_fil_B
    term_B = dwth_fil_B * dwth_fil_B + 2.0 * dsto_add_B / river_length / (gradient_B + 1e-9)
    dwth_add_B = -dwth_fil_B + tl.sqrt(tl.maximum(term_B, 0.0))
    ddph_add_B = dwth_add_B * gradient_B
    p_dph_B_found = levee_base_height + ddph_fil_B + ddph_add_B
    f_frc_B_found = (dwth_fil_B + levee_distance) / (dwth_inc * num_flood_levels)
    
    # If not found (extrapolate)
    ddph_add_B_extra = dsto_add_B / (dwth_fil_B * river_length + 1e-9)
    p_dph_B_extra = levee_base_height + ddph_fil_B + ddph_add_B_extra
    f_frc_B_extra = 1.0
    
    p_dph_B = tl.where(found_B != 0, p_dph_B_found, p_dph_B_extra)
    f_frc_B = tl.where(found_B != 0, f_frc_B_found, f_frc_B_extra)
    
    f_dph_c3 = levee_crown_height
    r_sto_c3 = river_max_storage + river_length * river_width * f_dph_c3
    r_dph_c3 = r_sto_c3 / river_length / river_width
    f_sto_c3 = tl.maximum(s_top - r_sto_c3, 0.0)
    p_sto_c3 = tl.maximum(total_storage - r_sto_c3 - f_sto_c3, 0.0)
    p_dph_c3 = p_dph_B
    f_frc_c3 = tl.clamp(f_frc_B, 0.0, 1.0)
    
    # --- Logic for Case 4 ---
    f_dph_c4 = flood_depth_curr
    r_sto_c4 = river_storage_curr
    dsto_add_wedge_c4 = (f_dph_c4 - levee_crown_height) * (levee_distance + river_width) * river_length
    f_sto_c4 = tl.maximum(s_top + dsto_add_wedge_c4 - r_sto_c4, 0.0)
    p_sto_c4 = tl.maximum(total_storage - r_sto_c4 - f_sto_c4, 0.0)
    p_dph_c4 = f_dph_c4
    
    # --- Select Results ---
    r_dph_curr = tl.load(river_depth_ptr + levee_catchment_idx, mask=mask, other=0.0)
    
    r_sto = tl.where(is_case2, r_sto_c2,
             tl.where(is_case3, r_sto_c3,
              tl.where(is_case4, r_sto_c4, river_storage_curr)))
              
    f_sto = tl.where(is_case2, f_sto_c2,
             tl.where(is_case3, f_sto_c3,
              tl.where(is_case4, f_sto_c4, flood_storage_curr)))
              
    p_sto = tl.where(is_case2, 0.0,
             tl.where(is_case3, p_sto_c3,
              tl.where(is_case4, p_sto_c4, 0.0)))
              
    r_dph = tl.where(is_case2, r_dph_c2,
             tl.where(is_case3, r_dph_c3, r_dph_curr))
             
    f_dph = tl.where(is_case2, f_dph_c2,
             tl.where(is_case3, f_dph_c3, flood_depth_curr))
             
    p_dph = tl.where(is_case2, 0.0,
             tl.where(is_case3, p_dph_c3,
              tl.where(is_case4, p_dph_c4, 0.0)))
    
    # If levee_fraction == 1.0, protected_depth is flood_depth
    p_dph = tl.where(levee_fraction == 1.0, f_dph, p_dph)
              
    f_frc = tl.where(is_case2, f_frc_c2,
             tl.where(is_case3, f_frc_c3, 
              tl.load(flood_fraction_ptr + levee_catchment_idx, mask=mask, other=0.0)))

    # Log variables
    total_storage_stage_new = r_sto + f_sto + p_sto
    tl.atomic_add(total_storage_stage_sum_ptr + current_step, tl.sum(total_storage_stage_new) * 1e-9)
    tl.atomic_add(river_storage_sum_ptr + current_step, tl.sum(r_sto) * 1e-9)
    tl.atomic_add(flood_storage_sum_ptr + current_step, tl.sum(f_sto) * 1e-9)
    tl.atomic_add(flood_area_sum_ptr + current_step, tl.sum(f_frc * catchment_area) * 1e-9)
    tl.atomic_add(total_stage_error_sum_ptr + current_step, tl.sum(total_storage_stage_new - total_storage) * 1e-9)

    # Store results
    tl.store(river_storage_ptr + levee_catchment_idx, r_sto, mask=mask)
    tl.store(flood_storage_ptr + levee_catchment_idx, f_sto, mask=mask)
    tl.store(protected_storage_ptr + levee_offs, p_sto, mask=mask)
    tl.store(river_depth_ptr + levee_catchment_idx, r_dph, mask=mask)
    tl.store(flood_depth_ptr + levee_catchment_idx, f_dph, mask=mask)
    tl.store(protected_depth_ptr + levee_catchment_idx, p_dph, mask=mask)
    tl.store(flood_fraction_ptr + levee_catchment_idx, f_frc, mask=mask)
    tl.store(flood_area_ptr + levee_catchment_idx, f_frc * catchment_area, mask=mask)


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
    water_surface_elevation_ptr,                # *f32: River depth
    protected_water_surface_elevation_ptr,      # *f32: Protected water surface elevation
    total_storage_ptr,                          # *f32: Total storage (in/out)
    outgoing_storage_ptr,                       # *f32: Outgoing storage (in/out)
    gravity: tl.constexpr,                      # f32: Gravity constant
    time_step,                                  # f32: Time step
    num_bifurcation_paths: tl.constexpr,        # Total number of bifurcation paths
    num_bifurcation_levels: tl.constexpr,       # int: Number of bifurcation levels    
    BLOCK_SIZE: tl.constexpr                    # Block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_bifurcation_paths
    
    # Load indices
    bifurcation_catchment_idx = tl.load(bifurcation_catchment_idx_ptr + offs, mask=mask, other=0)
    bifurcation_downstream_idx = tl.load(bifurcation_downstream_idx_ptr + offs, mask=mask, other=0)
    
    # Load bifurcation properties
    
    bifurcation_length = tl.load(bifurcation_length_ptr + offs, mask=mask, other=0.0)
    
    # Load river properties for catchment and downstream
    bifurcation_water_surface_elevation = tl.load(water_surface_elevation_ptr + bifurcation_catchment_idx, mask=mask, other=0.0)
    bifurcation_water_surface_elevation_downstream = tl.load(water_surface_elevation_ptr + bifurcation_downstream_idx, mask=mask, other=0.0)
    max_bifurcation_water_surface_elevation = tl.maximum(bifurcation_water_surface_elevation, bifurcation_water_surface_elevation_downstream)

    # Load protected properties
    bifurcation_protected_water_surface_elevation = tl.load(protected_water_surface_elevation_ptr + bifurcation_catchment_idx, mask=mask, other=0.0)
    bifurcation_protected_water_surface_elevation_downstream = tl.load(protected_water_surface_elevation_ptr + bifurcation_downstream_idx, mask=mask, other=0.0)
    max_bifurcation_protected_water_surface_elevation = tl.maximum(bifurcation_protected_water_surface_elevation, bifurcation_protected_water_surface_elevation_downstream)

    # Bifurcation slope (clamped similarly to flood slope)
    bifurcation_slope = (bifurcation_water_surface_elevation - bifurcation_water_surface_elevation_downstream) / bifurcation_length
    bifurcation_slope = tl.clamp(bifurcation_slope, -0.005, 0.005)

    # Storage change limiter calculation
    bifurcation_total_storage = tl.load(total_storage_ptr + bifurcation_catchment_idx, mask=mask, other=0.0)
    bifurcation_total_storage_downstream = tl.load(total_storage_ptr + bifurcation_downstream_idx, mask=mask, other=0.0)
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
            bifurcation_semi_implicit_flow_depth = tl.maximum(
                tl.sqrt(updated_bifurcation_cross_section_depth * bifurcation_cross_section_depth),
                tl.sqrt(updated_bifurcation_cross_section_depth * 0.01)
            )
        else:
            bifurcation_semi_implicit_flow_depth = updated_bifurcation_cross_section_depth
        
        bifurcation_width = tl.load(bifurcation_width_ptr + level_idx, mask=mask, other=0.0)
        bifurcation_outflow = tl.load(bifurcation_outflow_ptr + level_idx, mask=mask, other=0.0)

        unit_bifurcation_outflow = bifurcation_outflow / bifurcation_width

        numerator = bifurcation_width * (
            unit_bifurcation_outflow + gravity * time_step 
            * bifurcation_semi_implicit_flow_depth * bifurcation_slope
        )
        denominator = 1.0 + gravity * time_step * (bifurcation_manning * bifurcation_manning) * tl.abs(unit_bifurcation_outflow) \
                    * tl.exp((-7.0/3.0) * tl.log(bifurcation_semi_implicit_flow_depth))
        
        updated_bifurcation_outflow = numerator / denominator
        bifurcation_condition = (bifurcation_semi_implicit_flow_depth > 1e-5)
        updated_bifurcation_outflow = tl.where(bifurcation_condition, updated_bifurcation_outflow, 0.0)
        sum_bifurcation_outflow += updated_bifurcation_outflow
        tl.store(bifurcation_cross_section_depth_ptr + level_idx, updated_bifurcation_cross_section_depth, mask=mask)
        tl.store(bifurcation_outflow_ptr + level_idx, updated_bifurcation_outflow, mask=mask)
    limit_rate = tl.minimum(0.05 * tl.minimum(bifurcation_total_storage, bifurcation_total_storage_downstream) / (tl.abs(sum_bifurcation_outflow) * time_step), 1.0)
    sum_bifurcation_outflow *= limit_rate
    for level in tl.static_range(num_bifurcation_levels):
        level_idx = offs * num_bifurcation_levels + level
        updated_bifurcation_outflow = tl.load(bifurcation_outflow_ptr + level_idx, mask=mask)
        updated_bifurcation_outflow *= limit_rate
        tl.store(bifurcation_outflow_ptr + level_idx, updated_bifurcation_outflow, mask=mask)

    pos_flow = tl.maximum(sum_bifurcation_outflow, 0.0)
    neg_flow = tl.minimum(sum_bifurcation_outflow, 0.0)
    tl.atomic_add(outgoing_storage_ptr + bifurcation_catchment_idx, pos_flow * time_step, mask=mask)
    tl.atomic_add(outgoing_storage_ptr + bifurcation_downstream_idx, -neg_flow * time_step, mask=mask)
