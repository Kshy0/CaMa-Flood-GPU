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
    levee_base_storage_ptr,
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
    levee_base_storage = tl.load(levee_base_storage_ptr + levee_offs, mask=mask, other=0.0)
    
    # Load current state
    river_storage_curr = tl.load(river_storage_ptr + levee_catchment_idx, mask=mask, other=0.0)
    flood_storage_curr = tl.load(flood_storage_ptr + levee_catchment_idx, mask=mask, other=0.0)
    total_storage = river_storage_curr + flood_storage_curr
    
    # Derived parameters
    river_max_storage = river_length * river_width * river_height
    dwth_inc = (catchment_area / river_length) / num_flood_levels
    levee_distance = levee_fraction * (catchment_area / river_length)
    
    # Calculate s_top
    dhgt_dif = levee_crown_height - levee_base_height
    s_top = levee_base_storage + (levee_distance + river_width) * dhgt_dif * river_length
    
    # --- 1. Calculate s_fill ---
    s_fill = river_max_storage
    dsto_fil = river_max_storage
    dwth_fil = river_width
    dhgt_pre = 0.0
    fill_complete = 0
    
    for i in tl.static_range(num_flood_levels):
        depth_val = tl.load(flood_depth_table_ptr + levee_catchment_idx * num_flood_levels + i, mask=mask, other=0.0)
        gradient = (depth_val - dhgt_pre) / dwth_inc
        
        # Conditions
        cond_full = (fill_complete == 0) & (levee_crown_height > depth_val)
        cond_part = (fill_complete == 0) & (levee_crown_height <= depth_val)
        
        # Update for full
        dsto_add_full = river_length * (dwth_fil + 0.5 * dwth_inc) * (depth_val - dhgt_pre)
        
        # Update for part
        dhgt_now = levee_crown_height - dhgt_pre
        dwth_add_part = tl.where(gradient > 1e-6, dhgt_now / gradient, 0.0)
        dsto_add_part = (dwth_add_part * 0.5 + dwth_fil) * dhgt_now * river_length
        
        s_fill = s_fill + tl.where(cond_full, dsto_add_full, tl.where(cond_part, dsto_add_part, 0.0))
        
        dsto_fil = tl.where(cond_full, s_fill, dsto_fil)
        dwth_fil = tl.where(cond_full, dwth_fil + dwth_inc, dwth_fil)
        dhgt_pre = tl.where(cond_full, depth_val, dhgt_pre)
        
        fill_complete = fill_complete | cond_part

    # Extrapolation if not complete
    cond_extra = (fill_complete == 0)
    dhgt_now = levee_crown_height - dhgt_pre
    dsto_add_extra = dwth_fil * dhgt_now * river_length
    s_fill = s_fill + tl.where(cond_extra, dsto_add_extra, 0.0)
    
    # --- 2. Determine Cases ---
    is_case0 = total_storage <= river_max_storage
    is_case1 = (total_storage > river_max_storage) & (total_storage < levee_base_storage)
    is_case2 = (total_storage >= levee_base_storage) & (total_storage < s_top)
    is_case3 = (total_storage >= s_top) & (total_storage < s_fill)
    is_case4 = (total_storage >= s_fill)
    
    # --- 3. Search A (Standard Profile) - Used by Case 1 & 4 ---
    dsto_fil_A = river_max_storage
    dwth_fil_A = river_width
    dhgt_fil_A = 0.0
    gradient_A = 0.0
    found_A = 0
    
    s_prof_curr = river_max_storage
    dhgt_pre_prof = 0.0
    dwth_pre_prof = river_width
    
    for i in tl.static_range(num_flood_levels):
        depth_val = tl.load(flood_depth_table_ptr + levee_catchment_idx * num_flood_levels + i, mask=mask, other=0.0)
        gradient = (depth_val - dhgt_pre_prof) / dwth_inc
        
        dsto_seg = river_length * (dwth_pre_prof + 0.5 * dwth_inc) * (depth_val - dhgt_pre_prof)
        s_prof_next = s_prof_curr + dsto_seg
        
        cond_found = (found_A == 0) & (total_storage <= s_prof_next)
        
        dsto_fil_A = tl.where(cond_found, s_prof_curr, dsto_fil_A)
        dwth_fil_A = tl.where(cond_found, dwth_pre_prof, dwth_fil_A)
        dhgt_fil_A = tl.where(cond_found, dhgt_pre_prof, dhgt_fil_A)
        gradient_A = tl.where(cond_found, gradient, gradient_A)
        
        found_A = found_A | cond_found
        
        s_prof_curr = s_prof_next
        dhgt_pre_prof = depth_val
        dwth_pre_prof += dwth_inc
        
    # Calc results for A
    dsto_add_A = total_storage - dsto_fil_A
    
    # If found
    term_A = dwth_fil_A * dwth_fil_A + 2.0 * dsto_add_A / river_length / gradient_A
    dwth_add_A = -dwth_fil_A + tl.sqrt(tl.maximum(term_A, 0.0))
    f_dph_A_found = dhgt_fil_A + gradient_A * dwth_add_A
    f_frc_A_found = (-river_width + dwth_fil_A + dwth_add_A) / (dwth_inc * num_flood_levels)
    
    # If not found (extrapolate)
    dwth_add_A_extra = 0.0
    f_dph_A_extra = dhgt_fil_A + dsto_add_A / dwth_fil_A / river_length
    f_frc_A_extra = (-river_width + dwth_fil_A) / (dwth_inc * num_flood_levels)
    
    f_dph_A = tl.where(found_A != 0, f_dph_A_found, f_dph_A_extra)
    f_frc_A = tl.where(found_A != 0, f_frc_A_found, f_frc_A_extra)
    
    # --- 4. Search B (Case 3) ---
    ilev = (levee_fraction * num_flood_levels).to(tl.int32)
    
    dsto_fil_B = s_top
    dwth_fil_B = 0.0
    ddph_fil_B = 0.0
    gradient_B = 0.0
    found_B = 0
    
    s_prof_curr = river_max_storage
    dhgt_pre_prof = 0.0
    dwth_pre_prof = river_width
    
    for i in tl.static_range(num_flood_levels):
        depth_val = tl.load(flood_depth_table_ptr + levee_catchment_idx * num_flood_levels + i, mask=mask, other=0.0)
        gradient = (depth_val - dhgt_pre_prof) / dwth_inc
        
        dsto_seg = river_length * (dwth_pre_prof + 0.5 * dwth_inc) * (depth_val - dhgt_pre_prof)
        s_prof_next = s_prof_curr + dsto_seg
        
        # Check logic for Case 3
        dsto_add_wedge = (levee_distance + river_width) * (levee_crown_height - depth_val) * river_length
        threshold = s_prof_next + dsto_add_wedge
        
        cond_check = (i >= ilev) & (found_B == 0)
        cond_found = cond_check & (total_storage < threshold)
        
        # Update for next step
        dsto_fil_B_next = threshold
        dwth_fil_B_next = dwth_inc * (i + 1) - levee_distance
        ddph_fil_B_next = depth_val - levee_base_height
        
        # If found, we don't update dsto_fil_B anymore (it holds the correct base)
        dsto_fil_B = tl.where(cond_check & (cond_found == 0), dsto_fil_B_next, dsto_fil_B)
        dwth_fil_B = tl.where(cond_check & (cond_found == 0), dwth_fil_B_next, dwth_fil_B)
        ddph_fil_B = tl.where(cond_check & (cond_found == 0), ddph_fil_B_next, ddph_fil_B)
        
        gradient_B = tl.where(cond_found, gradient, gradient_B)
        found_B = found_B | cond_found
        
        s_prof_curr = s_prof_next
        dhgt_pre_prof = depth_val
        dwth_pre_prof += dwth_inc

    # Calc results for B
    dsto_add_B = total_storage - dsto_fil_B
    term_B = dwth_fil_B * dwth_fil_B + 2.0 * dsto_add_B / river_length / gradient_B
    dwth_add_B = -dwth_fil_B + tl.sqrt(tl.maximum(term_B, 0.0))
    ddph_add_B = dwth_add_B * gradient_B
    p_dph_B = levee_base_height + ddph_fil_B + ddph_add_B
    f_frc_B = (dwth_fil_B + levee_distance) / (dwth_inc * num_flood_levels)
    
    # --- 5. Assemble Final Results ---
    
    # Case 0
    r_sto_c0 = total_storage
    r_dph_c0 = total_storage / river_length / river_width
    
    # Case 1
    r_sto_c1 = river_max_storage + river_length * river_width * f_dph_A
    r_dph_c1 = r_sto_c1 / river_length / river_width
    f_sto_c1 = tl.maximum(total_storage - r_sto_c1, 0.0)
    f_frc_c1 = tl.clamp(f_frc_A, 0.0, 1.0)
    
    # Case 2
    dsto_add_c2 = total_storage - levee_base_storage
    dwth_add_c2 = levee_distance + river_width
    f_dph_c2 = levee_base_height + dsto_add_c2 / dwth_add_c2 / river_length
    r_sto_c2 = river_max_storage + river_length * river_width * f_dph_c2
    r_dph_c2 = r_sto_c2 / river_length / river_width
    f_sto_c2 = tl.maximum(total_storage - r_sto_c2, 0.0)
    f_frc_c2 = levee_fraction
    
    # Case 3
    f_dph_c3 = levee_crown_height
    r_sto_c3 = river_max_storage + river_length * river_width * f_dph_c3
    r_dph_c3 = r_sto_c3 / river_length / river_width
    f_sto_c3 = tl.maximum(s_top - r_sto_c3, 0.0)
    p_sto_c3 = tl.maximum(total_storage - r_sto_c3 - f_sto_c3, 0.0)
    p_dph_c3 = p_dph_B
    f_frc_c3 = tl.clamp(f_frc_B, 0.0, 1.0)
    
    # Case 4
    r_sto_c4 = river_max_storage + river_length * river_width * f_dph_A
    r_dph_c4 = r_sto_c4 / river_length / river_width
    dsto_add_wedge_c4 = (f_dph_A - levee_crown_height) * (levee_distance + river_width) * river_length
    f_sto_c4 = tl.maximum(s_top + dsto_add_wedge_c4 - r_sto_c4, 0.0)
    p_sto_c4 = tl.maximum(total_storage - r_sto_c4 - f_sto_c4, 0.0)
    p_dph_c4 = f_dph_A
    f_frc_c4 = tl.clamp(f_frc_A, 0.0, 1.0)
    
    # Select
    r_sto = tl.where(is_case0, r_sto_c0,
             tl.where(is_case1, r_sto_c1,
              tl.where(is_case2, r_sto_c2,
               tl.where(is_case3, r_sto_c3, r_sto_c4))))
               
    f_sto = tl.where(is_case0, 0.0,
             tl.where(is_case1, f_sto_c1,
              tl.where(is_case2, f_sto_c2,
               tl.where(is_case3, f_sto_c3, f_sto_c4))))
               
    p_sto = tl.where(is_case0, 0.0,
             tl.where(is_case1, 0.0,
              tl.where(is_case2, 0.0,
               tl.where(is_case3, p_sto_c3, p_sto_c4))))
               
    r_dph = tl.where(is_case0, r_dph_c0,
             tl.where(is_case1, r_dph_c1,
              tl.where(is_case2, r_dph_c2,
               tl.where(is_case3, r_dph_c3, r_dph_c4))))
               
    f_dph = tl.where(is_case0, 0.0,
             tl.where(is_case1, f_dph_A, # f_dph_c1 is same as f_dph_A
              tl.where(is_case2, f_dph_c2,
               tl.where(is_case3, f_dph_c3, f_dph_A)))) # f_dph_c4 is f_dph_A
               
    p_dph = tl.where(is_case0, 0.0,
             tl.where(is_case1, 0.0,
              tl.where(is_case2, 0.0,
               tl.where(is_case3, p_dph_c3, p_dph_c4))))
               
    f_frc = tl.where(is_case0, 0.0,
             tl.where(is_case1, f_frc_c1,
              tl.where(is_case2, f_frc_c2,
               tl.where(is_case3, f_frc_c3, f_frc_c4))))

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
    levee_base_storage_ptr,
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
    levee_base_storage = tl.load(levee_base_storage_ptr + levee_offs, mask=mask, other=0.0)
    
    # Load current state
    river_storage_curr = tl.load(river_storage_ptr + levee_catchment_idx, mask=mask, other=0.0)
    flood_storage_curr = tl.load(flood_storage_ptr + levee_catchment_idx, mask=mask, other=0.0)
    total_storage = river_storage_curr + flood_storage_curr
    
    # Derived parameters
    river_max_storage = river_length * river_width * river_height
    dwth_inc = (catchment_area / river_length) / num_flood_levels
    levee_distance = levee_fraction * (catchment_area / river_length)
    
    # Calculate s_top
    dhgt_dif = levee_crown_height - levee_base_height
    s_top = levee_base_storage + (levee_distance + river_width) * dhgt_dif * river_length
    
    # --- 1. Calculate s_fill ---
    s_fill = river_max_storage
    dsto_fil = river_max_storage
    dwth_fil = river_width
    dhgt_pre = 0.0
    fill_complete = 0
    
    for i in tl.static_range(num_flood_levels):
        depth_val = tl.load(flood_depth_table_ptr + levee_catchment_idx * num_flood_levels + i, mask=mask, other=0.0)
        gradient = (depth_val - dhgt_pre) / dwth_inc
        
        # Conditions
        cond_full = (fill_complete == 0) & (levee_crown_height > depth_val)
        cond_part = (fill_complete == 0) & (levee_crown_height <= depth_val)
        
        # Update for full
        dsto_add_full = river_length * (dwth_fil + 0.5 * dwth_inc) * (depth_val - dhgt_pre)
        
        # Update for part
        dhgt_now = levee_crown_height - dhgt_pre
        dwth_add_part = tl.where(gradient > 1e-6, dhgt_now / gradient, 0.0)
        dsto_add_part = (dwth_add_part * 0.5 + dwth_fil) * dhgt_now * river_length
        
        s_fill = s_fill + tl.where(cond_full, dsto_add_full, tl.where(cond_part, dsto_add_part, 0.0))
        
        dsto_fil = tl.where(cond_full, s_fill, dsto_fil)
        dwth_fil = tl.where(cond_full, dwth_fil + dwth_inc, dwth_fil)
        dhgt_pre = tl.where(cond_full, depth_val, dhgt_pre)
        
        fill_complete = fill_complete | cond_part

    # Extrapolation if not complete
    cond_extra = (fill_complete == 0)
    dhgt_now = levee_crown_height - dhgt_pre
    dsto_add_extra = dwth_fil * dhgt_now * river_length
    s_fill = s_fill + tl.where(cond_extra, dsto_add_extra, 0.0)
    
    # --- 2. Determine Cases ---
    is_case0 = total_storage <= river_max_storage
    is_case1 = (total_storage > river_max_storage) & (total_storage < levee_base_storage)
    is_case2 = (total_storage >= levee_base_storage) & (total_storage < s_top)
    is_case3 = (total_storage >= s_top) & (total_storage < s_fill)
    is_case4 = (total_storage >= s_fill)
    
    # --- 3. Search A (Standard Profile) - Used by Case 1 & 4 ---
    dsto_fil_A = river_max_storage
    dwth_fil_A = river_width
    dhgt_fil_A = 0.0
    gradient_A = 0.0
    found_A = 0
    
    s_prof_curr = river_max_storage
    dhgt_pre_prof = 0.0
    dwth_pre_prof = river_width
    
    for i in tl.static_range(num_flood_levels):
        depth_val = tl.load(flood_depth_table_ptr + levee_catchment_idx * num_flood_levels + i, mask=mask, other=0.0)
        gradient = (depth_val - dhgt_pre_prof) / dwth_inc
        
        dsto_seg = river_length * (dwth_pre_prof + 0.5 * dwth_inc) * (depth_val - dhgt_pre_prof)
        s_prof_next = s_prof_curr + dsto_seg
        
        cond_found = (found_A == 0) & (total_storage <= s_prof_next)
        
        dsto_fil_A = tl.where(cond_found, s_prof_curr, dsto_fil_A)
        dwth_fil_A = tl.where(cond_found, dwth_pre_prof, dwth_fil_A)
        dhgt_fil_A = tl.where(cond_found, dhgt_pre_prof, dhgt_fil_A)
        gradient_A = tl.where(cond_found, gradient, gradient_A)
        
        found_A = found_A | cond_found
        
        s_prof_curr = s_prof_next
        dhgt_pre_prof = depth_val
        dwth_pre_prof += dwth_inc
        
    # Calc results for A
    dsto_add_A = total_storage - dsto_fil_A
    
    # If found
    term_A = dwth_fil_A * dwth_fil_A + 2.0 * dsto_add_A / river_length / gradient_A
    dwth_add_A = -dwth_fil_A + tl.sqrt(tl.maximum(term_A, 0.0))
    f_dph_A_found = dhgt_fil_A + gradient_A * dwth_add_A
    f_frc_A_found = (-river_width + dwth_fil_A + dwth_add_A) / (dwth_inc * num_flood_levels)
    
    # If not found (extrapolate)
    dwth_add_A_extra = 0.0
    f_dph_A_extra = dhgt_fil_A + dsto_add_A / dwth_fil_A / river_length
    f_frc_A_extra = (-river_width + dwth_fil_A) / (dwth_inc * num_flood_levels)
    
    f_dph_A = tl.where(found_A != 0, f_dph_A_found, f_dph_A_extra)
    f_frc_A = tl.where(found_A != 0, f_frc_A_found, f_frc_A_extra)
    
    # --- 4. Search B (Case 3) ---
    ilev = (levee_fraction * num_flood_levels).to(tl.int32)
    
    dsto_fil_B = s_top
    dwth_fil_B = 0.0
    ddph_fil_B = 0.0
    gradient_B = 0.0
    found_B = 0
    
    s_prof_curr = river_max_storage
    dhgt_pre_prof = 0.0
    dwth_pre_prof = river_width
    
    for i in tl.static_range(num_flood_levels):
        depth_val = tl.load(flood_depth_table_ptr + levee_catchment_idx * num_flood_levels + i, mask=mask, other=0.0)
        gradient = (depth_val - dhgt_pre_prof) / dwth_inc
        
        dsto_seg = river_length * (dwth_pre_prof + 0.5 * dwth_inc) * (depth_val - dhgt_pre_prof)
        s_prof_next = s_prof_curr + dsto_seg
        
        # Check logic for Case 3
        dsto_add_wedge = (levee_distance + river_width) * (levee_crown_height - depth_val) * river_length
        threshold = s_prof_next + dsto_add_wedge
        
        cond_check = (i >= ilev) & (found_B == 0)
        cond_found = cond_check & (total_storage < threshold)
        
        # Update for next step
        dsto_fil_B_next = threshold
        dwth_fil_B_next = dwth_inc * (i + 1) - levee_distance
        ddph_fil_B_next = depth_val - levee_base_height
        
        # If found, we don't update dsto_fil_B anymore (it holds the correct base)
        dsto_fil_B = tl.where(cond_check & (cond_found == 0), dsto_fil_B_next, dsto_fil_B)
        dwth_fil_B = tl.where(cond_check & (cond_found == 0), dwth_fil_B_next, dwth_fil_B)
        ddph_fil_B = tl.where(cond_check & (cond_found == 0), ddph_fil_B_next, ddph_fil_B)
        
        gradient_B = tl.where(cond_found, gradient, gradient_B)
        found_B = found_B | cond_found
        
        s_prof_curr = s_prof_next
        dhgt_pre_prof = depth_val
        dwth_pre_prof += dwth_inc

    # Calc results for B
    dsto_add_B = total_storage - dsto_fil_B
    term_B = dwth_fil_B * dwth_fil_B + 2.0 * dsto_add_B / river_length / gradient_B
    dwth_add_B = -dwth_fil_B + tl.sqrt(tl.maximum(term_B, 0.0))
    ddph_add_B = dwth_add_B * gradient_B
    p_dph_B = levee_base_height + ddph_fil_B + ddph_add_B
    f_frc_B = (dwth_fil_B + levee_distance) / (dwth_inc * num_flood_levels)
    
    # --- 5. Assemble Final Results ---
    
    # Case 0
    r_sto_c0 = total_storage
    r_dph_c0 = total_storage / river_length / river_width
    
    # Case 1
    r_sto_c1 = river_max_storage + river_length * river_width * f_dph_A
    r_dph_c1 = r_sto_c1 / river_length / river_width
    f_sto_c1 = tl.maximum(total_storage - r_sto_c1, 0.0)
    f_frc_c1 = tl.clamp(f_frc_A, 0.0, 1.0)
    
    # Case 2
    dsto_add_c2 = total_storage - levee_base_storage
    dwth_add_c2 = levee_distance + river_width
    f_dph_c2 = levee_base_height + dsto_add_c2 / dwth_add_c2 / river_length
    r_sto_c2 = river_max_storage + river_length * river_width * f_dph_c2
    r_dph_c2 = r_sto_c2 / river_length / river_width
    f_sto_c2 = tl.maximum(total_storage - r_sto_c2, 0.0)
    f_frc_c2 = levee_fraction
    
    # Case 3
    f_dph_c3 = levee_crown_height
    r_sto_c3 = river_max_storage + river_length * river_width * f_dph_c3
    r_dph_c3 = r_sto_c3 / river_length / river_width
    f_sto_c3 = tl.maximum(s_top - r_sto_c3, 0.0)
    p_sto_c3 = tl.maximum(total_storage - r_sto_c3 - f_sto_c3, 0.0)
    p_dph_c3 = p_dph_B
    f_frc_c3 = tl.clamp(f_frc_B, 0.0, 1.0)
    
    # Case 4
    r_sto_c4 = river_max_storage + river_length * river_width * f_dph_A
    r_dph_c4 = r_sto_c4 / river_length / river_width
    dsto_add_wedge_c4 = (f_dph_A - levee_crown_height) * (levee_distance + river_width) * river_length
    f_sto_c4 = tl.maximum(s_top + dsto_add_wedge_c4 - r_sto_c4, 0.0)
    p_sto_c4 = tl.maximum(total_storage - r_sto_c4 - f_sto_c4, 0.0)
    p_dph_c4 = f_dph_A
    f_frc_c4 = tl.clamp(f_frc_A, 0.0, 1.0)
    
    # Select
    r_sto = tl.where(is_case0, r_sto_c0,
             tl.where(is_case1, r_sto_c1,
              tl.where(is_case2, r_sto_c2,
               tl.where(is_case3, r_sto_c3, r_sto_c4))))
               
    f_sto = tl.where(is_case0, 0.0,
             tl.where(is_case1, f_sto_c1,
              tl.where(is_case2, f_sto_c2,
               tl.where(is_case3, f_sto_c3, f_sto_c4))))
               
    p_sto = tl.where(is_case0, 0.0,
             tl.where(is_case1, 0.0,
              tl.where(is_case2, 0.0,
               tl.where(is_case3, p_sto_c3, p_sto_c4))))
               
    r_dph = tl.where(is_case0, r_dph_c0,
             tl.where(is_case1, r_dph_c1,
              tl.where(is_case2, r_dph_c2,
               tl.where(is_case3, r_dph_c3, r_dph_c4))))
               
    f_dph = tl.where(is_case0, 0.0,
             tl.where(is_case1, f_dph_A, # f_dph_c1 is same as f_dph_A
              tl.where(is_case2, f_dph_c2,
               tl.where(is_case3, f_dph_c3, f_dph_A)))) # f_dph_c4 is f_dph_A
               
    p_dph = tl.where(is_case0, 0.0,
             tl.where(is_case1, 0.0,
              tl.where(is_case2, 0.0,
               tl.where(is_case3, p_dph_c3, p_dph_c4))))
               
    f_frc = tl.where(is_case0, 0.0,
             tl.where(is_case1, f_frc_c1,
              tl.where(is_case2, f_frc_c2,
               tl.where(is_case3, f_frc_c3, f_frc_c4))))

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
