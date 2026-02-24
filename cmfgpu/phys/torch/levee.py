# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Pure PyTorch implementation of levee-aware flood stage kernels."""

import torch


def compute_levee_stage_kernel(
    levee_catchment_idx: torch.Tensor,
    river_storage: torch.Tensor,
    flood_storage: torch.Tensor,
    protected_storage: torch.Tensor,
    river_depth: torch.Tensor,
    flood_depth: torch.Tensor,
    protected_depth: torch.Tensor,
    river_height: torch.Tensor,
    flood_depth_table: torch.Tensor,
    catchment_area: torch.Tensor,
    river_width: torch.Tensor,
    river_length: torch.Tensor,
    levee_base_height: torch.Tensor,
    levee_crown_height: torch.Tensor,
    levee_fraction: torch.Tensor,
    flood_fraction: torch.Tensor,
    num_levees: int,
    num_flood_levels: int,
    BLOCK_SIZE: int = 128,
) -> None:
    NL = num_levees
    lci = levee_catchment_idx.long()

    rl = river_length[lci]
    rw = river_width[lci]
    rh = river_height[lci]
    ca = catchment_area[lci]

    l_crown = levee_crown_height
    l_frac = levee_fraction
    l_base_h = levee_base_height

    riv_sto_curr = river_storage[lci]
    fld_sto_curr = flood_storage[lci]
    fld_depth_curr = flood_depth[lci]

    total_sto = riv_sto_curr + fld_sto_curr

    riv_max_sto = rl * rw * rh
    dwth_inc = (ca / rl) / num_flood_levels
    levee_dist = l_frac * (ca / rl)

    # Scan for levee_base_storage, levee_fill_storage, and Case 3 search B
    s_curr = riv_max_sto.clone()
    dhgt_pre = torch.zeros_like(s_curr)
    dwth_pre = rw.clone()

    levee_base_sto = riv_max_sto.clone()
    levee_fill_sto = riv_max_sto.clone()

    found_base = s_curr.new_zeros(NL, dtype=torch.bool)
    found_fill = s_curr.new_zeros(NL, dtype=torch.bool)

    ilev = (l_frac * num_flood_levels).to(torch.int32)

    dsto_fil_B = torch.zeros_like(s_curr)
    dwth_fil_B = torch.zeros_like(s_curr)
    ddph_fil_B = torch.zeros_like(s_curr)
    gradient_B = torch.zeros_like(s_curr)
    found_B = s_curr.new_zeros(NL, dtype=torch.bool)

    # table is (total_catchments, num_flood_levels)
    for i in range(num_flood_levels):
        depth_val = flood_depth_table[lci, i]

        dhgt_seg = torch.clamp(depth_val - dhgt_pre, min=1e-6)
        dwth_mid = dwth_pre + 0.5 * dwth_inc
        dsto_seg = rl * dwth_mid * dhgt_seg
        s_next = s_curr + dsto_seg
        gradient = dhgt_seg / dwth_inc

        # Check base
        cond_base = (l_base_h > dhgt_pre) & (l_base_h <= depth_val)
        ratio_base = (l_base_h - dhgt_pre) / dhgt_seg
        dsto_base_p = rl * (dwth_pre + 0.5 * ratio_base * dwth_inc) * (ratio_base * dhgt_seg)
        s_base_cand = s_curr + dsto_base_p
        levee_base_sto = torch.where(cond_base & ~found_base, s_base_cand, levee_base_sto)
        found_base = found_base | cond_base

        # Check fill
        cond_fill = (l_crown > dhgt_pre) & (l_crown <= depth_val)
        ratio_fill = (l_crown - dhgt_pre) / dhgt_seg
        dsto_fill_p = rl * (dwth_pre + 0.5 * ratio_fill * dwth_inc) * (ratio_fill * dhgt_seg)
        s_fill_cand = s_curr + dsto_fill_p
        levee_fill_sto = torch.where(cond_fill & ~found_fill, s_fill_cand, levee_fill_sto)
        found_fill = found_fill | cond_fill

        # Case 3 search
        dhgt_dif_loop = l_crown - l_base_h
        s_top_loop = levee_base_sto + (levee_dist + rw) * dhgt_dif_loop * rl

        dsto_add_wedge = (levee_dist + rw) * (l_crown - depth_val) * rl
        threshold = s_next + dsto_add_wedge

        cond_check = (i >= ilev) & ~found_B
        cond_found = cond_check & (total_sto < threshold)

        current_lb = torch.where(i == ilev, s_top_loop, dsto_fil_B)
        dsto_fil_B = torch.where(cond_check & ~cond_found, threshold, current_lb)
        dwth_fil_B = torch.where(cond_check & ~cond_found, dwth_inc * (i + 1) - levee_dist, dwth_fil_B)
        ddph_fil_B = torch.where(cond_check & ~cond_found, depth_val - l_base_h, ddph_fil_B)
        gradient_B = torch.where(cond_found, gradient, gradient_B)
        found_B = found_B | cond_found

        s_curr = s_next
        dhgt_pre = depth_val
        dwth_pre = dwth_pre + dwth_inc

    # Out of bounds
    s_base_extra = s_curr + rl * dwth_pre * (l_base_h - dhgt_pre)
    levee_base_sto = torch.where(found_base, levee_base_sto,
                                 torch.where(l_base_h > dhgt_pre, s_base_extra, riv_max_sto))
    s_fill_extra = s_curr + rl * dwth_pre * (l_crown - dhgt_pre)
    levee_fill_sto = torch.where(found_fill, levee_fill_sto,
                                 torch.where(l_crown > dhgt_pre, s_fill_extra, riv_max_sto))

    dhgt_dif = l_crown - l_base_h
    s_top = levee_base_sto + (levee_dist + rw) * dhgt_dif * rl

    # Case determination
    is_case4 = total_sto >= levee_fill_sto
    is_case3 = ~is_case4 & (total_sto >= s_top)
    is_case2 = ~is_case4 & ~is_case3 & (total_sto >= levee_base_sto)

    # Case 2
    dsto_add_c2 = total_sto - levee_base_sto
    dwth_add_c2 = levee_dist + rw
    f_dph_c2 = l_base_h + dsto_add_c2 / dwth_add_c2 / rl
    r_sto_c2 = riv_max_sto + rl * rw * f_dph_c2
    r_dph_c2 = r_sto_c2 / rl / rw
    f_sto_c2 = torch.clamp(total_sto - r_sto_c2, min=0.0)
    f_frc_c2 = l_frac

    # Case 3
    dsto_add_B = total_sto - dsto_fil_B
    term_B = dwth_fil_B ** 2 + 2.0 * dsto_add_B / rl / (gradient_B + 1e-9)
    dwth_add_B = -dwth_fil_B + torch.sqrt(torch.clamp(term_B, min=0.0))
    ddph_add_B = dwth_add_B * gradient_B
    p_dph_B_found = l_base_h + ddph_fil_B + ddph_add_B
    f_frc_B_found = (dwth_fil_B + levee_dist) / (dwth_inc * num_flood_levels)

    ddph_add_B_extra = dsto_add_B / (dwth_fil_B * rl + 1e-9)
    p_dph_B_extra = l_base_h + ddph_fil_B + ddph_add_B_extra
    f_frc_B_extra = torch.ones_like(f_frc_B_found)

    p_dph_B = torch.where(found_B, p_dph_B_found, p_dph_B_extra)
    f_frc_B = torch.where(found_B, f_frc_B_found, f_frc_B_extra)

    f_dph_c3 = l_crown
    r_sto_c3 = riv_max_sto + rl * rw * f_dph_c3
    r_dph_c3 = r_sto_c3 / rl / rw
    f_sto_c3 = torch.clamp(s_top - r_sto_c3, min=0.0)
    p_sto_c3 = torch.clamp(total_sto - r_sto_c3 - f_sto_c3, min=0.0)
    p_dph_c3 = p_dph_B
    f_frc_c3 = torch.clamp(f_frc_B, 0.0, 1.0)

    # Case 4
    f_dph_c4 = fld_depth_curr
    r_sto_c4 = riv_sto_curr
    dsto_add_c4 = (f_dph_c4 - l_crown) * (levee_dist + rw) * rl
    f_sto_c4 = torch.clamp(s_top + dsto_add_c4 - r_sto_c4, min=0.0)
    p_sto_c4 = torch.clamp(total_sto - r_sto_c4 - f_sto_c4, min=0.0)
    p_dph_c4 = f_dph_c4

    # Select results
    r_dph_curr = river_depth[lci]

    r_sto = torch.where(is_case2, r_sto_c2,
            torch.where(is_case3, r_sto_c3,
            torch.where(is_case4, r_sto_c4, riv_sto_curr)))
    f_sto = torch.where(is_case2, f_sto_c2,
            torch.where(is_case3, f_sto_c3,
            torch.where(is_case4, f_sto_c4, fld_sto_curr)))
    p_sto = torch.where(is_case2, 0.0,
            torch.where(is_case3, p_sto_c3,
            torch.where(is_case4, p_sto_c4, 0.0)))
    r_dph = torch.where(is_case2, r_dph_c2,
            torch.where(is_case3, r_dph_c3, r_dph_curr))
    f_dph = torch.where(is_case2, f_dph_c2,
            torch.where(is_case3, f_dph_c3, fld_depth_curr))
    p_dph = torch.where(is_case2, 0.0,
            torch.where(is_case3, p_dph_c3,
            torch.where(is_case4, p_dph_c4, 0.0)))
    f_frc = torch.where(is_case2, f_frc_c2,
            torch.where(is_case3, f_frc_c3, flood_fraction[lci]))

    # Store results (scatter to catchment indices)
    river_storage[lci] = r_sto
    flood_storage[lci] = f_sto
    protected_storage[lci] = p_sto
    river_depth[lci] = r_dph
    flood_depth[lci] = f_dph
    protected_depth[lci] = p_dph
    flood_fraction[lci] = f_frc


def compute_levee_stage_log_kernel(
    levee_catchment_idx: torch.Tensor,
    river_storage: torch.Tensor,
    flood_storage: torch.Tensor,
    protected_storage: torch.Tensor,
    river_depth: torch.Tensor,
    flood_depth: torch.Tensor,
    protected_depth: torch.Tensor,
    river_height: torch.Tensor,
    flood_depth_table: torch.Tensor,
    catchment_area: torch.Tensor,
    river_width: torch.Tensor,
    river_length: torch.Tensor,
    levee_base_height: torch.Tensor,
    levee_crown_height: torch.Tensor,
    levee_fraction: torch.Tensor,
    flood_fraction: torch.Tensor,
    total_storage_stage_sum: torch.Tensor,
    river_storage_sum: torch.Tensor,
    flood_storage_sum: torch.Tensor,
    flood_area_sum: torch.Tensor,
    total_stage_error_sum: torch.Tensor,
    current_step: int,
    num_levees: int,
    num_flood_levels: int,
    BLOCK_SIZE: int = 128,
) -> None:
    NL = num_levees
    lci = levee_catchment_idx.long()
    ca = catchment_area[lci]

    total_pre = river_storage[lci] + flood_storage[lci]

    compute_levee_stage_kernel(
        levee_catchment_idx=levee_catchment_idx,
        river_storage=river_storage,
        flood_storage=flood_storage,
        protected_storage=protected_storage,
        river_depth=river_depth,
        flood_depth=flood_depth,
        protected_depth=protected_depth,
        river_height=river_height,
        flood_depth_table=flood_depth_table,
        catchment_area=catchment_area,
        river_width=river_width,
        river_length=river_length,
        levee_base_height=levee_base_height,
        levee_crown_height=levee_crown_height,
        levee_fraction=levee_fraction,
        flood_fraction=flood_fraction,
        num_levees=num_levees,
        num_flood_levels=num_flood_levels,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    r_sto = river_storage[lci]
    f_sto = flood_storage[lci]
    p_sto = protected_storage[lci]
    ff = flood_fraction[lci]

    total_new = r_sto + f_sto + p_sto
    total_storage_stage_sum[current_step] += total_new.sum() * 1e-9
    river_storage_sum[current_step] += r_sto.sum() * 1e-9
    flood_storage_sum[current_step] += f_sto.sum() * 1e-9
    flood_area_sum[current_step] += (ff * ca).sum() * 1e-9
    total_stage_error_sum[current_step] += (total_new - total_pre).sum() * 1e-9


def compute_levee_bifurcation_outflow_kernel(
    bifurcation_catchment_idx: torch.Tensor,
    bifurcation_downstream_idx: torch.Tensor,
    bifurcation_manning: torch.Tensor,
    bifurcation_outflow: torch.Tensor,
    bifurcation_width: torch.Tensor,
    bifurcation_length: torch.Tensor,
    bifurcation_elevation: torch.Tensor,
    bifurcation_cross_section_depth: torch.Tensor,
    water_surface_elevation: torch.Tensor,
    protected_water_surface_elevation: torch.Tensor,
    total_storage: torch.Tensor,
    outgoing_storage: torch.Tensor,
    gravity: float,
    time_step: float,
    num_bifurcation_paths: int,
    num_bifurcation_levels: int,
    BLOCK_SIZE: int = 128,
) -> None:
    P = num_bifurcation_paths
    L = num_bifurcation_levels

    bci = bifurcation_catchment_idx.long()
    bdi = bifurcation_downstream_idx.long()

    blen = bifurcation_length
    bwse = water_surface_elevation[bci]
    bwse_ds = water_surface_elevation[bdi]
    max_bwse = torch.maximum(bwse, bwse_ds)

    bpwse = protected_water_surface_elevation[bci]
    bpwse_ds = protected_water_surface_elevation[bdi]
    max_bpwse = torch.maximum(bpwse, bpwse_ds)

    bslope = torch.clamp((bwse - bwse_ds) / blen, -0.005, 0.005)

    b_total_sto = total_storage[bci]
    b_total_sto_ds = total_storage[bdi]

    gt = gravity * time_step
    sum_bif_out = blen.new_zeros(P)

    for level in range(L):
        bman = bifurcation_manning[:, level]
        bcs = bifurcation_cross_section_depth[:, level]
        belev = bifurcation_elevation[:, level]

        current_max_wse = max_bwse if level == 0 else max_bpwse
        upd_bcs = torch.clamp(current_max_wse - belev, min=0.0)

        if level == 0:
            b_semi = torch.maximum(
                torch.sqrt(upd_bcs * bcs),
                torch.sqrt(upd_bcs * 0.01),
            )
        else:
            b_semi = upd_bcs

        bw = bifurcation_width[:, level]
        bout = bifurcation_outflow[:, level]
        unit_bout = bout / bw

        num = bw * (unit_bout + gt * b_semi * bslope)
        den = 1.0 + gt * (bman ** 2) * torch.abs(unit_bout) * torch.pow(b_semi, -7.0 / 3.0)

        upd_bout = torch.where(b_semi > 1e-5, num / den, 0.0)
        sum_bif_out += upd_bout

        bifurcation_cross_section_depth[:, level] = upd_bcs
        bifurcation_outflow[:, level] = upd_bout

    lr = torch.clamp(
        0.05 * torch.minimum(b_total_sto, b_total_sto_ds) / (torch.abs(sum_bif_out) * time_step).clamp(min=1e-30),
        max=1.0,
    )
    sum_bif_out = sum_bif_out * lr
    for level in range(L):
        bifurcation_outflow[:, level] = bifurcation_outflow[:, level] * lr

    pos = torch.clamp(sum_bif_out, min=0.0)
    neg = torch.clamp(sum_bif_out, max=0.0)
    outgoing_storage.scatter_add_(0, bci, pos * time_step)
    outgoing_storage.scatter_add_(0, bdi, -neg * time_step)


# Batched variants not implemented
compute_levee_stage_batched_kernel = None
compute_levee_bifurcation_outflow_batched_kernel = None
