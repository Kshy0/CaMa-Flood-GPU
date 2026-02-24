# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Pure PyTorch implementation of flood-stage / storage kernels."""

import torch


def compute_flood_stage_kernel(
    river_inflow: torch.Tensor,
    flood_inflow: torch.Tensor,
    river_outflow: torch.Tensor,
    flood_outflow: torch.Tensor,
    global_bifurcation_outflow: torch.Tensor,
    runoff: torch.Tensor,
    time_step: float,
    outgoing_storage: torch.Tensor,
    river_storage: torch.Tensor,
    flood_storage: torch.Tensor,
    protected_storage: torch.Tensor,
    river_depth: torch.Tensor,
    flood_depth: torch.Tensor,
    protected_depth: torch.Tensor,
    flood_fraction: torch.Tensor,
    river_height: torch.Tensor,
    flood_depth_table: torch.Tensor,
    catchment_area: torch.Tensor,
    river_width: torch.Tensor,
    river_length: torch.Tensor,
    num_catchments: int,
    num_flood_levels: int,
    BLOCK_SIZE: int = 128,
) -> None:
    N = num_catchments

    # ---- 1. Storage update ----
    riv_sto_upd = river_storage + (river_inflow - river_outflow) * time_step
    fld_sto_upd = flood_storage + torch.where(riv_sto_upd < 0.0, riv_sto_upd, 0.0) + (flood_inflow - flood_outflow - global_bifurcation_outflow) * time_step
    riv_sto_upd = torch.clamp(riv_sto_upd, min=0.0)
    riv_sto_upd = torch.where(
        fld_sto_upd < 0.0,
        torch.clamp(riv_sto_upd + fld_sto_upd, min=0.0),
        riv_sto_upd,
    )
    fld_sto_upd = torch.clamp(fld_sto_upd, min=0.0)
    total_sto = torch.clamp(riv_sto_upd + fld_sto_upd + protected_storage + runoff * time_step, min=0.0)

    # ---- 2. Flood stage computation ----
    rh = river_height
    ca = catchment_area
    rw = river_width
    rl = river_length

    riv_max_sto = rl * rw * rh
    w_inc = ca / rl / num_flood_levels

    # Reshape table: (N, num_flood_levels)
    table = flood_depth_table.reshape(N, num_flood_levels)

    # Vectorized scan over flood levels
    level = (total_sto > riv_max_sto).to(torch.int32) - 1

    S_accum = riv_max_sto
    prev_H = total_sto.new_zeros(N)
    prev_W = rw

    prev_total_sto = riv_max_sto
    prev_flood_depth = total_sto.new_zeros(N)
    next_flood_depth = total_sto.new_zeros(N)

    for i in range(num_flood_levels):
        H_curr = table[:, i]
        W_curr = rw + (i + 1) * w_inc
        dS = rl * 0.5 * (prev_W + W_curr) * (H_curr - prev_H)
        S_curr = S_accum + dS

        next_flood_depth = torch.where(level == i, H_curr, next_flood_depth)

        is_above = total_sto > S_curr
        level = level + is_above.to(level.dtype)
        prev_total_sto = torch.where(is_above, S_curr, prev_total_sto)
        prev_flood_depth = torch.where(is_above, H_curr, prev_flood_depth)

        S_accum = S_curr
        prev_H = H_curr
        prev_W = W_curr

    no_flood = level < 0
    level = torch.clamp(level, min=0)

    prev_total_width = rw + level.float() * w_inc

    flood_grad = torch.where(level == num_flood_levels, 0.0, (next_flood_depth - prev_flood_depth) / w_inc)

    # Guard against division by zero when flood_grad == 0
    # (happens for no-flood catchments or flat table entries).
    # Use flood_grad=1 as a safe denominator; the result is masked
    # by no_flood / level==num_flood_levels later anyway.
    flood_grad_safe = torch.where(flood_grad == 0.0, 1.0, flood_grad)
    sqrt_arg = prev_total_width * prev_total_width + 2.0 * (total_sto - prev_total_sto) / (flood_grad_safe * rl)
    diff_width = torch.sqrt(sqrt_arg) - prev_total_width
    fld_depth_mid = prev_flood_depth + diff_width * flood_grad
    fld_depth_top = prev_flood_depth + (total_sto - prev_total_sto) / (prev_total_width * rl)

    fld_d = torch.where(no_flood, 0.0, torch.where(level == num_flood_levels, fld_depth_top, fld_depth_mid))

    riv_sto_final = torch.where(
        no_flood,
        total_sto,
        torch.minimum(riv_max_sto + rl * rw * fld_d, total_sto),
    )
    riv_d = riv_sto_final / (rl * rw)

    fld_frac_mid = torch.clamp((prev_total_width + diff_width - rw) * rl / ca, 0.0, 1.0)
    fld_frac = torch.where(no_flood, 0.0, torch.where(level == num_flood_levels, 1.0, fld_frac_mid))

    fld_sto_final = torch.clamp(total_sto - riv_sto_final, min=0.0)

    # Write outputs
    outgoing_storage.zero_()
    river_storage.copy_(riv_sto_final)
    flood_storage.copy_(fld_sto_final)
    protected_storage.zero_()
    river_depth.copy_(riv_d)
    flood_depth.copy_(fld_d)
    protected_depth.copy_(fld_d)
    flood_fraction.copy_(fld_frac)


def compute_flood_stage_log_kernel(
    river_inflow: torch.Tensor,
    flood_inflow: torch.Tensor,
    river_outflow: torch.Tensor,
    flood_outflow: torch.Tensor,
    global_bifurcation_outflow: torch.Tensor,
    runoff: torch.Tensor,
    time_step: float,
    outgoing_storage: torch.Tensor,
    river_storage: torch.Tensor,
    flood_storage: torch.Tensor,
    protected_storage: torch.Tensor,
    river_depth: torch.Tensor,
    flood_depth: torch.Tensor,
    protected_depth: torch.Tensor,
    flood_fraction: torch.Tensor,
    river_height: torch.Tensor,
    flood_depth_table: torch.Tensor,
    catchment_area: torch.Tensor,
    river_width: torch.Tensor,
    river_length: torch.Tensor,
    is_levee: torch.Tensor,
    total_storage_pre_sum: torch.Tensor,
    total_storage_next_sum: torch.Tensor,
    total_storage_new_sum: torch.Tensor,
    total_inflow_sum: torch.Tensor,
    total_outflow_sum: torch.Tensor,
    total_storage_stage_sum: torch.Tensor,
    river_storage_sum: torch.Tensor,
    flood_storage_sum: torch.Tensor,
    flood_area_sum: torch.Tensor,
    total_inflow_error_sum: torch.Tensor,
    total_stage_error_sum: torch.Tensor,
    current_step: int,
    num_catchments: int,
    num_flood_levels: int,
    log_buffer_size: int = 1000,
    BLOCK_SIZE: int = 128,
) -> None:
    """Flood stage kernel with logging. Delegates to base kernel + log."""
    N = num_catchments
    non_levee = ~is_levee

    # Pre-logging (all tensor ops, no .item() – keeps the graph intact)
    total_pre = river_storage + flood_storage + protected_storage
    total_storage_pre_sum[current_step] += total_pre.sum() * 1e-9

    riv_in = river_inflow.clone()
    fld_in = flood_inflow.clone()
    riv_out = river_outflow.clone()
    fld_out = flood_outflow.clone()
    bif_out = global_bifurcation_outflow.clone()
    ro = runoff.clone()

    # Run the core computation
    compute_flood_stage_kernel(
        river_inflow=river_inflow,
        flood_inflow=flood_inflow,
        river_outflow=river_outflow,
        flood_outflow=flood_outflow,
        global_bifurcation_outflow=global_bifurcation_outflow,
        runoff=runoff,
        time_step=time_step,
        outgoing_storage=outgoing_storage,
        river_storage=river_storage,
        flood_storage=flood_storage,
        protected_storage=protected_storage,
        river_depth=river_depth,
        flood_depth=flood_depth,
        protected_depth=protected_depth,
        flood_fraction=flood_fraction,
        river_height=river_height,
        flood_depth_table=flood_depth_table,
        catchment_area=catchment_area,
        river_width=river_width,
        river_length=river_length,
        num_catchments=num_catchments,
        num_flood_levels=num_flood_levels,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Post-logging (pure tensor ops – no .item())
    ca = catchment_area
    riv_sto = river_storage
    fld_sto = flood_storage
    ff = flood_fraction
    total_next = riv_sto + fld_sto
    total_inflow_sum[current_step] += torch.where(non_levee, (riv_in + fld_in) * time_step, 0.0).sum() * 1e-9
    total_outflow_sum[current_step] += torch.where(non_levee, (riv_out + fld_out) * time_step, 0.0).sum() * 1e-9
    total_storage_next_sum[current_step] += torch.where(non_levee, total_next, 0.0).sum() * 1e-9
    total_storage_new_sum[current_step] += torch.where(non_levee, total_next, 0.0).sum() * 1e-9
    total_storage_stage_sum[current_step] += total_next.sum() * 1e-9
    river_storage_sum[current_step] += torch.where(non_levee, riv_sto, 0.0).sum() * 1e-9
    flood_storage_sum[current_step] += torch.where(non_levee, fld_sto, 0.0).sum() * 1e-9
    flood_area = ff * ca
    flood_area_sum[current_step] += torch.where(non_levee, flood_area, 0.0).sum() * 1e-9
    total_inflow_error_sum[current_step] += torch.where(non_levee,
        total_pre - total_next + (riv_in + fld_in + ro - riv_out - fld_out - bif_out) * time_step, 0.0).sum() * 1e-9
    total_stage_error_sum[current_step] += torch.where(non_levee,
        (total_next - (riv_sto + fld_sto)) * 1e-9, 0.0).sum()


# Batched variant not implemented
compute_flood_stage_batched_kernel = None
