# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Pure PyTorch implementation of the reservoir outflow kernel.

Mirrors the Triton kernel in ``cmfgpu.phys.triton.reservoir``:

1. Undo the main outflow kernel's outgoing_storage contribution for
   reservoir catchments.
2. Compute reservoir outflow using the Yamazaki & Funato
   four-zone dispatch rule.
3. Write back ``river_outflow``, zero ``flood_outflow``, and update
   ``outgoing_storage`` for reservoir catchments.

Compiled body is separated from ``scatter_add_`` so that
``torch.compile`` can JIT-fuse the heavy arithmetic while scatter
operations run eagerly.
"""

import torch

from cmfgpu.phys._backend import _torch_compile


def _reservoir_outflow_body(
    reservoir_catchment_idx: torch.Tensor,
    downstream_idx: torch.Tensor,
    reservoir_total_inflow: torch.Tensor,
    river_outflow: torch.Tensor,
    flood_outflow: torch.Tensor,
    river_storage: torch.Tensor,
    flood_storage: torch.Tensor,
    conservation_volume: torch.Tensor,
    emergency_volume: torch.Tensor,
    adjustment_volume: torch.Tensor,
    normal_outflow: torch.Tensor,
    adjustment_outflow: torch.Tensor,
    flood_control_outflow: torch.Tensor,
    runoff: torch.Tensor,
    total_storage: torch.Tensor,
    outgoing_storage: torch.Tensor,
    time_step: float,
    num_reservoirs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compilable body – all computation *except* ``scatter_add_``.

    Returns
    -------
    scatter_undo : Tensor, shape (R,)
        Values to ``scatter_add_`` into *outgoing_storage* at ``didx``
        for undoing the main kernel's negative-flow downstream contribution.
    scatter_new : Tensor, shape (R,)
        Values to ``scatter_add_`` into *outgoing_storage* at ``didx``
        for the new reservoir outflow's negative-flow downstream part.
    """
    cidx = reservoir_catchment_idx
    didx = downstream_idx[cidx]
    is_mouth = didx == cidx

    # ================================================================ #
    # 1. Undo main outflow kernel's outgoing_storage contribution
    # ================================================================ #
    old_riv = river_outflow[cidx]
    old_fld = flood_outflow[cidx]

    old_pos = torch.clamp(old_riv, min=0.0) + torch.clamp(old_fld, min=0.0)
    old_neg = torch.clamp(old_riv, max=0.0) + torch.clamp(old_fld, max=0.0)

    outgoing_storage[cidx] -= old_pos * time_step
    scatter_undo = torch.where(~is_mouth, old_neg * time_step, torch.zeros_like(old_neg))

    # ================================================================ #
    # 2. Compute reservoir outflow
    # ================================================================ #
    tot_sto = river_storage[cidx] + flood_storage[cidx]

    total_inflow = reservoir_total_inflow[cidx]
    reservoir_total_inflow[cidx] = 0.0

    res_inflow = total_inflow + runoff[cidx]

    res_out = torch.zeros_like(conservation_volume)

    # ---- Case 1: below conservation volume ----
    c1 = tot_sto <= conservation_volume
    res_out = torch.where(c1, normal_outflow * torch.sqrt(tot_sto / conservation_volume), res_out)

    # ---- Case 2: between conservation and adjustment volume ----
    c2 = (tot_sto > conservation_volume) & (tot_sto <= adjustment_volume)
    frac2 = (tot_sto - conservation_volume) / (adjustment_volume - conservation_volume)
    res_out = torch.where(c2, normal_outflow + frac2.pow(3) * (adjustment_outflow - normal_outflow), res_out)

    # ---- Case 3: between adjustment and emergency volume ----
    c3 = (tot_sto > adjustment_volume) & (tot_sto <= emergency_volume)
    flood_period = res_inflow >= flood_control_outflow
    frac3 = (tot_sto - adjustment_volume) / (emergency_volume - adjustment_volume)

    out_flood = normal_outflow + (
        (tot_sto - conservation_volume) / (emergency_volume - conservation_volume)
    ) * (res_inflow - normal_outflow)
    out_tmp = adjustment_outflow + frac3.pow(0.1) * (flood_control_outflow - adjustment_outflow)
    out_combined = torch.maximum(out_flood, out_tmp)

    out_nonflood = adjustment_outflow + frac3.pow(0.1) * (flood_control_outflow - adjustment_outflow)

    res_out = torch.where(c3 & flood_period, out_combined, res_out)
    res_out = torch.where(c3 & ~flood_period, out_nonflood, res_out)

    # ---- Case 4: above emergency volume ----
    c4 = tot_sto > emergency_volume
    out_emergency = torch.where(res_inflow >= flood_control_outflow, res_inflow, flood_control_outflow)
    res_out = torch.where(c4, out_emergency, res_out)

    # Clamp to [0, total_storage / time_step]
    max_out = tot_sto / time_step
    res_out = torch.clamp(res_out, min=0.0)
    res_out = torch.min(res_out, max_out)

    # ================================================================ #
    # 3. Store results
    # ================================================================ #
    river_outflow[cidx] = res_out
    flood_outflow[cidx] = 0.0
    total_storage[cidx] = tot_sto

    new_pos = torch.clamp(res_out, min=0.0)
    outgoing_storage[cidx] += new_pos * time_step

    new_neg = torch.clamp(res_out, max=0.0)
    scatter_new = torch.where(~is_mouth, -(new_neg * time_step), torch.zeros_like(new_neg))

    return scatter_undo, scatter_new


_reservoir_outflow_compiled = _torch_compile(_reservoir_outflow_body)


def compute_reservoir_outflow_kernel(
    reservoir_catchment_idx: torch.Tensor,
    downstream_idx: torch.Tensor,
    reservoir_total_inflow: torch.Tensor,
    river_outflow: torch.Tensor,
    flood_outflow: torch.Tensor,
    river_storage: torch.Tensor,
    flood_storage: torch.Tensor,
    conservation_volume: torch.Tensor,
    emergency_volume: torch.Tensor,
    adjustment_volume: torch.Tensor,
    normal_outflow: torch.Tensor,
    adjustment_outflow: torch.Tensor,
    flood_control_outflow: torch.Tensor,
    runoff: torch.Tensor,
    total_storage: torch.Tensor,
    outgoing_storage: torch.Tensor,
    time_step: float,
    num_reservoirs: int,
    BLOCK_SIZE: int = 128,
) -> None:
    """Reservoir outflow: compiled body + eager scatter."""
    scatter_undo, scatter_new = _reservoir_outflow_compiled(
        reservoir_catchment_idx=reservoir_catchment_idx,
        downstream_idx=downstream_idx,
        reservoir_total_inflow=reservoir_total_inflow,
        river_outflow=river_outflow,
        flood_outflow=flood_outflow,
        river_storage=river_storage,
        flood_storage=flood_storage,
        conservation_volume=conservation_volume,
        emergency_volume=emergency_volume,
        adjustment_volume=adjustment_volume,
        normal_outflow=normal_outflow,
        adjustment_outflow=adjustment_outflow,
        flood_control_outflow=flood_control_outflow,
        runoff=runoff,
        total_storage=total_storage,
        outgoing_storage=outgoing_storage,
        time_step=time_step,
        num_reservoirs=num_reservoirs,
    )
    didx = downstream_idx[reservoir_catchment_idx]
    outgoing_storage.scatter_add_(0, didx, scatter_undo)
    outgoing_storage.scatter_add_(0, didx, scatter_new)
