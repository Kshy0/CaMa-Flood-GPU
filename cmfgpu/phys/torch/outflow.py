# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Pure PyTorch implementation of outflow kernels, optimised for MPS.

The computation-heavy *body* of each kernel is compiled with
``torch.compile``.  ``scatter_add_`` calls (which ``torch._dynamo``
cannot trace on MPS) are executed **outside** the compiled graph so
that the bulk of the arithmetic still benefits from JIT fusion while
the scatter runs eagerly without triggering recompilation storms.
"""

import torch

from cmfgpu.phys._backend import _torch_compile


# ======================================================================
# Outflow kernel
# ======================================================================

def _compute_outflow_body(
    downstream_idx: torch.Tensor,
    river_inflow: torch.Tensor,
    river_outflow: torch.Tensor,
    river_manning: torch.Tensor,
    river_depth: torch.Tensor,
    river_width: torch.Tensor,
    river_length: torch.Tensor,
    river_height: torch.Tensor,
    river_storage: torch.Tensor,
    flood_inflow: torch.Tensor,
    flood_outflow: torch.Tensor,
    flood_manning: torch.Tensor,
    flood_depth: torch.Tensor,
    protected_depth: torch.Tensor,
    catchment_elevation: torch.Tensor,
    downstream_distance: torch.Tensor,
    flood_storage: torch.Tensor,
    protected_storage: torch.Tensor,
    river_cross_section_depth: torch.Tensor,
    flood_cross_section_depth: torch.Tensor,
    flood_cross_section_area: torch.Tensor,
    global_bifurcation_outflow: torch.Tensor,
    total_storage: torch.Tensor,
    outgoing_storage: torch.Tensor,
    water_surface_elevation: torch.Tensor,
    protected_water_surface_elevation: torch.Tensor,
    gravity: float,
    time_step: float,
    num_catchments: int,
) -> torch.Tensor:
    """Compilable body – all computation *except* ``scatter_add_``.

    Returns
    -------
    to_scatter : Tensor
        Values to ``scatter_add_`` into *outgoing_storage* keyed by
        ``downstream_idx``.  River-mouth entries are zeroed so that
        scattering the full array is equivalent to the original
        masked scatter.
    """
    N = num_catchments
    idx = torch.arange(N, device=downstream_idx.device)
    is_river_mouth = downstream_idx == idx

    # (2) Water surface elevations
    river_elevation = catchment_elevation - river_height
    wse = river_depth + river_elevation
    prot_wse = torch.minimum(catchment_elevation + protected_depth, wse)
    tot_sto = river_storage + flood_storage + protected_storage

    # Downstream
    ds = downstream_idx.long()
    river_depth_ds = river_depth[ds]
    river_height_ds = river_height[ds]
    elev_ds = catchment_elevation[ds]
    river_elev_ds = elev_ds - river_height_ds
    wse_ds = river_depth_ds + river_elev_ds

    # (3) Max wse
    max_wse = torch.maximum(wse, wse_ds)

    # River mouth fix
    wse_ds = torch.where(is_river_mouth, catchment_elevation, wse_ds)

    # (4) Slopes
    river_slope = (wse - wse_ds) / downstream_distance
    flood_slope = torch.clamp(river_slope, -0.005, 0.005)

    # (5) Cross-section depths
    upd_riv_cs_depth = max_wse - river_elevation
    riv_semi = torch.sqrt(upd_riv_cs_depth * river_cross_section_depth).clamp(min=1e-6)

    upd_fld_cs_depth = torch.clamp(max_wse - catchment_elevation, min=0.0)
    fld_semi = torch.sqrt(upd_fld_cs_depth * flood_cross_section_depth).clamp(min=1e-6)

    # (6) Flood area
    upd_fld_cs_area = torch.clamp(
        flood_storage / river_length - flood_depth * river_width, min=0.0
    )
    fld_impl_area = torch.sqrt(
        upd_fld_cs_area * flood_cross_section_area.clamp(min=1e-6)
    ).clamp(min=1e-6)

    # (7) River outflow
    gt = gravity * time_step
    riv_cs_area = upd_riv_cs_depth * river_width
    riv_cond = (riv_semi > 1e-5) & (riv_cs_area > 1e-5)
    unit_riv_out = river_outflow / river_width
    num_riv = river_width * (unit_riv_out + gt * riv_semi * river_slope)
    den_riv = 1.0 + gt * (river_manning ** 2) * torch.abs(unit_riv_out) * torch.pow(riv_semi, -7.0 / 3.0)
    upd_riv_out = torch.where(riv_cond, num_riv / den_riv, 0.0)

    # (8) Flood outflow
    fld_cond = (fld_semi > 1e-5) & (upd_fld_cs_area > 1e-5)
    num_fld = flood_outflow + gt * fld_impl_area * flood_slope
    den_fld = 1.0 + gt * (flood_manning ** 2) * torch.abs(flood_outflow) * torch.pow(fld_semi, -4.0 / 3.0) / fld_impl_area
    upd_fld_out = torch.where(fld_cond, num_fld / den_fld, 0.0)

    # (9) Opposite direction & negative flow limiting
    opp = (upd_riv_out * upd_fld_out) < 0.0
    upd_fld_out = torch.where(opp, 0.0, upd_fld_out)
    is_neg = (upd_riv_out < 0.0) & ~is_river_mouth
    total_neg_flow = torch.where(is_neg, (-upd_riv_out - upd_fld_out) * time_step, 1.0)
    limit_rate = torch.clamp(
        torch.where(is_neg, 0.05 * tot_sto / total_neg_flow, 1.0),
        max=1.0,
    )
    upd_riv_out = torch.where(is_neg, upd_riv_out * limit_rate, upd_riv_out)
    upd_fld_out = torch.where(is_neg, upd_fld_out * limit_rate, upd_fld_out)

    # (10) Store results (in-place ops – fine for torch.compile)
    river_outflow.copy_(upd_riv_out)
    flood_outflow.copy_(upd_fld_out)
    water_surface_elevation.copy_(wse)
    protected_water_surface_elevation.copy_(prot_wse)
    river_cross_section_depth.copy_(upd_riv_cs_depth)
    flood_cross_section_depth.copy_(upd_fld_cs_depth)
    flood_cross_section_area.copy_(upd_fld_cs_area)
    total_storage.copy_(tot_sto)
    river_inflow.zero_()
    flood_inflow.zero_()
    global_bifurcation_outflow.zero_()

    # (11) Outgoing storage – non-scatter part
    pos = torch.clamp(upd_riv_out, min=0.0) + torch.clamp(upd_fld_out, min=0.0)
    neg = torch.clamp(upd_riv_out, max=0.0) + torch.clamp(upd_fld_out, max=0.0)
    outgoing_storage += pos * time_step

    # River-mouth entries are zeroed so scatter_add_ is a harmless
    # no-op for them (they scatter to themselves).
    to_scatter = torch.where(is_river_mouth, 0.0, -neg * time_step)
    return to_scatter


_compute_outflow_compiled = _torch_compile(_compute_outflow_body)


def compute_outflow_kernel(
    downstream_idx: torch.Tensor,
    river_inflow: torch.Tensor,
    river_outflow: torch.Tensor,
    river_manning: torch.Tensor,
    river_depth: torch.Tensor,
    river_width: torch.Tensor,
    river_length: torch.Tensor,
    river_height: torch.Tensor,
    river_storage: torch.Tensor,
    flood_inflow: torch.Tensor,
    flood_outflow: torch.Tensor,
    flood_manning: torch.Tensor,
    flood_depth: torch.Tensor,
    protected_depth: torch.Tensor,
    catchment_elevation: torch.Tensor,
    downstream_distance: torch.Tensor,
    flood_storage: torch.Tensor,
    protected_storage: torch.Tensor,
    river_cross_section_depth: torch.Tensor,
    flood_cross_section_depth: torch.Tensor,
    flood_cross_section_area: torch.Tensor,
    global_bifurcation_outflow: torch.Tensor,
    total_storage: torch.Tensor,
    outgoing_storage: torch.Tensor,
    water_surface_elevation: torch.Tensor,
    protected_water_surface_elevation: torch.Tensor,
    gravity: float,
    time_step: float,
    num_catchments: int,
    BLOCK_SIZE: int = 128,
) -> None:
    """Outflow kernel: compiled body + eager scatter."""
    to_scatter = _compute_outflow_compiled(
        downstream_idx=downstream_idx,
        river_inflow=river_inflow,
        river_outflow=river_outflow,
        river_manning=river_manning,
        river_depth=river_depth,
        river_width=river_width,
        river_length=river_length,
        river_height=river_height,
        river_storage=river_storage,
        flood_inflow=flood_inflow,
        flood_outflow=flood_outflow,
        flood_manning=flood_manning,
        flood_depth=flood_depth,
        protected_depth=protected_depth,
        catchment_elevation=catchment_elevation,
        downstream_distance=downstream_distance,
        flood_storage=flood_storage,
        protected_storage=protected_storage,
        river_cross_section_depth=river_cross_section_depth,
        flood_cross_section_depth=flood_cross_section_depth,
        flood_cross_section_area=flood_cross_section_area,
        global_bifurcation_outflow=global_bifurcation_outflow,
        total_storage=total_storage,
        outgoing_storage=outgoing_storage,
        water_surface_elevation=water_surface_elevation,
        protected_water_surface_elevation=protected_water_surface_elevation,
        gravity=gravity,
        time_step=time_step,
        num_catchments=num_catchments,
    )
    # Scatter downstream negative outflow (eager – not compiled)
    ds = downstream_idx.long()
    outgoing_storage.scatter_add_(0, ds, to_scatter)


# ======================================================================
# Inflow kernel
# ======================================================================

def _compute_inflow_body(
    downstream_idx: torch.Tensor,
    river_outflow: torch.Tensor,
    flood_outflow: torch.Tensor,
    river_storage: torch.Tensor,
    flood_storage: torch.Tensor,
    outgoing_storage: torch.Tensor,
    river_inflow: torch.Tensor,
    flood_inflow: torch.Tensor,
    limit_rate: torch.Tensor,
    num_catchments: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compilable body – returns values to scatter into inflows.

    Instead of boolean-indexing ``ds[non_mouth]`` (which creates
    dynamic-shaped tensors), river-mouth entries are zeroed before
    scatter so the operation covers the full array with a fixed shape.
    """
    N = num_catchments
    idx = torch.arange(N, device=downstream_idx.device)
    ds = downstream_idx.long()
    is_river_mouth = ds == idx

    riv_out = river_outflow
    fld_out = flood_outflow
    out_sto = outgoing_storage
    rate_sto = river_storage + flood_storage

    # Local limit
    lr = torch.where(out_sto > 1e-8, torch.clamp(rate_sto / out_sto, max=1.0), 1.0)

    # Downstream limit
    out_sto_ds = outgoing_storage[ds]
    rate_sto_ds = river_storage[ds] + flood_storage[ds]
    lr_ds = torch.where(out_sto_ds > 1e-8, torch.clamp(rate_sto_ds / out_sto_ds, max=1.0), 1.0)

    upd_riv = torch.where(riv_out >= 0.0, riv_out * lr, riv_out * lr_ds)
    upd_fld = torch.where(fld_out >= 0.0, fld_out * lr, fld_out * lr_ds)

    river_outflow.copy_(upd_riv)
    flood_outflow.copy_(upd_fld)
    limit_rate.copy_(lr)

    # Zero out river-mouth entries so scatter_add_ over the full
    # array is equivalent to the original masked scatter.
    scatter_riv = torch.where(is_river_mouth, 0.0, upd_riv)
    scatter_fld = torch.where(is_river_mouth, 0.0, upd_fld)
    return scatter_riv, scatter_fld


_compute_inflow_compiled = _torch_compile(_compute_inflow_body)


def compute_inflow_kernel(
    downstream_idx: torch.Tensor,
    river_outflow: torch.Tensor,
    flood_outflow: torch.Tensor,
    river_storage: torch.Tensor,
    flood_storage: torch.Tensor,
    outgoing_storage: torch.Tensor,
    river_inflow: torch.Tensor,
    flood_inflow: torch.Tensor,
    limit_rate: torch.Tensor,
    num_catchments: int,
    BLOCK_SIZE: int = 128,
) -> None:
    """Inflow kernel: compiled body + eager scatter."""
    scatter_riv, scatter_fld = _compute_inflow_compiled(
        downstream_idx=downstream_idx,
        river_outflow=river_outflow,
        flood_outflow=flood_outflow,
        river_storage=river_storage,
        flood_storage=flood_storage,
        outgoing_storage=outgoing_storage,
        river_inflow=river_inflow,
        flood_inflow=flood_inflow,
        limit_rate=limit_rate,
        num_catchments=num_catchments,
    )
    # Accumulate inflows via scatter_add_ (eager – not compiled)
    ds = downstream_idx.long()
    river_inflow.scatter_add_(0, ds, scatter_riv)
    flood_inflow.scatter_add_(0, ds, scatter_fld)


# Batched variants not implemented for MPS / torch backend
compute_outflow_batched_kernel = None
compute_inflow_batched_kernel = None
