# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Pure PyTorch implementation of bifurcation kernels, optimised for MPS.

Compiled bodies are separated from ``scatter_add_`` so that
``torch.compile`` can JIT-fuse the heavy arithmetic while scatter
operations run eagerly.
"""

import torch

from cmfgpu.phys._backend import _torch_compile

# ======================================================================
# Bifurcation outflow
# ======================================================================

def _bifurcation_outflow_body(
    bifurcation_catchment_idx: torch.Tensor,
    bifurcation_downstream_idx: torch.Tensor,
    bifurcation_manning: torch.Tensor,
    bifurcation_outflow: torch.Tensor,
    bifurcation_width: torch.Tensor,
    bifurcation_length: torch.Tensor,
    bifurcation_elevation: torch.Tensor,
    bifurcation_cross_section_depth: torch.Tensor,
    water_surface_elevation: torch.Tensor,
    total_storage: torch.Tensor,
    gravity: float,
    time_step: float,
    num_bifurcation_paths: int,
    num_bifurcation_levels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compilable body – all computation *except* ``scatter_add_``.

    Returns
    -------
    scatter_pos : Tensor, shape (P,)
        Positive outflow values to scatter into *outgoing_storage*
        at ``bifurcation_catchment_idx``.
    scatter_neg : Tensor, shape (P,)
        Negative outflow values (already negated) to scatter into
        *outgoing_storage* at ``bifurcation_downstream_idx``.
    """
    P = num_bifurcation_paths
    L = num_bifurcation_levels

    bci = bifurcation_catchment_idx
    bdi = bifurcation_downstream_idx

    blen = bifurcation_length

    bwse = water_surface_elevation[bci]
    bwse_ds = water_surface_elevation[bdi]
    max_bwse = torch.maximum(bwse, bwse_ds)

    bslope = torch.clamp((bwse - bwse_ds) / blen, -0.005, 0.005)

    b_total_sto = total_storage[bci]
    b_total_sto_ds = total_storage[bdi]

    gt = gravity * time_step
    sum_bif_out = blen.new_zeros(P)

    for level in range(L):
        bman = bifurcation_manning[:, level]
        bcs = bifurcation_cross_section_depth[:, level]
        belev = bifurcation_elevation[:, level]

        upd_bcs = torch.clamp(max_bwse - belev, min=0.0)
        b_semi = torch.maximum(
            torch.sqrt(upd_bcs * bcs),
            torch.sqrt(upd_bcs * 0.01),
        )

        bw = bifurcation_width[:, level]
        bout = bifurcation_outflow[:, level]
        unit_bout = bout / bw

        num = bw * (unit_bout + gt * b_semi * bslope)
        den = 1.0 + gt * (bman ** 2) * torch.abs(unit_bout) * torch.pow(b_semi, -7.0 / 3.0)

        upd_bout = torch.where(b_semi > 1e-5, num / den, 0.0)
        sum_bif_out = sum_bif_out + upd_bout

        bifurcation_cross_section_depth[:, level] = upd_bcs
        bifurcation_outflow[:, level] = upd_bout

    limit_rate = torch.clamp(
        0.05 * torch.minimum(b_total_sto, b_total_sto_ds) / (torch.abs(sum_bif_out) * time_step).clamp(min=1e-30),
        max=1.0,
    )
    sum_bif_out = sum_bif_out * limit_rate

    for level in range(L):
        bifurcation_outflow[:, level] = bifurcation_outflow[:, level] * limit_rate

    pos = torch.clamp(sum_bif_out, min=0.0)
    neg = torch.clamp(sum_bif_out, max=0.0)

    scatter_pos = pos * time_step
    scatter_neg = -neg * time_step
    return scatter_pos, scatter_neg


_bifurcation_outflow_compiled = _torch_compile(_bifurcation_outflow_body)


def compute_bifurcation_outflow_kernel(
    bifurcation_catchment_idx: torch.Tensor,
    bifurcation_downstream_idx: torch.Tensor,
    bifurcation_manning: torch.Tensor,
    bifurcation_outflow: torch.Tensor,
    bifurcation_width: torch.Tensor,
    bifurcation_length: torch.Tensor,
    bifurcation_elevation: torch.Tensor,
    bifurcation_cross_section_depth: torch.Tensor,
    water_surface_elevation: torch.Tensor,
    total_storage: torch.Tensor,
    outgoing_storage: torch.Tensor,
    gravity: float,
    time_step: float,
    num_bifurcation_paths: int,
    num_bifurcation_levels: int,
    BLOCK_SIZE: int = 128,
) -> None:
    """Bifurcation outflow: compiled body + eager scatter."""
    scatter_pos, scatter_neg = _bifurcation_outflow_compiled(
        bifurcation_catchment_idx=bifurcation_catchment_idx,
        bifurcation_downstream_idx=bifurcation_downstream_idx,
        bifurcation_manning=bifurcation_manning,
        bifurcation_outflow=bifurcation_outflow,
        bifurcation_width=bifurcation_width,
        bifurcation_length=bifurcation_length,
        bifurcation_elevation=bifurcation_elevation,
        bifurcation_cross_section_depth=bifurcation_cross_section_depth,
        water_surface_elevation=water_surface_elevation,
        total_storage=total_storage,
        gravity=gravity,
        time_step=time_step,
        num_bifurcation_paths=num_bifurcation_paths,
        num_bifurcation_levels=num_bifurcation_levels,
    )
    bci = bifurcation_catchment_idx
    bdi = bifurcation_downstream_idx
    outgoing_storage.scatter_add_(0, bci, scatter_pos)
    outgoing_storage.scatter_add_(0, bdi, scatter_neg)


# ======================================================================
# Bifurcation inflow
# ======================================================================

def _bifurcation_inflow_body(
    bifurcation_catchment_idx: torch.Tensor,
    bifurcation_downstream_idx: torch.Tensor,
    limit_rate: torch.Tensor,
    bifurcation_outflow: torch.Tensor,
    num_bifurcation_paths: int,
    num_bifurcation_levels: int,
) -> torch.Tensor:
    """Compilable body – returns sum to scatter into
    ``global_bifurcation_outflow``.
    """
    P = num_bifurcation_paths
    L = num_bifurcation_levels

    bci = bifurcation_catchment_idx
    bdi = bifurcation_downstream_idx

    lr = limit_rate[bci]
    lr_ds = limit_rate[bdi]

    sum_bif_out = lr.new_zeros(P)

    for level in range(L):
        bout = bifurcation_outflow[:, level]
        bout = torch.where(bout >= 0.0, bout * lr, bout * lr_ds)
        sum_bif_out = sum_bif_out + bout
        bifurcation_outflow[:, level] = bout

    return sum_bif_out


_bifurcation_inflow_compiled = _torch_compile(_bifurcation_inflow_body)


def compute_bifurcation_inflow_kernel(
    bifurcation_catchment_idx: torch.Tensor,
    bifurcation_downstream_idx: torch.Tensor,
    limit_rate: torch.Tensor,
    bifurcation_outflow: torch.Tensor,
    global_bifurcation_outflow: torch.Tensor,
    num_bifurcation_paths: int,
    num_bifurcation_levels: int,
    BLOCK_SIZE: int = 128,
) -> None:
    """Bifurcation inflow: compiled body + eager scatter."""
    sum_bif_out = _bifurcation_inflow_compiled(
        bifurcation_catchment_idx=bifurcation_catchment_idx,
        bifurcation_downstream_idx=bifurcation_downstream_idx,
        limit_rate=limit_rate,
        bifurcation_outflow=bifurcation_outflow,
        num_bifurcation_paths=num_bifurcation_paths,
        num_bifurcation_levels=num_bifurcation_levels,
    )
    bci = bifurcation_catchment_idx
    bdi = bifurcation_downstream_idx
    global_bifurcation_outflow.scatter_add_(0, bci, sum_bif_out)
    global_bifurcation_outflow.scatter_add_(0, bdi, -sum_bif_out)


# Batched variants not implemented
compute_bifurcation_outflow_batched_kernel = None
compute_bifurcation_inflow_batched_kernel = None
