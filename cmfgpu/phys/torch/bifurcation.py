# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Pure PyTorch implementation of bifurcation kernels."""

import torch


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
    P = num_bifurcation_paths
    L = num_bifurcation_levels

    bci = bifurcation_catchment_idx.long()
    bdi = bifurcation_downstream_idx.long()

    blen = bifurcation_length

    bwse = water_surface_elevation[bci]
    bwse_ds = water_surface_elevation[bdi]
    max_bwse = torch.maximum(bwse, bwse_ds)

    bslope = torch.clamp((bwse - bwse_ds) / blen, -0.005, 0.005)

    b_total_sto = total_storage[bci]
    b_total_sto_ds = total_storage[bdi]

    # Level-indexed arrays: shape (P, L)
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
        sum_bif_out += upd_bout

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
    outgoing_storage.scatter_add_(0, bci, pos * time_step)
    outgoing_storage.scatter_add_(0, bdi, -neg * time_step)


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
    P = num_bifurcation_paths
    L = num_bifurcation_levels

    bci = bifurcation_catchment_idx.long()
    bdi = bifurcation_downstream_idx.long()

    lr = limit_rate[bci]
    lr_ds = limit_rate[bdi]

    sum_bif_out = lr.new_zeros(P)

    for level in range(L):
        bout = bifurcation_outflow[:, level]
        bout = torch.where(bout >= 0.0, bout * lr, bout * lr_ds)
        sum_bif_out += bout
        bifurcation_outflow[:, level] = bout

    global_bifurcation_outflow.scatter_add_(0, bci, sum_bif_out)
    global_bifurcation_outflow.scatter_add_(0, bdi, -sum_bif_out)


# Batched variants not implemented
compute_bifurcation_outflow_batched_kernel = None
compute_bifurcation_inflow_batched_kernel = None
