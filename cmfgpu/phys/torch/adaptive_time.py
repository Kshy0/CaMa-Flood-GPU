# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Pure PyTorch implementation of adaptive time-step kernel."""

import torch


def compute_adaptive_time_step_kernel(
    river_depth: torch.Tensor,
    downstream_distance: torch.Tensor,
    is_dam_related: torch.Tensor,
    max_sub_steps: torch.Tensor,
    time_step: float,
    adaptive_time_factor: float,
    gravity: float,
    num_catchments: int,
    BLOCK_SIZE: int = 128,
    HAS_RESERVOIR: bool = False,
) -> None:
    dd = downstream_distance.float()
    rd = river_depth.float()

    # Skip dam-related cells (Fortran: I2MASK==0 only)
    if HAS_RESERVOIR and is_dam_related is not None:
        valid = ~is_dam_related
        dd = dd[valid]
        rd = rd[valid]

    depth = torch.clamp(rd, min=0.01)
    dt = adaptive_time_factor * dd / torch.sqrt(gravity * depth)
    dt_clamped = torch.clamp(dt, max=time_step)

    min_dt = dt_clamped.min()

    n_steps = (torch.floor(time_step / min_dt + 0.49) + 1.0).to(torch.int32)

    # atomic_max equivalent: pure tensor op (no .item())
    max_sub_steps[0] = torch.maximum(max_sub_steps[0], n_steps)


# Batched variant not implemented
compute_adaptive_time_step_batched_kernel = None
