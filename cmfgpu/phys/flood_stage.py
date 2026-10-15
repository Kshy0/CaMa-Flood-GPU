# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Diagnose initial river, floodplain and protected-area water levels.

The ``compute_flood_stage`` and ``compute_levee_stage`` kernels without
their storage update: like them, each function repartitions the storages
and sets the depths and flooded fraction of the base module in place, here
to diagnose the stage of the initial storage. Storages keep their storage
precision (``hpfloat``) and every other quantity is computed in the
precision of ``river_height``, as in the kernels. Tensors may carry a
leading member axis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from cmfgpu.modules.base import BaseModule
    from cmfgpu.modules.levee import LeveeModule


def flood_stage(base: BaseModule) -> None:
    """Split each catchment's total storage between river and floodplain.

    With levees, the river storage is not capped by the total (the levee
    stage repartitions levee cells afterwards) and the protected storage and
    depth are zero.
    """
    has_levee = base.protected_storage is not None
    total_storage = base.river_storage + base.flood_storage
    if has_levee:
        total_storage = total_storage + base.protected_storage
    river_height, river_width = base.river_height, base.river_width
    river_length, catchment_area = base.river_length, base.catchment_area
    flood_depth_table = base.flood_depth_table
    compute = river_height.dtype
    total = total_storage.to(compute)
    stage_total = total if has_levee else total_storage
    num_levels = flood_depth_table.shape[-1]

    river_max_storage = river_length * river_width * river_height
    width_increment = catchment_area / river_length / num_levels

    level = torch.where(stage_total > river_max_storage, 0, -1)
    accumulated = river_max_storage
    previous_height = torch.zeros_like(river_max_storage)
    previous_width = river_width
    previous_storage = river_max_storage
    previous_depth = torch.zeros_like(river_max_storage)
    next_depth = torch.zeros_like(river_max_storage)
    for i in range(num_levels):
        height = flood_depth_table[..., i]
        width = river_width + (i + 1) * width_increment
        current = accumulated + river_length * 0.5 * (previous_width + width) * (
            height - previous_height
        )
        next_depth = torch.where(level == i, height, next_depth)
        above = stage_total > current
        level = level + above.to(level.dtype)
        previous_storage = torch.where(above, current, previous_storage)
        previous_depth = torch.where(above, height, previous_depth)
        accumulated = current
        previous_height = height
        previous_width = width

    no_flood = level < 0
    level = level.clamp_min(0)
    top = level == num_levels
    total_width = river_width + level * width_increment
    gradient = torch.where(
        top, 0.0, (next_depth - previous_depth) / width_increment
    )
    width_added = (
        torch.sqrt(
            total_width * total_width
            + 2.0 * (total - previous_storage) / (gradient * river_length)
        )
        - total_width
    )
    flood_depth = torch.where(
        no_flood,
        0.0,
        torch.where(
            top,
            previous_depth + (total - previous_storage) / (total_width * river_length),
            previous_depth + width_added * gradient,
        ),
    )

    capacity = (river_max_storage + river_length * river_width * flood_depth).to(
        total_storage.dtype
    )
    if not has_levee:
        capacity = torch.minimum(capacity, total_storage)
    river_storage = torch.where(no_flood, total_storage, capacity)
    river_depth = (river_storage.to(compute) / river_length / river_width).clamp_min(0.0)
    flood_fraction = torch.where(
        no_flood,
        0.0,
        torch.where(
            top,
            1.0,
            (
                (total_width + width_added - river_width) * river_length / catchment_area
            ).clamp(0.0, 1.0),
        ),
    )
    base.river_storage.copy_(river_storage)
    base.flood_storage.copy_((total_storage - river_storage).clamp_min(0.0))
    base.river_depth.copy_(river_depth)
    base.flood_depth.copy_(flood_depth)
    base.flood_fraction.copy_(flood_fraction)
    if has_levee:
        base.protected_storage.zero_()
        base.protected_depth.zero_()


def levee_stage(levee: LeveeModule) -> None:
    """Repartition the levee cells of a ``has_levee`` default stage."""
    base = levee.base
    flood_depth_table = base.flood_depth_table
    idx = levee.levee_catchment_idx.long()

    def at_levees(value: torch.Tensor) -> torch.Tensor:
        return value[..., idx]

    num_levels = flood_depth_table.shape[-1]
    table = flood_depth_table[..., idx, :]
    length = at_levees(base.river_length)
    width = at_levees(base.river_width)
    area = at_levees(base.catchment_area)
    river_max_storage = levee.levee_river_max_storage
    width_increment = area / length / num_levels
    fraction = levee.levee_fraction
    distance = fraction * (area / length)
    levee_level = (fraction * num_levels).to(torch.int32)
    base_height = levee.levee_base_height
    crown = torch.maximum(levee.levee_crown_height, base_height)
    base_storage = levee.levee_base_storage
    top_storage = levee.levee_top_storage
    fill_storage = levee.levee_fill_storage

    river_storage_hp = at_levees(base.river_storage)
    flood_storage_hp = at_levees(base.flood_storage)
    total_hp = river_storage_hp + flood_storage_hp
    total = total_hp.to(length.dtype)

    above_base = (total > river_max_storage) & ~(total < base_storage)
    is_case2 = above_base & (total < top_storage)
    above_top = above_base & ~(total < top_storage)
    is_case3 = above_top & (total < fill_storage)
    is_case4 = above_top & ~(total < fill_storage)

    zero = torch.zeros_like(total)
    not_found = torch.zeros_like(total, dtype=torch.bool)
    fill_storage_b, fill_width_b, fill_depth_b, gradient_b = zero, zero, zero, zero
    found_b = not_found
    fill_storage_c4, fill_width_c4, gradient_c4 = zero, zero, zero
    found_c4 = not_found
    current = river_max_storage
    previous_height = zero
    previous_width = width
    for i in range(num_levels):
        depth = table[..., i]
        segment = depth - previous_height
        middle_width = previous_width + 0.5 * width_increment
        following = current + length * middle_width * segment
        gradient = segment / width_increment

        # Case 3: the protected side fills layer by layer from the levee layer.
        threshold = following + (distance + width) * (crown - depth) * length
        check = (i >= levee_level) & ~found_b
        found = check & (total < threshold)
        passed = check & ~found
        fill_storage_b = torch.where(
            passed,
            threshold,
            torch.where(levee_level == i, levee.levee_layer_top_storage, fill_storage_b),
        )
        fill_width_b = torch.where(
            passed, width_increment * (i + 1) - distance, fill_width_b
        )
        fill_depth_b = torch.where(passed, depth - base_height, fill_depth_b)
        gradient_b = torch.where(found, gradient, gradient_b)
        found_b = found_b | found

        # Case 4 stops at the first layer the storage does not exceed.
        stop = ~found_c4 & ~(total > following)
        fill_storage_c4 = torch.where(stop, current, fill_storage_c4)
        fill_width_c4 = torch.where(stop, previous_width, fill_width_c4)
        gradient_c4 = torch.where(stop, gradient, gradient_c4)
        found_c4 = found_c4 | stop

        current = following
        previous_height = depth
        previous_width = previous_width + width_increment

    # Case 2: river side below the crown, protected side dry.
    flood_depth_c2 = base_height + (total - base_storage) / (
        distance + width
    ) / length
    river_storage_c2 = river_max_storage + length * width * flood_depth_c2
    river_depth_c2 = river_storage_c2 / length / width

    # Case 3: river side at the crown, protected side filling.
    river_storage_c3 = river_max_storage + length * width * crown
    river_depth_c3 = river_storage_c3 / length / width
    added_storage_b = total - fill_storage_b
    added_width_b = -fill_width_b + torch.sqrt(
        fill_width_b * fill_width_b + 2.0 * added_storage_b / length / gradient_b
    )
    protected_depth_c3 = torch.where(
        found_b,
        base_height + fill_depth_b + added_width_b * gradient_b,
        base_height + fill_depth_b + added_storage_b / fill_width_b / length,
    )
    flood_fraction_c3 = torch.where(
        found_b,
        ((fill_width_b + distance) / (width_increment * num_levels)).clamp(0.0, 1.0),
        1.0,
    )

    # Case 4: above the crown; the default river stage stands, with the
    # unclamped default-stage flood fraction.
    added_width_c4 = torch.where(
        found_c4,
        -fill_width_c4
        + torch.sqrt(
            fill_width_c4 * fill_width_c4
            + 2.0 * (total - fill_storage_c4) / length / gradient_c4
        ),
        zero,
    )
    fill_width_c4 = torch.where(found_c4, fill_width_c4, previous_width)
    flood_fraction_c4 = (-width + fill_width_c4 + added_width_c4) / (
        width_increment * num_levels
    )
    flood_depth = at_levees(base.flood_depth)
    added_storage_c4 = (flood_depth - crown) * (distance + width) * length

    partitioned = is_case2 | is_case3 | is_case4
    river_storage = torch.where(
        is_case2 | is_case3,
        torch.where(is_case2, river_storage_c2, river_storage_c3).to(total_hp.dtype),
        river_storage_hp,
    )
    flood_top = torch.where(
        is_case3, top_storage, top_storage + added_storage_c4
    ).to(total_hp.dtype)
    flood_storage = torch.where(
        partitioned,
        (torch.where(is_case2, total_hp, flood_top) - river_storage).clamp_min(0.0),
        flood_storage_hp,
    )
    protected_storage = torch.where(
        is_case3 | is_case4,
        (total_hp - river_storage - flood_storage).clamp_min(0.0),
        0.0,
    )
    river_depth = torch.where(
        is_case2,
        river_depth_c2,
        torch.where(is_case3, river_depth_c3, at_levees(base.river_depth)),
    )
    protected_depth = torch.where(
        is_case3, protected_depth_c3, torch.where(is_case4, flood_depth, 0.0)
    )
    flood_depth = torch.where(
        is_case2, flood_depth_c2, torch.where(is_case3, crown, flood_depth)
    )
    flood_fraction = torch.where(
        is_case2,
        fraction,
        torch.where(
            is_case3,
            flood_fraction_c3,
            torch.where(is_case4, flood_fraction_c4, at_levees(base.flood_fraction)),
        ),
    )

    for target, value in (
        (base.river_storage, river_storage),
        (base.flood_storage, flood_storage),
        (base.protected_storage, protected_storage),
        (base.river_depth, river_depth),
        (base.flood_depth, flood_depth),
        (base.protected_depth, protected_depth),
        (base.flood_fraction, flood_fraction),
    ):
        target[..., idx] = value.to(target.dtype)
