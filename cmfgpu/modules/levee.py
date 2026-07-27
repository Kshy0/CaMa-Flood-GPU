# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Levee module definitions for CaMa-Flood-GPU.
"""

from __future__ import annotations

from functools import cached_property
from typing import ClassVar, Literal, Optional, Self, Tuple

import torch
from hydroforge.model.module import (
    AbstractModule,
    CoordinateField,
    ReferenceField,
    ReferenceIndexField,
    TensorField,
    computed_tensor_field,
)
from pydantic import Field, computed_field, model_validator

from cmfgpu.modules.base import BaseModule


def LeveeField(
    description: str,
    shape: Tuple[str, ...] = ("num_levees",),
    dtype: Literal["float", "int", "idx", "bool"] = "float",
    dim_coords: Optional[str] = "levee_id",
    category: Literal["topology", "param", "init_state"] = "param",
    mode: Literal["device", "cpu", "discard"] = "device",
    **kwargs,
):
    return TensorField(
        description=description,
        shape=shape,
        dtype=dtype,
        dim_coords=dim_coords,
        category=category,
        mode=mode,
        **kwargs,
    )


def computed_levee_field(
    description: str,
    shape: Tuple[str, ...] = ("num_levees",),
    dtype: Literal["float", "int", "idx", "bool"] = "float",
    dim_coords: Optional[str] = "levee_id",
    category: Literal[
        "topology", "derived_param", "state", "virtual"
    ] = "derived_param",
    expr: Optional[str] = None,
    **kwargs,
):
    return computed_tensor_field(
        description=description,
        shape=shape,
        dtype=dtype,
        dim_coords=dim_coords,
        category=category,
        expr=expr,
        **kwargs,
    )


class LeveeModule(AbstractModule):
    """Container for levee-related tensors."""

    module_name: ClassVar[str] = "levee"
    description: ClassVar[str] = "Levee protection module with protected storage states"
    dependencies: ClassVar[list[str]] = ["base"]

    base: BaseModule = Field(
        exclude=True,
        description="Reference to BaseModule",
    )

    # ------------------------------------------------------------------ #
    # Levee metadata and topology
    # ------------------------------------------------------------------ #

    levee_catchment_id: torch.Tensor = ReferenceField(
        description="Catchment ID hosting each levee",
        dtype="int",
        shape=("num_levees",),
        dim_coords="levee_id",
        references="catchment_id",
        is_key=True,
    )

    levee_id: torch.Tensor = CoordinateField(
        description="Unique ID for each levee",
        dtype="int",
        shape=("num_levees",),
        partition_by="levee_catchment_id",
    )

    # ------------------------------------------------------------------ #
    # Static levee parameters (num_levees)
    # ------------------------------------------------------------------ #
    levee_crown_height: torch.Tensor = LeveeField(
        description="Levee crown height above river bed (m)",
        category="param",
    )

    levee_fraction: torch.Tensor = LeveeField(
        description="Relative distance between river and levee (0 close to channel, 1 far end)",
        category="param",
    )

    # ------------------------------------------------------------------ #
    # Computed tensors (levee-aligned)
    # ------------------------------------------------------------------ #
    @computed_field(description="Total number of levees")
    @cached_property
    def num_levees(self) -> int:
        return self.levee_catchment_id.shape[0]

    levee_catchment_idx = ReferenceIndexField("levee_catchment_id")

    @computed_tensor_field(
        description="Boolean mask for catchments governed by levee physics",
        shape=("base.num_catchments",),
        dtype="bool",
        dim_coords="base.catchment_id",
        category="topology",
    )
    @cached_property
    def is_levee(self) -> torch.Tensor:
        mask = torch.zeros(
            self.base.num_catchments,
            dtype=torch.bool,
            device=self.base.catchment_id.device,
        )
        mask[self.levee_catchment_idx] = True
        return mask

    def _interp_lookup(
        self, table: torch.Tensor, position: torch.Tensor
    ) -> torch.Tensor:
        # table: (N, M) or (T, N, M)
        # position: (L,) or (T, L)

        # Gather rows for levees
        rows = self.gather_tensor(
            table,
            self.levee_catchment_idx,
            batched=self.base.is_batched("flood_depth_table"),
        )
        # rows: (L, M) or (T, L, M)

        # Clamp position to valid range [0, num_flood_levels]
        position = torch.clamp(position, 0.0, float(self.base.num_flood_levels))

        lower = torch.floor(position).to(torch.int32)
        upper = lower + 1

        frac = position - lower

        # Map virtual indices 0..N to table indices -1..N-1
        # Virtual index 0 -> 0.0 value
        # Virtual index k (>0) -> table[:, k-1]

        # Lower value
        lower_idx_table = lower - 1
        lower_is_zero = lower == 0
        # Clamp negative index to 0 for gather (will be masked out)
        lower_gather_idx = torch.clamp(lower_idx_table, min=0)

        # Handle broadcasting for gather
        # We need rows and indices to have compatible shapes for gather
        # rows: (L, M) or (T, L, M)
        # lower_gather_idx: (L,) or (T, L)

        # Case 1: rows is batched (T, L, M), index is shared (L,) -> Expand index to (T, L)
        if rows.ndim == 3 and lower_gather_idx.ndim == 1:
            target_shape = (rows.shape[0], rows.shape[1])
            lower_gather_idx = lower_gather_idx.expand(target_shape)
            lower_is_zero = lower_is_zero.expand(target_shape)

        # Case 2: rows is shared (L, M), index is batched (T, L) -> Expand rows to (T, L, M)
        elif rows.ndim == 2 and lower_gather_idx.ndim == 2:
            target_shape = (lower_gather_idx.shape[0], rows.shape[0], rows.shape[1])
            rows = rows.expand(target_shape)

        lower_val_from_table = rows.gather(-1, lower_gather_idx.unsqueeze(-1)).squeeze(
            -1
        )
        zero = torch.tensor(0.0, device=rows.device, dtype=rows.dtype)
        lower_val = torch.where(lower_is_zero, zero, lower_val_from_table)

        # Upper value
        upper_idx_table = upper - 1
        upper_idx_table = torch.clamp(
            upper_idx_table, max=self.base.num_flood_levels - 1
        )

        # Ensure upper_idx_table matches rows shape (which might have been expanded above)
        if rows.ndim == 3 and upper_idx_table.ndim == 1:
            target_shape = (rows.shape[0], rows.shape[1])
            upper_idx_table = upper_idx_table.expand(target_shape)

        upper_val = rows.gather(-1, upper_idx_table.unsqueeze(-1)).squeeze(-1)

        return (lower_val + frac * (upper_val - lower_val)).contiguous()

    @computed_levee_field(
        description="Levee base height above river bed (m)", category="derived_param"
    )
    @cached_property
    def levee_base_height(self) -> torch.Tensor:
        return self._interp_lookup(
            self.base.flood_depth_table,
            self.levee_fraction * self.base.num_flood_levels,
        )

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @model_validator(mode="after")
    def validate_levee_fraction(self) -> Self:
        invalid = (
            ~torch.isfinite(self.levee_fraction)
            | (self.levee_fraction < 0)
            | (self.levee_fraction >= 1)
        )
        if torch.any(invalid):
            raise ValueError("levee_fraction must be finite and lie within [0, 1)")
        return self

    @model_validator(mode="after")
    def validate_levee_height(self) -> Self:
        invalid = (
            ~torch.isfinite(self.levee_crown_height)
            | ~torch.isfinite(self.levee_base_height)
            | (self.levee_base_height >= self.levee_crown_height)
        )
        num_invalid = invalid.sum().item()
        if num_invalid > 0:
            raise ValueError(
                f"{num_invalid} levees have non-finite heights or a base "
                "height greater than or equal to the crown height"
            )
        return self
