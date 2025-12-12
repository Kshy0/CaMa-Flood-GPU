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
from pydantic import Field, computed_field, model_validator

from cmfgpu.modules.abstract_module import (AbstractModule, TensorField,
                                            computed_tensor_field)
from cmfgpu.modules.base import BaseModule
from cmfgpu.utils import find_indices_in_torch


def LeveeField(
    description: str,
    shape: Tuple[str, ...] = ("base.num_levees",),
    dtype: Literal["float", "int", "bool"] = "float",
    group_by: Optional[str] = "levee_basin_id",
    save_idx: Optional[str] = "levee_save_idx",
    save_coord: Optional[str] = "levee_save_id",
    intermediate: bool = False,
    **kwargs,
):
    return TensorField(
        description=description,
        shape=shape,
        dtype=dtype,
        group_by=group_by,
        save_idx=save_idx,
        save_coord=save_coord,
        intermediate=intermediate,
        **kwargs,
    )


def computed_levee_field(
    description: str,
    shape: Tuple[str, ...] = ("base.num_levees",),
    dtype: Literal["float", "int", "bool"] = "float",
    save_idx: Optional[str] = "levee_save_idx",
    save_coord: Optional[str] = "levee_save_id",
    intermediate: bool = False,
    **kwargs,
):
    return computed_tensor_field(
        description=description,
        shape=shape,
        dtype=dtype,
        save_idx=save_idx,
        save_coord=save_coord,
        intermediate=intermediate,
        **kwargs,
    )


class LeveeModule(AbstractModule):
    """Container for levee-related tensors."""

    module_name: ClassVar[str] = "levee"
    description: ClassVar[str] = "Levee protection module with protected storage states"
    dependencies: ClassVar[list[str]] = ["base"]

    base: Optional[BaseModule] = Field(default=None, exclude=True, description="Reference to BaseModule")

    # ------------------------------------------------------------------ #
    # Levee metadata and topology
    # ------------------------------------------------------------------ #

    levee_basin_id: torch.Tensor = LeveeField(
        description="Basin ID per levee (used for data distribution)",
        dtype="int",
    )

    levee_save_mask: Optional[torch.Tensor] = LeveeField(
        description="Mask of levees whose diagnostics should be saved",
        dtype="bool",
        default=None,
    )

    levee_id: torch.Tensor = LeveeField(
        description="Unique ID for each levee",
        dtype="int",
    )

    levee_catchment_id: torch.Tensor = LeveeField(
        description="Catchment ID for each levee",
        dtype="int",
    )

    # ------------------------------------------------------------------ #
    # Static levee parameters (num_levees)
    # ------------------------------------------------------------------ #
    levee_crown_height: torch.Tensor = LeveeField(
        description="Levee crown height above river bed (m)",
    )

    levee_fraction: torch.Tensor = LeveeField(
        description="Relative distance between river and levee (0 close to channel, 1 far end)",
    )

    # ------------------------------------------------------------------ #
    # Hidden / intermediate states
    # ------------------------------------------------------------------ #
    @computed_levee_field(description="Total number of catchments represented in this module")
    @cached_property
    def protected_storage(self) -> torch.Tensor:
        return torch.zeros_like(self.levee_crown_height)

    # ------------------------------------------------------------------ #
    # Computed scalar metadata
    # ------------------------------------------------------------------ #

    @computed_field(description="Number of levees whose outputs are persisted")
    @cached_property
    def num_saved_levees(self) -> int:
        if self.levee_save_mask is None:
            return self.base.num_levees
        active = torch.count_nonzero(self.levee_save_mask)
        return int(active.item())

    # ------------------------------------------------------------------ #
    # Computed tensors (levee-aligned)
    # ------------------------------------------------------------------ #
    @computed_levee_field(description="Indices of catchments hosting each levee", dtype="int")
    @cached_property
    def levee_catchment_idx(self) -> torch.Tensor:
        return find_indices_in_torch(self.levee_catchment_id, self.base.catchment_id)

    def _interp_lookup(self, table: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        rows = table[self.levee_catchment_idx]
        
        # Clamp position to valid range [0, num_flood_levels]
        position = torch.clamp(position, 0.0, float(self.base.num_flood_levels))
        
        lower = torch.floor(position).to(torch.int64)
        upper = lower + 1
        
        frac = position - lower
        
        # Map virtual indices 0..N to table indices -1..N-1
        # Virtual index 0 -> 0.0 value
        # Virtual index k (>0) -> table[:, k-1]
        
        # Lower value
        lower_idx_table = lower - 1
        lower_is_zero = (lower == 0)
        # Clamp negative index to 0 for gather (will be masked out)
        lower_gather_idx = torch.clamp(lower_idx_table, min=0)
        lower_val_from_table = rows.gather(1, lower_gather_idx.unsqueeze(-1)).squeeze(-1)
        lower_val = torch.where(lower_is_zero, torch.tensor(0.0, device=rows.device, dtype=rows.dtype), lower_val_from_table)
        
        # Upper value
        upper_idx_table = upper - 1
        upper_idx_table = torch.clamp(upper_idx_table, max=self.base.num_flood_levels - 1)
        upper_val = rows.gather(1, upper_idx_table.unsqueeze(-1)).squeeze(-1)
        
        return (lower_val + frac * (upper_val - lower_val)).contiguous()
    
    @computed_levee_field(description="Levee base height above river bed (m)")
    @cached_property
    def levee_base_height(self) -> torch.Tensor:
        return self._interp_lookup(self.base.flood_depth_table, self.levee_fraction * self.base.num_flood_levels)
    
    @computed_levee_field(description="Storage when water first touches levee base (m³)")
    @cached_property
    def levee_base_storage(self) -> torch.Tensor:
        # Gather parameters for levee catchments
        idx = self.levee_catchment_idx
        river_length = self.base.river_length[idx]
        river_width = self.base.river_width[idx]
        river_height = self.base.river_height[idx]
        catchment_area = self.base.catchment_area[idx]
        flood_depth_table = self.base.flood_depth_table[idx] # (num_levees, num_flood_levels)

        num_flood_levels = self.base.num_flood_levels
        
        # Compute river_max_storage
        river_max_storage = river_length * river_width * river_height
        
        # Compute width_increment
        catchment_width = catchment_area / river_length
        width_increment = catchment_width / num_flood_levels
        
        # Compute widths at each level
        # levels 1 to M
        levels = torch.arange(1, num_flood_levels + 1, device=self.device, dtype=torch.float32)
        # W shape: (num_levees, num_flood_levels)
        W = river_width.unsqueeze(1) + levels.unsqueeze(0) * width_increment.unsqueeze(1)
        
        # Pad H and W for vectorization
        # H_padded: 0, H_0, H_1, ...
        H_padded = torch.cat([torch.zeros((idx.shape[0], 1), device=self.device), flood_depth_table], dim=1)
        
        # W_padded: W_riv, W_0, W_1, ...
        W_padded = torch.cat([river_width.unsqueeze(1), W], dim=1)
        
        # Compute dS
        # dS_i = L * 0.5 * (W_{i-1} + W_i) * (H_i - H_{i-1})
        # indices in padded: i and i+1
        # i ranges from 0 to M-1
        
        W_avg = 0.5 * (W_padded[:, :-1] + W_padded[:, 1:])
        dH = H_padded[:, 1:] - H_padded[:, :-1]
        dS = river_length.unsqueeze(1) * W_avg * dH
        
        # Cumulative storage
        S_accum = torch.cumsum(dS, dim=1)
        S_table = river_max_storage.unsqueeze(1) + S_accum
        
        # Interpolate
        position = self.levee_fraction * self.base.num_flood_levels
        
        lower = torch.floor(position).to(torch.int64)
        upper = lower + 1
        frac = position - lower
        
        # Lower value
        lower_idx_table = lower - 1
        lower_is_zero = (lower == 0)
        lower_gather_idx = torch.clamp(lower_idx_table, min=0)
        
        lower_val_from_table = S_table.gather(1, lower_gather_idx.unsqueeze(-1)).squeeze(-1)
        lower_val = torch.where(lower_is_zero, river_max_storage, lower_val_from_table)
        
        # Upper value
        upper_idx_table = upper - 1
        upper_idx_table = torch.clamp(upper_idx_table, max=num_flood_levels - 1)
        upper_val = S_table.gather(1, upper_idx_table.unsqueeze(-1)).squeeze(-1)
        
        return (lower_val + frac * (upper_val - lower_val)).contiguous()

    # ------------------------------------------------------------------ #
    # Save/selection helpers
    # ------------------------------------------------------------------ #
    @computed_levee_field(
        description="Indices of levees whose outputs are saved",
        dtype="int",
        shape=("num_saved_levees",),
    )
    @cached_property
    def levee_save_idx(self) -> Optional[torch.Tensor]:
        if self.levee_save_mask is None:
            return torch.arange(self.base.num_levees, dtype=torch.int64, device=self.device)
        idx = torch.nonzero(self.levee_save_mask, as_tuple=False).squeeze(-1).to(torch.int64)
        return idx if idx.numel() > 0 else None

    @computed_levee_field(
        description="Levee IDs that correspond to saved outputs",
        dtype="int",
        shape=("num_saved_levees",),
    )
    @cached_property
    def levee_save_id(self) -> Optional[torch.Tensor]:
        if self.levee_save_idx is None:
            return None
        return self.levee_catchment_id[self.levee_save_idx]

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @model_validator(mode="after")
    def validate_levee_fraction(self) -> Self:
        # Check for strictly invalid values (outside [0, 1])
        if torch.any((self.levee_fraction < 0) | (self.levee_fraction > 1)):
            raise ValueError("levee_fraction must lie within [0, 1]")
            
        # Check for boundary values (0 or 1)
        num_zeros = (self.levee_fraction == 0).sum().item()
        num_ones = (self.levee_fraction == 1).sum().item()
        
        if num_zeros > 0 or num_ones > 0:
            print(
                f"[rank {self.rank}][LeveeModule] Warning: Found {num_zeros} levees with fraction=0 "
                f"and {num_ones} with fraction=1. Boundary values are allowed but not expected."
            )
        return self

    @model_validator(mode="after")
    def validate_levee_catchment_idx(self) -> Self:
        if torch.any(self.levee_catchment_idx < 0):
            raise ValueError("levee_catchment_id contains entries absent from catchment_id")
        return self

    @model_validator(mode="after")
    def validate_levee_height(self) -> Self:
        invalid = self.levee_base_height >= self.levee_crown_height
        num_invalid = invalid.sum().item()
        if num_invalid > 0:
            print(
                f"[rank {self.rank}][LeveeModule] Found {num_invalid} levees with invalid height (base >= crown). "
                "Fixing by setting crown = max(crown, base)."
            )
            self.levee_crown_height = torch.maximum(self.levee_crown_height, self.levee_base_height)
        return self
