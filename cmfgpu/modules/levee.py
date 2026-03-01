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
    dtype: Literal["float", "int", "idx", "bool"] = "float",
    group_by: Optional[str] = "levee_basin_id",
    dim_coords: Optional[str] = "base.levee_catchment_id",
    category: Literal["topology", "param", "init_state"] = "param",
    mode: Literal["device", "cpu", "discard"] = "device",
    **kwargs,
):
    return TensorField(
        description=description,
        shape=shape,
        dtype=dtype,
        group_by=group_by,
        save_idx=None,
        save_coord=None,
        dim_coords=dim_coords,
        category=category,
        mode=mode,
        **kwargs,
    )


def computed_levee_field(
    description: str,
    shape: Tuple[str, ...] = ("base.num_levees",),
    dtype: Literal["float", "int", "idx", "bool"] = "float",
    dim_coords: Optional[str] = "base.levee_catchment_id",
    category: Literal["topology", "derived_param", "state", "virtual"] = "derived_param",
    expr: Optional[str] = None,
    **kwargs,
):
    return computed_tensor_field(
        description=description,
        shape=shape,
        dtype=dtype,
        save_idx=None,
        save_coord=None,
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

    base: Optional[BaseModule] = Field(
        default=None,
        exclude=True,
        description="Reference to BaseModule",
    )

    # ------------------------------------------------------------------ #
    # Levee metadata and topology
    # ------------------------------------------------------------------ #

    levee_id: torch.Tensor = LeveeField(
        description="Unique ID for each levee",
        dtype="int",
        category="topology",
        mode="cpu",
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
    @computed_levee_field(description="Indices of catchments hosting each levee", dtype="idx", category="topology")
    @cached_property
    def levee_catchment_idx(self) -> torch.Tensor:
        return find_indices_in_torch(self.base.levee_catchment_id, self.base.catchment_id)

    def _interp_lookup(self, table: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        # table: (N, M) or (T, N, M)
        # position: (L,) or (T, L)
        
        # Gather rows for levees
        rows = self.gather_tensor(table, self.levee_catchment_idx)
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
        lower_is_zero = (lower == 0)
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

        lower_val_from_table = rows.gather(-1, lower_gather_idx.unsqueeze(-1)).squeeze(-1)
        zero = torch.tensor(0.0, device=rows.device, dtype=rows.dtype)
        lower_val = torch.where(lower_is_zero, zero, lower_val_from_table)
        
        # Upper value
        upper_idx_table = upper - 1
        upper_idx_table = torch.clamp(upper_idx_table, max=self.base.num_flood_levels - 1)
        
        # Ensure upper_idx_table matches rows shape (which might have been expanded above)
        if rows.ndim == 3 and upper_idx_table.ndim == 1:
             target_shape = (rows.shape[0], rows.shape[1])
             upper_idx_table = upper_idx_table.expand(target_shape)

        upper_val = rows.gather(-1, upper_idx_table.unsqueeze(-1)).squeeze(-1)
        
        return (lower_val + frac * (upper_val - lower_val)).contiguous()
    
    @computed_levee_field(description="Levee base height above river bed (m)", category="derived_param")
    @cached_property
    def levee_base_height(self) -> torch.Tensor:
        return self._interp_lookup(self.base.flood_depth_table, self.levee_fraction * self.base.num_flood_levels)
    
    @computed_levee_field(description="Storage when water first touches levee base (m³)", category="derived_param")
    @cached_property
    def levee_base_storage(self) -> torch.Tensor:
        # Gather parameters for levee catchments
        idx = self.levee_catchment_idx
        river_length = self.gather_tensor(self.base.river_length, idx)
        river_width = self.gather_tensor(self.base.river_width, idx)
        river_height = self.gather_tensor(self.base.river_height, idx)
        catchment_area = self.gather_tensor(self.base.catchment_area, idx)
        flood_depth_table = self.gather_tensor(self.base.flood_depth_table, idx) # (L, M) or (T, L, M)

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
        
        # Check if any input is batched
        is_batched = (
            river_length.ndim > 1 or 
            river_width.ndim > 1 or 
            river_height.ndim > 1 or 
            catchment_area.ndim > 1 or 
            flood_depth_table.ndim > 2
        )
        
        if is_batched:
            # Ensure all inputs are batched (T, L, ...)
            num_trials = self.num_trials or 1
            if river_length.ndim == 1: river_length = river_length.expand(num_trials, -1)
            if river_width.ndim == 1: river_width = river_width.expand(num_trials, -1)
            if river_height.ndim == 1: river_height = river_height.expand(num_trials, -1)
            if catchment_area.ndim == 1: catchment_area = catchment_area.expand(num_trials, -1)
            if flood_depth_table.ndim == 2: flood_depth_table = flood_depth_table.expand(num_trials, -1, -1)
            
            # Recalculate derived batched vars
            river_max_storage = river_length * river_width * river_height
            catchment_width = catchment_area / river_length
            width_increment = catchment_width / num_flood_levels
            
            # W calculation for batched inputs
            # river_width: (T, L)
            # levels: (M,)
            # width_increment: (T, L)
            W = river_width.unsqueeze(-1) + levels.view(1, 1, -1) * width_increment.unsqueeze(-1)
        else:
            # Shared inputs
            W = river_width.unsqueeze(-1) + levels.view(1, -1) * width_increment.unsqueeze(-1)
        
        # Pad H and W for vectorization
        # H_padded: 0, H_0, H_1, ...
        
        zeros_shape = list(flood_depth_table.shape)
        zeros_shape[-1] = 1
        zeros = torch.zeros(zeros_shape, device=self.device, dtype=flood_depth_table.dtype)
        
        H_padded = torch.cat([zeros, flood_depth_table], dim=-1)
        
        # W_padded: W_riv, W_0, W_1, ...
        W_padded = torch.cat([river_width.unsqueeze(-1), W], dim=-1)
        
        # Compute dS
        # dS_i = L * 0.5 * (W_{i-1} + W_i) * (H_i - H_{i-1})
        # indices in padded: i and i+1
        # i ranges from 0 to M-1
        
        W_avg = 0.5 * (W_padded[..., :-1] + W_padded[..., 1:])
        dH = H_padded[..., 1:] - H_padded[..., :-1]
        dS = river_length.unsqueeze(-1) * W_avg * dH
        
        # Cumulative storage
        S_accum = torch.cumsum(dS, dim=-1)
        S_table = river_max_storage.unsqueeze(-1) + S_accum
        
        # Interpolate
        position = self.levee_fraction * self.base.num_flood_levels
        
        lower = torch.floor(position).to(torch.int32)
        upper = lower + 1
        frac = position - lower
        
        # Lower value
        lower_idx_table = lower - 1
        lower_is_zero = (lower == 0)
        lower_gather_idx = torch.clamp(lower_idx_table, min=0)
        
        # Handle broadcasting for gather
        # S_table: (L, M) or (T, L, M)
        # lower_gather_idx: (L,) or (T, L)
        
        if S_table.ndim == 3 and lower_gather_idx.ndim == 1:
             target_shape = (S_table.shape[0], S_table.shape[1])
             lower_gather_idx = lower_gather_idx.expand(target_shape)
             lower_is_zero = lower_is_zero.expand(target_shape)
        elif S_table.ndim == 2 and lower_gather_idx.ndim == 2:
             target_shape = (lower_gather_idx.shape[0], S_table.shape[0], S_table.shape[1])
             S_table = S_table.expand(target_shape)

        lower_val_from_table = S_table.gather(-1, lower_gather_idx.unsqueeze(-1)).squeeze(-1)
        
        # river_max_storage needs to match shape for where
        if river_max_storage.ndim < lower_val_from_table.ndim:
             river_max_storage = river_max_storage.expand_as(lower_val_from_table)
             
        lower_val = torch.where(lower_is_zero, river_max_storage, lower_val_from_table)
        
        # Upper value
        upper_idx_table = upper - 1
        upper_idx_table = torch.clamp(upper_idx_table, max=num_flood_levels - 1)
        
        if S_table.ndim == 3 and upper_idx_table.ndim == 1:
             target_shape = (S_table.shape[0], S_table.shape[1])
             upper_idx_table = upper_idx_table.expand(target_shape)

        upper_val = S_table.gather(-1, upper_idx_table.unsqueeze(-1)).squeeze(-1)
        
        return (lower_val + frac * (upper_val - lower_val))

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @model_validator(mode="after")
    def validate_levee_fraction(self) -> Self:
        # Check for strictly invalid values (outside [0, 1])
        if torch.any((self.levee_fraction < 0) | (self.levee_fraction >= 1)):
            raise ValueError("levee_fraction must lie within [0, 1)")
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

    # ------------------------------------------------------------------ #
    # Batched flags
    # ------------------------------------------------------------------ #
    @computed_field
    @cached_property
    def batched_levee_base_height(self) -> bool:
        return self._is_batched(self.levee_base_height)

    @computed_field
    @cached_property
    def batched_levee_crown_height(self) -> bool:
        return self._is_batched(self.levee_crown_height)

    @computed_field
    @cached_property
    def batched_levee_fraction(self) -> bool:
        return self._is_batched(self.levee_fraction)
