# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Reservoir module for CaMa-Flood-GPU using TensorField / computed_tensor_field helpers.
"""
from __future__ import annotations

from functools import cached_property
from typing import ClassVar, Literal, Optional, Self, Tuple

import torch
from pydantic import computed_field, model_validator

from cmfgpu.modules.abstract_module import (AbstractModule, TensorField,
                                            computed_tensor_field)
from cmfgpu.utils import find_indices_in_torch


def ReservoirField(
    description: str,
    shape: Tuple[str, ...] = ("num_reservoirs",),
    dtype: Literal["float", "int", "bool"] = "float",
    group_by: Optional[str] = "reservoir_basin_id",
    save_idx: Optional[str] = "reservoir_save_idx",
    save_coord: Optional[str] = "reservoir_save_id",
    dim_coords: Optional[str] = "reservoir_catchment_id",
    **kwargs
):
    return TensorField(
        description=description,
        shape=shape,
        dtype=dtype,
        group_by=group_by,
        save_idx=save_idx,
        save_coord=save_coord,
        dim_coords=dim_coords,
        **kwargs
    )

def computed_reservoir_field(
    description: str,
    shape: Tuple[str, ...] = ("num_reservoirs",),
    dtype: Literal["float", "int", "bool"] = "float",
    save_idx: Optional[str] = "reservoir_save_idx",
    save_coord: Optional[str] = "reservoir_save_id",
    dim_coords: Optional[str] = "reservoir_catchment_id",
    **kwargs
):
    return computed_tensor_field(
        description=description,
        shape=shape,
        dtype=dtype,
        save_idx=save_idx,
        save_coord=save_coord,
        dim_coords=dim_coords,
        **kwargs
    )

class ReservoirModule(AbstractModule):
    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    module_name: ClassVar[str] = "reservoir"
    description: ClassVar[str] = "Reservoir operation module with storage and outflow regulation"
    dependencies: ClassVar[list] = ["base"]

    # ------------------------------------------------------------------ #
    # Reservoir topology
    # ------------------------------------------------------------------ #
    reservoir_save_mask: Optional[torch.Tensor] = ReservoirField(
        description="Mask of reservoirs to include in output",
        dtype="bool",
        default=None,
    )

    reservoir_catchment_id: torch.Tensor = ReservoirField(
        description="Catchment IDs where reservoirs are located",
        dtype="int",
    )

    # ------------------------------------------------------------------ #
    # Physical properties
    # ------------------------------------------------------------------ #
    reservoir_capacity: torch.Tensor = ReservoirField(
        description="Maximum storage capacity (m³)",
    )

    conservation_volume: torch.Tensor = ReservoirField(
        description="Conservation storage volume (m³)",
    )

    emergency_volume: torch.Tensor = ReservoirField(
        description="Emergency storage volume (m³)",
    )

    normal_outflow: torch.Tensor = ReservoirField(
        description="Normal outflow rate (m³ s⁻¹)",
    )

    flood_control_outflow: torch.Tensor = ReservoirField(
        description="Flood-control outflow rate (m³ s⁻¹)",
    )

    reservoir_area: torch.Tensor = ReservoirField(
        description="Surface area at normal water level (m²)",
    )

    # ------------------------------------------------------------------ #
    # State variables
    # ------------------------------------------------------------------ #
    reservoir_storage: torch.Tensor = ReservoirField(
        description="Current storage (m³)",
        default=0,
    )

    reservoir_outflow: torch.Tensor = ReservoirField(
        description="Current outflow (m³ s⁻¹)",
        default=0,
    )

    reservoir_water_level: torch.Tensor = ReservoirField(
        description="Current water level (m)",
        default=0,
    )

    # ------------------------------------------------------------------ #
    # Computed tensor indices
    # ------------------------------------------------------------------ #
    @computed_reservoir_field(
        description="Catchment-array indices for each reservoir",
        shape=("num_reservoirs",),
        dtype="int",
    )
    @cached_property
    def reservoir_catchment_idx(self) -> torch.Tensor:
        return find_indices_in_torch(self.reservoir_catchment_id, self.catchment_id)

    @computed_reservoir_field(
        description="Indices of reservoirs saved in output",
        dtype="int",
    )
    @cached_property
    def reservoir_save_idx(self) -> torch.Tensor:
        if self.reservoir_save_mask is None:
            return torch.arange(self.num_reservoirs, dtype=torch.int64, device=self.device)
        return torch.nonzero(self.reservoir_save_mask, as_tuple=False).squeeze(-1)

    # ------------------------------------------------------------------ #
    # Computed scalar dimensions
    # ------------------------------------------------------------------ #
    @computed_field(description="Number of reservoirs saved in output.")
    @cached_property
    def num_saved_reservoirs(self) -> int:
        return len(self.reservoir_save_idx)

    @computed_field(description="Total number of reservoirs.")
    @cached_property
    def num_reservoirs(self) -> int:
        return self.reservoir_capacity.shape[0]

    @computed_field(description="Total number of catchments.")
    @cached_property
    def num_catchments(self) -> int:
        return self.catchment_id.shape[0]

    # ------------------------------------------------------------------ #
    # Computed tensors (operations)
    # ------------------------------------------------------------------ #
    @computed_reservoir_field(
        description="Volume threshold triggering regulation (m³)",
    )
    @cached_property
    def adjustment_volume(self) -> torch.Tensor:
        return self.conservation_volume + self.emergency_volume * 0.1

    @computed_reservoir_field(
        description="Regulated outflow rate (m³ s⁻¹)",
    )
    @cached_property
    def adjustment_outflow(self) -> torch.Tensor:
        seconds_per_year = 365.0 * 24 * 60 * 60
        seconds_per_180_days = 180.0 * 24 * 60 * 60
        term1 = self.conservation_volume * 0.7
        term2 = self.normal_outflow * (seconds_per_year / 4)
        combined = (term1 + term2) / seconds_per_180_days
        min_outflow = torch.minimum(self.normal_outflow, combined)
        return (min_outflow * 1.5 + self.flood_control_outflow) * 0.5

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @model_validator(mode="after")
    def validate_reservoir_save_idx(self) -> Self:
        if not torch.all(
            (self.reservoir_save_idx >= 0) & (self.reservoir_save_idx < self.num_reservoirs)
        ):
            raise ValueError("reservoir_save_idx contains invalid indices")
        return self

    @model_validator(mode="after")
    def validate_reservoir_catchment_idx(self) -> Self:
        if not torch.all(
            (self.reservoir_catchment_idx >= 0) & (self.reservoir_catchment_idx < self.num_catchments)
        ):
            raise ValueError("reservoir_catchment_idx contains invalid indices")
        return self

    @model_validator(mode="after")
    def validate_num_reservoirs(self) -> Self:
        if self.num_reservoirs < 0:
            raise ValueError("num_reservoirs must be non-negative")
        return self

    @model_validator(mode="after")
    def validate_reservoir_volumes(self) -> Self:
        total_operational = self.conservation_volume + self.emergency_volume
        if torch.any(total_operational > self.reservoir_capacity):
            raise ValueError("Conservation + emergency volume exceeds capacity")
        if torch.any(self.reservoir_storage > self.reservoir_capacity):
            raise ValueError("Reservoir storage exceeds capacity")
        return self
