"""
Reservoir module for CaMa-Flood-GPU using TensorField / computed_tensor_field helpers.
"""
from __future__ import annotations

from typing import ClassVar, Self, Optional
from functools import cached_property
import torch
from pydantic import Field, model_validator, computed_field

from cmfgpu.modules.abstract_module import (
    AbstractModule,
    TensorField,
    computed_tensor_field,
)
from cmfgpu.utils import find_indices_in_torch


class ReservoirModule(AbstractModule):
    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    module_name: ClassVar[str] = "reservoir"
    description: ClassVar[str] = "Reservoir operation module with storage and outflow regulation"
    dependencies: ClassVar[list] = ["base"]

    # ------------------------------------------------------------------ #
    # Shared catchment IDs
    # ------------------------------------------------------------------ #
    catchment_id: torch.Tensor = TensorField(
        description="Unique ID of each catchment",
        shape=("num_catchments",),
        dtype="int",
        group_by="catchment_basin_id",
    )

    # ------------------------------------------------------------------ #
    # Reservoir topology
    # ------------------------------------------------------------------ #
    reservoir_save_mask: Optional[torch.Tensor] = TensorField(
        description="Mask of reservoirs to include in output",
        shape=("num_reservoirs",),
        dtype="bool",
        group_by="reservoir_basin_id",
        default=None,
    )

    reservoir_catchment_id: torch.Tensor = TensorField(
        description="Catchment IDs where reservoirs are located",
        shape=("num_reservoirs",),
        dtype="int",
        group_by="reservoir_basin_id",
    )

    reservoir_catchment_x: torch.Tensor = TensorField(
        description="Longitude index of reservoir catchment",
        shape=("num_reservoirs",),
        dtype="int",
        group_by="reservoir_basin_id",
        default=None,
    )

    reservoir_catchment_y: torch.Tensor = TensorField(
        description="Latitude index of reservoir catchment",
        shape=("num_reservoirs",),
        dtype="int",
        group_by="reservoir_basin_id",
        default=None,
    )

    # ------------------------------------------------------------------ #
    # Physical properties
    # ------------------------------------------------------------------ #
    reservoir_capacity: torch.Tensor = TensorField(
        description="Maximum storage capacity (m³)",
        shape=("num_reservoirs",),
        group_by="reservoir_basin_id",
    )

    conservation_volume: torch.Tensor = TensorField(
        description="Conservation storage volume (m³)",
        shape=("num_reservoirs",),
        group_by="reservoir_basin_id",
    )

    emergency_volume: torch.Tensor = TensorField(
        description="Emergency storage volume (m³)",
        shape=("num_reservoirs",),
        group_by="reservoir_basin_id",
    )

    normal_outflow: torch.Tensor = TensorField(
        description="Normal outflow rate (m³ s⁻¹)",
        shape=("num_reservoirs",),
        group_by="reservoir_basin_id",
    )

    flood_control_outflow: torch.Tensor = TensorField(
        description="Flood-control outflow rate (m³ s⁻¹)",
        shape=("num_reservoirs",),
        group_by="reservoir_basin_id",
    )

    reservoir_area: torch.Tensor = TensorField(
        description="Surface area at normal water level (m²)",
        shape=("num_reservoirs",),
        group_by="reservoir_basin_id",
    )

    # ------------------------------------------------------------------ #
    # State variables
    # ------------------------------------------------------------------ #
    reservoir_storage: torch.Tensor = TensorField(
        description="Current storage (m³)",
        shape=("num_reservoirs",),
        group_by="reservoir_basin_id",
        default=0,
    )

    reservoir_outflow: torch.Tensor = TensorField(
        description="Current outflow (m³ s⁻¹)",
        shape=("num_reservoirs",),
        group_by="reservoir_basin_id",
        default=0,
    )

    reservoir_water_level: torch.Tensor = TensorField(
        description="Current water level (m)",
        shape=("num_reservoirs",),
        group_by="reservoir_basin_id",
        default=0,
    )

    # ------------------------------------------------------------------ #
    # Computed tensor indices
    # ------------------------------------------------------------------ #
    @computed_tensor_field(
        description="Catchment-array indices for each reservoir",
        shape=("num_reservoirs",),
        dtype="int",
    )
    @cached_property
    def reservoir_catchment_idx(self) -> torch.Tensor:
        return find_indices_in_torch(self.reservoir_catchment_id, self.catchment_id).contiguous()

    @computed_tensor_field(
        description="Indices of reservoirs saved in output",
        shape=("num_saved_reservoirs",),
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
    @computed_tensor_field(
        description="Volume threshold triggering regulation (m³)",
        shape=("num_reservoirs",),
    )
    @cached_property
    def adjustment_volume(self) -> torch.Tensor:
        return self.conservation_volume + self.emergency_volume * 0.1

    @computed_tensor_field(
        description="Regulated outflow rate (m³ s⁻¹)",
        shape=("num_reservoirs",),
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
