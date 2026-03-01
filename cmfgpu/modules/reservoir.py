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
from pydantic import Field, model_validator

from cmfgpu.modules.abstract_module import (AbstractModule, TensorField,
                                            computed_tensor_field)
from cmfgpu.modules.base import BaseModule
from cmfgpu.utils import find_indices_in_torch


def ReservoirField(
    description: str,
    shape: Tuple[str, ...] = ("base.num_reservoirs",),
    dtype: Literal["float", "int", "idx", "bool"] = "float",
    group_by: Optional[str] = "reservoir_basin_id",
    dim_coords: Optional[str] = "base.reservoir_catchment_id",
    category: Literal["topology", "param"] = "param",
    mode: Literal["device", "cpu", "discard"] = "device",
    **kwargs
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
        **kwargs
    )

def computed_reservoir_field(
    description: str,
    shape: Tuple[str, ...] = ("base.num_reservoirs",),
    dtype: Literal["float", "int", "idx", "bool"] = "float",
    dim_coords: Optional[str] = "base.reservoir_catchment_id",
    category: Literal["topology", "derived_param", "state", "virtual"] = "derived_param",
    expr: Optional[str] = None,
    **kwargs
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
        **kwargs
    )

class ReservoirModule(AbstractModule):
    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    module_name: ClassVar[str] = "reservoir"
    description: ClassVar[str] = "Reservoir operation module with storage and outflow regulation"
    dependencies: ClassVar[list] = ["base"]

    base: Optional[BaseModule] = Field(
        default=None,
        exclude=True,
        description="Reference to BaseModule",
    )

    # ------------------------------------------------------------------ #
    # Reservoir topology
    # ------------------------------------------------------------------ #
    reservoir_id: torch.Tensor = ReservoirField(
        description="Unique ID for each reservoir",
        dtype="int",
        category="topology",
        mode="cpu",
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
    # Computed tensor fields
    # ------------------------------------------------------------------ #
    @computed_reservoir_field(
        description="Catchment-array indices for each reservoir",
        dtype="idx",
        category="topology",
    )
    @cached_property
    def reservoir_catchment_idx(self) -> torch.Tensor:
        return find_indices_in_torch(self.base.reservoir_catchment_id, self.base.catchment_id)

    # ------------------------------------------------------------------ #
    # Computed tensors (operations)
    # ------------------------------------------------------------------ #
    @computed_reservoir_field(
        description="Flood control storage capacity (m³), derived from emergency and conservation volumes",
    )
    @cached_property
    def flood_volume(self) -> torch.Tensor:
        """FldVol = (EmeVol - ConVol) / 0.95, inverse of EmeVol = ConVol + FldVol * 0.95"""
        return (self.emergency_volume - self.conservation_volume) / 0.95

    @computed_reservoir_field(
        description="Volume threshold triggering regulation (m³).  Fortran: AdjVol = ConVol + FldVol * 0.1",
    )
    @cached_property
    def adjustment_volume(self) -> torch.Tensor:
        return self.conservation_volume + self.flood_volume * 0.1

    @computed_reservoir_field(
        description="Effective normal outflow after Yamazaki & Funato modification (m³ s⁻¹). "
                    "Fortran: Qn = min(Qn, Qsto) * 1.5  where Qsto = (ConVol*0.7 + Vyr/4) / (180 days).",
    )
    @cached_property
    def effective_normal_outflow(self) -> torch.Tensor:
        seconds_per_year = 365.0 * 24 * 60 * 60
        seconds_per_180_days = 180.0 * 24 * 60 * 60
        Vyr = self.normal_outflow * seconds_per_year
        Qsto = (self.conservation_volume * 0.7 + Vyr / 4.0) / seconds_per_180_days
        return torch.minimum(self.normal_outflow, Qsto) * 1.5

    @computed_reservoir_field(
        description="Regulated outflow rate (m³ s⁻¹).  Fortran: Qa = (modified_Qn + Qf) * 0.5",
    )
    @cached_property
    def adjustment_outflow(self) -> torch.Tensor:
        return (self.effective_normal_outflow + self.flood_control_outflow) * 0.5

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @model_validator(mode="after")
    def validate_reservoir_catchment_idx(self) -> Self:
        if torch.any(self.reservoir_catchment_idx < 0):
            raise ValueError("reservoir_catchment_id contains entries absent from catchment_id")
        return self

    @model_validator(mode="after")
    def validate_reservoir_volumes(self) -> Self:
        if torch.any(self.emergency_volume > self.reservoir_capacity):
            raise ValueError("Emergency volume exceeds reservoir capacity")
        if torch.any(self.conservation_volume > self.emergency_volume):
            raise ValueError("Conservation volume exceeds emergency volume")
        return self
