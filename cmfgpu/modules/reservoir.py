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
from typing import ClassVar, Literal, Self

import torch
from hydroforge.model import (
    AbstractModule,
    CoordinateField,
    ReferenceField,
    ReferenceIndexField,
    TensorField,
    computed_tensor_field,
    module_ref,
    optional_module_ref,
)
from pydantic import computed_field, model_validator

from cmfgpu.config import DISABLED_BIFURCATION_ELEVATION
from cmfgpu.modules.base import BaseModule
from cmfgpu.modules.bifurcation import BifurcationModule


def ReservoirField(
    description: str,
    shape: tuple[str, ...] = ("num_reservoirs",),
    dtype: Literal["float", "int", "idx", "bool"] = "float",
    dim_coords: str | None = "reservoir_id",
    category: Literal["topology", "param"] = "param",
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


def computed_reservoir_field(
    description: str,
    shape: tuple[str, ...] = ("num_reservoirs",),
    dtype: Literal["float", "int", "idx", "bool"] = "float",
    dim_coords: str | None = "reservoir_id",
    category: Literal[
        "topology", "derived_param", "state", "virtual"
    ] = "derived_param",
    expr: str | None = None,
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


class ReservoirModule(AbstractModule):
    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    module_name: ClassVar[str] = "reservoir"
    description: ClassVar[str] = (
        "Reservoir operation module with storage and outflow regulation"
    )
    base = module_ref(BaseModule)
    bifurcation = optional_module_ref(BifurcationModule)

    # ------------------------------------------------------------------ #
    # Reservoir topology
    # ------------------------------------------------------------------ #
    reservoir_catchment_id: torch.Tensor = ReferenceField(
        description="Catchment ID hosting each reservoir",
        dtype="int",
        shape=("num_reservoirs",),
        dim_coords="reservoir_id",
        references="catchment_id",
        is_key=True,
    )

    reservoir_id: torch.Tensor = CoordinateField(
        description="Unique ID for each reservoir",
        dtype="int",
        shape=("num_reservoirs",),
        partition_by="reservoir_catchment_id",
    )

    # ------------------------------------------------------------------ #
    # Physical properties
    # ------------------------------------------------------------------ #
    reservoir_capacity: torch.Tensor = ReservoirField(
        description="Maximum storage capacity",
        units="m3",
        gt=0,
    )

    conservation_volume: torch.Tensor = ReservoirField(
        description="Conservation storage volume",
        units="m3",
        gt=0,
    )

    emergency_volume: torch.Tensor = ReservoirField(
        description="Emergency storage volume",
        units="m3",
    )

    normal_outflow: torch.Tensor = ReservoirField(
        description="Normal outflow rate",
        units="m3 s-1",
        ge=0,
    )

    flood_control_outflow: torch.Tensor = ReservoirField(
        description="Flood-control outflow rate",
        units="m3 s-1",
        ge=0,
    )

    reservoir_area: torch.Tensor = ReservoirField(
        description="Surface area at normal water level",
        units="m2",
        gt=0,
    )

    # ------------------------------------------------------------------ #
    # Computed tensor fields
    # ------------------------------------------------------------------ #
    @computed_field(description="Total number of reservoirs")
    @cached_property
    def num_reservoirs(self) -> int:
        return self.reservoir_catchment_id.shape[0]

    reservoir_catchment_idx = ReferenceIndexField("reservoir_catchment_id")

    @computed_tensor_field(
        description="Boolean mask for reservoir catchments",
        shape=("base.num_catchments",),
        dtype="bool",
        dim_coords="base.catchment_id",
        category="topology",
    )
    @cached_property
    def is_reservoir(self) -> torch.Tensor:
        mask = torch.zeros(
            self.base.num_catchments,
            dtype=torch.bool,
            device=self.base.catchment_id.device,
        )
        mask[self.reservoir_catchment_idx] = True
        return mask

    @computed_tensor_field(
        description="Mask for reservoir cells and their immediate upstream cells",
        shape=("base.num_catchments",),
        dtype="bool",
        dim_coords="base.catchment_id",
        category="topology",
    )
    @cached_property
    def is_dam_related(self) -> torch.Tensor:
        downstream_is_dam = self.is_reservoir[self.base.downstream_idx]
        idx = torch.arange(
            self.base.num_catchments, device=self.base.catchment_id.device
        )
        not_mouth = self.base.downstream_idx != idx
        upstream = (~self.is_reservoir) & not_mouth & downstream_is_dam
        return self.is_reservoir | upstream

    @computed_tensor_field(
        description="Mask for immediate upstream-of-reservoir cells",
        shape=("base.num_catchments",),
        dtype="bool",
        dim_coords="base.catchment_id",
        category="topology",
    )
    @cached_property
    def is_dam_upstream(self) -> torch.Tensor:
        return self.is_dam_related & ~self.is_reservoir

    reservoir_total_inflow: torch.Tensor = TensorField(
        description="Accumulated reservoir total inflow from upstream",
        units="m3 s-1",
        shape=("base.num_catchments",),
        dtype="hpfloat",
        dim_coords="base.catchment_id",
        category="init_state",
        default=0,
    )

    # ------------------------------------------------------------------ #
    # Computed tensors (operations)
    # ------------------------------------------------------------------ #
    @computed_reservoir_field(
        description="Flood control storage capacity, derived from emergency and conservation volumes",
        units="m3",
    )
    @cached_property
    def flood_volume(self) -> torch.Tensor:
        """FldVol = (EmeVol - ConVol) / 0.95, inverse of EmeVol = ConVol + FldVol * 0.95"""
        excess = self.emergency_volume - self.conservation_volume
        return excess / excess.new_tensor(0.95)

    @computed_reservoir_field(
        description="Volume threshold triggering regulation: AdjVol = ConVol + FldVol * 0.1",
        units="m3",
    )
    @cached_property
    def adjustment_volume(self) -> torch.Tensor:
        tenth = torch.tensor(0.1, dtype=torch.float32).item()
        return self.conservation_volume + self.flood_volume * tenth

    @computed_reservoir_field(
        description="Effective normal outflow after Yamazaki & Funato modification. Qn = min(Qn, Qsto) * 1.5 where Qsto = (ConVol*0.7 + Vyr/4) / (180 days)",
        units="m3 s-1",
    )
    @cached_property
    def effective_normal_outflow(self) -> torch.Tensor:
        normal_outflow = self.normal_outflow
        annual_inflow = normal_outflow * (365.0 * 24 * 60 * 60)
        seven_tenths = torch.tensor(0.7, dtype=torch.float32).item()
        dry_storage = self.conservation_volume * seven_tenths + annual_inflow / 4.0
        dry_outflow = dry_storage / dry_storage.new_tensor(180.0 * 24 * 60 * 60)
        return torch.minimum(normal_outflow, dry_outflow) * 1.5

    @computed_reservoir_field(
        description="Regulated outflow rate: Qa = (modified_Qn + Qf) * 0.5",
        units="m3 s-1",
    )
    @cached_property
    def adjustment_outflow(self) -> torch.Tensor:
        return (self.effective_normal_outflow + self.flood_control_outflow) * 0.5

    def initialize_dam_storage(self) -> int:
        """Raise cold-start dam storage to the conservation volume."""
        idx = self.reservoir_catchment_idx
        river = self.base.river_storage[..., idx]
        flood = self.base.flood_storage[..., idx]
        conservation = self.conservation_volume.to(river.dtype)
        update = (river + flood) < conservation
        if not update.any():
            return 0
        self.base.river_storage[..., idx] = torch.where(
            update,
            conservation,
            river,
        )
        self.base.flood_storage[..., idx] = torch.where(
            update,
            torch.zeros_like(flood),
            flood,
        )
        return int(update.sum().item())

    def mask_bifurcation_paths(self) -> int:
        """Disable paths whose upstream or downstream cell is dam-related."""
        bifurcation = self.bifurcation
        if bifurcation is None:
            return 0
        masked = (
            self.is_dam_related[bifurcation.bifurcation_catchment_idx]
            | self.is_dam_related[bifurcation.bifurcation_downstream_idx]
        )
        bifurcation.bifurcation_elevation[..., masked, :] = (
            DISABLED_BIFURCATION_ELEVATION
        )
        return int(masked.sum().item())

    def initialize_state(self, *, supplied: frozenset[str]) -> None:
        """Apply reservoir-owned cold starts without replacing restart state.

        CMF_DAMOUT_INIT raises dam storage to ConVol only without restart
        data.  Every checkpoint carries the reservoir's restart state
        ``reservoir_total_inflow``, so its presence among the ``supplied``
        inputs marks a restart.  Dam-related bifurcation paths close always.
        """
        if "reservoir_total_inflow" not in supplied:
            fixed = self.initialize_dam_storage()
            if fixed:
                self._emit(
                    "info",
                    "reservoir.storage_initialized",
                    "Initialized dam cells to conservation storage",
                    cells=fixed,
                )
        masked = self.mask_bifurcation_paths()
        if masked:
            self._emit(
                "info",
                "reservoir.bifurcation_masked",
                "Masked dam-related bifurcation paths",
                paths=masked,
            )

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @model_validator(mode="after")
    def validate_reservoir_volumes(self) -> Self:
        # Both interpolation regimes divide by the gap above conservation.
        if torch.any(self.emergency_volume <= self.conservation_volume).item():
            raise ValueError("Emergency volume must exceed conservation volume")
        if torch.any(self.emergency_volume > self.reservoir_capacity):
            raise ValueError("Emergency volume exceeds reservoir capacity")
        adjustment = self.adjustment_volume
        if torch.any(
            ~torch.isfinite(adjustment)
            | (adjustment <= self.conservation_volume)
            | (adjustment >= self.emergency_volume)
        ).item():
            raise ValueError("Adjustment volume must remain strictly between conservation and emergency")
        return self
