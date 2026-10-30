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
from hydroforge.contracts import opt
from hydroforge.model import (
    AbstractModule,
    CoordinateField,
    ReferenceField,
    ReferenceIndexField,
    TensorField,
    computed_tensor_field,
    kernel_field,
    module_ref,
)
from pydantic import computed_field, model_validator

from cmfgpu.modules.base import BaseModule


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

    reservoir_upstream_area: torch.Tensor | None = ReservoirField(
        description="Drainage area upstream of the dam; required by the Hanazaki (2022) scheme",
        units="m2",
        gt=0,
        depends_on=opt("reservoir.release_scheme") == "hanazaki_2022",
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
    def is_reservoir(self) -> torch.Tensor:
        mask = torch.zeros(
            self.base.catchment_id.shape[0],
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
    def is_dam_related(self) -> torch.Tensor:
        downstream_is_dam = self.is_reservoir[self.base.downstream_idx]
        idx = torch.arange(
            self.base.catchment_id.shape[0], device=self.base.catchment_id.device
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
    def flood_volume(self) -> torch.Tensor:
        """Invert the configured emergency-volume fraction of the flood pool."""
        excess = self.emergency_volume - self.conservation_volume
        fraction = self.options.value("reservoir.flood_volume_fraction")
        return excess / excess.new_tensor(fraction)

    @computed_reservoir_field(
        description="Regulation threshold above conservation, using the configured flood-pool fraction",
        units="m3",
        depends_on=opt("reservoir.release_scheme") == "yamazaki_funato",
    )
    def adjustment_volume(self) -> torch.Tensor | None:
        fraction = torch.tensor(
            self.options.value("reservoir.adjustment_volume_fraction"), dtype=torch.float32
        ).item()
        return self.conservation_volume + self.flood_volume * fraction

    @computed_reservoir_field(
        description="Normal release: raw Qn for H22, storage-limited supply for Yamazaki & Funato",
        units="m3 s-1",
    )
    def effective_normal_outflow(self) -> torch.Tensor:
        if self.uses_h22:
            return self.normal_outflow
        else:
            normal_outflow = self.normal_outflow
            annual_inflow = normal_outflow * (365.0 * 24 * 60 * 60)
            fraction = torch.tensor(
                self.options.value("reservoir.dry_storage_fraction"), dtype=torch.float32
            ).item()
            dry_storage = (
                self.conservation_volume * fraction
                + annual_inflow * self.options.value("reservoir.dry_inflow_fraction")
            )
            dry_days = self.options.value("reservoir.dry_season_days")
            dry_outflow = dry_storage / dry_storage.new_tensor(dry_days * 24 * 60 * 60)
            factor = self.options.value("reservoir.normal_outflow_factor")
            return torch.minimum(normal_outflow, dry_outflow) * factor

    @computed_reservoir_field(
        description="Regulated outflow rate: Qa = (modified_Qn + Qf) * 0.5",
        units="m3 s-1",
        depends_on=opt("reservoir.release_scheme") == "yamazaki_funato",
    )
    def adjustment_outflow(self) -> torch.Tensor | None:
        return (self.effective_normal_outflow + self.flood_control_outflow) * 0.5

    @kernel_field
    def uses_h22(self) -> bool:
        """Uniform construction-time release choice, consumed by kernel specs."""
        return self.options.option_code("reservoir.release_scheme") == 1

    @kernel_field
    def uses_yamazaki_funato(self) -> bool:
        return self.options.option_code("reservoir.release_scheme") == 0

    @kernel_field
    def release_exponent(self) -> float:
        return self.options.value("reservoir.release_exponent")

    @computed_reservoir_field(
        description="Hanazaki (2022) normal volume as a configured conservation-volume fraction",
        units="m3",
        depends_on=opt("reservoir.release_scheme") == "hanazaki_2022",
    )
    def normal_volume(self) -> torch.Tensor | None:
        return self.conservation_volume * self.options.value("reservoir.normal_volume_fraction")

    @computed_reservoir_field(
        description="Hanazaki (2022) flood-inflow share using the configured flood-storage depth",
        depends_on=opt("reservoir.release_scheme") == "hanazaki_2022",
    )
    def flood_inflow_share(self) -> torch.Tensor | None:
        # CaMa-Flood v4 stores FldVol/area in R_VolUpa(NDAM) instead of
        # R_VolUpa(IDAM), so every dam but the last reads an unset ratio; this
        # is the per-dam ratio of Hanazaki et al. (2022).
        area = self.reservoir_upstream_area
        depth = self.flood_volume / area
        threshold = torch.tensor(
            self.options.value("reservoir.flood_storage_depth"), dtype=torch.float32
        ).item()
        return torch.clamp(1 - depth / threshold, min=0)

    def fill_dams(self) -> None:
        """Fill cold-start reservoirs to conservation volume when below it,
        leaving their floodplains empty."""
        base = self.base
        idx = self.reservoir_catchment_idx
        river = base.river_storage[..., idx]
        flood = base.flood_storage[..., idx]
        conservation = self.conservation_volume.to(river.dtype)
        fill = (river + flood) < conservation
        base.river_storage[..., idx] = torch.where(fill, conservation, river)
        base.flood_storage[..., idx] = torch.where(fill, torch.zeros_like(flood), flood)
        count = int(fill.sum().item())
        if count:
            self._emit(
                "info",
                "reservoir.storage_initialized",
                "Initialized dam cells to conservation storage",
                cells=count,
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
        if self.uses_yamazaki_funato:
            adjustment = self.adjustment_volume
            if torch.any(
                ~torch.isfinite(adjustment)
                | (adjustment <= self.conservation_volume)
                | (adjustment >= self.emergency_volume)
            ).item():
                raise ValueError("Adjustment volume must remain strictly between conservation and emergency")
        return self
