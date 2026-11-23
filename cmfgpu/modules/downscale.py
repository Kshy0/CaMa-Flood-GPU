# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""High-resolution flood downscaling modules restoring Fortran CaMa-Flood projection schemes."""

from __future__ import annotations

from typing import ClassVar, Self

import torch
from hydroforge.model import (
    AbstractModule,
    CoordinateField,
    ReferenceField,
    ReferenceIndexField,
    SelectionField,
    TensorField,
    computed_field,
    computed_tensor_field,
    kernel_field,
    module_ref,
)
from pydantic import model_validator


class DownscaleBaseModule(AbstractModule):
    """Core high-resolution pixel topology and flood-depth projection module."""

    module_name: ClassVar[str] = "base"
    description: ClassVar[str] = (
        "High-resolution flood-depth projection on sub-grid topography"
    )

    @kernel_field
    def permanent_water_depth(self) -> float:
        return self.options.value("permanent_water_depth")

    @kernel_field
    def protected_depth_offset(self) -> float:
        return self.options.value("protected_depth_offset")

    catchment_id: torch.Tensor = CoordinateField(
        description="Unique identifier of each coarse catchment",
        dtype="int",
        shape=("num_catchments",),
    )

    catchment_basin_id: torch.Tensor = TensorField(
        description="Basin identifier for distributed catchment sharding",
        shape=("num_catchments",),
        dtype="int",
        dim_coords="catchment_id",
        category="topology",
        mode="cpu",
    )

    flood_depth: torch.Tensor = TensorField(
        description="Coarse catchment flood depth above river bankfull",
        shape=("num_catchments",),
        dtype="float",
        dim_coords="catchment_id",
        category="forcing",
        units="m",
        output="disabled",
        default=0,
    )

    pixel_catchment_id: torch.Tensor = ReferenceField(
        description="Parent coarse catchment identifier of each high-resolution pixel",
        dtype="int",
        shape=("num_pixels",),
        dim_coords="pixel_id",
        references="catchment_id",
    )

    pixel_id: torch.Tensor = CoordinateField(
        description="Unique flat raster index of each valid high-resolution catchment pixel",
        dtype="int",
        shape=("num_pixels",),
        partition_by="pixel_catchment_id",
    )

    output_pixel_id: torch.Tensor | None = SelectionField(
        description="High-resolution pixel identifiers selected for output",
        dtype="int",
        shape=("num_output_pixels",),
        selects="pixel_id",
        default=None,
    )

    pixel_flood_diff: torch.Tensor = TensorField(
        description="High-resolution pixel height above channel bankfull",
        shape=("num_pixels",),
        dtype="float",
        dim_coords="pixel_id",
        category="param",
        units="m",
    )

    pixel_is_permanent_water: torch.Tensor = TensorField(
        description="Mask of permanent river-channel pixels derived from high-resolution width",
        shape=("num_pixels",),
        dtype="bool",
        dim_coords="pixel_id",
        category="param",
        default=False,
    )

    pixel_catchment_idx = ReferenceIndexField("pixel_catchment_id")

    @computed_field(description="Coarse catchments owned by this rank")
    def num_catchments(self) -> int:
        return self.owned_rows("catchment_id")

    @computed_field(description="High-resolution pixels owned by this rank")
    def num_pixels(self) -> int:
        return self.pixel_id.shape[0]

    @computed_field(description="Number of high-resolution pixels selected for output")
    def num_output_pixels(self) -> int:
        if self.output_pixel_id is None:
            return self.num_pixels
        return self.output_pixel_id.shape[0]

    @computed_tensor_field(
        description="Downscaled flood water depth on high-resolution pixels",
        shape=("num_pixels",),
        dtype="float",
        dim_coords="pixel_id",
        category="state",
        units="m",
    )
    def pixel_flood_depth(self) -> torch.Tensor:
        shape = (
            (self.ensemble_size, self.num_pixels)
            if self.ensemble_size is not None
            else (self.num_pixels,)
        )
        return torch.zeros(shape, dtype=self.precision, device=self.device)


class DownscaleLeveeModule(AbstractModule):
    """Levee-aware high-resolution flood-depth projection module."""

    module_name: ClassVar[str] = "levee"
    description: ClassVar[str] = (
        "Levee-protected high-resolution flood-depth projection"
    )
    conflicts: ClassVar[tuple[str, ...]] = ("duration",)
    base = module_ref(DownscaleBaseModule)

    levee_catchment_id: torch.Tensor = ReferenceField(
        description="Catchment identifier hosting each levee",
        dtype="int",
        shape=("num_levees",),
        dim_coords="levee_id",
        references="catchment_id",
        is_key=True,
    )

    levee_id: torch.Tensor = CoordinateField(
        description="Unique identifier for each levee",
        dtype="int",
        shape=("num_levees",),
        partition_by="levee_catchment_id",
    )

    levee_fraction: torch.Tensor = TensorField(
        description="Relative unprotected distance fraction between river and levee",
        shape=("num_levees",),
        dtype="float",
        dim_coords="levee_id",
        category="param",
        units="1",
        ge=0,
        le=1,
    )

    pixel_elevation_percentile: torch.Tensor = TensorField(
        description="Intra-catchment pixel elevation rank from zero to one hundred",
        shape=("base.num_pixels",),
        dtype="float",
        dim_coords="base.pixel_id",
        category="param",
        units="%",
        ge=0,
        le=100,
    )

    protected_depth: torch.Tensor = TensorField(
        description="Coarse catchment flood depth on the levee-protected land side",
        shape=("base.num_catchments",),
        dtype="float",
        dim_coords="base.catchment_id",
        category="forcing",
        units="m",
        output="disabled",
        default=0,
    )

    levee_catchment_idx = ReferenceIndexField("levee_catchment_id")

    @computed_field(description="Total number of levees owned by this rank")
    def num_levees(self) -> int:
        return self.levee_catchment_id.shape[0]

    @computed_tensor_field(
        description="Mask of high-resolution pixels located inside levee-protected zones",
        shape=("base.num_pixels",),
        dtype="bool",
        dim_coords="base.pixel_id",
        category="derived_param",
    )
    def pixel_is_protected(self) -> torch.Tensor:
        num_catchments = self.base.catchment_id.shape[0]
        levee_idx = self.levee_catchment_idx.long()
        pixel_cidx = self.base.pixel_catchment_idx.long()
        has_levee = torch.zeros(
            num_catchments, dtype=torch.bool, device=self.device
        )
        has_levee[levee_idx] = True

        fraction = self.levee_fraction
        catchment_fraction = torch.zeros(
            (*fraction.shape[:-1], num_catchments),
            dtype=fraction.dtype,
            device=self.device,
        )
        catchment_fraction[..., levee_idx] = fraction
        pixel_threshold = catchment_fraction[..., pixel_cidx] * 100.0
        return has_levee[pixel_cidx] & (
            self.pixel_elevation_percentile > pixel_threshold
        )


class DownscaleTributaryModule(AbstractModule):
    """Sub-grid tributary interpolation and hillslope backwater module."""

    module_name: ClassVar[str] = "tributary"
    description: ClassVar[str] = (
        "Sub-grid tributary depth interpolation and backwater smoothing"
    )
    conflicts: ClassVar[tuple[str, ...]] = ("duration",)
    base = module_ref(DownscaleBaseModule)

    tributary_pixel_id: torch.Tensor = CoordinateField(
        description="Identifiers of high-resolution sub-grid tributary channel pixels",
        dtype="int",
        shape=("num_tributary_pixels",),
        references="pixel_id",
    )

    tributary_target_id: torch.Tensor = ReferenceField(
        description="Target tributary pixel identifier for each interpolation link",
        dtype="int",
        shape=("num_tributary_links",),
        dim_coords="tributary_link_id",
        references="tributary_pixel_id",
    )

    tributary_link_id: torch.Tensor = CoordinateField(
        description="Unique identifier of each sub-grid tributary interpolation link",
        dtype="int",
        shape=("num_tributary_links",),
        partition_by="tributary_target_id",
    )

    tributary_upstream_pixel_id: torch.Tensor = ReferenceField(
        description="Upstream catchment outlet pixel identifier for each tributary link",
        dtype="int",
        shape=("num_tributary_links",),
        dim_coords="tributary_link_id",
        references="pixel_id",
    )

    tributary_downstream_pixel_id: torch.Tensor = ReferenceField(
        description="Downstream mainstem confluence pixel identifier for each tributary link",
        dtype="int",
        shape=("num_tributary_links",),
        dim_coords="tributary_link_id",
        references="pixel_id",
    )

    tributary_chain_step: torch.Tensor = TensorField(
        description="One-based pixel step count from upstream outlet along the tributary chain",
        shape=("num_tributary_links",),
        dtype="float",
        dim_coords="tributary_link_id",
        category="param",
        units="1",
        gt=0,
    )

    tributary_chain_length: torch.Tensor = TensorField(
        description="Total pixel step length from upstream outlet to mainstem confluence",
        shape=("num_tributary_links",),
        dtype="float",
        dim_coords="tributary_link_id",
        category="param",
        units="1",
        gt=1,
    )

    backwater_pixel_id: torch.Tensor = CoordinateField(
        description="Identifiers of off-channel pixels eligible for tributary backwater",
        dtype="int",
        shape=("num_backwater_pixels",),
        references="pixel_id",
    )

    backwater_target_pixel_id: torch.Tensor = ReferenceField(
        description="Terminal channel pixel identifier governing backwater elevation",
        dtype="int",
        shape=("num_backwater_pixels",),
        dim_coords="backwater_pixel_id",
        references="pixel_id",
    )

    backwater_floor_elevation: torch.Tensor = TensorField(
        description="Maximum intermediate sill height above channel along the backwater path",
        shape=("num_backwater_pixels",),
        dtype="float",
        dim_coords="backwater_pixel_id",
        category="param",
        units="m",
    )

    tributary_pixel_idx = ReferenceIndexField("tributary_pixel_id")
    tributary_target_idx = ReferenceIndexField("tributary_target_id")
    tributary_upstream_pixel_idx = ReferenceIndexField("tributary_upstream_pixel_id")
    tributary_downstream_pixel_idx = ReferenceIndexField("tributary_downstream_pixel_id")
    backwater_pixel_idx = ReferenceIndexField("backwater_pixel_id")
    backwater_target_pixel_idx = ReferenceIndexField("backwater_target_pixel_id")

    @computed_field(description="Number of sub-grid tributary pixels owned by this rank")
    def num_tributary_pixels(self) -> int:
        return self.tributary_pixel_id.shape[0]

    @computed_field(description="Number of sub-grid tributary links owned by this rank")
    def num_tributary_links(self) -> int:
        return self.tributary_link_id.shape[0]

    @computed_field(description="Number of off-channel backwater pixels owned by this rank")
    def num_backwater_pixels(self) -> int:
        return self.backwater_pixel_id.shape[0]

    @computed_tensor_field(
        description="Flag indicating catchments whose tributary depth increased during smoothing",
        shape=("base.num_catchments",),
        dtype="idx",
        dim_coords="base.catchment_id",
        category="state",
        output="disabled",
    )
    def catchment_tributary_modified(self) -> torch.Tensor:
        shape = (
            (self.ensemble_size, self.base.num_catchments)
            if self.ensemble_size is not None
            else (self.base.num_catchments,)
        )
        return torch.zeros(shape, dtype=torch.int32, device=self.device)

    @model_validator(mode="after")
    def validate_tributary_chains(self) -> Self:
        if self.num_tributary_links > 0 and torch.any(
            self.tributary_chain_step >= self.tributary_chain_length
        ).item():
            raise ValueError(
                "tributary_chain_step must be strictly less than tributary_chain_length"
            )
        return self


class DownscaleDurationModule(AbstractModule):
    """High-resolution inundation-duration projection module."""

    module_name: ClassVar[str] = "duration"
    description: ClassVar[str] = (
        "High-resolution inundation-duration projection from depth-binned histograms"
    )
    conflicts: ClassVar[tuple[str, ...]] = ("levee", "tributary")
    base = module_ref(DownscaleBaseModule)

    @kernel_field
    def duration_depth_interval(self) -> float:
        return self.options.value("duration_depth_interval")

    @kernel_field
    def permanent_water_duration(self) -> float:
        return self.options.value("permanent_water_duration")

    duration_bin_id: torch.Tensor = CoordinateField(
        description="Zero-based index of each coarse depth-duration histogram bin",
        dtype="int",
        shape=("num_duration_bins",),
        replicated=True,
    )

    flood_duration_table: torch.Tensor = TensorField(
        description="Coarse inundation duration per catchment and flood-depth bin",
        shape=("base.num_catchments", "num_duration_bins"),
        dtype="float",
        dim_coords="base.catchment_id",
        category="forcing",
        units="day",
        output="disabled",
        default=0,
    )

    @computed_field(description="Number of depth bins in the coarse duration histogram")
    def num_duration_bins(self) -> int:
        return self.duration_bin_id.shape[0]

    @computed_tensor_field(
        description="Downscaled inundation duration on high-resolution pixels",
        shape=("base.num_pixels",),
        dtype="float",
        dim_coords="base.pixel_id",
        category="state",
        units="day",
    )
    def pixel_flood_duration(self) -> torch.Tensor:
        shape = (
            (self.ensemble_size, self.base.num_pixels)
            if self.ensemble_size is not None
            else (self.base.num_pixels,)
        )
        return torch.zeros(shape, dtype=self.precision, device=self.device)

    @model_validator(mode="after")
    def validate_duration_bins(self) -> Self:
        if self.num_duration_bins < 1:
            raise ValueError("duration_bin_id must contain at least one depth bin")
        return self
