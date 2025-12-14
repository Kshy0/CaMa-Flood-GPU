# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Bifurcation module for CaMa-Flood-GPU using TensorField / computed_tensor_field
for concise tensor metadata.
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


def BifurcationField(
    description: str,
    shape: Tuple[str, ...] = ("num_bifurcation_paths",),
    dtype: Literal["float", "int", "bool"] = "float",
    group_by: Optional[str] = "bifurcation_basin_id",
    save_idx: Optional[str] = "bifurcation_path_save_idx",
    save_coord: Optional[str] = "bifurcation_path_save_id",
    dim_coords: Optional[str] = "bifurcation_path_id",
    intermediate: bool = False,
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
        intermediate=intermediate,
        **kwargs
    )

def computed_bifurcation_field(
    description: str,
    shape: Tuple[str, ...] = ("num_bifurcation_paths",),
    dtype: Literal["float", "int", "bool"] = "float",
    save_idx: Optional[str] = "bifurcation_path_save_idx",
    save_coord: Optional[str] = "bifurcation_path_save_id",
    dim_coords: Optional[str] = "bifurcation_path_id",
    intermediate: bool = False,
    **kwargs
):
    return computed_tensor_field(
        description=description,
        shape=shape,
        dtype=dtype,
        save_idx=save_idx,
        save_coord=save_coord,
        dim_coords=dim_coords,
        intermediate=intermediate,
        **kwargs
    )

class BifurcationModule(AbstractModule):
    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    module_name: ClassVar[str] = "bifurcation"
    description: ClassVar[str] = "Bifurcation flow module with multi-level channel calculations"
    dependencies: ClassVar[list] = ["base"]

    base: Optional[BaseModule] = Field(default=None, exclude=True, description="Reference to BaseModule")

    # ------------------------------------------------------------------ #
    # IDs
    # ------------------------------------------------------------------ #
    bifurcation_path_id: torch.Tensor = BifurcationField(
        description="Unique ID for each bifurcation path (used to distinguish paths for identification in saved results)",
        dtype="int",
    )

    bifurcation_basin_id: torch.Tensor = BifurcationField(
        description="Basin ID for each bifurcation path (used to group paths by basin)",
        dtype="int",
    )

    # ------------------------------------------------------------------ #
    # Bifurcation topology
    # ------------------------------------------------------------------ #
    bifurcation_save_mask: Optional[torch.Tensor] = BifurcationField(
        description="Mask of bifurcation paths to save in output",
        dtype="bool",
        default=None,
    )

    bifurcation_catchment_id: torch.Tensor = BifurcationField(
        description="Upstream catchment IDs for each bifurcation path",
        dtype="int",
    )

    bifurcation_downstream_id: torch.Tensor = BifurcationField(
        description="Downstream catchment IDs for each bifurcation path",
        dtype="int",
    )

    # ------------------------------------------------------------------ #
    # Channel properties
    # ------------------------------------------------------------------ #
    bifurcation_manning: torch.Tensor = BifurcationField(
        description="Manning roughness coefficients for bifurcation channels (-)",
        shape=("num_bifurcation_paths", "num_bifurcation_levels"),
        default=0.03,
    )

    bifurcation_width: torch.Tensor = BifurcationField(
        description="Channel widths by path and level (m)",
        shape=("num_bifurcation_paths", "num_bifurcation_levels"),
    )

    bifurcation_length: torch.Tensor = BifurcationField(
        description="Channel lengths for each bifurcation path (m)",
    )

    bifurcation_elevation: torch.Tensor = BifurcationField(
        description="Channel-bed elevations by path and level (m a.s.l.)",
        shape=("num_bifurcation_paths", "num_bifurcation_levels"),
    )

    # ------------------------------------------------------------------ #
    # State variables
    # ------------------------------------------------------------------ #
    bifurcation_outflow: torch.Tensor = BifurcationField(
        description="Outflow through each bifurcation path & level (m³ s⁻¹)",
        shape=("num_bifurcation_paths", "num_bifurcation_levels"),
        default=0,
    )

    bifurcation_cross_section_depth: torch.Tensor = BifurcationField(
        description="Cross-sectional water depth (m)",
        shape=("num_bifurcation_paths", "num_bifurcation_levels"),
        default=0,
    )

    # ------------------------------------------------------------------ #
    # Computed tensor indices
    # ------------------------------------------------------------------ #
    @computed_bifurcation_field(
        description="Indices of upstream catchments for each bifurcation path",
        dtype="int",
    )
    @cached_property
    def bifurcation_catchment_idx(self) -> torch.Tensor:
        return find_indices_in_torch(self.bifurcation_catchment_id, self.base.catchment_id)

    @computed_bifurcation_field(
        description="Indices of downstream catchments for each bifurcation path",
        dtype="int",
    )
    @cached_property
    def bifurcation_downstream_idx(self) -> torch.Tensor:
        return find_indices_in_torch(self.bifurcation_downstream_id, self.base.catchment_id)

    @computed_bifurcation_field(
        description="Indices of bifurcation paths to save in output",
        shape=("num_saved_bifurcation_paths",),
        dtype="int",
    )
    @cached_property
    def bifurcation_path_save_idx(self) -> torch.Tensor:
        if self.bifurcation_save_mask is None:
            return torch.arange(self.num_bifurcation_paths, dtype=torch.int64, device=self.device)
        return torch.nonzero(self.bifurcation_save_mask, as_tuple=False).squeeze(-1)
    
    @computed_bifurcation_field(
        description="Indices of bifurcation paths to save in output",
        shape=("num_saved_bifurcation_paths",),
        dtype="int",
    )
    @cached_property
    def bifurcation_path_save_id(self) -> torch.Tensor:
        if self.bifurcation_path_save_idx is None:
            return self.bifurcation_path_id
        return self.bifurcation_path_id[self.bifurcation_path_save_idx]

    # ------------------------------------------------------------------ #
    # Computed scalar dimensions
    # ------------------------------------------------------------------ #
    @computed_field(
        description="Number of paths saved in output."
    )
    @cached_property
    def num_saved_bifurcation_paths(self) -> int:
        return len(self.bifurcation_path_save_idx)

    @computed_field(
        description="Total number of bifurcation paths."
    )
    @cached_property
    def num_bifurcation_paths(self) -> int:
        return self.bifurcation_width.shape[0]

    @computed_field(
        description="Number of levels in each bifurcation path."
    )
    @cached_property
    def num_bifurcation_levels(self) -> int:
        return self.bifurcation_width.shape[1]

    @computed_field(
        description="Total number of catchments in the domain."
    )
    @cached_property
    def num_catchments(self) -> int:
        return self.base.catchment_id.shape[0]

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @model_validator(mode="after")
    def validate_bifurcation_path_save_idx(self) -> Self:
        if not torch.all(
            (self.bifurcation_path_save_idx >= 0)
            & (self.bifurcation_path_save_idx < self.num_bifurcation_paths)
        ):
            raise ValueError("bifurcation_path_save_idx contains invalid indices")
        return self

    @model_validator(mode="after")
    def validate_bifurcation_catchment_idx(self) -> Self:
        if not torch.all(
            (self.bifurcation_catchment_idx >= 0)
            & (self.bifurcation_catchment_idx < self.num_catchments)
        ):
            raise ValueError("bifurcation_catchment_idx contains invalid indices")
        return self

    @model_validator(mode="after")
    def validate_bifurcation_downstream_idx(self) -> Self:
        if not torch.all(
            (self.bifurcation_downstream_idx >= 0)
            & (self.bifurcation_downstream_idx < self.num_catchments)
        ):
            raise ValueError("bifurcation_downstream_idx contains invalid indices")
        return self

    @model_validator(mode="after")
    def validate_num_bifurcation_paths(self) -> Self:
        if self.num_bifurcation_paths <= 0:
            raise ValueError("num_bifurcation_paths must be positive")
        return self
