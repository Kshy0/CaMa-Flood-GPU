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
from typing import ClassVar, Literal, Optional, Tuple

import torch
from hydroforge.modeling.module import (AbstractModule, CoordinateField,
                                        ReferenceField, ReferenceIndexField,
                                        SelectionField, TensorField,
                                        computed_tensor_field)
from pydantic import Field, computed_field

from cmfgpu.modules.base import BaseModule


def BifurcationField(
    description: str,
    shape: Tuple[str, ...] = ("num_bifurcation_paths",),
    dtype: Literal["float", "int", "idx", "bool"] = "float",
    dim_coords: Optional[str] = "bifurcation_path_id",
    category: Literal["topology", "param", "init_state"] = "param",
    mode: Literal["device", "cpu", "discard"] = "device",
    **kwargs
):
    return TensorField(
        description=description,
        shape=shape,
        dtype=dtype,
        dim_coords=dim_coords,
        category=category,
        mode=mode,
        **kwargs
    )

def computed_bifurcation_field(
    description: str,
    shape: Tuple[str, ...] = ("num_bifurcation_paths",),
    dtype: Literal["float", "int", "idx", "bool"] = "float",
    dim_coords: Optional[str] = "bifurcation_path_id",
    category: Literal["topology", "derived_param", "state", "virtual"] = "derived_param",
    expr: Optional[str] = None,
    **kwargs
):
    return computed_tensor_field(
        description=description,
        shape=shape,
        dtype=dtype,
        dim_coords=dim_coords,
        category=category,
        expr=expr,
        **kwargs
    )

class BifurcationModule(AbstractModule):
    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    module_name: ClassVar[str] = "bifurcation"
    description: ClassVar[str] = "Bifurcation flow module with multi-level channel calculations"
    dependencies: ClassVar[list] = ["base"]

    base: BaseModule = Field(
        exclude=True,
        description="Reference to BaseModule",
    )

    # ------------------------------------------------------------------ #
    # IDs
    # ------------------------------------------------------------------ #
    bifurcation_path_id: torch.Tensor = CoordinateField(
        description="Unique ID for each bifurcation path (used to distinguish paths for identification in saved results)",
        dtype="int",
        shape=("num_bifurcation_paths",),
        partition_by="bifurcation_catchment_id",
    )

    # ------------------------------------------------------------------ #
    # Bifurcation topology
    # ------------------------------------------------------------------ #
    output_bifurcation_path_id: Optional[torch.Tensor] = SelectionField(
        description="Bifurcation path IDs to save in output. "
                    "None means save all paths.",
        dtype="int",
        shape=("num_output_bifurcation_paths",),
        selects="bifurcation_path_id",
        default=None,
    )

    bifurcation_catchment_id: torch.Tensor = ReferenceField(
        description="Upstream catchment IDs for each bifurcation path",
        dtype="int",
        shape=("num_bifurcation_paths",),
        dim_coords="bifurcation_path_id",
        references="catchment_id",
    )

    bifurcation_downstream_id: torch.Tensor = ReferenceField(
        description="Downstream catchment IDs for each bifurcation path",
        dtype="int",
        shape=("num_bifurcation_paths",),
        dim_coords="bifurcation_path_id",
        references="catchment_id",
    )

    # ------------------------------------------------------------------ #
    # Channel properties
    # ------------------------------------------------------------------ #
    bifurcation_manning: torch.Tensor = BifurcationField(
        description="Manning roughness coefficients for bifurcation channels (-)",
        shape=("num_bifurcation_paths", "num_bifurcation_levels"),
        default=0.03,
        category="param",
    )

    bifurcation_width: torch.Tensor = BifurcationField(
        description="Channel widths by path and level (m)",
        shape=("num_bifurcation_paths", "num_bifurcation_levels"),
        category="param",
    )

    bifurcation_length: torch.Tensor = BifurcationField(
        description="Channel lengths for each bifurcation path (m)",
        category="param",
    )

    bifurcation_elevation: torch.Tensor = BifurcationField(
        description="Channel-bed elevations by path and level (m a.s.l.)",
        shape=("num_bifurcation_paths", "num_bifurcation_levels"),
        category="param",
    )

    # ------------------------------------------------------------------ #
    # State variables
    # ------------------------------------------------------------------ #
    bifurcation_outflow: torch.Tensor = BifurcationField(
        description="Outflow through each bifurcation path & level (m³ s⁻¹)",
        shape=("num_bifurcation_paths", "num_bifurcation_levels"),
        default=0,
        category="init_state",
    )

    bifurcation_cross_section_depth: torch.Tensor = BifurcationField(
        description="Cross-sectional water depth (m)",
        shape=("num_bifurcation_paths", "num_bifurcation_levels"),
        default=0,
        category="init_state",
    )

    # ------------------------------------------------------------------ #
    # Computed tensor indices
    # ------------------------------------------------------------------ #
    bifurcation_catchment_idx = ReferenceIndexField(
        "bifurcation_catchment_id",
    )
    bifurcation_downstream_idx = ReferenceIndexField(
        "bifurcation_downstream_id",
    )

    # ------------------------------------------------------------------ #
    # Computed scalar dimensions
    # ------------------------------------------------------------------ #
    @computed_field(
        description="Number of paths saved in output."
    )
    @cached_property
    def num_output_bifurcation_paths(self) -> int:
        if self.output_bifurcation_path_id is None:
            return self.num_bifurcation_paths
        return self.output_bifurcation_path_id.shape[0]

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

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
