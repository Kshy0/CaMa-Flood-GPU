# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Prescribed downstream water-level boundaries."""

from __future__ import annotations

from functools import cached_property
from typing import ClassVar

import torch
from hydroforge.model.module import (AbstractModule, CoordinateField,
                                        ReferenceIndexField, TensorField,
                                        module_ref)
from pydantic import computed_field

from cmfgpu.modules.base import BaseModule


class SeaLevelModule(AbstractModule):
    """Maintain prescribed absolute downstream water levels.

    Every selected catchment must be a river mouth.
    """

    module_name: ClassVar[str] = "sea_level"
    description: ClassVar[str] = "Prescribed downstream water-level boundary"
    base = module_ref(BaseModule)

    sea_level_catchment_id: torch.Tensor = CoordinateField(
        description="Catchment IDs whose downstream water level is prescribed",
        dtype="int",
        shape=("num_sea_level_boundaries",),
        references="catchment_id",
    )

    sea_level_catchment_idx = ReferenceIndexField("sea_level_catchment_id")
    catchment_sea_level_idx = ReferenceIndexField(
        "sea_level_catchment_id", inverse=True,
    )

    @computed_field(description="Number of prescribed water-level boundaries")
    @cached_property
    def num_sea_level_boundaries(self) -> int:
        return self.sea_level_catchment_id.shape[0]

    sea_surface_elevation: torch.Tensor = TensorField(
        description="Current absolute downstream water-surface elevation (m)",
        shape=("num_sea_level_boundaries",),
        dtype="float",
        dim_coords="sea_level_catchment_id",
        category="param",
        output="disabled",
        default=0,
    )

    def validate_linked_state(self) -> None:
        """Require every prescribed boundary catchment to be a river mouth."""
        idx = self.sea_level_catchment_idx.to(self.base.downstream_idx.device)
        invalid = self.base.downstream_idx[idx] != idx
        if torch.any(invalid):
            ids = self.sea_level_catchment_id[
                invalid.to(self.sea_level_catchment_id.device)
            ][:10].detach().cpu().tolist()
            raise ValueError(
                "Sea-level boundary catchments are not river mouths: "
                f"{ids}"
            )
