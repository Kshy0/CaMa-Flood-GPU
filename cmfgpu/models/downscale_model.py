# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Master controller for high-resolution CaMa-Flood-GPU downscaling."""

from collections.abc import Mapping
from typing import ClassVar

import torch
from hydroforge.contracts import BackendRequirement
from hydroforge.execution import ManagedStep, between_steps, managed_step
from hydroforge.model import (
    AbstractModel,
    copy_tensor_inputs,
    module_ref,
    optional_module_ref,
)
from pydantic import Field

from cmfgpu.config import DownscaleOptions
from cmfgpu.modules.downscale import (
    DownscaleBaseModule,
    DownscaleDurationModule,
    DownscaleLeveeModule,
    DownscaleTributaryModule,
)
from cmfgpu.phys.storage import (
    compute_downscale_backwater_depth,
    compute_downscale_flood_depth,
    compute_downscale_flood_duration,
    compute_downscale_permanent_water,
    compute_downscale_tributary_depth,
)


class CaMaDownscale(AbstractModel):
    """High-resolution flood-depth and inundation-duration downscaling model."""

    options: DownscaleOptions = Field(default_factory=DownscaleOptions)

    base = module_ref(DownscaleBaseModule)
    levee = optional_module_ref(DownscaleLeveeModule)
    tributary = optional_module_ref(DownscaleTributaryModule)
    duration = optional_module_ref(DownscaleDurationModule)

    partition_key: ClassVar[str] = "catchment_id"
    partition_group: ClassVar[str] = "catchment_basin_id"
    backend_requirements: ClassVar[Mapping[str, BackendRequirement]] = {
        "cuda": BackendRequirement(),
        "triton": BackendRequirement(),
        "metal": BackendRequirement(default_mixed_precision=False),
    }

    @between_steps
    @torch.inference_mode()
    def set_inputs(
        self,
        *,
        flood_depth: torch.Tensor | None = None,
        protected_depth: torch.Tensor | None = None,
        flood_duration_table: torch.Tensor | None = None,
    ) -> None:
        """Stage coarse catchment flood depths or depth-duration histograms."""

        staged_inputs = {}
        duration_module = self.duration
        levee_module = self.levee

        if duration_module is not None:
            if flood_duration_table is None:
                raise ValueError(
                    "the duration module requires flood_duration_table forcing"
                )
            if flood_depth is not None or protected_depth is not None:
                raise ValueError(
                    "flood_depth and protected_depth are not used when the duration module is open"
                )
            staged_inputs["flood_duration_table"] = (
                flood_duration_table,
                duration_module.flood_duration_table,
            )
        else:
            if flood_duration_table is not None:
                raise ValueError(
                    "flood_duration_table forcing requires the duration module"
                )
            if flood_depth is None:
                raise ValueError("flood_depth forcing is required for depth downscaling")
            staged_inputs["flood_depth"] = (flood_depth, self.base.flood_depth)
            if levee_module is None:
                if protected_depth is not None:
                    raise ValueError(
                        "protected_depth forcing requires the levee module"
                    )
            else:
                if protected_depth is None:
                    raise ValueError(
                        "the levee module requires protected_depth forcing"
                    )
                staged_inputs["protected_depth"] = (
                    protected_depth,
                    levee_module.protected_depth,
                )

        copy_tensor_inputs(staged_inputs)

    @managed_step
    @torch.inference_mode()
    def step_advance(self, step: ManagedStep) -> None:
        """Project staged coarse flood depths or durations onto high-resolution pixels."""

        del step
        if self.duration is not None:
            compute_downscale_flood_duration()
            return

        tributary = self.tributary
        if tributary is not None:
            tributary.catchment_tributary_modified.zero_()

        compute_downscale_flood_depth()

        if tributary is not None:
            if tributary.num_tributary_pixels > 0:
                compute_downscale_tributary_depth()
            if tributary.num_backwater_pixels > 0:
                compute_downscale_backwater_depth()
            if self.base.permanent_water_depth > 0.0:
                compute_downscale_permanent_water()
