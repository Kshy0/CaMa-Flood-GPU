# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Master controller class for managing all CaMa-Flood-GPU modules using Pydantic v2.
"""

from collections.abc import Mapping
from typing import ClassVar

import torch
from hydroforge.contracts import BackendRequirement, ModuleRequirement
from hydroforge.execution import (
    ManagedStep,
    all_reduce_,
    between_steps,
    managed_step,
)
from hydroforge.model import (
    AbstractModel,
    copy_tensor_inputs,
    module_ref,
    optional_module_ref,
)

from cmfgpu.modules.adaptive_time import AdaptiveTimeModule
from cmfgpu.modules.base import BaseModule
from cmfgpu.modules.bifurcation import BifurcationModule
from cmfgpu.modules.inflow import InflowModule
from cmfgpu.modules.levee import LeveeModule
from cmfgpu.modules.log import LogModule
from cmfgpu.modules.reservoir import ReservoirModule
from cmfgpu.modules.sea_level import SeaLevelModule
from cmfgpu.phys.adaptive_time import compute_adaptive_time_step
from cmfgpu.phys.bifurcation import (
    compute_bifurcation_inflow,
    compute_bifurcation_outflow,
)
from cmfgpu.phys.levee import (
    compute_levee_bifurcation_outflow,
    compute_levee_stage,
    compute_levee_stage_log,
)
from cmfgpu.phys.outflow import compute_inflow, compute_outflow
from cmfgpu.phys.reservoir import compute_reservoir_outflow
from cmfgpu.phys.storage import compute_flood_stage, compute_flood_stage_log


class CaMaFlood(AbstractModel):
    """
    CaMa-Flood GPU model master controller class
    """

    base = module_ref(BaseModule)
    inflow = optional_module_ref(InflowModule)
    sea_level = optional_module_ref(SeaLevelModule)
    bifurcation = optional_module_ref(BifurcationModule)
    log = optional_module_ref(LogModule)
    adaptive_time = optional_module_ref(AdaptiveTimeModule)
    levee = optional_module_ref(LeveeModule)
    reservoir = optional_module_ref(ReservoirModule)
    partition_key: ClassVar[str] = "catchment_id"
    partition_group: ClassVar[str] = "catchment_basin_id"
    cuda_extension_modules: ClassVar[tuple[str, ...]] = ("cmfgpu.phys.cuda",)
    backend_requirements: ClassVar[Mapping[str, BackendRequirement]] = {
        "cuda": BackendRequirement(),
    }
    module_requirements: ClassVar[Mapping[str, ModuleRequirement]] = {
        "log": ModuleRequirement(ensemble=False),
    }

    def initialize_model_state(self) -> None:
        reservoir = self.reservoir
        if reservoir is not None:
            reservoir.initialize_state()

    @between_steps
    @torch.inference_mode()
    def set_inputs(
        self,
        *,
        runoff: torch.Tensor,
        inflow: torch.Tensor | None = None,
        sea_surface_elevation: torch.Tensor | None = None,
    ) -> None:
        """Stage all public dynamic forcing without rebinding model buffers."""

        staged_inputs = {}
        staged_inputs["runoff"] = (runoff, self.base.runoff)
        inflow_module = self.inflow
        if inflow_module is None:
            if inflow is not None:
                raise ValueError("inflow forcing requires the inflow module")
        else:
            if inflow is None:
                raise ValueError("the inflow module requires inflow forcing")
            staged_inputs["inflow"] = (inflow, inflow_module.inflow)
        sea_level_module = self.sea_level
        if sea_level_module is None:
            if sea_surface_elevation is not None:
                raise ValueError(
                    "sea_surface_elevation forcing requires the sea_level module"
                )
        else:
            if sea_surface_elevation is None:
                raise ValueError(
                    "the sea_level module requires sea_surface_elevation forcing"
                )
            staged_inputs["sea_surface_elevation"] = (
                sea_surface_elevation, sea_level_module.sea_surface_elevation,
            )
        copy_tensor_inputs(staged_inputs)

    @managed_step
    @torch.inference_mode()
    def step_advance(self, step: ManagedStep) -> None:
        """Advance one step; fixed mode accepts managed ``num_sub_steps``."""

        time_step_seconds = step.duration.total_seconds()
        adaptive_time = self.adaptive_time
        log = self.log
        reservoir = self.reservoir
        bifurcation = self.bifurcation
        levee = self.levee
        self.base.outer_time_step.fill_(time_step_seconds)
        if adaptive_time is not None:
            adaptive_time.max_sub_steps.zero_()
            compute_adaptive_time_step()
            if self.spatial_world_size > 1:
                all_reduce_(adaptive_time.max_sub_steps, reduction="max")
            if self.parallel is not None and self.parallel.ensemble_partitions > 1:
                all_reduce_(adaptive_time.max_sub_steps, reduction="max", scope="ensemble")
            fixed_substeps = step.fixed(
                count=int(adaptive_time.max_sub_steps.item()),
            )
        else:
            fixed_substeps = step.fixed()
        fixed_count = fixed_substeps.count
        time_sub_step = time_step_seconds / fixed_count

        if log is not None:
            log.set_time(time_sub_step, fixed_count, step.current_time)

        self.base.time_step.fill_(time_sub_step)

        for sub_step in fixed_substeps:
            if log is not None:
                self.base.current_step.copy_(sub_step.index)
            compute_outflow()

            if reservoir is not None:
                compute_reservoir_outflow()

            if bifurcation is not None:
                if levee is not None:
                    compute_levee_bifurcation_outflow()
                else:
                    compute_bifurcation_outflow()

            compute_inflow()

            if bifurcation is not None:
                compute_bifurcation_inflow()

            if log is not None:
                compute_flood_stage_log()
            else:
                compute_flood_stage()

            if levee is not None:
                if log is not None:
                    compute_levee_stage_log()
                else:
                    compute_levee_stage()

        if log is None:
            self.base.current_step.fill_(fixed_count - 1)

        if log is not None:
            if self.spatial_world_size > 1:
                log.gather_results()
            if self.spatial_rank == 0 and step.output_enabled:
                log.write_step(self.log_path)
            else:
                log.clear_buffers()
