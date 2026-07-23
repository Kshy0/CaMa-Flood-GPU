# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Master controller class for managing all CaMa-Flood-GPU modules using Pydantic v2.
"""
from datetime import datetime
from typing import ClassVar, Dict, Mapping, Optional, Type, Union

import cftime
import torch
from hydroforge.model.model import AbstractModel
from hydroforge.contracts.events import emit
from hydroforge.contracts import BackendRequirement, ModuleRequirement
from hydroforge.execution import all_reduce_
from hydroforge.execution.inputs import copy_input
from hydroforge.execution.step import managed_step

from cmfgpu.modules.adaptive_time import AdaptiveTimeModule
from cmfgpu.modules.base import BaseModule
from cmfgpu.modules.bifurcation import BifurcationModule
from cmfgpu.modules.inflow import InflowModule
from cmfgpu.modules.levee import LeveeModule
from cmfgpu.modules.log import LogModule
from cmfgpu.modules.reservoir import ReservoirModule
from cmfgpu.modules.sea_level import SeaLevelModule
from cmfgpu.phys.adaptive_time import compute_adaptive_time_step
from cmfgpu.phys.bifurcation import (compute_bifurcation_inflow,
                                     compute_bifurcation_outflow)
from cmfgpu.phys.levee import (compute_levee_bifurcation_outflow,
                               compute_levee_stage, compute_levee_stage_log)
from cmfgpu.phys.outflow import compute_inflow, compute_outflow
from cmfgpu.phys.reservoir import compute_reservoir_outflow
from cmfgpu.phys.storage import compute_flood_stage, compute_flood_stage_log


class CaMaFlood(AbstractModel):
    """
    CaMa-Flood GPU model master controller class
    """

    module_list: ClassVar[Dict[str, Type[BaseModule]]] = {
        "base": BaseModule,
        "inflow": InflowModule,
        "sea_level": SeaLevelModule,
        "bifurcation": BifurcationModule,
        "log": LogModule,
        "adaptive_time": AdaptiveTimeModule,
        "levee": LeveeModule,
        "reservoir": ReservoirModule,
    }
    partition_key: ClassVar[str] = "catchment_id"
    partition_group: ClassVar[str] = "catchment_basin_id"
    cuda_extension_modules: ClassVar[tuple[str, ...]] = ("cmfgpu.phys.cuda",)
    backend_requirements: ClassVar[Mapping[str, BackendRequirement]] = {
        "cuda": BackendRequirement(trials=False),
    }
    module_requirements: ClassVar[Mapping[str, ModuleRequirement]] = {
        "log": ModuleRequirement(trials=False),
    }
    def initialize_model_state(self) -> None:
        if self.has_module("reservoir"):
            fixed = self.reservoir.initialize_dam_storage()
            if fixed:
                emit(
                    self, "info", "reservoir.storage_initialized",
                    "Initialized dam cells to conservation storage", cells=fixed,
                )
            if self.has_module("bifurcation"):
                masked = self.reservoir.mask_bifurcation_paths(self.bifurcation)
                if masked:
                    emit(
                        self, "info", "reservoir.bifurcation_masked",
                        "Masked dam-related bifurcation paths", paths=masked,
                    )

    @managed_step
    @torch.inference_mode()
    def step_advance(
        self,
        runoff: torch.Tensor,
        time_step: float,
        default_num_sub_steps: int,
        current_time: Optional[Union[datetime, cftime.datetime]],
        inflow: Optional[torch.Tensor] = None,
        sea_surface_elevation: Optional[torch.Tensor] = None,
        output_enabled: bool = True) -> None:
        """
        Advance the model by one time step using the provided runoff input.

        Statistics windows are inferred from the model's configured intervals.

        Args:
            runoff (torch.Tensor): Input runoff tensor for this time step.
            inflow (torch.Tensor, optional): Compact prescribed discharge on
                ``InflowModule.inflow_catchment_id``. It remains a separate
                component until the primal physics boundary.
            sea_surface_elevation (torch.Tensor, optional): Compact absolute
                downstream water levels on ``SeaLevelModule.sea_level_catchment_id``.
            time_step (float): Duration of the time step (seconds).
            default_num_sub_steps (int): Default sub-steps if adaptive time stepping is disabled.
            current_time (Optional[datetime]): Current simulation time. Used for logging.
        """
        copy_input(
            self.base.runoff, runoff, when_none="required", name="runoff",
            trial_broadcast=self.num_trials is not None,
        )
        copy_input(
            self.inflow.inflow if self.has_module("inflow") else None,
            inflow, name="inflow",
            trial_broadcast=self.num_trials is not None,
        )
        copy_input(
            (
                self.sea_level.sea_surface_elevation
                if self.has_module("sea_level") else None
            ),
            sea_surface_elevation,
            when_none="required" if self.has_module("sea_level") else "keep",
            name="sea_surface_elevation",
            trial_broadcast=self.num_trials is not None,
        )
        if self.has_module("adaptive_time"):
            self.adaptive_time.max_sub_steps.fill_(0)
            compute_adaptive_time_step(outer_time_step=time_step)
            if self.world_size > 1:
                all_reduce_(self.adaptive_time.max_sub_steps, reduction="max")
            
            # Take the maximum across all trials if batched
            num_sub_steps = int(self.adaptive_time.max_sub_steps.max().item())
            if num_sub_steps < 1:
                num_sub_steps = 1
            time_sub_step = time_step / num_sub_steps
        else:
            num_sub_steps = int(default_num_sub_steps)
            if num_sub_steps < 1:
                raise ValueError("default_num_sub_steps must be positive")
            time_sub_step = time_step / num_sub_steps
        if self.has_module("log"):
            self.log.set_time(time_sub_step, num_sub_steps, current_time)

        self.base.time_step.fill_(time_sub_step)

        for sub_step in self.substeps.fixed(count=num_sub_steps):
            # Tensor expressions are captured together with the registered
            # kernels; no backend/march code belongs in the model.
            if self.has_module("log"):
                self.base.current_step.copy_(sub_step.index)
            compute_outflow()

            if self.has_module("reservoir"):
                compute_reservoir_outflow()

            if self.has_module("bifurcation"):
                if self.has_module("levee"):
                    compute_levee_bifurcation_outflow()
                else:
                    compute_bifurcation_outflow()

            compute_inflow()

            if self.has_module("bifurcation"):
                compute_bifurcation_inflow()

            if self.has_module("log"):
                compute_flood_stage_log()
            else:
                compute_flood_stage()

            if self.has_module("levee"):
                if self.has_module("log"):
                    compute_levee_stage_log()
                else:
                    compute_levee_stage()

        # Keep the historical public state without adding scalar work to each
        # compiled physics iteration when per-substep logging is disabled.
        if not self.has_module("log"):
            self.base.current_step.fill_(num_sub_steps - 1)

        if self.has_module("log"):
            if self.world_size > 1:
                self.log.gather_results()
            if self.rank == 0 and output_enabled:
                self.log.write_step(self.log_path)
