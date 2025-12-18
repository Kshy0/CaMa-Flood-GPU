# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Master controller class for managing all CaMa-Flood-GPU modules using Pydantic v2.
"""
from datetime import datetime
from functools import cached_property
from typing import Callable, ClassVar, Dict, Optional, Type, Union

import cftime
import torch
import triton
from pydantic import PrivateAttr, computed_field
from torch import distributed as dist

from cmfgpu.models.abstract_model import AbstractModel
from cmfgpu.modules.adaptive_time import AdaptiveTimeModule
from cmfgpu.modules.base import BaseModule
from cmfgpu.modules.bifurcation import BifurcationModule
from cmfgpu.modules.levee import LeveeModule
from cmfgpu.modules.log import LogModule
from cmfgpu.phys.adaptive_time import (
    compute_adaptive_time_step_batched_kernel,
    compute_adaptive_time_step_kernel)
from cmfgpu.phys.bifurcation import (
    compute_bifurcation_inflow_batched_kernel,
    compute_bifurcation_inflow_kernel,
    compute_bifurcation_outflow_batched_kernel,
    compute_bifurcation_outflow_kernel)
from cmfgpu.phys.levee import (
    compute_levee_bifurcation_outflow_batched_kernel,
    compute_levee_bifurcation_outflow_kernel,
    compute_levee_stage_batched_kernel, compute_levee_stage_kernel,
    compute_levee_stage_log_kernel)
from cmfgpu.phys.outflow import (compute_inflow_batched_kernel,
                                 compute_inflow_kernel,
                                 compute_outflow_batched_kernel,
                                 compute_outflow_kernel)
from cmfgpu.phys.storage import (compute_flood_stage_batched_kernel,
                                 compute_flood_stage_kernel,
                                 compute_flood_stage_log_kernel)


class CaMaFlood(AbstractModel):
    """
    CaMa-Flood GPU model master controller class
    """
    module_list: ClassVar[Dict[str, Type[BaseModule]]] = {
        "base": BaseModule,
        "bifurcation": BifurcationModule,
        "log": LogModule,
        "adaptive_time": AdaptiveTimeModule,
        "levee": LeveeModule,
    }
    group_by: ClassVar[str] = "catchment_basin_id"
    _stats_elapsed_time: float = PrivateAttr(default=0.0)
    output_start_time: Optional[Union[datetime, cftime.datetime]] = None

    @cached_property
    def base(self) -> BaseModule:
        return self.get_module("base")
    
    @cached_property
    def bifurcation(self) -> Optional[BifurcationModule]:
        return self.get_module("bifurcation")

    @cached_property
    def levee(self) -> Optional[LeveeModule]:
        return self.get_module("levee")

    @cached_property
    def log(self) -> Optional[LogModule]:
        return self.get_module("log")

    @cached_property
    def adaptive_time(self) -> Optional[AdaptiveTimeModule]:
        return self.get_module("adaptive_time")
    
    @computed_field
    @cached_property
    def bifurcation_flag(self) -> bool:
        return self.bifurcation is not None

    @computed_field
    @cached_property
    def levee_flag(self) -> bool:
        return self.levee is not None

    @cached_property
    def base_grid(self) -> Callable:
        return lambda META: (triton.cdiv(self.base.num_catchments * (self.num_trials or 1), META["BLOCK_SIZE"]),)

    @cached_property
    def bifurcation_grid(self) -> Callable:
        if not self.bifurcation_flag:
            return None
        return lambda META: (triton.cdiv(self.bifurcation.num_bifurcation_paths * (self.num_trials or 1), META["BLOCK_SIZE"]),)

    @cached_property
    def levee_grid(self) -> Callable:
        if not self.levee_flag:
            return None
        return lambda META: (triton.cdiv(self.base.num_levees * (self.num_trials or 1), META["BLOCK_SIZE"]),)

    def step_advance(
        self,
        runoff: torch.Tensor,
        time_step: float,
        default_num_sub_steps: int,
        current_time: Optional[Union[datetime, cftime.datetime]],
        stat_is_first: bool = True,
        stat_is_last: bool = True,        
        output_enabled: bool = True) -> None:
        """
        Advance the model by one time step using the provided runoff input.

        Notes on time-weighted statistics:
          - Time-weighted accumulation is performed every sub-step with weight = dt (seconds).
          - If stat_is_first is True, the accumulation window is reset at the first sub-step.
          - If stat_is_last is True, the window is finalized at the last sub-step: all accumulated
            sums are divided by the total elapsed time (total_weight) to yield means.
          - By default (stat_is_first=True, stat_is_last=True), each call to step_advance forms an
            independent window and is saved once at the end of this call.

        Example: If you want to save daily statistics while the input runoff is hourly:
          - At 00:00-01:00:    call step_advance(..., stat_is_first=True,  stat_is_last=False)
          - At 01:00-23:00:    call step_advance(..., stat_is_first=False, stat_is_last=False)
          - At 23:00-24:00:    call step_advance(..., stat_is_first=False, stat_is_last=True)
          This makes the whole day one averaging window; only the last hour finalizes the mean.

        Args:
            runoff (torch.Tensor): Input runoff tensor for this time step.
            time_step (float): Duration of the time step (seconds).
            default_num_sub_steps (int): Default sub-steps if adaptive time stepping is disabled.
            current_time (Optional[datetime]): Current simulation time. Used for logging.
            stat_is_first (bool): Whether this call starts a new statistics window (resets accumulation).
            stat_is_last (bool): Whether this call ends the current statistics window (finalize average).
        """
        self.execute_parameter_change_plan(current_time)

        if self.adaptive_time is not None:
            self.adaptive_time.min_time_sub_step.fill_(float('inf'))
            if self.num_trials is not None:
                compute_adaptive_time_step_batched_kernel[self.base_grid](
                    river_depth_ptr=self.base.river_depth,
                    downstream_distance_ptr=self.base.downstream_distance,
                    min_time_sub_step_ptr=self.adaptive_time.min_time_sub_step,
                    time_step=time_step,
                    adaptive_time_factor=self.adaptive_time.adaptive_time_factor,
                    gravity=self.base.gravity,
                    num_catchments=self.base.num_catchments,
                    num_trials=self.num_trials,
                    BLOCK_SIZE=self.BLOCK_SIZE,
                    batched_downstream_distance=False,
                )
            else:
                compute_adaptive_time_step_kernel[self.base_grid](
                    river_depth_ptr=self.base.river_depth,
                    downstream_distance_ptr=self.base.downstream_distance,
                    min_time_sub_step_ptr=self.adaptive_time.min_time_sub_step,
                    time_step=time_step,
                    adaptive_time_factor=self.adaptive_time.adaptive_time_factor,
                    gravity=self.base.gravity,
                    num_catchments=self.base.num_catchments,
                    BLOCK_SIZE=self.BLOCK_SIZE
                )
            if self.world_size > 1:
                dist.all_reduce(self.adaptive_time.min_time_sub_step, op=dist.ReduceOp.MIN)
            
            # Take the minimum across all trials if batched
            min_dt = self.adaptive_time.min_time_sub_step.min().item()
            num_sub_steps = int(round(time_step / min_dt - 0.01) + 1)
            time_sub_step = time_step / num_sub_steps
        else:
            num_sub_steps = default_num_sub_steps
            time_sub_step = time_step / num_sub_steps
        if self.log is not None:
            self.log.set_time(time_sub_step, num_sub_steps, current_time)

        if stat_is_first:
            # Reset elapsed time counter at the beginning of a stats window
            self._stats_elapsed_time = 0.0

        # Check if output is enabled
        if self.output_start_time is not None and current_time is not None:
            if current_time < self.output_start_time:
                output_enabled = False

        for sub_step in range(num_sub_steps):
            # Determine flags for the first/last sub-step of this model step
            is_first = stat_is_first and (sub_step == 0)
            is_last = stat_is_last and (sub_step == num_sub_steps - 1)
            self.do_one_sub_step(time_sub_step, runoff, sub_step)
            # Accumulate elapsed time in seconds for the current window
            self._stats_elapsed_time += time_sub_step
            # Compute total_weight only when finalizing
            total_weight = self._stats_elapsed_time if is_last else 0.0
            # Update stats after physics for each sub-step
            if output_enabled:
                self.update_statistics(
                    weight=time_sub_step,
                    total_weight=total_weight,
                    is_first=is_first,
                    is_last=is_last,
                    BLOCK_SIZE=self.BLOCK_SIZE,
                )

        # Reset elapsed counter after closing a window
        if stat_is_last:
            if output_enabled:
                self.finalize_time_step(current_time)
            self._stats_elapsed_time = 0.0
        
        if self.log is not None:
            if self.world_size > 1:
                self.log.gather_results()
            if self.rank == 0 and output_enabled:
                self.log.write_step(self.log_path)
        if self.rank == 0:
            msg = f"Processed step at {current_time}, adaptive_time_step={num_sub_steps}"
            print(f"\r{msg:<80}", end="", flush=True)
    def do_one_sub_step(self, time_sub_step: float, runoff: torch.Tensor, sub_step: int) -> None:
        """Execute one sub time step calculation"""
        # Outflow computation
        if self.num_trials is not None:
            compute_outflow_batched_kernel[self.base_grid](
                is_river_mouth_ptr=self.base.is_river_mouth,
                downstream_idx_ptr=self.base.downstream_idx,
                river_inflow_ptr=self.base.river_inflow,
                flood_inflow_ptr=self.base.flood_inflow,
                global_bifurcation_outflow_ptr=self.base.global_bifurcation_outflow,
                river_outflow_ptr=self.base.river_outflow,
                flood_outflow_ptr=self.base.flood_outflow,
                river_manning_ptr=self.base.river_manning,
                flood_manning_ptr=self.base.flood_manning,
                river_depth_ptr=self.base.river_depth,
                river_width_ptr=self.base.river_width,
                river_length_ptr=self.base.river_length,
                river_height_ptr=self.base.river_height,
                river_storage_ptr=self.base.river_storage,
                flood_depth_ptr=self.base.flood_depth,
                protected_depth_ptr=self.base.protected_depth,
                catchment_elevation_ptr=self.base.catchment_elevation,
                downstream_distance_ptr=self.base.downstream_distance,
                flood_storage_ptr=self.base.flood_storage,
                protected_storage_ptr=self.base.protected_storage,
                river_cross_section_depth_ptr=self.base.river_cross_section_depth,
                flood_cross_section_depth_ptr=self.base.flood_cross_section_depth,
                flood_cross_section_area_ptr=self.base.flood_cross_section_area,
                total_storage_ptr=self.base.total_storage,
                outgoing_storage_ptr=self.base.outgoing_storage,
                water_surface_elevation_ptr=self.base.water_surface_elevation,
                protected_water_surface_elevation_ptr=self.base.protected_water_surface_elevation,
                gravity=self.base.gravity,
                time_step=time_sub_step,
                num_catchments=self.base.num_catchments,
                num_trials=self.num_trials,
                BLOCK_SIZE=self.BLOCK_SIZE,
                batched_river_manning=self.base.batched_river_manning,
                batched_flood_manning=self.base.batched_flood_manning,
                batched_river_width=self.base.batched_river_width,
                batched_river_length=self.base.batched_river_length,
                batched_river_height=self.base.batched_river_height,
                batched_catchment_elevation=self.base.batched_catchment_elevation,
            )
        else:
            compute_outflow_kernel[self.base_grid](
                is_river_mouth_ptr=self.base.is_river_mouth,
                downstream_idx_ptr=self.base.downstream_idx,
                river_inflow_ptr=self.base.river_inflow,
                flood_inflow_ptr=self.base.flood_inflow,
                global_bifurcation_outflow_ptr=self.base.global_bifurcation_outflow,
                river_outflow_ptr=self.base.river_outflow,
                flood_outflow_ptr=self.base.flood_outflow,
                river_manning_ptr=self.base.river_manning,
                flood_manning_ptr=self.base.flood_manning,
                river_depth_ptr=self.base.river_depth,
                river_width_ptr=self.base.river_width,
                river_length_ptr=self.base.river_length,
                river_height_ptr=self.base.river_height,
                river_storage_ptr=self.base.river_storage,
                flood_depth_ptr=self.base.flood_depth,
                protected_depth_ptr=self.base.protected_depth,
                catchment_elevation_ptr=self.base.catchment_elevation,
                downstream_distance_ptr=self.base.downstream_distance,
                flood_storage_ptr=self.base.flood_storage,
                protected_storage_ptr=self.base.protected_storage,
                river_cross_section_depth_ptr=self.base.river_cross_section_depth,
                flood_cross_section_depth_ptr=self.base.flood_cross_section_depth,
                flood_cross_section_area_ptr=self.base.flood_cross_section_area,
                total_storage_ptr=self.base.total_storage,
                outgoing_storage_ptr=self.base.outgoing_storage,
                water_surface_elevation_ptr=self.base.water_surface_elevation,
                protected_water_surface_elevation_ptr=self.base.protected_water_surface_elevation,
                gravity=self.base.gravity,
                time_step=time_sub_step,
                num_catchments=self.base.num_catchments,
                BLOCK_SIZE=self.BLOCK_SIZE
            )
        
        # Bifurcation outflow computation
        if self.bifurcation_flag:
            if self.levee_flag:
                if self.num_trials is not None:
                    compute_levee_bifurcation_outflow_batched_kernel[self.bifurcation_grid](
                        bifurcation_catchment_idx_ptr=self.bifurcation.bifurcation_catchment_idx,
                        bifurcation_downstream_idx_ptr=self.bifurcation.bifurcation_downstream_idx,
                        bifurcation_manning_ptr=self.bifurcation.bifurcation_manning,
                        bifurcation_outflow_ptr=self.bifurcation.bifurcation_outflow,
                        bifurcation_width_ptr=self.bifurcation.bifurcation_width,
                        bifurcation_length_ptr=self.bifurcation.bifurcation_length,
                        bifurcation_elevation_ptr=self.bifurcation.bifurcation_elevation,
                        bifurcation_cross_section_depth_ptr=self.bifurcation.bifurcation_cross_section_depth,
                        water_surface_elevation_ptr=self.base.water_surface_elevation,
                        protected_water_surface_elevation_ptr=self.base.protected_water_surface_elevation,
                        total_storage_ptr=self.base.total_storage,
                        outgoing_storage_ptr=self.base.outgoing_storage,
                        gravity=self.base.gravity,
                        time_step=time_sub_step,
                        num_bifurcation_paths=self.bifurcation.num_bifurcation_paths,
                        num_bifurcation_levels=self.bifurcation.num_bifurcation_levels,
                        num_trials=self.num_trials,
                        BLOCK_SIZE=self.BLOCK_SIZE,
                        num_catchments=self.base.num_catchments,
                        batched_bifurcation_manning=self.bifurcation.batched_bifurcation_manning,
                        batched_bifurcation_width=self.bifurcation.batched_bifurcation_width,
                        batched_bifurcation_length=self.bifurcation.batched_bifurcation_length,
                        batched_bifurcation_elevation=self.bifurcation.batched_bifurcation_elevation,
                    )
                else:
                    compute_levee_bifurcation_outflow_kernel[self.bifurcation_grid](
                        bifurcation_catchment_idx_ptr=self.bifurcation.bifurcation_catchment_idx,
                        bifurcation_downstream_idx_ptr=self.bifurcation.bifurcation_downstream_idx,
                        bifurcation_manning_ptr=self.bifurcation.bifurcation_manning,
                        bifurcation_outflow_ptr=self.bifurcation.bifurcation_outflow,
                        bifurcation_width_ptr=self.bifurcation.bifurcation_width,
                        bifurcation_length_ptr=self.bifurcation.bifurcation_length,
                        bifurcation_elevation_ptr=self.bifurcation.bifurcation_elevation,
                        bifurcation_cross_section_depth_ptr=self.bifurcation.bifurcation_cross_section_depth,
                        water_surface_elevation_ptr=self.base.water_surface_elevation,
                        protected_water_surface_elevation_ptr=self.base.protected_water_surface_elevation,
                        total_storage_ptr=self.base.total_storage,
                        outgoing_storage_ptr=self.base.outgoing_storage,
                        gravity=self.base.gravity,
                        time_step=time_sub_step,
                        num_bifurcation_paths=self.bifurcation.num_bifurcation_paths,
                        num_bifurcation_levels=self.bifurcation.num_bifurcation_levels,
                        BLOCK_SIZE=self.BLOCK_SIZE
                    )
            else:
                if self.num_trials is not None:
                    compute_bifurcation_outflow_batched_kernel[self.bifurcation_grid](
                        bifurcation_catchment_idx_ptr=self.bifurcation.bifurcation_catchment_idx,
                        bifurcation_downstream_idx_ptr=self.bifurcation.bifurcation_downstream_idx,
                        bifurcation_manning_ptr=self.bifurcation.bifurcation_manning,
                        bifurcation_outflow_ptr=self.bifurcation.bifurcation_outflow,
                        bifurcation_width_ptr=self.bifurcation.bifurcation_width,
                        bifurcation_length_ptr=self.bifurcation.bifurcation_length,
                        bifurcation_elevation_ptr=self.bifurcation.bifurcation_elevation,
                        bifurcation_cross_section_depth_ptr=self.bifurcation.bifurcation_cross_section_depth,
                        water_surface_elevation_ptr=self.base.water_surface_elevation,
                        total_storage_ptr=self.base.total_storage,
                        outgoing_storage_ptr=self.base.outgoing_storage,
                        gravity=self.base.gravity,
                        time_step=time_sub_step,
                        num_bifurcation_paths=self.bifurcation.num_bifurcation_paths,
                        num_bifurcation_levels=self.bifurcation.num_bifurcation_levels,
                        num_trials=self.num_trials,
                        BLOCK_SIZE=self.BLOCK_SIZE,
                        num_catchments=self.base.num_catchments,
                        batched_bifurcation_manning=self.bifurcation.batched_bifurcation_manning,
                        batched_bifurcation_width=self.bifurcation.batched_bifurcation_width,
                        batched_bifurcation_length=self.bifurcation.batched_bifurcation_length,
                        batched_bifurcation_elevation=self.bifurcation.batched_bifurcation_elevation,
                    )
                else:
                    compute_bifurcation_outflow_kernel[self.bifurcation_grid](
                        bifurcation_catchment_idx_ptr=self.bifurcation.bifurcation_catchment_idx,
                        bifurcation_downstream_idx_ptr=self.bifurcation.bifurcation_downstream_idx,
                        bifurcation_manning_ptr=self.bifurcation.bifurcation_manning,
                        bifurcation_outflow_ptr=self.bifurcation.bifurcation_outflow,
                        bifurcation_width_ptr=self.bifurcation.bifurcation_width,
                        bifurcation_length_ptr=self.bifurcation.bifurcation_length,
                        bifurcation_elevation_ptr=self.bifurcation.bifurcation_elevation,
                        bifurcation_cross_section_depth_ptr=self.bifurcation.bifurcation_cross_section_depth,
                        water_surface_elevation_ptr=self.base.water_surface_elevation,
                        total_storage_ptr=self.base.total_storage,
                        outgoing_storage_ptr=self.base.outgoing_storage,
                        gravity=self.base.gravity,
                        time_step=time_sub_step,
                        num_bifurcation_paths=self.bifurcation.num_bifurcation_paths,
                        num_bifurcation_levels=self.bifurcation.num_bifurcation_levels,
                        BLOCK_SIZE=self.BLOCK_SIZE
                    )


        # Inflow computation
        if self.num_trials is not None:
            compute_inflow_batched_kernel[self.base_grid](
                is_river_mouth_ptr=self.base.is_river_mouth,
                downstream_idx_ptr=self.base.downstream_idx,
                river_outflow_ptr=self.base.river_outflow,
                flood_outflow_ptr=self.base.flood_outflow,
                total_storage_ptr=self.base.total_storage,
                outgoing_storage_ptr=self.base.outgoing_storage,
                river_inflow_ptr=self.base.river_inflow,
                flood_inflow_ptr=self.base.flood_inflow,
                limit_rate_ptr=self.base.limit_rate,
                num_catchments=self.base.num_catchments,
                num_trials=self.num_trials,
                BLOCK_SIZE=self.BLOCK_SIZE,
            )
        else:
            compute_inflow_kernel[self.base_grid](
                is_river_mouth_ptr=self.base.is_river_mouth,
                downstream_idx_ptr=self.base.downstream_idx,
                river_outflow_ptr=self.base.river_outflow,
                flood_outflow_ptr=self.base.flood_outflow,
                total_storage_ptr=self.base.total_storage,
                outgoing_storage_ptr=self.base.outgoing_storage,
                river_inflow_ptr=self.base.river_inflow,
                flood_inflow_ptr=self.base.flood_inflow,
                limit_rate_ptr=self.base.limit_rate,
                num_catchments=self.base.num_catchments,
                BLOCK_SIZE=self.BLOCK_SIZE
            )
        
        # Bifurcation inflow computation (if enabled)
        if self.bifurcation_flag:
            if self.num_trials is not None:
                compute_bifurcation_inflow_batched_kernel[self.bifurcation_grid](
                    bifurcation_catchment_idx_ptr=self.bifurcation.bifurcation_catchment_idx,
                    bifurcation_downstream_idx_ptr=self.bifurcation.bifurcation_downstream_idx,
                    limit_rate_ptr=self.base.limit_rate,
                    bifurcation_outflow_ptr=self.bifurcation.bifurcation_outflow,
                    global_bifurcation_outflow_ptr=self.base.global_bifurcation_outflow,
                    num_bifurcation_paths=self.bifurcation.num_bifurcation_paths,
                    num_bifurcation_levels=self.bifurcation.num_bifurcation_levels,
                    num_trials=self.num_trials,
                    BLOCK_SIZE=self.BLOCK_SIZE,
                    num_catchments=self.base.num_catchments,
                )
            else:
                compute_bifurcation_inflow_kernel[self.bifurcation_grid](
                    bifurcation_catchment_idx_ptr=self.bifurcation.bifurcation_catchment_idx,
                    bifurcation_downstream_idx_ptr=self.bifurcation.bifurcation_downstream_idx,
                    limit_rate_ptr=self.base.limit_rate,
                    bifurcation_outflow_ptr=self.bifurcation.bifurcation_outflow,
                    global_bifurcation_outflow_ptr=self.base.global_bifurcation_outflow,
                    num_bifurcation_paths=self.bifurcation.num_bifurcation_paths,
                    num_bifurcation_levels=self.bifurcation.num_bifurcation_levels,
                    BLOCK_SIZE=self.BLOCK_SIZE
                )

        # Flood stage computation for non-levee catchments
        if self.num_trials is not None:
            compute_flood_stage_batched_kernel[self.base_grid](
                river_inflow_ptr=self.base.river_inflow,
                flood_inflow_ptr=self.base.flood_inflow,
                river_outflow_ptr=self.base.river_outflow,
                flood_outflow_ptr=self.base.flood_outflow,
                global_bifurcation_outflow_ptr=self.base.global_bifurcation_outflow,
                runoff_ptr=runoff, 
                time_step=time_sub_step,
                outgoing_storage_ptr=self.base.outgoing_storage,
                river_storage_ptr=self.base.river_storage,
                flood_storage_ptr=self.base.flood_storage,
                protected_storage_ptr=self.base.protected_storage,
                river_depth_ptr=self.base.river_depth,
                flood_depth_ptr=self.base.flood_depth,
                protected_depth_ptr=self.base.protected_depth,
                flood_fraction_ptr=self.base.flood_fraction,
                flood_area_ptr=self.base.flood_area,
                river_height_ptr=self.base.river_height,
                flood_depth_table_ptr=self.base.flood_depth_table,
                catchment_area_ptr=self.base.catchment_area,
                river_width_ptr=self.base.river_width,
                river_length_ptr=self.base.river_length,
                num_catchments=self.base.num_catchments,
                num_flood_levels=self.base.num_flood_levels,
                num_trials=self.num_trials,
                BLOCK_SIZE=self.BLOCK_SIZE,
                batched_runoff=(runoff.ndim > 1 and runoff.shape[0] == self.num_trials),
                batched_river_height=self.base.batched_river_height,
                batched_flood_depth_table=self.base.batched_flood_depth_table,
                batched_catchment_area=self.base.batched_catchment_area,
                batched_river_width=self.base.batched_river_width,
                batched_river_length=self.base.batched_river_length,
            )
        elif self.log is not None:
            compute_flood_stage_log_kernel[self.base_grid](
                river_inflow_ptr=self.base.river_inflow,
                flood_inflow_ptr=self.base.flood_inflow,
                river_outflow_ptr=self.base.river_outflow,
                flood_outflow_ptr=self.base.flood_outflow,
                global_bifurcation_outflow_ptr=self.base.global_bifurcation_outflow,
                runoff_ptr=runoff,
                time_step=time_sub_step,
                outgoing_storage_ptr=self.base.outgoing_storage,
                river_storage_ptr=self.base.river_storage,
                flood_storage_ptr=self.base.flood_storage,
                protected_storage_ptr=self.base.protected_storage,
                river_depth_ptr=self.base.river_depth,
                flood_depth_ptr=self.base.flood_depth,
                protected_depth_ptr=self.base.protected_depth,
                flood_fraction_ptr=self.base.flood_fraction,
                flood_area_ptr=self.base.flood_area,
                river_height_ptr=self.base.river_height,
                flood_depth_table_ptr=self.base.flood_depth_table,
                catchment_area_ptr=self.base.catchment_area,
                river_width_ptr=self.base.river_width,
                river_length_ptr=self.base.river_length,
                is_levee_ptr=self.base.is_levee,
                total_storage_pre_sum_ptr=self.log.total_storage_pre_sum,
                total_storage_next_sum_ptr=self.log.total_storage_next_sum,
                total_storage_new_sum_ptr=self.log.total_storage_new_sum,
                total_inflow_sum_ptr=self.log.total_inflow_sum,
                total_outflow_sum_ptr=self.log.total_outflow_sum,
                total_storage_stage_sum_ptr=self.log.total_storage_stage_sum,
                river_storage_sum_ptr=self.log.river_storage_sum,
                flood_storage_sum_ptr=self.log.flood_storage_sum,
                total_inflow_error_sum_ptr=self.log.total_inflow_error_sum,
                total_stage_error_sum_ptr=self.log.total_stage_error_sum,
                flood_area_sum_ptr=self.log.flood_area_sum,
                current_step=sub_step,
                num_catchments=self.base.num_catchments,
                num_flood_levels=self.base.num_flood_levels,
                BLOCK_SIZE=self.BLOCK_SIZE
            )
        else:
            compute_flood_stage_kernel[self.base_grid](
                river_inflow_ptr=self.base.river_inflow,
                flood_inflow_ptr=self.base.flood_inflow,
                river_outflow_ptr=self.base.river_outflow,
                flood_outflow_ptr=self.base.flood_outflow,
                global_bifurcation_outflow_ptr=self.base.global_bifurcation_outflow,
                runoff_ptr=runoff, 
                time_step=time_sub_step,
                outgoing_storage_ptr=self.base.outgoing_storage,
                river_storage_ptr=self.base.river_storage,
                flood_storage_ptr=self.base.flood_storage,
                protected_storage_ptr=self.base.protected_storage,
                river_depth_ptr=self.base.river_depth,
                flood_depth_ptr=self.base.flood_depth,
                protected_depth_ptr=self.base.protected_depth,
                flood_fraction_ptr=self.base.flood_fraction,
                flood_area_ptr=self.base.flood_area,
                river_height_ptr=self.base.river_height,
                flood_depth_table_ptr=self.base.flood_depth_table,
                catchment_area_ptr=self.base.catchment_area,
                river_width_ptr=self.base.river_width,
                river_length_ptr=self.base.river_length,
                num_catchments=self.base.num_catchments,
                num_flood_levels=self.base.num_flood_levels,
                BLOCK_SIZE=self.BLOCK_SIZE
            )

        # Levee stage computation (if enabled)
        if self.levee_flag:
            if self.num_trials is not None:
                compute_levee_stage_batched_kernel[self.levee_grid](
                    levee_catchment_idx_ptr=self.levee.levee_catchment_idx,
                    river_storage_ptr=self.base.river_storage,
                    flood_storage_ptr=self.base.flood_storage,
                    protected_storage_ptr=self.base.protected_storage,
                    river_depth_ptr=self.base.river_depth,
                    flood_depth_ptr=self.base.flood_depth,
                    protected_depth_ptr=self.base.protected_depth,
                    river_height_ptr=self.base.river_height,
                    flood_depth_table_ptr=self.base.flood_depth_table,
                    catchment_area_ptr=self.base.catchment_area,
                    river_width_ptr=self.base.river_width,
                    river_length_ptr=self.base.river_length,
                    levee_base_height_ptr=self.levee.levee_base_height,
                    levee_crown_height_ptr=self.levee.levee_crown_height,
                    levee_fraction_ptr=self.levee.levee_fraction,
                    flood_fraction_ptr=self.base.flood_fraction,
                    flood_area_ptr=self.base.flood_area,
                    num_levees=self.base.num_levees,
                    num_flood_levels=self.base.num_flood_levels,
                    num_trials=self.num_trials,
                    BLOCK_SIZE=self.BLOCK_SIZE,
                    num_catchments=self.base.num_catchments,
                    batched_river_height=self.base.batched_river_height,
                    batched_flood_depth_table=self.base.batched_flood_depth_table,
                    batched_catchment_area=self.base.batched_catchment_area,
                    batched_river_width=self.base.batched_river_width,
                    batched_river_length=self.base.batched_river_length,
                    batched_levee_base_height=self.levee.batched_levee_base_height,
                    batched_levee_crown_height=self.levee.batched_levee_crown_height,
                    batched_levee_fraction=self.levee.batched_levee_fraction,
                )
            elif self.log is not None:
                compute_levee_stage_log_kernel[self.levee_grid](
                    levee_catchment_idx_ptr=self.levee.levee_catchment_idx,
                    river_storage_ptr=self.base.river_storage,
                    flood_storage_ptr=self.base.flood_storage,
                    protected_storage_ptr=self.base.protected_storage,
                    river_depth_ptr=self.base.river_depth,
                    flood_depth_ptr=self.base.flood_depth,
                    protected_depth_ptr=self.base.protected_depth,
                    river_height_ptr=self.base.river_height,
                    flood_depth_table_ptr=self.base.flood_depth_table,
                    catchment_area_ptr=self.base.catchment_area,
                    river_width_ptr=self.base.river_width,
                    river_length_ptr=self.base.river_length,
                    levee_base_height_ptr=self.levee.levee_base_height,
                    levee_crown_height_ptr=self.levee.levee_crown_height,
                    levee_fraction_ptr=self.levee.levee_fraction,
                    flood_fraction_ptr=self.base.flood_fraction,
                    flood_area_ptr=self.base.flood_area,
                    total_storage_stage_sum_ptr=self.log.total_storage_stage_sum,
                    river_storage_sum_ptr=self.log.river_storage_sum,
                    flood_storage_sum_ptr=self.log.flood_storage_sum,
                    flood_area_sum_ptr=self.log.flood_area_sum,
                    total_stage_error_sum_ptr=self.log.total_stage_error_sum,
                    current_step=sub_step,
                    num_levees=self.base.num_levees,
                    num_flood_levels=self.base.num_flood_levels,
                    BLOCK_SIZE=self.BLOCK_SIZE
                )
            else:
                compute_levee_stage_kernel[self.levee_grid](
                    levee_catchment_idx_ptr=self.levee.levee_catchment_idx,
                    river_storage_ptr=self.base.river_storage,
                    flood_storage_ptr=self.base.flood_storage,
                    protected_storage_ptr=self.base.protected_storage,
                    river_depth_ptr=self.base.river_depth,
                    flood_depth_ptr=self.base.flood_depth,
                    protected_depth_ptr=self.base.protected_depth,
                    river_height_ptr=self.base.river_height,
                    flood_depth_table_ptr=self.base.flood_depth_table,
                    catchment_area_ptr=self.base.catchment_area,
                    river_width_ptr=self.base.river_width,
                    river_length_ptr=self.base.river_length,
                    levee_base_height_ptr=self.levee.levee_base_height,
                    levee_crown_height_ptr=self.levee.levee_crown_height,
                    levee_fraction_ptr=self.levee.levee_fraction,
                    flood_fraction_ptr=self.base.flood_fraction,
                    flood_area_ptr=self.base.flood_area,
                    num_levees=self.base.num_levees,
                    num_flood_levels=self.base.num_flood_levels,
                    BLOCK_SIZE=self.BLOCK_SIZE
                )
