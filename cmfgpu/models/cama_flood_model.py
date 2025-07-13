"""
Master controller class for managing all CaMa-Flood-GPU modules using Pydantic v2.
Updated to work with the new AbstractModule hierarchy and SimulationConfig validation.
"""
import triton
import torch
from typing import Dict, List, Optional, Type, ClassVar, Callable
from pathlib import Path
from datetime import datetime
from torch import distributed as dist
from pydantic import Field, computed_field
from functools import cached_property
from cmfgpu.models.abstract_model import AbstractModel
from cmfgpu.modules.base import BaseModule
from cmfgpu.modules.bifurcation import BifurcationModule
from cmfgpu.modules.log import LogModule
from cmfgpu.modules.adaptive_time import AdaptiveTimeModule
from cmfgpu.phys.triton.outflow import compute_outflow_kernel, compute_inflow_kernel
from cmfgpu.phys.triton.bifurcation import compute_bifurcation_outflow_kernel, compute_bifurcation_inflow_kernel
from cmfgpu.phys.triton.adaptive_time import compute_adaptive_time_step_kernel
from cmfgpu.phys.triton.storage import compute_flood_stage_kernel, compute_flood_stage_log_kernel


class CaMaFlood(AbstractModel):
    """
    CaMa-Flood GPU model master controller class
    """
    module_list: ClassVar[Dict[str, Type[BaseModule]]] = {
        "base": BaseModule,
        "bifurcation": BifurcationModule,
        "log": LogModule,
        "adaptive_time": AdaptiveTimeModule
    }
    group_by: ClassVar[str] = "catchment_basin_id"

    @cached_property
    def base(self) -> BaseModule:
        return self.get_module("base")
    
    @cached_property
    def bifurcation(self) -> Optional[BifurcationModule]:
        return self.get_module("bifurcation")

    @cached_property
    def log(self) -> Optional[LogModule]:
        return self.get_module("log")

    @cached_property
    def adaptive_time(self) -> Optional[AdaptiveTimeModule]:
        return self.get_module("adaptive_time")
    
    @computed_field
    @cached_property
    def bifurcation_flag(self) -> bool:
        """
        Check if bifurcation module is enabled
        """
        return self.bifurcation is not None and self.bifurcation.num_bifurcation_paths > 0

    @cached_property
    def base_grid(self) -> Callable:
        return lambda META: (triton.cdiv(self.base.num_catchments, META["BLOCK_SIZE"]),)

    @cached_property
    def bifurcation_grid(self) -> Callable:
        return lambda META: (triton.cdiv(self.bifurcation.num_bifurcation_paths, META["BLOCK_SIZE"]),) if self.bifurcation_flag else None

    def step_advance(self, runoff: torch.Tensor, time_step: float, default_num_sub_steps: int, current_time: Optional[datetime]) -> None:
        """
        Advance model by one time step
        This method orchestrates the entire step advance process, including outflow, inflow, and flood stage calculations
        """
        if self.adaptive_time is not None:
            self.adaptive_time.min_time_sub_step.fill_(float('inf'))
            compute_adaptive_time_step_kernel[self.base_grid](
                is_reservoir_ptr=self.base.is_reservoir,
                downstream_idx_ptr=self.base.downstream_idx,
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
            num_sub_steps = int(round(time_step / self.adaptive_time.min_time_sub_step.item() - 0.01) + 1)
            time_sub_step = time_step / num_sub_steps
        else:
            num_sub_steps = default_num_sub_steps
            time_sub_step = time_step / num_sub_steps
        if self.log is not None:
            self.log.set_time(time_sub_step, num_sub_steps, current_time)
        for sub_step in range(num_sub_steps):
            self.do_one_sub_step(time_sub_step, runoff, sub_step, num_sub_steps)
        self.finalize_time_step(current_time)
        if self.log is not None:
            self.log.write_step()

    def do_one_sub_step(self, time_sub_step: float, runoff: torch.Tensor, sub_step: int, num_sub_steps: int) -> None:
        """Execute one sub time step calculation"""
        # Outflow computation
        compute_outflow_kernel[self.base_grid](
            is_river_mouth_ptr=self.base.is_river_mouth,
            is_reservoir_ptr=self.base.is_reservoir,
            downstream_idx_ptr=self.base.downstream_idx,
            river_outflow_ptr=self.base.river_outflow,
            flood_outflow_ptr=self.base.flood_outflow,
            river_manning_ptr=self.base.river_manning,
            flood_manning_ptr=self.base.flood_manning,
            river_depth_ptr=self.base.river_depth,
            river_width_ptr=self.base.river_width,
            river_length_ptr=self.base.river_length,
            river_elevation_ptr=self.base.river_elevation,
            river_storage_ptr=self.base.river_storage,
            catchment_elevation_ptr=self.base.catchment_elevation,
            downstream_distance_ptr=self.base.downstream_distance,
            flood_depth_ptr=self.base.flood_depth,
            flood_storage_ptr=self.base.flood_storage,
            river_cross_section_depth_ptr=self.base.river_cross_section_depth,
            flood_cross_section_depth_ptr=self.base.flood_cross_section_depth,
            flood_cross_section_area_ptr=self.base.flood_cross_section_area,
            total_storage_ptr=self.base.total_storage,
            outgoing_storage_ptr=self.base.outgoing_storage,
            water_surface_elevation_ptr=self.base.water_surface_elevation,
            gravity=self.base.gravity,
            time_step=time_sub_step,
            num_catchments=self.base.num_catchments,
            BLOCK_SIZE=self.BLOCK_SIZE
        )
        
        # Bifurcation outflow computation
        if self.bifurcation_flag:
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
        
        # Flood stage computation
        if self.log is not None:
            compute_flood_stage_log_kernel[self.base_grid](
                river_inflow_ptr=self.base.river_inflow,
                flood_inflow_ptr=self.base.flood_inflow,
                river_outflow_ptr=self.base.river_outflow,
                flood_outflow_ptr=self.base.flood_outflow,
                global_bifurcation_outflow_ptr=self.base.global_bifurcation_outflow,
                runoff_ptr=runoff,
                time_step=time_sub_step,
                river_storage_ptr=self.base.river_storage,
                flood_storage_ptr=self.base.flood_storage,
                outgoing_storage_ptr=self.base.outgoing_storage,
                river_depth_ptr=self.base.river_depth,
                flood_depth_ptr=self.base.flood_depth,
                flood_fraction_ptr=self.base.flood_fraction,
                flood_area_ptr=self.base.flood_area,
                river_max_storage_ptr=self.base.river_max_storage,
                total_storage_table_ptr=self.base.total_storage_table,
                flood_depth_table_ptr=self.base.flood_depth_table,
                total_width_table_ptr=self.base.total_width_table,
                flood_gradient_table_ptr=self.base.flood_gradient_table,
                catchment_area_ptr=self.base.catchment_area,
                river_width_ptr=self.base.river_width,
                river_length_ptr=self.base.river_length,
                total_storage_pre_sum_ptr=self.log.total_storage_pre_sum,
                total_storage_next_sum_ptr=self.log.total_storage_next_sum,
                total_storage_new_sum_ptr=self.log.total_storage_new_sum,
                total_inflow_sum_ptr=self.log.total_inflow_sum,
                total_outflow_sum_ptr=self.log.total_outflow_sum,
                total_storage_stage_sum_ptr=self.log.total_storage_stage_sum,
                river_storage_sum_ptr=self.log.river_storage_sum,
                flood_storage_sum_ptr=self.log.flood_storage_sum,
                flood_area_sum_ptr=self.log.flood_area_sum,
                total_inflow_error_sum_ptr=self.log.total_inflow_error_sum,
                total_stage_error_sum_ptr=self.log.total_stage_error_sum,
                num_catchments=self.base.num_catchments,
                current_step=sub_step,
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
                river_storage_ptr=self.base.river_storage,
                flood_storage_ptr=self.base.flood_storage,
                outgoing_storage_ptr=self.base.outgoing_storage,
                river_depth_ptr=self.base.river_depth,
                flood_depth_ptr=self.base.flood_depth,
                flood_fraction_ptr=self.base.flood_fraction,
                flood_area_ptr=self.base.flood_area,
                river_max_storage_ptr=self.base.river_max_storage,
                total_storage_table_ptr=self.base.total_storage_table,
                flood_depth_table_ptr=self.base.flood_depth_table,
                total_width_table_ptr=self.base.total_width_table,
                flood_gradient_table_ptr=self.base.flood_gradient_table,
                catchment_area_ptr=self.base.catchment_area,
                river_width_ptr=self.base.river_width,
                river_length_ptr=self.base.river_length,
                num_catchments=self.base.num_catchments,
                num_flood_levels=self.base.num_flood_levels,
                BLOCK_SIZE=self.BLOCK_SIZE
            )
        self.update_statistics(
            weight=num_sub_steps,
            refresh=(sub_step == num_sub_steps - 1)
        )

    def save_states(self, filepath: Path) -> None:
        """
        Save current simulation states
        
        Args:
            filepath: Save path
        """
        
        # Save states for all modules
        for module_name in self.opened_modules:
            module_instance = self._modules.get(module_name)
            if module_instance is not None:
                self.save_h5data(module_name, module_instance)
        
        print(f"States saved to: {filepath}")