# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Master controller class for managing all CaMa-Flood-GPU modules using Pydantic v2.
"""
from datetime import datetime
from functools import cached_property, partial
from typing import Any, ClassVar, Dict, Optional, Self, Type, Union

import cftime
import torch
from hydroforge.modeling.model import AbstractModel
from hydroforge.runtime.cuda_graph import CUDAGraphMixin
from pydantic import PrivateAttr, computed_field, model_validator
from torch import distributed as dist

from cmfgpu.modules.adaptive_time import AdaptiveTimeModule
from cmfgpu.modules.base import BaseModule
from cmfgpu.modules.bifurcation import BifurcationModule
from cmfgpu.modules.levee import LeveeModule
from cmfgpu.modules.log import LogModule
from cmfgpu.modules.reservoir import ReservoirModule
from cmfgpu.phys.adaptive_time import compute_adaptive_time_step
from cmfgpu.phys.bifurcation import (compute_bifurcation_inflow,
                                     compute_bifurcation_outflow)
from cmfgpu.phys.levee import (compute_levee_bifurcation_outflow,
                               compute_levee_stage, compute_levee_stage_log)
from cmfgpu.phys.outflow import compute_inflow, compute_outflow
from cmfgpu.phys.reservoir import compute_reservoir_outflow
from cmfgpu.phys.storage import compute_flood_stage, compute_flood_stage_log


class CaMaFlood(CUDAGraphMixin, AbstractModel):
    """
    CaMa-Flood GPU model master controller class
    """

    module_list: ClassVar[Dict[str, Type[BaseModule]]] = {
        "base": BaseModule,
        "bifurcation": BifurcationModule,
        "log": LogModule,
        "adaptive_time": AdaptiveTimeModule,
        "levee": LeveeModule,
        "reservoir": ReservoirModule,
    }
    group_by: ClassVar[str] = "catchment_basin_id"
    _stats_elapsed_time: float = PrivateAttr(default=0.0)
    _stats_start_time: Optional[Union[datetime, cftime.datetime]] = PrivateAttr(default=None)
    _stats_macro_step: int = PrivateAttr(default=0)
    _stats_cg: Optional[Any] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        # Auto-enable CUDA graphs for backends that support it
        from hydroforge.runtime.backend import KERNEL_BACKEND
        if KERNEL_BACKEND in ("triton", "cuda") and torch.cuda.is_available():
            self.enable_cuda_graph()

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

    @cached_property
    def reservoir(self) -> Optional[ReservoirModule]:
        return self.get_module("reservoir")
    
    @computed_field
    @cached_property
    def bifurcation_flag(self) -> bool:
        return self.bifurcation is not None

    @computed_field
    @cached_property
    def levee_flag(self) -> bool:
        return self.levee is not None

    @computed_field
    @cached_property
    def reservoir_flag(self) -> bool:
        return self.reservoir is not None
    
    @model_validator(mode="after")
    def validate_log_compatibility(self) -> Self:
        if self.num_trials is not None and self.num_trials > 1 and "log" in self.opened_modules:
            raise ValueError("The 'log' module cannot be used when num_trials > 1.")
        return self

    @model_validator(mode='after')
    def validate_backend_precision(self) -> Self:
        """Non-triton/cuda backends only support float32, no mixed precision."""
        from hydroforge.runtime.backend import KERNEL_BACKEND
        if KERNEL_BACKEND not in ("triton", "cuda"):
            if self.precision != "float32":
                raise ValueError(
                    f"Backend '{KERNEL_BACKEND}' only supports float32 precision, "
                    f"got '{self.precision}'. Use the triton (CUDA) backend for float64."
                )
            if self.mixed_precision:
                raise ValueError(
                    f"Backend '{KERNEL_BACKEND}' does not support mixed precision. "
                    f"Set mixed_precision=False or use the triton backend."
                )
        return self

    @model_validator(mode='after')
    def validate_cuda_backend_limitations(self) -> Self:
        """CUDA C++ / Metal backends do not support batched (num_trials>1);
        CUDA C++ additionally lacks the log module kernels."""
        from hydroforge.runtime.backend import KERNEL_BACKEND
        if KERNEL_BACKEND not in ("cuda", "metal"):
            return self
        if self.num_trials is not None and self.num_trials > 1:
            raise ValueError(
                f"'{KERNEL_BACKEND}' backend does not support batched execution "
                f"(num_trials={self.num_trials}). Use the triton backend instead."
            )
        if KERNEL_BACKEND == "cuda" and "log" in self.opened_modules:
            raise ValueError(
                f"'{KERNEL_BACKEND}' backend does not implement the 'log' module kernels "
                "(compute_flood_stage_log, compute_levee_stage_log). "
                "Use the triton or torch backend instead."
            )
        # Auto-configure CUDA storage precision to match the model's setting.
        # mixed_precision=False → all float32 → compile CUDA with -DCMF_STORAGE_FLOAT
        if KERNEL_BACKEND == "cuda" and not self.mixed_precision:
            from cmfgpu.phys.cuda import configure_storage_precision
            configure_storage_precision(use_float32=True)
        return self

    @model_validator(mode="after")
    def init_dam_cell_storage(self) -> Self:
        """
        At cold start, bump river_storage at dam cells to conservation_volume,
        following Fortran CMF_DAMOUT_INIT:
          P2DAMSTO(ISEQ) = MAX(P2RIVSTO(ISEQ)+P2FLDSTO(ISEQ), DamVol_cons)
          P2RIVSTO(ISEQ) = P2DAMSTO(ISEQ)
          P2FLDSTO(ISEQ) = 0
        """
        if not self.reservoir_flag:
            return self
        res_idx = self.reservoir.reservoir_catchment_idx   # (num_reservoirs,)
        con_vol = self.reservoir.conservation_volume        # (num_reservoirs,)
        current = self.base.river_storage[res_idx]          # (num_reservoirs,)
        need_fix = current < con_vol.to(current.dtype)
        if need_fix.any():
            n_fixed = int(need_fix.sum().item())
            self.base.river_storage[res_idx] = torch.where(
                need_fix,
                con_vol.to(self.base.river_storage.dtype),
                current,
            )
            # Also zero out flood_storage at fixed dam cells (Fortran: P2FLDSTO=0)
            flood_current = self.base.flood_storage[res_idx]
            self.base.flood_storage[res_idx] = torch.where(
                need_fix,
                torch.zeros_like(flood_current),
                flood_current,
            )
            print(f"  [init_dam_cell_storage] Clamped {n_fixed}/{len(con_vol)} dam-cell "
                  f"river_storage to conservation_volume")
        return self

    @model_validator(mode="after")
    def mask_bifurcation_at_dam_cells(self) -> Self:
        """
        Disable bifurcation at dam and upstream-of-dam cells by setting
        bifurcation elevation to 1E20, following Fortran CMF_DAMOUT_INIT:
          IF( I1DAM(ISEQP)>0 .or. I1DAM(JSEQP)>0 ) PTH_ELV(IPTH,:)=1.E20
        """
        if not self.reservoir_flag or not self.bifurcation_flag:
            return self
        is_dam = self.base.is_dam_related
        if is_dam is None:
            return self
        # Check each bifurcation path: if upstream or downstream touches a dam-related cell
        up_idx = self.bifurcation.bifurcation_catchment_idx
        dn_idx = self.bifurcation.bifurcation_downstream_idx
        up_is_dam = is_dam[up_idx]
        dn_is_dam = is_dam[dn_idx]
        mask_paths = up_is_dam | dn_is_dam  # (num_paths,)
        if mask_paths.any():
            n_masked = int(mask_paths.sum().item())
            self.bifurcation.bifurcation_elevation[mask_paths] = 1.0e20
            print(f"  Masked {n_masked} bifurcation paths at dam-related cells (elevation → 1E20)")
        return self

    # ------------------------------------------------------------------ #
    # Bound kernel callables (built once via partial; only dynamic args at call-time)
    # ------------------------------------------------------------------ #
    @cached_property
    def _call_outflow(self):
        return partial(compute_outflow,
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
            time_step_ptr=self.base.time_step,
            num_catchments=self.base.num_catchments,
            BLOCK_SIZE=self.BLOCK_SIZE,
            HAS_BIFURCATION=self.bifurcation_flag,
            is_dam_upstream_ptr=self.base.is_dam_upstream,
            HAS_RESERVOIR=self.reservoir_flag,
            MIN_KINEMATIC_SLOPE=self.base.min_kinematic_slope,
            num_trials=self.num_trials,
            batched_river_manning=self.base.batched_river_manning,
            batched_flood_manning=self.base.batched_flood_manning,
            batched_river_width=self.base.batched_river_width,
            batched_river_length=self.base.batched_river_length,
            batched_river_height=self.base.batched_river_height,
            batched_catchment_elevation=self.base.batched_catchment_elevation,
        )

    @cached_property
    def _call_reservoir_outflow(self):
        return partial(compute_reservoir_outflow,
            reservoir_catchment_idx_ptr=self.reservoir.reservoir_catchment_idx,
            downstream_idx_ptr=self.base.downstream_idx,
            reservoir_total_inflow_ptr=self.base.reservoir_total_inflow,
            river_outflow_ptr=self.base.river_outflow,
            flood_outflow_ptr=self.base.flood_outflow,
            river_storage_ptr=self.base.river_storage,
            flood_storage_ptr=self.base.flood_storage,
            conservation_volume_ptr=self.reservoir.conservation_volume,
            emergency_volume_ptr=self.reservoir.emergency_volume,
            adjustment_volume_ptr=self.reservoir.adjustment_volume,
            normal_outflow_ptr=self.reservoir.effective_normal_outflow,
            adjustment_outflow_ptr=self.reservoir.adjustment_outflow,
            flood_control_outflow_ptr=self.reservoir.flood_control_outflow,
            total_storage_ptr=self.base.total_storage,
            outgoing_storage_ptr=self.base.outgoing_storage,
            num_reservoirs=self.base.num_reservoirs,
            time_step_ptr=self.base.time_step,
            BLOCK_SIZE=self.BLOCK_SIZE,
        )

    @cached_property
    def _call_bif_outflow(self):
        kw = dict(
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
            time_step_ptr=self.base.time_step,
            num_bifurcation_paths=self.bifurcation.num_bifurcation_paths,
            num_bifurcation_levels=self.bifurcation.num_bifurcation_levels,
            BLOCK_SIZE=self.BLOCK_SIZE,
            num_trials=self.num_trials,
            num_catchments=self.base.num_catchments,
            batched_bifurcation_manning=self.bifurcation.batched_bifurcation_manning,
            batched_bifurcation_width=self.bifurcation.batched_bifurcation_width,
            batched_bifurcation_length=self.bifurcation.batched_bifurcation_length,
            batched_bifurcation_elevation=self.bifurcation.batched_bifurcation_elevation,
        )
        if self.levee_flag:
            kw['protected_water_surface_elevation_ptr'] = self.base.protected_water_surface_elevation
            return partial(compute_levee_bifurcation_outflow, **kw)
        return partial(compute_bifurcation_outflow, **kw)

    @cached_property
    def _call_inflow(self):
        return partial(compute_inflow,
            downstream_idx_ptr=self.base.downstream_idx,
            river_outflow_ptr=self.base.river_outflow,
            flood_outflow_ptr=self.base.flood_outflow,
            river_storage_ptr=self.base.river_storage,
            flood_storage_ptr=self.base.flood_storage,
            outgoing_storage_ptr=self.base.outgoing_storage,
            river_inflow_ptr=self.base.river_inflow,
            flood_inflow_ptr=self.base.flood_inflow,
            limit_rate_ptr=self.base.limit_rate,
            reservoir_total_inflow_ptr=self.base.reservoir_total_inflow,
            is_reservoir_ptr=self.base.is_reservoir,
            num_catchments=self.base.num_catchments,
            HAS_RESERVOIR=self.reservoir_flag,
            BLOCK_SIZE=self.BLOCK_SIZE,
            num_trials=self.num_trials,
        )

    @cached_property
    def _call_bif_inflow(self):
        return partial(compute_bifurcation_inflow,
            bifurcation_catchment_idx_ptr=self.bifurcation.bifurcation_catchment_idx,
            bifurcation_downstream_idx_ptr=self.bifurcation.bifurcation_downstream_idx,
            limit_rate_ptr=self.base.limit_rate,
            bifurcation_outflow_ptr=self.bifurcation.bifurcation_outflow,
            global_bifurcation_outflow_ptr=self.base.global_bifurcation_outflow,
            num_bifurcation_paths=self.bifurcation.num_bifurcation_paths,
            num_bifurcation_levels=self.bifurcation.num_bifurcation_levels,
            BLOCK_SIZE=self.BLOCK_SIZE,
            num_trials=self.num_trials,
            num_catchments=self.base.num_catchments,
        )

    @cached_property
    def _call_flood_stage(self):
        return partial(compute_flood_stage,
            river_inflow_ptr=self.base.river_inflow,
            flood_inflow_ptr=self.base.flood_inflow,
            river_outflow_ptr=self.base.river_outflow,
            flood_outflow_ptr=self.base.flood_outflow,
            global_bifurcation_outflow_ptr=self.base.global_bifurcation_outflow,
            outgoing_storage_ptr=self.base.outgoing_storage,
            river_storage_ptr=self.base.river_storage,
            flood_storage_ptr=self.base.flood_storage,
            protected_storage_ptr=self.base.protected_storage,
            river_depth_ptr=self.base.river_depth,
            flood_depth_ptr=self.base.flood_depth,
            protected_depth_ptr=self.base.protected_depth,
            flood_fraction_ptr=self.base.flood_fraction,
            river_height_ptr=self.base.river_height,
            flood_depth_table_ptr=self.base.flood_depth_table,
            catchment_area_ptr=self.base.catchment_area,
            river_width_ptr=self.base.river_width,
            river_length_ptr=self.base.river_length,
            num_catchments=self.base.num_catchments,
            time_step_ptr=self.base.time_step,
            num_flood_levels=self.base.num_flood_levels,
            BLOCK_SIZE=self.BLOCK_SIZE,
            HAS_BIFURCATION=self.bifurcation_flag,
            num_trials=self.num_trials,
            batched_river_height=self.base.batched_river_height,
            batched_flood_depth_table=self.base.batched_flood_depth_table,
            batched_catchment_area=self.base.batched_catchment_area,
            batched_river_width=self.base.batched_river_width,
            batched_river_length=self.base.batched_river_length,
        )

    @cached_property
    def _call_flood_stage_log(self):
        return partial(compute_flood_stage_log,
            river_inflow_ptr=self.base.river_inflow,
            flood_inflow_ptr=self.base.flood_inflow,
            river_outflow_ptr=self.base.river_outflow,
            flood_outflow_ptr=self.base.flood_outflow,
            global_bifurcation_outflow_ptr=self.base.global_bifurcation_outflow,
            outgoing_storage_ptr=self.base.outgoing_storage,
            river_storage_ptr=self.base.river_storage,
            flood_storage_ptr=self.base.flood_storage,
            protected_storage_ptr=self.base.protected_storage,
            river_depth_ptr=self.base.river_depth,
            flood_depth_ptr=self.base.flood_depth,
            protected_depth_ptr=self.base.protected_depth,
            flood_fraction_ptr=self.base.flood_fraction,
            river_height_ptr=self.base.river_height,
            flood_depth_table_ptr=self.base.flood_depth_table,
            catchment_area_ptr=self.base.catchment_area,
            river_width_ptr=self.base.river_width,
            river_length_ptr=self.base.river_length,
            is_levee_ptr=self.base.is_levee,
            log_sums_ptr=self.log.log_sums,
            num_catchments=self.base.num_catchments,
            time_step_ptr=self.base.time_step,
            current_step_ptr=self.base.current_step,
            num_flood_levels=self.base.num_flood_levels,
            log_buffer_size=self.log.log_buffer_size,
            BLOCK_SIZE=self.BLOCK_SIZE,
            HAS_BIFURCATION=self.bifurcation_flag,
        )

    @cached_property
    def _call_levee_stage(self):
        return partial(compute_levee_stage,
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
            num_levees=self.base.num_levees,
            num_flood_levels=self.base.num_flood_levels,
            BLOCK_SIZE=self.BLOCK_SIZE,
            num_trials=self.num_trials,
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

    @cached_property
    def _call_levee_stage_log(self):
        return partial(compute_levee_stage_log,
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
            log_sums_ptr=self.log.log_sums,
            num_levees=self.base.num_levees,
            current_step_ptr=self.base.current_step,
            num_flood_levels=self.base.num_flood_levels,
            log_buffer_size=self.log.log_buffer_size,
            BLOCK_SIZE=self.BLOCK_SIZE,
        )

    @cached_property
    def _call_adaptive_time(self):
        return partial(compute_adaptive_time_step,
            river_depth_ptr=self.base.river_depth,
            downstream_distance_ptr=self.base.downstream_distance,
            is_dam_related_ptr=self.base.is_dam_related,
            max_sub_steps_ptr=self.adaptive_time.max_sub_steps,
            adaptive_time_factor=self.adaptive_time.adaptive_time_factor,
            gravity=self.base.gravity,
            num_catchments=self.base.num_catchments,
            BLOCK_SIZE=self.BLOCK_SIZE,
            HAS_RESERVOIR=self.reservoir_flag,
            num_trials=self.num_trials,
            batched_downstream_distance=self.base.batched_downstream_distance,
        )

    # ------------------------------------------------------------------ #
    # CUDA Graph support (via CUDAGraphMixin)
    # ------------------------------------------------------------------ #
    def cuda_graph_target(self, *, runoff, **kw):
        """The kernel sequence captured into a CUDA Graph."""
        self.do_one_sub_step(runoff, output_enabled=(self.log is not None))

    def disable_cuda_graph(self) -> None:
        """Disable CUDA Graph and release all cached graphs including statistics."""
        self._stats_cg = None
        super().disable_cuda_graph()

    def _stats_graph_replay(self, sub_step, num_sub_steps, flags, weight,
                            total_weight, num_macro_steps, macro_step_index,
                            BLOCK_SIZE):
        """Replay or capture a CUDA graph for the statistics aggregator kernel.

        All varying scalars are stored as 1-element device tensors in
        ``_kernel_states`` so the graph is address-stable and a single
        capture can be replayed for *every* combination of values.
        """
        agg = self._statistics_aggregator
        states = agg._kernel_states

        # Fill scalar tensors — graph loads from fixed addresses
        states['__weight'].fill_(weight)
        states['__total_weight'].fill_(total_weight)
        states['__num_macro_steps'].fill_(num_macro_steps)
        states['__sub_step'].fill_(sub_step)
        states['__num_sub_steps'].fill_(num_sub_steps)
        states['__flags'].fill_(flags)
        states['__macro_step_index'].fill_(macro_step_index)

        if self._stats_cg is None:
            pool = self.__dict__.get("_cg_pool")
            # Warmup
            for _ in range(3):
                agg._aggregator_function(states, BLOCK_SIZE)
            # Capture
            self._stats_cg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._stats_cg, pool=pool):
                agg._aggregator_function(states, BLOCK_SIZE)

        self._stats_cg.replay()

    @torch.inference_mode()
    def step_advance(
        self,
        runoff: torch.Tensor,
        time_step: float,
        default_num_sub_steps: int,
        current_time: Optional[Union[datetime, cftime.datetime]],
        stat_is_first: bool = True,
        stat_is_last: bool = True,
        stat_is_outer_first: Optional[bool] = None,
        stat_is_outer_last: Optional[bool] = None,
        output_enabled: bool = True) -> None:
        """
        Advance the model by one time step using the provided runoff input.

        Notes on time-weighted statistics:
          - Time-weighted accumulation is performed every sub-step with weight = dt (seconds).
          - If stat_is_first is True, the accumulation window is reset at the first sub-step.
          - If stat_is_last is True, the window is finalized at the last sub-step.
          - By default (stat_is_first=True, stat_is_last=True), each call to step_advance forms an
            independent window and is saved once at the end of this call.

        Args:
            runoff (torch.Tensor): Input runoff tensor for this time step.
            time_step (float): Duration of the time step (seconds).
            default_num_sub_steps (int): Default sub-steps if adaptive time stepping is disabled.
            current_time (Optional[datetime]): Current simulation time. Used for logging.
            stat_is_first (bool): Whether this call starts a new statistics window (resets accumulation).
            stat_is_last (bool): Whether this call ends the current statistics window (finalize average).
            stat_is_outer_first (bool, optional): Explicit start of outer window. Defaults to None.
            stat_is_outer_last (bool, optional): Explicit end of outer window. Defaults to None.
        """
        self.execute_parameter_change_plan(current_time)

        # Handle defaults for outer stats (default to False, only running inner stats)
        if stat_is_outer_first is None:
            stat_is_outer_first = False
        if stat_is_outer_last is None:
            stat_is_outer_last = False

        if self.adaptive_time is not None:
            self.adaptive_time.max_sub_steps.fill_(0)
            self._call_adaptive_time(time_step=time_step)
            if self.world_size > 1:
                dist.all_reduce(self.adaptive_time.max_sub_steps, op=dist.ReduceOp.MAX)
            
            # Take the maximum across all trials if batched
            num_sub_steps = int(self.adaptive_time.max_sub_steps.max().item())
            if num_sub_steps < 1:
                num_sub_steps = 1
            time_sub_step = time_step / num_sub_steps
        else:
            num_sub_steps = int(default_num_sub_steps)
            time_sub_step = time_step / num_sub_steps
        if self.log is not None:
            self.log.set_time(time_sub_step, num_sub_steps, current_time)

        if stat_is_first:
            # Reset elapsed time counter at the beginning of a stats window
            self._stats_elapsed_time = 0.0
            self._stats_start_time = current_time
            # Keep _stats_macro_step cumulative across inner windows
            
        if stat_is_outer_first:
            self._stats_macro_step = 0

        # Check if output is enabled
        if self.output_start_time is not None and current_time is not None:
            if current_time < self.output_start_time:
                output_enabled = False

        # Pre-compute constants for the sub-step loop
        flags = (int(stat_is_first) | (int(stat_is_last) << 1) |
                 (int(stat_is_outer_first) << 2) | (int(stat_is_outer_last) << 3))
        total_weight = ((0.0 if stat_is_first else self._stats_elapsed_time)
                        + num_sub_steps * time_sub_step)

        # Determine if stats CUDA graph is safe to use:
        # only when no outer windowing (num_macro_steps/macro_step_index don't affect output)
        # torch backend's @torch.compile is incompatible with CUDA graph stream capture
        from hydroforge.runtime.backend import KERNEL_BACKEND
        agg = self._statistics_aggregator
        use_stats_cg = (self.cuda_graph_enabled
                        and KERNEL_BACKEND != "torch"
                        and not stat_is_outer_first and not stat_is_outer_last
                        and output_enabled and agg is not None and agg._aggregator_generated)

        self.base.time_step.fill_(time_sub_step)

        # ------------------------------------------------------------------ #
        # Sub-step 0: use standard code paths (handles first-time capture)
        # ------------------------------------------------------------------ #
        self.base.current_step.fill_(0)
        if self.cuda_graph_enabled:
            self.cuda_graph_replay(cache_key=0, runoff=runoff)
        else:
            self.do_one_sub_step(runoff, output_enabled)
        self._stats_elapsed_time += time_sub_step

        if output_enabled:
            if use_stats_cg:
                is_inner_last_0 = bool(flags & 2) and (num_sub_steps == 1)
                if is_inner_last_0:
                    for out_name, is_outer in agg._output_is_outer.items():
                        if not is_outer:
                            agg._dirty_outputs.add(out_name)
                    agg._current_macro_step_count += 1.0
                self._stats_graph_replay(
                    sub_step=0,
                    num_sub_steps=num_sub_steps,
                    flags=flags,
                    weight=time_sub_step,
                    total_weight=total_weight,
                    num_macro_steps=agg._current_macro_step_count,
                    macro_step_index=agg._macro_step_index,
                    BLOCK_SIZE=self.BLOCK_SIZE,
                )
            else:
                self.update_statistics(
                    sub_step=0,
                    num_sub_steps=num_sub_steps,
                    flags=flags,
                    weight=time_sub_step,
                    total_weight=total_weight,
                    BLOCK_SIZE=self.BLOCK_SIZE
                )

        # ------------------------------------------------------------------ #
        # Sub-steps 1..N-1
        # ------------------------------------------------------------------ #
        if num_sub_steps > 1:
            if use_stats_cg:
                # === FAST PATH ===
                # Both physics and stats graphs are now captured (sub-step 0
                # ensured that).  Bypass cuda_graph_replay / _stats_graph_replay
                # to eliminate per-iteration overhead:
                #   - no **kwargs dict allocation
                #   - no redundant runoff.copy_() (runoff unchanged between sub-steps)
                #   - only fill the 1 scalar that actually changes (sub_step)
                #     (the other 6 constants were set in sub-step 0)
                phys_graph = self.__dict__["_cg_cache"][0][0]
                stats_graph = self._stats_cg
                states = agg._kernel_states
                sub_step_t = states['__sub_step']
                current_step_t = self.base.current_step
                last_sub = num_sub_steps - 1
                is_stat_last = bool(flags & 2)

                for sub_step in range(1, num_sub_steps):
                    current_step_t.fill_(sub_step)
                    phys_graph.replay()
                    sub_step_t.fill_(sub_step)
                    if sub_step == last_sub and is_stat_last:
                        for out_name, is_outer in agg._output_is_outer.items():
                            if not is_outer:
                                agg._dirty_outputs.add(out_name)
                        agg._current_macro_step_count += 1.0
                        states['__num_macro_steps'].fill_(agg._current_macro_step_count)
                    stats_graph.replay()

                self._stats_elapsed_time += (num_sub_steps - 1) * time_sub_step
            else:
                # Standard path for remaining sub-steps
                for sub_step in range(1, num_sub_steps):
                    self.base.current_step.fill_(sub_step)
                    if self.cuda_graph_enabled:
                        self.cuda_graph_replay(
                            cache_key=0,
                            runoff=runoff,
                        )
                    else:
                        self.do_one_sub_step(runoff, output_enabled)
                    self._stats_elapsed_time += time_sub_step
                    if output_enabled:
                        self.update_statistics(
                            sub_step=sub_step,
                            num_sub_steps=num_sub_steps,
                            flags=flags,
                            weight=time_sub_step,
                            total_weight=total_weight,
                            BLOCK_SIZE=self.BLOCK_SIZE
                        )

        # Reset elapsed counter after closing a window
        if stat_is_last:
            if output_enabled:
                self.finalize_time_step(self._stats_start_time if self._stats_start_time is not None else current_time)
            self._stats_elapsed_time = 0.0
            self._stats_start_time = None
            self._stats_macro_step += 1
        
        if self.log is not None:
            if self.world_size > 1:
                self.log.gather_results()
            if self.rank == 0 and output_enabled:
                self.log.write_step(self.log_path)
        if self.rank == 0:
            self.progress_tick()
            progress = self.format_progress()
            if progress:
                msg = f"Processed step at {current_time}, adaptive_time_step={num_sub_steps} | {progress}"
            else:
                msg = f"Processed step at {current_time}, adaptive_time_step={num_sub_steps}"
            print(f"\r\033[K{msg}", end="", flush=True)
    def do_one_sub_step(self, runoff: torch.Tensor, output_enabled: bool = True) -> None:
        """Execute one sub time step calculation."""
        self._call_outflow()

        if self.reservoir_flag:
            self._call_reservoir_outflow(runoff_ptr=runoff)

        if self.bifurcation_flag:
            self._call_bif_outflow()

        self._call_inflow()

        if self.bifurcation_flag:
            self._call_bif_inflow()

        if self.log is not None and output_enabled and compute_flood_stage_log is not None and self.num_trials is None:
            self._call_flood_stage_log(runoff_ptr=runoff)
        else:
            self._call_flood_stage(
                runoff_ptr=runoff,
                batched_runoff=(runoff.ndim > 1 and runoff.shape[0] == (self.num_trials or 0)),
            )

        if self.levee_flag:
            if self.log is not None and output_enabled and compute_levee_stage_log is not None and self.num_trials is None:
                self._call_levee_stage_log()
            else:
                self._call_levee_stage()
