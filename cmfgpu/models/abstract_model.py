# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import (Any, ClassVar, Dict, Iterator, List, Literal, Optional,
                    Self, Tuple, Type, Union)

import cftime
import numpy as np
import torch
import torch.distributed as dist
from pydantic import (BaseModel, ConfigDict, Field, PrivateAttr,
                      field_validator, model_validator)

from cmfgpu.models.aggregator import StatisticsAggregator
from cmfgpu.models.utils import compute_group_to_rank
from cmfgpu.modules.abstract_module import AbstractModule
from cmfgpu.params.input_proxy import InputProxy
from cmfgpu.utils import find_indices_in_torch


@dataclass
class PlanItem:
    # User inputs
    variable_name: str
    start_time: Union[datetime, cftime.datetime]
    active_steps: int = 1
    delta: Union[float, torch.Tensor] = 0.0
    target_value: Optional[Union[float, torch.Tensor]] = None
    target_ids: Optional[Union[List[int], torch.Tensor]] = None
    
    # Cached execution context (resolved once)
    _module: Optional[Any] = None
    _attr_name: str = ""
    _indices: Optional[torch.Tensor] = None
    _is_ready: bool = False

    @property
    def is_set_value(self) -> bool:
        return self.target_value is not None

    @property
    def is_incremental(self) -> bool:
        return not self.is_set_value

@dataclass
class ActivePlan:
    item: PlanItem
    steps_executed: int = 0
    executed_once: bool = False



class AbstractModel(BaseModel, ABC):
    """
    Master controller for CaMa-Flood-GPU workflow using the AbstractModule hierarchy.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra='forbid'
    )

    # Class variables
    module_list: ClassVar[Dict[str, Type[AbstractModule]]] = {}
    group_by: ClassVar[str] = "group_id"  # Default group variable, override in subclasses

    # Instance fields
    experiment_name: str = Field(default="experiment", description="Name of the experiment")
    input_proxy: InputProxy = Field(default=..., description="InputProxy object containing model data")
    output_dir: Path = Field(default_factory=lambda: Path("./out"), description="Path to the output directory")
    opened_modules: List[str] = Field(default_factory=list, description="List of active modules")
    # Preferred shape: dict[op -> str | list[str]]; op in {mean,max,min,last};
    # one variable can appear under multiple ops.
    variables_to_save: Optional[Dict[str, Union[str, List[str]]]] = Field(
        None, description="Statistics to save, in the form {op: [vars...]}. Supported ops: mean, max, min, last."
    )
    precision: Literal["float32", "float64"] = Field(default="float32", description="Precision of the model")
    world_size: int = Field(default=1, description="Total number of distributed processes")
    rank: int = Field(default=0, description="Current process rank in distributed setup")
    device: torch.device = Field(default=torch.device("cpu"), description="Device for tensors (e.g., 'cuda:0', 'cpu')")
    BLOCK_SIZE: int = Field(default=256, description="GPU block size for kernels")
    output_workers: int = Field(default=2, description="Number of workers for writing output files")
    output_complevel: int = Field(default=4, description="Compression level for output NetCDF files", ge=0, le=9)
    output_split_by_year: bool = Field(default=False, description="Whether to split output files by year")
    num_trials: Optional[int] = Field(default=None, description="Number of parallel simulations (ensemble members)")
    save_kernels: bool = Field(default=False, description="Whether to save generated Triton kernels")

    _modules: Dict[str, AbstractModule] = PrivateAttr(default_factory=dict)
    _statistics_aggregator: Optional[StatisticsAggregator] = PrivateAttr(default=None)
    
    # Parameter Change Plan State
    _plans: List[PlanItem] = PrivateAttr(default_factory=list)
    _active_plans: List[ActivePlan] = PrivateAttr(default_factory=list)
    _next_plan_idx: int = PrivateAttr(default=0)
    _cached_grouped_plans: Optional[Dict[Tuple[int, str], List[ActivePlan]]] = PrivateAttr(default=None)

    @field_validator('num_trials')
    @classmethod
    def validate_num_trials(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 1:
            raise ValueError("num_trials must be greater than 1 if specified. For single trial, use None.")
        return v

    def _iter_all_fields(self, include_computed: bool = True) -> Iterator[Tuple[str, Type[AbstractModule], str, Any]]:
        """
        Iterate over all fields in all opened modules.
        Yields: (module_name, module_class, field_name, field_info)
        """
        for module_name in self.opened_modules:
            if module_name not in self.module_list:
                continue
            module_class = self.module_list[module_name]
            
            # Regular fields
            for name, info in module_class.get_model_fields().items():
                if name not in module_class.nc_excluded_fields:
                    yield module_name, module_class, name, info
            
            # Computed fields
            if include_computed:
                for name, info in module_class.get_model_computed_fields().items():
                    if name not in module_class.nc_excluded_fields:
                        yield module_name, module_class, name, info

    def _apply_grouped_changes(self, module: Any, attr: str, plans: List[ActivePlan]):
        try:
            current_val = getattr(module, attr)
            is_tensor = isinstance(current_val, torch.Tensor)
            
            # Optimization: Calculate global delta and set value first
            global_delta = 0.0
            global_set_value = None
            
            # Separate sparse updates
            sparse_updates = [] # List of (indices, value, is_set)

            # Sort plans: Set values first, then Incremental
            # This ensures that if we have Set and Add in the same step, Add is applied on top of Set
            plans.sort(key=lambda x: x.item.is_incremental)

            for active in plans:
                item = active.item
                
                # Determine value and type
                if item.is_set_value:
                    val = item.target_value
                    is_set = True
                else:
                    val = item.delta
                    is_set = False
                
                # Update execution counters
                active.steps_executed += 1
                active.executed_once = True

                # Check if global or sparse
                if item._indices is None:
                    if is_set:
                        global_set_value = val
                        global_delta = 0.0 # Reset delta if global set happens? 
                        # Logic: Set establishes baseline. Previous deltas are overwritten by Set.
                        # Subsequent deltas (in sorted order) will add to this baseline.
                    else:
                        global_delta += val
                else:
                    sparse_updates.append((item._indices, val, is_set))

            # Apply Global Changes
            if is_tensor:
                if global_set_value is not None:
                    current_val.fill_(global_set_value)
                
                if global_delta != 0.0:
                    current_val.add_(global_delta)
                
                # Apply Sparse Changes
                for indices, val, is_set in sparse_updates:
                    if is_set:
                        current_val[indices] = val
                    else:
                        current_val[indices] += val
            else:
                # Scalar handling
                new_val = current_val
                if global_set_value is not None:
                    new_val = global_set_value
                new_val += global_delta
                
                if sparse_updates:
                     print(f"ParameterChangePlan Warning: Sparse updates ignored for scalar variable {attr}.")
                
                setattr(module, attr, new_val)

        except Exception as e:
            print(f"ParameterChangePlan Error: Failed to update {attr}. {e}")

    def execute_parameter_change_plan(self, current_time: Union[datetime, cftime.datetime]) -> None:
        """
        Execute the plans for the current time step.
        """
        if current_time is None:
            return

        plans_changed = False

        # 1. Activate new plans
        while self._next_plan_idx < len(self._plans):
            plan = self._plans[self._next_plan_idx]
            if current_time >= plan.start_time:
                self._active_plans.append(ActivePlan(item=plan))
                self._next_plan_idx += 1
                plans_changed = True
            else:
                break
        
        if not self._active_plans:
            self._cached_grouped_plans = None
            return

        # 2. Filter finished plans and check if grouping update is needed
        valid_active_plans = []
        for active in self._active_plans:
            item = active.item
            is_finished = False
            
            if active.steps_executed >= item.active_steps:
                is_finished = True
            
            if not is_finished:
                valid_active_plans.append(active)
            else:
                plans_changed = True
        
        self._active_plans = valid_active_plans

        if not self._active_plans:
            self._cached_grouped_plans = None
            return

        # 3. Update cached grouping if needed
        if plans_changed or self._cached_grouped_plans is None:
            grouped_plans: Dict[Tuple[int, str], List[ActivePlan]] = {}
            for active in self._active_plans:
                if active.item._is_ready:
                    # Use id(module) as key because Pydantic models are not hashable
                    key = (id(active.item._module), active.item._attr_name)
                    if key not in grouped_plans:
                        grouped_plans[key] = []
                    grouped_plans[key].append(active)
            self._cached_grouped_plans = grouped_plans

        # 4. Execute grouped plans
        for (_, attr), plans in self._cached_grouped_plans.items():
            if not plans:
                continue
            module = plans[0].item._module
            self._apply_grouped_changes(module, attr, plans)

    def _resolve_id_tensor(self, module: Any, id_attr: Optional[str]) -> Optional[torch.Tensor]:
        """
        Helper to resolve the ID tensor from a module given an attribute path.
        """
        if not id_attr:
            return None
            
        if "." in id_attr:
            # Try to resolve nested attribute
            parts = id_attr.split(".")
            curr = module
            for part in parts:
                if hasattr(curr, part):
                    curr = getattr(curr, part)
                else:
                    return None
            return curr if isinstance(curr, torch.Tensor) else None
        elif hasattr(module, id_attr):
            return getattr(module, id_attr)
        return None

    def _resolve_plan_item(self, item: PlanItem):
        variable_map = self.variable_map
        
        if item.variable_name in variable_map:
            module, attr, id_attr = variable_map[item.variable_name]
            item._module = module
            item._attr_name = attr
            
            if item.target_ids is not None:
                # Handle nested attributes (e.g. base.levee_catchment_id)
                id_tensor = self._resolve_id_tensor(module, id_attr)
                
                if id_tensor is not None:
                    if item.target_ids.device != id_tensor.device:
                        item.target_ids = item.target_ids.to(id_tensor.device)
                    
                    indices = find_indices_in_torch(item.target_ids, id_tensor)
                    
                    # Strict check: All IDs must be found
                    if torch.any(indices < 0):
                        raise ValueError(f"ParameterChangePlan Error: Some target_ids for {item.variable_name} were not found in {id_attr}.")
                        
                    item._indices = indices
                else:
                    print(f"ParameterChangePlan Warning: Cannot find ID tensor '{id_attr}' for {item.variable_name}. Applying to ALL.")
            
            # Move tensor values to correct device
            if isinstance(item.delta, torch.Tensor):
                item.delta = item.delta.to(module.device)
            if isinstance(item.target_value, torch.Tensor):
                item.target_value = item.target_value.to(module.device)

            item._is_ready = True
        else:
            print(f"ParameterChangePlan Warning: Variable {item.variable_name} not found in model.")

    def add_parameter_change_plan(
        self,
        variable_name: str,
        start_time: Union[datetime, cftime.datetime],
        active_steps: int = 1,
        delta: Union[float, torch.Tensor] = 0.0,
        target_value: Optional[Union[float, torch.Tensor]] = None,
        target_ids: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> None:
        """
        Add a parameter change plan.
        """
        if active_steps < 1:
            raise ValueError("active_steps must be >= 1")

        if target_ids is not None and not isinstance(target_ids, torch.Tensor):
            target_ids = torch.tensor(target_ids, dtype=torch.int64)

        # Ensure tensor values are on the correct device if possible, 
        # but we don't have easy access to the module's device here until resolve.
        # We will handle device movement in _resolve_plan_item or execution.

        item = PlanItem(
            variable_name=variable_name,
            start_time=start_time,
            active_steps=active_steps,
            delta=delta,
            target_value=target_value,
            target_ids=target_ids
        )
        
        self._resolve_plan_item(item)
        self._plans.append(item)
        # Keep plans sorted by start time for efficient activation
        self._plans.sort(key=lambda x: x.start_time)
        # Reset execution pointer if plans change
        self._next_plan_idx = 0
        self._active_plans.clear()

    def set_variable_value(
        self,
        variable_name: str,
        value: Union[float, torch.Tensor],
        target_ids: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> None:
        """
        Directly set the value of a variable for specific IDs immediately.
        
        Args:
            variable_name: Name of the variable to update.
            value: New value (scalar or tensor).
            target_ids: List of IDs to apply the change to. If None, applies to all.
            
        Raises:
            ValueError: If variable not found or IDs not found.
        """
        if variable_name not in self.variable_map:
            raise ValueError(f"Variable '{variable_name}' not found in model.")
            
        module, attr, id_attr = self.variable_map[variable_name]
        current_val = getattr(module, attr)
        
        # Prepare value
        if isinstance(value, torch.Tensor):
            value = value.to(self.device)
        
        # Case 1: Global update
        if target_ids is None:
            if isinstance(current_val, torch.Tensor):
                current_val[:] = value
            else:
                setattr(module, attr, value)
            return

        # Case 2: Sparse update (requires ID resolution)
        if not isinstance(current_val, torch.Tensor):
            print(f"Warning: Ignoring target_ids for scalar variable '{variable_name}'. Updating globally.")
            setattr(module, attr, value)
            return

        # Resolve ID tensor
        id_tensor = self._resolve_id_tensor(module, id_attr)
        
        if id_tensor is None:
             raise ValueError(f"Cannot resolve ID tensor '{id_attr}' for variable '{variable_name}', so target_ids cannot be used.")

        # Prepare target_ids
        if not isinstance(target_ids, torch.Tensor):
            target_ids = torch.tensor(target_ids, dtype=torch.int64, device=self.device)
        else:
            target_ids = target_ids.to(self.device)
            
        # Ensure id_tensor is on correct device (should be)
        if id_tensor.device != target_ids.device:
            target_ids = target_ids.to(id_tensor.device)

        # Find indices
        indices = find_indices_in_torch(target_ids, id_tensor)
        
        # Validate
        if torch.any(indices < 0):
            raise ValueError(f"Some target_ids for '{variable_name}' were not found in '{id_attr}'.")
            
        # Apply
        current_val[indices] = value

    def summarize_plan(self) -> None:
        """
        Print a summary of the parameter change plan and check for conflicts.
        Raises ValueError if conflicts are detected (e.g. setting the same variable twice at the same time for the same location).
        """
        print(f"\n[rank {self.rank}] === Parameter Change Plan Summary ===")
        
        if not self._plans:
            print("No parameter change plans defined.")
            return

        # Sort by time
        sorted_plans = sorted(self._plans, key=lambda x: x.start_time)
        
        # Conflict Detection
        # Group SET plans by (variable, time)
        set_plans_map = {}
        
        for plan in sorted_plans:
            if plan.is_set_value:
                key = (plan.variable_name, plan.start_time)
                if key not in set_plans_map:
                    set_plans_map[key] = []
                set_plans_map[key].append(plan)

        conflicts = []
        
        for (var_name, time), plans in set_plans_map.items():
            if len(plans) > 1:
                # Check overlaps
                for i in range(len(plans)):
                    for j in range(i + 1, len(plans)):
                        p1 = plans[i]
                        p2 = plans[j]
                        
                        # If either targets ALL (None), it conflicts with everything
                        if p1.target_ids is None or p2.target_ids is None:
                            conflicts.append(f"Conflict: Variable '{var_name}' set multiple times at {time}. One or both plans target ALL.")
                            continue
                            
                        # Check intersection of IDs
                        # Ensure tensors are on CPU for set operation
                        ids1 = p1.target_ids
                        if isinstance(ids1, torch.Tensor):
                            ids1 = ids1.detach().cpu().numpy()
                        else:
                            ids1 = np.array(ids1)
                            
                        ids2 = p2.target_ids
                        if isinstance(ids2, torch.Tensor):
                            ids2 = ids2.detach().cpu().numpy()
                        else:
                            ids2 = np.array(ids2)
                        
                        # Use numpy intersect1d
                        intersection = np.intersect1d(ids1, ids2)
                        if intersection.size > 0:
                            sample_conflict = intersection[:5].tolist()
                            conflicts.append(f"Conflict: Variable '{var_name}' set multiple times at {time} for IDs {sample_conflict}...")

        # Print Summary Table
        print(f"{'Time':<25} | {'Variable':<20} | {'Type':<8} | {'Value':<10} | {'Steps':<10} | {'Target'}")
        print("-" * 100)
        
        for plan in sorted_plans:
            type_str = "SET" if plan.is_set_value else "ADD"
            
            # Handle Tensor values for display
            val = plan.target_value if plan.is_set_value else plan.delta
            if isinstance(val, torch.Tensor):
                if val.numel() == 1:
                    val_str = f"{val.item():.4g}"
                else:
                    val_str = "Tensor"
            else:
                val_str = f"{val:.4g}"

            dur_str = f"{plan.active_steps}" if plan.is_incremental else "-"
            
            if plan.target_ids is None:
                target_str = "ALL"
            else:
                count = len(plan.target_ids)
                # Resolve ID attribute name for display
                id_attr_name = "IDs"
                if plan.variable_name in self.variable_map:
                    _, _, id_attr = self.variable_map[plan.variable_name]
                    if id_attr:
                        id_attr_name = id_attr

                if count <= 5:
                    # Show IDs
                    ids_list = plan.target_ids.tolist() if isinstance(plan.target_ids, torch.Tensor) else plan.target_ids
                    target_str = f"{str(ids_list)} ({id_attr_name})"
                else:
                    target_str = f"{count} {id_attr_name}"
            
            print(f"{str(plan.start_time):<25} | {plan.variable_name:<20} | {type_str:<8} | {val_str:<10} | {dur_str:<10} | {target_str}")
            
        print("-" * 100)

        if conflicts:
            error_msg = "\n".join(conflicts)
            raise ValueError(f"Parameter Plan Conflicts Detected:\n{error_msg}")

    @cached_property
    def dtype(self) -> torch.dtype:
        return torch.float32 if self.precision == "float32" else torch.float64

    @cached_property
    def output_full_dir(self) -> Path:
        output_full_dir = self.output_dir / self.experiment_name
        return output_full_dir

    @cached_property
    def log_path(self) -> Path:
        log_path = self.output_full_dir / "log.txt"
        return log_path

    def check_namespace_conflicts(self) -> None:
        """
        Check for namespace conflicts across all opened modules.
        """
        field_definitions: Dict[str, Tuple[str, Any]] = {}

        for module_name, _, field_name, field_info in self._iter_all_fields(include_computed=True):
            if field_name in field_definitions:
                existing_module, existing_info = field_definitions[field_name]
                
                # Compare definitions
                # 1. Compare annotation (type)
                new_type = getattr(field_info, 'annotation', getattr(field_info, 'return_type', None))
                old_type = getattr(existing_info, 'annotation', getattr(existing_info, 'return_type', None))
                
                # 2. Compare json_schema_extra (shape, dtype, etc.)
                new_extra = getattr(field_info, 'json_schema_extra', {}) or {}
                old_extra = getattr(existing_info, 'json_schema_extra', {}) or {}
                
                if new_type != old_type or new_extra != old_extra:
                    raise ValueError(
                        f"Namespace conflict detected for field '{field_name}':\n"
                        f"  - Defined in '{existing_module}' with type={old_type}, extra={old_extra}\n"
                        f"  - Defined in '{module_name}' with type={new_type}, extra={new_extra}\n"
                        f"Please rename one of the fields to avoid ambiguity."
                    )
            else:
                field_definitions[field_name] = (module_name, field_info)

    def model_post_init(self, __context):
        """
        Post-initialization hook to validate opened modules and register them.
        """
        print(f"[rank {self.rank}]: Initializing ModelManager with opened modules:", self.opened_modules)
        
        self.check_namespace_conflicts()
        
        print(f"Using primary group variable: {self.group_by}")

        # Validate that all opened modules are registered
        module_data = self.shard_param()  # reads from NetCDF
        
        # Sort modules by dependency
        from graphlib import TopologicalSorter
        sorter = TopologicalSorter()
        for module_name in self.opened_modules:
            if module_name not in self.module_list:
                raise ValueError(f"Module {module_name} not found in module_list")
            deps = self.module_list[module_name].dependencies
            # Only include dependencies that are in opened_modules
            active_deps = [d for d in deps if d in self.opened_modules]
            sorter.add(module_name, *active_deps)
            
        # Get sorted order
        sorted_modules = list(sorter.static_order())

        for module_name in sorted_modules:
            if module_name not in self.opened_modules:
                continue

            # Register the module instance with data
            module_class = self.module_list[module_name]
            module_instance = module_class(
                opened_modules=self.opened_modules,
                rank=self.rank,
                device=self.device,
                world_size=self.world_size,
                precision=self.dtype,
                num_trials=self.num_trials,
                **self._modules,
                **module_data
            )
            self._modules[module_name] = module_instance

        self.initialize_statistics_aggregator()
        self.print_memory_summary()
        print("All modules initialized successfully.")

    def print_memory_summary(self) -> None:
        """
        Print a summary of memory usage by module.
        """
        total_memory = 0
        print(f"\n[rank {self.rank}] Memory Usage Summary:")
        print(f"{'Module':<30} | {'Memory (MB)':<15}")
        print(f"{'-' * 50}")
        
        for module_name in self.opened_modules:
            if module_name not in self._modules:
                continue
            module = self._modules[module_name]
            mem_bytes = module.get_memory_usage()
            mem_mb = mem_bytes / (1024 * 1024)
            total_memory += mem_bytes
            print(f"{module_name:<30} | {mem_mb:<15.2f}")
            
        print(f"{'-' * 50}")
        print(f"{'Total':<30} | {total_memory / (1024 * 1024):<15.2f} MB\n")

    def get_module(self, module_name: str) -> AbstractModule:
        return self._modules[module_name] if module_name in self.opened_modules else None

    @cached_property
    def variable_group_mapping(self) -> Dict[str, str]:
        """
        Build a mapping from each variable to its group variable.

        Returns:
            Dictionary mapping variable_name -> group_by_name
        """
        variable_group_mapping = {}

        # Iterate through all opened modules to collect field information
        for _, _, field_name, field_info in self._iter_all_fields(include_computed=False):
            json_schema_extra = getattr(field_info, 'json_schema_extra', None)
            if json_schema_extra is None:
                json_schema_extra = {}
            group_var = json_schema_extra.get('group_by', None)
            if group_var:
                variable_group_mapping[field_name] = group_var

        return variable_group_mapping

    @cached_property
    def variable_map(self) -> Dict[str, Tuple[AbstractModule, str, Optional[str]]]:
        """
        Map variable names to (module_instance, field_name, id_attr).
        This provides a unified way to lookup variables across all modules.
        """
        mapping = {}
        for module_name, _, field_name, field_info in self._iter_all_fields(include_computed=True):
            module = self.get_module(module_name)
            if module is None:
                continue
            
            # Determine ID attribute for coordinate lookup
            id_attr = None
            
            # Check dim_coords in field metadata
            dim_coords = None
            if hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
                dim_coords = field_info.json_schema_extra.get("dim_coords")
            
            if dim_coords:
                id_attr = dim_coords
            
            entry = (module, field_name, id_attr)
            mapping[field_name] = entry
            mapping[f"{module_name}.{field_name}"] = entry
            
        return mapping

    @cached_property
    def group_id_to_rank(self) -> np.ndarray:
        """
        Load primary group variable from InputProxy and compute
        a full ID->rank map using compute_group_to_rank.
        """
        if self.group_by not in self.input_proxy:
            raise ValueError(f"Missing primary group variable '{self.group_by}' in InputProxy.")
        grp = self.input_proxy[self.group_by]
        group_id_to_rank = compute_group_to_rank(self.world_size, grp)
        return group_id_to_rank

    def initialize_statistics_aggregator(self) -> None:
        """
        Initialize the statistics aggregator for streaming NetCDF output.
        Registers all variables to save, including their save_idx and save_coord if present.
        Avoids duplicate registration.
        """
        if not self.variables_to_save:
            return
        self._statistics_aggregator = StatisticsAggregator(
            device=self.device,
            output_dir=self.output_full_dir,
            rank=self.rank,
            num_workers=self.output_workers,
            complevel=self.output_complevel,
            output_split_by_year=self.output_split_by_year,
            num_trials=self.num_trials or 1,
            save_kernels=self.save_kernels,
        )

        registered_vars = set()

        # Normalize variables_to_save (op -> vars) into var -> set[ops]
        allowed_ops = {"mean", "max", "min", "last"}
        var_to_ops: Dict[str, List[str]] = {}
        for op, vars_val in self.variables_to_save.items():
            op_l = str(op).lower()
            if op_l not in allowed_ops:
                raise ValueError(f"Invalid op '{op}'. Allowed: {sorted(allowed_ops)}")
            if isinstance(vars_val, str):
                names = [vars_val]
            elif isinstance(vars_val, list):
                names = list(vars_val)
            else:
                raise ValueError(f"variables_to_save['{op}'] must be a string or list of strings")
            for name in names:
                var_to_ops.setdefault(name, [])
                if op_l not in var_to_ops[name]:
                    var_to_ops[name].append(op_l)

        registered_vars_by_shape: Dict[str, List[str]] = {}

        for var_name in var_to_ops.keys():
            for module_name in self.opened_modules:
                module_instance = self.get_module(module_name)
                if not hasattr(module_instance, var_name):
                    continue

                tensor = getattr(module_instance, var_name)
                field_info = module_instance.get_model_fields().get(var_name)
                if field_info is None:
                    field_info = module_instance.get_model_computed_fields().get(var_name)
                
                if field_info is None:
                    continue

                # Check category
                category = field_info.json_schema_extra.get("category", "param")
                if category not in ("state", "shared_state", "init_state"):
                     print(f"[rank {self.rank}] Warning: Variable '{var_name}' is category '{category}', skipping output (only state/shared_state/init_state allowed).")
                     continue

                # Register the main tensor if not already done
                if var_name not in registered_vars:
                    self._statistics_aggregator.register_tensor(var_name, tensor, field_info)
                    registered_vars.add(var_name)
                    shape_str = str(tuple(tensor.shape))
                    if shape_str not in registered_vars_by_shape:
                        registered_vars_by_shape[shape_str] = []
                    registered_vars_by_shape[shape_str].append(var_name)

                # Check for save_idx
                save_idx = field_info.json_schema_extra.get("save_idx")
                if save_idx and save_idx not in registered_vars:
                    if hasattr(module_instance, save_idx):
                        save_tensor = getattr(module_instance, save_idx)
                        self._statistics_aggregator.register_tensor(save_idx, save_tensor, {})
                        registered_vars.add(save_idx)
                        shape_str = str(tuple(save_tensor.shape))
                        if shape_str not in registered_vars_by_shape:
                            registered_vars_by_shape[shape_str] = []
                        registered_vars_by_shape[shape_str].append(save_idx)
                    else:
                        raise ValueError(
                            f"save_idx '{save_idx}' not found in module '{module_name}' for variable '{var_name}'"
                        )

                # Check for save_coord
                save_coord = field_info.json_schema_extra.get("save_coord")
                if save_coord and save_coord not in registered_vars:
                    if hasattr(module_instance, save_coord):
                        coord_tensor = getattr(module_instance, save_coord)
                        self._statistics_aggregator.register_tensor(save_coord, coord_tensor, {})
                        registered_vars.add(save_coord)
                        shape_str = str(tuple(coord_tensor.shape))
                        if shape_str not in registered_vars_by_shape:
                            registered_vars_by_shape[shape_str] = []
                        registered_vars_by_shape[shape_str].append(save_coord)
                    else:
                        print(f"Warning: save_coord '{save_coord}' not found in module '{module_name}' for variable '{var_name}'")

                break  # break once var_name is found in a module
        
        if registered_vars_by_shape:
            for shape_str, vars_list in registered_vars_by_shape.items():
                print(f"[rank {self.rank}]: Registered tensors for streaming: {', '.join(vars_list)} (shape: {shape_str})")

        self._statistics_aggregator.initialize_streaming_aggregation(
            variable_ops=var_to_ops
        )

    def update_statistics(self, weight: float, total_weight: float = 0.0, is_first: bool = False, is_last: bool = False, BLOCK_SIZE: int = 128) -> None:
        """
        Update streaming statistics with a time weight.
        Args:
            weight: dt in seconds for this sub-step (time-weighted accumulation)
            is_first: whether this sub-step is the first of a stats window
            is_last: whether this sub-step is the last of a stats window
        """
        if self._statistics_aggregator is not None:
            self._statistics_aggregator.update_statistics(weight, total_weight, is_first, is_last, BLOCK_SIZE)

    def finalize_time_step(self, current_time: Union[datetime, cftime.datetime]) -> None:
        """
        Finalize time step in aggregator (write current means to disk).
        """
        if self._statistics_aggregator is not None:
            self._statistics_aggregator.finalize_time_step(current_time)

    def shard_param(self) -> Dict[str, Any]:
        """
        Load fields by reading from InputProxy and slicing in-memory per rank.
        """
        module_data: Dict[str, torch.Tensor] = {}

        # Collect unique fields to load across all opened modules
        fields_to_load: Dict[str, Any] = {}
        for _, _, field_name, field_info in self._iter_all_fields(include_computed=False):
            if field_name not in fields_to_load:
                fields_to_load[field_name] = field_info

        try:
            # Validate required fields exist
            missing_required = [
                name for name, info in fields_to_load.items()
                if info.is_required() and name not in self.input_proxy
            ]
            if missing_required:
                raise KeyError(
                    f"Required fields missing from InputProxy: {missing_required}. "
                    f"Available fields: {list(self.input_proxy.data.keys())}"
                )

            # Pre-compute indices per group var for current rank
            group_vars_needed = set()
            for name in fields_to_load.keys():
                if name in self.variable_group_mapping:
                    # Only need group var if the variable itself is in the dataset
                    if name in self.input_proxy:
                        group_vars_needed.add(self.variable_group_mapping[name])
            group_indices_cache: Dict[str, np.ndarray] = {}
            for group_var in group_vars_needed:
                if group_var not in self.input_proxy:
                    raise ValueError(f"Group variable '{group_var}' not found in InputProxy.")
                grp = self.input_proxy[group_var]
                idx = np.nonzero(self.group_id_to_rank[grp] == self.rank)[0]
                group_indices_cache[group_var] = idx

            print(f"[rank {self.rank}]: Loading data for modules {self.opened_modules}")

            def to_torch(arr: Any) -> torch.Tensor:
                # Use as_tensor to avoid unnecessary copy; unify float dtype only
                t = torch.as_tensor(arr)
                if t.is_floating_point() and t.dtype != self.dtype:
                    t = t.to(self.dtype)
                if not t.is_contiguous():
                    t = t.contiguous()
                return t

            # Buckets for logging
            missing_fields = []
            no_local_fields: Dict[str, List[str]] = {}
            distributed_fields: Dict[Tuple[Tuple[int, ...], str], List[str]] = {}
            full_fields = []

            # Sort fields for deterministic processing order
            def sort_key(item):
                name, _ = item
                group = self.variable_group_mapping.get(name, "")
                if group is None:
                    group = ""
                return (str(group), name)

            sorted_fields = sorted(fields_to_load.items(), key=sort_key)

            for field_name, field_info in sorted_fields:
                if field_name not in self.input_proxy:
                    missing_fields.append(field_name)
                    continue

                full_np = self.input_proxy[field_name]

                group_var = self.variable_group_mapping.get(field_name, None)
                if group_var is not None:
                    idx = group_indices_cache[group_var]
                    if idx.size == 0:
                        # Construct empty with correct trailing shape
                        base_shape = full_np.shape[1:] if isinstance(full_np, np.ndarray) else ()
                        empty_np = np.empty((0, *base_shape), dtype=getattr(full_np, "dtype", np.float32))
                        module_data[field_name] = to_torch(empty_np)
                        
                        if group_var not in no_local_fields:
                            no_local_fields[group_var] = []
                        no_local_fields[group_var].append(field_name)
                    else:
                        # Handle batched parameters (num_trials, num_catchments, ...)
                        # If the first dimension is num_trials, we need to index the second dimension
                        if full_np.ndim > 1 and self.num_trials is not None and full_np.shape[0] == self.num_trials:
                            # Batched parameter: (T, N, ...)
                            # We want to select indices from the second dimension (N)
                            # Result should be (T, L, ...) where L is len(idx)
                            local_np = full_np[:, idx]
                        else:
                            # Standard parameter: (N, ...)
                            local_np = full_np[idx]
                            
                        module_data[field_name] = to_torch(local_np)
                        
                        shape = local_np.shape
                        key = (shape, group_var)
                        if key not in distributed_fields:
                            distributed_fields[key] = []
                        distributed_fields[key].append(field_name)
                else:
                    module_data[field_name] = to_torch(full_np)
                    full_fields.append(field_name)
            
            # Flush logs
            for group_var, fields in no_local_fields.items():
                print(f"[rank {self.rank}]: No local data for distributed fields: {', '.join(fields)} (group_by: {group_var})")
            
            for (shape, group_var), fields in distributed_fields.items():
                print(f"[rank {self.rank}]: Loaded distributed fields: {', '.join(fields)} (shape: {shape}, group_by: {group_var})")
            
            if full_fields:
                print(f"[rank {self.rank}]: Loaded full fields: {', '.join(full_fields)} (no group_by)")
            
            if missing_fields:
                print(f"[rank {self.rank}]: Optional fields not in InputProxy, using default: {', '.join(missing_fields)}")

        except Exception as e:
            raise RuntimeError(f"Error loading data from InputProxy: {e}")

        return module_data

    def save_state(self, current_time: Optional[Union[datetime, cftime.datetime]]) -> InputProxy:
        """
        Save model state to InputProxy and NetCDF files (.nc).
        """
        if self.num_trials is not None:
            print(f"[rank {self.rank}] Warning: save_state is not supported for multi-trial simulations.")
            return None

        timestamp = current_time.strftime("%Y%m%d_%H%M%S") if current_time else "latest"

        # Determine file path per-rank
        if self.world_size > 1:
            nc_path = self.output_full_dir / f"model_state_rank{self.rank}_{timestamp}.nc"
        else:
            nc_path = self.output_full_dir / f"model_state_{timestamp}.nc"

        # Collect data
        data = {}
        visited_fields = set()
        
        saved_distributed = []
        saved_global = []
        skipped_none = set()

        for module_name in self.opened_modules:
            module = self._modules[module_name]
            for field_name, field_info in module.get_model_fields().items():
                if field_name in module.nc_excluded_fields or field_name in visited_fields:
                    continue
                
                if field_info.exclude:
                    continue

                is_distributed = field_name in self.variable_group_mapping

                # Only rank 0 saves non-distributed variables
                if not is_distributed and self.rank != 0:
                    continue

                val = getattr(module, field_name)
                
                if isinstance(val, torch.Tensor):
                    val = val.detach().cpu().numpy()
                
                if val is None:
                    skipped_none.add(field_name)
                    continue
                    
                data[field_name] = val
                visited_fields.add(field_name)
                skipped_none.discard(field_name)

                if is_distributed:
                    saved_distributed.append(field_name)
                else:
                    saved_global.append(field_name)

        # Create InputProxy
        proxy = InputProxy(data, attrs={
            "title": "CaMa-Flood-GPU Model State",
            "history": f"Created by CaMa-Flood-GPU at {datetime.now().isoformat()}",
            "source": "AbstractModel.save_state"
        })
        
        # Write to file
        if nc_path.exists():
             print(f"[rank {self.rank}] Warning: Overwriting existing model state file: {nc_path}")
             
        proxy.to_nc(nc_path, output_complevel=self.output_complevel if self.world_size == 1 else 0)
        
        if saved_distributed:
            print(f"[rank {self.rank}] Saved distributed fields: {', '.join(saved_distributed)}")
        if saved_global:
            print(f"[rank {self.rank}] Saved global fields: {', '.join(saved_global)}")
        if skipped_none:
            print(f"[rank {self.rank}] Skipped None fields (not saved): {', '.join(sorted(list(skipped_none)))}")

        if self.world_size > 1:
            dist.barrier()

        # Merge step only done by rank 0
        if self.rank == 0 and self.world_size > 1:
            merged_path = self.output_full_dir / f"model_state_{timestamp}.nc"
            rank_paths = [self.output_full_dir / f"model_state_rank{r}_{timestamp}.nc" for r in range(self.world_size)]
            
            InputProxy.merge(merged_path, rank_paths, self.variable_group_mapping, self.output_complevel)
            
            # Remove rank files
            for p in rank_paths:
                try:
                    p.unlink()
                except Exception:
                    pass
            
            print(f"[rank 0] Model state merged to: {merged_path}")
            
        return proxy

    def load_state(self, proxy: InputProxy) -> None:
        """
        Restore model state from an InputProxy.
        Supports loading from both global (merged) and local (sharded) proxies.
        """
        print(f"[rank {self.rank}] Loading state from InputProxy...")
        
        loaded_count = 0
        
        # Cache group indices for sharding
        group_indices_cache: Dict[str, np.ndarray] = {}

        for module_name in self.opened_modules:
            module = self._modules[module_name]
            
            for field_name, field_info in module.get_model_fields().items():
                if field_name not in proxy:
                    continue
                
                # Skip excluded fields if they happen to be in proxy (unlikely but safe)
                if field_info.exclude:
                    continue

                new_val = proxy[field_name]
                current_val = getattr(module, field_name)
                
                # Handle Tensor fields
                if isinstance(current_val, torch.Tensor):
                    # Convert new_val to numpy if it's a tensor (InputProxy might hold tensors)
                    if isinstance(new_val, torch.Tensor):
                        new_val = new_val.detach().cpu().numpy()
                    
                    new_val = np.asarray(new_val)
                    
                    # Check 1: Direct shape match (Local file or scalar)
                    if new_val.shape == tuple(current_val.shape):
                        current_val.copy_(torch.as_tensor(new_val).to(current_val.device))
                        loaded_count += 1
                        continue
                        
                    # Check 2: Distributed variable needing sharding (Global file)
                    if field_name in self.variable_group_mapping:
                        group_var = self.variable_group_mapping[field_name]
                        
                        # We rely on self.input_proxy (static params) for sharding info
                        if group_var not in self.input_proxy:
                            print(f"[rank {self.rank}] Warning: Cannot shard '{field_name}' because group var '{group_var}' is missing in static inputs.")
                            continue
                            
                        # Get indices (cached)
                        if group_var not in group_indices_cache:
                            grp = self.input_proxy[group_var]
                            idx = np.nonzero(self.group_id_to_rank[grp] == self.rank)[0]
                            group_indices_cache[group_var] = idx
                        
                        idx = group_indices_cache[group_var]
                        
                        # Shard the global data
                        try:
                            local_val = new_val[idx]
                        except IndexError:
                             print(f"[rank {self.rank}] Warning: Indexing error sharding '{field_name}'. Shape: {new_val.shape}, Indices max: {idx.max() if len(idx)>0 else 'N/A'}")
                             continue

                        if local_val.shape == tuple(current_val.shape):
                            current_val.copy_(torch.as_tensor(local_val).to(current_val.device))
                            loaded_count += 1
                        else:
                            print(f"[rank {self.rank}] Warning: Shape mismatch for '{field_name}' after sharding. Expected {tuple(current_val.shape)}, got {local_val.shape}.")
                    else:
                        print(f"[rank {self.rank}] Warning: Shape mismatch for '{field_name}'. Expected {tuple(current_val.shape)}, got {new_val.shape}.")
                
                # Handle Scalar/Other fields
                else:
                    # For scalars, we just set the value
                    # If it's a numpy scalar, convert to python type if needed, or just set
                    setattr(module, field_name, new_val)
                    loaded_count += 1

        print(f"[rank {self.rank}] Successfully loaded {loaded_count} variables from InputProxy.")

    @field_validator("opened_modules")
    @classmethod
    def validate_modules(cls, v: List[str]) -> List[str]:
        """Validate module names are valid"""
        if not v:
            raise ValueError("No modules opened. Please specify at least one module in opened_modules.")
        for module in v:
            if module not in cls.module_list:
                raise ValueError(f"Invalid module name: {module}. Available modules: {list(cls.module_list.keys())}")
        return v

    @model_validator(mode="after")
    def validate_variables_to_save(self) -> Self:
        if self.variables_to_save is None:
            return self
        # Validate shape: dict[op -> vars]
        allowed_ops = {"mean", "max", "min", "last"}
        if not isinstance(self.variables_to_save, dict):
            # Optional convenience: list[str] => mean
            names = list(self.variables_to_save) if isinstance(self.variables_to_save, list) else []
            pairs = [(n, "mean") for n in names]
        else:
            pairs = []
            for op, vs in self.variables_to_save.items():
                op_l = str(op).lower()
                if op_l not in allowed_ops:
                    raise ValueError(f"Invalid statistics op '{op}'. Allowed: {sorted(allowed_ops)}")
                if isinstance(vs, str):
                    vars_list = [vs]
                elif isinstance(vs, list):
                    vars_list = vs
                else:
                    raise ValueError(f"variables_to_save['{op}'] must be a string or list of strings")
                for var in vars_list:
                    pairs.append((var, op_l))

        # Validate each variable exists and has save_idx
        for var, _ in pairs:
            found = False
            has_save_idx = False
            for module in self.opened_modules:
                module_class = self.module_list[module]
                fields = module_class.model_fields | module_class.model_computed_fields
                if var in fields:
                    found = True
                    field_info = fields[var]
                    extra = getattr(field_info, "json_schema_extra", {})
                    if extra and extra.get("save_idx") is not None:
                        has_save_idx = True
                    break
            if not found:
                raise ValueError(f"Variable '{var}' not found in any opened module.")
            if not has_save_idx:
                raise ValueError(f"Variable '{var}' does not have `save_idx` defined, and cannot be saved.")
        return self

    @model_validator(mode="after")
    def validate_rank(self) -> Self:
        """
        Validate that the current rank is within the world size.
        """
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"Invalid rank {self.rank} for world size {self.world_size}.")
        return self

    @model_validator(mode="after")
    def validate_output_full_dir(self) -> Self:
        if self.rank == 0:
            if not self.output_full_dir.exists():
                self.output_full_dir.mkdir(parents=True, exist_ok=True)
            else:
                print(f"Warning: Output directory {self.output_full_dir} already exists. Contents may be overwritten.")
        return self
