# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import re
from abc import ABC
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

from cmfgpu.aggregator import StatisticsAggregator
from cmfgpu.models.utils import (ActivePlan, ParameterPlanMixin, PlanItem,
                                  ProgressMixin, ProgressTracker,
                                  compute_group_to_rank)
from cmfgpu.modules.abstract_module import AbstractModule
from cmfgpu.params.input_proxy import InputProxy


class AbstractModel(ParameterPlanMixin, ProgressMixin, BaseModel, ABC):
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
    experiment_name: str = Field(
        default="experiment",
        description="Name of the experiment",
    )
    input_proxy: InputProxy = Field(
        default=...,
        description="InputProxy object containing model data",
    )
    output_dir: Path = Field(
        default_factory=lambda: Path("./out"),
        description="Path to the output directory",
    )
    opened_modules: List[str] = Field(
        default_factory=list,
        description="List of active modules",
    )
    # Preferred shape: dict[op -> str | list[str]]; op in {mean,max,min,last};
    # one variable can appear under multiple ops.
    variables_to_save: Optional[Dict[str, Union[str, List[Union[str, Dict[str, str]]]]]] = Field(
        default=None,
        description=(
            "Statistics to save, in the form {op: [vars...]}. "
            "Supported ops: mean, max, min, last, first, mid, sum. "
            "For max/min, argmax/argmin are automatically computed. "
            "Variables can be strings or {alias: expr} dicts."
        ),
    )
    precision: Literal["bfloat16", "float32", "float64"] = Field(
        default="float32",
        description="Base precision of the model",
    )
    mixed_precision: bool = Field(
        default=False,
        description=(
            "Enable mixed precision for hpfloat (storage) tensors.\n"
            "When True, hpfloat tensors are promoted one level above base precision:\n"
            "  bfloat16 → float32, float32 → float64, float64 → float64 (no promotion)."
        ),
    )
    world_size: int = Field(
        default=1,
        description="Total number of distributed processes",
    )
    rank: int = Field(
        default=0,
        description="Current process rank in distributed setup",
    )
    device: torch.device = Field(
        default=torch.device("cpu"),
        description="Device for tensors (e.g., 'cuda:0', 'cpu')",
    )
    BLOCK_SIZE: int = Field(
        default=256,
        description="GPU block size for kernels",
    )
    output_workers: int = Field(
        default=2,
        description="Number of workers for writing output files",
    )
    output_complevel: int = Field(
        default=4,
        description="Compression level for output NetCDF files",
        ge=0,
        le=9,
    )
    output_split_by_year: bool = Field(
        default=False,
        description="Whether to split output files by year",
    )
    num_trials: Optional[int] = Field(
        default=None,
        description="Number of parallel simulations (ensemble members)",
    )
    save_kernels: bool = Field(
        default=False,
        description="Whether to save generated Triton kernels",
    )
    max_pending_steps: int = Field(
        default=20,
        description="Maximum number of pending time steps for output buffering",
    )
    output_start_time: Optional[Union[datetime, cftime.datetime]] = Field(
        default=None,
        description="Time to start saving output",
    )
    calendar: str = Field(
        default="standard",
        description="Calendar type for time handling (e.g., standard, noleap)",
    )
    in_memory_output: bool = Field(
        default=False,
        description="Store output in memory instead of writing to NC files",
    )
    result_device: Optional[torch.device] = Field(
        default=None,
        description="Device for in-memory results (default: CPU)",
    )

    _modules: Dict[str, AbstractModule] = PrivateAttr(default_factory=dict)

    _statistics_aggregator: Optional[StatisticsAggregator] = PrivateAttr(default=None)
    
    # Parameter Change Plan State
    _plans: List[PlanItem] = PrivateAttr(default_factory=list)
    _active_plans: List[ActivePlan] = PrivateAttr(default_factory=list)
    _next_plan_idx: int = PrivateAttr(default=0)
    _cached_grouped_plans: Optional[Dict[Tuple[int, str], List[ActivePlan]]] = PrivateAttr(default=None)

    # Progress Tracking
    _progress: Optional[ProgressTracker] = PrivateAttr(default=None)

    @model_validator(mode='after')
    def align_output_start_time(self) -> Self:
        """
        Ensures output_start_time matches the specified calendar type.
        """
        if self.output_start_time is None or self.calendar == "standard":
            return self

        # If output_start_time is standard datetime but calendar is not standard (e.g. noleap)
        # we try to convert it to the appropriate cftime object.
        if isinstance(self.output_start_time, datetime) and not isinstance(self.output_start_time, cftime.datetime):
            try:
                # Create a dummy object to get the class type for this calendar
                dummy = cftime.num2date([0], units="days since 1900-01-01", calendar=self.calendar)[0]
                Cls = dummy.__class__
                
                kwargs = {}
                if hasattr(dummy, "has_year_zero"):
                    kwargs["has_year_zero"] = dummy.has_year_zero

                self.output_start_time = Cls(
                    self.output_start_time.year, self.output_start_time.month, self.output_start_time.day,
                    self.output_start_time.hour, self.output_start_time.minute, self.output_start_time.second,
                    self.output_start_time.microsecond, **kwargs
                )
            except Exception:
                # If conversion fails, we leave it as is and hope for the best or fail later
                pass
        return self

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

    @cached_property
    def dtype(self) -> torch.dtype:
        _dtype_map = {
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float64": torch.float64,
        }
        return _dtype_map[self.precision]

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

    def model_post_init(self, __context: Any) -> None:
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
                mixed_precision=self.mixed_precision,
                num_trials=self.num_trials,
                **self._modules,
                **module_data
            )
            self._modules[module_name] = module_instance

        self.initialize_statistics_aggregator()

        for module_name in self.opened_modules:
            mod = self.get_module(module_name)
            if mod:
                mod.handle_tensor_mode()

        self.print_memory_summary()
        print("All modules initialized successfully.")

    def print_memory_summary(self) -> None:
        """
        Print a summary of memory usage by module.
        
        Uses ``data_ptr`` de-duplication across all modules so that tensors
        shared between modules (e.g. a computed field that returns a view of
        another module's tensor) are counted only once.
        """
        if self.rank != 0:
            return
        total_memory = 0
        global_seen_ptrs: set = set()
        print(f"\n[rank {self.rank}] Memory Usage Summary:")
        print(f"{'Module':<30} | {'Memory (MB)':<15} | {'Unique (MB)':<15}")
        print(f"{'-' * 65}")
        
        for module_name in self.opened_modules:
            if module_name not in self._modules:
                continue
            module = self._modules[module_name]
            # Module-local memory (may include tensors shared with other modules)
            mem_bytes = module.get_memory_usage()
            mem_mb = mem_bytes / (1024 * 1024)
            
            # Count only tensors not yet seen globally
            unique_bytes = 0
            all_fields = module.get_model_fields().copy()
            all_fields.update(module.get_model_computed_fields())
            for name, field_info in all_fields.items():
                if not hasattr(module, name):
                    continue
                value = getattr(module, name)
                if isinstance(value, torch.Tensor) and value.device == module.device:
                    ptr = value.data_ptr()
                    if ptr not in global_seen_ptrs:
                        global_seen_ptrs.add(ptr)
                        unique_bytes += value.element_size() * value.nelement()
            
            total_memory += unique_bytes
            unique_mb = unique_bytes / (1024 * 1024)
            if unique_bytes != mem_bytes:
                print(f"{module_name:<30} | {mem_mb:<15.2f} | {unique_mb:<15.2f} (shared)")
            else:
                print(f"{module_name:<30} | {mem_mb:<15.2f} | {unique_mb:<15.2f}")
        
        # Add StatisticsAggregator memory usage
        if self._statistics_aggregator is not None:
            aggregator_mem = self._statistics_aggregator.get_memory_usage()
            aggregator_mem_mb = aggregator_mem / (1024 * 1024)
            total_memory += aggregator_mem
            print(f"{'StatisticsAggregator':<30} | {aggregator_mem_mb:<15.2f} | {aggregator_mem_mb:<15.2f}")
            
        print(f"{'-' * 65}")
        print(f"{'Total':<30} | {'':<15} | {total_memory / (1024 * 1024):<15.2f} MB\n")

    def get_module(self, module_name: str) -> Optional[AbstractModule]:
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
            max_pending_steps=self.max_pending_steps,
            calendar=self.calendar,
            in_memory_mode=self.in_memory_output,
            result_device=self.result_device,
            save_precision=self.dtype,
        )

        registered_vars = set()

        # Normalize variables_to_save (op -> vars) into var -> set[ops]
        # Note: argmax/argmin can now be specified explicitly for timing information
        allowed_ops = {"mean", "max", "min", "last", "first", "mid", "sum", "argmax", "argmin"}
        import re
        topk_pattern = re.compile(r'^(max|min)(\d+)$')
        argtopk_pattern = re.compile(r'^arg(max|min)(\d*)$')  # argmax, argmin, argmax3, argmin3

        var_to_ops: Dict[str, List[str]] = {}
        # Stores explicit expressions provided by user: alias -> expression
        explicit_expressions: Dict[str, str] = {}

        for op, vars_val in self.variables_to_save.items():
            op_l = str(op).lower()
            op_parts = op_l.split('_')
            
            # Validate op parts
            for p in op_parts:
                # Check against allowed patterns
                if p not in allowed_ops and not topk_pattern.match(p) and not argtopk_pattern.match(p):
                     raise ValueError(f"Invalid op '{op}'. Component '{p}' not in allowed ops: {sorted(allowed_ops)}, top-k pattern, or arg-top-k pattern.")

            # Single operation (no underscore) - this is the inner aggregation
            # arg operations are NOT allowed as inner operations
            if len(op_parts) == 1:
                single_op = op_parts[0]
                if argtopk_pattern.match(single_op) or single_op in ('argmax', 'argmin'):
                    raise ValueError(f"Invalid op '{op}': arg operations (argmax, argmin, argmax3, etc.) cannot be used alone. They are only valid as outer operations in compound form like 'argmax_mean'.")

            if len(op_parts) > 1:
                outer, inner = op_parts[0], op_parts[1]
                # Check for outer restriction: mid cannot be an outer op
                if outer == 'mid':
                    raise ValueError(f"Invalid composite op '{op}': 'mid' cannot be used as an outer operation. It is only valid as an inner op (standalone 'mid').")
                # Check for inner restriction: top-k, arg-top-k, argmax/argmin cannot be inner ops
                if topk_pattern.match(inner) or argtopk_pattern.match(inner):
                    raise ValueError(f"Invalid composite op '{op}': '{inner}' (top-k/arg-top-k) cannot be used as an inner operation.")
                if inner in ('argmax', 'argmin'):
                    raise ValueError(f"Invalid composite op '{op}': '{inner}' cannot be used as an inner operation. arg operations are only valid as outer ops.")
                if len(op_parts) > 2:
                    raise ValueError(f"Invalid composite op '{op}': Only 2 levels of operations are supported.")

            # Standardize vars_val to a list of items
            if isinstance(vars_val, (str, tuple)):
                items = [vars_val]
            elif isinstance(vars_val, list):
                items = vars_val
            else:
                raise ValueError(f"variables_to_save['{op}'] must be a string, tuple (name, expr), or list of them.")
            
            for item in items:
                if isinstance(item, str):
                    name = item
                elif isinstance(item, (tuple, list)) and len(item) == 2:
                    name = item[0]
                    expr = item[1]
                    if name in explicit_expressions and explicit_expressions[name] != expr:
                         raise ValueError(f"Conflicting expressions for alias '{name}': '{explicit_expressions[name]}' vs '{expr}'")
                    explicit_expressions[name] = expr
                elif isinstance(item, dict):
                    if len(item) != 1:
                        raise ValueError(f"Dictionary item in variables_to_save['{op}'] must have exactly one key-value pair {{alias: expr}}. Got: {item}")
                    name, expr = next(iter(item.items()))
                    if name in explicit_expressions and explicit_expressions[name] != expr:
                         raise ValueError(f"Conflicting expressions for alias '{name}': '{explicit_expressions[name]}' vs '{expr}'")
                    explicit_expressions[name] = expr
                else:
                    raise ValueError(f"Invalid item in variables_to_save['{op}']: {item}. Must be string, dict {{name: expr}}, or (name, expr) tuple.")

                var_to_ops.setdefault(name, [])
                if op_l not in var_to_ops[name]:
                    var_to_ops[name].append(op_l)

        # Handle ad-hoc expressions in var_to_ops
        adhoc_virtuals: Dict[str, Any] = {}
        
        current_vars = list(var_to_ops.keys())
        for var_name in current_vars:
            if var_name in self.variable_map:
                continue
            
            # If provided as tuple, usage is explicit. If string, assume string IS the expression.
            expression = explicit_expressions.get(var_name, var_name)
            
            # Check if it looks like an expression (simple heuristic)
            # and if we can resolve its tokens
            import re
            tokens = set(re.findall(r'\b[a-zA-Z_]\w*\b', expression))
            
            valid_deps = []
            for t in tokens:
                 if t in self.variable_map:
                      valid_deps.append(t)
            
            # If no dependencies found, skip (likely invalid or constant)
            if not valid_deps:
                 continue

            # Consistency Check
            # Ensure all dependencies share the same save_idx/save_coord/dim_coords
            def get_field_meta(name):
                 mod, attr, _ = self.variable_map[name]
                 info = mod.get_model_fields().get(attr) or mod.get_model_computed_fields().get(attr)
                 if info:
                      extra = info.json_schema_extra or {}
                      return (extra.get("save_idx"), extra.get("save_coord"), extra.get("dim_coords"))
                 return (None, None, None)

            ref_meta = get_field_meta(valid_deps[0])
            # Only enforce save_idx and dim_coords roughly.
            # save_coord might act differently but usually matches too.
            
            for dep in valid_deps[1:]:
                 curr_meta = get_field_meta(dep)
                 if curr_meta != ref_meta:
                      raise ValueError(
                          f"Inconsistent metadata in virtual variable '{var_name}' (expression: '{expression}'). "
                          f"Dependency '{valid_deps[0]}' has {ref_meta}, but '{dep}' has {curr_meta}. "
                          "All dependencies in an expression must share the same 'save_idx', 'save_coord', and 'dim_coords' to ensure correct parallel iteration."
                      )

            # Create FieldInfo
            save_idx, save_coord, dim_coords = ref_meta
            
            new_info = Field(
                description=f"Ad-hoc expression: {expression}",
                json_schema_extra={
                    "category": "virtual",
                    "expr": expression,
                    "save_idx": save_idx,
                    "save_coord": save_coord,
                    "dim_coords": dim_coords
                }
            )
            
            adhoc_virtuals[var_name] = new_info

        # No need to sanitize names or update var_to_ops keys further, 
        # as var_to_ops already uses the keys we intend to register.


        registered_vars_by_shape: Dict[str, List[str]] = {}
        
        # 1. Expand variables to include dependencies of virtual variables
        vars_to_process = list(var_to_ops.keys())
        vars_seen = set(vars_to_process)
        idx = 0
        while idx < len(vars_to_process):
            curr_var = vars_to_process[idx]
            idx += 1
            
            if curr_var not in self.variable_map:
                if curr_var in adhoc_virtuals:
                    field_info = adhoc_virtuals[curr_var]
                    # Check for deps in adhoc virtuals
                    expr = field_info.json_schema_extra.get("expr", "")
                    deps = re.findall(r'\b[a-zA-Z_]\w*\b', expr)
                    for d in deps:
                        if d not in vars_seen:
                            vars_seen.add(d)
                            vars_to_process.append(d)
                continue

            module_instance, attr_name, _ = self.variable_map[curr_var]
            
            # Check normal fields
            field_info = module_instance.get_model_fields().get(attr_name)
            if field_info is None:
                # Check computed fields
                field_info = module_instance.get_model_computed_fields().get(attr_name)
            
            if field_info:
                cat = field_info.json_schema_extra.get("category", "param")
                if cat == 'virtual':
                    expr = field_info.json_schema_extra.get("expr", "")
                    # Simple regex for potential tokens
                    deps = re.findall(r'\b[a-zA-Z_]\w*\b', expr)
                    for d in deps:
                        if d not in vars_seen:
                            vars_seen.add(d)
                            vars_to_process.append(d)

        # 2. Register all variables (original + dependencies)
        registered_vars = set()
        for var_name in vars_to_process:
            if var_name not in self.variable_map:
                if var_name in adhoc_virtuals:
                     if var_name not in registered_vars:
                         self._statistics_aggregator.register_virtual_tensor(var_name, adhoc_virtuals[var_name])
                         registered_vars.add(var_name)
                     continue

                if self.rank == 0:
                    print(f"Warning: Variable '{var_name}' not found in variable_map or adhoc list. Skipping.")
                continue

            module_instance, attr_name, _ = self.variable_map[var_name]

            if not hasattr(module_instance, attr_name):
                continue

            tensor = getattr(module_instance, attr_name)
            field_info = module_instance.get_model_fields().get(attr_name)
            if field_info is None:
                field_info = module_instance.get_model_computed_fields().get(attr_name)
            
            if field_info is None:
                continue

            # Check category
            category = field_info.json_schema_extra.get("category", "param")
            allowed_cats = ("state", "shared_state", "init_state", "param", "virtual")
            if category not in allowed_cats:
                    # Only warn if it's an explicitly requested output
                    if var_name in var_to_ops:
                        print(f"[rank {self.rank}] Warning: Variable '{var_name}' is category '{category}', skipping output (allowed: {allowed_cats}).")
                    continue

            # Check dimensionality and restrictions (only for requested outputs)
            if var_name in var_to_ops:
                ops = var_to_ops[var_name]
                if category != 'virtual':
                    # 1D is usually (N,), 2D is (N, Level).
                    # With trials: 1D is (T, N), 2D is (T, N, Level)
                    limit = 1
                    if self.num_trials and self.num_trials > 1:
                        limit = 2
                    
                    is_real_2d = tensor.ndim > limit

                    if is_real_2d:
                        for op in ops:
                            op_base = op.split('_')[0]
                            if topk_pattern.match(op_base) or op_base in ('max', 'min'):
                                raise ValueError(f"Operation '{op}' is not allowed for 2D variable '{var_name}' (ndim={tensor.ndim}). Only 'mean', 'sum', 'last', 'first', 'mid' are supported for 2D variables.")

            # Register the main tensor if not already done
            if var_name not in registered_vars:
                if category == 'virtual':
                    self._statistics_aggregator.register_virtual_tensor(var_name, field_info)
                else:
                    self._statistics_aggregator.register_tensor(var_name, tensor, field_info)
                    shape_str = str(tuple(tensor.shape))
                    if shape_str not in registered_vars_by_shape:
                        registered_vars_by_shape[shape_str] = []
                    registered_vars_by_shape[shape_str].append(var_name)
                
                registered_vars.add(var_name)

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
                        f"save_idx '{save_idx}' not found in module '{type(module_instance).__name__}' for variable '{var_name}'"
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
                    print(f"Warning: save_coord '{save_coord}' not found in module '{type(module_instance).__name__}' for variable '{var_name}'")
        
        if registered_vars_by_shape:
            for shape_str, vars_list in registered_vars_by_shape.items():
                print(f"[rank {self.rank}]: Registered tensors for streaming: {', '.join(vars_list)} (shape: {shape_str})")

        self._statistics_aggregator.initialize_streaming_aggregation(
            variable_ops=var_to_ops
        )

    def update_statistics(self, weight: float, total_weight: float = 0.0, is_first: bool = False, is_last: bool = False, is_middle: bool = False, is_macro_step_end: bool = False, is_inner_first: bool = False, is_inner_last: bool = False, is_outer_first: bool = False, is_outer_last: bool = False, BLOCK_SIZE: int = 128, custom_step_index: Optional[int] = None) -> None:
        """
        Update streaming statistics with a time weight.
        Args:
            weight: dt in seconds for this sub-step (time-weighted accumulation)
            is_first: whether this sub-step is the first of a stats window
            is_last: whether this sub-step is the last of a stats window
            is_middle: whether this sub-step is the middle of a stats window
            is_macro_step_end: whether this sub-step is the end of a macro-step
        """
        if self._statistics_aggregator is not None:
            self._statistics_aggregator.update_statistics(weight, total_weight, is_inner_first=is_inner_first, is_inner_last=is_inner_last, is_outer_first=is_outer_first, is_outer_last=is_outer_last, BLOCK_SIZE=BLOCK_SIZE, custom_step_index=custom_step_index, is_first=is_first, is_last=is_last, is_middle=is_middle, is_macro_step_end=is_macro_step_end)

    def finalize_time_step(self, current_time: Union[datetime, cftime.datetime]) -> None:
        """
        Finalize time step in aggregator (write current means to disk).
        """
        if self._statistics_aggregator is not None:
            self._statistics_aggregator.finalize_time_step(current_time)

    def get_output_results(self, as_stacked: bool = True) -> Dict[str, torch.Tensor]:
        """
        Get the in-memory output results (only available when in_memory_output=True).
        
        Args:
            as_stacked: If True (default), stack all time steps into a single tensor.
                       If False, return list of per-time-step tensors.
                            
        Returns:
            Dictionary mapping output names to result tensors.
            Shape (when stacked): (time_steps, *actual_shape)
            
        Raises:
            RuntimeError: If not in in_memory_output mode or aggregator not initialized.
        """
        if self._statistics_aggregator is None:
            raise RuntimeError("Statistics aggregator not initialized")
        return self._statistics_aggregator.get_results(as_stacked=as_stacked)
    
    def get_output_result(self, variable_name: str, op: str = "mean", as_stacked: bool = True) -> torch.Tensor:
        """
        Get a specific output result tensor by variable name and operation.
        
        Args:
            variable_name: Name of the variable
            op: Operation type (mean, max, min, last, etc.)
            as_stacked: If True (default), stack all time steps into a single tensor.
            
        Returns:
            Result tensor for the specified variable and operation.
            
        Raises:
            RuntimeError: If not in in_memory_output mode or aggregator not initialized.
            KeyError: If the specified variable/op combination doesn't exist.
        """
        if self._statistics_aggregator is None:
            raise RuntimeError("Statistics aggregator not initialized")
        return self._statistics_aggregator.get_result(variable_name, op, as_stacked=as_stacked)
    
    def get_output_time_index(self) -> int:
        """Get the current output time index (number of finalized time steps)."""
        if self._statistics_aggregator is None:
            return 0
        return self._statistics_aggregator.get_time_index()
    
    def reset_output_time_index(self) -> None:
        """Reset the output time index to 0 for a new simulation run (in-memory mode only)."""
        if self._statistics_aggregator is not None:
            self._statistics_aggregator.reset_time_index()

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

                group_var = self.variable_group_mapping.get(field_name, None)
                if group_var is not None:
                    idx = group_indices_cache[group_var]

                    # Use get_var_shape to check dimensions without loading full data
                    full_shape = self.input_proxy.get_var_shape(field_name)

                    # Handle batched parameters (num_trials, num_catchments, ...)
                    if len(full_shape) > 1 and self.num_trials is not None and full_shape[0] == self.num_trials:
                         # Batched parameter: (T, N, ...) -> slice dim 1
                         slicer = (slice(None), idx)
                    else:
                         # Standard parameter: (N, ...) -> slice dim 0
                         slicer = idx

                    # Read only the subset
                    local_np = self.input_proxy.get_subset(field_name, slicer)
                    module_data[field_name] = to_torch(local_np)

                    if idx.size == 0:
                        if group_var not in no_local_fields:
                            no_local_fields[group_var] = []
                        no_local_fields[group_var].append(field_name)
                    else:
                        shape = local_np.shape
                        key = (shape, group_var)
                        if key not in distributed_fields:
                            distributed_fields[key] = []
                        distributed_fields[key].append(field_name)
                else:
                    module_data[field_name] = to_torch(self.input_proxy[field_name])
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
                    if isinstance(new_val, (np.ndarray, np.generic)):
                        if new_val.ndim == 0:
                            new_val = new_val.item()
                    
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
        # Note: argmax/argmin can now be specified explicitly for timing information
        allowed_ops = {"mean", "max", "min", "last", "first", "mid", "sum", "argmax", "argmin"}
        topk_pattern = re.compile(r'^(max|min)(\d+)$')
        argtopk_pattern = re.compile(r'^arg(max|min)(\d*)$')  # argmax, argmin, argmax3, argmin3

        if not isinstance(self.variables_to_save, dict):
            # Optional convenience: list[str] => mean
            names = list(self.variables_to_save) if isinstance(self.variables_to_save, list) else []
            pairs = [(n, "mean") for n in names]
        else:
            pairs = []
            for op, vs in self.variables_to_save.items():
                op_l = str(op).lower()
                op_parts = op_l.split('_')
                
                for p in op_parts:
                    if p not in allowed_ops and not topk_pattern.match(p) and not argtopk_pattern.match(p):
                        raise ValueError(f"Invalid statistics op '{op}'. Component '{p}' not in allowed ops: {sorted(allowed_ops)}, top-k pattern, or arg-top-k pattern.")
                
                # Single operation (no underscore) - this is the inner aggregation
                # arg operations are NOT allowed as inner operations
                if len(op_parts) == 1:
                    single_op = op_parts[0]
                    if argtopk_pattern.match(single_op) or single_op in ('argmax', 'argmin'):
                        raise ValueError(f"Invalid op '{op}': arg operations (argmax, argmin, argmax3, etc.) cannot be used alone. They are only valid as outer operations in compound form like 'argmax_mean'.")

                if len(op_parts) > 1:
                    outer, inner = op_parts[0], op_parts[1]
                    # Disallow mid as outer op
                    if outer == 'mid':
                        raise ValueError(f"Invalid composite op '{op}': 'mid' cannot be used as an outer operation. It is only valid as an inner op (standalone 'mid').")
                    # Disallow top-k, arg-top-k, argmax/argmin as inner ops
                    if topk_pattern.match(inner) or argtopk_pattern.match(inner):
                        raise ValueError(f"Invalid composite op '{op}': '{inner}' (top-k/arg-top-k) cannot be used as an inner operation.")
                    if inner in ('argmax', 'argmin'):
                        raise ValueError(f"Invalid composite op '{op}': '{inner}' cannot be used as an inner operation. arg operations are only valid as outer ops.")
                    # Disallow more than 2 levels for now
                    if len(op_parts) > 2:
                        raise ValueError(f"Invalid composite op '{op}': Only 2 levels of operations are supported.")

                if isinstance(vs, str):
                    vars_list = [vs]
                elif isinstance(vs, list):
                    vars_list = vs
                else:
                    raise ValueError(f"variables_to_save['{op}'] must be a string or list of strings/dicts")
                for var in vars_list:
                    if isinstance(var, dict):
                        var_name = next(iter(var.keys()))
                        # Explicit definition {alias: expr} -> Treat as valid virtual
                        pairs.append((var_name, op_l, True))
                    elif isinstance(var, (tuple, list)):
                        var_name = var[0]
                         # Explicit definition (alias, expr) -> Treat as valid virtual
                        pairs.append((var_name, op_l, True))
                    else:
                        pairs.append((var, op_l, False))

        # Validate each variable exists and has save_idx
        for var, _, is_explicit in pairs:
            if is_explicit:
                continue

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
            
            # If not found as direct field, check if it is a valid expression
            if not found:
                 # If the variable name contains characters other than alphanumeric and underscore,
                 # we assume it is a mathematical expression.
                 if re.search(r'[^a-zA-Z0-9_]', var):
                     found = True
                     has_save_idx = True  # We assume expressions are valid for now and let Aggregator handle/validate them.

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
