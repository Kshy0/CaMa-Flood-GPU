# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Abstract base class for all CaMa-Flood-GPU modules using Pydantic v2.
This is the highest level abstraction that all modules inherit from.
"""
from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, Dict, List, Literal, Optional, Self, Tuple

import torch
from pydantic import (BaseModel, ConfigDict, Field, PrivateAttr,
                      computed_field, field_validator, model_validator)
from pydantic.fields import FieldInfo


def TensorField(
    description: str,
    shape: Tuple[str, ...],
    dtype: Literal["float", "int", "bool", "hpfloat"] = "float",
    group_by: Optional[str] = None,
    save_idx: Optional[str] = None,
    save_coord: Optional[str] = None,
    dim_coords: Optional[str] = None,
    category: Literal["topology", "param", "init_state"] = "param",
    mode: Literal["device", "cpu", "discard"] = "device",
    **kwargs
):
    """
    Create a tensor field with shape information directly in AbstractModule.
    
    Args:
        description: Human-readable description of the variable
        shape: Tuple of dimension names (scalar variable names)
        dtype: Data type ('float', 'int', 'bool')
        group_by: Name of the variable that indicates basin membership for this tensor.
                       If None, the full data will be loaded without distribution.
        dim_coords: Variable name that provides coordinates (IDs) for the 0th dimension.
                    Useful for selecting elements by ID (e.g. for parameter changes).
        category: Category of the variable:
                  - 'topology': Static structure (NEVER batched)
                  - 'param': Input parameter (can be batched)
                  - 'init_state': Initializable state variable (ALWAYS batched if num_trials > 1)
        mode: Handling of variables after initialization:
                  - 'device': Keep on current device (default)
                  - 'cpu': Move to CPU memory to save GPU memory
                  - 'discard': Set to None after initialization to maximize memory saving
        **kwargs: Additional Field parameters
    """
    if dtype not in ("float", "hpfloat"):
        save_idx = None

    if save_idx is None:
        save_coord = None

    return Field(
        description=description,
        **kwargs,
        json_schema_extra={
            "tensor_shape": shape,
            "tensor_dtype": dtype,
            "group_by": group_by,
            "save_idx": save_idx,
            "save_coord": save_coord,
            "dim_coords": dim_coords,
            "category": category,
            "mode": mode,
        }
    )

def computed_tensor_field(
    description: str,
    shape: Tuple[str, ...],
    dtype: Literal["float", "int", "bool", "hpfloat"] = "float",
    save_idx: Optional[str] = None,
    save_coord: Optional[str] = None,
    dim_coords: Optional[str] = None,
    category: Literal["topology", "derived_param", "state", "shared_state", "virtual"] = "derived_param",
    expr: Optional[str] = None,
    **kwargs
):
    """
    Create a computed tensor field with shape information for AbstractModule.
    
    Args:
        description: Human-readable description of the variable
        shape: Tuple of dimension names (scalar variable names)
        dtype: Data type ('float', 'int', 'bool')
        group_by: Name of the variable that indicates basin membership for this tensor.
                       If None, the full data will be loaded without distribution.
        dim_coords: Variable name that provides coordinates (IDs) for the 0th dimension.
        category: Category of the variable:
                  - 'topology': Static structure (NEVER batched)
                  - 'derived_param': Computed parameter (can be batched)
                  - 'state': Computed state variable (ALWAYS batched if num_trials > 1)
                  - 'shared_state': Computed state variable (NEVER batched)
                  - 'virtual': Computed on-demand during analysis/output (not stored in memory)
        expr: Expression string for virtual variables
        **kwargs: Additional computed_field parameters
    """
    if expr is not None and category != "virtual":
        raise ValueError("expr can only be provided when category is 'virtual'")

    if dtype not in ("float", "hpfloat"):
        save_idx = None

    if save_idx is None:
        save_coord = None

    return computed_field(
        description=description,
        json_schema_extra={
            "tensor_shape": shape,
            "tensor_dtype": dtype,
            "save_idx": save_idx,
            "save_coord": save_coord,
            "dim_coords": dim_coords,
            "category": category,
            "expr": expr
        },
        **kwargs
    )


class AbstractModule(BaseModel, ABC):
    """
    Abstract base class for all CaMa-Flood-GPU modules.
    
    This class provides the fundamental framework that all modules must follow:
    - Field discovery and validation using Pydantic v2
    - Shape information for tensor fields
    - Type safety for variables
    - Distinction between input variables and computed fields
    - Integration with PyTorch tensors
    - Device and precision management
    - Support for distributed data splitting
    
    All specific modules (base, bifurcation, reservoir, etc.) inherit from this class.
    """
    
    # Pydantic configuration
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow torch.Tensor types
        validate_assignment=False,      # Validate on assignment
        extra='ignore'                
    )
    
    # Module metadata - must be overridden in subclasses
    module_name: ClassVar[str] = "abstract"
    description: ClassVar[str] = "Abstract base module"
    dependencies: ClassVar[List[str]] = []  # List of modules this module depends on
    conflicts: ClassVar[List[str]] = []  # List of modules that cannot co-exist with this module
    group_by: ClassVar[Optional[str]] = None  # Variable indicating basin membership
    nc_excluded_fields: ClassVar[List[str]] = [
        "opened_modules", "device", "precision", "mixed_precision", "rank"
    ]  # Fields to exclude from HDF5

    opened_modules: List[str] = Field(
        default_factory=list,
    )
    rank: int = Field(
        default=0,
        description="Current process rank in distributed setup",
    )
    device: torch.device = Field(
        default=torch.device("cpu"),
        description="Device for tensors (e.g., 'cuda:0', 'cpu')",
    )
    precision: torch.dtype = Field(
        default=torch.float32,
        description="Data type for tensors",
    )
    mixed_precision: bool = Field(
        default=False,
        description=(
            "Enable mixed precision for hpfloat tensors (storage variables).\n"
            "When True, hpfloat tensors are promoted one level above base precision:\n"
            "  bfloat16 → float32, float32 → float64, float64 → float64 (no promotion)."
        ),
    )
    num_trials: Optional[int] = Field(
        default=None,
        description="Number of parallel simulations (ensemble members)",
    )

    @property
    def high_precision(self) -> torch.dtype:
        """Return the dtype for hpfloat tensors.

        When ``mixed_precision`` is False (default), hpfloat uses the same
        dtype as ``precision`` — all tensors share one precision level.

        When ``mixed_precision`` is True, hpfloat is promoted one level:
          bfloat16 → float32, float32 → float64, float64 → float64.
        This mirrors Fortran CaMa-Flood's JPRD (double-precision) for
        storage variables (P2RIVSTO, P2FLDSTO, etc.).
        """
        if not self.mixed_precision:
            return self.precision
        _hp_map = {
            torch.bfloat16: torch.float32,
            torch.float32: torch.float64,
            torch.float64: torch.float64,
        }
        return _hp_map.get(self.precision, self.precision)
    
    _expanded_params: set = PrivateAttr(default_factory=set)

    @field_validator('num_trials')
    @classmethod
    def validate_num_trials(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 1:
            raise ValueError("num_trials must be greater than 1 if specified. For single trial, use None.")
        return v

    def model_post_init(self, __context: Any):
        if self.module_name not in self.opened_modules:
            raise ValueError(
                f"`{self.module_name}` is not listed in `opened_modules`. "
                f"All active modules must include themselves in that list."
            )
        self.validate_tensors()
        self.init_optional_tensors()
        self.validate_computed_tensors()

    @classmethod
    def get_model_fields(cls) -> Dict[str, FieldInfo]:
        return cls.model_fields
    
    @classmethod
    def get_model_computed_fields(cls) -> Dict[str, FieldInfo]:
        return cls.model_computed_fields

    
    def init_optional_tensors(self) -> None:
        """
        Initialize optional tensor fields:
        - If None -> zeros with expected shape
        - If scalar default -> full with that value and expected shape
        - If already a tensor -> skip
        """
        for name, field_info in self.get_model_fields().items():
            # Check if it is a TensorField by looking for tensor_shape in json_schema_extra
            json_schema_extra = getattr(field_info, 'json_schema_extra', None)
            if json_schema_extra is None or 'tensor_shape' not in json_schema_extra:
                continue

            if name in self.model_fields_set:
                continue
            value = getattr(self, name, None)
            # shape
            expected_shape = self.get_expected_shape(name)
            # dtype
            tensor_dtype = field_info.json_schema_extra.get('tensor_dtype', 'float')
            dtype_map = {
                'float': self.precision,
                'hpfloat': self.high_precision,
                'int': torch.int64,
                'bool': torch.bool
            }
            target_dtype = dtype_map.get(tensor_dtype)

            if value is None:
                tensor_value = None
            elif isinstance(value, (int, float, bool)):
                tensor_value = torch.full(expected_shape, fill_value=value, dtype=target_dtype, device=self.device)
            else:
                raise TypeError(f"Unsupported default type for {name}: {type(value)}")

            setattr(self, name, tensor_value)

    def get_expected_shape(self, field_name: str) -> Tuple[int, ...]:
        """
        Get the expected shape for a tensor field based on current scalar values.
        
        Args:
            field_name: Name of the tensor field
            
        Returns:
            Tuple of integer dimensions
        """
        model_fields = self.get_model_fields() | self.get_model_computed_fields()
        if field_name not in model_fields:
            raise ValueError(f"Field {field_name} is not a tensor field")
        json_schema_extra = getattr(model_fields[field_name], 'json_schema_extra', None)
        if json_schema_extra is None:
            json_schema_extra = {}
        shape_spec = json_schema_extra.get('tensor_shape', None)
        if shape_spec is None:
            return None
        
        # Get current scalar values from instance
        scalar_values = {}
        for dim_name in shape_spec:
            # Handle dotted notation (e.g., "base.num_flood_levels")
            if "." in dim_name:
                parts = dim_name.split(".")
                if len(parts) != 2:
                    raise ValueError(f"Invalid dimension format: {dim_name}. Expected 'module.attribute'")
                module_name, attr_name = parts
                if not hasattr(self, module_name):
                    raise ValueError(f"Module {module_name} not found in {self.module_name} for dimension {dim_name}")
                module_obj = getattr(self, module_name)
                if not hasattr(module_obj, attr_name):
                    raise ValueError(f"Attribute {attr_name} not found in module {module_name} for dimension {dim_name}")
                scalar_values[dim_name] = getattr(module_obj, attr_name)
                continue

            if hasattr(self, dim_name):
                scalar_values[dim_name] = getattr(self, dim_name)
            else:
                raise ValueError(f"Dimension {dim_name} not found in module")
        
        shape = tuple(scalar_values[dim] for dim in shape_spec)
        
        category = json_schema_extra.get('category', 'param')
        if self.num_trials is not None:
             if category in ('state', 'init_state') or \
                (category in ('param', 'derived_param') and field_name in self._expanded_params):
                 return (self.num_trials,) + shape
        
        return shape
    

    def get_expected_dtype(self, field_name: str) -> torch.dtype:
        """
        Get the expected data type for a tensor field based on its definition.
        
        Args:
            field_name: Name of the tensor field
            
        Returns:
            Expected torch.dtype for the tensor
        """
        model_fields = self.get_model_fields() | self.get_model_computed_fields()
        if field_name not in model_fields:
            raise ValueError(f"Field {field_name} is not a tensor field")
        json_schema_extra = getattr(model_fields[field_name], 'json_schema_extra', None)
        if json_schema_extra is None:
            json_schema_extra = {}
        dtype_str = json_schema_extra.get('tensor_dtype', 'float')
        
        dtype_map = {
            'float': self.precision,
            'hpfloat': self.high_precision,
            'int': torch.int64,
            'bool': torch.bool
        }
        
        return dtype_map.get(dtype_str, torch.float32)
    
    def validate_tensors(self) -> bool:
        """
        Validate and auto-fix tensor consistency issues.
        - Ensures contiguity
        - Validates shapes (fails on mismatch)
        - Ensures device consistency (moves to self.device if needed)
        - Ensures precision consistency for floating-point tensors
        - Ensures int tensors are int64
        """
        auto_fix_log = {}

        for field_name, field_info in self.get_model_fields().items():
            # Check if it is a TensorField by looking for tensor_shape in json_schema_extra
            json_schema_extra = getattr(field_info, 'json_schema_extra', None)
            if json_schema_extra is None or 'tensor_shape' not in json_schema_extra:
                continue

            tensor = getattr(self, field_name, None)
            
            if tensor is None or not isinstance(tensor, torch.Tensor):
                continue
                
            # 1. Shape validation (fail fast)
            expected_shape = self.get_expected_shape(field_name)
            if tensor.shape != expected_shape:
                # Try to expand if it's a state/init_state variable and shape matches without batch dim
                category = json_schema_extra.get('category', 'param')
                if category in ('state', 'init_state') and self.num_trials is not None:
                    # Check if it matches expected shape excluding the first dim (num_trials)
                    if tensor.shape == expected_shape[1:]:
                        tensor = tensor.unsqueeze(0).expand(expected_shape).clone()
                        setattr(self, field_name, tensor)
                    else:
                        raise ValueError(f"Shape mismatch for {field_name}: expected {expected_shape}, got {tensor.shape}")
                elif category in ('param', 'derived_param') and self.num_trials is not None:
                    # Check if it is already batched but not marked as expanded
                    if tensor.shape[0] == self.num_trials and tensor.shape[1:] == expected_shape:
                        # It is batched, so we should mark it as expanded
                        self._expanded_params.add(field_name)
                        # Re-check expected shape now that it is marked expanded
                        expected_shape = self.get_expected_shape(field_name)
                        if tensor.shape != expected_shape:
                             raise ValueError(f"Shape mismatch for {field_name}: expected {expected_shape}, got {tensor.shape}")
                    else:
                        raise ValueError(f"Shape mismatch for {field_name}: expected {expected_shape}, got {tensor.shape}")
                else:
                    raise ValueError(f"Shape mismatch for {field_name}: expected {expected_shape}, got {tensor.shape}")
            
            # 2. Auto-fix contiguity
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
                
            # 3. Auto-fix device mismatch
            if tensor.device != self.device:
                tensor = tensor.to(self.device)
                
            # 4. Auto-fix precision for floating-point tensors
            expected_dtype = self.get_expected_dtype(field_name)
            tensor_dtype = tensor.dtype
            if tensor_dtype != expected_dtype:
                tensor = tensor.to(expected_dtype)
                key = f"{tensor_dtype} -> {expected_dtype}"
                if key not in auto_fix_log:
                    auto_fix_log[key] = []
                auto_fix_log[key].append(field_name)
            # Update tensor if it was modified
            setattr(self, field_name, tensor)
        
        if auto_fix_log:
            for key, fields in auto_fix_log.items():
                print(f"Auto-fixed dtype for fields: {', '.join(fields)} ({key})")
        return True
    
    def validate_computed_tensors(self) -> bool:
        """
        Validate computed tensors to ensure they are correctly defined.
        """
        auto_fix_log = {}
        for field_name, field_info in self.get_model_computed_fields().items():
            json_schema_extra = getattr(field_info, 'json_schema_extra', None)
            if json_schema_extra is None:
                continue

            # Need to skip virtual fields
            category = json_schema_extra.get('category', 'derived_param')
            if category == 'virtual':
                continue

            tensor = getattr(self, field_name)
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.device != self.device:
                raise ValueError(
                    f"Computed field {field_name} must be on device {self.device}, "
                    f"but is on {tensor.device}"
                )

            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
                setattr(self, field_name, tensor)
            
            expected_shape = self.get_expected_shape(field_name)
            if tensor.shape != expected_shape:
                # Check if it is batched (num_trials > 1)
                if self.num_trials is not None and tensor.shape == (self.num_trials,) + expected_shape:
                    # Mark as expanded so get_expected_shape returns the correct batched shape
                    self._expanded_params.add(field_name)
                else:
                    raise ValueError(
                        f"Computed field {field_name} has shape {tensor.shape}, "
                        f"but expected shape is {expected_shape}"
                    )
            
            expected_dtype = self.get_expected_dtype(field_name)
            if tensor.dtype != expected_dtype:
                key = f"{tensor.dtype} -> {expected_dtype}"
                if key not in auto_fix_log:
                    auto_fix_log[key] = []
                auto_fix_log[key].append(field_name)
                tensor = tensor.to(expected_dtype)
                setattr(self, field_name, tensor)
        
        if auto_fix_log:
            for key, fields in auto_fix_log.items():
                print(f"Auto-fixed dtype for computed fields: {', '.join(fields)} ({key})")
        return True
    
    @model_validator(mode="after")
    def validate_opened_modules(self) -> Self:
        v = self.opened_modules
        if self.module_name not in v:
            raise ValueError(
                f"Current module '{self.module_name}' must be included in opened_modules. "
                f"Available modules: {v}"
            )

        missing_deps = [dep for dep in self.dependencies if dep not in v]
        if missing_deps:
            raise ValueError(
                f"Module '{self.module_name}' has missing dependencies in opened_modules: {missing_deps}. "
                f"Required dependencies: {self.dependencies}. "
                f"Available modules: {v}"
            )

        present_conflicts = [c for c in self.conflicts if c in v and c != self.module_name]
        if present_conflicts:
            raise ValueError(
                f"Module '{self.module_name}' conflicts with modules present in opened_modules: {present_conflicts}. "
                f"These modules cannot be enabled together."
            )
        
        return self
    

    def get_memory_usage(self) -> int:
        """
        Calculate the memory usage of the module in bytes.
        Excludes intermediate tensors.
        Only counts tensors currently on the active computing device.
        Uses data_ptr deduplication to avoid double-counting shared tensors.
        """
        total_bytes = 0
        seen_ptrs: set = set()
        
        # Combine fields and computed fields
        all_fields = self.get_model_fields().copy()
        all_fields.update(self.get_model_computed_fields())
        
        for name, field_info in all_fields.items():
                
            # Get the value
            if not hasattr(self, name):
                continue
            value = getattr(self, name)
            
            # Check if it's a tensor
            if isinstance(value, torch.Tensor):
                # Only count if on the main computing device
                if value.device == self.device:
                    ptr = value.data_ptr()
                    if ptr not in seen_ptrs:
                        seen_ptrs.add(ptr)
                        total_bytes += value.element_size() * value.nelement()
                
        return total_bytes

    def gather_tensor(self, tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Gather values from a tensor using indices, handling potential batch dimensions.
        
        If tensor is (N, ...), returns (L, ...) where L = len(indices).
        If tensor is (T, N, ...), returns (T, L, ...).
        """
        if self.num_trials is not None and tensor.shape[0] == self.num_trials:
            # Batched tensor (T, N, ...)
            # We want to gather along the second dimension (N)
            # indices is (L,)
            # Result should be (T, L, ...)
            # We can use tensor[:, indices]
            return tensor[:, indices]
        else:
            # Shared tensor (N, ...)
            # Result should be (L, ...)
            return tensor[indices]


    def _is_batched(self, tensor: torch.Tensor) -> bool:
        """
        Check if a tensor is batched (has num_trials as first dimension).
        """
        if self.num_trials is None:
            return False
        return tensor.ndim > 0 and tensor.shape[0] == self.num_trials

    @property
    def batched_params(self) -> Dict[str, bool]:
        """
        Returns a dictionary mapping parameter names to whether they are batched.
        Checks both fields and computed fields.
        """
        res = {}
        # Helper to check a field
        def check_field(name, field_info):
            # Skip state variables and topology
            category = field_info.json_schema_extra.get("category", "param") if field_info.json_schema_extra else "param"
            if category == "state" or category == "topology":
                return
            
            if not hasattr(self, name):
                return
                
            val = getattr(self, name)
            if isinstance(val, torch.Tensor):
                # Check if batched: (num_trials, ...) and num_trials > 1
                # Note: We assume that if shape[0] == num_trials and num_trials > 1, it is batched.
                is_batched = (self.num_trials is not None and val.ndim > 0 and val.shape[0] == self.num_trials)
                res[name] = is_batched

        for name, field in self.get_model_fields().items():
            check_field(name, field)
            
        for name, field in self.get_model_computed_fields().items():
            check_field(name, field)
                
        return res

    def handle_tensor_mode(self) -> None:
        """
        Process variables based on their mode setting:
        - 'device': Keep on current device (default)
        - 'cpu': Move to CPU memory
        - 'discard': Remove from memory (set to None) to save space
        """
        for name, field_info in self.get_model_fields().items():
            json_schema_extra = getattr(field_info, 'json_schema_extra', {})
            if json_schema_extra is None:
                continue

            mode = json_schema_extra.get('mode', 'device')
            
            if mode == 'device':
                continue
                
            if not hasattr(self, name):
                continue
                
            val = getattr(self, name)
            if not isinstance(val, torch.Tensor):
                # Could be None already
                continue
                
            if mode == 'cpu':
                if val.device.type != 'cpu':
                    setattr(self, name, val.cpu())
            elif mode == 'discard':
                setattr(self, name, None)
