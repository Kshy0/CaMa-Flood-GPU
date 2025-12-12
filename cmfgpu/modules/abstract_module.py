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
from pydantic import (BaseModel, ConfigDict, Field, computed_field,
                      model_validator)
from pydantic.fields import FieldInfo


def TensorField(
    description: str,
    shape: Tuple[str, ...],
    dtype: Literal["float", "int", "bool"] = "float",
    group_by: Optional[str] = None,
    save_idx: Optional[str] = None,
    save_coord: Optional[str] = None,
    intermediate: bool = False,
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
        intermediate: If True, the tensor is considered an intermediate variable
                      that can be cleared after initialization to save memory.
        **kwargs: Additional Field parameters
    """
    if dtype != "float":
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
            "intermediate": intermediate,
        }
    )

def computed_tensor_field(
    description: str,
    shape: Tuple[str, ...],
    dtype: Literal["float", "int", "bool"] = "float",
    save_idx: Optional[str] = None,
    save_coord: Optional[str] = None,
    intermediate: bool = False,
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
        intermediate: If True, the tensor is considered an intermediate variable
                      that can be cleared after initialization to save memory.
        **kwargs: Additional computed_field parameters
    """
    if dtype != "float":
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
            "intermediate": intermediate,
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
        "opened_modules", "device", "precision", "rank"
    ]  # Fields to exclude from HDF5

    opened_modules: List[str] = Field(default_factory=list)
    rank: int = Field(default=0, description="Current process rank in distributed setup")
    device: torch.device = Field(default=torch.device("cpu"), description="Device for tensors (e.g., 'cuda:0', 'cpu')")
    precision: torch.dtype = Field(default=torch.float32, description="Data type for tensors")

    def model_post_init(self, __context: Any):
        if self.module_name not in self.opened_modules:
            raise ValueError(
                f"`{self.module_name}` is not listed in `opened_modules`. "
                f"All active modules must include themselves in that list."
            )
        self.validate_tensors()
        self.init_optional_tensors()
        self.validate_computed_tensors()
        # self.clear_intermediate_tensors() # Deferred to AbstractModel.model_post_init


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
        
        return tuple(scalar_values[dim] for dim in shape_spec)
    
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
                print(f"Auto-fixed dtype for {field_name}: {tensor_dtype} -> {expected_dtype}")   
            # Update tensor if it was modified
            setattr(self, field_name, tensor)
        return True
    
    def validate_computed_tensors(self) -> bool:
        """
        Validate computed tensors to ensure they are correctly defined.
        """
        for field_name in self.get_model_computed_fields():
            tensor = getattr(self, field_name)
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.device != self.device:
                raise ValueError(
                    f"Computed field {field_name} must be on device {self.device}, "
                    f"but is on {tensor.device}"
                )

            if not tensor.is_contiguous():
                raise ValueError(f"Computed field {field_name} must be contiguous, but is not")
            if tensor.shape != self.get_expected_shape(field_name):
                raise ValueError(
                    f"Computed field {field_name} has shape {tensor.shape}, "
                    f"but expected shape is {self.get_expected_shape(field_name)}"
                )
            
            expected_dtype = self.get_expected_dtype(field_name)
            if tensor.dtype != expected_dtype:
                print(f"Auto-fixed dtype for computed field {field_name}: {tensor.dtype} -> {expected_dtype}")
                tensor = tensor.to(expected_dtype)
                setattr(self, field_name, tensor)
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
    
    def clear_intermediate_tensors(self) -> None:
        """
        Clear intermediate tensors to save memory.
        These tensors are marked with `intermediate=True` in their field definition.
        """
        for name, field_info in (self.get_model_fields() | self.get_model_computed_fields()).items():
            if field_info.json_schema_extra and field_info.json_schema_extra.get("intermediate"):
                if hasattr(self, name):
                    setattr(self, name, None)
                    print(f"Cleared intermediate tensor: {name}")

    def get_memory_usage(self) -> int:
        """
        Calculate the memory usage of the module in bytes.
        Excludes intermediate tensors.
        """
        total_bytes = 0
        
        # Combine fields and computed fields
        all_fields = self.get_model_fields().copy()
        all_fields.update(self.get_model_computed_fields())
        
        for name, field_info in all_fields.items():
            # Check if it's an intermediate variable
            json_schema_extra = getattr(field_info, 'json_schema_extra', None)
            if json_schema_extra and json_schema_extra.get('intermediate'):
                continue
                
            # Get the value
            if not hasattr(self, name):
                continue
            value = getattr(self, name)
            
            # Check if it's a tensor
            if isinstance(value, torch.Tensor):
                total_bytes += value.element_size() * value.nelement()
                
        return total_bytes
