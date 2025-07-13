from __future__ import annotations
import torch
import shutil
import h5py
from functools import cached_property
from datetime import datetime
from typing import Dict, Any, Literal, ClassVar, Optional, List, Type, Self
from abc import ABC
from pydantic import BaseModel, Field, ConfigDict, field_validator, FilePath, model_validator, PrivateAttr
from pathlib import Path
from cmfgpu.models.utils import compute_group_to_rank, StatisticsAggregator
from cmfgpu.modules.abstract_module import AbstractModule
import numpy as np

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
    input_file: FilePath = Field(default=..., description="Path to the H5 data file")
    output_dir: Path = Field(default_factory=lambda :Path("./out"), description="Path to the H5 data file")
    opened_modules: List[str] = Field(default_factory=list, description="List of active modules")
    variables_to_save: Optional[List[str]] = Field(None, description="Variables to be collected during the simulation")
    precision: Literal["float32", "float64"] = Field(default="float32", description="Precision of the model")
    world_size: int = Field(default=1, description="Total number of distributed processes")
    rank: int = Field(default=0, description="Current process rank in distributed setup")
    device: torch.device = Field(default=torch.device("cpu"), description="Device for tensors (e.g., 'cuda:0', 'cpu')")
    BLOCK_SIZE: int = Field(default=256, description="GPU block size for kernels")
    output_workers: int = Field(default=2, description="Number of workers for writing output files")

    _modules: Dict[str, AbstractModule] = PrivateAttr(default_factory=dict)
    _statistics_aggregator: Optional[StatisticsAggregator] = PrivateAttr(default=None)

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
    
    def model_post_init(self, __context):
        """
        Post-initialization hook to validate opened modules and register them.
        """
        print(f"Initializing ModelManager (rank: {self.rank} world size: {self.world_size}) with opened modules:", self.opened_modules)
        print(f"Using primary group variable: {self.group_by}")
        
        # Validate that all opened modules are registered
        module_data = self.get_h5data()
        for module_name in self.opened_modules:
            
            # Register the module instance with data
            module_class = self.module_list[module_name]
            module_instance = module_class(
                opened_modules=self.opened_modules, 
                rank=self.rank,
                device=self.device,
                world_size=self.world_size,
                precision=self.dtype, 
                log_path=self.log_path,
                **module_data
            )
            self._modules[module_name] = module_instance
        self.initialize_statistics_aggregator()
        print("All modules initialized successfully.")

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
        for module_name in self.opened_modules:
            module_class = self.module_list[module_name]
            # Get all fields from the module class
            for field_name, field_info in module_class.get_model_fields().items():
                if field_name in module_class.h5_excluded_fields:
                    continue
                json_schema_extra = getattr(field_info, 'json_schema_extra', None)
                if json_schema_extra is None:
                    json_schema_extra = {}
                group_var = json_schema_extra.get('group_by', None)
                if group_var:
                    variable_group_mapping[field_name] = group_var

        return variable_group_mapping

    @cached_property
    def group_id_to_rank(self) -> np.ndarray:
        """
        Load primary group variable from H5 and compute
        a full ID->rank map using compute_group_to_rank.
        """
        with h5py.File(self.input_file, 'r') as f:
            if self.group_by not in f:
                raise ValueError(f"Missing primary group variable '{self.group_by}'")
            grp = f[self.group_by][()]
        # call the new single-step Numba function
        group_id_to_rank = compute_group_to_rank(self.world_size, grp)
        return group_id_to_rank

    def compute_rank_indices_for_group_var(
        self, group_var: str, group_id_to_rank: np.ndarray
    ) -> np.ndarray:
        """
        Given a group-variable dataset and a full ID->rank map,
        return all indices that belong to this process.
        """
        with h5py.File(self.input_file, 'r') as f:
            data = f[group_var][()]
        # boolean mask where assigned rank == current rank
        mask = (group_id_to_rank[data] == self.rank)
        return np.nonzero(mask)[0]

    def initialize_statistics_aggregator(self) -> None:
        """
        Initialize the statistics aggregator for streaming NetCDF output.
        Registers all variables to save, including their save_idx and save_coord if present.
        Avoids duplicate registration.
        """
        self._statistics_aggregator = StatisticsAggregator(
            device=self.device,
            output_dir=self.output_full_dir,
            rank=self.rank,
            num_workers=self.output_workers,
        )

        if not self.variables_to_save:
            return

        registered_vars = set()

        for var_name in self.variables_to_save:
            for module_name in self.opened_modules:
                module_instance = self.get_module(module_name)
                if not hasattr(module_instance, var_name):
                    continue

                tensor = getattr(module_instance, var_name)
                field_info = module_instance.get_model_fields().get(var_name)
                if field_info is None:
                    continue

                # Register the main tensor if not already done
                if var_name not in registered_vars:
                    self._statistics_aggregator.register_tensor(var_name, tensor, field_info)
                    registered_vars.add(var_name)

                # Check for save_idx
                save_idx = field_info.json_schema_extra.get("save_idx")
                if save_idx and save_idx not in registered_vars:
                    if hasattr(module_instance, save_idx):
                        save_tensor = getattr(module_instance, save_idx)
                        self._statistics_aggregator.register_tensor(save_idx, save_tensor, {})
                        registered_vars.add(save_idx)
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
                    else:
                        print(f"Warning: save_coord '{save_coord}' not found in module '{module_name}' for variable '{var_name}'")

                break  # break once var_name is found in a module

        self._statistics_aggregator.initialize_streaming_aggregation(
            variable_names=self.variables_to_save
        )

    def update_statistics(self, weight: int, refresh: bool = False) -> None:
        """
        Call the statistics aggregator to update mean values at current step.
        """
        if self._statistics_aggregator is not None:
            self._statistics_aggregator.update_statistics(weight, refresh, self.BLOCK_SIZE)

    def finalize_time_step(self, dt: datetime) -> None:
        """
        Finalize time step in aggregator (write current means to disk).
        """
        if self._statistics_aggregator is not None:
            self._statistics_aggregator.finalize_time_step(dt)
    
    def get_h5data(self) -> Dict[str, Any]:
        """
        Load data for a specific module from the H5 file with unified distribution logic.
        Handles cases where rank has no data.
        """
        module_data = {}
        visited_fields = set()  # Track visited fields to avoid duplicates
        for module_name in self.opened_modules:
            module_class = self.module_list[module_name]
            
            try:
                with h5py.File(self.input_file, 'r') as f:
                    # Get field information from the module class

                    print(f"Loading data for module '{module_name}' (rank {self.rank}):")
                    
                    # Load fields based on distribution strategy
                    
                    for field_name, field_info in module_class.get_model_fields().items():

                        if field_name in module_class.h5_excluded_fields or field_name in visited_fields:
                            continue
                        
                        # Check if field is required but missing
                        if field_info.is_required() and field_name not in f:
                            raise KeyError(
                                f"Required field '{field_name}' missing from H5 file. "
                                f"Available fields: {list(f.keys())}"
                            )
                        
                        # Skip optional fields not present in H5
                        if field_name not in f:
                            print(f"  Optional field not in H5, will use default: {field_name}")
                            continue
                        
                        # Check if this field needs distribution
                        if field_name in self.variable_group_mapping:
                            group_var = self.variable_group_mapping[field_name]

                            if group_var in f:
                                # Compute indices for this specific group variable
                                current_rank_indices = self.compute_rank_indices_for_group_var(group_var, self.group_id_to_rank)
                                
                                # Handle empty indices case
                                if len(current_rank_indices) == 0:
                                    # Create empty tensor with correct dtype
                                    h5_dataset = f[field_name]
                                    if len(h5_dataset.shape) == 1:
                                        empty_shape = (0,)
                                    else:
                                        empty_shape = (0,) + h5_dataset.shape[1:]
                                    
                                    filtered_data = np.empty(empty_shape, dtype=h5_dataset.dtype)
                                    print(f"  Created empty tensor for field: {field_name} (shape: {empty_shape})")
                                else:
                                    # Normal case - read with indices
                                    h5_dataset = f[field_name]
                                    filtered_data = h5_dataset[:][current_rank_indices]
                                    print(f"  Loaded distributed field: {field_name} (shape: {filtered_data.shape} group_by: {group_var})")
                                
                                module_data[field_name] = torch.tensor(filtered_data)
                            else:
                                raise ValueError(f"Group variable '{group_var}' not found in H5 file for field '{field_name}'")
                        else:
                            # No group_by specified, load full data
                            full_data = f[field_name][()]
                            if isinstance(full_data, np.ndarray):
                                module_data[field_name] = torch.tensor(full_data)
                            else:
                                module_data[field_name] = full_data
                            print(f"  Loaded full field: {field_name} (no group_by)")
                        visited_fields.add(field_name)
                        
            except Exception as e:
                raise RuntimeError(f"Error loading data for module '{module_name}': {e}")
            
        return module_data
        
    def save_h5data(self, module_name: str, module_instance: AbstractModule) -> None:
        """
        Save module data to H5 file, including both required and optional fields.
        
        Args:
            module_name: Name of the module
            module_instance: Instance of the module to save
        """
        h5_path = Path(self.input_file)
        
        try:
            with h5py.File(h5_path, 'a') as f:  # 'a' mode for append/create
                # Get both required and optional fields from the module
                
                # Combine both field types
                all_fields = module_instance.get_model_fields()

                print(f"Saving module '{module_name}' data to H5 file:")
                
                for field_name, field_info in all_fields.items():
                    if field_name in module_instance.h5_excluded_fields:
                        # Skip manager-controlled fields
                        continue
                    
                    value = getattr(module_instance, field_name)
                    
                    # Skip None values for optional fields
                    if value is None and not field_info.is_required():
                        print(f"  Skipping None optional field: {field_name}")
                        continue

                    if isinstance(value, torch.Tensor):
                        # Convert tensor to numpy for H5 storage
                        numpy_data = value.detach().cpu().numpy()
                        f.create_dataset(field_name, data=numpy_data)
                        print(f"  Saved tensor field: {field_name} (shape: {numpy_data.shape}, dtype: {numpy_data.dtype})")
                    elif isinstance(value, (int, float, bool)):
                        f.create_dataset(field_name, data=value)
                        print(f"  Saved scalar field: {field_name} = {value}")
                    elif isinstance(value, str):
                        f.create_dataset(field_name, data=value.encode())
                        print(f"  Saved string field: {field_name}")
                    elif isinstance(value, np.ndarray):
                        f.create_dataset(field_name, data=value)
                        print(f"  Saved numpy array field: {field_name} (shape: {value.shape}, dtype: {value.dtype})")
                    else:
                        raise TypeError(f"Unsupported type for field '{field_name}': {type(value)}")
                
                print(f"Successfully saved module '{module_name}' data to H5 file")
                
        except Exception as e:
            raise RuntimeError(f"Error saving module '{module_name}' to H5 file: {e}")
    
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
        for var in self.variables_to_save:
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
            if self.output_full_dir.exists():
                shutil.rmtree(self.output_full_dir)
            self.output_full_dir.mkdir(parents=True, exist_ok=True)
        return self