from __future__ import annotations

import shutil
from abc import ABC
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Set, Literal, Optional, Self, Type

import h5py
import numpy as np
import torch
import torch.distributed as dist

from pydantic import (BaseModel, ConfigDict, Field, FilePath, PrivateAttr,
                      field_validator, model_validator)

from cmfgpu.models.utils import StatisticsAggregator, compute_group_to_rank
from cmfgpu.modules.abstract_module import AbstractModule


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
    output_complevel: int = Field(default=4, description="Compression level for output NetCDF files", ge=0, le=9)

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
        print(f"[{self.rank}]: Initializing ModelManager with opened modules:", self.opened_modules)
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
        if not self.variables_to_save:
            return
        self._statistics_aggregator = StatisticsAggregator(
            device=self.device,
            output_dir=self.output_full_dir,
            rank=self.rank,
            num_workers=self.output_workers,
            complevel=self.output_complevel,
        )

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

    def update_statistics(self, weight: int, refresh: bool = False, BLOCK_SIZE: int = 128) -> None:
        """
        Call the statistics aggregator to update mean values at current step.
        """
        if self._statistics_aggregator is not None:
            self._statistics_aggregator.update_statistics(weight, refresh, BLOCK_SIZE)

    def finalize_time_step(self, current_time: datetime) -> None:
        """
        Finalize time step in aggregator (write current means to disk).
        """
        if self._statistics_aggregator is not None:
            self._statistics_aggregator.finalize_time_step(current_time)
    
    def get_h5data(self) -> Dict[str, Any]:
        """
        Load data for a specific module from the H5 file with unified distribution logic.
        Handles cases where rank has no data.
        """
        module_data: Dict[str, torch.Tensor] = {}
        visited_fields: Set[str] = set()  # Track visited fields to avoid duplicates
        group_indices_cache: Dict[str, np.ndarray] = {}
        
        for module_name in self.opened_modules:
            module_class = self.module_list[module_name]
            
            try:
                with h5py.File(self.input_file, 'r') as f:
                    # Get field information from the module class

                    print(f"[rank {self.rank}]: Loading data for module '{module_name}':")
                    
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
                            print(f"[rank {self.rank}]: Optional field not in H5, will use default: {field_name}")
                            continue
                        
                        # Check if this field needs distribution
                        if field_name in self.variable_group_mapping:
                            group_var = self.variable_group_mapping[field_name]

                            if group_var in f:
                                # Compute indices for this specific group variable
                                if group_var not in group_indices_cache:
                                    group_indices_cache[group_var] = self.compute_rank_indices_for_group_var(
                                        group_var, self.group_id_to_rank
                                    )

                                current_rank_indices = group_indices_cache[group_var]
                                
                                if len(current_rank_indices) == 0:
                                    raise ValueError(
                                        f"No data found for group variable '{group_var}' on rank {self.rank} for field '{field_name}'."
                                    )
                                else:
                                    # Normal case - read with indices
                                    h5_dataset = f[field_name]
                                    filtered_data = h5_dataset[:][current_rank_indices]
                                    print(f"[rank {self.rank}]: Loaded distributed field: {field_name} (shape: {filtered_data.shape} group_by: {group_var})")
                                
                                module_data[field_name] = torch.tensor(filtered_data)
                            else:
                                raise ValueError(
                                    f"Group variable '{group_var}' not found in H5 file for field '{field_name}'"
                                )
                        else:
                            # No group_by specified, load full data
                            full_data = f[field_name][()]
                            if isinstance(full_data, np.ndarray):
                                module_data[field_name] = torch.tensor(full_data)
                            else:
                                module_data[field_name] = full_data
                            print(f"[rank {self.rank}]: Loaded full field: {field_name} (no group_by)")
                        
                        visited_fields.add(field_name)
                        
            except Exception as e:
                raise RuntimeError(f"Error loading data for module '{module_name}': {e}")
            
        return module_data
        
        
    def save_states(self, current_time: Optional[datetime]) -> None:
        """
        Save model state to HDF5 files, handling distributed and global variable logic.
        Variables not in `variable_group_mapping` are saved only by rank 0.
        """
        timestamp = current_time.strftime("%Y%m%d_%H%M%S") if current_time else "latest"

        # Determine file path per-rank
        if self.world_size > 1:
            h5_path = self.output_full_dir / f"model_state_rank{self.rank}_{timestamp}.h5"
        else:
            h5_path = self.output_full_dir / f"model_state_{timestamp}.h5"

        visited_fields = set()

        for module_name in self.opened_modules:
            module = self._modules[module_name]
            try:
                with h5py.File(h5_path, 'a') as f:
                    for field_name in module.get_model_fields():
                        if field_name in module.h5_excluded_fields or field_name in visited_fields:
                            continue

                        is_distributed = field_name in self.variable_group_mapping

                        # Only rank 0 saves non-distributed variables
                        if not is_distributed and self.rank != 0:
                            continue

                        data = getattr(module, field_name)
                        if isinstance(data, torch.Tensor):
                            data = data.detach().cpu().numpy()

                        if data is not None:
                            if np.isscalar(data) or (isinstance(data, np.ndarray) and data.shape == ()):
                                f.create_dataset(field_name, data=data)
                            else:
                                f.create_dataset(field_name, data=data, compression="gzip")
                            visited_fields.add(field_name)

            except Exception as e:
                raise RuntimeError(f"Error saving data for module '{module_name}': {e}")

        if self.world_size > 1:
            dist.barrier()

        # Merge step only done by rank 0
        if self.rank == 0 and self.world_size > 1:
            merged_path = self.output_full_dir / f"model_state_{timestamp}.h5"
            with h5py.File(merged_path, 'w') as merged_file:
                for rank in range(self.world_size):
                    rank_path = self.output_full_dir / f"model_state_rank{rank}_{timestamp}.h5"
                    if not rank_path.exists():
                        raise FileNotFoundError(f"Missing file: {rank_path}")

                    with h5py.File(rank_path, 'r') as rank_file:
                        for field_name in rank_file:
                            data = rank_file[field_name][()]
                            if field_name in merged_file:
                                # Append distributed variable
                                merged_dset = merged_file[field_name]
                                merged_dset.resize((merged_dset.shape[0] + data.shape[0]), axis=0)
                                merged_dset[-data.shape[0]:] = data
                            else:
                                # Create new dataset (set maxshape if we expect appending)
                                if isinstance(data, np.ndarray) and data.ndim >= 1:
                                    merged_file.create_dataset(
                                        field_name,
                                        data=data,
                                        maxshape=(None,) + data.shape[1:],
                                        compression='gzip'
                                    )
                                else:
                                    merged_file.create_dataset(field_name, data=data)

                    rank_path.unlink()

            print(f"[rank 0] Model state merged to: {merged_path}")
                    

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
            if not self.output_full_dir.exists():
                self.output_full_dir.mkdir(parents=True, exist_ok=True)
            else:
                print(f"Warning: Output directory {self.output_full_dir} already exists. Contents may be overwritten.")
        return self
