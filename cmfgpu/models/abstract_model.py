from __future__ import annotations

from abc import ABC
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Self, Type

import numpy as np
import numpy.ma as ma
import torch
import torch.distributed as dist
from netCDF4 import Dataset
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
    input_file: FilePath = Field(default=..., description="Path to the NetCDF (.nc) data file")
    output_dir: Path = Field(default_factory=lambda: Path("./out"), description="Path to the output directory")
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
        module_data = self.shard_param()  # reads from NetCDF
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
                if field_name in module_class.nc_excluded_fields:
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
        Load primary group variable from NetCDF and compute
        a full ID->rank map using compute_group_to_rank.
        """
        with Dataset(self.input_file, 'r') as ds:
            if self.group_by not in ds.variables:
                raise ValueError(f"Missing primary group variable '{self.group_by}' in NetCDF file.")
            grp = np.asarray(ds.variables[self.group_by][:])
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
        with Dataset(self.input_file, 'r') as ds:
            if group_var not in ds.variables:
                raise ValueError(f"Group variable '{group_var}' not found in NetCDF file.")
            data = np.asarray(ds.variables[group_var][:])
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

    def shard_param(self) -> Dict[str, Any]:
        """
        Load fields by reading full datasets once and slicing in-memory per rank.
        This implementation reads from a NetCDF (.nc) file via netCDF4.
        """
        module_data: Dict[str, torch.Tensor] = {}

        # Collect unique fields to load across all opened modules
        fields_to_load: Dict[str, Any] = {}
        for module_name in self.opened_modules:
            module_class = self.module_list[module_name]
            for field_name, field_info in module_class.get_model_fields().items():
                if field_name in module_class.nc_excluded_fields:
                    continue
                if field_name not in fields_to_load:
                    fields_to_load[field_name] = field_info

        def read_var(ds: Dataset, name: str) -> np.ndarray:
            v = ds.variables[name][:]
            if ma.isMaskedArray(v):
                # Fill masked values conservatively depending on dtype
                if np.issubdtype(v.dtype, np.floating):
                    return np.asarray(v.filled(np.nan))
                else:
                    return np.asarray(v.filled(-1))
            return np.asarray(v)

        try:
            with Dataset(self.input_file, "r") as ds:
                # Validate required fields exist
                missing_required = [
                    name for name, info in fields_to_load.items()
                    if info.is_required() and name not in ds.variables
                ]
                if missing_required:
                    raise KeyError(
                        f"Required fields missing from NetCDF file: {missing_required}. "
                        f"Available fields: {list(ds.variables.keys())}"
                    )

                # Pre-compute indices per group var for current rank (single file open)
                group_vars_needed = {
                    self.variable_group_mapping[name]
                    for name in fields_to_load.keys()
                    if name in self.variable_group_mapping
                }
                group_indices_cache: Dict[str, np.ndarray] = {}
                for group_var in group_vars_needed:
                    if group_var not in ds.variables:
                        raise ValueError(f"Group variable '{group_var}' not found in NetCDF file.")
                    grp = read_var(ds, group_var)
                    idx = np.nonzero(self.group_id_to_rank[grp] == self.rank)[0]
                    group_indices_cache[group_var] = idx

                print(f"[rank {self.rank}]: Loading data for modules {self.opened_modules}")

                def to_torch(arr: Any) -> torch.Tensor:
                    # Use as_tensor to avoid unnecessary copy; unify float dtype only
                    t = torch.as_tensor(arr)
                    if t.is_floating_point() and t.dtype != self.dtype:
                        t = t.to(self.dtype)
                    return t

                for field_name, field_info in fields_to_load.items():
                    if field_name not in ds.variables:
                        print(f"[rank {self.rank}]: Optional field not in NetCDF, will use default: {field_name}")
                        continue

                    full_np = read_var(ds, field_name)

                    group_var = self.variable_group_mapping.get(field_name, None)
                    if group_var is not None:
                        idx = group_indices_cache[group_var]
                        if idx.size == 0:
                            # Construct empty with correct trailing shape
                            base_shape = full_np.shape[1:] if isinstance(full_np, np.ndarray) else ()
                            empty_np = np.empty((0, *base_shape), dtype=getattr(full_np, "dtype", np.float32))
                            module_data[field_name] = to_torch(empty_np)
                            print(f"[rank {self.rank}]: No local data for distributed field: {field_name} (group_by: {group_var})")
                        else:
                            local_np = full_np[idx]
                            module_data[field_name] = to_torch(local_np)
                            print(f"[rank {self.rank}]: Loaded distributed field: {field_name} (shape: {local_np.shape}, group_by: {group_var})")
                    else:
                        module_data[field_name] = to_torch(full_np)
                        print(f"[rank {self.rank}]: Loaded full field: {field_name} (no group_by)")

        except Exception as e:
            raise RuntimeError(f"Error loading data from NetCDF: {e}")

        return module_data

    def save_state(self, current_time: Optional[datetime]) -> None:
        """
        Save model state to NetCDF files (.nc), handling distributed and global variable logic.
        Variables not in `variable_group_mapping` are saved only by rank 0.
        Rank>0 will skip non-distributed variables.
        """
        timestamp = current_time.strftime("%Y%m%d_%H%M%S") if current_time else "latest"

        # Determine file path per-rank
        if self.world_size > 1:
            nc_path = self.output_full_dir / f"model_state_rank{self.rank}_{timestamp}.nc"
        else:
            nc_path = self.output_full_dir / f"model_state_{timestamp}.nc"

        # Helpers for NetCDF writing
        def _ensure_dim(ds: Dataset, name: str, size: Optional[int], unlimited: bool = False) -> None:
            if name in ds.dimensions:
                # If exists, optionally validate size (skip if unlimited or None)
                return
            ds.createDimension(name, None if unlimited else size)

        def _infer_and_write_var(ds: Dataset, name: str, data: np.ndarray, output_complevel: int) -> None:
            arr = np.asarray(data)
            # netCDF4 does not support bool reliably across libs; write as unsigned byte
            if arr.dtype == np.bool_:
                vtype = 'u1'
                arr_to_write = arr.astype('u1')
            else:
                vtype = arr.dtype
                arr_to_write = arr

            # Define dimensions
            if arr.ndim == 0:
                dims = ()
            else:
                dims = []
                for ax, sz in enumerate(arr.shape):
                    dim_name = f"{name}_dim{ax}"
                    _ensure_dim(ds, dim_name, sz, unlimited=False)
                    dims.append(dim_name)

            # Create variable if missing
            if name not in ds.variables:
                kwargs = {}
                if len(dims) > 0:
                    kwargs = dict(zlib=True, complevel=output_complevel, shuffle=True)
                var = ds.createVariable(name, vtype, dims, **kwargs)
            else:
                var = ds.variables[name]

            # Write data
            if arr.ndim == 0:
                var.assignValue(arr_to_write)
            else:
                var[:] = arr_to_write

        visited_fields: set[str] = set()
        if nc_path.exists():
            print(f"[rank {self.rank}] Warning: Overwriting existing model state file: {nc_path}")
        # Write per-rank file
        with Dataset(nc_path, 'w', format='NETCDF4') as ds:
            ds.title = "CaMa-Flood-GPU Model State (per-rank)" if self.world_size > 1 else "CaMa-Flood-GPU Model State"
            ds.history = f"Created by CaMa-Flood-GPU at {datetime.now().isoformat()}"
            ds.source = "AbstractModel.save_states (netCDF4)"
            for module_name in self.opened_modules:
                module = self._modules[module_name]
                for field_name in module.get_model_fields():
                    if field_name in module.nc_excluded_fields or field_name in visited_fields:
                        continue

                    is_distributed = field_name in self.variable_group_mapping

                    # Only rank 0 saves non-distributed variables
                    if not is_distributed and self.rank != 0:
                        continue

                    data = getattr(module, field_name)
                    if isinstance(data, torch.Tensor):
                        data = data.detach().cpu().numpy()

                    if data is None:
                        continue

                    _infer_and_write_var(ds, field_name, np.asarray(data), output_complevel=self.output_complevel if self.world_size == 1 else 0)
                    visited_fields.add(field_name)

        if self.world_size > 1:
            dist.barrier()

        # Merge step only done by rank 0
        if self.rank == 0 and self.world_size > 1:
            merged_path = self.output_full_dir / f"model_state_{timestamp}.nc"
            offsets: Dict[str, int] = {}

            with Dataset(merged_path, 'w', format='NETCDF4') as merged_ds:
                merged_ds.title = "CaMa-Flood-GPU Model State (merged)"
                merged_ds.history = f"Created by CaMa-Flood-GPU at {datetime.now().isoformat()}"
                merged_ds.source = "AbstractModel.save_states (netCDF4 merge)"

                for r in range(self.world_size):
                    rank_path = self.output_full_dir / f"model_state_rank{r}_{timestamp}.nc"
                    if not rank_path.exists():
                        raise FileNotFoundError(f"Missing file: {rank_path}")

                    with Dataset(rank_path, 'r') as rank_ds:
                        for var_name, var_in in rank_ds.variables.items():
                            is_distributed = var_name in self.variable_group_mapping
                            data = np.asarray(var_in[:])

                            # Define/create dims and variable in merged file
                            if var_name not in merged_ds.variables:
                                # Build dims
                                if data.ndim == 0:
                                    dims = ()
                                else:
                                    dims = []
                                    for ax, sz in enumerate(data.shape):
                                        if is_distributed and ax == 0:
                                            dname = f"{var_name}_n"
                                            _ensure_dim(merged_ds, dname, None, unlimited=True)
                                        else:
                                            dname = f"{var_name}_dim{ax}"
                                            _ensure_dim(merged_ds, dname, sz, unlimited=False)
                                        dims.append(dname)

                                # Dtype handling
                                if data.dtype == np.bool_:
                                    vtype = 'u1'
                                else:
                                    vtype = data.dtype

                                kwargs = {}
                                if len(dims) > 0:
                                    kwargs = dict(zlib=True, complevel=self.output_complevel, shuffle=True)
                                merged_var = merged_ds.createVariable(var_name, vtype, tuple(dims), **kwargs)
                            else:
                                merged_var = merged_ds.variables[var_name]

                            # Write/append
                            if data.ndim == 0:
                                # Only copy from rank 0 for non-distributed scalars
                                if r == 0:
                                    if data.dtype == np.bool_:
                                        merged_var.assignValue(data.astype('u1'))
                                    else:
                                        merged_var.assignValue(data)
                            else:
                                if is_distributed:
                                    off = offsets.get(var_name, 0)
                                    n = data.shape[0]
                                    if data.dtype == np.bool_:
                                        data = data.astype('u1')
                                    merged_var[off:off + n, ...] = data
                                    offsets[var_name] = off + n
                                else:
                                    # Only copy non-distributed arrays from rank 0
                                    if r == 0:
                                        if data.dtype == np.bool_:
                                            data = data.astype('u1')
                                        merged_var[:] = data

                    # Remove rank file after merging
                    try:
                        rank_path.unlink()
                    except Exception:
                        pass

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
