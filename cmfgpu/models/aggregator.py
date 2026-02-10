# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import random
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cftime
import netCDF4 as nc
import numpy as np
import torch
from pydantic.fields import FieldInfo

from cmfgpu.models.utils import torch_to_numpy_dtype


def _is_wsl() -> bool:
    """Check if the current system is Windows Subsystem for Linux (WSL)."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        with open("/proc/version", "r") as f:
            version_info = f.read().lower()
            return "microsoft" in version_info or "wsl" in version_info
    except Exception:
        return False

def sanitize_symbol(name: str) -> str:
    """Sanitize a string to be a valid python identifier/filename."""
    # Replace common operators with text
    name = name.replace('**', '_pow_')
    name = name.replace('^', '_pow_')
    name = name.replace('+', '_plus_')
    name = name.replace('-', '_minus_')
    name = name.replace('*', '_mul_')
    name = name.replace('/', '_div_')
    name = name.replace('.', '_dot_')
    # Replace any other non-alphanumeric with _
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Remove leading numbers
    if name and name[0].isdigit():
        name = '_' + name
    # Collapse underscores
    name = re.sub(r'_+', '_', name)
    return name.strip('_')

def _write_time_step_netcdf_process(args: Tuple) -> Tuple[str, int]:
    """Write a single time step to a NetCDF file. Each file contains a single variable."""
    (var_name, time_step_data, output_path, time_datetime) = args
    
    with nc.Dataset(output_path, 'a') as ncfile:
        safe = sanitize_symbol(var_name)
        time_var = ncfile.variables['time']
        
        # Find the data variable
        target_var = None
        if var_name in ncfile.variables:
            target_var = var_name
        elif safe in ncfile.variables:
            target_var = safe
        else:
            # Find first variable that is not dimension/coord related
            for v in ncfile.variables:
                if v not in ('time', 'trial', 'saved_points', 'levels', 'catchment_id'):
                    target_var = v
                    break
        
        if target_var is None:
            raise KeyError(f"Could not find variable for '{var_name}' (safe: '{safe}') in {output_path}")

        nc_var = ncfile.variables[target_var]
        current_len = len(nc_var)
        
        # Append data
        if time_step_data.ndim == 1:
            nc_var[current_len, :] = time_step_data
        elif time_step_data.ndim == 2:
            nc_var[current_len, :, :] = time_step_data
        elif time_step_data.ndim == 3:
            nc_var[current_len, :, :, :] = time_step_data
        
        # Append datetime
        time_unit = time_var.getncattr("units")
        calendar = time_var.getncattr("calendar")
        time_val = nc.date2num(time_datetime, units=time_unit, calendar=calendar)
        time_var[current_len] = time_val
    
    # WSL optimization: Clear page cache for the written file to prevent memory bloat
    if _is_wsl() and hasattr(os, 'posix_fadvise'):
        try:
            with open(output_path, 'rb') as f:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except Exception:
            pass
    
    return (var_name, current_len)


def _create_netcdf_file_process(args: Tuple) -> Union[Path, List[Path]]:
    """
    Process function for creating empty NetCDF files with proper structure.
    This function runs in a separate process.
    
    Args:
        args: Tuple containing (mean_var_name, metadata, coord_values, 
              output_dir, complevel, rank, year, calendar, time_unit, num_trials)
        
    Returns:
        Path or List[Path] to the created NetCDF file(s)
    """
    (mean_var_name, metadata, coord_values, output_dir, complevel, rank, year, calendar, time_unit, num_trials) = args

    safe_name = sanitize_symbol(mean_var_name)

    actual_shape = metadata.get('actual_shape', ())  # Spatial shape
    tensor_shape = metadata.get('tensor_shape', ())  # Logical grid shape
    coord_name = metadata.get('save_coord', None)
    dtype = metadata.get('dtype', 'f8')
    k_val = metadata.get('k', 1)
    
    # Helper to create a single NetCDF file
    def create_single_file(file_safe_name: str, file_var_name: str, description_suffix: str = "") -> Path:
        if year is not None:
            filename = f"{file_safe_name}_rank{rank}_{year}.nc"
        else:
            filename = f"{file_safe_name}_rank{rank}.nc"
        output_path = output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with nc.Dataset(output_path, 'w', format='NETCDF4') as ncfile:
            # Write global attributes
            ncfile.setncattr('title', f'Time series for rank {rank}: {file_var_name}')
            ncfile.setncattr('original_variable_name', file_var_name)

            # Create time dimension (unlimited for streaming)
            ncfile.createDimension('time', None)
            
            # Create spatial/vertical dimensions based on actual shape
            dim_names = ['time']  # Always start with time
            
            if num_trials > 1:
                dim_names.append('trial')
                ncfile.createDimension('trial', num_trials)
                
                dim_names.append('saved_points')
                ncfile.createDimension('saved_points', actual_shape[1])
                
                if len(actual_shape) > 2:
                    dim_names.append('levels')
                    ncfile.createDimension('levels', actual_shape[2])
            else:
                if len(actual_shape) == 1:
                    # 1D spatial: time + saved points
                    dim_names.append('saved_points')
                    ncfile.createDimension('saved_points', actual_shape[0])
                elif len(actual_shape) == 2:
                    # 2D: time + saved points + levels
                    dim_names.extend(['saved_points', 'levels'])
                    ncfile.createDimension('saved_points', actual_shape[0])
                    ncfile.createDimension('levels', actual_shape[1])
                    
            if coord_name and coord_values is not None:
                coord_var = ncfile.createVariable(
                    coord_name,
                    coord_values.dtype,
                    ('saved_points',),
                )
                coord_var[:] = coord_values

            time_var = ncfile.createVariable('time', 'f8', ('time',))
            time_var.setncattr('units', time_unit)
            time_var.setncattr('calendar', calendar)
            
            # Create single data variable
            nc_var = ncfile.createVariable(
                file_safe_name,
                dtype,
                dim_names,
                zlib=True,
                complevel=complevel)
            desc = metadata.get("description", "") + description_suffix
            nc_var.setncattr('description', desc)
            nc_var.setncattr('actual_shape', str(actual_shape))
            nc_var.setncattr('tensor_shape', str(tensor_shape))
            nc_var.setncattr('long_name', file_var_name)
        
        return output_path
    
    # For k > 1, create separate files for each k index
    if k_val > 1:
        paths = []
        for k_idx in range(k_val):
            file_safe_name = f"{safe_name}_{k_idx}"
            file_var_name = f"{mean_var_name}_{k_idx}"
            desc_suffix = f" [rank {k_idx}]"
            path = create_single_file(file_safe_name, file_var_name, desc_suffix)
            paths.append(path)
        return paths
    else:
        return create_single_file(safe_name, mean_var_name)

class StatisticsAggregator:
    """
    Handles statistics aggregation with streaming NetCDF output to minimize memory usage.
    Each time step is immediately written to disk after accumulation.
    
    Supports two modes:
    1. Streaming mode (default): Write each time step to NetCDF files incrementally.
    2. In-memory mode: Store all time steps in memory (CPU by default) for small-scale analysis.
       Results are dynamically appended, no need to pre-specify total time steps.
    """
    
    def __init__(self, device: torch.device, output_dir: Path, rank: int, 
                 num_workers: int = 4, complevel: int = 4, save_kernels: bool = False,
                 output_split_by_year: bool = False, num_trials: int = 1,
                 max_pending_steps: int = 10, calendar: str = "standard",
                 time_unit: str = "days since 1900-01-01 00:00:00",
                 in_memory_mode: bool = False, result_device: Optional[torch.device] = None):
        """
        Initialize the statistics aggregator.
        
        Args:
            device: PyTorch device for computations
            output_dir: Output directory for NetCDF files
            rank: Process rank identifier (int)
            num_workers: Number of worker processes for parallel NetCDF writing
            complevel: Compression level (1-9)
            save_kernels: Whether to save generated kernel files for inspection
            output_split_by_year: Whether to split output files by year
            num_trials: Number of parallel simulations
            max_pending_steps: Maximum number of time steps to buffer in memory before blocking.
                               Increase this to allow GPU to run ahead of disk I/O.
            calendar: CF calendar type (e.g., 'standard', 'noleap', '360_day')
            time_unit: CF time unit string (e.g., 'days since 1900-01-01 00:00:00')
            in_memory_mode: If True, store results in memory instead of writing to NC files.
                           Results are dynamically appended as time steps are finalized.
            result_device: Device for storing in-memory results. Defaults to CPU.
                          Only used when in_memory_mode=True.
        """
        self.device = device
        self.output_dir = output_dir
        self.rank = rank
        self.num_workers = num_workers
        self.complevel = complevel
        self.save_kernels = save_kernels
        self.output_split_by_year = output_split_by_year
        self.num_trials = num_trials
        self.max_pending_steps = max(1, max_pending_steps)
        self.calendar = calendar
        self.time_unit = time_unit
        self._current_year = None
        
        # In-memory mode settings
        self.in_memory_mode = in_memory_mode
        self.result_device = result_device if result_device is not None else torch.device("cpu")
        
        self._step_count = 0
        self._macro_step_index = 0  # Current macro step index (outer loop counter)
        
        # Time index tracking for argmax/argmin conversion
        # Maps macro step index -> datetime, populated during finalize_time_step
        self._macro_step_times: List[Union[datetime, cftime.datetime]] = []

        # Create kernels directory if saving is enabled
        if self.save_kernels:
            self.kernels_dir = self.output_dir / "generated_kernels"
            self.kernels_dir.mkdir(parents=True, exist_ok=True)

        # Internal state
        # Generic stats state (for all ops)
        self._variables: Set[str] = set()  # original variable names
        self._variable_ops: Dict[str, List[str]] = {}  # var -> list[ops]
        self._storage: Dict[str, torch.Tensor] = {}  # out_name -> tensor
        self._output_keys: List[str] = [] # list of keys in storage that are outputs
        self._metadata: Dict[str, Dict[str, Any]] = {}  # out_name -> meta
        self._coord_cache: Dict[str, np.ndarray] = {}
        
        self._tensor_registry: Dict[str, torch.Tensor] = {}
        self._field_registry: Dict[str, FieldInfo] = {}
        
        # Cache for sanitized names
        self._safe_name_cache: Dict[str, str] = {}

        # Streaming mode support
        self._netcdf_files: Dict[str, Path] = {}  # out_name -> NetCDF file path
        
        self._all_created_files: Set[Path] = set()
        self._files_created: bool = False

        # Thread pool for background writing
        self._write_executors: List[ProcessPoolExecutor] = []
        self._write_futures: List = []

        # Kernel state (mean fast-path)
        self._aggregator_function = None
        self._aggregator_generated = False
        self._kernel_states: Optional[Dict[str, torch.Tensor]] = None

        # Temporary file for generated kernels
        self._temp_kernel_file = None
        self._kernel_module = None
        self._saved_kernel_file = None
        self._dirty_outputs: Set[str] = set()
        
        # In-memory result tensors: out_name -> list of tensors (one per time step)
        # Only used when in_memory_mode=True
        self._result_tensors: Dict[str, List[torch.Tensor]] = {}
        self._current_time_index: int = 0  # Current time index for in-memory writing
        
        print(f"Initialized StreamingStatisticsAggregator for rank {self.rank} with {self.num_workers} workers")
        if in_memory_mode:
            print(f"  In-memory mode enabled, results will be stored on {self.result_device}")
        if self.save_kernels:
            print(f"Generated kernels will be saved to: {self.kernels_dir}")

    def get_memory_usage(self) -> int:
        """
        Calculate GPU/CPU memory usage by this aggregator's **own** buffers.
        
        Only counts tensors in ``_storage`` that are exclusively owned by the
        aggregator (accumulation buffers, inner-state buffers, weight buffers,
        etc.).  ``_kernel_states`` is intentionally excluded because it is
        merely a dict of *references* to tensors already present in
        ``_storage`` or ``_tensor_registry`` (module source tensors), and
        counting them again would lead to double-counting.
        
        In-memory result tensors are also excluded; use
        ``get_result_memory_usage()`` for those.
        
        Returns:
            Total memory usage in bytes.
        """
        total_bytes = 0
        seen_ptrs: set = set()
        
        # Storage tensors (accumulation buffers) – these are owned by the aggregator
        for name, tensor in self._storage.items():
            if isinstance(tensor, torch.Tensor):
                ptr = tensor.data_ptr()
                if ptr not in seen_ptrs:
                    seen_ptrs.add(ptr)
                    total_bytes += tensor.element_size() * tensor.numel()
        
        # _kernel_states is NOT counted here.
        
        return total_bytes
    
    def get_result_memory_usage(self) -> int:
        """
        Calculate memory usage by in-memory result tensors.
        
        Only applicable when in_memory_mode=True.
        
        Returns:
            Total memory usage in bytes for result tensors.
        """
        if not self.in_memory_mode:
            return 0
        
        total_bytes = 0
        for name, tensor_list in self._result_tensors.items():
            for tensor in tensor_list:
                if isinstance(tensor, torch.Tensor):
                    total_bytes += tensor.element_size() * tensor.numel()
        
        return total_bytes
        


    def _cleanup_temp_files(self):
        """Remove temporary kernel files."""
        if self._temp_kernel_file and os.path.exists(self._temp_kernel_file):
            try:
                os.unlink(self._temp_kernel_file)
            except Exception:
                pass

    def _cleanup_lock_files(self):
        """Remove lock files associated with NetCDF outputs."""
        # Use _all_created_files if available, fallback to _netcdf_files
        paths = getattr(self, '_all_created_files', None)
        if paths is None and hasattr(self, '_netcdf_files'):
            paths = self._netcdf_files.values()
            
        if paths:
            for output_path in paths:
                lock_path = output_path.with_suffix(output_path.suffix + '.lock')
                if lock_path.exists():
                    try:
                        os.unlink(lock_path)
                    except Exception:
                        pass
    
    def _cleanup_executor(self):
        """Clean up the write executor."""
        if self._write_executors:
            # Wait for pending writes to complete
            for future in self._write_futures:
                try:
                    future.result(timeout=30)  # Wait up to 30 seconds
                except:
                    pass
            for executor in self._write_executors:
                try:
                    executor.shutdown(wait=True)
                except Exception:
                    pass
            self._write_executors = []
            self._write_futures = []
    
    def __del__(self):
        """Clean up temporary files and executor when the object is destroyed."""
        self._cleanup_temp_files()
        self._cleanup_executor()
        self._cleanup_lock_files()
    
    def _get_safe_name(self, name: str) -> str:
        """Get or create a sanitized name for a variable/expression."""
        if name not in self._safe_name_cache:
            self._safe_name_cache[name] = sanitize_symbol(name)
        return self._safe_name_cache[name]
    
    def _generate_unique_name(self) -> str:
        """Generate a unique name for kernel files using timestamp + rank + hash."""
        timestamp = datetime.now().strftime("%H%M%S")
        random_seed = f"{self.rank}_{timestamp}_{random.randint(1000, 9999)}"
        hash_short = hashlib.md5(random_seed.encode()).hexdigest()[:6]
        return f"{timestamp}_r{self.rank}_{hash_short}"
    
    def register_tensor(self, name: str, tensor: torch.Tensor, field_info: FieldInfo) -> None:
        """
        Register a tensor with its metadata for potential aggregation.
        
        Args:
            name: Variable name
            tensor: PyTorch tensor (actual sampled data)
            field_info: Pydantic field information
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for {name}, got {type(tensor)}")
        

        self._tensor_registry[name] = tensor
        self._field_registry[name] = field_info
        
        # Pre-cache safe name
        self._get_safe_name(name)
        
        # Invalidate pre-computed states when new tensors are registered
        self._kernel_states = None

    def register_virtual_tensor(self, name: str, field_info: FieldInfo) -> None:
        """
        Register a virtual tensor (no data, just metadata).
        
        Args:
            name: Variable name
            field_info: Pydantic field information (must contain expr)
        """
        self._field_registry[name] = field_info
        self._get_safe_name(name)
        # Do NOT add to _tensor_registry since it has no storage
        self._kernel_states = None
        
    def initialize_streaming_aggregation(self, variable_ops: Optional[Dict[str, List[str]]] = None, variable_names: Optional[List[str]] = None) -> None:
        """
        Initialize streaming aggregation for specified variables.
        Creates NetCDF file structure but writes time steps incrementally.
        
        Args:
            variable_ops: Dict of variable -> op (mean|max|min|last)
            variable_names: Backward-compatible list of variable names (defaults to mean)
        """
        if variable_ops is None:
            if variable_names is None:
                raise ValueError("Either variable_ops or variable_names must be provided")
            # list[str] convenience => all mean
            variable_ops = {v: ["mean"] for v in variable_names}
        else:
            # normalize values to list[str] lowercased
            norm_ops: Dict[str, List[str]] = {}
            for var, ops in variable_ops.items():
                if ops is None:
                    ops_list = ["mean"]
                elif isinstance(ops, str):
                    ops_list = [ops]
                else:
                    ops_list = list(ops)
                norm_ops[var] = [str(op).lower() for op in ops_list]
            variable_ops = norm_ops
        print(f"Variables: {variable_ops}")
        
        # Enable streaming mode
        self._files_created = False
        
        # Initialize single time step aggregation (generic)
        self.initialize_statistics(variable_ops)
        
        # If in-memory mode, initialize result storage lists instead of starting file writers
        if self.in_memory_mode:
            self._init_result_storage()
            print(f"In-memory aggregation initialized with {len(self._result_tensors)} output variables")
        else:
            # Start the write executors (one per worker to guarantee serialization per variable)
            self._write_executors = [ProcessPoolExecutor(max_workers=1) for _ in range(self.num_workers)]
            self._write_futures = []
            print(f"Streaming aggregation system initialized successfully ({len(self._write_executors)} partitioned executors)")
    
    def initialize_in_memory_aggregation(self, variable_ops: Optional[Dict[str, List[str]]] = None, 
                                          variable_names: Optional[List[str]] = None) -> None:
        """
        Initialize in-memory aggregation for specified variables.
        Results are stored in memory (CPU by default) instead of being written to files.
        
        This is a convenience method that ensures in_memory_mode is enabled.
        
        Args:
            variable_ops: Dict of variable -> op (mean|max|min|last)
            variable_names: Backward-compatible list of variable names (defaults to mean)
            
        Raises:
            ValueError: If in_memory_mode was not enabled during initialization.
        """
        if not self.in_memory_mode:
            raise ValueError("in_memory_mode must be True to use initialize_in_memory_aggregation. "
                           "Set in_memory_mode=True when creating the aggregator.")
        
        self.initialize_streaming_aggregation(variable_ops=variable_ops, variable_names=variable_names)
    
    def _init_result_storage(self) -> None:
        """Initialize empty lists for in-memory result storage."""
        if not self.in_memory_mode:
            return
            
        self._result_tensors.clear()
        self._current_time_index = 0
        
        # Initialize empty lists for each output
        for out_name in self._output_keys:
            self._result_tensors[out_name] = []
    
    def get_results(self, as_stacked: bool = True) -> Dict[str, torch.Tensor]:
        """
        Get the in-memory result tensors.
        
        Args:
            as_stacked: If True (default), stack all time steps into a single tensor.
                       If False, return list of per-time-step tensors.
                            
        Returns:
            Dictionary mapping output names to result tensors.
            Shape (when stacked): (time_steps, *actual_shape)
            
        Raises:
            RuntimeError: If not in in-memory mode.
        """
        if not self.in_memory_mode:
            raise RuntimeError("get_results() is only available in in_memory_mode")
        
        if as_stacked:
            result = {}
            for name, tensor_list in self._result_tensors.items():
                if tensor_list:
                    result[name] = torch.stack(tensor_list, dim=0)
                else:
                    result[name] = torch.tensor([], device=self.result_device)
            return result
        else:
            return {name: list(tensor_list) for name, tensor_list in self._result_tensors.items()}
    
    def get_result(self, variable_name: str, op: str = "mean", as_stacked: bool = True) -> torch.Tensor:
        """
        Get a specific result tensor by variable name and operation.
        
        Args:
            variable_name: Name of the variable
            op: Operation type (mean, max, min, last, etc.)
            as_stacked: If True (default), stack all time steps into a single tensor.
            
        Returns:
            Result tensor for the specified variable and operation.
            
        Raises:
            RuntimeError: If not in in-memory mode.
            KeyError: If the specified variable/op combination doesn't exist.
        """
        if not self.in_memory_mode:
            raise RuntimeError("get_result() is only available in in_memory_mode")
        
        out_name = f"{variable_name}_{op}"
        if out_name not in self._result_tensors:
            raise KeyError(f"No result found for {out_name}. Available: {list(self._result_tensors.keys())}")
        
        tensor_list = self._result_tensors[out_name]
        if as_stacked and tensor_list:
            return torch.stack(tensor_list, dim=0)
        elif as_stacked:
            return torch.tensor([], device=self.result_device)
        else:
            return list(tensor_list)
    
    def get_time_index(self) -> int:
        """Get the current time index (number of finalized time steps)."""
        return self._current_time_index
    
    def reset_time_index(self) -> None:
        """Reset the time index to 0 for a new simulation run (in-memory mode only)."""
        if not self.in_memory_mode:
            raise RuntimeError("reset_time_index() is only available in in_memory_mode")
        self._current_time_index = 0
        self._macro_step_times.clear()
        # Clear result lists
        for out_name in self._result_tensors:
            self._result_tensors[out_name] = []
    
    def _create_netcdf_files(self, year: Optional[int] = None) -> None:
        """Create empty NetCDF files with proper structure for streaming."""
        if self.in_memory_mode:
            # Skip file creation in in-memory mode
            return
            
        if not self.output_split_by_year and self._files_created:
            return
        
        print(f"Creating NetCDF file structure...{' (Year: ' + str(year) + ')' if year else ''}")
        
        # Prepare file creation tasks
        creation_futures = {}
        # Use number of outputs instead of variables (supports multiple ops)
        n_outputs = len(self._metadata)
        actual_workers = max(1, min(self.num_workers, n_outputs))
        
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            items = list(self._metadata.items())
            for out_name, metadata in items:
                coord_name = metadata.get('save_coord')
                coord_values = self._coord_cache.get(coord_name, None)
                args = (out_name, metadata, coord_values, self.output_dir, self.complevel, self.rank, year, self.calendar, self.time_unit, self.num_trials)
                future = executor.submit(_create_netcdf_file_process, args)
                creation_futures[future] = (out_name, metadata.get('k', 1))
            
            # Collect results
            for future in as_completed(creation_futures):
                out_name, k_val = creation_futures[future]
                try:
                    result = future.result()
                    # Handle both single path and list of paths (for k > 1)
                    if isinstance(result, list):
                        # Multiple files for k > 1
                        self._netcdf_files[out_name] = result  # Store as list
                        for p in result:
                            self._all_created_files.add(p)
                            print(f"  Created {p.name}")
                    else:
                        self._netcdf_files[out_name] = result
                        self._all_created_files.add(result)
                        print(f"  Created {result.name}")
                except Exception as exc:
                    print(f"  Failed to create file for {out_name}: {exc}")
                    raise exc
        
        self._files_created = True
        total_files = sum(len(v) if isinstance(v, list) else 1 for v in self._netcdf_files.values())
        print(f"Created {total_files} NetCDF files for streaming")
    
    def _prepare_kernel_states(self) -> None:
        """Pre-compute and cache all tensors required for kernel execution."""
        required_tensors: Dict[str, torch.Tensor] = {}

        def get_dependencies(expr: str) -> Set[str]:
            tokens = set(re.findall(r'\b[a-zA-Z_]\w*\b', expr))
            deps = set()
            for token in tokens:
                if token in self._tensor_registry:
                    deps.add(token)
                elif token in self._field_registry:
                    f_info = self._field_registry[token]
                    cat = getattr(f_info, 'json_schema_extra', {}).get('category')
                    if cat == 'virtual':
                        sub = getattr(f_info, 'json_schema_extra', {}).get('expr')
                        if sub:
                            deps.update(get_dependencies(sub))
            return deps

        # Add original variables and their output buffers
        for var_name, ops in self._variable_ops.items():
            field_info = self._field_registry.get(var_name)
            json_extra = getattr(field_info, 'json_schema_extra', {})
            category = json_extra.get('category', 'param')
            
            if category == 'virtual':
                expr = json_extra.get('expr')
                if expr:
                    deps = get_dependencies(expr)
                    for dep in deps:
                        if dep in self._tensor_registry:
                            required_tensors[dep] = self._tensor_registry[dep]
            elif var_name in self._tensor_registry:
                required_tensors[var_name] = self._tensor_registry[var_name]

            for op in ops:
                out_name = f"{var_name}_{op}"
                required_tensors[out_name] = self._storage[out_name]
                
                # For explicit argmax/argmin operations, add their auxiliary storage
                op_parts = op.split('_')
                outer_op = op_parts[0]
                arg_match = re.match(r'arg(max|min)(\d*)$', outer_op)
                if arg_match:
                    arg_type = arg_match.group(1)
                    arg_k_str = arg_match.group(2)
                    aux_name = f"{var_name}_{arg_type}{arg_k_str or ''}_aux"
                    if aux_name in self._storage:
                        required_tensors[aux_name] = self._storage[aux_name]
                
                if op.startswith('median') or re.match(r'^q\d+', op):
                    # Compound median ops use per-inner state
                    if '_' in op:
                        inner = op.split('_')[1]
                        q_name = f"{var_name}_median_{inner}_q_state"
                        n_name = f"{var_name}_median_{inner}_n_state"
                    else:
                        q_name = f"{var_name}_median_q_state"
                        n_name = f"{var_name}_median_n_state"
                    required_tensors[q_name] = self._storage[q_name]
                    required_tensors[n_name] = self._storage[n_name]

                # Add inner states for compound ops
                if '_' in op:
                    parts = op.split('_')
                    inner = parts[1]
                    # 'last' inner op doesn't need cross-step state
                    # 'median' inner op uses its own q/n state, not generic inner_state
                    if inner not in ('last', 'median'):
                        inner_name = f"{var_name}_{inner}_inner_state"
                        if inner_name in self._storage:
                            required_tensors[inner_name] = self._storage[inner_name]
                        # Only 'mean' needs weight state
                        if inner == 'mean':
                            w_name = f"{var_name}_{inner}_weight_state"
                            if w_name in self._storage:
                                required_tensors[w_name] = self._storage[w_name]
                        elif inner == 'median':
                            qi_name = f"{var_name}_median_inner_q_state"
                            ni_name = f"{var_name}_median_inner_n_state"
                            if qi_name in self._storage:
                                required_tensors[qi_name] = self._storage[qi_name]
                            if ni_name in self._storage:
                                required_tensors[ni_name] = self._storage[ni_name]

        # Collect required dimensions and save indices
        required_dims: Set[str] = set()
        required_save_indices: Set[str] = set()
        for var_name in self._variables:
            field_info = self._field_registry[var_name]
            json_schema_extra = getattr(field_info, 'json_schema_extra', {})
            tensor_shape = json_schema_extra.get('tensor_shape', ())
            save_idx = json_schema_extra.get('save_idx')
            if save_idx:
                required_save_indices.add(save_idx)
            for dim_name in tensor_shape:
                if isinstance(dim_name, str):
                    required_dims.add(dim_name)

        # Add save_idx tensors
        for save_idx in required_save_indices:
            if save_idx in self._tensor_registry:
                required_tensors[save_idx] = self._tensor_registry[save_idx]
            else:
                raise RuntimeError(f"Save index tensor '{save_idx}' not registered")

        # Add dimension tensors/scalars
        for dim_name in required_dims:
            if dim_name in self._tensor_registry:
                tensor = self._tensor_registry[dim_name]
                if isinstance(tensor, (int, float)):
                    required_tensors[dim_name] = torch.tensor(tensor, device=self.device)
                else:
                    required_tensors[dim_name] = tensor

        self._kernel_states = required_tensors

    
    def _generate_kernel_header(self) -> List[str]:
        """Generate the header for the kernel file with documentation."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        var_list = sorted(list(self._variables))
        
        header = [
            '"""',
            f'Auto-generated Triton kernels for CaMa-Flood-GPU statistics aggregation (mean/max/min/last)',
            f'Generated at: {timestamp}',
            f'Rank: {self.rank}',
            f'Variables: {", ".join(var_list)}',
            f'Device: {self.device}',
            '',
            'Kernel Logic:',
            '- Load save_idx values to get original grid indices',
            '- Use idx to access original data: data[idx]',
            '- Store outputs using sequential indexing: out[offs]',
            '- max/min ops automatically update corresponding argmax/argmin (step index)',
            '- argmax/argmin indices are converted to datetime on NC file write',
            '- For mid: stores val when is_middle is True',
            '',
            'Optimizations Applied:',
            '- tl.static_range for compile-time loop unrolling (num_trials, bubble sort)',
            '- Merged max3+argmax3 and min3+argmin3 when coexisting to share comparisons',
            '- Base offset precomputation (shared across max/min/argmax/argmin for same var+K)',
            '- Merged maxK+minK bubble insert in single loop with shared offset',
            '- Precise mask for tl.store: mask & swap_mask to reduce write pressure',
            '"""',
            "",
            "import triton",
            "import triton.language as tl",
            "from triton.language.extra import libdevice",
            "",
            '# ============================================================================',
            f"# Generated Triton kernels for statistics aggregation - Rank {self.rank}",
            "# ============================================================================",
            "",
        ]
        return header
    
    def _save_kernel_file(self, kernel_code: str) -> None:
        """
        Save the generated kernel code to a permanent file for inspection.
        
        Args:
            kernel_code: Generated kernel code as string
        """
        # Use unique name generation
        unique_name = self._generate_unique_name()
        filename = f"kern_{unique_name}.py"
        
        self._saved_kernel_file = self.kernels_dir / filename
        
        with open(self._saved_kernel_file, 'w', encoding='utf-8') as f:
            f.write(kernel_code)

    
    def _write_and_import_kernels(self, kernel_code: str) -> None:
        """
        Write kernel code to a temporary file and import the module.
        
        Args:
            kernel_code: Generated kernel code as string
        """
        # Create temporary file with unique name
        unique_name = self._generate_unique_name()
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{unique_name}.py', delete=False) as f:
            f.write(kernel_code)
            self._temp_kernel_file = f.name
        
        # Import the module from the temporary file
        module_name = f"aggr_kernels_r{self.rank}_{unique_name}"
        spec = importlib.util.spec_from_file_location(module_name, self._temp_kernel_file)
        module = importlib.util.module_from_spec(spec)
        # Add to sys.modules to ensure proper import
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # Bind to instance
        self._kernel_module = module
        self._aggregator_function = getattr(module, 'internal_update_statistics')
        self._aggregator_generated = True

    def _transform_pow_expr(self, expr: str) -> str:
        """
        Transform power operations in an expression string to Triton-compatible tl.exp(log()).
        Power operator ** or ^ is transformed.
        """
        if '**' not in expr and '^' not in expr:
            return expr
            
        safe_expr = expr.replace('^', '**')
        
        def _visit_and_transform_pow(node):
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    new_list = []
                    for item in value:
                        if isinstance(item, ast.AST):
                            new_list.append(_visit_and_transform_pow(item))
                        else:
                            new_list.append(item)
                    setattr(node, field, new_list)
                elif isinstance(value, ast.AST):
                    setattr(node, field, _visit_and_transform_pow(value))
            
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                return ast.Call(
                    func=ast.Attribute(value=ast.Name(id='libdevice', ctx=ast.Load()), attr='pow', ctx=ast.Load()),
                    args=[node.left, node.right],
                    keywords=[]
                )
            return node

        try:
            expr_tree = ast.parse(safe_expr, mode='eval')
            expr_tree = _visit_and_transform_pow(expr_tree)
            return ast.unparse(expr_tree)
        except Exception as e:
            print(f"Warning: Failed to transform power expression '{safe_expr}': {e}")
            return safe_expr

    def _emit_variable_load(self, var_name: str, code_lines: List[str], emitted: Set[str], is_2d: bool = False):
        """Helper to emit load instructions or expression evaluation recursively."""
        if var_name in emitted:
            return
        
        # Get safe name for this variable
        safe_var_name = self._get_safe_name(var_name)
        
        info = self._field_registry.get(var_name)
        json_extra = getattr(info, 'json_schema_extra', {})
        cat = json_extra.get('category', 'param')
        
        if cat == 'virtual':
             expr = json_extra.get('expr')
             # Find dependencies
             tokens = set(re.findall(r'\b[a-zA-Z_]\w*\b', expr))
             safe_expr = expr
             for token in tokens:
                  # Only recurse if it's a known variable (field or registered tensor)
                  if token in self._field_registry or token in self._tensor_registry:
                       self._emit_variable_load(token, code_lines, emitted, is_2d)
                       # Replace token in expression with safe token
                       safe_token = self._get_safe_name(token)
                       safe_expr = re.sub(r'\b' + token + r'\b', safe_token, safe_expr)
             
             safe_expr = self._transform_pow_expr(safe_expr)
             
             indent = "        " if is_2d else "    "
             code_lines.append(f"{indent}{safe_var_name} = {safe_expr}")
        else:
             # Real tensor load
             indent = "        " if is_2d else "    "
             if is_2d:
                  code_lines.append(f"{indent}{safe_var_name} = tl.load({safe_var_name}_ptr + idx * n_levels + level, mask=mask, other=0.0)")
             else:
                  code_lines.append(f"{indent}{safe_var_name} = tl.load({safe_var_name}_ptr + idx, mask=mask, other=0.0)")
        
        emitted.add(var_name)
    
    def _generate_psquare_code(self, lines: List[str], var: str, q_ptr: str, n_ptr: str,
                               val_var: str, step_var: str, out_ptrs: 'dict[int, str] | str | None',
                               stride_expr: str,
                               offset_expr: str, indent: str, indent2: str, indent3: str) -> None:
        """
        Generate P-Square algorithm code for median/quantile computation.
        This is a reusable helper for both outer and inner median operations.
        
        The 5 markers track quantiles at positions 0, 0.25, 0.5, 0.75, 1.0.
        - q0 = min, q1 = 25th percentile, q2 = median, q3 = 75th percentile, q4 = max
        
        Args:
            lines: List to append code lines to
            var: Variable name (safe)
            q_ptr: Base pointer for q state
            n_ptr: Base pointer for n state
            val_var: Variable containing the value to insert
            step_var: Variable containing the step index
            out_ptrs: Output pointer(s) for result. Can be:
                - None: only state update, no output store
                - str: store q2 (median) to this pointer
                - dict: {marker_index: ptr} e.g. {1: q25_ptr, 2: median_ptr, 3: q75_ptr}
            stride_expr: Expression for stride between q/n values
            offset_expr: Expression for base offset
            indent, indent2, indent3: Indentation strings
        """
        # Normalize out_ptrs to dict form
        if out_ptrs is None:
            out_ptrs_dict = {}
        elif isinstance(out_ptrs, str):
            out_ptrs_dict = {2: out_ptrs}  # median = q2
        else:
            out_ptrs_dict = out_ptrs
        lines.extend([
            f"{indent}# P-Square Median Update",
            f"{indent}if {step_var} < 5:",
            f"{indent2}q_ptr_k = {q_ptr} + {step_var} * {stride_expr} + {offset_expr}",
            f"{indent2}tl.store(q_ptr_k, {val_var}, mask=mask)",
            f"{indent2}if {step_var} == 4:",
            # Load 5 values
            f"{indent3}q0 = tl.load({q_ptr} + 0 * {stride_expr} + {offset_expr}, mask=mask)",
            f"{indent3}q1 = tl.load({q_ptr} + 1 * {stride_expr} + {offset_expr}, mask=mask)",
            f"{indent3}q2 = tl.load({q_ptr} + 2 * {stride_expr} + {offset_expr}, mask=mask)",
            f"{indent3}q3 = tl.load({q_ptr} + 3 * {stride_expr} + {offset_expr}, mask=mask)",
            f"{indent3}q4 = tl.load({q_ptr} + 4 * {stride_expr} + {offset_expr}, mask=mask)",
            # Bubble sort (10 comparisons for 5 elements)
            f"{indent3}tmp=q0; q0=tl.minimum(tmp,q1); q1=tl.maximum(tmp,q1)",
            f"{indent3}tmp=q1; q1=tl.minimum(tmp,q2); q2=tl.maximum(tmp,q2)",
            f"{indent3}tmp=q2; q2=tl.minimum(tmp,q3); q3=tl.maximum(tmp,q3)",
            f"{indent3}tmp=q3; q3=tl.minimum(tmp,q4); q4=tl.maximum(tmp,q4)",
            f"{indent3}tmp=q0; q0=tl.minimum(tmp,q1); q1=tl.maximum(tmp,q1)",
            f"{indent3}tmp=q1; q1=tl.minimum(tmp,q2); q2=tl.maximum(tmp,q2)",
            f"{indent3}tmp=q2; q2=tl.minimum(tmp,q3); q3=tl.maximum(tmp,q3)",
            f"{indent3}tmp=q0; q0=tl.minimum(tmp,q1); q1=tl.maximum(tmp,q1)",
            f"{indent3}tmp=q1; q1=tl.minimum(tmp,q2); q2=tl.maximum(tmp,q2)",
            f"{indent3}tmp=q0; q0=tl.minimum(tmp,q1); q1=tl.maximum(tmp,q1)",
            # Store sorted values
            f"{indent3}tl.store({q_ptr} + 0 * {stride_expr} + {offset_expr}, q0, mask=mask)",
            f"{indent3}tl.store({q_ptr} + 1 * {stride_expr} + {offset_expr}, q1, mask=mask)",
            f"{indent3}tl.store({q_ptr} + 2 * {stride_expr} + {offset_expr}, q2, mask=mask)",
            f"{indent3}tl.store({q_ptr} + 3 * {stride_expr} + {offset_expr}, q3, mask=mask)",
            f"{indent3}tl.store({q_ptr} + 4 * {stride_expr} + {offset_expr}, q4, mask=mask)",
            # Initialize n positions
            f"{indent3}tl.store({n_ptr} + 0 * {stride_expr} + {offset_expr}, 0, mask=mask)",
            f"{indent3}tl.store({n_ptr} + 1 * {stride_expr} + {offset_expr}, 1, mask=mask)",
            f"{indent3}tl.store({n_ptr} + 2 * {stride_expr} + {offset_expr}, 2, mask=mask)",
            f"{indent3}tl.store({n_ptr} + 3 * {stride_expr} + {offset_expr}, 3, mask=mask)",
            f"{indent3}tl.store({n_ptr} + 4 * {stride_expr} + {offset_expr}, 4, mask=mask)",
        ])
        q_names = {0: 'q0', 1: 'q1', 2: 'q2', 3: 'q3', 4: 'q4'}
        for qi, ptr in out_ptrs_dict.items():
            lines.append(f"{indent3}tl.store({ptr}, {q_names[qi]}, mask=mask)")
        
        lines.extend([
            f"{indent}elif {step_var} >= 5:",
            # Load state
            f"{indent2}q0 = tl.load({q_ptr} + 0 * {stride_expr} + {offset_expr}, mask=mask)",
            f"{indent2}q1 = tl.load({q_ptr} + 1 * {stride_expr} + {offset_expr}, mask=mask)",
            f"{indent2}q2 = tl.load({q_ptr} + 2 * {stride_expr} + {offset_expr}, mask=mask)",
            f"{indent2}q3 = tl.load({q_ptr} + 3 * {stride_expr} + {offset_expr}, mask=mask)",
            f"{indent2}q4 = tl.load({q_ptr} + 4 * {stride_expr} + {offset_expr}, mask=mask)",
            f"{indent2}n0 = tl.load({n_ptr} + 0 * {stride_expr} + {offset_expr}, mask=mask).to(tl.float32)",
            f"{indent2}n1 = tl.load({n_ptr} + 1 * {stride_expr} + {offset_expr}, mask=mask).to(tl.float32)",
            f"{indent2}n2 = tl.load({n_ptr} + 2 * {stride_expr} + {offset_expr}, mask=mask).to(tl.float32)",
            f"{indent2}n3 = tl.load({n_ptr} + 3 * {stride_expr} + {offset_expr}, mask=mask).to(tl.float32)",
            f"{indent2}n4 = tl.load({n_ptr} + 4 * {stride_expr} + {offset_expr}, mask=mask).to(tl.float32)",
            # Update bounds and counts
            f"{indent2}q0 = tl.minimum(q0, {val_var})",
            f"{indent2}q4 = tl.maximum(q4, {val_var})",
            f"{indent2}n4 = n4 + 1.0",
            f"{indent2}n1 = n1 + tl.where({val_var} < q1, 1.0, 0.0)",
            f"{indent2}n2 = n2 + tl.where({val_var} < q2, 1.0, 0.0)",
            f"{indent2}n3 = n3 + tl.where({val_var} < q3, 1.0, 0.0)",
            # Desired positions
            f"{indent2}N_total = {step_var} + 1.0",
            f"{indent2}d1 = (N_total - 1) * 0.25; d2 = (N_total - 1) * 0.50; d3 = (N_total - 1) * 0.75",
            # Adjust marker 1
            f"{indent2}d = d1 - n1",
            f"{indent2}cond = ((d >= 1.0) & ((n2 - n1) > 1.0)) | ((d <= -1.0) & ((n1 - n0) > 1.0))",
            f"{indent2}dsign = tl.where(d >= 0.0, 1.0, -1.0)",
            f"{indent2}q_next = q1 + (dsign / (n2 - n0)) * ((n1 - n0 + dsign) * (q2 - q1) / (n2 - n1) + (n2 - n1 - dsign) * (q1 - q0) / (n1 - n0))",
            f"{indent2}q_linear = q1 + dsign * tl.where(dsign > 0.0, (q2 - q1) / (n2 - n1), (q1 - q0) / (n1 - n0))",
            f"{indent2}q_cand = tl.where((q0 < q_next) & (q_next < q2), q_next, q_linear)",
            f"{indent2}q1 = tl.where(cond, q_cand, q1); n1 = tl.where(cond, n1 + dsign, n1)",
            # Adjust marker 2
            f"{indent2}d = d2 - n2",
            f"{indent2}cond = ((d >= 1.0) & ((n3 - n2) > 1.0)) | ((d <= -1.0) & ((n2 - n1) > 1.0))",
            f"{indent2}dsign = tl.where(d >= 0.0, 1.0, -1.0)",
            f"{indent2}q_next = q2 + (dsign / (n3 - n1)) * ((n2 - n1 + dsign) * (q3 - q2) / (n3 - n2) + (n3 - n2 - dsign) * (q2 - q1) / (n2 - n1))",
            f"{indent2}q_linear = q2 + dsign * tl.where(dsign > 0.0, (q3 - q2) / (n3 - n2), (q2 - q1) / (n2 - n1))",
            f"{indent2}q_cand = tl.where((q1 < q_next) & (q_next < q3), q_next, q_linear)",
            f"{indent2}q2 = tl.where(cond, q_cand, q2); n2 = tl.where(cond, n2 + dsign, n2)",
            # Adjust marker 3
            f"{indent2}d = d3 - n3",
            f"{indent2}cond = ((d >= 1.0) & ((n4 - n3) > 1.0)) | ((d <= -1.0) & ((n3 - n2) > 1.0))",
            f"{indent2}dsign = tl.where(d >= 0.0, 1.0, -1.0)",
            f"{indent2}q_next = q3 + (dsign / (n4 - n2)) * ((n3 - n2 + dsign) * (q4 - q3) / (n4 - n3) + (n4 - n3 - dsign) * (q3 - q2) / (n3 - n2))",
            f"{indent2}q_linear = q3 + dsign * tl.where(dsign > 0.0, (q4 - q3) / (n4 - n3), (q3 - q2) / (n3 - n2))",
            f"{indent2}q_cand = tl.where((q2 < q_next) & (q_next < q4), q_next, q_linear)",
            f"{indent2}q3 = tl.where(cond, q_cand, q3); n3 = tl.where(cond, n3 + dsign, n3)",
            # Store state
            f"{indent2}tl.store({q_ptr} + 0 * {stride_expr} + {offset_expr}, q0, mask=mask)",
            f"{indent2}tl.store({q_ptr} + 1 * {stride_expr} + {offset_expr}, q1, mask=mask)",
            f"{indent2}tl.store({q_ptr} + 2 * {stride_expr} + {offset_expr}, q2, mask=mask)",
            f"{indent2}tl.store({q_ptr} + 3 * {stride_expr} + {offset_expr}, q3, mask=mask)",
            f"{indent2}tl.store({q_ptr} + 4 * {stride_expr} + {offset_expr}, q4, mask=mask)",
            f"{indent2}tl.store({n_ptr} + 0 * {stride_expr} + {offset_expr}, n0.to(tl.int32), mask=mask)",
            f"{indent2}tl.store({n_ptr} + 1 * {stride_expr} + {offset_expr}, n1.to(tl.int32), mask=mask)",
            f"{indent2}tl.store({n_ptr} + 2 * {stride_expr} + {offset_expr}, n2.to(tl.int32), mask=mask)",
            f"{indent2}tl.store({n_ptr} + 3 * {stride_expr} + {offset_expr}, n3.to(tl.int32), mask=mask)",
            f"{indent2}tl.store({n_ptr} + 4 * {stride_expr} + {offset_expr}, n4.to(tl.int32), mask=mask)",
        ])
        for qi, ptr in out_ptrs_dict.items():
            lines.append(f"{indent2}tl.store({ptr}, {q_names[qi]}, mask=mask)")

    def _generate_1d_vars_grouped(self, kernel_code_lines: List[str], dims_1d: List[str],
                                    indent: str, indent2: str, indent3: str, indent4: str, indent5: str) -> None:
        """
        Generate 1D variable processing code with conditions grouped for efficiency.
        All operations under the same condition are emitted in a single if block.
        Supports all ops including maxK/minK bubble insert and median P-Square.
        
        Optimization: arg operations (argmax, argmin, argmax3, etc.) can only be outer ops.
        When max+argmax or max3+argmax3 coexist, they are merged to share comparisons.
        """
        from collections import defaultdict
        
        if not dims_1d:
            return
            
        kernel_code_lines.append(f"{indent}# 1D variables")
        
        # Phase 0: Analyze max/argmax pairs for merging optimization
        # For each variable, detect which ops coexist to enable merging
        var_op_analysis = {}  # var -> {'max': bool, 'argmax': bool, 'max3': int, 'argmax3': int, ...}
        for var in dims_1d:
            ops = self._variable_ops[var]
            analysis = {
                'max': 'max' in ops,
                'argmax': 'argmax' in ops,
                'min': 'min' in ops,
                'argmin': 'argmin' in ops,
                'maxK': {},   # k -> True
                'argmaxK': {},  # k -> True
                'minK': {},
                'argminK': {},
            }
            for op in ops:
                match_maxk = re.match(r'^max(\d+)$', op)
                match_argmaxk = re.match(r'^argmax(\d+)$', op)
                match_mink = re.match(r'^min(\d+)$', op)
                match_argmink = re.match(r'^argmin(\d+)$', op)
                if match_maxk:
                    analysis['maxK'][int(match_maxk.group(1))] = True
                if match_argmaxk:
                    analysis['argmaxK'][int(match_argmaxk.group(1))] = True
                if match_mink:
                    analysis['minK'][int(match_mink.group(1))] = True
                if match_argmink:
                    analysis['argminK'][int(match_argmink.group(1))] = True
            var_op_analysis[var] = analysis
        
        # Phase 1: Classify variables by when their value is needed
        vars_need_val = set()  # vars that need val loaded unconditionally (every sub-step)
        vars_conditional_only = set()  # vars only used conditionally (first/last/mid or compound with last/first/mid inner)
        
        # Simple ops that are conditional (only need val at specific points)
        simple_conditional_ops = {'first', 'last', 'mid'}
        # Inner op types that only need the value conditionally (at is_inner_last)
        conditional_inner_ops = {'last', 'first', 'mid'}
        
        for var in dims_1d:
            ops = self._variable_ops[var]
            needs_unconditional = False
            for op in ops:
                if op in simple_conditional_ops:
                    continue  # Conditional simple op
                op_parts = op.split('_')
                if len(op_parts) > 1:
                    inner = op_parts[1]
                    if inner in conditional_inner_ops:
                        continue  # Compound op with conditional inner type
                # Any other op needs unconditional val (mean, sum, max, min, median, etc.)
                needs_unconditional = True
                break
            if needs_unconditional:
                vars_need_val.add(var)
            else:
                vars_conditional_only.add(var)
        
        # Helper to emit variable value load
        emitted_vars = set()
        def emit_val(v_name, to_lines):
            safe_v_name = self._get_safe_name(v_name)
            if safe_v_name in emitted_vars:
                return f"{safe_v_name}_val"
            
            info = self._field_registry.get(v_name)
            cat = getattr(info, 'json_schema_extra', {}).get('category', 'param')
            
            if cat == 'virtual':
                expr = getattr(info, 'json_schema_extra', {}).get('expr', '')
                safe_expr = expr
                toks = set(re.findall(r'\b[a-zA-Z_]\w*\b', expr))
                for t in toks:
                    if t in self._field_registry or t in self._tensor_registry:
                        emit_val(t, to_lines)
                        safe_t = self._get_safe_name(t)
                        safe_expr = re.sub(r'\b' + t + r'\b', f"{safe_t}_val", safe_expr)
                safe_expr = self._transform_pow_expr(safe_expr)
                to_lines.append(f"{indent}{safe_v_name}_val = {safe_expr}")
            else:
                in_ptr_loc = f"{safe_v_name}_ptr + t * stride_input + idx"
                to_lines.append(f"{indent}{safe_v_name}_val = tl.load({in_ptr_loc}, mask=mask, other=0.0)")
            
            emitted_vars.add(safe_v_name)
            return f"{safe_v_name}_val"
        
        # Phase 2: Collect all operations grouped by condition
        ops_unconditional = []
        ops_is_inner_first = []
        ops_not_is_inner_first = []
        ops_is_inner_last = []
        ops_is_inner_last_is_outer_first = []
        
        # Special storage for maxK/minK operations (need for loop)
        self._maxk_ops = []
        self._argmaxk_ops = []
        ops_is_inner_last_not_is_outer_first = []
        ops_is_inner_last_is_outer_last = []
        ops_is_inner_last_not_is_outer_last = []
        ops_is_middle = []
        
        # Track which inner aggregations are needed
        inner_aggregations_needed = defaultdict(set)  # inner_type -> set of vars
        
        # Track which merged operations we've already processed
        processed_merged_ops = set()  # (var, op_type, k) tuples
        
        for var in dims_1d:
            safe_var = self._get_safe_name(var)
            ops = self._variable_ops[var]
            out_offset = "t * n_saved_points + offs"
            analysis = var_op_analysis[var]
            
            # Check for compound ops that need inner aggregation
            for op in ops:
                if '_' in op:
                    inner = op.split('_')[1]
                    inner_aggregations_needed[inner].add(var)
                
            # Process each operation
            for op in ops:
                out_ptr = f"{safe_var}_{op}_ptr + {out_offset}"
                op_parts = op.split('_')
                
                # ===== Compound operations (e.g., max_mean, min_mean) =====
                if len(op_parts) > 1:
                    outer = op_parts[0]
                    inner = op_parts[1]
                    
                    # Parse maxK/minK/argmaxK/argminK pattern
                    k_val = 1
                    match_k = re.match(r'(arg)?(max|min)(\d+)$', outer)
                    is_arg_compound = outer.startswith('arg')
                    if match_k:
                        is_arg_compound = match_k.group(1) is not None
                        outer_base = match_k.group(2)  # 'max' or 'min'
                        k_val = int(match_k.group(3))
                    else:
                        outer_base = outer.lstrip('arg')  # Remove 'arg' prefix if present
                    
                    # Use variable-specific inner aggregation result
                    # For 'last' inner type, directly use the variable value (no intermediate state)
                    if inner == 'last':
                        val_var = f"{safe_var}_val"
                    else:
                        val_var = f"val_for_{safe_var}_{inner}"
                    
                    if is_arg_compound:
                        # Compound argmax/argmin (e.g., argmax_mean, argmax3_mean)
                        arg_type = outer_base  # 'max' or 'min'
                        aux_ptr_base = f"{safe_var}_{arg_type}{k_val if k_val > 1 else ''}_aux_ptr"
                        
                        if k_val == 1:
                            aux_ptr = f"{aux_ptr_base} + {out_offset}"
                            if arg_type == 'max':
                                ops_is_inner_last_is_outer_first.extend([
                                    f"tl.store({out_ptr}, macro_step_index, mask=mask)",
                                    f"tl.store({aux_ptr}, {val_var}, mask=mask)",
                                ])
                                ops_is_inner_last_not_is_outer_first.extend([
                                    f"{safe_var}_{op}_aux_old = tl.load({aux_ptr}, mask=mask, other={val_var})",
                                    f"{safe_var}_{op}_cond = {val_var} > {safe_var}_{op}_aux_old",
                                    f"tl.store({aux_ptr}, tl.where({safe_var}_{op}_cond, {val_var}, {safe_var}_{op}_aux_old), mask=mask)",
                                    f"tl.store({out_ptr}, macro_step_index, mask=mask & {safe_var}_{op}_cond)",
                                ])
                            else:  # min
                                ops_is_inner_last_is_outer_first.extend([
                                    f"tl.store({out_ptr}, macro_step_index, mask=mask)",
                                    f"tl.store({aux_ptr}, {val_var}, mask=mask)",
                                ])
                                ops_is_inner_last_not_is_outer_first.extend([
                                    f"{safe_var}_{op}_aux_old = tl.load({aux_ptr}, mask=mask, other={val_var})",
                                    f"{safe_var}_{op}_cond = {val_var} < {safe_var}_{op}_aux_old",
                                    f"tl.store({aux_ptr}, tl.where({safe_var}_{op}_cond, {val_var}, {safe_var}_{op}_aux_old), mask=mask)",
                                    f"tl.store({out_ptr}, macro_step_index, mask=mask & {safe_var}_{op}_cond)",
                                ])
                        else:
                            # ArgmaxK/ArgminK compound bubble insert
                            self._argmaxk_ops.append({
                                'var': safe_var, 'op': op, 'k': k_val, 'val_var': val_var,
                                'out_offset': out_offset, 'type': f'arg{arg_type}',
                                'has_val_output': False  # compound arg doesn't need val output
                            })
                    elif outer_base == 'max':
                        # Compound max without automatic arg (e.g., max_mean, max3_mean)
                        if k_val == 1:
                            ops_is_inner_last_is_outer_first.append(
                                f"tl.store({out_ptr}, {val_var}, mask=mask)")
                            ops_is_inner_last_not_is_outer_first.extend([
                                f"{safe_var}_{op}_old = tl.load({out_ptr}, mask=mask, other={val_var})",
                                f"tl.store({out_ptr}, tl.maximum({safe_var}_{op}_old, {val_var}), mask=mask)",
                            ])
                        else:
                            # maxK bubble insert without arg tracking
                            self._maxk_ops.append({
                                'var': safe_var, 'op': op, 'k': k_val, 'val_var': val_var,
                                'out_offset': out_offset, 'type': 'max'
                            })
                            
                    elif outer_base == 'min':
                        # Compound min without automatic arg (e.g., min_mean, min3_mean)
                        if k_val == 1:
                            ops_is_inner_last_is_outer_first.append(
                                f"tl.store({out_ptr}, {val_var}, mask=mask)")
                            ops_is_inner_last_not_is_outer_first.extend([
                                f"{safe_var}_{op}_old = tl.load({out_ptr}, mask=mask, other={val_var})",
                                f"tl.store({out_ptr}, tl.minimum({safe_var}_{op}_old, {val_var}), mask=mask)",
                            ])
                        else:
                            # minK bubble insert without arg tracking
                            self._maxk_ops.append({
                                'var': safe_var, 'op': op, 'k': k_val, 'val_var': val_var,
                                'out_offset': out_offset, 'type': 'min'
                            })
                            
                    elif outer == 'mean':
                        ops_is_inner_last_is_outer_first.append(
                            f"{safe_var}_{op}_accum = {val_var}")
                        ops_is_inner_last_not_is_outer_first.append(
                            f"{safe_var}_{op}_accum = tl.load({out_ptr}, mask=mask, other=0.0) + {val_var}")
                        ops_is_inner_last_is_outer_last.append(
                            f"tl.store({out_ptr}, {safe_var}_{op}_accum / num_macro_steps, mask=mask)")
                        ops_is_inner_last_not_is_outer_last.append(
                            f"tl.store({out_ptr}, {safe_var}_{op}_accum, mask=mask)")
                            
                    elif outer == 'sum':
                        ops_is_inner_last_is_outer_first.append(
                            f"tl.store({out_ptr}, {val_var}, mask=mask)")
                        ops_is_inner_last_not_is_outer_first.extend([
                            f"{safe_var}_{op}_old = tl.load({out_ptr}, mask=mask, other=0.0)",
                            f"tl.store({out_ptr}, {safe_var}_{op}_old + {val_var}, mask=mask)",
                        ])
                    elif outer == 'median':
                        # Compound median (e.g., median_mean, median_max)
                        # Track for deferred P-Square generation inside is_inner_last block
                        if not hasattr(self, '_median_compound_ops'):
                            self._median_compound_ops = {}
                        key = (safe_var, inner)
                        if key not in self._median_compound_ops:
                            self._median_compound_ops[key] = []
                        self._median_compound_ops[key].append((op, out_ptr, val_var, 2))  # marker index 2 = median
                    elif outer in ('q25', 'q75'):
                        # Compound quantile (e.g., q25_mean, q75_mean)
                        # Same P-Square state as median, different marker output
                        if not hasattr(self, '_median_compound_ops'):
                            self._median_compound_ops = {}
                        key = (safe_var, inner)
                        if key not in self._median_compound_ops:
                            self._median_compound_ops[key] = []
                        qi = 1 if outer == 'q25' else 3  # q1=25th, q3=75th
                        self._median_compound_ops[key].append((op, out_ptr, val_var, qi))
                    elif outer == 'last':
                        # Compound last (e.g., last_mean) — store the last inner value
                        # Simply overwrite on every is_inner_last step
                        ops_is_inner_last.append(f"tl.store({out_ptr}, {val_var}, mask=mask)")
                    elif outer == 'first':
                        # Compound first (e.g., first_mean) — store only at is_outer_first
                        ops_is_inner_last_is_outer_first.append(
                            f"tl.store({out_ptr}, {val_var}, mask=mask)")
                    continue
                
                # ===== Simple operations (non-compound) =====
                if op == 'mean':
                    inner_ops = set(o.split('_')[1] for o in ops if '_' in o)
                    if 'mean' in inner_ops:
                        # Reuse val_for_{safe_var}_mean from inner aggregation
                        ops_is_inner_last.append(f"tl.store({out_ptr}, val_for_{safe_var}_mean, mask=mask)")
                    else:
                        # Standalone mean - needs state (use variable-specific val)
                        ops_unconditional.extend([
                            f"# Standalone mean for {safe_var}",
                            f"{safe_var}_mean_old = tl.where(is_inner_first, tl.zeros_like({safe_var}_val), tl.load({out_ptr}, mask=mask, other=0.0))",
                            f"{safe_var}_mean_new = {safe_var}_mean_old + {safe_var}_val * weight",
                            f"{safe_var}_mean_out = tl.where(is_inner_last, {safe_var}_mean_new / total_weight, {safe_var}_mean_new)",
                        ])
                        ops_unconditional.append(f"tl.store({out_ptr}, {safe_var}_mean_out, mask=mask)")
                        
                elif op == 'sum':
                    ops_unconditional.extend([
                        f"{safe_var}_sum_old = tl.where(is_inner_first, tl.zeros_like({safe_var}_val), tl.load({out_ptr}, mask=mask, other=0.0))",
                        f"tl.store({out_ptr}, {safe_var}_sum_old + {safe_var}_val * weight, mask=mask)",
                    ])
                
                # ===== max/argmax MERGED handling =====
                elif op == 'max':
                    has_argmax = analysis['argmax']
                    merge_key = (var, 'max', 1)
                    if merge_key in processed_merged_ops:
                        continue  # Already processed as merged
                    processed_merged_ops.add(merge_key)
                    
                    if has_argmax:
                        # MERGED: max + argmax share the same comparison
                        aux_ptr = f"{safe_var}_max_aux_ptr + {out_offset}"
                        argmax_ptr = f"{safe_var}_argmax_ptr + {out_offset}"
                        ops_is_inner_first.extend([
                            f"# Merged max + argmax for {safe_var}",
                            f"tl.store({aux_ptr}, {safe_var}_val, mask=mask)",  # aux stores the max value
                            f"tl.store({out_ptr}, {safe_var}_val, mask=mask)",  # max output
                            f"tl.store({argmax_ptr}, macro_step_index, mask=mask)",  # argmax output
                        ])
                        ops_not_is_inner_first.extend([
                            f"{safe_var}_max_old = tl.load({aux_ptr}, mask=mask, other={safe_var}_val)",
                            f"{safe_var}_max_cond = {safe_var}_val > {safe_var}_max_old",
                            f"{safe_var}_max_new = tl.where({safe_var}_max_cond, {safe_var}_val, {safe_var}_max_old)",
                            f"tl.store({aux_ptr}, {safe_var}_max_new, mask=mask)",
                            f"tl.store({out_ptr}, {safe_var}_max_new, mask=mask)",
                            f"tl.store({argmax_ptr}, macro_step_index, mask=mask & {safe_var}_max_cond)",
                        ])
                    else:
                        # Simple max only (no argmax)
                        ops_is_inner_first.extend([
                            f"tl.store({out_ptr}, {safe_var}_val, mask=mask)",
                        ])
                        ops_not_is_inner_first.extend([
                            f"{safe_var}_max_old = tl.load({out_ptr}, mask=mask, other={safe_var}_val)",
                            f"tl.store({out_ptr}, tl.where({safe_var}_val > {safe_var}_max_old, {safe_var}_val, {safe_var}_max_old), mask=mask)",
                        ])
                
                elif op == 'argmax':
                    has_max = analysis['max']
                    merge_key = (var, 'max', 1)
                    if has_max:
                        # Will be handled by 'max' branch
                        continue
                    
                    # argmax only (no max)
                    aux_ptr = f"{safe_var}_max_aux_ptr + {out_offset}"
                    ops_is_inner_first.extend([
                        f"tl.store({out_ptr}, macro_step_index, mask=mask)",
                        f"tl.store({aux_ptr}, {safe_var}_val, mask=mask)",
                    ])
                    ops_not_is_inner_first.extend([
                        f"{safe_var}_argmax_aux_old = tl.load({aux_ptr}, mask=mask, other={safe_var}_val)",
                        f"{safe_var}_argmax_cond = {safe_var}_val > {safe_var}_argmax_aux_old",
                        f"tl.store({aux_ptr}, tl.where({safe_var}_argmax_cond, {safe_var}_val, {safe_var}_argmax_aux_old), mask=mask)",
                        f"tl.store({out_ptr}, macro_step_index, mask=mask & {safe_var}_argmax_cond)",
                    ])
                    
                # ===== min/argmin MERGED handling =====
                elif op == 'min':
                    has_argmin = analysis['argmin']
                    merge_key = (var, 'min', 1)
                    if merge_key in processed_merged_ops:
                        continue
                    processed_merged_ops.add(merge_key)
                    
                    if has_argmin:
                        # MERGED: min + argmin share the same comparison
                        aux_ptr = f"{safe_var}_min_aux_ptr + {out_offset}"
                        argmin_ptr = f"{safe_var}_argmin_ptr + {out_offset}"
                        ops_is_inner_first.extend([
                            f"# Merged min + argmin for {safe_var}",
                            f"tl.store({aux_ptr}, {safe_var}_val, mask=mask)",
                            f"tl.store({out_ptr}, {safe_var}_val, mask=mask)",
                            f"tl.store({argmin_ptr}, macro_step_index, mask=mask)",
                        ])
                        ops_not_is_inner_first.extend([
                            f"{safe_var}_min_old = tl.load({aux_ptr}, mask=mask, other={safe_var}_val)",
                            f"{safe_var}_min_cond = {safe_var}_val < {safe_var}_min_old",
                            f"{safe_var}_min_new = tl.where({safe_var}_min_cond, {safe_var}_val, {safe_var}_min_old)",
                            f"tl.store({aux_ptr}, {safe_var}_min_new, mask=mask)",
                            f"tl.store({out_ptr}, {safe_var}_min_new, mask=mask)",
                            f"tl.store({argmin_ptr}, macro_step_index, mask=mask & {safe_var}_min_cond)",
                        ])
                    else:
                        # Simple min only (no argmin)
                        ops_is_inner_first.extend([
                            f"tl.store({out_ptr}, {safe_var}_val, mask=mask)",
                        ])
                        ops_not_is_inner_first.extend([
                            f"{safe_var}_min_old = tl.load({out_ptr}, mask=mask, other={safe_var}_val)",
                            f"tl.store({out_ptr}, tl.where({safe_var}_val < {safe_var}_min_old, {safe_var}_val, {safe_var}_min_old), mask=mask)",
                        ])
                
                elif op == 'argmin':
                    has_min = analysis['min']
                    merge_key = (var, 'min', 1)
                    if has_min:
                        # Will be handled by 'min' branch
                        continue
                    
                    # argmin only (no min)
                    aux_ptr = f"{safe_var}_min_aux_ptr + {out_offset}"
                    ops_is_inner_first.extend([
                        f"tl.store({out_ptr}, macro_step_index, mask=mask)",
                        f"tl.store({aux_ptr}, {safe_var}_val, mask=mask)",
                    ])
                    ops_not_is_inner_first.extend([
                        f"{safe_var}_argmin_aux_old = tl.load({aux_ptr}, mask=mask, other={safe_var}_val)",
                        f"{safe_var}_argmin_cond = {safe_var}_val < {safe_var}_argmin_aux_old",
                        f"tl.store({aux_ptr}, tl.where({safe_var}_argmin_cond, {safe_var}_val, {safe_var}_argmin_aux_old), mask=mask)",
                        f"tl.store({out_ptr}, macro_step_index, mask=mask & {safe_var}_argmin_cond)",
                    ])
                
                # ===== maxK / argmaxK handling =====
                elif op.startswith('max') and re.match(r'^max(\d+)$', op):
                    match = re.match(r'^max(\d+)$', op)
                    k_val = int(match.group(1))
                    has_argmaxk = k_val in analysis['argmaxK']
                    merge_key = (var, 'max', k_val)
                    if merge_key in processed_merged_ops:
                        continue
                    processed_merged_ops.add(merge_key)
                    
                    if has_argmaxk:
                        # MERGED: maxK + argmaxK - store both val and idx in bubble insert
                        self._argmaxk_ops.append({
                            'var': safe_var, 'op': f'argmax{k_val}', 'k': k_val, 'val_var': f'{safe_var}_val',
                            'out_offset': out_offset, 'type': 'argmax',
                            'has_val_output': True,  # Also output max values
                            'val_output_ptr': f"{safe_var}_max{k_val}_ptr"
                        })
                    else:
                        # maxK only - simple bubble insert storing values
                        self._maxk_ops.append({
                            'var': safe_var, 'op': op, 'k': k_val, 'val_var': f'{safe_var}_val',
                            'out_offset': out_offset, 'type': 'max'
                        })
                
                elif op.startswith('argmax') and re.match(r'^argmax(\d+)$', op):
                    match = re.match(r'^argmax(\d+)$', op)
                    k_val = int(match.group(1))
                    has_maxk = k_val in analysis['maxK']
                    merge_key = (var, 'max', k_val)
                    if has_maxk:
                        # Will be handled by maxK branch
                        continue
                    
                    # argmaxK only - bubble insert with aux for values
                    self._argmaxk_ops.append({
                        'var': safe_var, 'op': op, 'k': k_val, 'val_var': f'{safe_var}_val',
                        'out_offset': out_offset, 'type': 'argmax',
                        'has_val_output': False
                    })
                
                # ===== minK / argminK handling =====
                elif op.startswith('min') and re.match(r'^min(\d+)$', op):
                    match = re.match(r'^min(\d+)$', op)
                    k_val = int(match.group(1))
                    has_argmink = k_val in analysis['argminK']
                    merge_key = (var, 'min', k_val)
                    if merge_key in processed_merged_ops:
                        continue
                    processed_merged_ops.add(merge_key)
                    
                    if has_argmink:
                        # MERGED: minK + argminK
                        self._argmaxk_ops.append({
                            'var': safe_var, 'op': f'argmin{k_val}', 'k': k_val, 'val_var': f'{safe_var}_val',
                            'out_offset': out_offset, 'type': 'argmin',
                            'has_val_output': True,
                            'val_output_ptr': f"{safe_var}_min{k_val}_ptr"
                        })
                    else:
                        # minK only
                        self._maxk_ops.append({
                            'var': safe_var, 'op': op, 'k': k_val, 'val_var': f'{safe_var}_val',
                            'out_offset': out_offset, 'type': 'min'
                        })
                
                elif op.startswith('argmin') and re.match(r'^argmin(\d+)$', op):
                    match = re.match(r'^argmin(\d+)$', op)
                    k_val = int(match.group(1))
                    has_mink = k_val in analysis['minK']
                    merge_key = (var, 'min', k_val)
                    if has_mink:
                        # Will be handled by minK branch
                        continue
                    
                    # argminK only
                    self._argmaxk_ops.append({
                        'var': safe_var, 'op': op, 'k': k_val, 'val_var': f'{safe_var}_val',
                        'out_offset': out_offset, 'type': 'argmin',
                        'has_val_output': False
                    })
                    
                elif op == 'last':
                    if var in vars_conditional_only:
                        # Check if this var also has compound ops with 'last' inner type
                        # If so, the deferred load inside is_inner_last block will already load the val
                        has_compound_last = any(
                            '_last' in other_op and other_op != 'last' 
                            for other_op in self._variable_ops[var]
                        )
                        if has_compound_last:
                            # Reuse the val loaded by deferred load (no duplicate load needed)
                            ops_is_inner_last.append(f"tl.store({out_ptr}, {safe_var}_val, mask=mask)")
                        else:
                            # Load inline
                            in_ptr_loc = f"{safe_var}_ptr + t * stride_input + idx"
                            ops_is_inner_last.extend([
                                f"{safe_var}_val = tl.load({in_ptr_loc}, mask=mask, other=0.0)",
                                f"tl.store({out_ptr}, {safe_var}_val, mask=mask)",
                            ])
                    else:
                        ops_is_inner_last.append(f"tl.store({out_ptr}, {safe_var}_val, mask=mask)")
                        
                elif op == 'first':
                    if var in vars_conditional_only:
                        has_compound_first = any(
                            '_first' in other_op and other_op != 'first'
                            for other_op in self._variable_ops[var]
                        )
                        if has_compound_first:
                            # Val will be loaded elsewhere (from unconditional or conditional load)
                            ops_is_inner_first.append(f"tl.store({out_ptr}, {safe_var}_val, mask=mask)")
                        else:
                            in_ptr_loc = f"{safe_var}_ptr + t * stride_input + idx"
                            ops_is_inner_first.extend([
                                f"{safe_var}_val = tl.load({in_ptr_loc}, mask=mask, other=0.0)",
                                f"tl.store({out_ptr}, {safe_var}_val, mask=mask)",
                            ])
                    else:
                        ops_is_inner_first.append(f"tl.store({out_ptr}, {safe_var}_val, mask=mask)")
                        
                elif op == 'mid':
                    if var in vars_conditional_only:
                        in_ptr_loc = f"{safe_var}_ptr + t * stride_input + idx"
                        ops_is_middle.extend([
                            f"{safe_var}_val = tl.load({in_ptr_loc}, mask=mask, other=0.0)",
                            f"tl.store({out_ptr}, {safe_var}_val, mask=mask)",
                        ])
                    else:
                        ops_is_middle.append(f"tl.store({out_ptr}, {safe_var}_val, mask=mask)")
                
                # ===== Median/Quantile operations =====
                elif op in ('median', 'q25', 'q75') or (op.startswith('q') and len(op) > 1 and op[1:].isdigit()):
                    # Outer median/quantile - P-Square algorithm applied to macro_step_index
                    # Track for deferred generation (all median/quantile ops for a var share state)
                    if not hasattr(self, '_median_outer_ops'):
                        self._median_outer_ops = {}
                    if safe_var not in self._median_outer_ops:
                        self._median_outer_ops[safe_var] = []
                    self._median_outer_ops[safe_var].append((op, out_ptr))
        
        # Phase 3: Emit loads for vars that need unconditional val
        for var in vars_need_val:
            emit_val(var, kernel_code_lines)
        
        # For conditional-only vars used in compound ops with 'last' inner type,
        # ensure the variable val is emitted (will be loaded inside is_inner_last block later)
        # We need to track them but NOT emit unconditional loads here.
        # The load will be emitted inside the is_inner_last block in Phase 6.
        
        # Check if 'val' is needed (for 2D processing or outer median ops)
        needs_val_alias = False
        if hasattr(self, '_median_outer_ops') and self._median_outer_ops:
            needs_val_alias = True
        # 'val' is also used in 2D variable processing
        # (that code path sets val independently, so we only need it for median here)
        if needs_val_alias and vars_need_val:
            first_var = next(iter(vars_need_val))
            safe_first = self._get_safe_name(first_var)
            kernel_code_lines.append(f"{indent}val = {safe_first}_val")
        
        # Phase 4: Emit inner aggregation state updates (per-variable)
        # Each variable gets its own inner aggregation state (val_for_{safe_var}_{inner_type})
        # For 'last' inner type, no state is needed - the value is simply the current variable value
        # used directly inside the `if is_inner_last:` block.
        for inner_type, inner_vars in inner_aggregations_needed.items():
            for var in inner_vars:
                safe_var = self._get_safe_name(var)
                out_offset = "t * n_saved_points + offs"
                val_for_var_inner = f"val_for_{safe_var}_{inner_type}"
                var_val = f"{safe_var}_val"
                
                if inner_type == 'last':
                    # 'last' is the simplest: val_for_X_last == X_val at is_inner_last.
                    # No state storage, no load/store needed.
                    pass
                elif inner_type == 'mean':
                    inner_ptr = f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    weight_ptr = f"{safe_var}_{inner_type}_weight_state_ptr + {out_offset}"
                    kernel_code_lines.append(f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})")
                    kernel_code_lines.extend([
                        f"{indent}{safe_var}_inner_{inner_type}_old = tl.load({inner_ptr}, mask=mask, other=0.0)",
                        f"{indent}{safe_var}_weight_{inner_type}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                        f"{indent}{safe_var}_inner_{inner_type}_new = {safe_var}_inner_{inner_type}_old + {var_val} * weight",
                        f"{indent}{safe_var}_weight_{inner_type}_new = {safe_var}_weight_{inner_type}_old + weight",
                    ])
                    # Store based on condition - use tl.where for efficiency
                    kernel_code_lines.extend([
                        f"{indent}tl.store({inner_ptr}, tl.where(is_inner_last, 0.0, {safe_var}_inner_{inner_type}_new), mask=mask)",
                        f"{indent}tl.store({weight_ptr}, tl.where(is_inner_last, 0.0, {safe_var}_weight_{inner_type}_new), mask=mask)",
                        f"{indent}{val_for_var_inner} = tl.where(is_inner_last, {safe_var}_inner_{inner_type}_new / {safe_var}_weight_{inner_type}_new, {val_for_var_inner})",
                    ])
                elif inner_type == 'sum':
                    inner_ptr = f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    kernel_code_lines.append(f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})")
                    kernel_code_lines.extend([
                        f"{indent}{safe_var}_inner_{inner_type}_old = tl.load({inner_ptr}, mask=mask, other=0.0)",
                        f"{indent}{safe_var}_inner_{inner_type}_new = {safe_var}_inner_{inner_type}_old + {var_val} * weight",
                        f"{indent}tl.store({inner_ptr}, tl.where(is_inner_last, 0.0, {safe_var}_inner_{inner_type}_new), mask=mask)",
                        f"{indent}{val_for_var_inner} = tl.where(is_inner_last, {safe_var}_inner_{inner_type}_new, {val_for_var_inner})",
                    ])
                elif inner_type == 'max':
                    inner_ptr = f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    kernel_code_lines.append(f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})")
                    kernel_code_lines.extend([
                        f"{indent}{safe_var}_inner_{inner_type}_old = tl.load({inner_ptr}, mask=mask, other={var_val})",
                        f"{indent}{safe_var}_inner_{inner_type}_new = tl.where(is_inner_first & (macro_step_index==0), {var_val}, tl.maximum({safe_var}_inner_{inner_type}_old, {var_val}))",
                        f"{indent}tl.store({inner_ptr}, tl.where(is_inner_last, -float('inf'), {safe_var}_inner_{inner_type}_new), mask=mask)",
                        f"{indent}{val_for_var_inner} = tl.where(is_inner_last, {safe_var}_inner_{inner_type}_new, {val_for_var_inner})",
                    ])
                elif inner_type == 'min':
                    inner_ptr = f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    kernel_code_lines.append(f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})")
                    kernel_code_lines.extend([
                        f"{indent}{safe_var}_inner_{inner_type}_old = tl.load({inner_ptr}, mask=mask, other={var_val})",
                        f"{indent}{safe_var}_inner_{inner_type}_new = tl.where(is_inner_first & (macro_step_index==0), {var_val}, tl.minimum({safe_var}_inner_{inner_type}_old, {var_val}))",
                        f"{indent}tl.store({inner_ptr}, tl.where(is_inner_last, float('inf'), {safe_var}_inner_{inner_type}_new), mask=mask)",
                        f"{indent}{val_for_var_inner} = tl.where(is_inner_last, {safe_var}_inner_{inner_type}_new, {val_for_var_inner})",
                    ])
                elif inner_type == 'first':
                    # 'first' inner: store the value at is_inner_first, read it back at is_inner_last
                    inner_ptr = f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    kernel_code_lines.append(f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})")
                    kernel_code_lines.extend([
                        f"{indent}tl.store({inner_ptr}, {var_val}, mask=mask & is_inner_first)",
                        f"{indent}{val_for_var_inner} = tl.where(is_inner_last, tl.load({inner_ptr}, mask=mask, other=0.0), {val_for_var_inner})",
                    ])
                elif inner_type == 'mid':
                    inner_ptr = f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    kernel_code_lines.append(f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})")
                    kernel_code_lines.extend([
                        f"{indent}tl.store({inner_ptr}, {var_val}, mask=mask & is_middle)",
                        f"{indent}{val_for_var_inner} = tl.where(is_inner_last, tl.load({inner_ptr}, mask=mask, other=0.0), {val_for_var_inner})",
                    ])
                elif inner_type == 'median':
                    # Inner P-Square median - use step_count_val as step index
                    # Track for deferred generation (all inner median ops share state)
                    if not hasattr(self, '_median_inner_ops'):
                        self._median_inner_ops = {}
                    self._median_inner_ops[safe_var] = out_offset
                # Note: removed 'break' - now generate for each variable in inner_vars
        
        # Phase 5: Emit unconditional ops
        for line in ops_unconditional:
            kernel_code_lines.append(f"{indent}{line}")
        
        # Phase 6: Emit grouped conditional blocks
        if ops_is_inner_first:
            kernel_code_lines.append(f"{indent}if is_inner_first:")
            for line in ops_is_inner_first:
                kernel_code_lines.append(f"{indent2}{line}")
                
        if ops_not_is_inner_first:
            if ops_is_inner_first:
                kernel_code_lines.append(f"{indent}else:")
            else:
                kernel_code_lines.append(f"{indent}if not is_inner_first:")
            for line in ops_not_is_inner_first:
                kernel_code_lines.append(f"{indent2}{line}")
        
        if ops_is_middle:
            kernel_code_lines.append(f"{indent}if is_middle:")
            for line in ops_is_middle:
                kernel_code_lines.append(f"{indent2}{line}")
        
        # Phase 6.5: Emit inner median P-Square operations BEFORE is_inner_last block
        # (because compound ops like max_median reference val_for_x_median inside is_inner_last)
        has_median_inner = hasattr(self, '_median_inner_ops') and self._median_inner_ops
        if has_median_inner:
            kernel_code_lines.append(f"{indent}# Inner Median P-Square Algorithm")
            kernel_code_lines.append(f"{indent}inner_step = step_count_val.to(tl.int32)")
            for safe_var, out_offset in self._median_inner_ops.items():
                var_val = f"{safe_var}_val"
                val_for_var_median = f"val_for_{safe_var}_median"
                kernel_code_lines.append(f"{indent}{val_for_var_median} = tl.zeros_like({var_val})")
                q_ptr = f"{safe_var}_median_inner_q_state_ptr"
                n_ptr = f"{safe_var}_median_inner_n_state_ptr"
                stride_k = "n_saved_points"
                offset_expr = "offs"
                
                self._generate_psquare_code(
                    kernel_code_lines, safe_var, q_ptr, n_ptr,
                    var_val, "inner_step", None, stride_k, offset_expr,
                    indent, indent2, indent3
                )
                # Extract median on is_inner_last
                kernel_code_lines.extend([
                    f"{indent}if is_inner_last:",
                    f"{indent2}{val_for_var_median} = tl.load({q_ptr} + 2 * {stride_k} + {offset_expr}, mask=mask, other=0.0)",
                ])
            self._median_inner_ops = {}  # Reset
        
        # Nested conditions for is_inner_last with outer conditions
        has_argmaxk_ops = hasattr(self, '_argmaxk_ops') and self._argmaxk_ops
        has_inner_last_ops = (ops_is_inner_last or ops_is_inner_last_is_outer_first or 
                             ops_is_inner_last_not_is_outer_first or ops_is_inner_last_is_outer_last or
                             ops_is_inner_last_not_is_outer_last or self._maxk_ops or has_argmaxk_ops)
        
        if has_inner_last_ops:
            kernel_code_lines.append(f"{indent}if is_inner_last:")
            
            # Emit deferred loads for conditional-only vars used in compound ops
            # These vars are only needed inside is_inner_last, so we load them here
            for var in dims_1d:
                if var in vars_conditional_only and var in inner_aggregations_needed.get('last', set()):
                    emit_val(var, kernel_code_lines)
                    # Patch the emitted line to use is_inner_last indentation
                    # The emit_val appends to kernel_code_lines with 'indent' (8 spaces)
                    # We need it at indent2 (12 spaces) since we're inside 'if is_inner_last:'
                    if kernel_code_lines and kernel_code_lines[-1].startswith(indent) and not kernel_code_lines[-1].startswith(indent2):
                        last_line = kernel_code_lines.pop()
                        kernel_code_lines.append(f"{indent2}{last_line.lstrip()}")
            
            # is_outer_first / not is_outer_first
            if ops_is_inner_last_is_outer_first or ops_is_inner_last_not_is_outer_first:
                kernel_code_lines.append(f"{indent2}if is_outer_first:")
                for line in ops_is_inner_last_is_outer_first:
                    kernel_code_lines.append(f"{indent3}{line}")
                if ops_is_inner_last_not_is_outer_first:
                    kernel_code_lines.append(f"{indent2}else:")
                    for line in ops_is_inner_last_not_is_outer_first:
                        kernel_code_lines.append(f"{indent3}{line}")
            
            # is_outer_last / not is_outer_last (for mean finalization)
            if ops_is_inner_last_is_outer_last or ops_is_inner_last_not_is_outer_last:
                kernel_code_lines.append(f"{indent2}if is_outer_last:")
                for line in ops_is_inner_last_is_outer_last:
                    kernel_code_lines.append(f"{indent3}{line}")
                if ops_is_inner_last_not_is_outer_last:
                    kernel_code_lines.append(f"{indent2}else:")
                    for line in ops_is_inner_last_not_is_outer_last:
                        kernel_code_lines.append(f"{indent3}{line}")
            
            # Simple is_inner_last ops
            for line in ops_is_inner_last:
                kernel_code_lines.append(f"{indent2}{line}")
            
            # ================================================================
            # Optimized MaxK/MinK + ArgmaxK/ArgminK bubble insert operations
            
            # Group by (var, k, out_offset) to share base offset across all operations
            from collections import defaultdict
            grouped_by_var_k = defaultdict(lambda: {'max': None, 'min': None, 'argmax': None, 'argmin': None})
            
            for maxk_op in self._maxk_ops:
                key = (maxk_op['var'], maxk_op['k'], maxk_op['out_offset'])
                grouped_by_var_k[key][maxk_op['type']] = maxk_op
            
            if hasattr(self, '_argmaxk_ops') and self._argmaxk_ops:
                for argk_op in self._argmaxk_ops:
                    key = (argk_op['var'], argk_op['k'], argk_op['out_offset'])
                    op_type = 'argmax' if 'max' in argk_op['type'] else 'argmin'
                    grouped_by_var_k[key][op_type] = argk_op
            
            # Process grouped operations with shared offset
            for (safe_var, k_val, out_offset), ops_dict in grouped_by_var_k.items():
                has_max = ops_dict['max'] is not None
                has_min = ops_dict['min'] is not None
                has_argmax = ops_dict['argmax'] is not None
                has_argmin = ops_dict['argmin'] is not None
                
                # Get val_var from each operation (may differ: val for max/min, val_for_mean for argmax/argmin)
                max_val_var = ops_dict['max']['val_var'] if has_max else None
                min_val_var = ops_dict['min']['val_var'] if has_min else None
                argmax_val_var = ops_dict['argmax']['val_var'] if has_argmax else None
                argmin_val_var = ops_dict['argmin']['val_var'] if has_argmin else None
                
                # Compute shared base offset once
                out_offset_k = f"({out_offset}) * {k_val}"
                
                # Generate header comment
                op_names = []
                if has_max: op_names.append(f"max{k_val}")
                if has_min: op_names.append(f"min{k_val}")
                if has_argmax: op_names.append(f"argmax{k_val}")
                if has_argmin: op_names.append(f"argmin{k_val}")
                kernel_code_lines.append(f"{indent2}# Merged Bubble Insert [{'+'.join(op_names)}] for {safe_var} (shared offset, precise mask)")
                
                # Shared base offset computation
                kernel_code_lines.append(f"{indent2}{safe_var}_k{k_val}_base_offs = {out_offset_k}")
                
                # Initialize new values for bubble insert (using correct val_var for each op type)
                if has_max:
                    kernel_code_lines.append(f"{indent2}new_val_max_{safe_var} = {max_val_var}")
                if has_min:
                    kernel_code_lines.append(f"{indent2}new_val_min_{safe_var} = {min_val_var}")
                if has_argmax:
                    kernel_code_lines.append(f"{indent2}new_val_argmax_{safe_var} = {argmax_val_var}")
                if has_argmin:
                    kernel_code_lines.append(f"{indent2}new_val_argmin_{safe_var} = {argmin_val_var}")
                if has_argmax or has_argmin:
                    kernel_code_lines.append(f"{indent2}new_idx_{safe_var} = tl.full([BLOCK_SIZE], macro_step_index, dtype=tl.int32)")
                
                # is_outer_first branch: initialize all arrays
                kernel_code_lines.append(f"{indent2}if is_outer_first:")
                
                # First position stores the initial value
                if has_max:
                    max_ptr = f"{safe_var}_{ops_dict['max']['op']}_ptr"
                    kernel_code_lines.append(f"{indent3}tl.store({max_ptr} + {safe_var}_k{k_val}_base_offs, new_val_max_{safe_var}, mask=mask)")
                if has_min:
                    min_ptr = f"{safe_var}_{ops_dict['min']['op']}_ptr"
                    kernel_code_lines.append(f"{indent3}tl.store({min_ptr} + {safe_var}_k{k_val}_base_offs, new_val_min_{safe_var}, mask=mask)")
                if has_argmax:
                    argmax_op = ops_dict['argmax']
                    argmax_aux_ptr = f"{safe_var}_max{k_val}_aux_ptr"
                    argmax_idx_ptr = f"{safe_var}_{argmax_op['op']}_ptr"
                    kernel_code_lines.append(f"{indent3}tl.store({argmax_idx_ptr} + {safe_var}_k{k_val}_base_offs, new_idx_{safe_var}, mask=mask)")
                    kernel_code_lines.append(f"{indent3}tl.store({argmax_aux_ptr} + {safe_var}_k{k_val}_base_offs, new_val_argmax_{safe_var}, mask=mask)")
                    if argmax_op.get('has_val_output') and argmax_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent3}tl.store({argmax_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs, new_val_argmax_{safe_var}, mask=mask)")
                if has_argmin:
                    argmin_op = ops_dict['argmin']
                    argmin_aux_ptr = f"{safe_var}_min{k_val}_aux_ptr"
                    argmin_idx_ptr = f"{safe_var}_{argmin_op['op']}_ptr"
                    kernel_code_lines.append(f"{indent3}tl.store({argmin_idx_ptr} + {safe_var}_k{k_val}_base_offs, new_idx_{safe_var}, mask=mask)")
                    kernel_code_lines.append(f"{indent3}tl.store({argmin_aux_ptr} + {safe_var}_k{k_val}_base_offs, new_val_argmin_{safe_var}, mask=mask)")
                    if argmin_op.get('has_val_output') and argmin_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent3}tl.store({argmin_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs, new_val_argmin_{safe_var}, mask=mask)")
                
                # Initialize remaining positions with inf/-inf
                kernel_code_lines.append(f"{indent3}for k in tl.static_range(1, {k_val}):")
                if has_max:
                    kernel_code_lines.append(f"{indent4}tl.store({max_ptr} + {safe_var}_k{k_val}_base_offs + k, -float('inf'), mask=mask)")
                if has_min:
                    kernel_code_lines.append(f"{indent4}tl.store({min_ptr} + {safe_var}_k{k_val}_base_offs + k, float('inf'), mask=mask)")
                if has_argmax:
                    kernel_code_lines.append(f"{indent4}tl.store({argmax_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, 0, mask=mask)")
                    kernel_code_lines.append(f"{indent4}tl.store({argmax_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, -float('inf'), mask=mask)")
                    if argmax_op.get('has_val_output') and argmax_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent4}tl.store({argmax_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs + k, -float('inf'), mask=mask)")
                if has_argmin:
                    kernel_code_lines.append(f"{indent4}tl.store({argmin_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, 0, mask=mask)")
                    kernel_code_lines.append(f"{indent4}tl.store({argmin_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, float('inf'), mask=mask)")
                    if argmin_op.get('has_val_output') and argmin_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent4}tl.store({argmin_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs + k, float('inf'), mask=mask)")
                
                # else branch: bubble insert
                kernel_code_lines.append(f"{indent2}else:")
                kernel_code_lines.append(f"{indent3}for k in tl.static_range({k_val}):")
                
                # Load old values and compute swap masks
                if has_max:
                    kernel_code_lines.extend([
                        f"{indent4}old_max_k = tl.load({max_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=-float('inf'))",
                        f"{indent4}swap_max = new_val_max_{safe_var} > old_max_k",
                        f"{indent4}max_to_store = tl.where(swap_max, new_val_max_{safe_var}, old_max_k)",
                        f"{indent4}new_val_max_{safe_var} = tl.where(swap_max, old_max_k, new_val_max_{safe_var})",
                        f"{indent4}tl.store({max_ptr} + {safe_var}_k{k_val}_base_offs + k, max_to_store, mask=mask & swap_max)",
                    ])
                if has_min:
                    kernel_code_lines.extend([
                        f"{indent4}old_min_k = tl.load({min_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=float('inf'))",
                        f"{indent4}swap_min = new_val_min_{safe_var} < old_min_k",
                        f"{indent4}min_to_store = tl.where(swap_min, new_val_min_{safe_var}, old_min_k)",
                        f"{indent4}new_val_min_{safe_var} = tl.where(swap_min, old_min_k, new_val_min_{safe_var})",
                        f"{indent4}tl.store({min_ptr} + {safe_var}_k{k_val}_base_offs + k, min_to_store, mask=mask & swap_min)",
                    ])
                if has_argmax:
                    kernel_code_lines.extend([
                        f"{indent4}old_argmax_aux_k = tl.load({argmax_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=-float('inf'))",
                        f"{indent4}old_argmax_idx_k = tl.load({argmax_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=0)",
                        f"{indent4}swap_argmax = new_val_argmax_{safe_var} > old_argmax_aux_k",
                        f"{indent4}argmax_aux_store = tl.where(swap_argmax, new_val_argmax_{safe_var}, old_argmax_aux_k)",
                        f"{indent4}argmax_idx_store = tl.where(swap_argmax, new_idx_{safe_var}, old_argmax_idx_k)",
                        f"{indent4}new_val_argmax_{safe_var} = tl.where(swap_argmax, old_argmax_aux_k, new_val_argmax_{safe_var})",
                        f"{indent4}new_idx_{safe_var} = tl.where(swap_argmax, old_argmax_idx_k, new_idx_{safe_var})",
                        f"{indent4}tl.store({argmax_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, argmax_aux_store, mask=mask & swap_argmax)",
                        f"{indent4}tl.store({argmax_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, argmax_idx_store, mask=mask & swap_argmax)",
                    ])
                    if argmax_op.get('has_val_output') and argmax_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent4}tl.store({argmax_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs + k, argmax_aux_store, mask=mask & swap_argmax)")
                if has_argmin:
                    kernel_code_lines.extend([
                        f"{indent4}old_argmin_aux_k = tl.load({argmin_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=float('inf'))",
                        f"{indent4}old_argmin_idx_k = tl.load({argmin_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=0)",
                        f"{indent4}swap_argmin = new_val_argmin_{safe_var} < old_argmin_aux_k",
                        f"{indent4}argmin_aux_store = tl.where(swap_argmin, new_val_argmin_{safe_var}, old_argmin_aux_k)",
                        f"{indent4}argmin_idx_store = tl.where(swap_argmin, new_idx_{safe_var}, old_argmin_idx_k)",
                        f"{indent4}new_val_argmin_{safe_var} = tl.where(swap_argmin, old_argmin_aux_k, new_val_argmin_{safe_var})",
                        f"{indent4}new_idx_{safe_var} = tl.where(swap_argmin, old_argmin_idx_k, new_idx_{safe_var})",
                        f"{indent4}tl.store({argmin_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, argmin_aux_store, mask=mask & swap_argmin)",
                        f"{indent4}tl.store({argmin_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, argmin_idx_store, mask=mask & swap_argmin)",
                    ])
                    if argmin_op.get('has_val_output') and argmin_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent4}tl.store({argmin_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs + k, argmin_aux_store, mask=mask & swap_argmin)")
            
            # Reset operation lists
            self._maxk_ops = []
            if hasattr(self, '_argmaxk_ops'):
                self._argmaxk_ops = []
        
        # Phase 8: Emit compound median/quantile P-Square operations (inside is_inner_last)
        has_median_compound = hasattr(self, '_median_compound_ops') and self._median_compound_ops
        if has_median_compound:
            # Ensure we're inside is_inner_last
            if not has_inner_last_ops:
                kernel_code_lines.append(f"{indent}if is_inner_last:")
            kernel_code_lines.append(f"{indent2}# Compound Median/Quantile P-Square Algorithm")
            for (safe_var, inner), ops_list in self._median_compound_ops.items():
                q_ptr = f"{safe_var}_median_{inner}_q_state_ptr"
                n_ptr = f"{safe_var}_median_{inner}_n_state_ptr"
                stride_k = "(num_trials * n_saved_points)"
                offset_expr = "(t * n_saved_points + offs)"
                val_var = ops_list[0][2]  # val_for_{var}_{inner}
                
                # Build out_ptrs dict: {marker_index: out_ptr}
                out_ptrs = {}
                for (op, out_ptr, _, qi) in ops_list:
                    out_ptrs[qi] = out_ptr
                
                self._generate_psquare_code(
                    kernel_code_lines, safe_var, q_ptr, n_ptr,
                    val_var, "macro_step_index", out_ptrs, stride_k, offset_expr,
                    indent2, indent3, indent4
                )
            self._median_compound_ops = {}
        
        # Phase 9: Emit outer median/quantile P-Square operations (standalone)
        if hasattr(self, '_median_outer_ops') and self._median_outer_ops:
            kernel_code_lines.append(f"{indent}# Outer Median/Quantile P-Square Algorithm")
            for safe_var, ops_list in self._median_outer_ops.items():
                q_ptr = f"{safe_var}_median_q_state_ptr"
                n_ptr = f"{safe_var}_median_n_state_ptr"
                stride_k = "(num_trials * n_saved_points)"
                offset_expr = "(t * n_saved_points + offs)"
                
                # Build out_ptrs dict: {marker_index: out_ptr}
                out_ptrs = {}
                for (op, out_ptr) in ops_list:
                    # Determine marker index from op name
                    if op.startswith('q25'):
                        out_ptrs[1] = out_ptr
                    elif op.startswith('q75'):
                        out_ptrs[3] = out_ptr
                    else:  # 'median' or default
                        out_ptrs[2] = out_ptr
                
                self._generate_psquare_code(
                    kernel_code_lines, safe_var, q_ptr, n_ptr,
                    "val", "macro_step_index", out_ptrs, stride_k, offset_expr,
                    indent, indent2, indent3
                )
            self._median_outer_ops = {}  # Reset
        
        kernel_code_lines.append("")

    def _generate_kernel_for_group(self, kernel_code_lines: List[str], kernel_name: str,
                                   save_idx: str, var_list: List[str],
                                   tensor_info: Dict[str, Dict[str, Any]]) -> None:
        """Generate kernel code for a specific save_idx group supporting ops."""
        if self.num_trials > 1:
            dims_1d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 2]
            dims_2d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 3]
        else:
            dims_1d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 1]
            dims_2d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 2]

        # Header
        kernel_code_lines.extend([
            f"# Kernel for save_idx: {save_idx}",
            f"# Variables: {', '.join(var_list)}",
            f"# 1D: {', '.join(dims_1d) if dims_1d else 'None'}",
            f"# 2D: {', '.join(dims_2d) if dims_2d else 'None'}",
            "",
            "@triton.jit",
            f"def {kernel_name}(",
            f"    {save_idx}_ptr,",
        ])

        # Gather input pointers (resolving virtuals)
        input_ptrs = set()
        def _gather_inputs(name):
             info = self._field_registry.get(name)
             if getattr(info, 'json_schema_extra', {}).get('category') == 'virtual':
                  expr = getattr(info, 'json_schema_extra', {}).get('expr', '')
                  toks = re.findall(r'\b[a-zA-Z_]\w*\b', expr)
                  for t in toks:
                       if t in self._field_registry or t in self._tensor_registry:
                            _gather_inputs(t)
             else:
                  input_ptrs.add(name)
        
        for var in var_list:
             _gather_inputs(var)

        # Pointers
        # Inputs
        sorted_inputs = sorted(list(input_ptrs))
        for var in sorted_inputs:
            safe_var = self._get_safe_name(var)
            # Avoid duplicate argument if save_idx matches input var
            if safe_var == save_idx:
                continue
            kernel_code_lines.append(f"    {safe_var}_ptr,")

        for var in var_list:
            safe_var = self._get_safe_name(var)
            # Track which extra state pointers have been added to avoid duplicates
            added_median_state = False
            added_aux_ptrs = set()  # Track aux pointers already added (for explicit argmax/argmin)
            
            for op in self._variable_ops[var]:
                kernel_code_lines.append(f"    {safe_var}_{op}_ptr,")
                
                # For EXPLICIT argmax/argmin operators, add aux pointer for tracking values
                # NO automatic argmax/argmin generation for max/min operations
                op_parts = op.split('_')
                outer_op = op_parts[0]
                
                # Check for explicit argmax/argmin (e.g., argmax, argmax3, argmin, argmin3)
                arg_match = re.match(r'arg(max|min)(\d*)$', outer_op)
                if arg_match:
                    arg_type = arg_match.group(1)  # 'max' or 'min'
                    arg_k_str = arg_match.group(2)  # '' or '3' etc
                    # aux pointer name: {safe_var}_{arg_type}{k}_aux_ptr (e.g., var_max_aux_ptr, var_max3_aux_ptr)
                    aux_name = f"{arg_type}{arg_k_str}_aux"  # e.g., 'max_aux', 'max3_aux'
                    if aux_name not in added_aux_ptrs:
                        kernel_code_lines.append(f"    {safe_var}_{aux_name}_ptr,")
                        added_aux_ptrs.add(aux_name)
                        
                if (op.startswith('median') or re.match(r'^q\d+', op)):
                     if '_' in op:
                         # Compound median/quantile: per-inner state
                         inner = op.split('_')[1]
                         cmp_key = f"median_{inner}"
                         if cmp_key not in added_aux_ptrs:
                             kernel_code_lines.append(f"    {safe_var}_median_{inner}_q_state_ptr,")
                             kernel_code_lines.append(f"    {safe_var}_median_{inner}_n_state_ptr,")
                             added_aux_ptrs.add(cmp_key)
                     elif not added_median_state:
                         kernel_code_lines.append(f"    {safe_var}_median_q_state_ptr,")
                         kernel_code_lines.append(f"    {safe_var}_median_n_state_ptr,")
                         added_median_state = True
            
            # Inner state pointers (only for ops that need cross-step state)
            added_inner = set()
            for op in self._variable_ops[var]:
                if '_' in op:
                    inner = op.split('_')[1]
                    if inner not in added_inner:
                        # 'last' inner op directly uses current value, no state needed
                        # 'median' inner op uses its own q/n state, not generic inner_state
                        if inner not in ('last', 'median'):
                            kernel_code_lines.append(f"    {safe_var}_{inner}_inner_state_ptr,")
                        if inner == 'mean':
                            kernel_code_lines.append(f"    {safe_var}_{inner}_weight_state_ptr,")
                        if inner == 'median':
                            kernel_code_lines.append(f"    {safe_var}_median_inner_q_state_ptr,")
                            kernel_code_lines.append(f"    {safe_var}_median_inner_n_state_ptr,")
                        added_inner.add(inner)

        kernel_code_lines.extend([
            "    weight,",
            "    total_weight,",
            "    num_macro_steps,",
            "    is_inner_first,",
            "    is_inner_last,",
            "    is_middle,",
            "    is_outer_first,",
            "    is_outer_last,",
            "    macro_step_index,",
            "    step_count_val,",
            "    n_saved_points: tl.constexpr,",
        ])
        if dims_2d:
            kernel_code_lines.append("    n_levels: tl.constexpr,")
        kernel_code_lines.extend([
            "    BLOCK_SIZE: tl.constexpr,",
            "    num_trials: tl.constexpr,",
            "    stride_input: tl.constexpr,",
            "):",
            "    pid = tl.program_id(0)",
            "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
            "    mask = offs < n_saved_points",
            "",
            f"    idx = tl.load({save_idx}_ptr + offs, mask=mask)",
            "",
        ])

        # Loop over trials - use tl.static_range for compile-time unrolling
        kernel_code_lines.append("    for t in tl.static_range(num_trials):")
        indent = "        "
        indent2 = indent + "    "
        indent3 = indent2 + "    "
        indent4 = indent3 + "    "
        indent5 = indent4 + "    "

        # 1D processing - use grouped generation for all vars (including median)
        if dims_1d:
            self._generate_1d_vars_grouped(kernel_code_lines, dims_1d, 
                                           indent, indent2, indent3, indent4, indent5)

        # 2D processing
        if dims_2d:
            non_last_only = [v for v in dims_2d if not (len(self._variable_ops[v]) == 1 and self._variable_ops[v][0] == 'last')]
            last_only_vars = [v for v in dims_2d if (len(self._variable_ops[v]) == 1 and self._variable_ops[v][0] == 'last')]

            if non_last_only:
                kernel_code_lines.extend([
                    f"{indent}# 2D variables (mean/min/max and mixed)",
                    f"{indent}for level in tl.static_range(n_levels):",
                ])
                emitted_vars_2d = set()
                def emit_val_2d(v_name):
                    safe_v_name = self._get_safe_name(v_name)
                    if safe_v_name in emitted_vars_2d: return f"{safe_v_name}_val"
                    
                    info = self._field_registry.get(v_name)
                    cat = getattr(info, 'json_schema_extra', {}).get('category', 'param') if info else 'param'
                    
                    if cat == 'virtual' and info:
                         expr = getattr(info, 'json_schema_extra', {}).get('expr')
                         import re
                         safe_expr = expr
                         toks = set(re.findall(r'\b[a-zA-Z_]\w*\b', expr))
                         for t in toks:
                              if t in self._field_registry or t in self._tensor_registry:
                                   emit_val_2d(t)
                                   safe_t = self._get_safe_name(t)
                                   safe_expr = re.sub(r'\b' + t + r'\b', f"{safe_t}_val", safe_expr)
                         safe_expr = self._transform_pow_expr(safe_expr)
                         kernel_code_lines.append(f"{indent2}{safe_v_name}_val = {safe_expr}")
                    else:
                         in_ptr_loc = f"{safe_v_name}_ptr + (t * stride_input + idx) * n_levels + level"
                         kernel_code_lines.append(f"{indent2}{safe_v_name}_val = tl.load({in_ptr_loc}, mask=mask, other=0.0)")
                    
                    emitted_vars_2d.add(safe_v_name)
                    return f"{safe_v_name}_val"

                for var in non_last_only:
                    safe_var = self._get_safe_name(var)
                    out_offset = f"(t * n_saved_points + offs) * n_levels + level"
                    
                    val_name = emit_val_2d(var)
                    kernel_code_lines.append(f"{indent2}val = {val_name}")

                    # Inner states update
                    ops = self._variable_ops[var]
                    inner_ops = set(op.split('_')[1] for op in ops if '_' in op)
                    for inner in inner_ops:
                        # Initialize val_for_inner to avoid UnboundLocalError/NameError in generated code
                        # This value is used if is_update_outer is True, where it gets overwritten.
                        kernel_code_lines.append(f"{indent2}val_for_{inner} = tl.zeros_like(val)")

                        inner_ptr = f"{safe_var}_{inner}_inner_state_ptr + {out_offset}"
                        if inner == 'mean':
                             weight_ptr = f"{safe_var}_{inner}_weight_state_ptr + {out_offset}"
                             kernel_code_lines.extend([
                                 f"{indent2}inner_{inner}_old = tl.load({inner_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}weight_{inner}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}inner_{inner}_new = inner_{inner}_old + val * weight",
                                 f"{indent2}weight_{inner}_new = weight_{inner}_old + weight",
                                 f"{indent2}if is_macro_step_end:",
                                 f"{indent3}tl.store({inner_ptr}, 0.0, mask=mask)",
                                 f"{indent3}tl.store({weight_ptr}, 0.0, mask=mask)",
                                 f"{indent3}val_for_{inner} = inner_{inner}_new / (weight_{inner}_new)",
                                 f"{indent2}else:",
                                 f"{indent3}tl.store({inner_ptr}, inner_{inner}_new, mask=mask)",
                                 f"{indent3}tl.store({weight_ptr}, weight_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'sum':
                             kernel_code_lines.extend([
                                 f"{indent2}inner_{inner}_old = tl.load({inner_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}inner_{inner}_new = inner_{inner}_old + val * weight",
                                 f"{indent2}if is_macro_step_end:",
                                 f"{indent3}tl.store({inner_ptr}, 0.0, mask=mask)",
                                 f"{indent3}val_for_{inner} = inner_{inner}_new",
                                 f"{indent2}else:",
                                 f"{indent3}tl.store({inner_ptr}, inner_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'max':
                             weight_ptr = f"{safe_var}_{inner}_weight_state_ptr + {out_offset}"
                             kernel_code_lines.extend([
                                 f"{indent2}weight_{inner}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}weight_{inner}_new = weight_{inner}_old + weight",
                                 f"{indent2}if is_first and macro_step_index==0:", 
                                 f"{indent3}inner_{inner}_new = val",
                                 f"{indent2}else:",
                                 f"{indent3}inner_{inner}_old = tl.load({inner_ptr}, mask=mask, other=val)",
                                 f"{indent3}inner_{inner}_new = tl.maximum(inner_{inner}_old, val)",
                                 f"{indent2}if is_macro_step_end:",
                                 f"{indent3}tl.store({inner_ptr}, -float('inf'), mask=mask)",
                                 f"{indent3}val_for_{inner} = inner_{inner}_new",
                                 f"{indent3}val_weight_for_{inner} = weight_{inner}_new",
                                 f"{indent3}tl.store({weight_ptr}, 0.0, mask=mask)",
                                 f"{indent2}else:",
                                 f"{indent3}tl.store({inner_ptr}, inner_{inner}_new, mask=mask)",
                                 f"{indent3}tl.store({weight_ptr}, weight_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'min':
                             weight_ptr = f"{safe_var}_{inner}_weight_state_ptr + {out_offset}"
                             kernel_code_lines.extend([
                                 f"{indent2}weight_{inner}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}weight_{inner}_new = weight_{inner}_old + weight",
                                 f"{indent2}if is_first and macro_step_index==0:",
                                 f"{indent3}inner_{inner}_new = val",
                                 f"{indent2}else:",
                                 f"{indent3}inner_{inner}_old = tl.load({inner_ptr}, mask=mask, other=val)",
                                 f"{indent3}inner_{inner}_new = tl.minimum(inner_{inner}_old, val)",
                                 f"{indent2}if is_macro_step_end:",
                                 f"{indent3}tl.store({inner_ptr}, float('inf'), mask=mask)",
                                 f"{indent3}val_for_{inner} = inner_{inner}_new",
                                 f"{indent3}val_weight_for_{inner} = weight_{inner}_new",
                                 f"{indent3}tl.store({weight_ptr}, 0.0, mask=mask)",
                                 f"{indent2}else:",
                                 f"{indent3}tl.store({inner_ptr}, inner_{inner}_new, mask=mask)",
                                 f"{indent3}tl.store({weight_ptr}, weight_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'first':
                             weight_ptr = f"{safe_var}_{inner}_weight_state_ptr + {out_offset}"
                             kernel_code_lines.extend([
                                 f"{indent2}weight_{inner}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}weight_{inner}_new = weight_{inner}_old + weight",
                                 f"{indent2}val_stored = tl.load({inner_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}if weight_{inner}_old == 0.0:",
                                 f"{indent3}inner_{inner}_new = val",
                                 f"{indent2}else:",
                                 f"{indent3}inner_{inner}_new = val_stored",
                                 f"{indent2}if is_macro_step_end:",
                                 f"{indent3}val_for_{inner} = inner_{inner}_new",
                                 f"{indent3}val_weight_for_{inner} = weight_{inner}_new",
                                 f"{indent3}tl.store({weight_ptr}, 0.0, mask=mask)",
                                 f"{indent3}tl.store({inner_ptr}, 0.0, mask=mask)",
                                 f"{indent2}else:",
                                 f"{indent3}tl.store({inner_ptr}, inner_{inner}_new, mask=mask)",
                                 f"{indent3}tl.store({weight_ptr}, weight_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'last':
                             weight_ptr = f"{safe_var}_{inner}_weight_state_ptr + {out_offset}"
                             kernel_code_lines.extend([
                                 f"{indent2}weight_{inner}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}weight_{inner}_new = weight_{inner}_old + weight",
                                 f"{indent2}if is_macro_step_end:",
                                 f"{indent3}val_for_{inner} = val",
                                 f"{indent3}val_weight_for_{inner} = weight_{inner}_new",
                                 f"{indent3}tl.store({weight_ptr}, 0.0, mask=mask)",
                                 f"{indent2}else:",
                                 f"{indent3}tl.store({weight_ptr}, weight_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'mid':
                             weight_ptr = f"{safe_var}_{inner}_weight_state_ptr + {out_offset}"
                             kernel_code_lines.extend([
                                 f"{indent2}weight_{inner}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}weight_{inner}_new = weight_{inner}_old + weight",
                                 f"{indent2}if is_middle:",
                                 f"{indent3}tl.store({inner_ptr}, val, mask=mask)",
                                 f"{indent2}if is_inner_last:",
                                 f"{indent3}val_for_{inner} = tl.load({inner_ptr}, mask=mask, other=0.0)",
                                 f"{indent3}val_weight_for_{inner} = weight_{inner}_new",
                                 f"{indent3}tl.store({weight_ptr}, 0.0, mask=mask)",
                                 f"{indent2}else:",
                                 f"{indent3}tl.store({weight_ptr}, weight_{inner}_new, mask=mask)",
                             ])

                    for op in self._variable_ops[var]:
                        out_ptr = f"{safe_var}_{op}_ptr + {out_offset}"
                        op_parts = op.split('_')
                        if len(op_parts) > 1:
                            outer = op_parts[0]
                            inner = op_parts[1]
                            
                            # Parse K
                            k_val = 1
                            match_k = re.match(r'(max|min)(\d+)$', outer)
                            if match_k:
                                outer = match_k.group(1) # normalize
                                k_val = int(match_k.group(2))

                            val_var = f"val_for_{inner}"
                            kernel_code_lines.append(f"{indent2}if is_macro_step_end:")
                            
                            if outer == 'max':
                                # argmax pointer (automatically created alongside max)
                                argmax_ptr = f"{safe_var}_arg{op}_ptr + {out_offset}"
                                if k_val == 1:
                                    kernel_code_lines.extend([
                                        f"{indent3}if is_outer_first and macro_step_index==0:",
                                        f"{indent4}tl.store({out_ptr}, {val_var}, mask=mask)",
                                        f"{indent4}tl.store({argmax_ptr}, macro_step_index, mask=mask)",
                                        f"{indent3}else:",
                                        f"{indent4}old = tl.load({out_ptr}, mask=mask, other={val_var})",
                                        f"{indent4}cond_mask = {val_var} > old",
                                        f"{indent4}new = tl.maximum(old, {val_var})",
                                        f"{indent4}tl.store({out_ptr}, new, mask=mask)",
                                        f"{indent4}tl.store({argmax_ptr}, macro_step_index, mask=mask & cond_mask)",
                                    ])
                                else:
                                    # Bubble Insert Max K with ArgMax
                                    argmax_op = f"arg{op}"
                                    kernel_code_lines.extend([
                                        f"{indent3}# Bubble Insert Max K={k_val} with ArgMax (static_range optimized)",
                                        f"{indent3}new_val = {val_var}",
                                        f"{indent3}new_idx = tl.full([BLOCK_SIZE], macro_step_index, tl.int32)",
                                        f"{indent3}k_offset = ({out_offset}) * {k_val}",
                                        f"{indent3}base_ptr = {safe_var}_{op}_ptr + k_offset",
                                        f"{indent3}idx_base_ptr = {safe_var}_{argmax_op}_ptr + k_offset",
                                        
                                        f"{indent3}if is_outer_first and macro_step_index==0:",
                                        f"{indent4}tl.store(base_ptr, new_val, mask=mask)",
                                        f"{indent4}tl.store(idx_base_ptr, new_idx, mask=mask)",
                                        f"{indent4}for k in tl.static_range(1, {k_val}):",
                                        f"{indent5}tl.store(base_ptr + k, -float('inf'), mask=mask)",
                                        f"{indent5}tl.store(idx_base_ptr + k, 0, mask=mask)",
                                        f"{indent3}else:",
                                        f"{indent4}for k in tl.static_range({k_val}):",
                                        f"{indent5}old_k = tl.load(base_ptr + k, mask=mask, other=-float('inf'))",
                                        f"{indent5}old_idx_k = tl.load(idx_base_ptr + k, mask=mask, other=0)",
                                        f"{indent5}swap_mask = new_val > old_k",
                                        f"{indent5}val_to_store = tl.where(swap_mask, new_val, old_k)",
                                        f"{indent5}idx_to_store = tl.where(swap_mask, new_idx, old_idx_k)",
                                        f"{indent5}new_val = tl.where(swap_mask, old_k, new_val)",
                                        f"{indent5}new_idx = tl.where(swap_mask, old_idx_k, new_idx)",
                                        f"{indent5}tl.store(base_ptr + k, val_to_store, mask=mask)",
                                        f"{indent5}tl.store(idx_base_ptr + k, idx_to_store, mask=mask)",
                                    ])

                            elif outer == 'min':
                                # argmin pointer (automatically created alongside min)
                                argmin_ptr = f"{safe_var}_arg{op}_ptr + {out_offset}"
                                if k_val == 1:
                                    kernel_code_lines.extend([
                                        f"{indent3}if is_outer_first and macro_step_index==0:",
                                        f"{indent4}tl.store({out_ptr}, {val_var}, mask=mask)",
                                        f"{indent4}tl.store({argmin_ptr}, macro_step_index, mask=mask)",
                                        f"{indent3}else:",
                                        f"{indent4}old = tl.load({out_ptr}, mask=mask, other={val_var})",
                                        f"{indent4}cond_mask = {val_var} < old",
                                        f"{indent4}new = tl.minimum(old, {val_var})",
                                        f"{indent4}tl.store({out_ptr}, new, mask=mask)",
                                        f"{indent4}tl.store({argmin_ptr}, macro_step_index, mask=mask & cond_mask)",
                                    ])
                                else:
                                    # Min K with ArgMin
                                    argmin_op = f"arg{op}"
                                    kernel_code_lines.extend([
                                        f"{indent3}# Bubble Insert Min K={k_val} with ArgMin (static_range optimized)",
                                        f"{indent3}new_val = {val_var}",
                                        f"{indent3}new_idx = tl.full([BLOCK_SIZE], macro_step_index, tl.int32)",
                                        f"{indent3}k_offset = ({out_offset}) * {k_val}",
                                        f"{indent3}base_ptr = {safe_var}_{op}_ptr + k_offset",
                                        f"{indent3}idx_base_ptr = {safe_var}_{argmin_op}_ptr + k_offset",
                                        
                                        f"{indent3}if is_outer_first and macro_step_index==0:",
                                        f"{indent4}tl.store(base_ptr, new_val, mask=mask)",
                                        f"{indent4}tl.store(idx_base_ptr, new_idx, mask=mask)",
                                        f"{indent4}for k in tl.static_range(1, {k_val}):",
                                        f"{indent5}tl.store(base_ptr + k, float('inf'), mask=mask)",
                                        f"{indent5}tl.store(idx_base_ptr + k, 0, mask=mask)",
                                        f"{indent3}else:",
                                        f"{indent4}for k in tl.static_range({k_val}):",
                                        f"{indent5}old_k = tl.load(base_ptr + k, mask=mask, other=float('inf'))",
                                        f"{indent5}old_idx_k = tl.load(idx_base_ptr + k, mask=mask, other=0)",
                                        f"{indent5}swap_mask = new_val < old_k",
                                        f"{indent5}val_to_store = tl.where(swap_mask, new_val, old_k)",
                                        f"{indent5}idx_to_store = tl.where(swap_mask, new_idx, old_idx_k)",
                                        f"{indent5}new_val = tl.where(swap_mask, old_k, new_val)",
                                        f"{indent5}new_idx = tl.where(swap_mask, old_idx_k, new_idx)",
                                        f"{indent5}tl.store(base_ptr + k, val_to_store, mask=mask)",
                                        f"{indent5}tl.store(idx_base_ptr + k, idx_to_store, mask=mask)",
                                    ])
                            elif outer == 'sum':
                                kernel_code_lines.extend([
                                    f"{indent3}if is_outer_first and macro_step_index==0:",
                                    f"{indent4}tl.store({out_ptr}, {val_var}, mask=mask)",
                                    f"{indent3}else:",
                                    f"{indent4}old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                    f"{indent4}tl.store({out_ptr}, old + {val_var}, mask=mask)",
                                ])
                            elif outer == 'mean':
                                term = f"{val_var}"
                                kernel_code_lines.extend([
                                    f"{indent3}if is_outer_first and macro_step_index==0:",
                                    f"{indent4}tl.store({out_ptr}, {term}, mask=mask)",
                                    f"{indent3}else:",
                                    f"{indent4}old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                    f"{indent4}tl.store({out_ptr}, old + {term}, mask=mask)",
                                    f"{indent3}if is_outer_last:",
                                    f"{indent4}chk = tl.load({out_ptr}, mask=mask)",
                                    f"{indent4}tl.store({out_ptr}, chk / num_macro_steps, mask=mask)",
                                ])
                            continue

                        if op == 'mean':
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}old = tl.zeros_like(val)",
                                f"{indent2}else:",
                                f"{indent3}old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                f"{indent2}new = old + val * weight",
                                f"{indent2}if is_inner_last:",
                                f"{indent3}new = new / total_weight",
                                f"{indent2}tl.store({out_ptr}, new, mask=mask)",
                            ])
                        elif op == 'sum':
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}old = tl.zeros_like(val)",
                                f"{indent2}else:",
                                f"{indent3}old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                f"{indent2}new = old + val * weight",
                                f"{indent2}tl.store({out_ptr}, new, mask=mask)",
                            ])
                        elif op == 'max':
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                                f"{indent2}else:",
                                f"{indent3}old = tl.load({out_ptr}, mask=mask, other=val)",
                                f"{indent3}new = tl.maximum(old, val)",
                                f"{indent3}tl.store({out_ptr}, new, mask=mask)",
                            ])
                        elif op == 'min':
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                                f"{indent2}else:",
                                f"{indent3}old = tl.load({out_ptr}, mask=mask, other=val)",
                                f"{indent3}new = tl.minimum(old, val)",
                                f"{indent3}tl.store({out_ptr}, new, mask=mask)",
                            ])
                        elif op == 'argmax':
                            # 2D argmax logic
                            max_ptr = f"{safe_var}_max_ptr + {out_offset}"
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}tl.store({out_ptr}, macro_step_index, mask=mask)",
                                f"{indent2}else:",
                                f"{indent3}curr_max = tl.load({max_ptr}, mask=mask, other=val)",
                                f"{indent3}cond_mask = val > curr_max",
                                f"{indent3}tl.store({out_ptr}, macro_step_index, mask=mask & cond_mask)",
                            ])
                        elif op == 'argmin':
                            min_ptr = f"{safe_var}_min_ptr + {out_offset}"
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}tl.store({out_ptr}, macro_step_index, mask=mask)",
                                f"{indent2}else:",
                                f"{indent3}curr_min = tl.load({min_ptr}, mask=mask, other=val)",
                                f"{indent3}cond_mask = val < curr_min",
                                f"{indent3}tl.store({out_ptr}, macro_step_index, mask=mask & cond_mask)",
                            ])
                        elif op == 'last':
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_last:",
                                f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                            ])
                        elif op == 'first':
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                            ])
                        elif op == 'mid':
                            kernel_code_lines.extend([
                                f"{indent2}if is_middle:",
                                f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                            ])
                        elif op == 'median':
                            pass # Median P-Square is 1D only for now, not supported in 2D with levels currently
                kernel_code_lines.append("")

            if last_only_vars:
                kernel_code_lines.extend([
                    f"{indent}# 2D variables (last-only)",
                    f"{indent}if is_inner_last:",
                    f"{indent2}for level in tl.static_range(n_levels):",
                ])
                for var in last_only_vars:
                    safe_var = self._get_safe_name(var)
                    out_offset = f"(t * n_saved_points + offs) * n_levels + level"
                    
                    val_name = emit_val_2d(var)
                    kernel_code_lines.extend([
                        f"{indent3}val = {val_name}",
                        f"{indent3}tl.store({safe_var}_last_ptr + {out_offset}, val, mask=mask)",
                    ])
        kernel_code_lines.append("")

    def _generate_main_function(self, kernel_code_lines: List[str],
                                grouped_by_save_idx: Dict[str, List[str]],
                                tensor_info: Dict[str, Dict[str, Any]]) -> None:
        """Generate the main python function that calls kernels."""
        kernel_code_lines.extend([
            "# Main update function",
            "def internal_update_statistics(states, weight, total_weight, num_macro_steps, is_inner_first, is_inner_last, is_middle, is_outer_first, is_outer_last, macro_step_index, step_count_val, BLOCK_SIZE):",
        ])
        
        if self.num_trials > 1:
             kernel_code_lines.append(f"    num_trials = {self.num_trials}")
        else:
             kernel_code_lines.append(f"    num_trials = 1")

        for save_idx, var_list in grouped_by_save_idx.items():
            kernel_name = f"kernel_{save_idx}"
            
            # Get stride_input from metadata of first variable
            first_var = var_list[0]
            stride_input = 0
            for out_name, meta in self._metadata.items():
                if meta['original_variable'] == first_var:
                    stride_input = meta.get('stride_input', 0)
                    break
            
            kernel_code_lines.extend([
                f"    # Launch kernel for {save_idx}",
                f"    save_idx_len = len(states['{save_idx}'])",
                f"    stride_input = {stride_input}",
                f"    grid_{save_idx} = lambda meta: (triton.cdiv(save_idx_len, meta['BLOCK_SIZE']),)",
                f"    {kernel_name}[grid_{save_idx}](",
                f"        {save_idx}_ptr=states['{save_idx}'],",
            ])
            
            # Gather input pointers (resolving virtuals)
            input_args = set()
            def _gather_inputs(name):
                 info = self._field_registry.get(name)
                 if getattr(info, 'json_schema_extra', {}).get('category') == 'virtual':
                      expr = getattr(info, 'json_schema_extra', {}).get('expr', '')
                      import re
                      toks = set(re.findall(r'\b[a-zA-Z_]\w*\b', expr))
                      for t in toks:
                           if t in self._field_registry or t in self._tensor_registry:
                                _gather_inputs(t)
                 else:
                      input_args.add(name)
            
            for var in var_list:
                 _gather_inputs(var)

            # Add Input pointers
            sorted_inputs = sorted(list(input_args))
            for var in sorted_inputs:
                 safe_var = self._get_safe_name(var)
                 # Avoid duplicate argument if save_idx matches input var
                 if safe_var == save_idx:
                     continue
                 kernel_code_lines.append(f"        {safe_var}_ptr=states['{var}'],")
            
            # Add variable output pointers
            for var in var_list:
                safe_var = self._get_safe_name(var)
                added_median_state = False
                added_aux_ptrs = set()  # Track aux pointers for explicit argmax/argmin
                for op in self._variable_ops[var]:
                    kernel_code_lines.append(f"        {safe_var}_{op}_ptr=states['{var}_{op}'],")
                    
                    # For EXPLICIT argmax/argmin operations, add aux pointer
                    # NO automatic argmax/argmin generation for max/min operations
                    op_parts = op.split('_')
                    outer_op = op_parts[0]
                    
                    # Check for explicit argmax/argmin (e.g., argmax, argmax3, argmin, argmin3)
                    arg_match = re.match(r'arg(max|min)(\d*)$', outer_op)
                    if arg_match:
                        arg_type = arg_match.group(1)  # 'max' or 'min'
                        arg_k_str = arg_match.group(2)  # '' or '3' etc
                        aux_name = f"{arg_type}{arg_k_str or ''}_aux"  # e.g., 'max_aux', 'max3_aux'
                        if aux_name not in added_aux_ptrs:
                            aux_storage_key = f"{var}_{arg_type}{arg_k_str if arg_k_str else ''}_aux"
                            kernel_code_lines.append(f"        {safe_var}_{aux_name}_ptr=states['{aux_storage_key}'],")
                            added_aux_ptrs.add(aux_name)
                    
                    if (op.startswith('median') or re.match(r'^q\d+', op)):
                        if '_' in op:
                            # Compound median/quantile: per-inner state
                            inner = op.split('_')[1]
                            cmp_key = f"median_{inner}"
                            if cmp_key not in added_aux_ptrs:
                                kernel_code_lines.append(f"        {safe_var}_median_{inner}_q_state_ptr=states['{var}_median_{inner}_q_state'],")
                                kernel_code_lines.append(f"        {safe_var}_median_{inner}_n_state_ptr=states['{var}_median_{inner}_n_state'],")
                                added_aux_ptrs.add(cmp_key)
                        elif not added_median_state:
                            kernel_code_lines.append(f"        {safe_var}_median_q_state_ptr=states['{var}_median_q_state'],")
                            kernel_code_lines.append(f"        {safe_var}_median_n_state_ptr=states['{var}_median_n_state'],")
                            added_median_state = True
                
                # Inner state pointers (only for ops that need cross-step state)
                added_inner = set()
                for op in self._variable_ops[var]:
                    if '_' in op:
                        inner = op.split('_')[1]
                        if inner not in added_inner:
                             # 'last' inner op directly uses current value, no state needed
                             # 'median' inner op uses its own q/n state, not generic inner_state
                             if inner not in ('last', 'median'):
                                 kernel_code_lines.append(f"        {safe_var}_{inner}_inner_state_ptr=states['{var}_{inner}_inner_state'],")
                             if inner == 'mean':
                                 kernel_code_lines.append(f"        {safe_var}_{inner}_weight_state_ptr=states['{var}_{inner}_weight_state'],")
                             if inner == 'median':
                                 kernel_code_lines.append(f"        {safe_var}_median_inner_q_state_ptr=states['{var}_median_inner_q_state'],")
                                 kernel_code_lines.append(f"        {safe_var}_median_inner_n_state_ptr=states['{var}_median_inner_n_state'],")
                             added_inner.add(inner)
            
            kernel_code_lines.extend([
                "        weight=weight,",
                "        total_weight=total_weight,",
                "        num_macro_steps=num_macro_steps,",
                "        is_inner_first=is_inner_first,",
                "        is_inner_last=is_inner_last,",
                "        is_middle=is_middle,",
                "        is_outer_first=is_outer_first,",
                "        is_outer_last=is_outer_last,",
                "        macro_step_index=macro_step_index,",
                "        step_count_val=step_count_val,",
                "        n_saved_points=save_idx_len,",
            ])
            
            # Add second dimension if needed (use actual shape)
            if self.num_trials > 1:
                dims_2d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 3]
            else:
                dims_2d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 2]

            if dims_2d:
                var_2d = dims_2d[0]
                actual_shape = tensor_info[var_2d]['actual_shape']
                n_levels = actual_shape[-1]
                kernel_code_lines.append(f"        n_levels={n_levels},")
            
            kernel_code_lines.extend([
                "        BLOCK_SIZE=BLOCK_SIZE,",
                "        num_trials=num_trials,",
                "        stride_input=stride_input,",
                "    )",
                "",
            ])

    def _generate_aggregator_function(self) -> None:
        """
        Generate and compile the aggregation kernel function.
        """
        if not self._variables:
            raise ValueError("No variables initialized for statistics aggregation")

        # Analyze tensor information and group by save_idx
        tensor_info = {}
        grouped_by_save_idx = {}
        
        for var_name in self._variables:
            field_info = self._field_registry[var_name]
            tensor = self._tensor_registry.get(var_name)
            
            # Virtual fallback for meta info construction
            if tensor is None:
                 # Check if meta info was constructed during init
                 first_op = self._variable_ops[var_name][0]
                 out_name = f"{var_name}_{first_op}"
                 meta = self._metadata[out_name]
                 
                 tensor_info[var_name] = {
                    'tensor': None,
                    'tensor_shape': meta['tensor_shape'],
                    'actual_shape': meta['actual_shape'],
                    'actual_ndim': meta['actual_ndim']
                }
            else:
                json_schema_extra = getattr(field_info, 'json_schema_extra', {})
                save_idx = json_schema_extra.get('save_idx')
                tensor_shape = json_schema_extra.get('tensor_shape', ())
                
                tensor_info[var_name] = {
                    'tensor': tensor,
                    'tensor_shape': tensor_shape,  # Logical grid shape
                    'actual_shape': tensor.shape,  # Sampled data shape
                    'actual_ndim': tensor.ndim     # Based on actual data
                }
                
            # Need save_idx to group
            json_schema_extra = getattr(field_info, 'json_schema_extra', {})
            save_idx = json_schema_extra.get('save_idx')
            
            if save_idx not in grouped_by_save_idx:
                grouped_by_save_idx[save_idx] = []
            grouped_by_save_idx[save_idx].append(var_name)
        
        # Generate kernel code
        kernel_code_lines = self._generate_kernel_header()
        
        # Generate kernels for each save_idx group
        for save_idx, var_list in grouped_by_save_idx.items():
            kernel_name = f"kernel_{save_idx}"
            self._generate_kernel_for_group(kernel_code_lines, kernel_name, save_idx, var_list, tensor_info)
        
        # Generate main function
        self._generate_main_function(kernel_code_lines, grouped_by_save_idx, tensor_info)
        
        # Write kernel code to temporary file and import
        kernel_code = "\n".join(kernel_code_lines)
        self._write_and_import_kernels(kernel_code)
        
        # Save kernel file for external inspection if enabled
        if self.save_kernels:
            self._save_kernel_file(kernel_code)

    def initialize_statistics(self, variable_ops: Dict[str, List[str]]) -> None:
        """Initialize aggregation tensors and metadata for provided variables and ops."""
        # Reset generic state
        self._variables = set()
        # Normalize to lower-case list for each variable
        self._variable_ops = {}
        for var, ops in variable_ops.items():
            if ops is None:
                ops_list = ["mean"]
            elif isinstance(ops, str):
                ops_list = [ops]
            else:
                ops_list = list(ops)
            self._variable_ops[var] = [str(o).lower() for o in ops_list]
        self._storage.clear()
        self._output_keys = []
        self._metadata.clear()
        self._output_is_outer: Dict[str, bool] = {}
        
        self._aggregator_function = None
        self._aggregator_generated = False
        self._kernel_states = None
        self._current_macro_step_count = 0.0

        # Clean up old temporary files
        self._cleanup_temp_files()

        # Validate and setup each variable
        for var_name, ops in self._variable_ops.items():
            # Note: argmax/argmin cannot be user-specified operations.
            # They are automatically created as auxiliary storage when max/min is requested.
            # The argmax/argmin values are stored as integer indices (macro_step_index),
            # which are converted to NC time values (days since epoch) when written to the
            # output files. This creates a time-series-like output where each extreme value
            # record also has its corresponding occurrence time.
            import re

            # Sort ops to ensure consistent processing order
            ops.sort()
            
            tensor = None
            field_info = self._field_registry[var_name]
            json_schema_extra = getattr(field_info, 'json_schema_extra', {})
            category = json_schema_extra.get('category', 'param')
            is_virtual = category == 'virtual'

            if var_name in self._tensor_registry:
                tensor = self._tensor_registry[var_name]
            elif is_virtual:
                tensor = None
            else:
                raise ValueError(f"Variable '{var_name}' not registered. Call register_tensor() first.")
            
            target_dtype = tensor.dtype if tensor is not None else torch.float32

            tensor_shape = json_schema_extra.get('tensor_shape', ())
            save_idx = json_schema_extra.get('save_idx')
            description = getattr(field_info, 'description', f"Variable {var_name}")
            save_coord = json_schema_extra.get('save_coord')

            if not save_idx:
                raise ValueError(f"Variable '{var_name}' must have save_idx in json_schema_extra")

            if save_idx in self._tensor_registry:
                ref_save_idx = self._tensor_registry[save_idx]
                if tensor is not None:
                     # Real tensor shape/dim logic
                     tensor_ndim = tensor.ndim
                     tensor_base_shape = tensor.shape
                else:
                     # Virtual tensor logic
                     # Infer ndim/shape from tensor_shape or dependencies
                     tensor_ndim = 1 + (1 if self.num_trials > 1 else 0) # Base guess
                     if len(tensor_shape) > 1: # Has extra dims
                          tensor_ndim += (len(tensor_shape) - 1)
                     
                     # Construct hypothetical shape for allocation size
                     tensor_base_shape = (len(ref_save_idx),) # minimum
                     if len(tensor_shape) > 1:
                           # Try to resolve dimensions from dependencies or registry
                           expr = json_schema_extra.get('expr')
                           toks = re.findall(r'\b[a-zA-Z_]\w*\b', expr)
                           found_dep = False
                           for t in toks:
                                if t in self._tensor_registry:
                                     dep = self._tensor_registry[t]
                                     tensor_ndim = dep.ndim
                                     tensor_base_shape = dep.shape
                                     target_dtype = dep.dtype
                                     found_dep = True
                                     break
                           if not found_dep:
                                # Try to resolve dimensions from registry
                                try:
                                    extra_dims = []
                                    # tensor_shape[0] is the grid dimension, skip it
                                    # tensor_shape[1:] are the extra dimensions (e.g. levels)
                                    for dim_name in tensor_shape[1:]:
                                         if dim_name in self._tensor_registry:
                                              d_val = self._tensor_registry[dim_name]
                                              if d_val.numel() == 1:
                                                   extra_dims.append(int(d_val.item()))
                                              else:
                                                   raise ValueError
                                         else:
                                              raise ValueError
                                    
                                    # Construct a fake base shape that satisfies the slicing logic below
                                    # The logic below uses [2:] (if trials) or [1:] (if no trials)
                                    # to get the EXTRA dims.
                                    prefix_len = 2 if self.num_trials > 1 else 1
                                    tensor_base_shape = (1,) * prefix_len + tuple(extra_dims)
                                    target_dtype = torch.float32

                                except ValueError:
                                     raise ValueError(f"Virtual variable '{var_name}' has multi-dimensional shape {tensor_shape}. Dependencies not found in registry, and dimensions could not be resolved directly.")

                if self.num_trials > 1:
                    actual_shape = (self.num_trials, len(ref_save_idx)) + tensor_base_shape[2:] if tensor_ndim > 1 else (self.num_trials, len(ref_save_idx))
                else:
                    actual_shape = (len(ref_save_idx),) + tensor_base_shape[1:]
            else:
                raise ValueError(f"Save index '{save_idx}' not registered in tensor registry")
            
            actual_ndim = tensor_ndim
            max_ndim = 3 if self.num_trials > 1 else 2
            if actual_ndim > max_ndim:
                raise ValueError(f"Variable '{var_name}' has {actual_ndim} actual dimensions. Only up to {max_ndim}D variables are supported.")

            is_2d = (self.num_trials > 1 and actual_ndim == 3) or (self.num_trials == 1 and actual_ndim == 2)
            if is_2d and any(op.split('_')[0] in ['max', 'min'] or re.match(r'(max|min)\d+$', op.split('_')[0]) for op in ops):
                raise ValueError(f"max/min operations are not supported for 2D variable '{var_name}' (with levels).")

            # Track
            self._variables.add(var_name)

            for op in ops:
                out_name = f"{var_name}_{op}"
                
                # Parse op parts
                op_parts = op.split('_')
                outer_op = op_parts[0]
                
                # Check for K in max/min ops (e.g., max3, min3)
                k_val = 1
                match_k = re.match(r'(max|min)(\d+)$', outer_op)
                if match_k:
                    outer_base = match_k.group(1)
                    k_val = int(match_k.group(2))
                    outer_op = outer_base # normalize for allocation logic below (mostly)
                
                # Check for explicit argmax/argmin operators (e.g., argmax, argmax3)
                arg_match = re.match(r'arg(max|min)(\d*)$', outer_op)
                arg_k_val = 1  # Default for arg ops
                if arg_match:
                    arg_k_str = arg_match.group(2)
                    arg_k_val = int(arg_k_str) if arg_k_str else 1
                
                # Allocate storage by op
                if k_val > 1:
                    alloc_shape = actual_shape + (k_val,)
                else:
                    alloc_shape = actual_shape

                if arg_match:
                    # Explicit argmax/argmin operator - store integer indices only
                    arg_type = arg_match.group(1)  # 'max' or 'min'
                    arg_k_str = arg_match.group(2)  # '' or '3' etc
                    # arg_k_val already computed above
                    
                    if arg_k_val > 1:
                        arg_alloc_shape = actual_shape + (arg_k_val,)
                    else:
                        arg_alloc_shape = actual_shape
                    
                    # Store integer indices (macro step index within the window)
                    init_tensor = torch.zeros(arg_alloc_shape, dtype=torch.int32, device=self.device)
                    # Also need to track the corresponding extreme values for comparison
                    aux_name = f"{var_name}_{arg_type}{arg_k_str or ''}_aux"
                    if arg_type == 'max':
                        self._storage[aux_name] = torch.full(arg_alloc_shape, -torch.inf, dtype=target_dtype, device=self.device)
                    else:
                        self._storage[aux_name] = torch.full(arg_alloc_shape, torch.inf, dtype=target_dtype, device=self.device)
                    # aux is not an output, just internal state
                elif outer_op == 'max':
                    # max or maxK - NO automatic argmax
                    init_tensor = torch.full(alloc_shape, -torch.inf, dtype=target_dtype, device=self.device)
                elif outer_op == 'min':
                    # min or minK - NO automatic argmin
                    init_tensor = torch.full(alloc_shape, torch.inf, dtype=target_dtype, device=self.device)
                elif outer_op == 'first':
                    # Similar to 'last', we just need storage. Zero initialization is fine as it will be overwritten on is_first.
                    init_tensor = torch.zeros(alloc_shape, dtype=target_dtype, device=self.device)
                elif outer_op.startswith('median') or (outer_op.startswith('q') and outer_op[1:].isdigit()):
                    init_tensor = torch.zeros(alloc_shape, dtype=target_dtype, device=self.device)
                    # Allocate P-Square states: 5 markers (q) and 5 positions (n)
                    q_shape = (5,) + actual_shape
                    n_shape = (5,) + actual_shape
                    
                    # For compound median ops (e.g., median_max, median_mean),
                    # each inner op needs its own P-Square state
                    if len(op_parts) > 1:
                        inner_op = op_parts[1]
                        q_state_name = f"{var_name}_median_{inner_op}_q_state"
                        n_state_name = f"{var_name}_median_{inner_op}_n_state"
                    else:
                        q_state_name = f"{var_name}_median_q_state"
                        n_state_name = f"{var_name}_median_n_state"
                    
                    if q_state_name not in self._storage:
                        # q state holds marker heights
                        self._storage[q_state_name] = torch.zeros(q_shape, dtype=target_dtype, device=self.device)
                    if n_state_name not in self._storage:
                        # n state holds marker positions (integer counts)
                        self._storage[n_state_name] = torch.zeros(n_shape, dtype=torch.int32, device=self.device)
                else:
                    init_tensor = torch.zeros(alloc_shape, dtype=target_dtype, device=self.device)
                self._storage[out_name] = init_tensor
                self._output_keys.append(out_name)

                # For compound ops, allocate inner state
                if len(op_parts) > 1:
                    inner_op = op_parts[1]
                    # 'last' inner op doesn't need cross-step state - it directly uses current value
                    # 'median' inner op uses its own q/n state, not generic inner_state
                    needs_inner_state = inner_op not in ('last', 'median')
                    inner_state_name = f"{var_name}_{inner_op}_inner_state"
                    
                    # Allocate inner median q/n state separately
                    if inner_op == 'median':
                        q_shape = (5,) + actual_shape
                        n_shape = (5,) + actual_shape
                        q_inner_name = f"{var_name}_median_inner_q_state"
                        n_inner_name = f"{var_name}_median_inner_n_state"
                        if q_inner_name not in self._storage:
                            self._storage[q_inner_name] = torch.zeros(q_shape, dtype=target_dtype, device=self.device)
                        if n_inner_name not in self._storage:
                            self._storage[n_inner_name] = torch.zeros(n_shape, dtype=torch.int32, device=self.device)
                    
                    if needs_inner_state and inner_state_name not in self._storage:
                         # Initialize inner state
                         if inner_op == 'mean':
                             init_inner = torch.zeros(actual_shape, dtype=target_dtype, device=self.device)
                         elif inner_op == 'max':
                             init_inner = torch.full(actual_shape, -torch.inf, dtype=target_dtype, device=self.device)
                         elif inner_op == 'min':
                             init_inner = torch.full(actual_shape, torch.inf, dtype=target_dtype, device=self.device)
                         elif inner_op == 'sum':
                             init_inner = torch.zeros(actual_shape, dtype=target_dtype, device=self.device)
                         elif inner_op in ('first', 'mid'):
                             init_inner = torch.zeros(actual_shape, dtype=target_dtype, device=self.device)
                         else:
                             # Should be caught by validator, but safe fallback
                             init_inner = torch.zeros(actual_shape, dtype=target_dtype, device=self.device)
                         self._storage[inner_state_name] = init_inner
                         
                         # Allocate weight state only for inner ops that need it (mean)
                         if inner_op == 'mean':
                             weight_state_name = f"{var_name}_{inner_op}_weight_state"
                             if weight_state_name not in self._storage:
                                 self._storage[weight_state_name] = torch.zeros(actual_shape, dtype=target_dtype, device=self.device)

                if save_coord and save_coord not in self._coord_cache:
                    coord_tensor = self._tensor_registry[save_coord]
                    self._coord_cache[save_coord] = coord_tensor.detach().cpu().numpy()
                
                out_dtype = torch_to_numpy_dtype(target_dtype)
                
                # Check if this is an argmax/argmin op and determine the k value
                is_arg_op = arg_match is not None
                # For arg ops, use arg_k_val; otherwise use k_val
                effective_k = arg_k_val if is_arg_op else k_val

                meta = {
                    'original_variable': var_name,
                    'op': op,
                    'save_idx': save_idx,
                    'tensor_shape': tensor_shape,
                    'dtype': 'i4' if is_arg_op else out_dtype,  # int32 for arg ops
                    'actual_shape': actual_shape,
                    'actual_ndim': actual_ndim,
                    'save_coord': save_coord,
                    'description': f"{description} ({op})",
                    'stride_input': tensor.shape[1] if tensor is not None and self.num_trials > 1 else 0,
                    'k': effective_k,
                    'is_time_index': is_arg_op,  # argmax/argmin store integer indices
                }
                self._metadata[out_name] = meta
                
                # Classify as outer if it is a compound op (e.g. max_mean)
                self._output_is_outer[out_name] = len(op_parts) > 1


        # Generate kernels and prepare states for all requested variables/ops
        self._generate_aggregator_function()
        self._prepare_kernel_states()

    
    def update_statistics(self, weight: float, total_weight: float = 0.0, 
                          is_inner_first: bool = False, is_inner_last: bool = False, 
                          is_outer_first: bool = False, is_outer_last: bool = False,
                          BLOCK_SIZE: int = 128, custom_step_index: Optional[int] = None,
                          # Legacy kwargs support
                          is_first: bool = False, is_last: bool = False, 
                          is_middle: bool = False, is_macro_step_end: bool = False) -> None:
        if not self._aggregator_generated:
            raise RuntimeError("Statistics aggregation not initialized. Call initialize_streaming_aggregation() first.")
        
        # Handle legacy or new parameters
        _is_inner_first = is_inner_first or is_first
        _is_inner_last = is_inner_last or is_last
        
        if _is_inner_first:
            self._step_count = 0
        
        # Reset macro_step_index at the start of each outer statistics period
        # This ensures argmax/argmin indices are always relative to the start of the period
        if is_outer_first:
            self._macro_step_index = 0
            self._current_macro_step_count = 0.0
            
        if _is_inner_last:
             for out_name, is_outer in self._output_is_outer.items():
                 # We only trigger dirty for non-outer (Standard) ops when inner loop ends
                 if not is_outer:
                     self._dirty_outputs.add(out_name)
        
        if is_outer_last:
             for out_name, is_outer in self._output_is_outer.items():
                 # We trigger dirty for outer ops when outer loop ends
                 if is_outer:
                     self._dirty_outputs.add(out_name)

        if is_macro_step_end: # Legacy support or counter
            self._current_macro_step_count += 1.0

        macro_count_val = self._current_macro_step_count
            
        # Ensure kernel states is actually populated correctly for new keys
            
        self._aggregator_function(self._kernel_states, weight, total_weight, macro_count_val, 
                                  _is_inner_first, _is_inner_last, is_middle, 
                                  is_outer_first, is_outer_last,
                                  self._macro_step_index, self._step_count, BLOCK_SIZE)
        
        self._step_count += 1

    
    def finalize_time_step(self, dt: Union[datetime, cftime.datetime]) -> None:
        """
        Finalize the current time step by writing results to output.
        
        In streaming mode: writes to NetCDF files incrementally.
        In in-memory mode: copies current storage to result tensors.
        
        Args:
            dt: Time step to finalize (datetime or cftime.datetime)
        """
        # Record this time step for argmax/argmin index-to-time conversion
        # This is called at the end of each outer loop iteration
        self._macro_step_times.append(dt)

        # Handle in-memory mode
        if self.in_memory_mode:
            self._finalize_time_step_in_memory(dt)
            return

        if self.output_split_by_year:
            if self._current_year is None:
                # First call - set up files
                self._create_netcdf_files(year=dt.year)
                self._current_year = dt.year
            elif self._current_year != dt.year:
                # Year transition - create new files for new year
                self._create_netcdf_files(year=dt.year)
                self._current_year = dt.year
                self._macro_step_times = []  # Reset time mapping for new year
        else:
            # Create NetCDF files if not already created
            if not self._files_created:
                self._create_netcdf_files()
        
        # Increment macro step index for next iteration
        # (Note: index is reset to 0 in update_statistics when is_outer_first=True)
        self._macro_step_index += 1
        
        # Write all outputs that are marked dirty
        # Use explicit list of output keys to maintain order/determinism
        keys_to_write = [k for k in self._output_keys if k in self._dirty_outputs]
        
        # Clear dirty set for next step
        self._dirty_outputs.clear()
        
        for out_name in keys_to_write:
            tensor = self._storage[out_name]

            if out_name not in self._netcdf_files:
                continue
            output_paths = self._netcdf_files[out_name]
            
            # Check if this is a time index variable (argmax/argmin)
            metadata = self._metadata.get(out_name, {})
            is_time_index = metadata.get('is_time_index', False)
            k_val = metadata.get('k', 1)
            
            # Convert tensor to numpy
            raw_data = tensor.detach().cpu().numpy()
            
            # For time index variables, keep as integer indices (no conversion to time)
            # They will be stored as int32 in the NC file
            if is_time_index:
                time_step_data = raw_data.astype(np.int32)
            else:
                time_step_data = raw_data
            
            # Handle k > 1 case: write to separate files
            if k_val > 1 and isinstance(output_paths, list):
                for k_idx, output_path in enumerate(output_paths):
                    # Extract k-th slice (last dimension is k)
                    if time_step_data.ndim == 2:
                        # (saved_points, k) -> (saved_points,)
                        k_data = time_step_data[:, k_idx]
                    elif time_step_data.ndim == 3:
                        # (trials, saved_points, k) or (saved_points, levels, k)
                        k_data = time_step_data[:, :, k_idx]
                    elif time_step_data.ndim == 4:
                        # (trials, saved_points, levels, k)
                        k_data = time_step_data[:, :, :, k_idx]
                    else:
                        k_data = time_step_data[..., k_idx]
                    
                    # Use a unique key for executor selection
                    exec_key = f"{out_name}_{k_idx}"
                    idx = abs(hash(exec_key)) % len(self._write_executors)
                    executor = self._write_executors[idx]
                    
                    file_var_name = f"{out_name}_{k_idx}"
                    args = (file_var_name, k_data, output_path, dt)
                    future = executor.submit(_write_time_step_netcdf_process, args)
                    self._write_futures.append(future)
            else:
                # Single file case (k=1 or legacy)
                output_path = output_paths if not isinstance(output_paths, list) else output_paths[0]
                
                idx = abs(hash(out_name)) % len(self._write_executors)
                executor = self._write_executors[idx]
                
                args = (out_name, time_step_data, output_path, dt)
                future = executor.submit(_write_time_step_netcdf_process, args)
                self._write_futures.append(future)
            
        # Note: _current_macro_step_count is reset in update_statistics when is_outer_first=True
        
        # Manage backlog: Wait if too many steps are pending
        batch_n = len(self._storage)
        max_futures = self.max_pending_steps * batch_n
        
        while len(self._write_futures) > max_futures:
            # Pop the oldest future and wait for it
            future = self._write_futures.pop(0)
            try:
                future.result()
            except Exception as exc:
                print(f"  Failed to write time step (backlog): {exc}")
                raise exc
        
        # If we are strictly synchronous (max_pending_steps=1), we can clear the list
        # to keep it perfectly clean, although the loop above handles it too.
        if self.max_pending_steps == 1 and len(self._write_futures) >= batch_n:
             # Wait for the current batch completely (old behavior)
             for future in self._write_futures:
                 try:
                     future.result()
                 except Exception as exc:
                     print(f"  Failed to write time step {dt}: {exc}")
                     raise exc
             self._write_futures.clear()

    def _finalize_time_step_in_memory(self, dt: Union[datetime, cftime.datetime]) -> None:
        """
        Finalize time step in in-memory mode by copying storage to result tensors.
        
        Args:
            dt: Time step to finalize
        """
        # Increment macro step index for next iteration
        self._macro_step_index += 1
        
        # Get dirty outputs to write
        keys_to_write = [k for k in self._output_keys if k in self._dirty_outputs]
        self._dirty_outputs.clear()
        
        # Append storage tensors to result lists
        for out_name in keys_to_write:
            if out_name not in self._result_tensors:
                continue
                
            storage_tensor = self._storage[out_name]
            
            # Clone and move to result device (default CPU)
            # This frees GPU memory and allows dynamic growth
            result_copy = storage_tensor.detach().clone().to(self.result_device)
            self._result_tensors[out_name].append(result_copy)
        
        # Advance time index
        self._current_time_index += 1
        
        # Note: _current_macro_step_count is reset in update_statistics when is_outer_first=True
