# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

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


def _write_time_step_netcdf_process(args: Tuple) -> Tuple[str, int]:
    (mean_var_name, time_step_data, output_path, time_datetime) = args
    
    with nc.Dataset(output_path, 'a') as ncfile:
        nc_var = ncfile.variables[mean_var_name]
        time_var = ncfile.variables['time']
        
        current_len = len(nc_var)
        
        # Append data
        if time_step_data.ndim == 1:
            nc_var[current_len, :] = time_step_data
        elif time_step_data.ndim == 2:
            nc_var[current_len, :, :] = time_step_data
        elif time_step_data.ndim == 3:
            # e.g. (saved_points, levels, k) or (saved_points, k) ??
            # time_step_data is (saved_points, k) or (saved_points, levels)
            # If K is present, shape is (saved_points, k) -> 1D var with K
            nc_var[current_len, :, :, :] = time_step_data
        
        # Append datetime
        time_unit = time_var.getncattr("units")
        calendar = time_var.getncattr("calendar")
        time_val = nc.date2num(time_datetime, units=time_unit, calendar=calendar)
        time_var[current_len] = time_val
    
    # WSL optimization: Clear page cache for the written file to prevent memory bloat
    # Windows does not automatically reclaim WSL page cache memory effectively
    if _is_wsl() and hasattr(os, 'posix_fadvise'):
        try:
            with open(output_path, 'rb') as f:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except Exception:
            pass
    
    return (mean_var_name, current_len)


def _create_netcdf_file_process(args: Tuple) -> Path:
    """
    Process function for creating empty NetCDF files with proper structure.
    This function runs in a separate process.
    
    Args:
        args: Tuple containing (mean_var_name, metadata, coord_values, 
              output_dir, complevel, rank, year, calendar, time_unit, num_trials)
        
    Returns:
        Path to the created NetCDF file
    """
    (mean_var_name, metadata, coord_values, output_dir, complevel, rank, year, calendar, time_unit, num_trials) = args

    if year is not None:
        filename = f"{mean_var_name}_rank{rank}_{year}.nc"
    else:
        filename = f"{mean_var_name}_rank{rank}.nc"
    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with nc.Dataset(output_path, 'w', format='NETCDF4') as ncfile:
        # Write global attributes
        ncfile.setncattr('title', f'Time series for rank {rank}: {mean_var_name}')
        actual_shape = metadata.get('actual_shape', ())  # Spatial shape
        tensor_shape = metadata.get('tensor_shape', ())  # Logical grid shape
        coord_name = metadata.get('save_coord', None)
        dtype = metadata.get('dtype', 'f8')
        
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
        
        # Add Rank/K dimension if needed
        k_val = metadata.get('k', 1)
        if k_val > 1:
            dim_names.append('rank')
            if 'rank' not in ncfile.dimensions:
                ncfile.createDimension('rank', k_val) 
            elif len(ncfile.dimensions['rank']) != k_val:
                # Fallback if different K used in same file (unlikely for now)
                # But to be safe, maybe use rank_{k}
                pass 
                
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
        # Create main data variable (empty, will be filled during streaming)
        nc_var = ncfile.createVariable(
            mean_var_name,
            dtype,
            dim_names,
            zlib=True,
            complevel=complevel)
        nc_var.setncattr('description', metadata.get("description", ""))
        nc_var.setncattr('actual_shape', str(actual_shape))
        nc_var.setncattr('tensor_shape', str(tensor_shape))
    
    return output_path

class StatisticsAggregator:
    """
    Handles statistics aggregation with streaming NetCDF output to minimize memory usage.
    Each time step is immediately written to disk after accumulation.
    """
    
    def __init__(self, device: torch.device, output_dir: Path, rank: int, 
                 num_workers: int = 4, complevel: int = 4, save_kernels: bool = False,
                 output_split_by_year: bool = False, num_trials: int = 1,
                 max_pending_steps: int = 10):
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
        self.calendar = None
        self.time_unit = None
        self._current_year = None
        
        self._step_count = 0

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
        
        print(f"Initialized StreamingStatisticsAggregator for rank {self.rank} with {self.num_workers} workers")
        if self.save_kernels:
            print(f"Generated kernels will be saved to: {self.kernels_dir}")
    
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
                executor.shutdown(wait=True)
            self._write_executors = []
            self._write_futures = []
    
    def __del__(self):
        """Clean up temporary files and executor when the object is destroyed."""
        self._cleanup_temp_files()
        self._cleanup_executor()
        self._cleanup_lock_files()
    
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
        
        # Invalidate pre-computed states when new tensors are registered
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
        
        # Start the write executors (one per worker to guarantee serialization per variable)
        self._write_executors = [ProcessPoolExecutor(max_workers=1) for _ in range(self.num_workers)]
        self._write_futures = []
        
        print(f"Streaming aggregation system initialized successfully ({len(self._write_executors)} partitioned executors)")
    
    def _create_netcdf_files(self, year: Optional[int] = None) -> None:
        """Create empty NetCDF files with proper structure for streaming."""
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
                creation_futures[future] = out_name
            
            # Collect results
            for future in as_completed(creation_futures):
                mean_var_name = creation_futures[future]
                try:
                    output_path = future.result()
                    self._netcdf_files[mean_var_name] = output_path
                    self._all_created_files.add(output_path)
                    print(f"  Created {output_path.name}")
                except Exception as exc:
                    print(f"  Failed to create file for {mean_var_name}: {exc}")
                    raise exc
        
        self._files_created = True
        print(f"Created {len(self._netcdf_files)} NetCDF files for streaming")
    
    def _prepare_kernel_states(self) -> None:
        """Pre-compute and cache all tensors required for kernel execution."""
        required_tensors: Dict[str, torch.Tensor] = {}

        # Add original variables and their output buffers
        for var_name, ops in self._variable_ops.items():
            required_tensors[var_name] = self._tensor_registry[var_name]
            for op in ops:
                out_name = f"{var_name}_{op}"
                required_tensors[out_name] = self._storage[out_name]
                
                # Add inner states for compound ops
                if '_' in op:
                    parts = op.split('_')
                    inner = parts[1]
                    inner_name = f"{var_name}_{inner}_inner_state"
                    if inner_name in self._storage:
                        required_tensors[inner_name] = self._storage[inner_name]
                        # Also add weight state if needed
                        if inner == 'mean':
                            w_name = f"{var_name}_{inner}_weight_state"
                            if w_name in self._storage:
                                required_tensors[w_name] = self._storage[w_name]

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
            '- For argmax: stores current_step when a new max is found',
            '- For argmin: stores current_step when a new min is found',
            '- For mid: stores val when is_middle is True',
            '"""',
            "",
            "import triton",
            "import triton.language as tl",
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
    
    def _generate_1d_processing(self, kernel_code_lines: List[str], dims_1d: List[str]) -> None:
        """Generate code for 1D variable processing - FIXED LOGIC."""
        kernel_code_lines.extend([
            "    # -------- 1-D variables --------",
            "    # Use idx to access original data, offs for mean storage",
        ])
        
        # Load current values using idx (original grid indices)
        for var in dims_1d:
            kernel_code_lines.append(f"    {var} = tl.load({var}_ptr + idx, mask=mask, other=0.0)")
        
        # Load old means using offs (sequential storage)
        kernel_code_lines.append("    if is_first:")
        for var in dims_1d:
            kernel_code_lines.append(f"        {var}_old_mean = tl.zeros_like({var})")
        kernel_code_lines.append("    else:")
        for var in dims_1d:
            kernel_code_lines.append(f"        {var}_old_mean = tl.load({var}_mean_ptr + offs, mask=mask, other=0.0)")
        
        # Update means (time-weighted)
        for var in dims_1d:
            kernel_code_lines.append(f"    {var}_new_mean = {var}_old_mean + {var} * weight")
        # Finalize mean on last by dividing by total_weight
        kernel_code_lines.append("    if is_last:")
        for var in dims_1d:
            kernel_code_lines.append(f"        {var}_new_mean = {var}_new_mean / total_weight")
        
        # Store new means using offs (sequential storage)
        for var in dims_1d:
            kernel_code_lines.append(f"    tl.store({var}_mean_ptr + offs, {var}_new_mean, mask=mask)")
        kernel_code_lines.append("")
    
    def _generate_2d_processing(self, kernel_code_lines: List[str], dims_2d: List[str]) -> None:
        """Generate code for 2D variable processing - FIXED LOGIC."""
        kernel_code_lines.extend([
            "    # -------- 2-D variables --------",
            "    # Use idx for original data access, offs for mean storage",
            "    for level in tl.static_range(n_levels):",
        ])
        
        # Load current values using idx (original grid indices)
        for var in dims_2d:
            kernel_code_lines.append(f"        {var} = tl.load({var}_ptr + idx * n_levels + level, mask=mask, other=0.0)")
        
        # Load old means using offs (sequential storage)
        kernel_code_lines.append("        if is_first:")
        for var in dims_2d:
            kernel_code_lines.append(f"            {var}_old_mean = tl.zeros_like({var})")
        kernel_code_lines.append("        else:")
        for var in dims_2d:
            kernel_code_lines.append(f"            {var}_old_mean = tl.load({var}_mean_ptr + offs * n_levels + level, mask=mask, other=0.0)")
        
        # Update means (time-weighted)
        for var in dims_2d:
            kernel_code_lines.append(f"        {var}_new_mean = {var}_old_mean + {var} * weight")
        kernel_code_lines.append("        if is_last:")
        for var in dims_2d:
            kernel_code_lines.append(f"            {var}_new_mean = {var}_new_mean / total_weight")
        
        # Store new means using offs (sequential storage)
        for var in dims_2d:
            kernel_code_lines.append(f"        tl.store({var}_mean_ptr + offs * n_levels + level, {var}_new_mean, mask=mask)")
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

        # Pointers
        for var in var_list:
            kernel_code_lines.append(f"    {var}_ptr,")
            for op in self._variable_ops[var]:
                kernel_code_lines.append(f"    {var}_{op}_ptr,")
            
            # Inner state pointers
            added_inner = set()
            for op in self._variable_ops[var]:
                if '_' in op:
                    inner = op.split('_')[1]
                    if inner not in added_inner:
                        kernel_code_lines.append(f"    {var}_{inner}_inner_state_ptr,")
                        if inner == 'mean':
                            kernel_code_lines.append(f"    {var}_{inner}_weight_state_ptr,")
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
            "    current_step,",
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

        # Loop over trials
        kernel_code_lines.append("    for t in range(num_trials):")
        indent = "        "
        indent2 = indent + "    "
        indent3 = indent2 + "    "
        indent4 = indent3 + "    "
        indent5 = indent4 + "    "

        # 1D processing
        if dims_1d:
            kernel_code_lines.extend([
                f"{indent}# 1D variables",
            ])
            for var in dims_1d:
                ops = self._variable_ops[var]
                # Adjust pointers
                in_ptr = f"{var}_ptr + t * stride_input + idx"
                out_offset = f"t * n_saved_points + offs"

                if len(ops) == 1 and ops[0] == 'last':
                    # last-only: load only when needed
                    kernel_code_lines.extend([
                        f"{indent}if is_inner_last:",
                        f"{indent2}val = tl.load({in_ptr}, mask=mask, other=0.0)",
                        f"{indent2}tl.store({var}_last_ptr + {out_offset}, val, mask=mask)",
                    ])
                else:
                    kernel_code_lines.append(f"{indent}val = tl.load({in_ptr}, mask=mask, other=0.0)")

                    # Inner states update
                    inner_ops = set(op.split('_')[1] for op in ops if '_' in op)
                    for inner in inner_ops:
                        kernel_code_lines.append(f"{indent}val_for_{inner} = tl.zeros_like(val)")
                        inner_ptr = f"{var}_{inner}_inner_state_ptr + {out_offset}"
                        if inner == 'mean':
                             weight_ptr = f"{var}_{inner}_weight_state_ptr + {out_offset}"
                             kernel_code_lines.extend([
                                 f"{indent}inner_{inner}_old = tl.load({inner_ptr}, mask=mask, other=0.0)",
                                 f"{indent}weight_{inner}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                 f"{indent}inner_{inner}_new = inner_{inner}_old + val * weight",
                                 f"{indent}weight_{inner}_new = weight_{inner}_old + weight",
                                 f"{indent}if is_inner_last:",
                                 f"{indent2}tl.store({inner_ptr}, 0.0, mask=mask)",
                                 f"{indent2}tl.store({weight_ptr}, 0.0, mask=mask)",
                                 # Avoid DBZ
                                 f"{indent2}val_for_{inner} = inner_{inner}_new / (weight_{inner}_new)",
                                 f"{indent}else:",
                                 f"{indent2}tl.store({inner_ptr}, inner_{inner}_new, mask=mask)",
                                 f"{indent2}tl.store({weight_ptr}, weight_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'sum':
                             kernel_code_lines.extend([
                                 f"{indent}inner_{inner}_old = tl.load({inner_ptr}, mask=mask, other=0.0)",
                                 f"{indent}inner_{inner}_new = inner_{inner}_old + val * weight",
                                 f"{indent}if is_inner_last:",
                                 f"{indent2}tl.store({inner_ptr}, 0.0, mask=mask)",
                                 f"{indent2}val_for_{inner} = inner_{inner}_new",
                                 f"{indent}else:",
                                 f"{indent2}tl.store({inner_ptr}, inner_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'max':
                             kernel_code_lines.extend([
                                 f"{indent}if is_inner_first and current_step==0:",
                                 f"{indent2}inner_{inner}_new = val",
                                 f"{indent}else:",
                                 f"{indent2}inner_{inner}_old = tl.load({inner_ptr}, mask=mask, other=val)",
                                 f"{indent2}inner_{inner}_new = tl.maximum(inner_{inner}_old, val)",
                                 f"{indent}if is_inner_last:",
                                 f"{indent2}tl.store({inner_ptr}, -float('inf'), mask=mask)",
                                 f"{indent2}val_for_{inner} = inner_{inner}_new",
                                 f"{indent}else:",
                                 f"{indent2}tl.store({inner_ptr}, inner_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'min':
                             kernel_code_lines.extend([
                                 f"{indent}if is_inner_first and current_step==0:",
                                 f"{indent2}inner_{inner}_new = val",
                                 f"{indent}else:",
                                 f"{indent2}inner_{inner}_old = tl.load({inner_ptr}, mask=mask, other=val)",
                                 f"{indent2}inner_{inner}_new = tl.minimum(inner_{inner}_old, val)",
                                 f"{indent}if is_inner_last:",
                                 f"{indent2}tl.store({inner_ptr}, float('inf'), mask=mask)",
                                 f"{indent2}val_for_{inner} = inner_{inner}_new",
                                 f"{indent}else:",
                                 f"{indent2}tl.store({inner_ptr}, inner_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'mid':
                             kernel_code_lines.extend([
                                 f"{indent}if is_middle:",
                                 f"{indent2}tl.store({inner_ptr}, val, mask=mask)",
                                 f"{indent}if is_inner_last:",
                                 f"{indent2}val_for_{inner} = tl.load({inner_ptr}, mask=mask, other=0.0)",
                             ])
                        elif inner == 'last':
                             kernel_code_lines.extend([
                                 f"{indent}inner_{inner}_new = val",
                                 f"{indent}if is_inner_last:",
                                 f"{indent2}val_for_{inner} = inner_{inner}_new",
                             ])

                    for op in ops:
                        out_ptr = f"{var}_{op}_ptr + {out_offset}"
                        op_parts = op.split('_')
                        if len(op_parts) > 1:
                            outer = op_parts[0]
                            
                            # Parse K
                            k_val = 1
                            match_k = re.match(r'(max|min|argmax|argmin)(\d+)$', outer)
                            if match_k:
                                outer = match_k.group(1) # normalize
                                k_val = int(match_k.group(2))
                            
                            inner = op_parts[1]
                            val_var = f"val_for_{inner}"
                            kernel_code_lines.append(f"{indent}if is_inner_last:")
                            
                            if outer == 'max':
                                if k_val == 1:
                                    kernel_code_lines.extend([
                                        f"{indent2}if is_outer_first and current_step==0:",
                                        f"{indent3}tl.store({out_ptr}, {val_var}, mask=mask)",
                                        f"{indent2}else:",
                                        f"{indent3}old = tl.load({out_ptr}, mask=mask, other={val_var})",
                                        f"{indent3}new = tl.maximum(old, {val_var})",
                                        f"{indent3}tl.store({out_ptr}, new, mask=mask)",
                                    ])
                                else:
                                    # Bubble Insert for maxK
                                    kernel_code_lines.extend([
                                        f"{indent2}# Bubble Insert Max K={k_val}",
                                        f"{indent2}new_val = {val_var}",
                                        f"{indent2}k_offset = ({out_offset}) * {k_val}",
                                        f"{indent2}base_ptr = {var}_{op}_ptr + k_offset",
                                        
                                        f"{indent2}if is_outer_first and current_step==0:",
                                        f"{indent3}tl.store(base_ptr, new_val, mask=mask)",
                                        f"{indent3}for k in range(1, {k_val}):",
                                        f"{indent4}tl.store(base_ptr + k, -float('inf'), mask=mask)",
                                        f"{indent2}else:",
                                        f"{indent3}for k in range({k_val}):",
                                        f"{indent4}old_k = tl.load(base_ptr + k, mask=mask, other=-float('inf'))",
                                        f"{indent4}swap_mask = new_val > old_k",
                                        f"{indent4}val_to_store = tl.where(swap_mask, new_val, old_k)",
                                        f"{indent4}new_val = tl.where(swap_mask, old_k, new_val)",
                                        f"{indent4}tl.store(base_ptr + k, val_to_store, mask=mask)",
                                    ])

                            elif outer == 'argmax':
                                if k_val == 1:
                                    comp_ptr = f"{var}_max_{inner}_ptr + {out_offset}"
                                    kernel_code_lines.extend([
                                        f"{indent2}if is_outer_first and current_step==0:",
                                        f"{indent3}tl.store({out_ptr}, current_step, mask=mask)",
                                        f"{indent2}else:",
                                        f"{indent3}curr_max = tl.load({comp_ptr}, mask=mask, other={val_var})",
                                        f"{indent3}cond_mask = {val_var} > curr_max",
                                        f"{indent3}tl.store({out_ptr}, current_step, mask=mask & cond_mask)",
                                    ])
                                else:
                                    # Argmax K
                                    # We duplicate the bubble logic here, reading maxK values. 
                                    # argmaxK reads OLD maxK values (before update), decides swaps, updates indices.
                                    # Then maxK runs, reads OLD maxK values, updates values.
                                    
                                    comp_op = f"max{k_val}_{inner}"
                                    comp_ptr_base = f"{var}_{comp_op}_ptr"
                                    
                                    kernel_code_lines.extend([
                                        f"{indent2}# Bubble Insert Argmax K={k_val}",
                                        f"{indent2}comp_val = {val_var}",
                                        f"{indent2}new_idx = tl.full([BLOCK_SIZE], current_step, tl.int32)",
                                        f"{indent2}k_offset = ({out_offset}) * {k_val}",
                                        # Pointer to values (read-only for comparison)
                                        f"{indent2}val_base_ptr = {comp_ptr_base} + k_offset",
                                        # Pointer to indices (read-write)
                                        f"{indent2}idx_base_ptr = {var}_{op}_ptr + k_offset",
                                        
                                        f"{indent2}if is_outer_first and current_step==0:",
                                        f"{indent3}tl.store(idx_base_ptr, new_idx, mask=mask)",
                                        # others initialized to 0 (default)
                                        f"{indent2}else:",
                                        f"{indent3}for k in range({k_val}):",
                                        f"{indent4}old_val_k = tl.load(val_base_ptr + k, mask=mask, other=-float('inf'))",
                                        f"{indent4}old_idx_k = tl.load(idx_base_ptr + k, mask=mask, other=0)", # 0 or -1?
                                        f"{indent4}swap_mask = comp_val > old_val_k",
                                        # Swap indices based on value comparison
                                        f"{indent4}idx_to_store = tl.where(swap_mask, new_idx, old_idx_k)",
                                        f"{indent4}new_idx = tl.where(swap_mask, old_idx_k, new_idx)",
                                        # Must also start swapping value component for next iterations comparison
                                        f"{indent4}comp_val = tl.where(swap_mask, old_val_k, comp_val)",
                                        f"{indent4}tl.store(idx_base_ptr + k, idx_to_store, mask=mask)",
                                    ])

                            elif outer == 'min':
                                if k_val == 1:
                                    kernel_code_lines.extend([
                                        f"{indent2}if is_outer_first and current_step==0:",
                                        f"{indent3}tl.store({out_ptr}, {val_var}, mask=mask)",
                                        f"{indent2}else:",
                                        f"{indent3}old = tl.load({out_ptr}, mask=mask, other={val_var})",
                                        f"{indent3}new = tl.minimum(old, {val_var})",
                                        f"{indent3}tl.store({out_ptr}, new, mask=mask)",
                                    ])
                                else:
                                    # Min K
                                    kernel_code_lines.extend([
                                        f"{indent2}# Bubble Insert Min K={k_val}",
                                        f"{indent2}new_val = {val_var}",
                                        f"{indent2}k_offset = ({out_offset}) * {k_val}",
                                        f"{indent2}base_ptr = {var}_{op}_ptr + k_offset",
                                        
                                        f"{indent2}if is_outer_first and current_step==0:",
                                        f"{indent3}tl.store(base_ptr, new_val, mask=mask)",
                                        f"{indent3}for k in range(1, {k_val}):",
                                        f"{indent3}    tl.store(base_ptr + k, float('inf'), mask=mask)",
                                        f"{indent2}else:",
                                        f"{indent3}for k in range({k_val}):",
                                        f"{indent4}old_k = tl.load(base_ptr + k, mask=mask, other=float('inf'))",
                                        f"{indent4}swap_mask = new_val < old_k",
                                        f"{indent4}val_to_store = tl.where(swap_mask, new_val, old_k)",
                                        f"{indent4}new_val = tl.where(swap_mask, old_k, new_val)",
                                        f"{indent4}tl.store(base_ptr + k, val_to_store, mask=mask)",
                                    ])
                                    
                            elif outer == 'argmin':
                                if k_val == 1:
                                    comp_ptr = f"{var}_min_{inner}_ptr + {out_offset}"
                                    kernel_code_lines.extend([
                                        f"{indent2}if is_outer_first and current_step==0:",
                                        f"{indent3}tl.store({out_ptr}, current_step, mask=mask)",
                                        f"{indent2}else:",
                                        f"{indent3}curr_min = tl.load({comp_ptr}, mask=mask, other={val_var})",
                                        f"{indent3}cond_mask = {val_var} < curr_min",
                                        f"{indent3}tl.store({out_ptr}, current_step, mask=mask & cond_mask)",
                                    ])
                                else:
                                    # Argmin K
                                    # Assume argminK runs before minK
                                    comp_op = f"min{k_val}_{inner}"
                                    comp_ptr_base = f"{var}_{comp_op}_ptr"
                                    
                                    kernel_code_lines.extend([
                                        f"{indent2}# Bubble Insert Argmin K={k_val}",
                                        f"{indent2}comp_val = {val_var}",
                                        f"{indent2}new_idx = tl.full([BLOCK_SIZE], current_step, tl.int32)",
                                        f"{indent2}k_offset = ({out_offset}) * {k_val}",
                                        f"{indent2}val_base_ptr = {comp_ptr_base} + k_offset",
                                        f"{indent2}idx_base_ptr = {var}_{op}_ptr + k_offset",
                                        
                                        f"{indent2}if is_outer_first and current_step==0:",
                                        f"{indent3}tl.store(idx_base_ptr, new_idx, mask=mask)",
                                        f"{indent2}else:",
                                        f"{indent3}for k in range({k_val}):",
                                        f"{indent4}old_val_k = tl.load(val_base_ptr + k, mask=mask, other=float('inf'))",
                                        f"{indent4}old_idx_k = tl.load(idx_base_ptr + k, mask=mask, other=0)",
                                        f"{indent4}swap_mask = comp_val < old_val_k",
                                        f"{indent4}idx_to_store = tl.where(swap_mask, new_idx, old_idx_k)",
                                        f"{indent4}new_idx = tl.where(swap_mask, old_idx_k, new_idx)",
                                        f"{indent4}comp_val = tl.where(swap_mask, old_val_k, comp_val)",
                                        f"{indent4}tl.store(idx_base_ptr + k, idx_to_store, mask=mask)",
                                    ])
                                    
                            elif outer == 'mean':
                                # Outer mean implementation
                                kernel_code_lines.extend([
                                    f"{indent2}if is_outer_first and current_step==0:",
                                    f"{indent3}new = {val_var}",
                                    f"{indent2}else:",
                                    f"{indent3}old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                    f"{indent3}new = old + {val_var}",
                                    f"{indent2}if is_outer_last:",
                                    f"{indent3}new = new / (num_macro_steps)",
                                    f"{indent3}tl.store({out_ptr}, new, mask=mask)",
                                    f"{indent2}else:",
                                    f"{indent3}tl.store({out_ptr}, new, mask=mask)",
                                ])
                            elif outer == 'sum':
                                kernel_code_lines.extend([
                                    f"{indent2}if is_outer_first and current_step==0:",
                                    f"{indent3}tl.store({out_ptr}, {val_var}, mask=mask)",
                                    f"{indent2}else:",
                                    f"{indent3}old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                    f"{indent3}new = old + {val_var}",
                                    f"{indent3}tl.store({out_ptr}, new, mask=mask)",
                                ])
                            continue

                        if op == 'mean':
                            kernel_code_lines.extend([
                                f"{indent}if is_inner_first:",
                                f"{indent2}old = tl.zeros_like(val)",
                                f"{indent}else:",
                                f"{indent2}old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                f"{indent}new = old + val * weight",
                                f"{indent}if is_inner_last:",
                                f"{indent2}new = new / total_weight",
                                f"{indent}tl.store({out_ptr}, new, mask=mask)",
                            ])
                        elif op == 'sum':
                            kernel_code_lines.extend([
                                f"{indent}if is_inner_first:",
                                f"{indent2}old = tl.zeros_like(val)",
                                f"{indent}else:",
                                f"{indent2}old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                # Simple sum accumulation (integral if input is scaled)
                                # Assuming val is already scaled if needed, or if it's simple accumulation.
                                # Consistent with mean: mean is sum(val*weight)/sum(weight).
                                # So sum should be sum(val*weight).
                                f"{indent}new = old + val * weight",
                                f"{indent}tl.store({out_ptr}, new, mask=mask)",
                            ])
                        elif op == 'max':
                            kernel_code_lines.extend([
                                f"{indent}if is_inner_first:",
                                f"{indent2}tl.store({out_ptr}, val, mask=mask)",
                                f"{indent}else:",
                                f"{indent2}old = tl.load({out_ptr}, mask=mask, other=val)",
                                f"{indent2}new = tl.maximum(old, val)",
                                f"{indent2}tl.store({out_ptr}, new, mask=mask)",
                            ])
                        elif op == 'min':
                            kernel_code_lines.extend([
                                f"{indent}if is_inner_first:",
                                f"{indent2}tl.store({out_ptr}, val, mask=mask)",
                                f"{indent}else:",
                                f"{indent2}old = tl.load({out_ptr}, mask=mask, other=val)",
                                f"{indent2}new = tl.minimum(old, val)",
                                f"{indent2}tl.store({out_ptr}, new, mask=mask)",
                            ])
                        elif op == 'argmax':
                            max_ptr = f"{var}_max_ptr + {out_offset}"
                            kernel_code_lines.extend([
                                f"{indent}if is_inner_first:",
                                f"{indent2}tl.store({out_ptr}, current_step, mask=mask)",
                                f"{indent2}if is_inner_last:", 
                                f"{indent}else:",
                                f"{indent2}curr_max = tl.load({max_ptr}, mask=mask, other=val)",
                                f"{indent2}cond_mask = val > curr_max",
                                f"{indent2}tl.store({out_ptr}, current_step, mask=mask & cond_mask)",
                            ])
                        elif op == 'argmin':
                            min_ptr = f"{var}_min_ptr + {out_offset}"
                            kernel_code_lines.extend([
                                f"{indent}if is_inner_first:",
                                f"{indent2}tl.store({out_ptr}, current_step, mask=mask)",
                                f"{indent}else:",
                                f"{indent2}curr_min = tl.load({min_ptr}, mask=mask, other=val)",
                                f"{indent2}cond_mask = val < curr_min",
                                f"{indent2}tl.store({out_ptr}, current_step, mask=mask & cond_mask)",
                            ])
                        elif op == 'last':
                            kernel_code_lines.extend([
                                f"{indent}if is_inner_last:",
                                f"{indent2}tl.store({out_ptr}, val, mask=mask)",
                            ])
                        elif op == 'first':
                            kernel_code_lines.extend([
                                f"{indent}if is_inner_first:",
                                f"{indent2}tl.store({out_ptr}, val, mask=mask)",
                            ])
                        elif op == 'mid':
                            kernel_code_lines.extend([
                                f"{indent}if is_middle:",
                                f"{indent2}tl.store({out_ptr}, val, mask=mask)",
                            ])
                kernel_code_lines.append("")

        # 2D processing
        if dims_2d:
            non_last_only = [v for v in dims_2d if not (len(self._variable_ops[v]) == 1 and self._variable_ops[v][0] == 'last')]
            last_only_vars = [v for v in dims_2d if (len(self._variable_ops[v]) == 1 and self._variable_ops[v][0] == 'last')]

            if non_last_only:
                kernel_code_lines.extend([
                    f"{indent}# 2D variables (mean/min/max and mixed)",
                    f"{indent}for level in tl.static_range(n_levels):",
                ])
                for var in non_last_only:
                    in_ptr = f"{var}_ptr + (t * stride_input + idx) * n_levels + level"
                    out_offset = f"(t * n_saved_points + offs) * n_levels + level"
                    
                    kernel_code_lines.append(f"{indent2}val = tl.load({in_ptr}, mask=mask, other=0.0)")

                    # Inner states update
                    ops = self._variable_ops[var]
                    inner_ops = set(op.split('_')[1] for op in ops if '_' in op)
                    for inner in inner_ops:
                        # Initialize val_for_inner to avoid UnboundLocalError/NameError in generated code
                        # This value is used if is_update_outer is True, where it gets overwritten.
                        kernel_code_lines.append(f"{indent2}val_for_{inner} = tl.zeros_like(val)")

                        inner_ptr = f"{var}_{inner}_inner_state_ptr + {out_offset}"
                        if inner == 'mean':
                             weight_ptr = f"{var}_{inner}_weight_state_ptr + {out_offset}"
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
                             weight_ptr = f"{var}_{inner}_weight_state_ptr + {out_offset}"
                             kernel_code_lines.extend([
                                 f"{indent2}weight_{inner}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}weight_{inner}_new = weight_{inner}_old + weight",
                                 f"{indent2}if is_first and current_step==0:", 
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
                             weight_ptr = f"{var}_{inner}_weight_state_ptr + {out_offset}"
                             kernel_code_lines.extend([
                                 f"{indent2}weight_{inner}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}weight_{inner}_new = weight_{inner}_old + weight",
                                 f"{indent2}if is_first and current_step==0:",
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
                             weight_ptr = f"{var}_{inner}_weight_state_ptr + {out_offset}"
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
                             weight_ptr = f"{var}_{inner}_weight_state_ptr + {out_offset}"
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
                             weight_ptr = f"{var}_{inner}_weight_state_ptr + {out_offset}"
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
                        out_ptr = f"{var}_{op}_ptr + {out_offset}"
                        op_parts = op.split('_')
                        if len(op_parts) > 1:
                            outer = op_parts[0]
                            inner = op_parts[1]
                            
                            # Parse K
                            k_val = 1
                            match_k = re.match(r'(max|min|argmax|argmin)(\d+)$', outer)
                            if match_k:
                                outer = match_k.group(1) # normalize
                                k_val = int(match_k.group(2))

                            val_var = f"val_for_{inner}"
                            kernel_code_lines.append(f"{indent2}if is_macro_step_end:")
                            
                            if outer == 'max':
                                if k_val == 1:
                                    kernel_code_lines.extend([
                                        f"{indent3}if is_outer_first and current_step==0:",
                                        f"{indent4}tl.store({out_ptr}, {val_var}, mask=mask)",
                                        f"{indent3}else:",
                                        f"{indent4}old = tl.load({out_ptr}, mask=mask, other={val_var})",
                                        f"{indent4}new = tl.maximum(old, {val_var})",
                                        f"{indent4}tl.store({out_ptr}, new, mask=mask)",
                                    ])
                                else:
                                    # Bubble Insert Max K
                                    kernel_code_lines.extend([
                                        f"{indent3}# Bubble Insert Max K={k_val}",
                                        f"{indent3}new_val = {val_var}",
                                        f"{indent3}k_offset = ({out_offset}) * {k_val}",
                                        f"{indent3}base_ptr = {var}_{op}_ptr + k_offset",
                                        
                                        f"{indent3}if is_outer_first and current_step==0:",
                                        f"{indent4}tl.store(base_ptr, new_val, mask=mask)",
                                        f"{indent4}for k in range(1, {k_val}):",
                                        f"{indent4}    tl.store(base_ptr + k, -float('inf'), mask=mask)",
                                        f"{indent3}else:",
                                        f"{indent4}for k in range({k_val}):",
                                        f"{indent5}old_k = tl.load(base_ptr + k, mask=mask, other=-float('inf'))",
                                        f"{indent5}swap_mask = new_val > old_k",
                                        f"{indent5}val_to_store = tl.where(swap_mask, new_val, old_k)",
                                        f"{indent5}new_val = tl.where(swap_mask, old_k, new_val)",
                                        f"{indent5}tl.store(base_ptr + k, val_to_store, mask=mask)",
                                    ])

                            elif outer == 'argmax':
                                if k_val == 1:
                                    comp_ptr = f"{var}_max_{inner}_ptr + {out_offset}"
                                    kernel_code_lines.extend([
                                        f"{indent3}if is_outer_first and current_step==0:",
                                        f"{indent4}tl.store({out_ptr}, current_step, mask=mask)",
                                        f"{indent3}else:",
                                        f"{indent4}curr_max = tl.load({comp_ptr}, mask=mask, other={val_var})",
                                        f"{indent4}cond_mask = {val_var} > curr_max",
                                        f"{indent4}tl.store({out_ptr}, current_step, mask=mask & cond_mask)",
                                    ])
                                else:
                                    # Argmax K
                                    comp_op = f"max{k_val}_{inner}"
                                    comp_ptr_base = f"{var}_{comp_op}_ptr"
                                    kernel_code_lines.extend([
                                        f"{indent3}# Bubble Insert Argmax K={k_val}",
                                        f"{indent3}comp_val = {val_var}",
                                        f"{indent3}new_idx = tl.full([BLOCK_SIZE], current_step, tl.int32)",
                                        f"{indent3}k_offset = ({out_offset}) * {k_val}",
                                        f"{indent3}val_base_ptr = {comp_ptr_base} + k_offset",
                                        f"{indent3}idx_base_ptr = {var}_{op}_ptr + k_offset",
                                        
                                        f"{indent3}if is_outer_first and current_step==0:",
                                        f"{indent4}tl.store(idx_base_ptr, new_idx, mask=mask)",
                                        f"{indent3}else:",
                                        f"{indent4}for k in range({k_val}):",
                                        f"{indent5}old_val_k = tl.load(val_base_ptr + k, mask=mask, other=-float('inf'))",
                                        f"{indent5}old_idx_k = tl.load(idx_base_ptr + k, mask=mask, other=0)",
                                        f"{indent5}swap_mask = comp_val > old_val_k",
                                        f"{indent5}idx_to_store = tl.where(swap_mask, new_idx, old_idx_k)",
                                        f"{indent5}new_idx = tl.where(swap_mask, old_idx_k, new_idx)",
                                        f"{indent5}comp_val = tl.where(swap_mask, old_val_k, comp_val)",
                                        f"{indent5}tl.store(idx_base_ptr + k, idx_to_store, mask=mask)",
                                    ])

                            elif outer == 'min':
                                if k_val == 1:
                                    kernel_code_lines.extend([
                                        f"{indent3}if is_outer_first and current_step==0:",
                                        f"{indent4}tl.store({out_ptr}, {val_var}, mask=mask)",
                                        f"{indent3}else:",
                                        f"{indent4}old = tl.load({out_ptr}, mask=mask, other={val_var})",
                                        f"{indent4}new = tl.minimum(old, {val_var})",
                                        f"{indent4}tl.store({out_ptr}, new, mask=mask)",
                                    ])
                                else:
                                    # Min K
                                    kernel_code_lines.extend([
                                        f"{indent3}# Bubble Insert Min K={k_val}",
                                        f"{indent3}new_val = {val_var}",
                                        f"{indent3}k_offset = ({out_offset}) * {k_val}",
                                        f"{indent3}base_ptr = {var}_{op}_ptr + k_offset",
                                        
                                        f"{indent3}if is_outer_first and current_step==0:",
                                        f"{indent4}tl.store(base_ptr, new_val, mask=mask)",
                                        f"{indent4}for k in range(1, {k_val}):",
                                        f"{indent4}    tl.store(base_ptr + k, float('inf'), mask=mask)",
                                        f"{indent3}else:",
                                        f"{indent4}for k in range({k_val}):",
                                        f"{indent5}old_k = tl.load(base_ptr + k, mask=mask, other=float('inf'))",
                                        f"{indent5}swap_mask = new_val < old_k",
                                        f"{indent5}val_to_store = tl.where(swap_mask, new_val, old_k)",
                                        f"{indent5}new_val = tl.where(swap_mask, old_k, new_val)",
                                        f"{indent5}tl.store(base_ptr + k, val_to_store, mask=mask)",
                                    ])

                            elif outer == 'argmin':
                                if k_val == 1:
                                    comp_ptr = f"{var}_min_{inner}_ptr + {out_offset}"
                                    kernel_code_lines.extend([
                                        f"{indent3}if is_outer_first and current_step==0:",
                                        f"{indent4}tl.store({out_ptr}, current_step, mask=mask)",
                                        f"{indent3}else:",
                                        f"{indent4}curr_min = tl.load({comp_ptr}, mask=mask, other={val_var})",
                                        f"{indent4}cond_mask = {val_var} < curr_min",
                                        f"{indent4}tl.store({out_ptr}, current_step, mask=mask & cond_mask)",
                                    ])
                                else:
                                    # Argmin K
                                    comp_op = f"min{k_val}_{inner}"
                                    comp_ptr_base = f"{var}_{comp_op}_ptr"
                                    kernel_code_lines.extend([
                                        f"{indent3}# Bubble Insert Argmin K={k_val}",
                                        f"{indent3}comp_val = {val_var}",
                                        f"{indent3}new_idx = tl.full([BLOCK_SIZE], current_step, tl.int32)",
                                        f"{indent3}k_offset = ({out_offset}) * {k_val}",
                                        f"{indent3}val_base_ptr = {comp_ptr_base} + k_offset",
                                        f"{indent3}idx_base_ptr = {var}_{op}_ptr + k_offset",
                                        
                                        f"{indent3}if is_outer_first and current_step==0:",
                                        f"{indent4}tl.store(idx_base_ptr, new_idx, mask=mask)",
                                        f"{indent3}else:",
                                        f"{indent4}for k in range({k_val}):",
                                        f"{indent5}old_val_k = tl.load(val_base_ptr + k, mask=mask, other=float('inf'))",
                                        f"{indent5}old_idx_k = tl.load(idx_base_ptr + k, mask=mask, other=0)",
                                        f"{indent5}swap_mask = comp_val < old_val_k",
                                        f"{indent5}idx_to_store = tl.where(swap_mask, new_idx, old_idx_k)",
                                        f"{indent5}new_idx = tl.where(swap_mask, old_idx_k, new_idx)",
                                        f"{indent5}comp_val = tl.where(swap_mask, old_val_k, comp_val)",
                                        f"{indent5}tl.store(idx_base_ptr + k, idx_to_store, mask=mask)",
                                    ])
                            elif outer == 'sum':
                                kernel_code_lines.extend([
                                    f"{indent3}if is_outer_first and current_step==0:",
                                    f"{indent4}tl.store({out_ptr}, {val_var}, mask=mask)",
                                    f"{indent3}else:",
                                    f"{indent4}old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                    f"{indent4}tl.store({out_ptr}, old + {val_var}, mask=mask)",
                                ])
                            elif outer == 'mean':
                                term = f"{val_var}"
                                kernel_code_lines.extend([
                                    f"{indent3}if is_outer_first and current_step==0:",
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
                            max_ptr = f"{var}_max_ptr + {out_offset}"
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}tl.store({out_ptr}, current_step, mask=mask)",
                                f"{indent2}else:",
                                f"{indent3}curr_max = tl.load({max_ptr}, mask=mask, other=val)",
                                f"{indent3}cond_mask = val > curr_max",
                                f"{indent3}tl.store({out_ptr}, current_step, mask=mask & cond_mask)",
                            ])
                        elif op == 'argmin':
                            min_ptr = f"{var}_min_ptr + {out_offset}"
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}tl.store({out_ptr}, current_step, mask=mask)",
                                f"{indent2}else:",
                                f"{indent3}curr_min = tl.load({min_ptr}, mask=mask, other=val)",
                                f"{indent3}cond_mask = val < curr_min",
                                f"{indent3}tl.store({out_ptr}, current_step, mask=mask & cond_mask)",
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
                kernel_code_lines.append("")

            if last_only_vars:
                kernel_code_lines.extend([
                    f"{indent}# 2D variables (last-only)",
                    f"{indent}if is_inner_last:",
                    f"{indent2}for level in tl.static_range(n_levels):",
                ])
                for var in last_only_vars:
                    in_ptr = f"{var}_ptr + (t * stride_input + idx) * n_levels + level"
                    out_offset = f"(t * n_saved_points + offs) * n_levels + level"
                    kernel_code_lines.extend([
                        f"{indent3}val = tl.load({in_ptr}, mask=mask, other=0.0)",
                        f"{indent3}tl.store({var}_last_ptr + {out_offset}, val, mask=mask)",
                    ])
        kernel_code_lines.append("")

    def _generate_main_function(self, kernel_code_lines: List[str],
                                grouped_by_save_idx: Dict[str, List[str]],
                                tensor_info: Dict[str, Dict[str, Any]]) -> None:
        """Generate the main python function that calls kernels."""
        kernel_code_lines.extend([
            "# Main update function",
            "def internal_update_statistics(states, weight, total_weight, num_macro_steps, is_inner_first, is_inner_last, is_middle, is_outer_first, is_outer_last, current_step, BLOCK_SIZE):",
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
            
            # Add variable pointers
            for var in var_list:
                kernel_code_lines.append(f"        {var}_ptr=states['{var}'],")
                for op in self._variable_ops[var]:
                    kernel_code_lines.append(f"        {var}_{op}_ptr=states['{var}_{op}'],")
                
                # Inner state pointers
                added_inner = set()
                for op in self._variable_ops[var]:
                    if '_' in op:
                        inner = op.split('_')[1]
                        if inner not in added_inner:
                             kernel_code_lines.append(f"        {var}_{inner}_inner_state_ptr=states['{var}_{inner}_inner_state'],")
                             if inner in ('mean', 'max', 'min', 'first', 'last', 'mid'):
                                 kernel_code_lines.append(f"        {var}_{inner}_weight_state_ptr=states['{var}_{inner}_weight_state'],")
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
                "        current_step=current_step,",
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
            tensor = self._tensor_registry[var_name]
            json_schema_extra = getattr(field_info, 'json_schema_extra', {})
            save_idx = json_schema_extra.get('save_idx')
            tensor_shape = json_schema_extra.get('tensor_shape', ())
            
            tensor_info[var_name] = {
                'tensor': tensor,
                'tensor_shape': tensor_shape,  # Logical grid shape
                'actual_shape': tensor.shape,  # Sampled data shape
                'actual_ndim': tensor.ndim     # Based on actual data
            }
            
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
            # Handle compound op dependencies
            extra_ops = []
            import re
            for op in ops:
                if '_' in op:
                    parts = op.split('_')
                    outer_op = parts[0]
                    # Check for maxk/mink pattern
                    match_k = re.match(r'(max|min|argmax|argmin)(\d+)$', outer_op)
                    if match_k:
                        base = match_k.group(1)
                        k = int(match_k.group(2))
                        if base in ['argmax', 'argmin']:
                            # argmaxK needs maxK
                            req_base = 'max' if base == 'argmax' else 'min'
                            req = f"{req_base}{k}_{parts[1]}"
                            if req not in ops and req not in extra_ops:
                                extra_ops.append(req)
                    
                    elif parts[0] == 'argmax':
                        req = f"max_{parts[1]}"
                        if req not in ops and req not in extra_ops:
                            extra_ops.append(req)
                    elif parts[0] == 'argmin':
                        req = f"min_{parts[1]}"
                        if req not in ops and req not in extra_ops:
                            extra_ops.append(req)
            ops.extend(extra_ops)
            
            # Sort ops to ensure deps are processed if needed, though dict order usually fine
            ops.sort()
            
            if var_name not in self._tensor_registry:
                raise ValueError(f"Variable '{var_name}' not registered. Call register_tensor() first.")

            tensor = self._tensor_registry[var_name]
            field_info = self._field_registry[var_name]

            json_schema_extra = getattr(field_info, 'json_schema_extra', {})
            tensor_shape = json_schema_extra.get('tensor_shape', ())
            save_idx = json_schema_extra.get('save_idx')
            description = getattr(field_info, 'description', f"Variable {var_name}")
            save_coord = json_schema_extra.get('save_coord')

            if not save_idx:
                raise ValueError(f"Variable '{var_name}' must have save_idx in json_schema_extra")

            if save_idx in self._tensor_registry:
                if self.num_trials > 1:
                    actual_shape = (self.num_trials, len(self._tensor_registry[save_idx])) + tensor.shape[2:]
                else:
                    actual_shape = (len(self._tensor_registry[save_idx]),) + tensor.shape[1:]
            else:
                raise ValueError(f"Save index '{save_idx}' not registered in tensor registry")
            actual_ndim = tensor.ndim
            max_ndim = 3 if self.num_trials > 1 else 2
            if actual_ndim > max_ndim:
                raise ValueError(f"Variable '{var_name}' has {actual_ndim} actual dimensions. Only up to {max_ndim}D variables are supported.")

            is_2d = (self.num_trials > 1 and actual_ndim == 3) or (self.num_trials == 1 and actual_ndim == 2)
            if is_2d and any(op in ops for op in ['argmax', 'argmin', 'max', 'min']):
                raise ValueError(f"argmax/argmin/max/min operations are not supported for 2D variable '{var_name}' (with levels).")

            # Track
            self._variables.add(var_name)

            for op in ops:
                out_name = f"{var_name}_{op}"
                
                # Parse op parts
                op_parts = op.split('_')
                outer_op = op_parts[0]
                
                # Check for K
                k_val = 1
                match_k = re.match(r'(max|min|argmax|argmin)(\d+)$', outer_op)
                if match_k:
                    outer_base = match_k.group(1)
                    k_val = int(match_k.group(2))
                    outer_op = outer_base # normalize for allocation logic below (mostly)
                
                # Allocate storage by op
                if k_val > 1:
                    alloc_shape = actual_shape + (k_val,)
                else:
                    alloc_shape = actual_shape

                if outer_op.startswith('max'):
                    # max or maxK
                    init_tensor = torch.full(alloc_shape, -torch.inf, dtype=tensor.dtype, device=self.device)
                elif outer_op.startswith('min'):
                    init_tensor = torch.full(alloc_shape, torch.inf, dtype=tensor.dtype, device=self.device)
                elif outer_op.startswith('argmax') or outer_op.startswith('argmin'):
                    init_tensor = torch.zeros(alloc_shape, dtype=torch.int32, device=self.device)
                elif outer_op == 'first':
                    # Similar to 'last', we just need storage. Zero initialization is fine as it will be overwritten on is_first.
                    init_tensor = torch.zeros(alloc_shape, dtype=tensor.dtype, device=self.device)
                else:
                    init_tensor = torch.zeros(alloc_shape, dtype=tensor.dtype, device=self.device)
                self._storage[out_name] = init_tensor
                self._output_keys.append(out_name)

                # For compound ops, allocate inner state
                if len(op_parts) > 1:
                    inner_op = op_parts[1]
                    inner_state_name = f"{var_name}_{inner_op}_inner_state"
                    if inner_state_name not in self._storage:
                         # Initialize inner state
                         if inner_op == 'mean':
                             init_inner = torch.zeros(actual_shape, dtype=tensor.dtype, device=self.device)
                         elif inner_op == 'max':
                             init_inner = torch.full(actual_shape, -torch.inf, dtype=tensor.dtype, device=self.device)
                         elif inner_op == 'min':
                             init_inner = torch.full(actual_shape, torch.inf, dtype=tensor.dtype, device=self.device)
                         elif inner_op == 'sum':
                             init_inner = torch.zeros(actual_shape, dtype=tensor.dtype, device=self.device)
                         elif inner_op in ('first', 'last', 'mid'):
                             init_inner = torch.zeros(actual_shape, dtype=tensor.dtype, device=self.device)
                         else:
                             # Should be caught by validator, but safe fallback
                             init_inner = torch.zeros(actual_shape, dtype=tensor.dtype, device=self.device)
                         self._storage[inner_state_name] = init_inner
                         
                         # Allocate weight state if inner op is mean (to calculate weighted average)
                         if inner_op in ('mean', 'max', 'min', 'first', 'last', 'mid'):
                             weight_state_name = f"{var_name}_{inner_op}_weight_state"
                             if weight_state_name not in self._storage:
                                 self._storage[weight_state_name] = torch.zeros(actual_shape, dtype=tensor.dtype, device=self.device)

                if save_coord and save_coord not in self._coord_cache:
                    coord_tensor = self._tensor_registry[save_coord]
                    self._coord_cache[save_coord] = coord_tensor.detach().cpu().numpy()
                
                out_dtype = torch_to_numpy_dtype(tensor.dtype)
                if outer_op in ('argmax', 'argmin'):
                    out_dtype = 'i4' # int32

                meta = {
                    'original_variable': var_name,
                    'op': op,
                    'save_idx': save_idx,
                    'tensor_shape': tensor_shape,
                    'dtype': out_dtype,
                    'actual_shape': actual_shape,
                    'actual_ndim': actual_ndim,
                    'save_coord': save_coord,
                    'description': f"{description} ({op})",
                    'stride_input': tensor.shape[1] if self.num_trials > 1 else 0,
                    'k': k_val
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
            
        if _is_inner_last:
             for out_name, is_outer in self._output_is_outer.items():
                 if not is_outer:
                     self._dirty_outputs.add(out_name)
        
        if is_outer_last:
             for out_name, is_outer in self._output_is_outer.items():
                 if is_outer:
                     self._dirty_outputs.add(out_name)

        if is_macro_step_end: # Legacy support or counter
            self._current_macro_step_count += 1.0

        step_val = custom_step_index if custom_step_index is not None else self._step_count
        macro_count_val = (float(custom_step_index) + 1.0) if custom_step_index is not None else self._current_macro_step_count
            
        self._aggregator_function(self._kernel_states, weight, total_weight, macro_count_val, 
                                  _is_inner_first, _is_inner_last, is_middle, 
                                  is_outer_first, is_outer_last,
                                  step_val, BLOCK_SIZE)
        
        self._step_count += 1

    
    def finalize_time_step(self, dt: Union[datetime, cftime.datetime]) -> None:
        """
        Finalize the current time step by immediately writing to NetCDF files
        and resetting mean storage for the next time step.
        
        Args:
            dt: Time step to finalize
        """
        # Infer calendar and time_unit if not set
        if self.calendar is None:
            if hasattr(dt, 'calendar'):
                self.calendar = dt.calendar
            else:
                self.calendar = 'standard'
        
        if self.time_unit is None:
            self.time_unit = "days since 1900-01-01 00:00:00"

        if self.output_split_by_year:
            if self._current_year != dt.year:
                self._create_netcdf_files(year=dt.year)
                self._current_year = dt.year
        else:
            # Create NetCDF files if not already created
            if not self._files_created:
                self._create_netcdf_files()
        
        # Write all outputs that are marked dirty
        # Use explicit list of output keys to maintain order/determinism
        keys_to_write = [k for k in self._output_keys if k in self._dirty_outputs]
        
        # Clear dirty set for next step
        self._dirty_outputs.clear()
        
        for out_name in keys_to_write:
            tensor = self._storage[out_name]

            if out_name not in self._netcdf_files:
                continue
            output_path = self._netcdf_files[out_name]
            time_step_data = tensor.detach().cpu().numpy()
            
            # Select executor based on variable name hash to ensure serialization
            # This avoids the need for file locks, as all writes for a specific variable
            # will always go to the same single-threaded executor.
            idx = abs(hash(out_name)) % len(self._write_executors)
            executor = self._write_executors[idx]
            
            args = (out_name, time_step_data, output_path, dt)
            future = executor.submit(_write_time_step_netcdf_process, args)
            self._write_futures.append(future)
            
        # Reset counters
        self._current_macro_step_count = 0.0
        
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
