# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import hashlib
import importlib.util
import os
import random
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import netCDF4 as nc
import numpy as np
import torch
from numba import njit
from pydantic.fields import FieldInfo


@njit
def compute_group_to_rank(world_size: int, group_assignments: np.ndarray):
    """
    Compute a mapping from each original group ID to a rank, using a greedy load balance.

    Returns:
      full_map: array of length (max_original_id+1), where
                full_map[original_group_id] = assigned_rank (or -1 if absent)
    """
    # Handle edge cases early
    if world_size <= 0 or group_assignments.size == 0:
        max_gid = int(group_assignments.max()) if group_assignments.size > 0 else -1
        return np.full(max_gid + 1, -1, np.int64)

    # 1) Compress original IDs → 0..n_groups-1
    # unique_ids: sorted unique original IDs
    # inv: for each entry in group_assignments, its compressed ID
    unique_ids, inv = np.unique(group_assignments), None
    # compute inv via a dense id_map:
    max_gid = int(unique_ids[-1]) if unique_ids.size > 0 else -1
    id_map = np.full(max_gid + 1, -1, np.int64)
    id_map[unique_ids] = np.arange(unique_ids.size, dtype=np.int64)

    inv = id_map[group_assignments]
    n_groups = unique_ids.size

    # 2) Count sizes of each compressed group
    group_sizes = np.bincount(inv, minlength=n_groups).astype(np.int64)

    # 3) Greedy assignment: largest groups first → argmin(rank_loads)
    order = np.argsort(group_sizes)          # ascending
    rank_loads = np.zeros(world_size, np.int64)
    comp_to_rank = np.empty(n_groups, np.int64)

    for i in range(order.size - 1, -1, -1):  # iterate from largest to smallest
        g = order[i]
        r = int(np.argmin(rank_loads))       # lightest rank
        comp_to_rank[g] = r
        rank_loads[r] += group_sizes[g]

    # 4) Expand back to the full original ID space (fill -1 for IDs not present)
    full_map = np.full(max_gid + 1, -1, np.int64)
    # unique_ids are the only valid original IDs; assign directly
    full_map[unique_ids] = comp_to_rank

    return full_map

def torch_to_numpy_dtype(torch_dtype):
    dtype_mapping = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.float16: np.float16,
    }
    if torch_dtype not in dtype_mapping:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
    return dtype_mapping[torch_dtype]

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
        
        # Append datetime
        time_unit = time_var.getncattr("units")
        calendar = time_var.getncattr("calendar")
        time_val = nc.date2num(time_datetime, units=time_unit, calendar=calendar)
        time_var[current_len] = time_val
    
    return (mean_var_name, current_len)


def _create_netcdf_file_process(args: Tuple) -> Path:
    """
    Process function for creating empty NetCDF files with proper structure.
    This function runs in a separate process.
    
    Args:
        args: Tuple containing (mean_var_name, metadata, coord_values, 
              output_dir, complevel, rank)
        
    Returns:
        Path to the created NetCDF file
    """
    (mean_var_name, metadata, coord_values, output_dir, complevel, rank) = args

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

        time_unit = "seconds since 1900-01-01 00:00:00"
        calendar = "standard"
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
                 num_workers: int = 4, complevel: int = 4, save_kernels: bool = False):
        """
        Initialize the statistics aggregator.
        
        Args:
            device: PyTorch device for computations
            output_dir: Output directory for NetCDF files
            rank: Process rank identifier (int)
            num_workers: Number of worker processes for parallel NetCDF writing
            save_kernels: Whether to save generated kernel files for inspection
        """
        self.device = device
        self.output_dir = output_dir
        self.rank = rank
        self.num_workers = num_workers
        self.complevel = complevel
        self.save_kernels = save_kernels

        # Create kernels directory if saving is enabled
        if self.save_kernels:
            self.kernels_dir = self.output_dir / "generated_kernels"
            self.kernels_dir.mkdir(parents=True, exist_ok=True)

        # Internal state
        # Generic stats state (for all ops)
        self._variables: Set[str] = set()  # original variable names
        self._variable_ops: Dict[str, List[str]] = {}  # var -> list[ops]
        self._storage: Dict[str, torch.Tensor] = {}  # out_name -> tensor
        self._metadata: Dict[str, Dict[str, Any]] = {}  # out_name -> meta
        self._coord_cache: Dict[str, np.ndarray] = {}
        # Backward-compat mean-only state (used by Triton path)
        self._mean_variables: Set[str] = set()
        self._mean_storage: Dict[str, torch.Tensor] = {}
        self._mean_metadata: Dict[str, Dict[str, Any]] = {}
        self._tensor_registry: Dict[str, torch.Tensor] = {}
        self._field_registry: Dict[str, FieldInfo] = {}

        # Streaming mode support
        self._netcdf_files: Dict[str, Path] = {}  # out_name -> NetCDF file path
        self._files_created: bool = False

        # Thread pool for background writing
        self._write_executor: Optional[ProcessPoolExecutor] = None
        self._write_futures: List = []

        # Kernel state (mean fast-path)
        self._aggregator_function = None
        self._aggregator_generated = False
        self._kernel_states: Optional[Dict[str, torch.Tensor]] = None

        # Temporary file for generated kernels
        self._temp_kernel_file = None
        self._kernel_module = None
        self._saved_kernel_file = None
        
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
    
    def _cleanup_executor(self):
        """Clean up the write executor."""
        if self._write_executor:
            # Wait for pending writes to complete
            for future in self._write_futures:
                try:
                    future.result(timeout=30)  # Wait up to 30 seconds
                except:
                    pass
            self._write_executor.shutdown(wait=True)
            self._write_executor = None
    
    def __del__(self):
        """Clean up temporary files and executor when the object is destroyed."""
        self._cleanup_temp_files()
        self._cleanup_executor()
    
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
        
        print(f"[{self.rank}]: Registered tensor: {name} (actual_shape: {tensor.shape}, device: {tensor.device})")
    
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
        
        # Start the write executor
        self._write_executor = ProcessPoolExecutor(max_workers=self.num_workers)
        self._write_futures = []
        
        print("Streaming aggregation system initialized successfully")
    
    def _create_netcdf_files(self) -> None:
        """Create empty NetCDF files with proper structure for streaming."""
        if self._files_created:
            return
        
        print("Creating NetCDF file structure...")
        
        # Prepare file creation tasks
        creation_futures = {}
        # Use number of outputs instead of variables (supports multiple ops)
        n_outputs = len(self._metadata) if self._metadata else len(self._mean_metadata)
        actual_workers = max(1, min(self.num_workers, n_outputs))
        
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            if self._metadata:
                items = list(self._metadata.items())
            else:
                # fallback for mean-only path
                items = list(self._mean_metadata.items())
            for out_name, metadata in items:
                coord_name = metadata.get('save_coord')
                coord_values = self._coord_cache.get(coord_name, None)
                args = (out_name, metadata, coord_values, self.output_dir, self.complevel, self.rank)
                future = executor.submit(_create_netcdf_file_process, args)
                creation_futures[future] = out_name
            
            # Collect results
            for future in as_completed(creation_futures):
                mean_var_name = creation_futures[future]
                try:
                    output_path = future.result()
                    self._netcdf_files[mean_var_name] = output_path
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

        kernel_code_lines.extend([
            "    weight,",
            "    total_weight,",
            "    is_first,",
            "    is_last,",
            "    n_saved_points: tl.constexpr,",
        ])
        if dims_2d:
            kernel_code_lines.append("    n_levels: tl.constexpr,")
        kernel_code_lines.extend([
            "    BLOCK_SIZE: tl.constexpr,",
            "):",
            "    pid = tl.program_id(0)",
            "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
            "    mask = offs < n_saved_points",
            "",
            f"    idx = tl.load({save_idx}_ptr + offs, mask=mask)",
            "",
        ])

        # 1D processing
        if dims_1d:
            kernel_code_lines.extend([
                "    # 1D variables",
            ])
            for var in dims_1d:
                ops = self._variable_ops[var]
                if len(ops) == 1 and ops[0] == 'last':
                    # last-only: load only when needed
                    kernel_code_lines.extend([
                        "    if is_last:",
                        f"        val = tl.load({var}_ptr + idx, mask=mask, other=0.0)",
                        f"        tl.store({var}_last_ptr + offs, val, mask=mask)",
                    ])
                else:
                    kernel_code_lines.append(f"    val = tl.load({var}_ptr + idx, mask=mask, other=0.0)")
                    for op in ops:
                        if op == 'mean':
                            kernel_code_lines.extend([
                                "    if is_first:",
                                f"        old = tl.zeros_like(val)",
                                "    else:",
                                f"        old = tl.load({var}_mean_ptr + offs, mask=mask, other=0.0)",
                                f"    new = old + val * weight",
                                "    if is_last:",
                                f"        new = new / total_weight",
                                f"    tl.store({var}_mean_ptr + offs, new, mask=mask)",
                            ])
                        elif op == 'max':
                            kernel_code_lines.extend([
                                "    if is_first:",
                                f"        tl.store({var}_max_ptr + offs, val, mask=mask)",
                                "    else:",
                                f"        old = tl.load({var}_max_ptr + offs, mask=mask, other=val)",
                                "        new = tl.maximum(old, val)",
                                f"        tl.store({var}_max_ptr + offs, new, mask=mask)",
                            ])
                        elif op == 'min':
                            kernel_code_lines.extend([
                                "    if is_first:",
                                f"        tl.store({var}_min_ptr + offs, val, mask=mask)",
                                "    else:",
                                f"        old = tl.load({var}_min_ptr + offs, mask=mask, other=val)",
                                "        new = tl.minimum(old, val)",
                                f"        tl.store({var}_min_ptr + offs, new, mask=mask)",
                            ])
                        elif op == 'last':
                            kernel_code_lines.extend([
                                "    if is_last:",
                                f"        tl.store({var}_last_ptr + offs, val, mask=mask)",
                            ])
                kernel_code_lines.append("")

    # 2D processing
        if dims_2d:
            non_last_only = [v for v in dims_2d if not (len(self._variable_ops[v]) == 1 and self._variable_ops[v][0] == 'last')]
            last_only_vars = [v for v in dims_2d if (len(self._variable_ops[v]) == 1 and self._variable_ops[v][0] == 'last')]

            if non_last_only:
                kernel_code_lines.extend([
                    "    # 2D variables (mean/min/max and mixed)",
                    "    for level in tl.static_range(n_levels):",
                ])
                for var in non_last_only:
                    kernel_code_lines.append(f"        val = tl.load({var}_ptr + idx * n_levels + level, mask=mask, other=0.0)")
                    for op in self._variable_ops[var]:
                        if op == 'mean':
                            kernel_code_lines.extend([
                                "        if is_first:",
                                f"            old = tl.zeros_like(val)",
                                "        else:",
                                f"            old = tl.load({var}_mean_ptr + offs * n_levels + level, mask=mask, other=0.0)",
                                f"        new = old + val * weight",
                                "        if is_last:",
                                f"            new = new / total_weight",
                                f"        tl.store({var}_mean_ptr + offs * n_levels + level, new, mask=mask)",
                            ])
                        elif op == 'max':
                            kernel_code_lines.extend([
                                "        if is_first:",
                                f"            tl.store({var}_max_ptr + offs * n_levels + level, val, mask=mask)",
                                "        else:",
                                f"            old = tl.load({var}_max_ptr + offs * n_levels + level, mask=mask, other=val)",
                                "            new = tl.maximum(old, val)",
                                f"            tl.store({var}_max_ptr + offs * n_levels + level, new, mask=mask)",
                            ])
                        elif op == 'min':
                            kernel_code_lines.extend([
                                "        if is_first:",
                                f"            tl.store({var}_min_ptr + offs * n_levels + level, val, mask=mask)",
                                "        else:",
                                f"            old = tl.load({var}_min_ptr + offs * n_levels + level, mask=mask, other=val)",
                                "            new = tl.minimum(old, val)",
                                f"            tl.store({var}_min_ptr + offs * n_levels + level, new, mask=mask)",
                            ])
                        elif op == 'last':
                            kernel_code_lines.extend([
                                "        if is_last:",
                                f"            tl.store({var}_last_ptr + offs * n_levels + level, val, mask=mask)",
                            ])
                kernel_code_lines.append("")

            if last_only_vars:
                kernel_code_lines.extend([
                    "    # 2D variables (last-only)",
                    "    if is_last:",
                    "        for level in tl.static_range(n_levels):",
                ])
                for var in last_only_vars:
                    kernel_code_lines.extend([
                        f"            val = tl.load({var}_ptr + idx * n_levels + level, mask=mask, other=0.0)",
                        f"            tl.store({var}_last_ptr + offs * n_levels + level, val, mask=mask)",
                    ])
                kernel_code_lines.append("")
    
    def _generate_main_function(self, kernel_code_lines: List[str],
                                grouped_by_save_idx: Dict[str, List[str]],
                                tensor_info: Dict[str, Dict[str, Any]]) -> None:
        """Generate the main aggregation function."""
        kernel_code_lines.extend([
            "",
            "# ============================================================================",
            "# Main aggregation function",
            "# ============================================================================",
            "",
            "def internal_update_statistics(states, weight, total_weight, is_first, is_last, BLOCK_SIZE):",
            '    """',
            '    Update statistics for all registered variables (mean/max/min/last).',
            '    ',
            '    Args:',
            '        states: Dictionary of tensor states',
            '        weight: dt (seconds) for this sub-step',
            '        total_weight: total dt (seconds) for the window; used only when is_last==1',
            '        is_first: 1 on first sub-step, 0 otherwise',
            '        is_last: 1 on final sub-step of the time step, else 0',
            '        BLOCK_SIZE: GPU block size',
            '    """',
        ])
        for save_idx, var_list in grouped_by_save_idx.items():
            kernel_name = f"kernel_{save_idx}"
            
            # Check if any 2D variables exist in this group
            dims_2d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 2]
            
            kernel_code_lines.extend([
                f"    # Process variables with save_idx: {save_idx}",
                f"    # Variables: {', '.join(var_list)}",
                f"    save_idx_len = states['{save_idx}'].shape[0]",
                f"    grid_{save_idx} = lambda meta: (triton.cdiv(save_idx_len, meta['BLOCK_SIZE']),)",
                f"    {kernel_name}[grid_{save_idx}](",
                f"        {save_idx}_ptr=states['{save_idx}'],",
            ])
            
            # Add variable pointers
            for var in var_list:
                kernel_code_lines.append(f"        {var}_ptr=states['{var}'],")
                for op in self._variable_ops[var]:
                    kernel_code_lines.append(f"        {var}_{op}_ptr=states['{var}_{op}'],")
            
            kernel_code_lines.extend([
                "        weight=weight,",
                "        total_weight=total_weight,",
                "        is_first=is_first,",
                "        is_last=is_last,",
                "        n_saved_points=save_idx_len,",
            ])
            
            # Add second dimension if needed (use actual shape)
            if dims_2d:
                var_2d = dims_2d[0]
                actual_shape = tensor_info[var_2d]['actual_shape']
                n_levels = actual_shape[1]
                kernel_code_lines.append(f"        n_levels={n_levels},")
            
            kernel_code_lines.extend([
                "        BLOCK_SIZE=BLOCK_SIZE",
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
        self._metadata.clear()
        # Reset mean fast-path holders
        self._mean_variables = {k for k, v in self._variable_ops.items() if 'mean' in v}
        self._mean_storage.clear()
        self._mean_metadata.clear()
        self._aggregator_function = None
        self._aggregator_generated = False
        self._kernel_states = None

        # Clean up old temporary files
        self._cleanup_temp_files()

        # Validate and setup each variable
        for var_name, ops in self._variable_ops.items():
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
                actual_shape = (len(self._tensor_registry[save_idx]),) + tensor.shape[1:]
            else:
                raise ValueError(f"Save index '{save_idx}' not registered in tensor registry")
            actual_ndim = tensor.ndim
            if actual_ndim > 2:
                raise ValueError(f"Variable '{var_name}' has {actual_ndim} actual dimensions. Only 1D and 2D variables are supported.")

            # Track
            self._variables.add(var_name)

            for op in ops:
                out_name = f"{var_name}_{op}"
                # Allocate storage by op
                if op == 'max':
                    init_tensor = torch.full(actual_shape, -torch.inf, dtype=tensor.dtype, device=self.device)
                elif op == 'min':
                    init_tensor = torch.full(actual_shape, torch.inf, dtype=tensor.dtype, device=self.device)
                else:
                    init_tensor = torch.zeros(actual_shape, dtype=tensor.dtype, device=self.device)
                self._storage[out_name] = init_tensor

                if save_coord and save_coord not in self._coord_cache:
                    coord_tensor = self._tensor_registry[save_coord]
                    self._coord_cache[save_coord] = coord_tensor.detach().cpu().numpy()

                meta = {
                    'original_variable': var_name,
                    'op': op,
                    'save_idx': save_idx,
                    'tensor_shape': tensor_shape,
                    'dtype': torch_to_numpy_dtype(tensor.dtype),
                    'actual_shape': actual_shape,
                    'actual_ndim': actual_ndim,
                    'save_coord': save_coord,
                    'description': f"{description} ({op})",
                }
                self._metadata[out_name] = meta

                if op == 'mean':
                    mean_var_name = out_name
                    self._mean_storage[mean_var_name] = init_tensor
                    self._mean_metadata[mean_var_name] = meta
                    self._tensor_registry[mean_var_name] = init_tensor

        # Generate kernels and prepare states for all requested variables/ops
        self._generate_aggregator_function()
        self._prepare_kernel_states()

    
    def update_statistics(self, weight: float, total_weight: float = 0.0, is_first: bool = False, is_last: bool = False, BLOCK_SIZE: int = 128) -> None:
        if not self._aggregator_generated:
            raise RuntimeError("Statistics aggregation not initialized. Call initialize_streaming_aggregation() first.")
        self._aggregator_function(self._kernel_states, weight, total_weight, is_first, is_last, BLOCK_SIZE)
    
    def finalize_time_step(self, dt: datetime) -> None:
        """
        Finalize the current time step by immediately writing to NetCDF files
        and resetting mean storage for the next time step.
        
        Args:
            time_step: Time step to finalize
        """

        # Create NetCDF files if not already created
        if not self._files_created:
            self._create_netcdf_files()
        
        # Write all outputs
        for out_name, tensor in (self._storage if self._storage else self._mean_storage).items():
            output_path = self._netcdf_files[out_name]
            time_step_data = tensor.detach().cpu().numpy()
            args = (out_name, time_step_data, output_path, dt)
            future = self._write_executor.submit(_write_time_step_netcdf_process, args)
            self._write_futures.append(future)
        
        # Wait for this batch to complete before continuing
        # This ensures time steps are written in order
        completed_count = 0
        batch_n = len(self._storage) if self._storage else len(self._mean_storage)
        for future in self._write_futures[-batch_n:]:  # Only wait for current batch
            try:
                var_name, written_time_step = future.result()
                completed_count += 1
            except Exception as exc:
                print(f"  Failed to write time step {dt}: {exc}")
                raise exc
