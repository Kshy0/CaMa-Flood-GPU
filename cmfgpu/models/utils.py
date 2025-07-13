from __future__ import annotations
import numpy as np
import torch
from typing import Dict, Any, List, Set, Tuple, Optional
from pathlib import Path
import netCDF4 as nc
from pydantic.fields import FieldInfo
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import os
import sys
import importlib.util
from datetime import datetime
import hashlib
import random
from numba import njit

@njit
def compute_group_to_rank(world_size: int, group_assignments: np.ndarray):
    """
    Compute a mapping from each original group ID to a rank, using a greedy load balance.

    Steps:
      1. Compress original group IDs to 0..n_groups-1
      2. Count each compressed group size via bincount
      3. Greedily assign largest groups first to the rank with minimal current load
      4. Expand the compressed mapping back to the full original ID space

    Returns:
      full_map: array of length (max_original_id+1), where
                full_map[original_group_id] = assigned_rank
    """
    # 1. build a map old_id -> new_id
    max_gid = group_assignments.max()
    id_map = np.full(max_gid + 1, -1, np.int64)
    counts = np.zeros(max_gid + 1, np.int64)
    for gid in group_assignments:
        counts[gid] += 1
    new_id = 0
    for old_id in range(max_gid + 1):
        if counts[old_id] > 0:
            id_map[old_id] = new_id
            new_id += 1
    n_groups = new_id

    # 2. count sizes of each compressed group
    compressed = id_map[group_assignments]
    group_sizes = np.bincount(compressed, minlength=n_groups)

    # 3. greedy assignment into ranks
    rank_loads = np.zeros(world_size, np.int64)
    comp_to_rank = np.empty(n_groups, np.int64)
    order = np.argsort(group_sizes)  # ascending
    for idx in order[::-1]:          # assign largest first
        # find rank with smallest load
        best = 0
        best_load = rank_loads[0]
        for r in range(1, world_size):
            if rank_loads[r] < best_load:
                best_load = rank_loads[r]
                best = r
        comp_to_rank[idx] = best
        rank_loads[best] += group_sizes[idx]

    # 4. expand back to full original ID space
    full_map = np.full(max_gid + 1, -1, np.int64)
    for old_id in range(max_gid + 1):
        nid = id_map[old_id]
        if nid >= 0:
            full_map[old_id] = comp_to_rank[nid]

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
        args: Tuple containing (mean_var_name, metadata, coord_registry, 
              output_dir, rank)
        
    Returns:
        Path to the created NetCDF file
    """
    (mean_var_name, metadata, coord_values, output_dir, rank) = args
    
    filename = f"{mean_var_name}_rank{rank}.nc"
    output_path = output_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with nc.Dataset(output_path, 'w', format='NETCDF4') as ncfile:
        # Write global attributes
        ncfile.setncattr('title', f'Streaming time series for rank {rank}: {mean_var_name}')
        ncfile.setncattr('description', f'Streaming time series simulation rank {rank}: {metadata.get("description", "")}')
        
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
            coord_var.setncattr('long_name', f'Coordinate values for {mean_var_name}')

        time_unit = "seconds since 1900-01-01 00:00:00"
        calendar = "standard"
        time_var = ncfile.createVariable('time', 'f8', ('time',))
        time_var.setncattr('units', time_unit)
        time_var.setncattr('calendar', calendar)
        # Create main data variable (empty, will be filled during streaming)
        nc_var = ncfile.createVariable(mean_var_name, dtype, dim_names, chunksizes=None)
        nc_var.setncattr('actual_shape', str(actual_shape))
        nc_var.setncattr('tensor_shape', str(tensor_shape))
    
    return output_path

class StatisticsAggregator:
    """
    Handles statistics aggregation with streaming NetCDF output to minimize memory usage.
    Each time step is immediately written to disk after accumulation.
    """
    
    def __init__(self, device: torch.device, output_dir: Path, rank: int, 
                 num_workers: int = 4, save_kernels: bool = True):
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
        self.save_kernels = save_kernels
        
        # Create kernels directory if saving is enabled
        if self.save_kernels:
            self.kernels_dir = self.output_dir / "generated_kernels"
            self.kernels_dir.mkdir(parents=True, exist_ok=True)
        
        # Internal state
        self._mean_variables: Set[str] = set()
        self._mean_storage: Dict[str, torch.Tensor] = {}
        self._coord_cache: Dict[str, np.ndarray] = {}
        self._mean_metadata: Dict[str, Dict[str, Any]] = {}
        self._tensor_registry: Dict[str, torch.Tensor] = {}
        self._field_registry: Dict[str, FieldInfo] = {}
        
        
        # Streaming mode support

        self._netcdf_files: Dict[str, Path] = {}  # Variable name -> NetCDF file path
        self._files_created: bool = False
        
        # Thread pool for background writing
        self._write_executor: Optional[ProcessPoolExecutor] = None
        self._write_futures: List = []
        
        # Kernel state
        self._aggregator_function = None
        self._aggregator_generated = False
        self._kernel_states: Optional[Dict[str, torch.Tensor]] = None
        
        # Temporary file for generated kernels
        self._temp_kernel_file = None
        self._kernel_module = None
        self._saved_kernel_file = None
        
        print(f"Initialized StreamingStatisticsAggregator for rank {rank} with {num_workers} workers")
        if self.save_kernels:
            print(f"Generated kernels will be saved to: {self.kernels_dir}")
    
    def __del__(self):
        """Clean up temporary files and executor when the object is destroyed."""
        self._cleanup_temp_files()
        self._cleanup_executor()
    
    def _cleanup_temp_files(self):
        """Remove temporary kernel files."""
        if self._temp_kernel_file and os.path.exists(self._temp_kernel_file):
            try:
                os.unlink(self._temp_kernel_file)
            except:
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
        
        print(f"Registered tensor: {name} (actual_shape: {tensor.shape}, device: {tensor.device})")
    
    def initialize_streaming_aggregation(self, variable_names: List[str]) -> None:
        """
        Initialize streaming aggregation for specified variables.
        Creates NetCDF file structure but writes time steps incrementally.
        
        Args:
            variable_names: List of variable names to compute means for
        """
        print(f"Variables: {variable_names}")
        
        # Enable streaming mode
        self._files_created = False
        
        # Initialize single time step aggregation
        self.initialize_mean_aggregation(variable_names)
        
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
        actual_workers = min(self.num_workers, len(self._mean_variables))
        
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            for var_name in self._mean_variables:
                mean_var_name = f"{var_name}_mean"
                metadata = self._mean_metadata[mean_var_name]
                coord_name   = metadata.get('save_coord')
                coord_values = self._coord_cache.get(coord_name, None)
                args = (
                    mean_var_name,
                    metadata,
                    coord_values,
                    self.output_dir,
                    self.rank,
                )
                
                future = executor.submit(_create_netcdf_file_process, args)
                creation_futures[future] = mean_var_name
            
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
    
    def initialize_mean_aggregation(self, variable_names: List[str]) -> None:
        """
        Initialize mean aggregation for specified variables.
        
        Args:
            variable_names: List of variable names to compute means for
        """
        
        # Reset state
        self._mean_variables = set()
        self._mean_storage.clear()
        self._mean_metadata.clear()
        self._aggregator_function = None
        self._aggregator_generated = False
        self._kernel_states = None
        
        # Clean up old temporary files
        self._cleanup_temp_files()
        
        # Validate and setup each variable
        for var_name in variable_names:
            if var_name not in self._tensor_registry:
                raise ValueError(f"Variable '{var_name}' not registered. Call register_tensor() first.")
            
            tensor = self._tensor_registry[var_name]
            field_info = self._field_registry[var_name]
            
            # Extract metadata
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
            
            # Add to tracking set
            self._mean_variables.add(var_name)
            
            # Create mean storage tensor with same shape as actual tensor
            mean_shape = actual_shape
            mean_tensor = torch.zeros(mean_shape, dtype=tensor.dtype, device=self.device)
            mean_var_name = f"{var_name}_mean"
            self._mean_storage[mean_var_name] = mean_tensor
            if save_coord and save_coord not in self._coord_cache:
                coord_tensor = self._tensor_registry[save_coord]
                self._coord_cache[save_coord] = coord_tensor.detach().cpu().numpy()
            # Store metadata for NetCDF writing
            self._mean_metadata[mean_var_name] = {
                'original_variable': var_name,
                'save_idx': save_idx,
                'tensor_shape': tensor_shape,
                'dtype': torch_to_numpy_dtype(tensor.dtype),
                'actual_shape': actual_shape,
                'actual_ndim': actual_ndim,
                'save_coord': save_coord,
                'description': description,
            }
            
            # Register mean tensor in registry for kernel access
            self._tensor_registry[mean_var_name] = mean_tensor
        
        # Generate aggregation kernels
        self._generate_aggregator_function()
        
        # Pre-compute kernel states for performance
        self._prepare_kernel_states()

    
    def update_statistics(self, weight: int, refresh: bool = False, BLOCK_SIZE: int = 256) -> None:
        if not self._aggregator_generated:
            raise RuntimeError("Mean aggregation not initialized. Call initialize_mean_aggregation() first.")
        
        # Call the generated aggregation function with pre-computed states
        self._aggregator_function(self._kernel_states, weight, refresh, BLOCK_SIZE)
    
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
        
        for var_name in self._mean_variables:
            mean_var_name = f"{var_name}_mean"
            mean_tensor = self._mean_storage[mean_var_name]
            output_path = self._netcdf_files[mean_var_name]
            
            # Convert to numpy for writing
            time_step_data = mean_tensor.detach().cpu().numpy()
            
            # Submit write task
            args = (mean_var_name, time_step_data, output_path, dt)
            future = self._write_executor.submit(_write_time_step_netcdf_process, args)
            self._write_futures.append(future)
        
        # Wait for this batch to complete before continuing
        # This ensures time steps are written in order
        completed_count = 0
        for future in self._write_futures[-len(self._mean_variables):]:  # Only wait for current batch
            try:
                var_name, written_time_step = future.result()
                completed_count += 1
            except Exception as exc:
                print(f"  Failed to write time step {dt}: {exc}")
                raise exc
    
    def _prepare_kernel_states(self) -> None:
        """Pre-compute and cache all tensors required for kernel execution."""
        
        required_tensors = {}
        
        # Add original variables and their means
        for var_name in self._mean_variables:
            required_tensors[var_name] = self._tensor_registry[var_name]
            mean_var_name = f"{var_name}_mean"
            required_tensors[mean_var_name] = self._mean_storage[mean_var_name]
        
        # Collect required dimensions and save indices
        required_dims = set()
        required_save_indices = set()
        
        for var_name in self._mean_variables:
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

    def _generate_aggregator_function(self) -> None:
        """
        Generate and compile the aggregation kernel function.
        """
        if not self._mean_variables:
            raise ValueError("No variables initialized for mean aggregation")

        # Analyze tensor information and group by save_idx
        tensor_info = {}
        grouped_by_save_idx = {}
        
        for var_name in self._mean_variables:
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
            kernel_name = f"mean_kernel_{save_idx}"
            self._generate_kernel_for_group(kernel_code_lines, kernel_name, save_idx, var_list, tensor_info)
        
        # Generate main function
        self._generate_main_function(kernel_code_lines, grouped_by_save_idx, tensor_info)
        
        # Write kernel code to temporary file and import
        kernel_code = "\n".join(kernel_code_lines)
        self._write_and_import_kernels(kernel_code)
        
        # Save kernel file for external inspection if enabled
        if self.save_kernels:
            self._save_kernel_file(kernel_code)

    
    def _generate_kernel_header(self) -> List[str]:
        """Generate the header for the kernel file with documentation."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        var_list = sorted(list(self._mean_variables))
        
        header = [
            '"""',
            f'Auto-generated Triton kernels for CaMa-Flood-GPU statistics aggregation',
            f'Generated at: {timestamp}',
            f'Rank: {self.rank}',
            f'Variables: {", ".join(var_list)}',
            f'Device: {self.device}',
            '',
            'Kernel Logic:',
            '- Load save_idx values to get original grid indices',
            '- Use idx to access original data: data[idx]',
            '- Store means using sequential indexing: mean[offs]',
            '"""',
            "",
            "import triton",
            "import triton.language as tl",
            "",
            "# ============================================================================",
            f"# Generated Triton kernels for mean statistics aggregation - Rank {self.rank}",
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
        self._kernel_module = importlib.util.module_from_spec(spec)
        
        # Add to sys.modules to ensure proper import
        sys.modules[module_name] = self._kernel_module
        spec.loader.exec_module(self._kernel_module)
        
        # Get the aggregation function
        self._aggregator_function = getattr(self._kernel_module, 'internal_update_mean_statistics')
        self._aggregator_generated = True
    
    def _generate_kernel_for_group(self, kernel_code_lines: List[str], kernel_name: str, 
                                  save_idx: str, var_list: List[str], 
                                  tensor_info: Dict[str, Dict[str, Any]]) -> None:
        """Generate kernel code for a specific save_idx group."""
        # Separate 1D and 2D variables based on actual dimensions
        dims_1d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 1]
        dims_2d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 2]
        
        # Add documentation for this kernel
        kernel_code_lines.extend([
            f"# Kernel for save_idx: {save_idx}",
            f"# Variables: {', '.join(var_list)}",
            f"# 1D variables (actual): {', '.join(dims_1d) if dims_1d else 'None'}",
            f"# 2D variables (actual): {', '.join(dims_2d) if dims_2d else 'None'}",
            "",
        ])
        
        # Generate kernel signature
        kernel_code_lines.extend([
            "@triton.jit",
            f"def {kernel_name}(",
            f"    {save_idx}_ptr,",
        ])
        
        # Add variable pointers
        for var in var_list:
            kernel_code_lines.extend([
                f"    {var}_ptr,",
                f"    {var}_mean_ptr,",
            ])
        
        kernel_code_lines.extend([
            "    weight,",
            "    refresh,",
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
            "    # Load save_idx to get original grid indices",
            f"    idx = tl.load({save_idx}_ptr + offs, mask=mask)",
            "",
        ])
        
        # Process 1D variables
        if dims_1d:
            self._generate_1d_processing(kernel_code_lines, dims_1d)
        
        # Process 2D variables
        if dims_2d:
            self._generate_2d_processing(kernel_code_lines, dims_2d)
        
        kernel_code_lines.append("")
    
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
        kernel_code_lines.append("    if refresh:")
        for var in dims_1d:
            kernel_code_lines.append(f"        {var}_old_mean = tl.zeros_like({var})")
        kernel_code_lines.append("    else:")
        for var in dims_1d:
            kernel_code_lines.append(f"        {var}_old_mean = tl.load({var}_mean_ptr + offs, mask=mask, other=0.0)")
        
        # Update means
        for var in dims_1d:
            kernel_code_lines.append(f"    {var}_new_mean = {var}_old_mean + {var} / weight")
        
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
        kernel_code_lines.append("        if refresh:")
        for var in dims_2d:
            kernel_code_lines.append(f"            {var}_old_mean = tl.zeros_like({var})")
        kernel_code_lines.append("        else:")
        for var in dims_2d:
            kernel_code_lines.append(f"            {var}_old_mean = tl.load({var}_mean_ptr + offs * n_levels + level, mask=mask, other=0.0)")
        
        # Update means
        for var in dims_2d:
            kernel_code_lines.append(f"        {var}_new_mean = {var}_old_mean + {var} / weight")
        
        # Store new means using offs (sequential storage)
        for var in dims_2d:
            kernel_code_lines.append(f"        tl.store({var}_mean_ptr + offs * n_levels + level, {var}_new_mean, mask=mask)")
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
            "def internal_update_mean_statistics(states, weight, refresh, BLOCK_SIZE):",
            '    """',
            '    Update mean statistics for all registered variables.',
            '    ',
            '    Args:',
            '        states: Dictionary of tensor states',
            '        weight: Total number of sub-steps per time step',
            '        refresh: 0 for first step, 1 for subsequent steps',
            '        BLOCK_SIZE: GPU block size',
            '    """',
        ])
        
        for save_idx, var_list in grouped_by_save_idx.items():
            kernel_name = f"mean_kernel_{save_idx}"
            
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
                kernel_code_lines.extend([
                    f"        {var}_ptr=states['{var}'],",
                    f"        {var}_mean_ptr=states['{var}_mean'],",
                ])
            
            kernel_code_lines.extend([
                "        weight=weight,",
                "        refresh=refresh,",
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