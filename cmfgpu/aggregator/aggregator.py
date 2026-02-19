# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import hashlib
import os
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import cftime
import numpy as np
import torch
from pydantic.fields import FieldInfo

from cmfgpu.aggregator.kernel_codegen import KernelCodegenMixin
from cmfgpu.aggregator.netcdf_io import NetCDFIOMixin
from cmfgpu.aggregator.statistics import StatisticsMixin
from cmfgpu.aggregator.utils import sanitize_symbol


class StatisticsAggregator(NetCDFIOMixin, KernelCodegenMixin, StatisticsMixin):
    """
    Handles statistics aggregation with streaming NetCDF output to minimize memory usage.
    Each time step is immediately written to disk after accumulation.
    
    Supports two modes:
    1. Streaming mode (default): Write each time step to NetCDF files incrementally.
    2. In-memory mode: Store all time steps in memory (CPU by default) for small-scale analysis.
       Results are dynamically appended, no need to pre-specify total time steps.
    """

    def __init__(self, device: torch.device, output_dir: Optional[Path] = None, rank: int = 0, 
                 num_workers: int = 4, complevel: int = 4, save_kernels: bool = False,
                 output_split_by_year: bool = False, num_trials: int = 1,
                 max_pending_steps: int = 10, calendar: str = "standard",
                 time_unit: str = "days since 1900-01-01 00:00:00",
                 in_memory_mode: bool = False, result_device: Optional[torch.device] = None):
        """
        Initialize the statistics aggregator.
        
        Args:
            device: PyTorch device for computations
            output_dir: Output directory for NetCDF files (required for streaming mode, optional for in-memory mode)
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
            if self.output_dir is None:
                raise ValueError("output_dir is required when save_kernels=True")
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
    

    def _init_result_storage(self) -> None:
        """Initialize empty lists for in-memory result storage."""
        if not self.in_memory_mode:
            return
            
        self._result_tensors.clear()
        self._current_time_index = 0
        
        # Initialize empty lists for each output
        for out_name in self._output_keys:
            self._result_tensors[out_name] = []
        

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
    
