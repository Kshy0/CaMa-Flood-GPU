# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import cftime
import netCDF4 as nc
import numpy as np

from cmfgpu.aggregator.utils import is_wsl, sanitize_symbol

if TYPE_CHECKING:
    from cmfgpu.aggregator.aggregator import StatisticsAggregator


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
    if is_wsl() and hasattr(os, 'posix_fadvise'):
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



class NetCDFIOMixin:
    """Mixin providing NetCDF file creation and streaming write functionality."""

    def _create_netcdf_files(self: StatisticsAggregator, year: Optional[int] = None) -> None:
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


    def _finalize_time_step_in_memory(self: StatisticsAggregator, dt: Union[datetime, cftime.datetime]) -> None:
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
    

    def finalize_time_step(self: StatisticsAggregator, dt: Union[datetime, cftime.datetime]) -> None:
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
