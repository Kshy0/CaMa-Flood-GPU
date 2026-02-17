# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Union

import cftime
import netCDF4 as nc
import numpy as np
import torch
import torch.distributed as dist
from scipy.sparse import csr_matrix
from tqdm import tqdm

from cmfgpu.utils import binread, find_indices_in, is_rank_zero, read_map


def compute_runoff_id(runoff_lon, runoff_lat, hires_lon, hires_lat):
    """
    Calculates runoff grid IDs considering ascending or descending order of coordinates.

    Notes on longitude handling:
    - runoff_lon is assumed to be strictly monotonic (usually ascending) and may be in
      [-180, 180] or [0, 360].
    - hires_lon values (e.g., generated from high-res maps) are typically in [-180, 180].
    - To avoid wrap-around mismatches, hires_lon is first normalized to the same range
      as runoff_lon before computing indices.
    """

    def _wrap_lon(lon_vals, mode):
        """Normalize longitudes to the target range.
        mode: '0-360' -> [0, 360), '-180-180' -> [-180, 180)
        """
        if mode == '0-360':
            # Use modulo to bring into [0, 360)
            return np.mod(lon_vals, 360.0)
        else:
            # Shift-mod-wrap into [-180, 180)
            wrapped = (np.mod(lon_vals + 180.0, 360.0)) - 180.0
            return wrapped

    # Decide target longitude range based on runoff grid
    rmin, rmax = float(np.min(runoff_lon)), float(np.max(runoff_lon))
    # if entirely non-negative and up to 360 -> treat as 0-360; otherwise -180-180
    target_mode = '0-360' if (rmin >= 0.0 and rmax <= 360.0) else '-180-180'

    # Normalize hires_lon to the same wrap as runoff_lon
    hires_lon = _wrap_lon(hires_lon, target_mode)

    lon_ascending = runoff_lon[1] > runoff_lon[0]
    lat_ascending = runoff_lat[1] > runoff_lat[0]

    gsize_lon = abs(runoff_lon[1] - runoff_lon[0])
    gsize_lat = abs(runoff_lat[1] - runoff_lat[0])

    if lon_ascending:
        westin = runoff_lon[0] - 0.5 * gsize_lon
        ixin = np.floor((hires_lon - westin) / gsize_lon).astype(int)
    else:
        westin = runoff_lon[0] + 0.5 * gsize_lon
        ixin = np.floor((westin - hires_lon) / gsize_lon).astype(int)

    if lat_ascending:
        southin = runoff_lat[0] - 0.5 * gsize_lat
        iyin = np.floor((hires_lat - southin) / gsize_lat).astype(int)
    else:
        northin = runoff_lat[0] + 0.5 * gsize_lat
        iyin = np.floor((northin - hires_lat) / gsize_lat).astype(int)

    
    nxin = len(runoff_lon)
    nyin = len(runoff_lat)
    ixin[ixin == nxin] = 0
    assert np.all((ixin >= 0) & (ixin < nxin)), "Some hires_lon points fall outside the runoff grid (longitude)"
    assert np.all((iyin >= 0) & (iyin < nyin)), "Some hires_lat points fall outside the runoff grid (latitude)"

    runoff_id = iyin * nxin + ixin

    return runoff_id

class AbstractDataset(torch.utils.data.Dataset, ABC):
    """
    Custom abstract class that inherits from PyTorch Dataset.
    Defines a common interface for accessing data with distributed support.
    """

    def _convert_to_calendar(self, dt: Union[datetime, cftime.datetime]) -> Union[datetime, cftime.datetime]:
        if dt is None:
            return None
        if self.calendar == "standard":
            if isinstance(dt, cftime.datetime):
                return datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
            return dt
        else:
            # Convert to cftime with self.calendar
            # If it's already cftime, we recreate it to ensure the calendar attribute matches
            return cftime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, calendar=self.calendar)
    def __init__(
        self,
        start_date: Union[datetime, cftime.datetime],
        end_date: Union[datetime, cftime.datetime],
        time_interval: timedelta,
        out_dtype: str = "float32",
        chunk_len: int = 1,
        spin_up_cycles: int = 0,
        spin_up_start_date: Optional[Union[datetime, cftime.datetime]] = None,
        spin_up_end_date: Optional[Union[datetime, cftime.datetime]] = None,
        calendar: str = "standard",
        clip_negative: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.out_dtype = out_dtype
        self.chunk_len = chunk_len
        self.start_date = start_date
        self.end_date = end_date
        self.spin_up_cycles = spin_up_cycles
        self.spin_up_start_date = spin_up_start_date
        self.spin_up_end_date = spin_up_end_date
        self.time_interval = time_interval
        self.calendar = calendar
        self.clip_negative = clip_negative
        
        # Local runoff indices for spatial compression (set by build_local_runoff_matrix)
        self._local_runoff_indices: Optional[np.ndarray] = None
        
        # Convert dates to the specified calendar immediately
        self.start_date = self._convert_to_calendar(start_date)
        self.end_date = self._convert_to_calendar(end_date)
        self.spin_up_start_date = self._convert_to_calendar(spin_up_start_date)
        self.spin_up_end_date = self._convert_to_calendar(spin_up_end_date)

    def update_calendar(self, calendar: str):
        """
        Updates the calendar and converts all date attributes to the new calendar.
        """
        self.calendar = calendar
        self.start_date = self._convert_to_calendar(self.start_date)
        self.end_date = self._convert_to_calendar(self.end_date)
        self.spin_up_start_date = self._convert_to_calendar(self.spin_up_start_date)
        self.spin_up_end_date = self._convert_to_calendar(self.spin_up_end_date)

    def validate_files_exist(self, file_paths: list[Union[str, Path]]) -> None:
        """
        Validates that all files in the provided list exist.
        Raises FileNotFoundError if any are missing.
        """
        missing_files = []
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                missing_files.append(str(path))
        
        if missing_files:
            raise FileNotFoundError(
                f"The following required data files are missing:\n" +
                "\n".join(missing_files)
            )

        if is_rank_zero() and self.spin_up_cycles > 0:
            print(f"[Dataset] Spin-up enabled: {self.spin_up_cycles} cycles.")

    def is_valid_time_index(self, idx: int) -> bool:
        """
        Checks if the given time index corresponds to a valid data step (not padding).
        Subclasses with chunking/padding should override this.
        """
        return True

    def time_iter(self):
        """Returns an iterator that yields (time, is_valid, is_spin_up) tuples step-by-step."""
        valid_steps_count = 0
        
        # Calculate spin-up steps
        spin_up_steps = 0
        if self.spin_up_cycles > 0 and self.time_interval is not None:
             duration = self.get_spin_up_duration()
             spin_up_steps = int(duration.total_seconds() / self.time_interval.total_seconds())

        # Iterate exactly as many times as the DataLoader will produce data points
        # This ensures we handle padding steps at the end of the last chunk correctly
        total_chunks = len(self)
        total_items = total_chunks * self.chunk_len

        for idx in range(total_items):
            try:
                dt = self.get_time_by_index(idx)
                valid = self.is_valid_time_index(idx)
            except IndexError:
                # Padding steps (out of bounds)
                dt = datetime.min
                valid = False
            
            is_spin_up = valid_steps_count < spin_up_steps
            yield dt, valid, is_spin_up
            
            if valid:
                valid_steps_count += 1

    def get_spin_up_duration(self) -> timedelta:
        """Calculates the total duration of the spin-up period."""
        if self.spin_up_cycles > 0:
            if self.time_interval is None:
                 raise ValueError("time_interval must be provided for spin-up calculation")
            
            if self.spin_up_start_date is None or self.spin_up_end_date is None:
                raise ValueError("spin_up_start_date and spin_up_end_date must be provided if spin_up_cycles > 0")

            # Calculate duration of one cycle
            # Assuming spin_up_end_date is inclusive, so we add one time_interval
            cycle_duration = self.spin_up_end_date - self.spin_up_start_date + self.time_interval
            
            return cycle_duration * self.spin_up_cycles
        return timedelta(0)

    def get_virtual_start_time(self, verbose: bool = False) -> datetime:
        """Calculates the virtual start time including spin-up."""
        if not hasattr(self, 'start_date'):
             raise AttributeError("Dataset must have 'start_date' to calculate virtual start time")
        
        duration = self.get_spin_up_duration()
        virtual_start = self.start_date - duration
        
        if verbose and is_rank_zero() and self.spin_up_cycles > 0:
             print(f"[Dataset] Spin-up duration: {duration}")
             print(f"[Dataset] Virtual start time: {virtual_start}")
             
        return virtual_start

    def _calc_spin_up_params(self):
        if self.spin_up_cycles > 0:
            if self.time_interval is None:
                 raise ValueError("time_interval must be provided for spin-up calculation")
            # Calculate number of chunks in spin-up period
            total_duration = self.spin_up_end_date - self.spin_up_start_date
            total_steps = int((total_duration.total_seconds() / self.time_interval.total_seconds())) + 1
            self._spin_up_num_chunks = (total_steps + self.chunk_len - 1) // self.chunk_len
        else:
            self._spin_up_num_chunks = 0

    @property
    def num_spin_up_chunks(self) -> int:
        if self.spin_up_cycles > 0:
             if not hasattr(self, '_spin_up_num_chunks'):
                 self._calc_spin_up_params()
             return self._spin_up_num_chunks * self.spin_up_cycles
        return 0

    def read_chunk(self, idx: int) -> np.ndarray:
        """
        Default implementation of read_chunk that handles spin-up logic.
        Requires time_interval to be set.
        """
        if self.time_interval is None:
             raise NotImplementedError("time_interval must be provided for default read_chunk")
        
        if self.spin_up_cycles > 0:
             if not hasattr(self, '_spin_up_num_chunks'):
                 self._calc_spin_up_params()
             
             total_spin_up_chunks = self._spin_up_num_chunks * self.spin_up_cycles
             
             if idx < total_spin_up_chunks:
                 # In spin-up
                 cycle_idx = idx % self._spin_up_num_chunks
                 # Time relative to spin_up_start_date
                 steps_offset = cycle_idx * self.chunk_len
                 current_time = self.spin_up_start_date + self.time_interval * steps_offset
                 return self.get_data(current_time, self.chunk_len)
             
             # Main simulation
             idx -= total_spin_up_chunks

        # Main simulation time
        steps_offset = idx * self.chunk_len
        
        if not hasattr(self, 'start_date'):
             raise AttributeError("Dataset must have 'start_date' attribute to use default read_chunk")

        current_time = self.start_date + self.time_interval * steps_offset
        return self.get_data(current_time, self.chunk_len)

    @property
    def num_spin_up_chunks(self) -> int:
        if self.spin_up_cycles > 0:
            if not hasattr(self, '_spin_up_num_chunks'):
                 self._calc_spin_up_params()
            return self._spin_up_num_chunks * self.spin_up_cycles
        return 0



    def shard_forcing(
        self,
        batch_runoff: torch.Tensor,
        local_runoff_matrix: torch.Tensor,
        world_size: int,
    ) -> torch.Tensor:
        """
        Map grid runoff to catchments and handle distributed sync.

        Expected input shape: 
          - (B, T, N) for single trial
          - (B, T, K, N) for K trials
        
        N should match data_size (compressed runoff grids after build_local_runoff_matrix).
        Output shape: (M, C) where M is the product of non-spatial dims, C = number of catchments.
        """
        if batch_runoff.dim() == 3:
            B, T, N = batch_runoff.shape
            flat = batch_runoff.reshape(B * T, N)
        elif batch_runoff.dim() == 4:
            B, T, K, N = batch_runoff.shape
            flat = batch_runoff.reshape(B * T * K, N)
        else:
            raise ValueError(f"batch_runoff must be 3D or 4D, got shape {tuple(batch_runoff.shape)}")

        if world_size > 1:
            dist.broadcast(flat, src=0)

        out = (flat @ local_runoff_matrix).contiguous()
        
        # If input was 4D (B, T, K, N), reshape output to (B*T, K, C)
        # This makes it ready for step-by-step slicing in the main loop
        if batch_runoff.dim() == 4:
            B, T, K, N = batch_runoff.shape
            # out is currently (B*T*K, C)
            # Reshape to (B*T, K, C) so that out[step] gives (K, C) for all trials
            out = out.view(B * T, K, -1)
            
        return out

    def build_local_runoff_matrix(self, 
                                  runoff_mapping_file: str, 
                                  desired_catchment_ids: Optional[np.ndarray] = None, 
                                  device: torch.device = None,
                                  precision: Literal["float32", "float64"]="float32") -> torch.Tensor:
        """
        Build PyTorch sparse matrix for mapping runoff data to specified catchments.
        
        This method:
        1. Loads the sparse mapping matrix from npz file
        2. Extracts submatrix for desired catchments (or all if desired_catchment_ids is None)
        3. Identifies non-zero columns (active runoff grids)
        4. Sets self._local_runoff_indices for use in __getitem__
        5. Returns the compressed mapping matrix
        
        After calling this method, data_size will return the compressed size,
        and __getitem__ will automatically extract only the needed columns.
        
        Args:
            runoff_mapping_file: Path to the npz file containing the mapping matrix
            desired_catchment_ids: Array of catchment IDs to include. If None, uses all 
                                   catchments from the mapping file in their original order.
            device: PyTorch device for the output tensor
            precision: Data precision ("float32" or "float64")
            
        Returns:
            local_runoff_matrix: Stensor of shape (n_active_grids, n_catchments)
        """
        runoff_mapping_file = Path(runoff_mapping_file)
        torch_precision = torch.float32 if precision == "float32" else torch.float64
        if runoff_mapping_file is None or not os.path.exists(runoff_mapping_file):
            raise ValueError("Runoff mapping file not found. Cannot build local matrix.")
        
        # Store the mapping file path for later use
        self._runoff_mapping_file = runoff_mapping_file
        
        # Load the scipy compressed sparsparse e matrix data
        mapping_data = np.load(runoff_mapping_file)
        
        all_catchment_ids = mapping_data['catchment_ids']
        sparse_data = mapping_data['sparse_data']
        sparse_indices = mapping_data['sparse_indices'] 
        sparse_indptr = mapping_data['sparse_indptr']
        matrix_shape = tuple(mapping_data['matrix_shape'])
        
        full_sparse_matrix = csr_matrix(
            (sparse_data, sparse_indices, sparse_indptr),
            shape=matrix_shape
        )
        
        # If desired_catchment_ids is None, use all catchments from the mapping file
        if desired_catchment_ids is None:
            desired_catchment_ids = all_catchment_ids
            desired_row_indices = np.arange(len(all_catchment_ids))
        else:
            # Use find_indices_in to get row indices for desired catchments
            desired_row_indices = find_indices_in(desired_catchment_ids, all_catchment_ids)
            
            # Check which catchments were found
            valid_idx = desired_row_indices != -1
            
            if np.any(~valid_idx):
                raise ValueError(
                    f"Some desired catchments ({np.sum(~valid_idx)}) were not found in the mapping file. "
                    "Please check your input data or mapping file."
                )
        
        # Verify mapping matrix columns match full grid size
        lon, lat = self.get_coordinates()
        full_grid_size = len(lon) * len(lat)
        if matrix_shape[1] != full_grid_size:
            raise ValueError(
                f"Mapping matrix columns ({matrix_shape[1]}) != full grid size ({full_grid_size}). "
                "Please regenerate the mapping file."
            )
        
        # Validate coordinate metadata if present in npz
        if 'coord_lon' in mapping_data and 'coord_lat' in mapping_data:
            saved_lon = mapping_data['coord_lon']
            saved_lat = mapping_data['coord_lat']
            
            lon_match = (len(lon) == len(saved_lon) and 
                        np.allclose(lon, saved_lon, rtol=1e-5, atol=1e-8))
            lat_match = (len(lat) == len(saved_lat) and 
                        np.allclose(lat, saved_lat, rtol=1e-5, atol=1e-8))
            
            if not lon_match or not lat_match:
                raise ValueError(
                    f"Coordinate mismatch between dataset and mapping file.\n"
                    f"Dataset: lon[{len(lon)}] range [{lon[0]:.4f}, {lon[-1]:.4f}], "
                    f"lat[{len(lat)}] range [{lat[0]:.4f}, {lat[-1]:.4f}]\n"
                    f"Mapping: lon[{len(saved_lon)}] range [{saved_lon[0]:.4f}, {saved_lon[-1]:.4f}], "
                    f"lat[{len(saved_lat)}] range [{saved_lat[0]:.4f}, {saved_lat[-1]:.4f}]\n"
                    "Please regenerate the mapping file with matching coordinates."
                )
        else:
            if is_rank_zero():
                print("[Dataset] Warning: Mapping file has no coordinate metadata. "
                      "Consider regenerating for better validation.")
        
        # Extract submatrix for desired catchments only
        submatrix = full_sparse_matrix[desired_row_indices, :]
        
        # Remove columns that are all zeros to optimize memory
        col_sums = np.array(submatrix.sum(axis=0)).flatten()
        non_zero_cols = np.where(col_sums != 0)[0].astype(np.int64)
        
        if len(non_zero_cols) == 0:
            raise ValueError("No non-zero runoff data found for the desired catchments.")
        
        # Store the local runoff indices for use in __getitem__
        self._local_runoff_indices = non_zero_cols
        
        # Store the desired catchment IDs for use in export_catchment_runoff
        self._desired_catchment_ids = np.asarray(desired_catchment_ids)
        
        # Extract final submatrix with only non-zero columns and transpose
        # Shape: (num_runoff_grids, num_catchments)
        final_submatrix = submatrix[:, non_zero_cols].T.tocoo()

        # Convert to PyTorch sparse COO tensor
        row_tensor = torch.from_numpy(final_submatrix.row.astype(np.int64)).to(device)
        col_tensor = torch.from_numpy(final_submatrix.col.astype(np.int64)).to(device)
        data_tensor = torch.from_numpy(final_submatrix.data.astype(np.float32)).to(device).to(torch_precision)

        indices = torch.stack([row_tensor, col_tensor])
        local_runoff_matrix = torch.sparse_coo_tensor(
            indices, data_tensor, 
            size=(len(non_zero_cols), len(desired_catchment_ids)),
            dtype=torch_precision,
            device=device
        ).coalesce()
        
        if is_rank_zero():
            print(f"[Dataset] Built local runoff matrix: {len(non_zero_cols)} active grids "
                  f"out of {full_grid_size} total")
            print(f"[Dataset] Mapping {len(non_zero_cols)} runoff grids -> {len(desired_catchment_ids)} catchments")
        
        return local_runoff_matrix

    def export_runoff_climatology(
        self,
        out_path: Union[str, Path],
        local_runoff_matrix: torch.Tensor,
        var_name: str = "runoff_clm",
        dtype: Literal["float32", "float64"] = "float32",
        complevel: int = 4,
        device: Union[str, torch.device] = "cpu",
        units: str = "m3/s",
        description: Optional[str] = None,
    ) -> Path:
        """
        Compute the temporal-mean (climatological average) runoff and export to NetCDF.

        This mirrors the logic of CaMa-Flood's ``calc_outclm.F90``: iterate over
        every timestep in the dataset, accumulate the sum, and divide by the number
        of steps to obtain the daily-mean runoff climatology mapped to catchments.

        Requires ``build_local_runoff_matrix()`` to be called first.
        The compressed runoff is mapped to catchments via sparse matmul and then
        time-averaged.  Output NetCDF has dimension ``(saved_points,)`` with a
        ``catchment_id`` coordinate variable.

        Args:
            out_path: Full path (including filename) for the output NetCDF file.
            local_runoff_matrix: Sparse tensor returned by ``build_local_runoff_matrix()``,
                shape ``(n_grids, n_catchments)``.
            var_name: Variable name written into the NetCDF file.
            dtype: Output data type (``"float32"`` or ``"float64"``).
            complevel: zlib compression level (0–9).
            device: Device for computation (``"cpu"`` or ``"cuda:X"``).
            units: Units attribute written to the output variable.
            description: Optional long description attribute.

        Returns:
            Path to the created NetCDF file.
        """
        if not hasattr(self, '_desired_catchment_ids') or self._desired_catchment_ids is None:
            raise ValueError(
                "build_local_runoff_matrix() must be called before "
                "export_runoff_climatology()."
            )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        catchment_ids = self._desired_catchment_ids
        n_catch = len(catchment_ids)

        torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if dev.type == "cuda" and not torch.cuda.is_available():
            print("[export_runoff_climatology] CUDA not available; falling back to CPU.")
            dev = torch.device("cpu")

        # Prepare transposed mapping matrix: (n_catch, n_grids)
        t_mapping = local_runoff_matrix.to(dev).to(torch_dtype)
        t_mapping_T = t_mapping.t().coalesce()

        # ----- Accumulate mean over all chunks -----
        n_chunks = len(self)
        total_steps = 0
        accumulator = torch.zeros(n_catch, dtype=torch.float64, device=dev)

        pbar = tqdm(range(n_chunks), desc="Computing climatology", unit="chunk")
        for ci in pbar:
            block = self.read_chunk(ci)  # (T, n_grids)

            # Determine how many valid steps in this block
            base_idx = ci * self.chunk_len
            T = block.shape[0]
            valid_T = sum(
                1 for k in range(T) if self.is_valid_time_index(base_idx + k)
            )
            if valid_T == 0:
                continue

            block = block[:valid_T]

            # block: (T, n_grids)
            block_t = torch.as_tensor(block, dtype=torch_dtype, device=dev)
            # (n_catch, n_grids) @ (n_grids, T) -> (n_catch, T)
            agg = torch.sparse.mm(t_mapping_T, block_t.T)
            accumulator += agg.sum(dim=1).to(torch.float64)

            total_steps += valid_T

        pbar.close()

        if total_steps == 0:
            raise RuntimeError("No valid timesteps found — cannot compute climatology.")

        mean_data = (accumulator / total_steps).cpu().numpy()
        print(f"[export_runoff_climatology] Averaged over {total_steps} timesteps")

        # ----- Write NetCDF -----
        dtype_nc = "f4" if dtype == "float32" else "f8"
        desc = description if description else f"Time-averaged {var_name} over {total_steps} steps"

        with nc.Dataset(str(out_path), "w", format="NETCDF4") as ds:
            ds.setncattr("title", f"Runoff climatology ({var_name})")
            ds.setncattr("total_timesteps", total_steps)

            ds.createDimension("saved_points", n_catch)

            cid_var = ds.createVariable("catchment_id", "i8", ("saved_points",))
            cid_var[:] = catchment_ids

            out_var = ds.createVariable(
                var_name, dtype_nc, ("saved_points",),
                zlib=True, complevel=int(complevel),
            )
            out_var[:] = mean_data.astype(
                np.float32 if dtype == "float32" else np.float64
            )
            out_var.setncattr("description", desc)
            out_var.setncattr("units", units)

        print(f"[export_runoff_climatology] Saved to {out_path}")
        return out_path

    def export_catchment_runoff(
        self,
        out_dir: str | Path,
        local_runoff_matrix: torch.Tensor,
        var_name: str = "var",
        dtype: Literal["float32", "float64"] = "float32",
        complevel: int = 4,
        normalized: bool = False,
        device: str | torch.device = "cpu",
        split_by_year: bool = False,
        units: str = "m3/s",
        description: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Union[Path, List[Path]]:
        """
        Export catchment-aggregated runoff to a NetCDF file readable by MultiRankStatsReader.
        
        Requires build_local_runoff_matrix() to be called first.

        - Output filename: {filename}_rank0.nc or {var_name}_rank0.nc if filename not specified
          (or with _{year} suffix if split_by_year)
        - Dimensions: time (unlimited), saved_points
        - Variables:
            * time: numeric with units and calendar
            * catchment_id: (saved_points,) catchment IDs
            * {var_name}: (time, saved_points) aggregated runoff (area-weighted mean)

        GPU acceleration:
        - Set `device="cuda:0"` (or any CUDA device) to enable GPU-accelerated sparse matmul.
        - Falls back to CPU if CUDA is not available.
        
        Args:
            out_dir: Output directory for NetCDF files
            local_runoff_matrix: Sparse tensor from build_local_runoff_matrix(), shape (n_grids, n_catchments)
            var_name: Variable name in output NetCDF
            dtype: Output data type
            complevel: Compression level (0-9)
            normalized: If True, normalize mapping weights to sum to 1 per catchment
            device: Device for computation ("cpu" or "cuda:X")
            split_by_year: If True, create separate files per year
            units: Units string for the output variable
            description: Optional description for the output variable
            filename: Optional custom filename prefix (default: var_name)
        """
        # Require build_local_runoff_matrix() to be called first
        if not hasattr(self, '_desired_catchment_ids') or self._desired_catchment_ids is None:
            raise ValueError(
                "build_local_runoff_matrix() must be called before export_catchment_runoff(). "
                "This sets the catchment IDs and runoff mapping."
            )
        
        catchment_ids = self._desired_catchment_ids
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        n_catch = len(catchment_ids)
        n_cols = self.data_size  # This is the compressed size after build_local_runoff_matrix

        # Prepare device and torch types
        torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if dev.type == "cuda" and not torch.cuda.is_available():
            print("CUDA not available; falling back to CPU for export_catchment_runoff.")
            dev = torch.device("cpu")

        # Use the provided local runoff matrix
        # Shape: (n_cols, n_catch) - maps compressed runoff grids to catchments
        t_mapping = local_runoff_matrix.to(dev).to(torch_dtype)
        
        if normalized:
            # Normalize by row sums (each catchment's total area)
            # t_mapping shape: (n_cols, n_catch)
            # We need to normalize columns (each catchment)
            col_sums = torch.sparse.sum(t_mapping, dim=0).to_dense()  # (n_catch,)
            # Create a diagonal scaling matrix or normalize in-place
            # For COO tensor, we need to work with the values
            t_mapping = t_mapping.coalesce()
            indices = t_mapping.indices()  # (2, nnz)
            values = t_mapping.values()    # (nnz,)
            col_indices = indices[1]       # column index for each value
            col_sums_expanded = col_sums[col_indices]
            nz_mask = col_sums_expanded > 0
            new_values = torch.zeros_like(values)
            new_values[nz_mask] = values[nz_mask] / col_sums_expanded[nz_mask]
            t_mapping = torch.sparse_coo_tensor(
                indices, new_values, t_mapping.size(), dtype=torch_dtype, device=dev
            ).coalesce()
        
        # Pre-compute transposed mapping matrix for efficient batch multiplication
        # t_mapping shape: (n_cols, n_catch)
        # t_mapping_T shape: (n_catch, n_cols) for sparse.mm(sparse, dense)
        t_mapping_T = t_mapping.t().coalesce()
        
        dtype_nc = "f4" if dtype == "float32" else "f8"
        file_prefix = filename if filename else var_name
        
        def _init_nc(path):
            ds = nc.Dataset(path, "w", format="NETCDF4")
            ds.setncattr("title", f"Aggregated catchment runoff ({var_name})")
            ds.createDimension("time", None)
            ds.createDimension("saved_points", n_catch)

            time_var = ds.createVariable("time", "f8", ("time",))
            time_var.setncattr("units", "seconds since 1900-01-01 00:00:00")
            time_var.setncattr("calendar", "standard")

            save_coord = ds.createVariable("catchment_id", "i8", ("saved_points",))
            save_coord[:] = catchment_ids

            out_var = ds.createVariable(
                var_name,
                dtype_nc,
                ("time", "saved_points"),
                zlib=True,
                complevel=int(complevel),
            )
            desc = description if description else f"Catchment-aggregated {var_name} (area-weighted mean)"
            out_var.setncattr("description", desc)
            out_var.setncattr("units", units)
            return ds, time_var, out_var

        created_files = []
        ds = None
        time_var = None
        out_var = None
        current_year = None
        write_idx = 0
        total_steps = self.num_main_steps + self.num_spin_up_steps

        try:
            if not split_by_year:
                nc_path = out_dir / f"{file_prefix}_rank0.nc"
                ds, time_var, out_var = _init_nc(nc_path)
                created_files.append(nc_path)

            n_chunks = len(self)
            pbar = tqdm(total=total_steps, desc="Exporting", unit="step")
            for ci in range(n_chunks):
                base_idx = ci * self.chunk_len
                block = self.read_chunk(ci)
                if block.ndim != 2 or block.shape[1] != n_cols:
                    raise ValueError(
                        f"Data block shape {tuple(block.shape)} incompatible with mapping columns {n_cols} at chunk {ci}. "
                        "Please call build_local_runoff_matrix() before export_catchment_runoff()."
                    )
                T = int(block.shape[0])

                # Move current block to device and compute in batch:
                # t_mapping_T shape: (n_catch, n_cols) - sparse COO tensor (pre-computed)
                # block shape: (T, n_cols)
                # We want: block @ t_mapping -> (T, n_catch)
                # Since torch.sparse.mm expects (sparse, dense), we compute:
                # t_mapping_T @ block.T = (n_catch, n_cols) @ (n_cols, T) = (n_catch, T)
                # Then transpose to get (T, n_catch)
                block_tensor = torch.as_tensor(
                    block, dtype=torch_dtype, device=dev
                )  # shape: (T, n_cols)
                agg_block = torch.sparse.mm(t_mapping_T, block_tensor.T)  # (n_catch, T)
                agg_block_np = agg_block.T.contiguous().to("cpu").numpy()  # (T, n_catch)

                # Write each timestep in the block
                for k in range(T):
                    dt_k = self.get_time_by_index(base_idx + k)
                    
                    if split_by_year:
                        year = dt_k.year
                        if year != current_year:
                            if ds:
                                ds.close()
                            current_year = year
                            nc_path = out_dir / f"{file_prefix}_rank0_{year}.nc"
                            ds, time_var, out_var = _init_nc(nc_path)
                            created_files.append(nc_path)
                            write_idx = 0
                    
                    out_var[write_idx, :] = agg_block_np[k, :].astype(np.float32 if dtype == "float32" else np.float64, copy=False)
                    time_val = nc.date2num(dt_k, units=time_var.getncattr("units"), calendar=time_var.getncattr("calendar"))
                    time_var[write_idx] = time_val
                    write_idx += 1
                    pbar.update(1)
            
            pbar.close()
        finally:
            if ds:
                ds.close()

        return created_files if split_by_year else created_files[0]
    
    def generate_runoff_mapping_table(
        self,
        map_dir: str,
        out_dir: str,
        npz_file: str = "runoff_mapping.npz",
        mapinfo_txt: str = "location.txt",
        hires_map_tag: str = "1min",
        lowres_idx_precision: str = "<i4",
        hires_idx_precision: str = "<i2",
        map_precision: str = "<f4",
        parameter_nc: str | Path | None = None,
    ):
        """
        Generate runoff mapping table and save as npz file.
                The mapping is stored as a sparse matrix format with catchment IDs array.

                Optional alignment/subsetting:
                - If parameter_nc is provided, rows (catchments) in the sparse matrix will be
                    aligned to the 1D catchment list read from that NetCDF. The saved
                    'catchment_ids' array (in NPZ) will follow the order from parameter_nc.
        """
        
        map_dir = Path(map_dir)
        hires_map_dir = map_dir / hires_map_tag
        mapdim_path = map_dir / "mapdim.txt"
        
        with open(mapdim_path, "r") as f:
            lines = f.readlines()
            nx = int(lines[0].split('!!')[0].strip())
            ny = int(lines[1].split('!!')[0].strip())

        nextxy_path = map_dir / "nextxy.bin"
            
        nextxy_data = binread(
            nextxy_path,
            (nx, ny, 2),
            dtype_str=lowres_idx_precision
        )
        catchment_x, catchment_y = np.where(nextxy_data[:, :, 0] != -9999)
        catchment_id = np.ravel_multi_index((catchment_x, catchment_y), (nx, ny))

        # Load location info
        with open(hires_map_dir/ mapinfo_txt, "r") as f:
            lines = f.readlines()
        data = lines[2].split()
        Nx, Ny = int(data[6]), int(data[7])
        West, East = float(data[2]), float(data[3])
        South, North = float(data[4]), float(data[5])
        Csize = float(data[8])

        hires_lon = np.linspace(West + 0.5 * Csize, East - 0.5 * Csize, Nx)
        hires_lat = np.linspace(North - 0.5 * Csize, South + 0.5 * Csize, Ny)
        lon2D, lat2D = np.meshgrid(hires_lon, hires_lat)
        hires_lon_2D = lon2D.T
        hires_lat_2D = lat2D.T

        # Load high-resolution maps
        HighResGridArea = read_map(
            hires_map_dir / f"{hires_map_tag}.grdare.bin", (Nx, Ny), precision=map_precision
        ) * 1E6
        HighResCatchmentId = read_map(
            hires_map_dir / f"{hires_map_tag}.catmxy.bin", (Nx, Ny, 2), precision=hires_idx_precision
        )

        valid_mask = HighResCatchmentId[:, :, 0] > 0
        x_idx, y_idx = np.where(valid_mask)
        HighResCatchmentId -= 1  # convert from 1-based to 0-based
        valid_x = HighResCatchmentId[x_idx, y_idx, 0]
        valid_y = HighResCatchmentId[x_idx, y_idx, 1]
        valid_areas = HighResGridArea[x_idx, y_idx]
        catchment_id_hires = np.ravel_multi_index((valid_x, valid_y), (nx, ny))

        # Get runoff coordinates from dataset class
        ro_lon, ro_lat = self.get_coordinates()
        valid_lon = hires_lon_2D[x_idx, y_idx]
        valid_lat = hires_lat_2D[x_idx, y_idx]

        # Compute catchment and runoff IDs
        catchment_idx = find_indices_in(catchment_id_hires, catchment_id)
        if np.any(catchment_idx == -1):
            print(
                f"Warning: Some high-resolution catchments ({np.sum(catchment_idx == -1)}) were not found in the low-resolution map. "
                "Please check your mapping files or input data."
            )
        runoff_idx = compute_runoff_id(ro_lon, ro_lat, valid_lon, valid_lat)

        # Full grid - no masking needed
        full_grid_size = len(ro_lat) * len(ro_lon)

        # Filter valid mappings (only check catchment validity)
        valid_mask = catchment_idx != -1
        row_idx = catchment_idx[valid_mask]
        col_idx = runoff_idx[valid_mask]
        data_values = valid_areas[valid_mask]

        # Optionally align/subset catchments to parameter_nc order/region
        save_catchment_ids = catchment_id.astype(np.int64)
        if parameter_nc is not None:
            try:
                path_nc = Path(parameter_nc)
                with nc.Dataset(path_nc, "r") as ds:
                    if "catchment_id" in ds.variables:
                        desired_ids = np.asarray(ds.variables["catchment_id"][...]).astype(np.int64)
                    else:
                        raise KeyError("'catchment_id' not found in parameter_nc")
                # Map desired ids to current full catchment_id list
                desired_to_full = find_indices_in(desired_ids, catchment_id)
                keep_mask_nc = desired_to_full >= 0
                if not np.all(keep_mask_nc):
                    n_miss = int((~keep_mask_nc).sum())
                    print(
                        f"Warning: {n_miss} IDs from parameter_nc are not present in current map; they will be skipped."
                    )
                # Build remap: full-index -> aligned-row
                base_to_aligned = -np.ones(len(catchment_id), dtype=np.int64)
                valid_full_idx = desired_to_full[keep_mask_nc]
                base_to_aligned[valid_full_idx] = np.arange(valid_full_idx.size, dtype=np.int64)
                # Remap existing row indices to aligned rows and drop negatives
                aligned_row_idx = base_to_aligned[row_idx]
                keep_rows = aligned_row_idx >= 0
                row_idx = aligned_row_idx[keep_rows]
                col_idx = col_idx[keep_rows]
                data_values = data_values[keep_rows]
                # Update outputs
                save_catchment_ids = desired_ids[keep_mask_nc]
                matrix_shape = (save_catchment_ids.size, full_grid_size)
            except Exception as e:
                raise ValueError(
                    f"Failed to read or process parameter_nc: {path_nc}. "
                    "Ensure it contains 'catchment_id' variable."
                ) from e
        else:
            matrix_shape = (len(catchment_id), full_grid_size)

        # Report missing mapping rows (relative to selected set)
        unique_row_count = len(np.unique(row_idx))
        baseline_rows = matrix_shape[0]
        missing_count = baseline_rows - unique_row_count
        if missing_count > 0:
            print(
                f"Warning: {missing_count} catchments were not mapped to runoff grids. "
                "Their runoff input will always be zero."
            )

        # Create sparse matrix using scipy and compress it
        sparse_matrix = csr_matrix(
            (data_values.astype(np.float32), (row_idx, col_idx)), 
            shape=matrix_shape,
            dtype=np.float32
        )
        
        # Eliminate zeros and compress
        sparse_matrix.eliminate_zeros()
        
        # Prepare mapping data for saving with compressed sparse matrix
        # Include coordinate metadata for validation during loading
        mapping_data = {
            'catchment_ids': save_catchment_ids.astype(np.int64),
            'sparse_data': sparse_matrix.data.astype(np.float32),
            'sparse_indices': sparse_matrix.indices.astype(np.int64),
            'sparse_indptr': sparse_matrix.indptr.astype(np.int64),
            'matrix_shape': np.array(matrix_shape, dtype=np.int64),
            # Coordinate metadata for validation
            'coord_lon': ro_lon.astype(np.float64),
            'coord_lat': ro_lat.astype(np.float64),
        }

        output_path = Path(out_dir) / npz_file

        np.savez_compressed(output_path, **mapping_data)

        print(f"Saved runoff mapping to {output_path}")
        print(f"Mapping contains {matrix_shape[0]} catchments "
            f"and {len(sparse_matrix.data)} non-zero runoff grid mappings")
        print(f"Matrix shape: {matrix_shape[0]} x {matrix_shape[1]}")
        print(f"Coordinate metadata: lon[{len(ro_lon)}] x lat[{len(ro_lat)}]")

    @abstractmethod
    def get_data(self, current_time: datetime, chunk_len: int) -> np.ndarray:
        """
        Read a contiguous time block starting at current_time.

        Inputs:
        - current_time: start datetime aligned to the dataset time grid
        - chunk_len: positive integer upper bound of steps to read

        Returns:
        - If _local_runoff_indices is None (before build_local_runoff_matrix):
            3D numpy array with shape (T, Y, X) where Y=lat, X=lon
        - If _local_runoff_indices is set (after build_local_runoff_matrix):
            2D numpy array with shape (T, N) where N = number of active grids
        - T ∈ [1, chunk_len]. The final block near the end may have T < chunk_len.

        Spatial convention:
        - Dimension order: (lat, lon) i.e. (Y, X)
        - When flattening: C-order (row-major), lon varies fastest
        - Coordinate arrays from get_coordinates(): (lon, lat)

        Implementation notes:
        - Do not read beyond the available time range; truncate instead.
        - Do not pad to chunk_len here; AbstractDataset.__getitem__ will pad.
        - Preserve chronological order for the returned timesteps.
        """
        raise NotImplementedError

    @abstractmethod
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        To be implemented by subclasses, returns the coordinates of the dataset.
        """
        raise NotImplementedError

    def get_time_by_index(self, idx: int) -> Union[datetime, cftime.datetime]:
        """
        Returns the datetime corresponding to the given index.
        Default implementation handles spin-up and linear time stepping.
        """
        if self.time_interval is None:
             raise NotImplementedError("time_interval must be provided for default get_time_by_index")

        if self.spin_up_cycles > 0:
            if self.spin_up_start_date is None or self.spin_up_end_date is None:
                 raise ValueError("Spin-up dates must be provided")
            
            # Calculate items (including padding) in one spin-up cycle
            chunks_per_cycle = self.num_spin_up_chunks // self.spin_up_cycles
            items_per_cycle = chunks_per_cycle * self.chunk_len
            
            total_spin_up_items = items_per_cycle * self.spin_up_cycles
            
            if idx < total_spin_up_items:
                # In spin-up
                # cycle_idx is which repetition of spin-up we are in
                # idx_in_cycle is the index within that repetition (including padding)
                idx_in_cycle = idx % items_per_cycle
                
                return self.spin_up_start_date + self.time_interval * idx_in_cycle
            
            # Main simulation
            idx -= total_spin_up_items

        if self.start_date is None:
             raise AttributeError("Dataset must have 'start_date'")

        return self.start_date + self.time_interval * idx

    def get_index_by_time(self, dt: Union[datetime, cftime.datetime]) -> int:
        """Returns the index in the main simulation timeline for a given datetime."""
        if self.start_date is None or self.time_interval is None:
             raise ValueError("start_date and time_interval required")
        
        offset = dt - self.start_date
        return int(offset.total_seconds() / self.time_interval.total_seconds())

    @property
    def num_main_steps(self) -> int:
        if self.start_date is None or self.end_date is None or self.time_interval is None:
            return 0
        duration = self.end_date - self.start_date
        return int(duration.total_seconds() / self.time_interval.total_seconds()) + 1

    @property
    def num_spin_up_steps(self) -> int:
        if self.spin_up_cycles <= 0:
            return 0
        cycle_duration = self.spin_up_end_date - self.spin_up_start_date
        steps_per_cycle = int(cycle_duration.total_seconds() / self.time_interval.total_seconds()) + 1
        return steps_per_cycle * self.spin_up_cycles

    @property
    def total_steps(self) -> int:
        return self.num_spin_up_steps + self.num_main_steps

    def is_valid_time_index(self, idx: int) -> bool:
        """
        Checks if the given time index corresponds to a valid data step.
        Handles padding gaps in spin-up and main simulation.
        """
        if idx < 0:
            return False

        if self.spin_up_cycles > 0:
            if not hasattr(self, '_spin_up_num_chunks'):
                 self._calc_spin_up_params()
            
            chunks_per_cycle = self._spin_up_num_chunks
            items_per_cycle = chunks_per_cycle * self.chunk_len
            total_spin_up_items = items_per_cycle * self.spin_up_cycles
            
            if idx < total_spin_up_items:
                # In spin-up region
                idx_in_cycle = idx % items_per_cycle
                
                # Calculate valid steps per cycle
                cycle_duration = self.spin_up_end_date - self.spin_up_start_date
                steps_per_cycle = int(cycle_duration.total_seconds() / self.time_interval.total_seconds()) + 1
                
                return idx_in_cycle < steps_per_cycle
            
            # Move to main simulation region
            idx -= total_spin_up_items

        # Main simulation region
        return idx < self.num_main_steps

    def _real_len(self) -> int:
        """Number of chunks in main simulation."""
        total = self.num_main_steps
        return (total + self.chunk_len - 1) // self.chunk_len

    def validate_files_in_range(self, file_path_gen: Callable[[datetime], Path]) -> None:
        """
        Validates that files exist for all time steps in the simulation, including spin-up.
        file_path_gen: function that takes a datetime and returns a Path to the file.
        """
        if self.time_interval is None:
             raise ValueError("time_interval must be provided for validation")

        file_paths = set()
        
        # Main simulation
        if self.start_date and self.end_date:
            curr = self.start_date
            while curr <= self.end_date:
                file_paths.add(file_path_gen(curr))
                curr += self.time_interval

        # Spin-up
        if self.spin_up_cycles > 0:
            if self.spin_up_start_date and self.spin_up_end_date:
                curr = self.spin_up_start_date
                while curr <= self.spin_up_end_date:
                    file_paths.add(file_path_gen(curr))
                    curr += self.time_interval
        
        self.validate_files_exist(list(file_paths))

    @abstractmethod
    def close(self) -> None:
        """
        Closes any open resources or files.
        """
        pass

    @property
    def data_size(self) -> int:
        """
        Returns the number of runoff grid points to be loaded per timestep.
        
        Before build_local_runoff_matrix is called: returns full grid size.
        After build_local_runoff_matrix is called: returns compressed size (active grids only).
        """
        if self._local_runoff_indices is not None:
            return len(self._local_runoff_indices)
        # Full grid size from coordinates
        lon, lat = self.get_coordinates()
        return len(lon) * len(lat)

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """
        Returns (ny, nx) = (lat_size, lon_size) grid dimensions.
        
        Spatial convention: (Y, X) = (lat, lon)
        """
        lon, lat = self.get_coordinates()
        return (len(lat), len(lon))

    def _combine(self, other, operation, reverse=False):
        is_dataset = isinstance(other, AbstractDataset)
        is_scalar = isinstance(other, (int, float, np.number))
        
        if not (is_dataset or is_scalar):
            return NotImplemented

        operands = [self, other] if not reverse else [other, self]
        return MixedDataset(operands, operation=operation)

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Fetch one chunk (T <= chunk_len) starting at chunk index `idx`.
        
        Returns:
        - If build_local_runoff_matrix has been called: (chunk_len, N) compressed data
        - Otherwise: (chunk_len, Y, X) full grid data
        
        Pads with zeros if the actual data is shorter than chunk_len.
        """
        # Compute absolute start index for this chunk
        if idx < 0:
            idx += len(self)
        
        compressed = self._local_runoff_indices is not None
        
        # Non-rank-0: return zeros to keep shapes consistent across ranks
        if not is_rank_zero():
            if compressed:
                return np.empty((self.chunk_len, self.data_size), dtype=self.out_dtype)
            else:
                ny, nx = self.grid_shape
                return np.empty((self.chunk_len, ny, nx), dtype=self.out_dtype)

        # Rank-0: fetch data
        data = self.read_chunk(idx)
        
        if compressed:
            # Expect (T, N)
            N = self.data_size
            if data.ndim != 2 or data.shape[1] != N:
                raise ValueError(
                    f"read_chunk returned shape {tuple(data.shape)}, expected (T, {N})"
                )
            T = data.shape[0]
            if T < self.chunk_len:
                pad = np.zeros((self.chunk_len - T, N), dtype=self.out_dtype)
                data = np.vstack([data, pad]) if data.size else pad
        else:
            # Expect (T, Y, X)
            ny, nx = self.grid_shape
            if data.ndim != 3 or data.shape[1] != ny or data.shape[2] != nx:
                raise ValueError(
                    f"read_chunk returned shape {tuple(data.shape)}, expected (T, {ny}, {nx})"
                )
            T = data.shape[0]
            if T < self.chunk_len:
                pad = np.zeros((self.chunk_len - T, ny, nx), dtype=self.out_dtype)
                data = np.vstack([data, pad]) if data.size else pad
        
        # Clip negative values to zero if enabled
        if self.clip_negative:
            np.maximum(data, 0, out=data)
            
        return np.ascontiguousarray(data)

    def __len__(self) -> int:
        real_len = self._real_len()
        return real_len + self.num_spin_up_chunks

    def __add__(self, other):
        return self._combine(other, "add")
    
    def __radd__(self, other):
        return self._combine(other, "add", reverse=True)

    def __sub__(self, other):
        return self._combine(other, "sub")
    
    def __rsub__(self, other):
        return self._combine(other, "sub", reverse=True)

    def __mul__(self, other):
        return self._combine(other, "mul")
    
    def __rmul__(self, other):
        return self._combine(other, "mul", reverse=True)

    def __truediv__(self, other):
        return self._combine(other, "div")
    
    def __rtruediv__(self, other):
        return self._combine(other, "div", reverse=True)


class MixedDataset(AbstractDataset):
    """
    A dataset that combines multiple datasets (or scalars) by applying an operation.
    """
    def __init__(self, operands: List[Union[AbstractDataset, float, int]], operation: str = "add"):
        if not operands:
            raise ValueError("operands list cannot be empty")
        
        base = None
        for op in operands:
            if isinstance(op, AbstractDataset):
                base = op
                break
        
        if base is None:
            raise ValueError("MixedDataset requires at least one AbstractDataset operand")
        
        self.base_dataset = base
        self.operands = []
        
        can_flatten = operation in ["add", "mul"]
        
        for op in operands:
            if can_flatten and isinstance(op, MixedDataset) and op.operation == operation:
                self.operands.extend(op.operands)
            else:
                self.operands.append(op)

        for i, op in enumerate(self.operands):
            if isinstance(op, AbstractDataset) and op is not base:
                if op.start_date != base.start_date:
                    raise ValueError(f"Operand {i} has different start_date")
                if op.end_date != base.end_date:
                    raise ValueError(f"Operand {i} has different end_date")
                if op.time_interval != base.time_interval:
                    raise ValueError(f"Operand {i} has different time_interval")
                if op.chunk_len != base.chunk_len:
                    raise ValueError(f"Operand {i} has different chunk_len")
                if op.data_size != base.data_size:
                    raise ValueError(f"Operand {i} has different data_size")

        # Initialize AbstractDataset using the base dataset's attributes
        super().__init__(
            start_date=base.start_date,
            end_date=base.end_date,
            time_interval=base.time_interval,
            out_dtype=base.out_dtype,
            chunk_len=base.chunk_len,
            spin_up_cycles=base.spin_up_cycles,
            spin_up_start_date=base.spin_up_start_date,
            spin_up_end_date=base.spin_up_end_date,
            calendar=base.calendar,
            clip_negative=base.clip_negative,
        )
        self.operation = operation

    def get_data(self, current_time: datetime, chunk_len: int) -> np.ndarray:
        def _fetch(op):
            if isinstance(op, AbstractDataset):
                return op.get_data(current_time, chunk_len)
            return op

        data = _fetch(self.operands[0])
        
        for op in self.operands[1:]:
            val = _fetch(op)
            if self.operation == "add":
                data = data + val
            elif self.operation == "sub":
                data = data - val
            elif self.operation == "mul":
                data = data * val
            elif self.operation == "div":
                data = data / val
            else:
                raise NotImplementedError(f"Operation {self.operation} not implemented")
        
        return data

    @property
    def data_mask(self) -> np.ndarray:
        return self.base_dataset.data_mask

    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.base_dataset.get_coordinates()

    def close(self) -> None:
        for op in self.operands:
            if hasattr(op, 'close'):
                op.close()

class StaticParameterDataset:
    """
    A dataset class for static or climatological parameter files (NetCDF).
    Does not inherit from torch.utils.data.Dataset.
    Supports generating mapping tables and exporting remapped data.
    """

    def __init__(
        self,
        nc_path: Union[str, Path],
        var_name: str,
        lat_name: str = "lat",
        lon_name: str = "lon",
    ):
        self.nc_path = Path(nc_path)
        self.var_name = var_name
        self.lat_name = lat_name
        self.lon_name = lon_name

        if not self.nc_path.exists():
            raise FileNotFoundError(f"File not found: {self.nc_path}")

        with nc.Dataset(self.nc_path, "r") as ds:
            if self.var_name not in ds.variables:
                raise ValueError(f"Variable {self.var_name} not found in {self.nc_path}")
            
            self.lat = ds.variables[self.lat_name][:]
            self.lon = ds.variables[self.lon_name][:]
            var = ds.variables[self.var_name]
            self.shape = var.shape
            self.ndim = var.ndim
            
            # Determine if it has a time/month dimension
            # Assuming (lat, lon) or (time, lat, lon)
            if self.ndim == 2:
                self.has_time = False
                self._len = 1
            elif self.ndim == 3:
                self.has_time = True
                self._len = self.shape[0]
            else:
                raise ValueError(f"Unsupported dimensions for variable {self.var_name}: {self.shape}")

    @property
    def data_mask(self) -> np.ndarray:
        """Returns a full mask (all True) for mapping table generation."""
        return np.ones((len(self.lat), len(self.lon)), dtype=bool)

    @property
    def data_size(self) -> int:
        return len(self.lat) * len(self.lon)

    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.lon, self.lat

    def read_chunk(self, idx: int) -> np.ndarray:
        """
        Reads data. For static data (ndim=2), idx is ignored (returns the single map).
        For climatology (ndim=3), returns the data at time index idx.
        Returns shape (1, lat*lon).
        """
        with nc.Dataset(self.nc_path, "r") as ds:
            var = ds.variables[self.var_name]
            if not self.has_time:
                data = var[:]
            else:
                data = var[idx]
            
            # Handle MaskedArray
            if isinstance(data, np.ma.MaskedArray):
                data = data.filled(np.nan)
            
            # Flatten to (N,) and return as (1, N)
            return data.flatten().reshape(1, -1)

    def generate_runoff_mapping_table(
        self,
        map_dir: str,
        out_dir: str,
        npz_file: str = "runoff_mapping.npz",
        mapinfo_txt: str = "location.txt",
        hires_map_tag: str = "1min",
        lowres_idx_precision: str = "<i4",
        hires_idx_precision: str = "<i2",
        map_precision: str = "<f4",
        parameter_nc: Union[str, Path, None] = None,
    ):
        """
        Generate runoff mapping table and save as npz file.
        Copied and adapted from AbstractDataset.
        """
        map_dir = Path(map_dir)
        hires_map_dir = map_dir / hires_map_tag
        mapdim_path = map_dir / "mapdim.txt"
        
        with open(mapdim_path, "r") as f:
            lines = f.readlines()
            nx = int(lines[0].split('!!')[0].strip())
            ny = int(lines[1].split('!!')[0].strip())

        nextxy_path = map_dir / "nextxy.bin"
            
        nextxy_data = binread(
            nextxy_path,
            (nx, ny, 2),
            dtype_str=lowres_idx_precision
        )
        catchment_x, catchment_y = np.where(nextxy_data[:, :, 0] != -9999)
        catchment_id = np.ravel_multi_index((catchment_x, catchment_y), (nx, ny))

        # Load location info
        with open(hires_map_dir/ mapinfo_txt, "r") as f:
            lines = f.readlines()
        data = lines[2].split()
        Nx, Ny = int(data[6]), int(data[7])
        West, East = float(data[2]), float(data[3])
        South, North = float(data[4]), float(data[5])
        Csize = float(data[8])

        hires_lon = np.linspace(West + 0.5 * Csize, East - 0.5 * Csize, Nx)
        hires_lat = np.linspace(North - 0.5 * Csize, South + 0.5 * Csize, Ny)

        # Load high-resolution maps
        HighResGridArea = read_map(
            hires_map_dir / f"{hires_map_tag}.grdare.bin", (Nx, Ny), precision=map_precision
        ) * 1E6
        HighResCatchmentId = read_map(
            hires_map_dir / f"{hires_map_tag}.catmxy.bin", (Nx, Ny, 2), precision=hires_idx_precision
        )

        valid_mask = HighResCatchmentId[:, :, 0] > 0
        x_idx, y_idx = np.where(valid_mask)
        HighResCatchmentId -= 1  # convert from 1-based to 0-based
        valid_x = HighResCatchmentId[x_idx, y_idx, 0]
        valid_y = HighResCatchmentId[x_idx, y_idx, 1]
        valid_areas = HighResGridArea[x_idx, y_idx]
        catchment_id_hires = np.ravel_multi_index((valid_x, valid_y), (nx, ny))

        # Get runoff coordinates
        ro_lon, ro_lat = self.get_coordinates()
        valid_lon = hires_lon[x_idx]
        valid_lat = hires_lat[y_idx]

        # Compute catchment and runoff IDs
        catchment_idx = find_indices_in(catchment_id_hires, catchment_id)
        if np.any(catchment_idx == -1):
            print(
                f"Warning: Some high-resolution catchments ({np.sum(catchment_idx == -1)}) were not found in the low-resolution map."
            )
        runoff_idx = compute_runoff_id(ro_lon, ro_lat, valid_lon, valid_lat)

        # Handle mask if available
        col_mask = self.data_mask
        if col_mask is not None:
            col_mask = np.ravel(col_mask, order="C")
        else:
            col_mask = np.ones((len(ro_lat) * len(ro_lon)), dtype=bool)

        col_mapping = -np.ones_like(col_mask, dtype=np.int64)
        col_mapping[np.flatnonzero(col_mask)] = np.arange(col_mask.sum())
        mapped_runoff_idx = col_mapping[runoff_idx]

        # Filter valid mappings
        valid_mask = (catchment_idx != -1) & (mapped_runoff_idx != -1)
        row_idx = catchment_idx[valid_mask]
        col_idx = mapped_runoff_idx[valid_mask]
        data_values = valid_areas[valid_mask]

        # Optionally align/subset catchments to parameter_nc order/region
        save_catchment_ids = catchment_id.astype(np.int64)
        matrix_shape = (len(catchment_id), int(col_mask.sum()))

        if parameter_nc is not None:
            try:
                with nc.Dataset(parameter_nc, "r") as pnc:
                    param_cx = pnc.variables["catchment_x"][:]
                    param_cy = pnc.variables["catchment_y"][:]
                    param_cids = np.ravel_multi_index((param_cx, param_cy), (nx, ny))
                
                # First, get the actual catchment IDs for the sparse entries
                actual_cids = catchment_id[row_idx]
                
                # Now find where these actual_cids are in param_cids
                new_row_idx = find_indices_in(actual_cids, param_cids)
                
                # Filter out those not in param_cids
                keep = new_row_idx != -1
                row_idx = new_row_idx[keep]
                col_idx = col_idx[keep]
                data_values = data_values[keep]
                
                save_catchment_ids = param_cids.astype(np.int64)
                matrix_shape = (len(param_cids), int(col_mask.sum()))
                
            except Exception as e:
                print(f"Warning: Failed to align with parameter_nc: {e}")
                print("Proceeding with default catchment list from nextxy.")

        # Create sparse matrix using scipy and compress it
        sparse_matrix = csr_matrix(
            (data_values.astype(np.float32), (row_idx, col_idx)), 
            shape=matrix_shape,
            dtype=np.float32
        )
        
        # Eliminate zeros and compress
        sparse_matrix.eliminate_zeros()
        
        # Prepare mapping data for saving with compressed sparse matrix
        mapping_data = {
            'catchment_ids': save_catchment_ids.astype(np.int64),
            'sparse_data': sparse_matrix.data.astype(np.float32),
            'sparse_indices': sparse_matrix.indices.astype(np.int64),
            'sparse_indptr': sparse_matrix.indptr.astype(np.int64),
            'matrix_shape': np.array(matrix_shape, dtype=np.int64)
        }

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / npz_file

        np.savez_compressed(output_path, **mapping_data)

        print(f"Saved runoff mapping to {output_path}")
        print(f"Mapping contains {matrix_shape[0]} catchments "
            f"and {len(sparse_matrix.data)} non-zero runoff grid mappings")
        print(f"Matrix shape: {matrix_shape[0]} x {matrix_shape[1]}")

    def export_catchment_runoff(
        self,
        out_dir: Union[str, Path],
        mapping_npz: Union[str, Path],
        var_name: Optional[str] = None,
        dtype: Literal["float32", "float64"] = "float32",
        complevel: int = 4,
        normalized: bool = False,
        device: Union[str, torch.device] = "cpu",
        description: Optional[str] = None,
        units: str = "mm",
    ) -> Path:
        """
        Export catchment-aggregated data to a NetCDF file.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        mapping_npz = Path(mapping_npz)
        
        if var_name is None:
            var_name = self.var_name

        if not mapping_npz.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_npz}")

        # Load mapping (SciPy CSR)
        m = np.load(mapping_npz)
        catchment_ids = m["catchment_ids"].astype(np.int64)
        sparse_data = m["sparse_data"].astype(np.float32 if dtype == "float32" else np.float64)
        sparse_indices = m["sparse_indices"].astype(np.int64)
        sparse_indptr = m["sparse_indptr"].astype(np.int64)
        mat_shape = tuple(np.array(m["matrix_shape"]).tolist())
        mapping = csr_matrix((sparse_data, sparse_indices, sparse_indptr), shape=mat_shape)

        n_catch = int(catchment_ids.shape[0])
        n_cols = int(mat_shape[1])
        
        # Check data size
        # Note: self.data_size is total pixels. If mask is used, n_cols should match mask.sum()
        
        # Prepare device and torch types
        torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if dev.type == "cuda" and not torch.cuda.is_available():
            print("CUDA not available; falling back to CPU.")
            dev = torch.device("cpu")

        # Build torch sparse CSR once on the target device
        crow = torch.from_numpy(mapping.indptr.astype(np.int64))
        ccol = torch.from_numpy(mapping.indices.astype(np.int64))
        cval = torch.from_numpy(mapping.data.astype(np.float32 if dtype == "float32" else np.float64))
        
        if normalized:
            row_lengths = crow[1:] - crow[:-1]
            row_ids = torch.repeat_interleave(
                torch.arange(n_catch, device=dev),
                row_lengths
            )
            row_sums = torch.zeros(n_catch, dtype=torch_dtype, device=dev)
            row_sums.scatter_add_(0, row_ids, cval)
            denom = row_sums[row_ids]
            nz_mask = denom > 0
            cval_new = torch.zeros_like(cval)
            cval_new[nz_mask] = cval[nz_mask] / denom[nz_mask]
            cval = cval_new

        t_mapping = torch.sparse_csr_tensor(
            crow, ccol, cval, size=(n_catch, n_cols), dtype=torch_dtype, device=dev
        )
        
        dtype_nc = "f4" if dtype == "float32" else "f8"
        
        nc_path = out_dir / f"{var_name}_rank0.nc"
        
        with nc.Dataset(nc_path, "w", format="NETCDF4") as ds:
            ds.setncattr("title", f"Aggregated catchment parameter ({var_name})")
            if self.has_time:
                ds.createDimension("time", None)
            ds.createDimension("saved_points", n_catch)

            if self.has_time:
                time_var = ds.createVariable("time", "f8", ("time",))
                time_var.setncattr("units", "months" if self.ndim==3 else "unknown") # Simplified
            
            save_coord = ds.createVariable("catchment_id", "i8", ("saved_points",))
            save_coord[:] = catchment_ids

            dims = ("time", "saved_points") if self.has_time else ("saved_points",)
            out_var = ds.createVariable(
                var_name,
                dtype_nc,
                dims,
                zlib=True,
                complevel=int(complevel),
            )
            desc = description if description else f"Catchment-aggregated {var_name}"
            out_var.setncattr("description", desc)
            out_var.setncattr("units", units)

            # Process data
            # If static, just one chunk (idx=0)
            # If monthly, loop over len(self)
            
            for idx in range(len(self)):
                block = self.read_chunk(idx) # (1, N)
                
                if block.shape[1] != n_cols:
                     raise ValueError(
                        f"Data block shape {tuple(block.shape)} incompatible with mapping columns {n_cols}."
                    )
                
                # block is (1, N). mapping is (C, N).
                # We want (C, 1).
                # mapping @ block.T -> (C, 1)
                
                block_t = torch.as_tensor(block, dtype=torch_dtype, device=dev).T
                agg_block = torch.sparse.mm(t_mapping, block_t) # (C, 1)
                agg_block_np = agg_block.T.contiguous().to("cpu").numpy() # (1, C)
                
                if self.has_time:
                    out_var[idx, :] = agg_block_np[0, :]
                    # time_var[idx] = idx + 1 # Dummy time
                else:
                    out_var[:] = agg_block_np[0, :]

        return nc_path

    def __len__(self) -> int:
        return self._len
