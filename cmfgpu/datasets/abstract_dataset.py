# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

import os
from abc import ABC, abstractmethod
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Literal, Tuple

import netCDF4 as nc
import numpy as np
import torch
import torch.distributed as dist
from scipy.sparse import csr_matrix

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
    def __init__(self, out_dtype: str = "float32", chunk_len: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_dtype = out_dtype
        self.chunk_len = chunk_len

    def shard_forcing(
        self,
        batch_runoff: torch.Tensor,
        local_runoff_matrix: torch.Tensor,
        local_runoff_indices: torch.Tensor,
        world_size: int,
    ) -> torch.Tensor:
        """
        Map grid runoff to catchments and handle distributed sync.

        Expected input shape: (B, T, N). We'll flatten the first two
        dimensions into a single time-like dimension before mapping to catchments.
        Output shape: ((B*T), C) where C = number of catchments mapped.
        """
        if batch_runoff.dim() == 3:
            B, T, N = batch_runoff.shape
            flat = batch_runoff.reshape(B * T, N)
        else:
            raise ValueError(f"batch_runoff must be 3D, got shape {tuple(batch_runoff.shape)}")

        if world_size > 1:
            dist.broadcast(flat, src=0)

        out = (flat[:, local_runoff_indices] @ local_runoff_matrix).contiguous()
        return out

    def build_local_runoff_matrix(self, runoff_mapping_file: str, desired_catchment_ids: np.ndarray, 
                                precision: Literal["float32", "float64"], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build PyTorch CSR matrix for mapping runoff data to specified catchments.
        Loads scipy compressed sparse matrix data and converts to PyTorch.
        The output catchment order will match the order of desired_catchment_ids.
        """
        runoff_mapping_file = Path(runoff_mapping_file)
        precision = torch.float32 if precision == "float32" else torch.float64
        if runoff_mapping_file is None or not os.path.exists(runoff_mapping_file):
            raise ValueError("Runoff mapping file not found. Cannot build local matrix.")
        
        # Load the scipy compressed sparse matrix data
        mapping_data = np.load(runoff_mapping_file)
        
        all_catchment_ids = mapping_data['catchment_ids']
        sparse_data = mapping_data['sparse_data']
        sparse_indices = mapping_data['sparse_indices'] 
        sparse_indptr = mapping_data['sparse_indptr']
        matrix_shape = mapping_data['matrix_shape']
        
        full_sparse_matrix = csr_matrix(
            (sparse_data, sparse_indices, sparse_indptr),
            shape=tuple(matrix_shape)
        )
        
        # Use find_indices_in to get row indices for desired catchments
        desired_row_indices = find_indices_in(desired_catchment_ids, all_catchment_ids)
        
        # Check which catchments were found
        valid_idx = desired_row_indices != -1
        
        if np.any(valid_idx == -1):
            raise ValueError(
                f"Some desired catchments ({np.sum(~valid_idx)}) were not found in the mapping file. "
                "Please check your input data or mapping file."
            )
        
        # Extract submatrix for desired catchments only
        submatrix = full_sparse_matrix[desired_row_indices, :]
        
        # Remove columns that are all zeros to optimize memory
        col_sums = np.array(submatrix.sum(axis=0)).flatten()
        non_zero_cols = np.where(col_sums != 0)[0]
        
        if len(non_zero_cols) == 0:
            raise ValueError("No non-zero runoff data found for the desired catchments.")
        
        # Extract final submatrix with only non-zero columns
        final_submatrix = submatrix[:, non_zero_cols].T.tocoo()
        
        # Store the information needed for data extraction
        local_runoff_indices = torch.tensor(non_zero_cols, dtype=torch.int64,device=device)

        # Convert to PyTorch tensors
        row_tensor = torch.from_numpy(final_submatrix.row.astype(np.int64)).to(device)
        col_tensor = torch.from_numpy(final_submatrix.col.astype(np.int64)).to(device)
        data_tensor = torch.from_numpy(final_submatrix.data.astype(np.float32)).to(device).to(precision)

        # Create PyTorch sparse CSR tensor
        indices = torch.stack([row_tensor, col_tensor])
        local_runoff_matrix = torch.sparse_coo_tensor(
            indices, data_tensor, 
            size=(len(non_zero_cols), len(desired_catchment_ids)),
            dtype=precision,
            device=device
        ).coalesce()
        
        print(f"Built local runoff matrix for {len(desired_catchment_ids)} catchments "
            f"and {len(non_zero_cols)} runoff grids on device {device}")
        print(f"Original matrix shape: {matrix_shape}, Local matrix shape: {local_runoff_matrix.shape}")
        print(f"Found {np.sum(valid_idx)} out of {len(desired_catchment_ids)} requested catchments")
        return local_runoff_matrix, local_runoff_indices

    def export_catchment_runoff(
        self,
        out_dir: str | Path,
        mapping_npz: str | Path,
        parameter_nc: str | Path,
        var_name: str = "runoff",
        dtype: Literal["float32", "float64"] = "float32",
        complevel: int = 4,
    ) -> Path:
        """
        Export catchment-aggregated runoff to a NetCDF file readable by MultiRankStatsReader.

        - Output filename: {var_name}_rank0.nc
        - Dimensions: time (unlimited), saved_points
        - Variables:
            * time: numeric with units and calendar
            * save_coord: (saved_points,) linear catchment IDs (compatible with nx, ny)
            * {var_name}: (time, saved_points) aggregated runoff (area-weighted mean in mm)
        - Inputs: sparse mapping NPZ (CSR matrix + catchment_ids) and a parameter NetCDF (to copy nx/ny attrs).

        Time range: uses the dataset's inherent length (e.g., defined by its __len__), no extra arguments.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        mapping_npz = Path(mapping_npz)
        parameter_nc = Path(parameter_nc)

        if not mapping_npz.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_npz}")

        m = np.load(mapping_npz)
        catchment_ids = m["catchment_ids"].astype(np.int64)
        sparse_data = m["sparse_data"].astype(np.float32 if dtype == "float32" else np.float64)
        sparse_indices = m["sparse_indices"].astype(np.int64)
        sparse_indptr = m["sparse_indptr"].astype(np.int64)
        mat_shape = tuple(np.array(m["matrix_shape"]).tolist())
        mapping = csr_matrix((sparse_data, sparse_indices, sparse_indptr), shape=mat_shape)

        # Normalize rows by total area to produce area-weighted mean (weights sum to 1 per catchment)
        mapping = mapping.tocsr(copy=True)
        row_sums = np.asarray(mapping.sum(axis=1)).ravel()
        # Avoid division by zero; rows with zero sum remain zeros
        for i in range(mapping.shape[0]):
            start, end = mapping.indptr[i], mapping.indptr[i + 1]
            s = row_sums[i]
            if s > 0:
                mapping.data[start:end] /= s

        n_catch = int(catchment_ids.shape[0])
        n_cols = int(mat_shape[1])
        if n_cols != self.data_size:
            raise ValueError(
                f"Mapping columns ({n_cols}) != dataset data_size ({self.data_size})."
            )

        nc_path = out_dir / f"{var_name}_rank0.nc"
        dtype_nc = "f4" if dtype == "float32" else "f8"
        with nc.Dataset(nc_path, "w", format="NETCDF4") as ds:

            try:
                with nc.Dataset(parameter_nc, "r") as pds:
                    if "nx" in pds.ncattrs():
                        ds.setncattr("nx", int(pds.getncattr("nx")))
                    if "ny" in pds.ncattrs():
                        ds.setncattr("ny", int(pds.getncattr("ny")))
                    dims = list(pds.dimensions.keys())
                    for a, b in [("nx", "ny"), ("x", "y"), ("lon", "lat")]:
                        if a in dims and b in dims and ("nx" not in ds.ncattrs() or "ny" not in ds.ncattrs()):
                            ds.setncattr("nx", int(pds.dimensions[a].size))
                            ds.setncattr("ny", int(pds.dimensions[b].size))
                            break
            except Exception:
                pass

            ds.setncattr("title", f"Aggregated catchment runoff ({var_name})")
            ds.createDimension("time", None)
            ds.createDimension("saved_points", n_catch)

            time_var = ds.createVariable("time", "f8", ("time",))
            time_var.setncattr("units", "seconds since 1900-01-01 00:00:00")
            time_var.setncattr("calendar", "standard")

            save_coord = ds.createVariable("save_coord", "i8", ("saved_points",))
            save_coord[:] = catchment_ids

            out_var = ds.createVariable(
                var_name,
                dtype_nc,
                ("time", "saved_points"),
                zlib=True,
                complevel=int(complevel),
            )
            out_var.setncattr("description", f"Catchment-aggregated {var_name} (area-weighted mean)")
            out_var.setncattr("units", "mm")

            write_idx = 0
            n_chunks = len(self)
            for ci in range(n_chunks):
                base_idx = ci * self.chunk_len
                dt0 = self.get_time_by_index(base_idx)
                block = self.get_data(dt0, length=self.chunk_len)
                if block.ndim != 2 or block.shape[1] != n_cols:
                    raise ValueError(
                        f"Data block shape {tuple(block.shape)} incompatible with mapping columns {n_cols} at chunk {ci}."
                    )
                T = int(block.shape[0])
                # Write each timestep in the block
                for k in range(T):
                    dt_k = self.get_time_by_index(base_idx + k)
                    vec = block[k].astype(np.float32 if dtype == "float32" else np.float64, copy=False)
                    agg = mapping @ vec
                    out_var[write_idx, :] = np.asarray(agg, dtype=np.float32 if dtype == "float32" else np.float64)
                    time_val = nc.date2num(dt_k, units=time_var.getncattr("units"), calendar=time_var.getncattr("calendar"))
                    time_var[write_idx] = time_val
                    write_idx += 1

        return nc_path
    
    def generate_runoff_mapping_table(
        self,
        map_dir: str,
        out_dir: str,
        npz_file: str = "runoff_mapping.npz",
        mapinfo_txt: str = "location.txt",
        hires_map_tag: str = "1min",
        lowres_idx_precision: str = "<i4",
        hires_idx_precision: str = "<i2",
        map_precision: str = "<f4"
    ):
        """
        Generate runoff mapping table and save as npz file.
        The mapping is stored as a sparse matrix format with catchment IDs array.
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
        unique_row_count = len(np.unique(row_idx))
        missing_count = len(catchment_id) - unique_row_count
        if missing_count > 0:
            print(f"Warning: {missing_count} catchments were not mapped to runoff grids. "
                "Their runoff input will always be zero.")
        # Create sparse matrix using scipy and compress it
        matrix_shape = (len(catchment_id), col_mask.sum())
        sparse_matrix = csr_matrix(
            (data_values.astype(np.float32), (row_idx, col_idx)), 
            shape=matrix_shape,
            dtype=np.float32
        )
        
        # Eliminate zeros and compress
        sparse_matrix.eliminate_zeros()
        
        # Prepare mapping data for saving with compressed sparse matrix
        mapping_data = {
            'catchment_ids': catchment_id.astype(np.int64),
            'sparse_data': sparse_matrix.data.astype(np.float32),
            'sparse_indices': sparse_matrix.indices.astype(np.int64),
            'sparse_indptr': sparse_matrix.indptr.astype(np.int64),
            'matrix_shape': np.array(matrix_shape, dtype=np.int64)
        }

        output_path = Path(out_dir) / npz_file
        
        np.savez_compressed(output_path, **mapping_data)
        
        print(f"Saved runoff mapping to {output_path}")
        print(f"Mapping contains {len(catchment_id)} catchments "
            f"and {len(sparse_matrix.data)} non-zero runoff grid mappings")
        print(f"Matrix shape: {matrix_shape[0]} x {matrix_shape[1]}")

    @abstractmethod
    def get_data(self, current_time: datetime, chunk_len: int) -> np.ndarray:
        """
        Read a contiguous time block starting at current_time.

        Inputs:
        - current_time: start datetime aligned to dataset time grid
        - chunk_len: positive integer number of steps to read

        Returns: 2D numpy array with shape (T, N), where T == chunk_len and
        N equals the number of valid spatial points (sum of data_mask).
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def data_mask(self) -> np.ndarray:
        """
        Returns the mask of the dataset.
        """
        pass

    @abstractmethod
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        To be implemented by subclasses, returns the coordinates of the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def get_time_by_index(self, idx: int) -> datetime:
        """
        Returns the datetime corresponding to the given index.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        Closes any open resources or files.
        """
        pass
    
    @cached_property
    def data_size(self) -> int:
        return int(self.data_mask.sum())

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Fetch one chunk (T <= chunk_len) starting at chunk index `idx` and pad to (chunk_len, N).

        Behavior:
        - Uses get_time_by_index(idx * chunk_len) to locate the first timestep of the chunk.
        - Calls get_data(current_time, length=chunk_len), which may return fewer than chunk_len rows.
        - Pads the tail with zeros to always return shape (chunk_len, N).
        - For non-rank-0, returns a zero array of the correct shape without reading.
        """
        # Compute absolute start index for this chunk
        base_idx = idx * self.chunk_len
        N = int(self.data_size)
        # Non-rank-0: return zeros to keep shapes consistent across ranks
        if not is_rank_zero():
            data = np.empty((self.chunk_len, N), dtype=self.out_dtype)
            return data

        # Rank-0: fetch data and pad if needed
        current_time = self.get_time_by_index(base_idx)
        data = self.get_data(current_time, chunk_len=self.chunk_len)
        if data.ndim != 2 or data.shape[1] != N:
            raise ValueError(
                f"get_data must return (T, N) with N={N}, got {tuple(data.shape)}"
            )
        T = int(data.shape[0])
        if T < self.chunk_len:
            pad = np.zeros((self.chunk_len - T, N), dtype=self.out_dtype)
            data = np.vstack([data, pad]) if data.size else pad
        return data
