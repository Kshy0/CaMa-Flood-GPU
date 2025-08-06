import os
from abc import ABC, abstractmethod
from functools import cached_property
from datetime import datetime
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
import torch.distributed as dist
from scipy.sparse import csr_matrix

from cmfgpu.utils import binread, find_indices_in, read_map, is_rank_zero

def compute_runoff_id(runoff_lon, runoff_lat, hires_lon, hires_lat):
    """
    Calculates runoff grid IDs considering ascending or descending order of coordinates.
    """
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

    assert np.all((ixin >= 0) & (ixin < nxin)), "Some hires_lon points fall outside the runoff grid (longitude)"
    assert np.all((iyin >= 0) & (iyin < nyin)), "Some hires_lat points fall outside the runoff grid (latitude)"

    runoff_id = iyin * nxin + ixin

    return runoff_id

class DefaultDataset(torch.utils.data.Dataset, ABC):
    """
    Custom abstract class that inherits from PyTorch Dataset.
    Defines a common interface for accessing data with distributed support.
    """
    def __init__(self, out_dtype: str = "float32", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_dtype = out_dtype

    def apply_runoff_to_catchments(self, 
                                   batch_runoff: torch.Tensor, 
                                   local_runoff_matrix: torch.Tensor, 
                                   local_runoff_indices: torch.Tensor,
                                   world_size: int) -> torch.Tensor:
        """
        Applies the local runoff matrix to the provided runoff data.
        Assumes runoff_data is a 1D array ordered according to the mapping file.
        Returns a tensor of catchment runoff values.
        """
        if world_size > 1:
            dist.broadcast(batch_runoff, src=0)
        return (batch_runoff[:, local_runoff_indices] @ local_runoff_matrix).contiguous()

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
    def get_data(self, current_time: datetime) -> np.ndarray:
        """
        To be implemented by subclasses, reads data of the specified date and time point from storage.
        Returns data which is 1D array ordered according to mapping file.
        """
        pass
    
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
        pass

    @abstractmethod
    def get_time_by_index(self, idx: int) -> datetime:
        """
        Returns the datetime corresponding to the given index.
        """
        pass

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
        Fetches data for a given index.
        Returns distributed data based on rank.
        """
        current_time = self.get_time_by_index(idx)
        if is_rank_zero():
            data = self.get_data(current_time)
        else:
            data = np.empty(self.data_size, dtype=self.out_dtype)

        return data
