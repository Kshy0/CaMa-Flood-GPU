# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from netCDF4 import Dataset

from cmfgpu.datasets.netcdf_dataset import NetCDFDataset
from cmfgpu.utils import find_indices_in, is_rank_zero


def exported_time_to_key(dt: datetime) -> str:
    """Constant key for exported format; we always use a single file path (prefix + suffix)."""
    return ""

class ExportedDataset(NetCDFDataset):
    """Dataset for pre-aggregated catchment runoff (time, saved_points).

    This dataset reads runoff data that has already been aggregated to catchment level,
    typically exported from a grid-based dataset using export_catchment_runoff().

    File convention (by default): f"{var_name}_rank{rank}.nc"
    Variables expected:
      - time: numeric with units/calendar
      - save_coord: (saved_points,) linear catchment ids
      - {var_name}: (time, saved_points) values

    Key differences from grid-based datasets:
      - Data is already at catchment level, no grid-to-catchment mapping needed
      - build_local_runoff_matrix only reorders columns to match desired catchment order
      - shard_forcing simply flattens (B, T, C) -> (B*T, C) without matrix multiplication
      - Each rank can read its own file independently
    """

    def __init__(
        self,
        base_dir: str,
        start_date: datetime,
        end_date: datetime,
        time_interval: timedelta = timedelta(days=1),
        var_name: str = "runoff",
        prefix: Optional[str] = "runoff_",
        suffix: str = "rank0.nc",
        time_to_key: Optional[Callable[[datetime], str]] = exported_time_to_key,
        coord_name: str = "catchment_id",
        *args,
        **kwargs,
    ):
        self.coord_name = coord_name
        super().__init__(
            base_dir=base_dir,
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval,
            var_name=var_name,
            prefix=prefix,
            suffix=suffix,
            time_to_key=time_to_key,
            *args,
            **kwargs,
        )

    # -------------------------
    # Coordinates (1D catchment IDs)
    # -------------------------
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return catchment coordinate arrays.

        Returns (save_coord, index) where:
          - save_coord: linear catchment id array of shape (C,)
          - index: simple 0..C-1 integer array of shape (C,)
        """
        key = self.time_to_key(self.start_date)
        path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
        with Dataset(path, "r") as ds:
            if self.coord_name not in ds.variables:
                raise ValueError(f"Coordinate variable '{self.coord_name}' not found in {path.name}. "
                               f"Available: {list(ds.variables.keys())}")
            arr = ds.variables[self.coord_name][:]
            sc = (arr.filled(0) if isinstance(arr, np.ma.MaskedArray) else np.asarray(arr)).astype(np.int64)
            return sc, np.arange(sc.shape[0], dtype=np.int64)

    @property
    def data_size(self) -> int:
        """Return number of catchments in the exported file."""
        if self._local_runoff_indices is not None:
            return len(self._local_runoff_indices)
        sc, _ = self.get_coordinates()
        return len(sc)

    # -------------------------
    # Reading helpers (T, C)
    # -------------------------
    @staticmethod
    def _ensure_tc(data: np.ndarray, t_idx: Optional[int], c_idx: Optional[int]) -> np.ndarray:
        """Transpose data to (T, C) format."""
        if t_idx is None:
            raise ValueError("A time dimension is required.")
        axes = list(range(data.ndim))
        if c_idx is None:
            rest = [a for a in axes if a != t_idx]
            if len(rest) != 1:
                raise ValueError(f"Expected one non-time axis, got shape={data.shape}")
            c_idx = rest[0]
        front = [t_idx, c_idx]
        back = [a for a in axes if a not in front]
        out = np.transpose(data, axes=front + back)
        if out.ndim > 2:
            tail = out.shape[2:]
            if any(s != 1 for s in tail):
                raise ValueError(f"Unsupported extra dims: shape={out.shape}")
            out = out.reshape(out.shape[0], out.shape[1])
        return out

    def _read_ops(self, ops: List[Tuple[str, List[int]]]) -> np.ndarray:
        """Read time steps and reorder columns if _local_runoff_indices is set."""
        # Determine output size
        if self._local_runoff_indices is not None:
            out_cols = len(self._local_runoff_indices)
        else:
            sc, _ = self.get_coordinates()
            out_cols = len(sc)
        
        if not ops:
            return np.empty((0, out_cols), dtype=self.out_dtype)
        
        chunks: List[np.ndarray] = []
        for key, abs_indices in ops:
            path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
            with Dataset(path, "r") as ds:
                var = ds.variables[self.var_name]
                dims = tuple(d.lower() for d in var.dimensions)
                if not (len(dims) == 2 and set(dims) == {"time", "saved_points"}):
                    raise ValueError(
                        f"Expected dims ('time','saved_points'), got {var.dimensions}"
                    )
                t_idx = dims.index("time")
                c_idx = dims.index("saved_points")
                if not abs_indices:
                    continue
                abs_idx = np.asarray(abs_indices, dtype=np.int32)
                sel = [slice(None)] * var.ndim
                sel[t_idx] = abs_idx
                arr = var[tuple(sel)]
                if isinstance(arr, np.ma.MaskedArray):
                    arr = arr.filled(0.0)
                else:
                    arr = np.nan_to_num(np.asarray(arr), nan=0.0)
                arr = self._ensure_tc(arr, t_idx, c_idx)
                
                # Reorder columns if indices are set
                if self._local_runoff_indices is not None:
                    arr = arr[:, self._local_runoff_indices]
                
                chunks.append(arr.astype(self.out_dtype, copy=False))
        
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)

    # -------------------------
    # Build local mapping (column reorder only)
    # -------------------------
    def build_local_runoff_matrix(
        self,
        desired_catchment_ids: np.ndarray,
    ) -> None:
        """Set up column reordering to match desired catchment order.
        
        Unlike grid-based datasets, this doesn't build a sparse matrix.
        It simply finds the column indices that map the file's catchment order
        to the desired order, and stores them in _local_runoff_indices.
        
        After calling this method:
          - _read_ops will return data with columns in the desired order
          - __getitem__ can be used (it requires _local_runoff_indices to be set)
          - shard_forcing simply flattens without matrix multiply
        
        Returns None (no matrix needed for exported data).
        """
        # Load catchment IDs from file
        key = self.time_to_key(self.start_date)
        path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
        with Dataset(path, "r") as ds:
            if self.coord_name not in ds.variables:
                raise ValueError(f"Coordinate variable '{self.coord_name}' not found in {path}")
            arr = ds.variables[self.coord_name][:]
            file_catchment_ids = (arr.filled(0) if isinstance(arr, np.ma.MaskedArray) 
                                  else np.asarray(arr)).astype(np.int64)

        # Find column positions for desired catchments
        col_pos = find_indices_in(desired_catchment_ids, file_catchment_ids)
        if np.any(col_pos == -1):
            missing = int(np.sum(col_pos == -1))
            raise ValueError(
                f"{missing} desired catchments not found in exported file {path.name}"
            )

        # Store indices for column reordering in _read_ops
        self._local_runoff_indices = col_pos.astype(np.int64)
        
        if is_rank_zero():
            print(f"[ExportedDataset] Mapped {len(desired_catchment_ids)} catchments "
                  f"from {len(file_catchment_ids)} in file")
        
        return None  # No matrix needed

    def shard_forcing(
        self,
        batch_runoff: torch.Tensor,
    ) -> torch.Tensor:
        """Flatten (B, T, C) -> (B*T, C).
        
        For ExportedDataset, data is already in the correct column order
        (set by build_local_runoff_matrix), so no matrix multiply is needed.
        """
        if batch_runoff.dim() == 3:
            B, T, C = batch_runoff.shape
            return batch_runoff.reshape(B * T, C).contiguous()
        elif batch_runoff.dim() == 4:
            B, T, K, C = batch_runoff.shape
            return batch_runoff.reshape(B * T, K, C).contiguous()
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {batch_runoff.dim()}D")

    # -------------------------
    # Override __getitem__ - no rank gating for exported data
    # -------------------------
    def __getitem__(self, idx: int) -> np.ndarray:
        """Fetch chunk - each rank reads independently for exported data."""
        if self._local_runoff_indices is None:
            raise RuntimeError(
                "build_local_runoff_matrix must be called before using __getitem__."
            )
        
        if idx < 0:
            idx += len(self)
        
        N = self.data_size
        data = self.read_chunk(idx)
        
        if data.ndim != 2 or data.shape[1] != N:
            raise ValueError(f"Expected shape (T, {N}), got {tuple(data.shape)}")
        
        T = data.shape[0]
        if T < self.chunk_len:
            pad = np.zeros((self.chunk_len - T, N), dtype=self.out_dtype)
            data = np.vstack([data, pad]) if data.size else pad
        return data

    # -------------------------
    # Disable grid-based methods
    # -------------------------
    def generate_runoff_mapping_table(self, *args, **kwargs):
        raise NotImplementedError("ExportedDataset does not require mapping tables.")

    def export_catchment_runoff(self, *args, **kwargs):
        raise NotImplementedError("ExportedDataset is already at catchment level.")
