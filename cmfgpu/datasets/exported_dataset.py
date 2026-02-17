# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from netCDF4 import Dataset

from cmfgpu.datasets.netcdf_dataset import NetCDFDataset
from cmfgpu.datasets.utils import single_file_key
from cmfgpu.utils import find_indices_in, is_rank_zero


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
        time_to_key: Optional[Callable[[datetime], str]] = single_file_key,
        coord_name: str = "catchment_id",
        in_memory: bool = False,
        *args,
        **kwargs,
    ):
        self.coord_name = coord_name
        self._in_memory = in_memory
        self._memory_cache: Optional[np.ndarray] = None  # Shape: (total_time_steps, num_catchments)
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

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """ExportedDataset has no grid; data is already at catchment level."""
        raise NotImplementedError(
            "ExportedDataset has no grid shape. Data is already catchment-aggregated."
        )

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
        
        # Auto-load to memory if in_memory mode is enabled
        if self._in_memory:
            self.load_to_memory()
        
        return None  # No matrix needed

    def load_to_memory(self) -> None:
        """Load all data into memory for faster repeated access.
        
        This method reads the entire dataset into a numpy array cached in memory,
        covering ALL files that span the [start_date, end_date] range.
        Subsequent __getitem__ calls will return slices from this cache instead
        of reading from disk.
        
        Note: build_local_runoff_matrix should be called first to set up
        the column reordering indices.
        """
        if self._memory_cache is not None:
            if is_rank_zero():
                print("[ExportedDataset] Data already in memory, skipping reload.")
            return
        
        # Read ALL time steps across all files using the multi-file infrastructure.
        ops = self._ops_from_times(self._global_times)
        all_data = self._read_ops(ops)
        
        # Store in cache with correct dtype and C-contiguous layout
        self._memory_cache = np.ascontiguousarray(all_data.astype(self.out_dtype, copy=False))
        
        if is_rank_zero():
            n_files = len(ops)
            mem_mb = self._memory_cache.nbytes / (1024 * 1024)
            print(f"[ExportedDataset] Loaded {self._memory_cache.shape} from {n_files} file(s) "
                  f"to memory ({mem_mb:.1f} MB)")

    def export_quantiles(
        self,
        out_path: Union[str, Path],
        quantiles: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
        var_name: Optional[str] = None,
        dtype: Literal["float32", "float64"] = "float32",
        complevel: int = 4,
        max_buffer_mb: float = 4096.0,
    ) -> Path:
        """Compute per-catchment temporal quantiles and save to NetCDF.

        For each catchment, computes the specified quantile values across the
        time dimension and writes the result to a single NetCDF file.

        Output format (consistent with ExportedDataset conventions):
          - Dimensions: ``quantile`` (Q), ``saved_points`` (C)
          - Variables:
            * ``quantile``     (Q,)    - quantile levels (e.g. 0.0 … 1.0)
            * ``catchment_id`` (C,)    - catchment IDs (int64)
            * ``{var_name}``   (Q, C)  - quantile values

        If ``build_local_runoff_matrix`` has been called, the output follows the
        reordered catchment order; otherwise it uses the file's native order.

        Exact quantile computation requires the full time series per catchment.
        When the full (T, C) array exceeds ``max_buffer_mb``, catchments are
        processed in column-batches whose size is automatically computed so that
        each batch (T × batch_catchments) fits within the buffer limit.

        Args:
            out_path: Output NetCDF file path.
            quantiles: Sequence of quantile levels in [0, 1].
            var_name: Variable name in output file (default: ``self.var_name``).
            dtype: Output data type.
            complevel: zlib compression level (0-9).
            max_buffer_mb: Maximum memory buffer in MB for reading data.
                When the full dataset exceeds this limit, catchments are
                processed in column-batches automatically. Default 4096 (4 GB).

        Returns:
            Path to the created NetCDF file.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        var_name = var_name or self.var_name
        quantiles_arr = np.asarray(quantiles, dtype=np.float64)
        Q = len(quantiles_arr)
        elem_bytes = 4 if dtype == "float32" else 8

        # ---- catchment IDs (respecting column reorder) ----
        file_catchment_ids, _ = self.get_coordinates()
        if self._local_runoff_indices is not None:
            catchment_ids = file_catchment_ids[self._local_runoff_indices]
        else:
            catchment_ids = file_catchment_ids
        C_total = len(catchment_ids)
        T_total = self.num_main_steps

        # ---- determine whether full (T, C) fits in buffer ----
        max_buffer_bytes = max_buffer_mb * 1024 * 1024
        full_size = T_total * C_total * elem_bytes
        fits_in_memory = (self._memory_cache is not None or full_size <= max_buffer_bytes)

        if not fits_in_memory:
            # Column-batch mode: compute batch_size (num catchments per batch)
            # so that T_total × batch_size × elem_bytes <= max_buffer_bytes
            batch_size = max(1, int(max_buffer_bytes / (T_total * elem_bytes)))
            n_batches = (C_total + batch_size - 1) // batch_size
            if is_rank_zero():
                print(f"[ExportedDataset] Dataset {full_size / 1e9:.1f} GB > "
                      f"buffer {max_buffer_mb:.0f} MB, "
                      f"processing {C_total} catchments in {n_batches} column-batches "
                      f"({batch_size} catchments/batch)")

        # ---- create output NetCDF ----
        dtype_nc = "f4" if dtype == "float32" else "f8"
        out_ds = Dataset(str(out_path), "w", format="NETCDF4")
        try:
            out_ds.createDimension("quantile", Q)
            out_ds.createDimension("saved_points", C_total)

            q_var = out_ds.createVariable("quantile", "f8", ("quantile",))
            q_var[:] = quantiles_arr
            q_var.long_name = "quantile level"

            cid_var = out_ds.createVariable("catchment_id", "i8", ("saved_points",))
            cid_var[:] = catchment_ids

            data_var = out_ds.createVariable(
                var_name, dtype_nc, ("quantile", "saved_points"),
                zlib=True, complevel=complevel,
            )
            data_var.long_name = f"{var_name} quantile values"

            if fits_in_memory:
                # ---- fits in memory: use __getitem__ with original chunk_len ----
                n_chunks = self._real_len()
                spin_up_offset = self.num_spin_up_chunks
                pieces: List[np.ndarray] = []
                for i in range(n_chunks):
                    pieces.append(self[spin_up_offset + i])
                all_data = np.concatenate(pieces, axis=0)[:T_total]
                q_values = np.quantile(all_data, quantiles_arr, axis=0)  # (Q, C)
                data_var[:] = q_values.astype(dtype)
            else:
                # ---- too large: batch by catchments (columns) ----
                # Exact quantile needs full time axis, so we read ALL time steps
                # for a subset of catchments per batch.
                ops = self._ops_from_times(self._global_times)
                for c_start in range(0, C_total, batch_size):
                    c_end = min(c_start + batch_size, C_total)
                    batch_cols = slice(c_start, c_end)

                    if self._local_runoff_indices is not None:
                        file_col_indices = self._local_runoff_indices[c_start:c_end]
                    else:
                        file_col_indices = np.arange(c_start, c_end, dtype=np.int64)

                    sort_order = np.argsort(file_col_indices)
                    sorted_cols = file_col_indices[sort_order]
                    unsort_order = np.argsort(sort_order)

                    file_chunks: List[np.ndarray] = []
                    for key, abs_indices in ops:
                        path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
                        with Dataset(path, "r") as ds_in:
                            var_in = ds_in.variables[self.var_name]
                            dims_in = tuple(d.lower() for d in var_in.dimensions)
                            t_idx = dims_in.index("time")
                            c_idx = dims_in.index("saved_points")

                            sel = [slice(None)] * var_in.ndim
                            sel[t_idx] = np.asarray(abs_indices, dtype=np.int32)
                            sel[c_idx] = sorted_cols
                            arr = var_in[tuple(sel)]
                            if isinstance(arr, np.ma.MaskedArray):
                                arr = arr.filled(0.0)
                            else:
                                arr = np.nan_to_num(np.asarray(arr), nan=0.0)
                            batch_data = self._ensure_tc(arr, t_idx, c_idx)
                            batch_data = batch_data[:, unsort_order]
                            file_chunks.append(batch_data)

                    all_batch = np.concatenate(file_chunks, axis=0) if len(file_chunks) > 1 else file_chunks[0]
                    q_batch = np.quantile(all_batch, quantiles_arr, axis=0)
                    data_var[:, batch_cols] = q_batch.astype(dtype)

        finally:
            out_ds.close()

        if is_rank_zero():
            print(f"[ExportedDataset] Saved quantiles to {out_path}")
            print(f"  Levels: {quantiles_arr.tolist()}")
            print(f"  Shape: ({Q}, {C_total})")

        return out_path

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
    # Disable grid-based methods
    # -------------------------
    def generate_runoff_mapping_table(self, *args, **kwargs):
        raise NotImplementedError("ExportedDataset does not require mapping tables.")

    def export_catchment_runoff(self, *args, **kwargs):
        raise NotImplementedError("ExportedDataset is already at catchment level.")

    # -------------------------
    # Override __getitem__ - no rank gating for exported data
    # -------------------------
    def __getitem__(self, idx: int) -> np.ndarray:
        """Fetch chunk - each rank reads independently for exported data.
        
        If build_local_runoff_matrix has been called, returns data reordered to
        match desired catchment order. Otherwise, returns data in file order.
        
        If in_memory mode is enabled (and load_to_memory has been called),
        returns a slice from the memory cache instead of reading from disk.
        """
        if idx < 0:
            idx += len(self)
        
        N = self.data_size
        
        # Use memory cache if available
        if self._memory_cache is not None:
            # Calculate time indices for this chunk
            start_time_idx = idx * self.chunk_len
            end_time_idx = min(start_time_idx + self.chunk_len, self._memory_cache.shape[0])
            
            data = self._memory_cache[start_time_idx:end_time_idx]
            T = data.shape[0]
            
            if T < self.chunk_len:
                pad = np.zeros((self.chunk_len - T, N), dtype=self.out_dtype)
                data = np.vstack([data, pad]) if data.size else pad
            
            # Already C-contiguous from load_to_memory, but slice may not be
            return np.ascontiguousarray(data)
        
        # Fall back to reading from disk
        data = self.read_chunk(idx)
        
        if data.ndim != 2 or data.shape[1] != N:
            raise ValueError(f"Expected shape (T, {N}), got {tuple(data.shape)}")
        
        T = data.shape[0]
        if T < self.chunk_len:
            pad = np.zeros((self.chunk_len - T, N), dtype=self.out_dtype)
            data = np.vstack([data, pad]) if data.size else pad
        return np.ascontiguousarray(data)
