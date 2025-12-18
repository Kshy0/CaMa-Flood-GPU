# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from netCDF4 import Dataset

from cmfgpu.datasets.netcdf_dataset import NetCDFDataset
from cmfgpu.utils import find_indices_in


def exported_time_to_key(dt: datetime) -> str:
    """Constant key for exported format; we always use a single file path (prefix + suffix)."""
    return ""

class ExportedDataset(NetCDFDataset):
    """Dataset for exported catchment runoff (time, saved_points).

    File convention (by default): f"{var_name}_rank{rank}.nc"
    Variables expected:
      - time: numeric with units/calendar
      - save_coord: (saved_points,) linear catchment ids
      - {var_name}: (time, saved_points) values in mm (or other), divided by unit_factor on read

    Notes on distributed:
      - Unlike the grid-backed pipeline, each rank reads its own file; no broadcast
        or sparse mapping is performed. shard_forcing() is overridden to just flatten
        (B, T, C) -> (B*T, C).
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
        *args,
        **kwargs,
    ):

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
    # Coordinates & mask (1D)
    # -------------------------
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return 1D coordinate arrays.

        For exported format, this returns (save_coord, index) where:
          - save_coord is the linear catchment id array of shape (C,)
          - index is a simple 0..C-1 integer array of shape (C,)
        This keeps a 2-tuple signature while reflecting the 1D nature.
        """
        key = self.time_to_key(self.start_date)
        path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
        with Dataset(path, "r") as ds:
            if "save_coord" not in ds.variables:
                raise ValueError(f"'save_coord' not found in {path.name}")
            arr = ds.variables["save_coord"][:]
            sc = (arr.filled(0) if isinstance(arr, np.ma.MaskedArray) else np.asarray(arr)).astype(np.int64)
            return sc, np.arange(sc.shape[0], dtype=np.int64)

    @cached_property
    def data_mask(self) -> np.ndarray:
        """1D mask selecting all saved points by default."""
        sc, _ = self.get_coordinates()
        return np.ones_like(sc, dtype=bool)

    # -------------------------
    # Reading helpers (T, C)
    # -------------------------
    @staticmethod
    def _ensure_tc(data: np.ndarray, t_idx: Optional[int], c_idx: Optional[int]) -> np.ndarray:
        """Transpose data so that axes become (T, C). Extra dims must be size-1.

        If c_idx is None, tries to infer the single non-time dimension.
        """
        if t_idx is None:
            raise ValueError("A time dimension is required in the variable.")
        axes = list(range(data.ndim))
        if c_idx is None:
            rest = [a for a in axes if a != t_idx]
            if len(rest) != 1:
                raise ValueError(f"Expected exactly one non-time axis, got {data.shape} with dims={data.ndim}")
            c_idx = rest[0]
        front = [t_idx, c_idx]
        back = [a for a in axes if a not in front]
        out = np.transpose(data, axes=front + back)
        if out.ndim > 2:
            tail = out.shape[2:]
            if any(s != 1 for s in tail):
                raise ValueError(f"Unsupported extra non-size-1 dims after time/points: shape={out.shape}")
            out = out.reshape(out.shape[0], out.shape[1])
        return out

    def _read_ops(self, ops: List[Tuple[str, List[int]]]) -> np.ndarray:
        """Execute per-file reads using absolute time indices for (T, C) arrays."""
        if not ops:
            return np.empty((0, int(self.data_size)), dtype=self.out_dtype)
        chunks: List[np.ndarray] = []
        for key, abs_indices in ops:
            path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
            with Dataset(path, "r") as ds:
                var = ds.variables[self.var_name]
                dims = tuple(d.lower() for d in var.dimensions)
                if not (len(dims) == 2 and set(dims) == {"time", "saved_points"}):
                    raise ValueError(
                        f"Exported dataset expects variable dims ('time','saved_points'), got {var.dimensions}"
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
                arr = np.asarray(arr)
                arr = self._ensure_tc(arr, t_idx, c_idx)
                out = arr.astype(self.out_dtype, copy=False)
                chunks.append(out)
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)

    # -------------------------
    # Public API tweaks
    # -------------------------
    def get_data(self, current_time: datetime, chunk_len: int) -> np.ndarray:
        """Read contiguous block starting at current_time -> (T, C)."""
        try:
            start_abs = self._global_times.index(current_time)
        except ValueError as e:
            raise ValueError(f"Start time {current_time} not found in global timeline") from e
        end_abs = min(start_abs + int(chunk_len), len(self._global_times))
        times = self._global_times[start_abs:end_abs]
        ops = self._ops_from_times(times)
        data = self._read_ops(ops)
        return data / self.unit_factor

    # Override mapping/broadcast to a no-op flatten
    def shard_forcing(
        self,
        batch_runoff: torch.Tensor,
        local_runoff_indices: torch.Tensor,
        world_size: int
    ):
        """Flatten (B, T, C) -> (B*T, C) and optionally reorder columns by indices.

        - No broadcast across ranks.
        - If local_runoff_indices is provided, select columns accordingly to match
          the model's desired catchment order for this rank.
        """
        x = batch_runoff
        if not hasattr(x, "dim") or x.dim() != 3:
            raise ValueError(f"batch_runoff must be 3D, got shape {getattr(x, 'shape', None)}")
        B, T, C = x.shape
        flat = x.reshape(B * T, C)
        if world_size > 1:
            dist.broadcast(flat, src=0)
        flat = flat[:, local_runoff_indices]
        return flat.contiguous()

    # -------------------------
    # Mapping-related overrides
    # -------------------------
    def build_local_runoff_matrix(
        self,
        desired_catchment_ids: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a simple column order for this rank without sparse matrices.

        We read save_coord from the exported file and compute the indices that map
        desired_catchment_ids (model order) to columns in this dataset. 
        """
        # Load 1D save_coord
        key = self.time_to_key(self.start_date)
        path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
        with Dataset(path, "r") as ds:
            if "save_coord" not in ds.variables:
                raise ValueError(f"'save_coord' not found in {path}")
            arr = ds.variables["save_coord"][:]
            sc = (arr.filled(0) if isinstance(arr, np.ma.MaskedArray) else np.asarray(arr)).astype(np.int64)

        # Find positions for desired catchments
        col_pos = find_indices_in(desired_catchment_ids, sc)
        if np.any(col_pos == -1):
            missing = int(np.sum(col_pos == -1))
            raise ValueError(
                f"{missing} desired catchments are not available in exported file {path.name}"
            )

        local_indices = torch.tensor(col_pos.astype(np.int64), dtype=torch.int64, device=device)
        return local_indices

    # Disable legacy mapping/export helpers not needed here
    def generate_runoff_mapping_table(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError("ExportedDataset does not require mapping tables.")

    def export_catchment_runoff(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError("ExportedDataset does not export data; use upstream dataset to export.")

    def __getitem__(self, idx: int) -> np.ndarray:
        """Each rank reads its own data; no rank-0 gating."""
        if idx < 0:
            idx += len(self)
        base_idx = idx * self.chunk_len
        N = self.data_size
        current_time = self.get_time_by_index(base_idx)
        data = self.get_data(current_time, chunk_len=self.chunk_len)
        if data.ndim != 2 or data.shape[1] != N:
            raise ValueError(f"get_data must return (T, N) with N={N}, got {tuple(data.shape)}")
        T = data.shape[0]
        if T < self.chunk_len:
            pad = np.zeros((self.chunk_len - T, N), dtype=self.out_dtype)
            data = np.vstack([data, pad]) if data.size else pad
        return data
