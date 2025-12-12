# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from netCDF4 import Dataset, num2date

from cmfgpu.datasets.abstract_dataset import AbstractDataset


def yearly_time_to_key(dt: datetime) -> str:
    """Default time-to-file key: one file per year."""
    return f"{dt.year}"


class NetCDFDataset(AbstractDataset):
    """NetCDF-backed dataset with minimal I/O and a compact design.

    Key ideas:
    - Scan only time variables to build a global timeline and a dt->(file_key, local_index)
      map. No heavy data read during initialization.
    - Group requested timestamps into contiguous slices per file so each chunk is read with
      as few NetCDF reads as possible (often 1-2 reads per chunk).
    - Normalize variable dimensions to (T, Y, X) once per read; precompute a spatial mask
      and use a linear index list to quickly collapse (Y, X) -> N.
    """

    def _validate_files_exist(self, keys: Set[str]) -> None:
        missing: List[str] = []
        for key in sorted(keys):
            path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
            if not path.exists():
                missing.append(str(path))
        if missing:
            raise FileNotFoundError(
                "The following required NetCDF data files are missing:\n" + "\n".join(missing)
            )

    def _scan_time_metadata(self, start_dt: datetime, end_dt: datetime) -> None:
        """Read only time vars to construct a global time index and lookup map."""
        # Build key -> first_dt map to help with date guessing
        key_to_first_dt: Dict[str, datetime] = {}
        t = start_dt
        while t <= end_dt:
            k = self.time_to_key(t)
            if k not in key_to_first_dt:
                key_to_first_dt[k] = t
            t += self.time_interval
        # Ensure end_date is covered
        k_end = self.time_to_key(end_dt)
        if k_end not in key_to_first_dt:
            key_to_first_dt[k_end] = end_dt

        keys = set(key_to_first_dt.keys())
        self._validate_files_exist(keys)

        for key in sorted(keys):
            path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
            with Dataset(path, "r") as ds:
                tvar = ds.variables.get("time") or ds.variables.get("valid_time")
                if tvar is None:
                    raise ValueError(f"Time variable not found in file: {path.name}")

                try:
                    raw_dates = num2date(tvar[:], tvar.units, getattr(tvar, "calendar", "standard"))
                    dates = [
                        datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
                        for d in raw_dates
                    ]
                except (ValueError, TypeError):
                    # Fallback for "days since start" or similar non-standard units
                    base = None
                    if key in key_to_first_dt:
                        sample_dt = key_to_first_dt[key]
                        # Try to snap to year start
                        dt_year = datetime(sample_dt.year, 1, 1)
                        if self.time_to_key(dt_year) == key:
                            base = dt_year
                        else:
                            # Try to snap to month start
                            dt_month = datetime(sample_dt.year, sample_dt.month, 1)
                            if self.time_to_key(dt_month) == key:
                                base = dt_month
                            else:
                                # Fallback to the sample date itself
                                base = sample_dt
                    
                    if base is None:
                        try:
                            year = int(key)
                            base = datetime(year, 1, 1)
                        except ValueError:
                            pass
                    
                    if base is not None:
                        dates = [base + timedelta(days=float(x)) for x in tvar[:]]
                    else:
                        raise ValueError(f"Cannot parse time for key '{key}' with units '{getattr(tvar, 'units', '')}'")

                self._file_times[key] = []
                for i, dt in enumerate(dates):
                    self._file_times[key].append(dt)
                    if start_dt <= dt <= end_dt:
                        self._dt_to_loc[dt] = (key, i)
        expected_times: List[datetime] = []
        t = start_dt
        while t <= end_dt:
            expected_times.append(t)
            t += self.time_interval
        missing = [dt for dt in expected_times if dt not in self._dt_to_loc]
        if missing:
            preview = ", ".join(str(m) for m in missing[:10])
            raise ValueError(
                f"Missing required timestamps for the chosen time_interval. "
                f"First missing: {preview} (total {len(missing)}). "
                f"Check start_date alignment and dataset temporal resolution."
            )
        self._global_times = expected_times

    def _ops_from_times(self, times: List[datetime]) -> List[Tuple[str, List[int]]]:
        """Group requested datetimes into per-file absolute index ops.

        Output format: List of (file_key, abs_indices), where abs_indices are
        absolute time indices in that file to fetch, in the requested order.
        """
        if not times:
            return []

        # Preserve file order as first encountered in the times list
        file_order: List[str] = []
        file_to_indices: Dict[str, List[int]] = {}

        for dt in times:
            key, idx = self._dt_to_loc[dt]
            if key not in file_to_indices:
                file_to_indices[key] = []
                file_order.append(key)
            file_to_indices[key].append(idx)

        ops: List[Tuple[str, List[int]]] = []
        for key in file_order:
            idxs = file_to_indices[key]
            # Keep the order user requested; deduplicate while preserving order
            seen = set()
            uniq = [i for i in idxs if (i not in seen and not seen.add(i))]
            ops.append((key, uniq))

        return ops

    def _build_simulation_plan(self) -> None:
        """
        Builds the sequence of chunks for the entire simulation, including spin-up.
        self._plan will be a list of (start_time, ops).
        """
        self._plan = []
        
        # Helper to build chunks for a time range
        def build_chunks_for_range(start_dt, end_dt):
            chunks = []
            times = []
            t = start_dt
            while t <= end_dt:
                times.append(t)
                t += self.time_interval
            
            total = len(times)
            if total == 0:
                return []
            
            n_chunks = (total + self.chunk_len - 1) // self.chunk_len
            for ci in range(n_chunks):
                a = ci * self.chunk_len
                b = min(a + self.chunk_len, total)
                chunk_times = times[a:b]
                ops = self._ops_from_times(chunk_times)
                # Store the start time of the chunk for reference
                chunks.append((chunk_times[0], ops))
            return chunks

        # 1. Spin-up chunks
        self._spin_up_chunks_template = []
        if self.spin_up_cycles > 0:
            if self.spin_up_start_date is None or self.spin_up_end_date is None:
                raise ValueError("Spin-up dates must be provided if spin_up_cycles > 0")
            self._spin_up_chunks_template = build_chunks_for_range(self.spin_up_start_date, self.spin_up_end_date)
            
            for _ in range(self.spin_up_cycles):
                self._plan.extend(self._spin_up_chunks_template)

        # 2. Main simulation chunks
        main_chunks = build_chunks_for_range(self.start_date, self.end_date)
        self._plan.extend(main_chunks)

    def __init__(
        self,
        base_dir: str,
        start_date: datetime,
        end_date: datetime,
        time_interval: timedelta = timedelta(days=1),
        unit_factor: float = 1.0,
        var_name: str = "Runoff",
        prefix: str = "e2o_ecmwf_wrr2_glob15_day_Runoff_",
        suffix: str = ".nc",
        out_dtype: str = "float32",
        chunk_len: int = 24,
        time_to_key: Optional[Callable[[datetime], str]] = yearly_time_to_key,
        *args,
        **kwargs,
    ):
        self.base_dir = base_dir
        self.time_interval = time_interval
        self.unit_factor = unit_factor
        self.var_name = var_name
        self.prefix = prefix
        self.suffix = suffix
        self.time_to_key = time_to_key

        # Runtime metadata
        self._file_times = {}
        self._global_times = []
        self._dt_to_loc = {}
        # Each chunk plan is a list of (file_key, abs_time_indices) operations.
        # We read exact timesteps per file using fancy indexing once per file.
        self._chunk_plan = []

        # Build time metadata and per-chunk minimal-IO plans up-front (cheap).
        super().__init__(out_dtype=out_dtype, chunk_len=chunk_len, time_interval=time_interval, start_date=start_date, end_date=end_date, *args, **kwargs)
        
        # Determine full data range needed
        scan_start = self.start_date
        scan_end = self.end_date
        if self.spin_up_cycles > 0:
            if self.spin_up_start_date is not None and self.spin_up_start_date < scan_start:
                scan_start = self.spin_up_start_date
            if self.spin_up_end_date is not None and self.spin_up_end_date > scan_end:
                scan_end = self.spin_up_end_date

        self._scan_time_metadata(scan_start, scan_end)
        self._build_simulation_plan()

    @property
    def num_spin_up_chunks(self) -> int:
        if self.spin_up_cycles > 0:
            return len(self._spin_up_chunks_template) * self.spin_up_cycles
        return 0

    # -------------------------
    # Variable shape helpers
    # -------------------------
    @staticmethod
    def _pick_dim(dim_names: Tuple[str, ...], *candidates: str) -> Optional[int]:
        m = {n.lower(): i for i, n in enumerate(dim_names)}
        for c in candidates:
            if c in m:
                return m[c]
        return None

    @staticmethod
    def _ensure_tyx(data: np.ndarray, t_idx: Optional[int], y_idx: int, x_idx: int) -> np.ndarray:
        """Transpose data so that axes become (T, Y, X). Extra dims must be size-1."""
        axes = list(range(data.ndim))
        if t_idx is None:
            raise ValueError("A time dimension is required in the variable.")
        front = [t_idx, y_idx, x_idx]
        back = [a for a in axes if a not in front]
        data = np.transpose(data, axes=front + back)
        if data.ndim > 3:
            # Collapse any trailing size-1 dims
            tail = data.shape[3:]
            if any(s != 1 for s in tail):
                raise ValueError(f"Unsupported extra non-size-1 dims after time/lat/lon: shape={data.shape}")
            data = data.reshape(data.shape[0], data.shape[1], data.shape[2])
        return data

    @cached_property
    def _mask_2d(self) -> np.ndarray:
        """Compute a boolean (Y, X) mask of valid spatial points from the first timestep."""
        key = self.time_to_key(self.start_date)
        path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
        with Dataset(path, "r") as ds:
            var = ds.variables[self.var_name]
            dims = var.dimensions
            t_idx = self._pick_dim(dims, "time", "valid_time")
            y_idx = self._pick_dim(dims, "lat", "latitude", "y")
            x_idx = self._pick_dim(dims, "lon", "longitude", "long", "x")
            if y_idx is None or x_idx is None:
                raise ValueError(f"Unable to recognize lat/lon dims in {dims}")

            # Build slice selecting the first time step
            sl = [slice(None)] * var.ndim
            if t_idx is not None:
                sl[t_idx] = 0
            snap = var[tuple(sl)]
            snap = snap.filled(np.nan) if isinstance(snap, np.ma.MaskedArray) else np.array(snap)

            # Move to (Y, X)
            if t_idx is not None:
                y_adj = y_idx - (1 if y_idx > t_idx else 0)
                x_adj = x_idx - (1 if x_idx > t_idx else 0)
            else:
                y_adj, x_adj = y_idx, x_idx

            if snap.ndim != 2:
                axes = list(range(snap.ndim))
                axes.remove(y_adj)
                axes.remove(x_adj if x_adj < len(axes) + 1 else x_adj)
                snap = np.transpose(snap, axes + [y_adj, x_adj])
                snap = snap.reshape(snap.shape[-2], snap.shape[-1])
            else:
                if not (y_adj == 0 and x_adj == 1):
                    snap = np.transpose(snap, axes=[y_adj, x_adj])

            return ~np.isnan(snap)

    @cached_property
    def _mask_linear(self) -> np.ndarray:
        """Linear indices (C-order) of valid spatial points."""
        m = self._mask_2d
        return np.flatnonzero(m.ravel(order="C")).astype(np.int64)

    def _read_ops(self, ops: List[Tuple[str, List[int]]]) -> np.ndarray:
        """Execute per-file reads using absolute time indices.

        Each op is (file_key, abs_indices). We'll fetch exactly these time steps
        from the file in a single fancy-indexing operation along the time axis.
        """
        if not ops:
            return np.empty((0, int(self.data_size)), dtype=self.out_dtype)
        chunks: List[np.ndarray] = []
        for key, abs_indices in ops:
            path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
            with Dataset(path, "r") as ds:
                var = ds.variables[self.var_name]
                dims = var.dimensions
                t_idx = self._pick_dim(dims, "time", "valid_time")
                y_idx = self._pick_dim(dims, "lat", "latitude", "y")
                x_idx = self._pick_dim(dims, "lon", "longitude", "long", "x")
                if t_idx is None or y_idx is None or x_idx is None:
                    raise ValueError(f"Expect at least time/lat/lon dims, got: {dims}")
                # Build absolute time indices and read them directly in one go
                if not abs_indices:
                    # No indices requested from this file; skip
                    continue
                abs_idx = np.asarray(abs_indices, dtype=np.int32)
                sel = [slice(None)] * var.ndim
                # Use fancy indexing along the time dimension to fetch only required steps
                sel[t_idx] = abs_idx
                arr = var[tuple(sel)]
                # Fill masks / NaNs
                if isinstance(arr, np.ma.MaskedArray):
                    arr = arr.filled(0.0)
                else:
                    arr = np.nan_to_num(np.asarray(arr), nan=0.0)
                arr = np.asarray(arr)
                # Normalize to (T, Y, X)
                arr = self._ensure_tyx(arr, t_idx, y_idx, x_idx)
                T, Y, X = arr.shape
                # Collapse spatial to (T, Y*X) and select valid indices
                flat = arr.reshape(T, Y * X)
                out = flat[:, self._mask_linear]
                out = out.astype(self.out_dtype, copy=False)
                chunks.append(out)
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)

    def read_chunk(self, idx: int) -> np.ndarray:
        """
        Reads the chunk at the specified index using the pre-computed plan.
        """
        if idx < 0 or idx >= len(self._plan):
            raise IndexError(f"Chunk index {idx} out of range (0-{len(self._plan)-1})")
        
        _, ops = self._plan[idx]
        data = self._read_ops(ops)
        return data / self.unit_factor

    def get_time_by_index(self, idx: int) -> datetime:
        """
        Returns the datetime corresponding to the given absolute timestep index.
        Note: This is a bit ambiguous with spin-up because times repeat.
        This implementation assumes idx maps linearly to the plan.
        """
        chunk_idx = idx // self.chunk_len
        offset = idx % self.chunk_len
        
        if chunk_idx >= len(self._plan):
             raise IndexError("time index out of range")

        start_time, _ = self._plan[chunk_idx]
        return start_time + offset * self.time_interval

    def close(self) -> None:
        """No persistent open handles are kept; provided for interface completeness."""
        pass

    def get_data(self, current_time: datetime, chunk_len: int) -> np.ndarray:
        """Read a contiguous block starting at current_time with minimal NetCDF I/O.

        Returns: (T, N) where T <= length and N is the number of valid points.
        """
        try:
            start_abs = self._global_times.index(current_time)
        except ValueError as e:
            raise ValueError(f"Start time {current_time} not found in global timeline") from e

        end_abs = min(start_abs + int(chunk_len), len(self._global_times))
        times = self._global_times[start_abs:end_abs]
        ops = self._ops_from_times(times)
        data = self._read_ops(ops)
        return data / self.unit_factor

    def get_index_by_time(self, dt: datetime) -> int:
        """Returns the absolute time index for a given datetime."""
        try:
            return self._global_times.index(dt)
        except ValueError:
            raise ValueError(f"Time {dt} not found in dataset timeline.")

    def _real_len(self) -> int:
        """Number of samples (chunks). Each item yields up to chunk_len steps."""
        total = len(self._global_times)
        return (total + self.chunk_len - 1) // self.chunk_len if total else 0

    def _collect_required_keys(self) -> Set[str]:
        """Collect file keys covering [start_date, end_date] stepping by time_interval."""
        keys: Set[str] = set()
        t = self.start_date
        # + one extra step to ensure inclusive end coverage for non-divisible ranges
        while t <= self.end_date:
            keys.add(self.time_to_key(t))
            t += self.time_interval
        keys.add(self.time_to_key(self.end_date))
        return keys

    # -------------------------
    # Public API
    # -------------------------
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lon, lat) 1D arrays from the first file."""
        key = self.time_to_key(self.start_date)
        path = Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}"
        with Dataset(path, "r") as ds:
            lat = ds.variables.get("lat") or ds.variables.get("latitude")
            lon = ds.variables.get("lon") or ds.variables.get("longitude") or ds.variables.get("long")
            if lat is None or lon is None:
                raise ValueError("Unable to find lat/lon variables in the dataset.")
            return np.array(lon[:]), np.array(lat[:])

    @cached_property
    def data_mask(self) -> np.ndarray:
        """Expose spatial mask as (Y, X) for mapping utilities."""
        return self._mask_2d

    def __len__(self) -> int:
        return len(self._plan)



if __name__ == "__main__":
    resolution = "glb_15min"
    dataset = NetCDFDataset(
        base_dir="/home/eat/cmf_v420_pkg/inp/test_15min_nc",
        start_date=datetime(2000, 1, 1),
        end_date=datetime(2000, 12, 31),
        prefix="e2o_ecmwf_wrr2_glob15_day_Runoff_",
        suffix=".nc",
        var_name="Runoff",
        chunk_len=24,
    )
    dataset.generate_runoff_mapping_table(
        map_dir=f"/home/eat/cmf_v420_pkg/map/{resolution}",
        hires_map_tag="1min",
        out_dir=f"/home/eat/CaMa-Flood-GPU/inp/{resolution}",
        npz_file="runoff_mapping_nc.npz",
    )
