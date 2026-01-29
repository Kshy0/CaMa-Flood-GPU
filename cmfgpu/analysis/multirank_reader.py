# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cftime
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

from matplotlib.ticker import FuncFormatter

class MultiRankStatsReader:
    """
    Manage per‑rank NetCDF outputs written by a StatisticsAggregator-like component.

    Major Features:
      - Auto-detect rank files: {var_name}_rank{rank}.nc
      - Derive (x, y) locations for saved_points using one (mutually exclusive) method:
          * coord_source=(nx, ny) tuple               -> treat coord_raw as linear indices
          * coord_source=NetCDF file path             -> extract map shape (nx, ny)
          * coord_source=callable(coord_raw)->(x,y)   -> custom conversion
      - Provide vector / grid / time series extraction APIs
      - Basic visualization (single time slice + animation)
      - Export time-sliced grids to CaMa-Flood-compatible Fortran-order binary
    """

    # ----------------------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------------------
    def _safe_time_str(self, t_obj, fmt="%Y-%m-%d %H:%M:%S") -> str:
        """Helper to safely format time objects (datetime, cftime, or others)."""
        # Try strftime first (works for datetime and modern cftime)
        if hasattr(t_obj, "strftime"):
             try:
                 return t_obj.strftime(fmt)
             except Exception:
                 pass
        
        # Fallback to isoformat
        if hasattr(t_obj, "isoformat"):
             try:
                 return t_obj.isoformat()
             except Exception:
                 pass
                 
        # Fallback to string
        return str(t_obj)

    def _select_coord_name(self, ds: nc.Dataset, saved_points: int) -> Optional[str]:
        """Pick a ('saved_points',) variable to serve as save_coord."""
        if self.coord_name and self.coord_name in ds.variables:
            v = ds.variables[self.coord_name]
            if v.dimensions == ("saved_points",) and len(v) == saved_points:
                return self.coord_name

        for name, v in ds.variables.items():
            if name in ("time", self.var_name):
                continue
            if v.dimensions == ("saved_points",) and len(v) == saved_points:
                return name
        return None

    def _scan_rank_files(self) -> List[dict]:
        """Locate rank files and collect basic structural metadata."""
        pattern = f"{self.var_name}_rank*.nc"
        files = sorted(self.base_dir.glob(pattern))
        rank_map: Dict[int, List[Tuple[int, Path]]] = {}
        
        # Regex to match rank and optional year: var_rank0.nc or var_rank0_2000.nc
        if self.split_by_year:
            rank_re = re.compile(rf"^{re.escape(self.var_name)}_rank(\d+)_(\d{{4}})\.nc$")
        else:
            rank_re = re.compile(rf"^{re.escape(self.var_name)}_rank(\d+)\.nc$")

        for fp in files:
            m = rank_re.match(fp.name)
            if not m:
                continue
            rank_id = int(m.group(1))
            if self.split_by_year:
                year = int(m.group(2))
            else:
                year = -1

            if rank_id not in rank_map:
                rank_map[rank_id] = []
            rank_map[rank_id].append((year, fp))

        rank_infos: List[dict] = []
        
        for rank_id in sorted(rank_map.keys()):
            # Sort files by year (or just by name if year is -1, but here we use the tuple)
            # If year is -1, it means no year suffix.
            files_with_year = sorted(rank_map[rank_id], key=lambda x: x[0])
            paths = [p for y, p in files_with_year]
            
            # Use the first file to get metadata
            first_fp = paths[0]
            
            try:
                with nc.Dataset(first_fp, "r") as ds:
                    if self.var_name not in ds.variables:
                        continue
                    var = ds.variables[self.var_name]
                    dims = var.dimensions
                    
                    has_trials = "trial" in ds.dimensions
                    n_trials = int(ds.dimensions["trial"].size) if has_trials else 0
                    
                    has_levels = "levels" in ds.dimensions
                    n_levels = int(ds.dimensions["levels"].size) if has_levels else 0
                    
                    saved_points = int(ds.dimensions["saved_points"].size)

                    coord_name = self._select_coord_name(ds, saved_points)
                    coord_raw = None
                    if coord_name is not None:
                        coord_raw = np.array(ds.variables[coord_name][:])

                    rank_infos.append(
                        {
                            "rank_id": rank_id,
                            "paths": paths, # List of paths
                            "path": first_fp, # Keep for backward compat / metadata
                            "saved_points": saved_points,
                            "has_trials": has_trials,
                            "n_trials": n_trials,
                            "has_levels": has_levels,
                            "n_levels": n_levels,
                            "coord_name": coord_name,
                            "coord_raw": coord_raw,
                            "x": None,
                            "y": None,
                        }
                    )
            except Exception as e:
                print(f"Warning: skipping rank {rank_id} (file {first_fp.name}), reason: {e}")

        return rank_infos

    def _read_time_axis(self) -> None:
        """
        Read the time axis from the first rank's files; validate / truncate against others.
        Produce:
          - self._time_values_num
          - self._time_datetimes (naive)
          - self._time_units / _time_calendar
          - self._time_len
          - self._file_time_offsets (list of (start, end) indices for each file in the first rank)
        """
        if not self._rank_files:
            raise RuntimeError("No rank files loaded.")

        # Use the first rank to build the master time axis
        first_rank = self._rank_files[0]
        all_times = []
        self._file_time_offsets = [] # For the first rank, but assumed same for all
        
        current_offset = 0
        
        # Read time from all files of the first rank
        for fp in first_rank["paths"]:
            with nc.Dataset(fp, "r") as ds:
                tvar = ds.variables["time"]
                if not hasattr(self, "_time_units"):
                    self._time_units = getattr(tvar, "units")
                    self._time_calendar = getattr(tvar, "calendar", "standard")
                
                t_vals = np.array(tvar[:])
                all_times.append(t_vals)
                
                length = len(t_vals)
                self._file_time_offsets.append((current_offset, current_offset + length))
                current_offset += length

        t0 = np.concatenate(all_times)
        dt0 = nc.num2date(t0, units=self._time_units, calendar=self._time_calendar)
        
        # Keep original time objects (python datetime or cftime.datetime)
        self._time_datetimes = list(dt0)

        self._time_values_num = t0
        self._time_len = len(t0)

        # Validate other ranks
        for info in self._rank_files[1:]:
            rank_len = 0
            for fp in info["paths"]:
                with nc.Dataset(fp, "r") as dsi:
                    rank_len += len(dsi.variables["time"])
            
            if rank_len != self._time_len:
                print(
                    f"Warning: Rank {info.get('rank_id')} has {rank_len} time steps, "
                    f"mismatch with first rank {self._time_len}. Truncating to min length."
                )
                self._time_len = min(self._time_len, rank_len)

        if self._time_len < len(self._time_values_num):
            self._time_values_num = self._time_values_num[: self._time_len]
            self._time_datetimes = self._time_datetimes[: self._time_len]
            # Re-adjust offsets if needed? For now assume simple truncation at end.

    def _compute_all_xy(self, force: bool = False) -> None:
        """Compute (x, y) for each rank (custom converter -> unravel -> None)."""
        for info in self._rank_files:
            if info["coord_raw"] is None or info["saved_points"] == 0:
                info["x"], info["y"] = None, None
                continue
            if (info["x"] is not None and info["y"] is not None) and not force:
                continue

            if self._coord_converter is not None:
                try:
                    x, y = self._coord_converter(info["coord_raw"])
                    info["x"] = np.asarray(x, dtype=np.int64)
                    info["y"] = np.asarray(y, dtype=np.int64)
                    continue
                except Exception as e:
                    print(f"Custom coord converter failed ({info['path'].name}): {e}. Trying fallback.")

            if self._map_shape is not None:
                nx_, ny_ = self._map_shape
                total = nx_ * ny_
                flat = np.asarray(info["coord_raw"]).astype(np.int64)
                if flat.ndim == 1 and np.all((flat >= 0) & (flat < total)):
                    x, y = np.unravel_index(flat, (nx_, ny_))
                    info["x"] = x.astype(np.int64)
                    info["y"] = y.astype(np.int64)
                else:
                    info["x"], info["y"] = None, None
                    print(
                        f"Note: {info['path'].name} save_coord is not a valid linear index; cannot auto-convert."
                    )
            else:
                info["x"], info["y"] = None, None

    def _preload_cache(self) -> None:
        """Preload only the chosen inclusive slice [self._slice_start, self._slice_end]."""
        if self._slice_start is None or self._slice_end is None:
            raise RuntimeError("Slice indices not set.")
        
        # We need to map global slice to file-specific slices
        # self._file_time_offsets contains [(0, 366), (366, 731), ...]
        
        for info in self._rank_files:
            if info["saved_points"] == 0:
                info["cache"] = None
                continue
            
            rank_data_parts = []
            current_global_idx = 0
            
            # Iterate through files and extract relevant parts
            for i, fp in enumerate(info["paths"]):
                file_start_global, file_end_global = self._file_time_offsets[i]
                
                # Check intersection with requested slice [self._slice_start, self._slice_end]
                # Intersection: max(start1, start2) to min(end1, end2)
                req_start = max(self._slice_start, file_start_global)
                req_end = min(self._slice_end + 1, file_end_global) # exclusive end
                
                if req_start < req_end:
                    # Calculate local indices
                    local_start = req_start - file_start_global
                    local_end = req_end - file_start_global
                    
                    try:
                        with nc.Dataset(fp, "r") as ds:
                            var = ds.variables[self.var_name]
                            # Slicing logic: always take all spatial/trial dims
                            # Dimensions are (time, [trial], saved_points, [levels])
                            data = var[local_start:local_end, ...]
                            rank_data_parts.append(np.array(data, copy=True))
                    except Exception as e:
                        print(f"Warning: failed to cache {fp.name}: {e}")
                        # Append zeros or handle error?
                        shape = [local_end - local_start]
                        if info["has_trials"]:
                            shape.append(info["n_trials"])
                        shape.append(info["saved_points"])
                        if info["has_levels"]:
                            shape.append(info["n_levels"])
                        rank_data_parts.append(np.zeros(tuple(shape), dtype=np.float32))

            if rank_data_parts:
                info["cache"] = np.concatenate(rank_data_parts, axis=0)
            else:
                info["cache"] = None

    # ----------------------------------------------------------------------------------
    # Constructor
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        base_dir: Union[str, Path],
        var_name: str,
        coord_name: Optional[str] = None,
        coord_source: Optional[
            Union[
                Tuple[int, int],
                str,
                Path,
                Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
            ]
        ] = None,
        time_range: Optional[Tuple[Union[datetime, cftime.datetime], Union[datetime, cftime.datetime]]] = None,
        cache_enabled: bool = False,
        split_by_year: bool = False,
    ):
        """
        time_range: CLOSED interval (start_dt, end_dt), both inclusive.
        """
        self.base_dir = Path(base_dir)
        self.var_name = var_name
        self.coord_name = coord_name

        self._map_shape: Optional[Tuple[int, int]] = None
        self._coord_converter: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None

        self._rank_files: List[dict] = []
        self.cache_enabled = cache_enabled
        self.split_by_year = split_by_year

        self._slice_start: Optional[int] = None
        self._slice_end: Optional[int] = None
        self._t_indices: Optional[np.ndarray] = None

        # Interpret coord_source
        if coord_source is not None:
            if callable(coord_source):
                self._coord_converter = coord_source
            elif isinstance(coord_source, (str, Path)):
                self.load_map_shape_from_nc(coord_source)
            else:
                nx, ny = coord_source  # type: ignore
                self.set_map_shape((int(nx), int(ny)))

        self._rank_files = self._scan_rank_files()
        if not self._rank_files:
            raise FileNotFoundError(
                f"No files found in {self.base_dir} matching: {self.var_name}_rank*.nc"
            )

        self._read_time_axis()

        # Apply closed datetime slice with strict range checking (no clamping)
        if time_range is not None:
            # Strategy: Convert input range to numeric values using the NetCDF unit/calendar.
            start_in, end_in = time_range
            
            try:
                # date2num handles mixing types gracefully usually (if calendar compatible)
                t_start_val = nc.date2num(start_in, self._time_units, self._time_calendar)
                t_end_val = nc.date2num(end_in, self._time_units, self._time_calendar)
                
                # Check coverage against numeric limits
                file_min = self._time_values_num[0]
                file_max = self._time_values_num[-1]
                
                if t_start_val < file_min or t_end_val > file_max:
                    # Provide informative error
                    # Try to format limits back to dates for message
                    try:
                        d_min = self._safe_time_str(self._time_datetimes[0])
                        d_max = self._safe_time_str(self._time_datetimes[-1])
                        d_req_start = self._safe_time_str(start_in)
                        d_req_end = self._safe_time_str(end_in)
                    except:
                        d_min, d_max = str(file_min), str(file_max)
                        d_req_start, d_req_end = str(start_in), str(end_in)
                        
                    raise ValueError(
                        f"time_range outside available coverage. "
                        f"Requested [{d_req_start} .. {d_req_end}] but coverage is [{d_min} .. {d_max}]."
                    )

                # Locate indices using vectorized numeric comparison
                # left: first index where time >= start
                # right: last index where time <= end
                
                valid_mask = (self._time_values_num >= t_start_val) & (self._time_values_num <= t_end_val)
                indices = np.where(valid_mask)[0]
                
                if len(indices) == 0:
                     raise ValueError("No time steps found in the request range.")
                
                left = indices[0]
                right = indices[-1]

            except Exception as e:
                print(f"Warning: Numeric time comparison failed ({e}), falling back to direct object comparison.")
                
                start_dt, end_dt = start_in, end_in
                if start_dt > end_dt:
                     raise ValueError("time_range start must be <= end (closed interval).")
                     
                first_dt = self._time_datetimes[0]
                last_dt = self._time_datetimes[-1]
                
                if start_dt < first_dt or end_dt > last_dt:
                    raise ValueError(f"time_range outside coverage [{first_dt} .. {last_dt}]")

                dts = self._time_datetimes
                left = None
                for i, dt in enumerate(dts):
                    if dt >= start_dt:
                        left = i
                        break
                
                right = None
                for j in range(len(dts) - 1, -1, -1):
                    if dts[j] <= end_dt:
                        right = j
                        break

            if left is None or right is None:
                raise ValueError("Failed to locate time slice indices.")

            self._slice_start = left
            self._slice_end = right
            self._t_indices = np.arange(left, right + 1, dtype=np.int64)

            self._time_values_num = self._time_values_num[self._t_indices]
            self._time_datetimes = [dts[i] for i in self._t_indices]
            self._time_len = len(self._t_indices)
        else:
            self._slice_start = 0
            self._slice_end = self._time_len - 1
            self._t_indices = np.arange(self._time_len, dtype=np.int64)

        self._compute_all_xy(force=True)

        if self.cache_enabled:
            self._preload_cache()

    # ----------------------------------------------------------------------------------
    # Data getters
    # ----------------------------------------------------------------------------------
    def _get_data_from_files(self, info: dict, t_index: int, level: Optional[int] = None, trial: int = 0) -> np.ndarray:
        """Helper to fetch data for a single time step from the correct file."""
        orig_time = int(self._t_indices[t_index])
        
        # Find which file contains orig_time
        for i, (start, end) in enumerate(self._file_time_offsets):
            if start <= orig_time < end:
                local_time = orig_time - start
                fp = info["paths"][i]
                with nc.Dataset(fp, "r") as ds:
                    var = ds.variables[self.var_name]
                    
                    # Build index tuple
                    # 1. time
                    indices = [local_time]
                    
                    # 2. trial
                    if info["has_trials"]:
                        indices.append(trial)
                        
                    # 3. saved_points (all)
                    indices.append(slice(None))
                    
                    # 4. levels
                    if info["has_levels"]:
                        indices.append(level if level is not None else 0)
                        
                    return var[tuple(indices)]
        
        # Should not happen if t_index is valid
        raise IndexError(f"Time index {orig_time} not found in any file.")

    def get_vector(
        self,
        t_index: int,
        level: Optional[int] = None,
        trial: int = 0,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        if t_index < 0 or t_index >= self._time_len:
            raise IndexError(f"t_index out of range [0, {self._time_len - 1}]")
        
        parts: List[np.ndarray] = []
        for info in self._rank_files:
            if info["saved_points"] == 0:
                parts.append(np.empty((0,), dtype=dtype or np.float32))
                continue
            
            cache_arr = info.get("cache")
            if cache_arr is not None:
                # cache_arr shape: (time, [trial], saved_points, [levels])
                indices = [t_index]
                if info["has_trials"]:
                    indices.append(trial)
                indices.append(slice(None))
                if info["has_levels"]:
                    indices.append(level if level is not None else 0)
                
                data = cache_arr[tuple(indices)]
            else:
                data = self._get_data_from_files(info, t_index, level, trial)
                
            arr = np.array(data, copy=False)
            if dtype is not None:
                arr = arr.astype(dtype, copy=False)
            parts.append(arr)
        return np.concatenate(parts, axis=0) if parts else np.array([])

    def get_grid(
        self,
        t_index: int,
        level: Optional[int] = None,
        trial: int = 0,
        fill_value: float = np.nan,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        if self._map_shape is None:
            raise RuntimeError("map_shape is not set; cannot project to grid.")
        if t_index < 0 or t_index >= self._time_len:
            raise IndexError(f"t_index out of range [0, {self._time_len - 1}]")

        nx_, ny_ = self._map_shape
        grid = np.full((nx_, ny_), fill_value, dtype=dtype or np.float32)

        for info in self._rank_files:
            if info["saved_points"] == 0:
                continue
            x = info.get("x")
            y = info.get("y")
            if x is None or y is None:
                raise RuntimeError(f"{info['path'].name} missing (x,y); set map_shape or coord converter.")
            
            cache_arr = info.get("cache")
            if cache_arr is not None:
                # cache_arr shape: (time, [trial], saved_points, [levels])
                indices = [t_index]
                if info["has_trials"]:
                    indices.append(trial)
                indices.append(slice(None))
                if info["has_levels"]:
                    if level is None:
                        raise ValueError("This variable has 'levels'; please specify 'level'.")
                    indices.append(level)
                
                vals = cache_arr[tuple(indices)]
            else:
                if info["has_levels"] and level is None:
                     raise ValueError("This variable has 'levels'; please specify 'level'.")
                vals = self._get_data_from_files(info, t_index, level, trial)
                
            grid[x, y] = np.array(vals, copy=False)
        return grid

    def get_series(
        self,
        points: Union[np.ndarray, Sequence[np.ndarray]],
        level: Optional[int] = None,
        trial: int = 0,
        fill_value: float = np.nan,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        def _as_list(v):
            # Heuristic: if input is a list/tuple that looks like (N, 2) coordinates, treat as single array
            if isinstance(v, (list, tuple)):
                try:
                    arr = np.array(v)
                    # If it forms a valid (N, 2) array, wrap it as a single item
                    # This handles [(x, y)] -> (1, 2) and [[x1, y1], [x2, y2]] -> (2, 2)
                    if arr.ndim == 2 and arr.shape[1] == 2:
                        return [arr]
                except Exception:
                    pass
            return [np.asarray(a) for a in v] if isinstance(v, (list, tuple)) else [np.asarray(v)]
        arr_list = _as_list(points)
        if not arr_list:
            return np.full((self._time_len, 0), fill_value, dtype=dtype or np.float32)

        def _kind(a: np.ndarray) -> str:
            if a.ndim == 2 and a.shape[1] == 2:
                return "xy"
            if a.ndim == 1 or a.ndim == 0:
                return "id"
            raise ValueError(f"Unsupported points shape: {a.shape}")

        kinds = {_kind(a) for a in arr_list}
        if len(kinds) != 1:
            raise ValueError("Provide either all XY (N,2) or all IDs (N,). Do not mix.")
        use_xy = kinds.pop() == "xy"

        if use_xy:
            queries = [(int(px), int(py)) for a in arr_list for (px, py) in np.asarray(a, dtype=np.int64)]
        else:
            queries = [int(v) for a in arr_list for v in np.asarray(a, dtype=np.int64).ravel()]

        N = len(queries)
        if len(set(queries)) != N:
            raise ValueError("Duplicate points not allowed.")
        col_to_hits: List[Optional[Tuple[int, int]]] = [None] * N

        # Map queries to (rank_idx, local_index) and check all found
        if use_xy:
            for r_idx, info in enumerate(self._rank_files):
                if info["saved_points"] == 0:
                    continue
                x, y = info.get("x"), info.get("y")
                if x is None or y is None:
                    continue
                
                # Build lookup map for this rank: (x, y) -> local_index
                if all(hit is not None for hit in col_to_hits):
                    break

                # Create a dictionary for O(1) lookup
                rank_lookup = { (int(xi), int(yi)): i for i, (xi, yi) in enumerate(zip(x, y)) }
                
                for c, (qx, qy) in enumerate(queries):
                    if col_to_hits[c] is not None:
                        continue
                    
                    if (qx, qy) in rank_lookup:
                        col_to_hits[c] = (r_idx, rank_lookup[(qx, qy)])

        else:
            for r_idx, info in enumerate(self._rank_files):
                if info["saved_points"] == 0 or info["coord_raw"] is None:
                    continue
                
                if all(hit is not None for hit in col_to_hits):
                    break

                raw = np.asarray(info["coord_raw"]).ravel()
                rank_lookup = { int(val): i for i, val in enumerate(raw) }

                for c, qid in enumerate(queries):
                    if col_to_hits[c] is not None:
                        continue
                    
                    if qid in rank_lookup:
                        col_to_hits[c] = (r_idx, rank_lookup[qid])

        if any(hit is None for hit in col_to_hits):
            raise ValueError("Some points not found in any rank.")

        # Print queried points info (Verbose: only if small number of points)
        if len(queries) <= 20:
            print(f"Querying {len(queries)} points:")
            for c, q in enumerate(queries):
                r_idx, li = col_to_hits[c]
                info = self._rank_files[r_idx]

                # Get catchment ID if available
                cid = "N/A"
                if info.get("coord_raw") is not None:
                    cid = info["coord_raw"][li]

                if use_xy:
                    qx, qy = q
                    print(f"  Point {c}: (x={qx}, y={qy}) -> Rank {r_idx}, Local Idx {li}, CatchmentID={cid}")
                else:
                    qid = q
                    # If querying by ID, we might want to print X, Y if available
                    x_val = info["x"][li] if info.get("x") is not None else "N/A"
                    y_val = info["y"][li] if info.get("y") is not None else "N/A"
                    print(f"  Point {c}: ID={qid} -> Rank {r_idx}, Local Idx {li}, (x={x_val}, y={y_val})")
        else:
            print(f"Querying {len(queries)} points (verbose output suppressed for >20 points)...")

        out = np.full((self._time_len, N), fill_value, dtype=dtype or np.float32)

        rank_to_cols: dict[int, List[Tuple[int, int]]] = {}
        for col, hit in enumerate(col_to_hits):
            r_idx, li = hit  # hit is guaranteed not None
            rank_to_cols.setdefault(r_idx, []).append((col, li))

        # Fast path: if cached in memory
        for r_idx, pairs in rank_to_cols.items():
            info = self._rank_files[r_idx]
            cache_arr = info.get("cache")
            if cache_arr is not None:
                # cache_arr shape: (time, [trial], saved_points, [levels])
                indices = [slice(None)] # time: all
                if info["has_trials"]:
                    indices.append(trial)
                indices.append(slice(None)) # saved_points: placeholder, will index later
                if info["has_levels"]:
                    if level is None:
                        raise ValueError("This variable has 'levels'; specify `level`.")
                    indices.append(level)
                
                # We need to be careful with indexing. 
                # cache_arr[indices] gives us (time, saved_points) or similar.

                # Construct base indices tuple
                base_indices = [slice(None)] # time
                if info["has_trials"]:
                    base_indices.append(trial)
                
                # saved_points dim is next.
                # levels dim is last.
                
                for col, li in pairs:
                    # Construct specific indices for this point
                    pt_indices = list(base_indices)
                    pt_indices.append(li) # saved_points index
                    if info["has_levels"]:
                        pt_indices.append(level)
                    
                    out[:, col] = np.asarray(cache_arr[tuple(pt_indices)], dtype=dtype or np.float32)
                continue

            # No cache: minimize I/O by opening once and slicing contiguous time window
            if self._slice_start is None or self._slice_end is None:
                raise RuntimeError("Internal error: time slice is not set.")
            
            idx = np.array([li for (_, li) in pairs], dtype=np.int64)
            
            # Iterate over files to fill data
            for i, fp in enumerate(info["paths"]):
                file_start_global, file_end_global = self._file_time_offsets[i]
                
                # Intersection with requested slice [self._slice_start, self._slice_end]
                req_start = max(self._slice_start, file_start_global)
                req_end = min(self._slice_end + 1, file_end_global)
                
                if req_start < req_end:
                    print(f"    Reading {fp.name} (indices: {len(idx)})...")
                    local_start = req_start - file_start_global
                    local_end = req_end - file_start_global
                    
                    # out is indexed 0..self._time_len-1 corresponding to self._slice_start..self._slice_end
                    out_start = req_start - self._slice_start
                    out_end = req_end - self._slice_start
                    
                    try:
                        with nc.Dataset(fp, "r") as ds:
                            var = ds.variables[self.var_name]
                            
                            # Build slicing tuple
                            # 1. time
                            slices = [slice(local_start, local_end)]
                            
                            # 2. trial
                            if info["has_trials"]:
                                slices.append(trial)
                                
                            # 3. saved_points (using advanced indexing with `idx`)
                            slices.append(idx)
                            
                            # 4. levels
                            if info["has_levels"]:
                                if level is None:
                                    raise ValueError("This variable has 'levels'; specify `level`.")
                                slices.append(level)
                                
                            chunk = np.asarray(var[tuple(slices)])
                        
                        # Scatter chunk to output columns
                        # chunk shape should be (time_len, num_points)
                        for k, (col, _) in enumerate(pairs):
                            out[out_start:out_end, col] = chunk[:, k].astype(dtype or np.float32, copy=False)
                    except Exception as e:
                        print(f"Warning: failed to read {fp.name}: {e}")

        return out

    # ----------------------------------------------------------------------------------
    # Basic info
    # ----------------------------------------------------------------------------------
    @property
    def num_ranks(self) -> int:
        return len(self._rank_files)

    @property
    def time_len(self) -> int:
        return self._time_len

    @property
    def times(self) -> List[datetime]:
        return self._time_datetimes

    @property
    def map_shape(self) -> Optional[Tuple[int, int]]:
        return self._map_shape

    def set_map_shape(self, map_shape: Tuple[int, int]) -> None:
        if len(map_shape) != 2:
            raise ValueError("map_shape must be a (nx, ny) tuple.")
        self._map_shape = (int(map_shape[0]), int(map_shape[1]))
        if getattr(self, "_rank_files", None):
            self._compute_all_xy(force=True)

    def load_map_shape_from_nc(
        self,
        nc_path: Union[str, Path],
    ) -> None:
        p = Path(nc_path)
        if not p.exists():
            raise FileNotFoundError(f"NetCDF file not found: {p}")

        nx = ny = None
        with nc.Dataset(p, "r") as ds:
            attrs = {a: ds.getncattr(a) for a in ds.ncattrs()}
            if "nx" in attrs and "ny" in attrs:
                nx = int(attrs["nx"])
                ny = int(attrs["ny"])
            if (nx is None or ny is None) and "nx" in ds.variables and "ny" in ds.variables:
                try:
                    nx = int(np.array(ds.variables["nx"][:]).squeeze())
                    ny = int(np.array(ds.variables["ny"][:]).squeeze())
                except Exception:
                    pass
            if (nx is None or ny is None) and "map_shape" in ds.variables:
                arr = np.array(ds.variables["map_shape"][:]).squeeze()
                if arr.size >= 2:
                    nx = int(arr[0]); ny = int(arr[1])
            if (nx is None or ny is None) and "map_shape" in attrs:
                arr = np.array(attrs["map_shape"]).squeeze()
                if np.size(arr) >= 2:
                    flat = np.ravel(arr)
                    nx = int(flat[0]); ny = int(flat[1])
            if nx is None or ny is None:
                dim_pairs = [("nx", "ny"), ("x", "y"), ("lon", "lat")]
                for a, b in dim_pairs:
                    if a in ds.dimensions and b in ds.dimensions:
                        nx = int(ds.dimensions[a].size)
                        ny = int(ds.dimensions[b].size)
                        break
        if nx is None or ny is None:
            raise KeyError("Could not find nx/ny or map_shape (attrs/vars/dims).")
        self.set_map_shape((nx, ny))

    def set_coord_converter(
        self,
        converter: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        self._coord_converter = converter
        self._compute_all_xy(force=True)

    # ----------------------------------------------------------------------------------
    # Visualization
    # ----------------------------------------------------------------------------------
    def plot_single_time(
        self,
        t_index: int = 0,
        level: Optional[int] = None,
        trial: int = 0,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (8, 6),
        as_scatter_if_no_map: bool = True,
        s: float = 1.0,
        auto_crop: bool = True,
        crop_pad: int = 10,
    ) -> None:
        if t_index < 0 or t_index >= self._time_len:
            raise IndexError(f"t_index out of range [0, {self._time_len - 1}]")
            
        t_str = f"t={t_index}"
        if self.times:
             t_str = self._safe_time_str(self.times[t_index])
        
        # Check if we have trials to display in title
        has_trials = False
        if self._rank_files and self._rank_files[0]["has_trials"]:
            has_trials = True
        
        title_str = f"{self.var_name} @ {t_str}"
        if has_trials:
            title_str += f" (Trial {trial})"

        fig, ax = plt.subplots(figsize=figsize)
        if self.map_shape is not None:
            grid = self.get_grid(t_index, level=level, trial=trial)
            im = ax.imshow(grid.T, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title_str)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            
            if auto_crop:
                valid_mask = np.isfinite(grid)
                if np.any(valid_mask):
                    xs, ys = np.where(valid_mask)
                    if len(xs) > 0:
                        xmin, xmax = xs.min(), xs.max()
                        ymin, ymax = ys.min(), ys.max()
                        
                        # Apply padding
                        xmin = max(0, xmin - crop_pad)
                        xmax = min(grid.shape[0] - 1, xmax + crop_pad)
                        ymin = max(0, ymin - crop_pad)
                        ymax = min(grid.shape[1] - 1, ymax + crop_pad)
                        
                        ax.set_xlim(xmin - 0.5, xmax + 0.5)
                        ax.set_ylim(ymax + 0.5, ymin - 0.5)

        elif as_scatter_if_no_map:
            xs: List[np.ndarray] = []
            ys: List[np.ndarray] = []
            vals: List[np.ndarray] = []
            for info in self._rank_files:
                if info["saved_points"] == 0:
                    continue
                if info["x"] is None or info["y"] is None:
                    raise RuntimeError("map_shape not set and no converter-provided (x,y).")
                xs.append(info["x"])
                ys.append(info["y"])
                cache_arr = info.get("cache")
                if cache_arr is not None:
                    # cache_arr shape: (time, [trial], saved_points, [levels])
                    indices = [t_index]
                    if info["has_trials"]:
                        indices.append(trial)
                    indices.append(slice(None))
                    if info["has_levels"]:
                        indices.append(level if level is not None else 0)
                    vv = cache_arr[tuple(indices)]
                else:
                    orig_t = int(self._t_indices[t_index])
                    with nc.Dataset(info["path"], "r") as ds:
                        var = ds.variables[self.var_name]
                        
                        # Build slicing tuple
                        # 1. time
                        indices = [orig_t]
                        
                        # 2. trial
                        if info["has_trials"]:
                            indices.append(trial)
                            
                        # 3. saved_points (all)
                        indices.append(slice(None))
                        
                        # 4. levels
                        if info["has_levels"]:
                            indices.append(level if level is not None else 0)
                            
                        vv = var[tuple(indices)]
                vals.append(np.array(vv))
            x_all = np.concatenate(xs) if xs else np.array([])
            y_all = np.concatenate(ys) if ys else np.array([])
            v_all = np.concatenate(vals) if vals else np.array([])
            sc = ax.scatter(x_all, y_all, c=v_all, s=s, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{title_str} (scatter)")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            
            if auto_crop and len(x_all) > 0:
                xmin, xmax = x_all.min(), x_all.max()
                ymin, ymax = y_all.min(), y_all.max()
                
                ax.set_xlim(xmin - crop_pad, xmax + crop_pad)
                ax.set_ylim(ymax + crop_pad, ymin - crop_pad)
                ax.invert_yaxis()

        else:
            raise RuntimeError("Cannot plot without map_shape and scatter fallback disabled.")
        fig.tight_layout()

    def animate(
        self,
        out_path: Union[str, Path],
        level: Optional[int] = None,
        trial: int = 0,
        x_range: Optional[Tuple[int, int]] = None,
        y_range: Optional[Tuple[int, int]] = None,
        t_range: Optional[Tuple[int, int]] = None,
        fps: int = 10,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (8, 6),
        auto_crop: bool = True,
        crop_pad: int = 10,
    ) -> None:
        if self._map_shape is None:
            raise RuntimeError("Animation requires map_shape.")
        t_start = 0 if t_range is None else max(0, int(t_range[0]))
        t_end = self._time_len if t_range is None else min(self._time_len, int(t_range[1]))
        if t_start >= t_end:
            raise ValueError("Invalid t_range: ensure t_start < t_end")

        nx_, ny_ = self._map_shape
        
        xmin = 0 
        xmax = nx_ - 1 
        ymin = 0 
        ymax = ny_ - 1
        
        if auto_crop and (x_range is None and y_range is None):
            # Fetch first frame
            grid_0 = self.get_grid(t_start, level=level, trial=trial)
            valid_mask = np.isfinite(grid_0)
            if np.any(valid_mask):
                xs, ys = np.where(valid_mask)
                xmin_c, xmax_c = xs.min(), xs.max()
                ymin_c, ymax_c = ys.min(), ys.max()
                
                xmin = max(0, xmin_c - crop_pad)
                xmax = min(nx_ - 1, xmax_c + crop_pad)
                ymin = max(0, ymin_c - crop_pad)
                ymax = min(ny_ - 1, ymax_c + crop_pad)
        
        # Override with manual ranges if provided
        if x_range is not None:
            xmin = max(0, int(x_range[0]))
            xmax = min(nx_ - 1, int(x_range[1]))
        if y_range is not None:
            ymin = max(0, int(y_range[0]))
            ymax = min(ny_ - 1, int(y_range[1]))

        if xmin > xmax or ymin > ymax:
            raise ValueError("Invalid x_range or y_range")

        first_grid = self.get_grid(t_start, level=level, trial=trial)
        window = first_grid[xmin:xmax + 1, ymin:ymax + 1]
        if vmin is None:
            vmin = np.nanmin(window) if np.isfinite(window).any() else 0.0
        if vmax is None:
            vmax = np.nanmax(window) if np.isfinite(window).any() else 1.0
        if not (vmax > vmin):
            vmax = vmin + 1.0
            
        extent = (xmin - 0.5, xmax + 0.5, ymax + 0.5, ymin - 0.5)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(window.T, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Use our robust time logic if available
        t_label = f"t={t_start}"
        if self.times:
             t_label = self._safe_time_str(self.times[t_start])
             
        ttl = ax.set_title(f"{self.var_name} @ {t_label}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.tight_layout()

        def _update(frame_idx: int):
            ti = t_start + frame_idx
            grid = self.get_grid(ti, level=level, trial=trial)
            win = grid[xmin:xmax + 1, ymin:ymax + 1]
            im.set_data(win.T)
            
            t_lbl = f"t={ti}"
            if self.times:
                t_lbl = self._safe_time_str(self.times[ti])
                
            ttl.set_text(f"{self.var_name} @ {t_lbl}")
            return [im, ttl]

        frames = t_end - t_start
        ani = animation.FuncAnimation(fig, _update, frames=frames, interval=1000 / fps, blit=False)

        out_path = Path(out_path)
        if out_path.suffix.lower() == ".gif":
            try:
                writer = animation.PillowWriter(fps=fps)
            except Exception:
                raise RuntimeError("Cannot create GIF (install Pillow) or choose .mp4.")
            ani.save(out_path, writer=writer)
        else:
            try:
                Writer = animation.writers["ffmpeg"]
                writer = Writer(fps=fps, metadata=dict(artist="MultiRankStatsReader"))
            except Exception:
                raise RuntimeError("ffmpeg writer not found. Install ffmpeg or use .gif.")
            ani.save(out_path, writer=writer)
        plt.close(fig)

    def plot_series(
        self,
        points: Union[np.ndarray, Sequence[np.ndarray], List[int]],
        level: Optional[int] = None,
        trial: Union[int, List[int]] = 0,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        labels: Optional[List[str]] = None,
        **kwargs
    ) -> plt.Axes:
        """
        Plot time series for specified points (IDs or XY coordinates).
        
        Args:
            points: One or more points. Can be a list of IDs/catchment_ids, or a list of (x,y) tuples.
            level: Level index if variable has levels.
            trial: Single trial index (int) or list of trial indices.
            figsize: Figure size tuple (width, height) if creating new figure.
            title: Title of the plot.
            ax: Existing matplotlib axis to plot on.
            labels: Optional list of labels for the points (length must match number of points).
            **kwargs: Additional keyword arguments passed to ax.plot
        
        Returns:
            The matplotlib Axes object.
        """
        if isinstance(trial, int):
            trials = [trial]
        else:
            trials = trial
            
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True
            
        # Select Time Axis Strategy
        # Prefer raw numeric values + FuncFormatter for perfect calendar support
        use_numeric_time = False
        if hasattr(self, "_time_values_num") and self._time_values_num is not None \
           and hasattr(self, "_time_units") and getattr(self, "_time_calendar", None):
            times_to_plot = self._time_values_num
            use_numeric_time = True
        elif self.times:
            # Fallback to datetime list (cached property)
            times_to_plot = self.times
        else:
            # Fallback to simple indices
            times_to_plot = np.arange(self.time_len)

        # Ensure points is in a format suitable for get_series
        
        for t in trials:
            # Fetch data: shape (time_len, num_points)
            data = self.get_series(points, level=level, trial=t)
            num_points = data.shape[1]
            
            for i in range(num_points):
                # Construct label
                # If multiple trials, include trial info. If multiple points, include point info.
                lbl_parts = []
                
                # Point Label
                if labels and i < len(labels):
                    lbl_parts.append(str(labels[i]))
                else:
                    # Try to give a sensible default label from points
                    if isinstance(points, (list, tuple, np.ndarray)):
                        # If points passed as [1, 2], points[i] is 1
                        # If points passed as [[1,2], [3,4]], points[i] is [1,2]
                        try:
                            pt_val = points[i]
                            lbl_parts.append(f"Pt {pt_val}")
                        except:
                            lbl_parts.append(f"Pt {i}")
                    else:
                        lbl_parts.append(f"Pt {i}")

                # Trial Label (only if ambiguous or multiple trials)
                if len(trials) > 1:
                    lbl_parts.append(f"(Trial {t})")
                elif not labels and num_points == 1:
                     # Single point, single trial, explicit label is nice
                     lbl_parts.append(f"(Trial {t})")

                label_str = " ".join(lbl_parts)
                
                ax.plot(times_to_plot, data[:, i], label=label_str, **kwargs)

        # Setup Axis Formatting
        if use_numeric_time:
            def time_tick_formatter(x, pos):
                try:
                    # Use netcdf4 num2date to convert scalar to cftime/datetime object
                    # This works for ALL calendars (360_day, noleap, etc)
                    d = nc.num2date(x, units=self._time_units, calendar=self._time_calendar)
                    return d.strftime('%Y-%m-%d')
                except Exception:
                    return f"{x:.1f}"
            
            ax.xaxis.set_major_formatter(FuncFormatter(time_tick_formatter))
            ax.set_xlabel(f"Time ({self._time_calendar})")
        else:
            ax.set_xlabel("Time")

        ax.set_ylabel(self.var_name)
        
        if title:
            ax.set_title(title)
        elif not ax.get_title():
            # Default title
            t_str = ""
            if len(times_to_plot) > 0:
                if use_numeric_time:
                     try:
                        start_d = nc.num2date(times_to_plot[0], units=self._time_units, calendar=self._time_calendar)
                        end_d = nc.num2date(times_to_plot[-1], units=self._time_units, calendar=self._time_calendar)
                        t_str = f"{start_d.strftime('%Y-%m-%d')} - {end_d.strftime('%Y-%m-%d')}"
                     except:
                        pass
                elif hasattr(times_to_plot[0], 'date'):
                    t_str = f"{times_to_plot[0].date()} - {times_to_plot[-1].date()}"
            ax.set_title(f"{self.var_name} Time Series {t_str}")
            
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # If we created the figure, layout tight
        if created_fig:
            plt.tight_layout()
            
        return ax

    # ----------------------------------------------------------------------------------
    # Export
    # ----------------------------------------------------------------------------------
    def export_to_cama_bin(
        self,
        out_dir: Union[str, Path],
        out_var_name: str,
        t_range: Optional[Tuple[int, int]] = None,
        trial: int = 0,
        fill_value: float = 1e20,
        dtype: np.dtype = np.float32,
        progress: bool = True,
    ) -> None:
        if self._map_shape is None:
            raise RuntimeError("map_shape is required to export .bin files.")
        if any(info["has_levels"] for info in self._rank_files):
            raise ValueError("Variables with 'levels' not supported for export.")
        if not self.times or self._time_len == 0:
            raise RuntimeError("No time axis available for export.")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        t_start = 0 if t_range is None else max(0, int(t_range[0]))
        t_end = self._time_len if t_range is None else min(self._time_len, int(t_range[1]))
        if t_start >= t_end:
            raise ValueError("Invalid t_range: ensure t_start < t_end")

        year_to_indices: dict[int, List[int]] = {}
        for ti in range(t_start, t_end):
            year = int(self.times[ti].year)
            year_to_indices.setdefault(year, []).append(ti)

        for year in sorted(year_to_indices.keys()):
            year_path = out_dir / f"{out_var_name}{year}.bin"
            if progress:
                print(f"[BIN] writing year {year} -> {year_path.name} ({len(year_to_indices[year])} frames)")
            with open(year_path, "wb") as fw:
                for ti in year_to_indices[year]:
                    grid = self.get_grid(ti, level=None, trial=trial, fill_value=fill_value, dtype=dtype)
                    grid = np.where(np.isfinite(grid), grid, fill_value).astype(dtype, copy=False)
                    fw.write(np.asfortranarray(grid).tobytes(order="F"))

    # ----------------------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------------------
    def get_all_coords_xy(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        for info in self._rank_files:
            if info["saved_points"] == 0:
                continue
            if info["x"] is None or info["y"] is None:
                return None, None
            xs.append(info["x"])
            ys.append(info["y"])
        if not xs:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        return np.concatenate(xs), np.concatenate(ys)

    def get_all_cids(self) -> Optional[np.ndarray]:
        cids: List[np.ndarray] = []
        for info in self._rank_files:
            if info["saved_points"] == 0 or info["coord_raw"] is None:
                continue
            cids.append(info["coord_raw"])
        if not cids:
            return None
        return np.concatenate(cids)

    def summary(self) -> str:
        slice_info = f"[{self._slice_start} .. {self._slice_end}] (inclusive)" if self._slice_start is not None else "N/A"
        lines = [
            f"Variable         : {self.var_name}",
            f"Base dir         : {self.base_dir}",
            f"Ranks            : {self.num_ranks}",
            f"Local time len   : {self.time_len}",
            f"Time slice idx   : {slice_info}",
            f"First time (loc) : {self._time_datetimes[0] if self._time_datetimes else 'N/A'}",
            f"Last  time (loc) : {self._time_datetimes[-1] if self._time_datetimes else 'N/A'}",
            f"Map shape        : {self._map_shape}",
            f"Coord converter  : {'yes' if self._coord_converter is not None else 'no'}",
        ]
        for i, info in enumerate(self._rank_files):
            lines.append(
                f"  - rank[{i}]: file={info['path'].name}, saved_points={info['saved_points']}, "
                f"trials={'yes (' + str(info['n_trials']) + ')' if info['has_trials'] else 'no'}, "
                f"levels={'yes (' + str(info['n_levels']) + ')' if info['has_levels'] else 'no'}, "
                f"coord={info['coord_name'] or 'N/A'}"
            )
        print("\n".join(lines))
