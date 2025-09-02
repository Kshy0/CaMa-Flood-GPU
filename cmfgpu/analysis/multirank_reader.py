# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np


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
        rank_infos: List[dict] = []
        rank_re = re.compile(rf"{re.escape(self.var_name)}_rank(\d+)\.nc$")

        for fp in files:
            try:
                with nc.Dataset(fp, "r") as ds:
                    if self.var_name not in ds.variables:
                        continue
                    var = ds.variables[self.var_name]
                    dims = var.dimensions
                    has_levels = len(dims) == 3 and dims[-1] == "levels"
                    n_levels = int(ds.dimensions["levels"].size) if has_levels else 0
                    saved_points = int(ds.dimensions["saved_points"].size)

                    coord_name = self._select_coord_name(ds, saved_points)
                    coord_raw = None
                    if coord_name is not None:
                        coord_raw = np.array(ds.variables[coord_name][:])

                    rank_infos.append(
                        {
                            "path": fp,
                            "saved_points": saved_points,
                            "has_levels": has_levels,
                            "n_levels": n_levels,
                            "coord_name": coord_name,
                            "coord_raw": coord_raw,
                            "x": None,
                            "y": None,
                        }
                    )
            except Exception as e:
                print(f"Warning: skipping file {fp.name}, reason: {e}")

        def _rank_key(info: dict):
            m = rank_re.search(info["path"].name)
            return int(m.group(1)) if m else 1_000_000

        rank_infos.sort(key=_rank_key)
        return rank_infos

    def _read_time_axis(self) -> None:
        """
        Read the time axis from the first rank file; validate / truncate against others.
        Produce:
          - self._time_values_num
          - self._time_datetimes (naive)
          - self._time_units / _time_calendar
          - self._time_len
        """
        if not self._rank_files:
            raise RuntimeError("No rank files loaded.")

        with nc.Dataset(self._rank_files[0]["path"], "r") as ds0:
            tvar = ds0.variables["time"]
            self._time_units = getattr(tvar, "units")
            self._time_calendar = getattr(tvar, "calendar", "standard")
            t0 = np.array(tvar[:])
            dt0 = nc.num2date(t0, units=self._time_units, calendar=self._time_calendar)
            self._time_datetimes = [
                datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in dt0
            ]
            self._time_values_num = t0
            self._time_len = len(t0)

        for info in self._rank_files[1:]:
            with nc.Dataset(info["path"], "r") as dsi:
                tvar = dsi.variables["time"]
                if len(tvar) != self._time_len:
                    print(
                        f"Warning: {info['path'].name} has {len(tvar)} time steps, "
                        f"mismatch with first file {self._time_len}. Truncating to min length."
                    )
                    self._time_len = min(self._time_len, len(tvar))
                units = getattr(tvar, "units")
                cal = getattr(tvar, "calendar", "standard")
                if units != self._time_units or cal != self._time_calendar:
                    print(
                        f"Warning: {info['path'].name} time units/calendar mismatch; using first file settings: "
                        f"units={self._time_units}, calendar={self._time_calendar}"
                    )

        if self._time_len < len(self._time_values_num):
            self._time_values_num = self._time_values_num[: self._time_len]
            self._time_datetimes = self._time_datetimes[: self._time_len]

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
        start = self._slice_start
        stop_exclusive = self._slice_end + 1
        for info in self._rank_files:
            if info["saved_points"] == 0:
                info["cache"] = None
                continue
            try:
                with nc.Dataset(info["path"], "r") as ds:
                    var = ds.variables[self.var_name]
                    if info["has_levels"]:
                        data = var[start:stop_exclusive, :, :]
                    else:
                        data = var[start:stop_exclusive, :]
                    info["cache"] = np.array(data, copy=True)
            except Exception as e:
                print(f"Warning: failed to cache {info['path'].name}: {e}")
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
        time_range: Optional[Tuple[datetime, datetime]] = None,
        cache_enabled: bool = False,
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
            start_dt, end_dt = time_range
            if start_dt > end_dt:
                raise ValueError("time_range start must be <= end (closed interval).")

            first_dt = self._time_datetimes[0]
            last_dt = self._time_datetimes[-1]
            # New strict behavior: raise if outside coverage
            if start_dt < first_dt or end_dt > last_dt:
                raise ValueError(
                    f"time_range outside available coverage. "
                    f"Requested [{start_dt} .. {end_dt}] but coverage is [{first_dt} .. {last_dt}]."
                )

            # Locate left boundary: exact DT >= start_dt (since start_dt within range it will exist)
            dts = self._time_datetimes
            left = None
            for i, dt in enumerate(dts):
                if dt >= start_dt:
                    left = i
                    break
            if left is None:
                raise ValueError("Failed to locate start index (unexpected).")

            # Locate right boundary: last dt <= end_dt
            right = None
            for j in range(len(dts) - 1, -1, -1):
                if dts[j] <= end_dt:
                    right = j
                    break
            if right is None or right < left:
                raise ValueError("Failed to locate end index (unexpected).")

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
    def get_vector(
        self,
        t_index: int,
        level: Optional[int] = None,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        if t_index < 0 or t_index >= self._time_len:
            raise IndexError(f"t_index out of range [0, {self._time_len - 1}]")
        orig_time = int(self._t_indices[t_index])
        parts: List[np.ndarray] = []
        for info in self._rank_files:
            if info["saved_points"] == 0:
                parts.append(np.empty((0,), dtype=dtype or np.float32))
                continue
            cache_arr = info.get("cache")
            if cache_arr is not None:
                if info["has_levels"]:
                    data = cache_arr[t_index, :, level if level is not None else 0]
                else:
                    data = cache_arr[t_index, :]
            else:
                with nc.Dataset(info["path"], "r") as ds:
                    var = ds.variables[self.var_name]
                    if info["has_levels"]:
                        data = var[orig_time, :, level if level is not None else 0]
                    else:
                        data = var[orig_time, :]
            arr = np.array(data, copy=False)
            if dtype is not None:
                arr = arr.astype(dtype, copy=False)
            parts.append(arr)
        return np.concatenate(parts, axis=0) if parts else np.array([])

    def get_grid(
        self,
        t_index: int,
        level: Optional[int] = None,
        fill_value: float = np.nan,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        if self._map_shape is None:
            raise RuntimeError("map_shape is not set; cannot project to grid.")
        if t_index < 0 or t_index >= self._time_len:
            raise IndexError(f"t_index out of range [0, {self._time_len - 1}]")

        orig_time = int(self._t_indices[t_index])
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
                if info["has_levels"]:
                    if level is None:
                        raise ValueError("This variable has 'levels'; please specify 'level'.")
                    vals = cache_arr[t_index, :, level]
                else:
                    vals = cache_arr[t_index, :]
            else:
                with nc.Dataset(info["path"], "r") as ds:
                    var = ds.variables[self.var_name]
                    if info["has_levels"]:
                        if level is None:
                            raise ValueError("This variable has 'levels'; please specify 'level'.")
                        vals = var[orig_time, :, level]
                    else:
                        vals = var[orig_time, :]
            grid[x, y] = np.array(vals, copy=False)
        return grid

    def get_series(
        self,
        points: Union[np.ndarray, Sequence[np.ndarray]],
        level: Optional[int] = None,
        fill_value: float = np.nan,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        def _as_list(v):
            return [np.asarray(a) for a in v] if isinstance(v, (list, tuple)) else [np.asarray(v)]
        arr_list = _as_list(points)
        if not arr_list:
            return np.full((self._time_len, 0), fill_value, dtype=dtype or np.float32)

        def _kind(a: np.ndarray) -> str:
            if a.ndim == 2 and a.shape[1] == 2:
                return "xy"
            if a.ndim == 1:
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
        out = np.full((self._time_len, N), fill_value, dtype=dtype or np.float32)
        col_to_hits: List[Optional[Tuple[int, int]]] = [None] * N

        # Map queries to (rank_idx, local_index)
        if use_xy:
            for r_idx, info in enumerate(self._rank_files):
                if info["saved_points"] == 0:
                    continue
                x, y = info.get("x"), info.get("y")
                if x is None or y is None:
                    continue
                for c, (qx, qy) in enumerate(queries):
                    if col_to_hits[c] is not None:
                        continue
                    matches = np.nonzero((x == qx) & (y == qy))[0]
                    if matches.size:
                        col_to_hits[c] = (r_idx, int(matches[0]))
        else:
            for r_idx, info in enumerate(self._rank_files):
                if info["saved_points"] == 0 or info["coord_raw"] is None:
                    continue
                raw = np.asarray(info["coord_raw"]).ravel()
                for c, qid in enumerate(queries):
                    if col_to_hits[c] is not None:
                        continue
                    matches = np.nonzero(raw == qid)[0]
                    if matches.size:
                        col_to_hits[c] = (r_idx, int(matches[0]))

        rank_to_cols: dict[int, List[Tuple[int, int]]] = {}
        for col, hit in enumerate(col_to_hits):
            if hit is None:
                continue
            r_idx, li = hit
            rank_to_cols.setdefault(r_idx, []).append((col, li))

        # Fast path: if cached in memory
        for r_idx, pairs in rank_to_cols.items():
            info = self._rank_files[r_idx]
            cache_arr = info.get("cache")
            if cache_arr is not None:
                if info["has_levels"]:
                    if level is None:
                        raise ValueError("This variable has 'levels'; specify `level`.")
                    for col, li in pairs:
                        out[:, col] = np.asarray(cache_arr[:, li, level], dtype=dtype or np.float32)
                else:
                    for col, li in pairs:
                        out[:, col] = np.asarray(cache_arr[:, li], dtype=dtype or np.float32)
                continue

            # No cache: minimize I/O by opening once and slicing contiguous time window
            if self._slice_start is None or self._slice_end is None:
                raise RuntimeError("Internal error: time slice is not set.")
            t0 = int(self._slice_start)
            t1 = int(self._slice_end) + 1  # exclusive
            idx = np.array([li for (_, li) in pairs], dtype=np.int64)

            with nc.Dataset(info["path"], "r") as ds:
                var = ds.variables[self.var_name]
                if info["has_levels"]:
                    if level is None:
                        raise ValueError("This variable has 'levels'; specify `level`.")
                    # Read (T, K) in one go
                    vals = np.asarray(var[t0:t1, idx, level])  # shape (T, K)
                else:
                    vals = np.asarray(var[t0:t1, idx])         # shape (T, K)

            # Scatter to output columns
            for k, (col, _) in enumerate(pairs):
                out[:, col] = vals[:, k].astype(dtype or np.float32, copy=False)

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
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (8, 6),
        as_scatter_if_no_map: bool = True,
        s: float = 1.0,
    ) -> None:
        if t_index < 0 or t_index >= self._time_len:
            raise IndexError(f"t_index out of range [0, {self._time_len - 1}]")
        t_str = self.times[t_index].isoformat() if self.times else f"t={t_index}"

        fig, ax = plt.subplots(figsize=figsize)
        if self.map_shape is not None:
            grid = self.get_grid(t_index, level=level)
            im = ax.imshow(grid.T, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{self.var_name} @ {t_str}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
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
                    if info["has_levels"]:
                        vv = cache_arr[t_index, :, level if level is not None else 0]
                    else:
                        vv = cache_arr[t_index, :]
                else:
                    orig_t = int(self._t_indices[t_index])
                    with nc.Dataset(info["path"], "r") as ds:
                        var = ds.variables[self.var_name]
                        if info["has_levels"]:
                            vv = var[orig_t, :, level] if level is not None else var[orig_t, :, 0]
                        else:
                            vv = var[orig_t, :]
                vals.append(np.array(vv))
            x_all = np.concatenate(xs) if xs else np.array([])
            y_all = np.concatenate(ys) if ys else np.array([])
            v_all = np.concatenate(vals) if vals else np.array([])
            sc = ax.scatter(x_all, y_all, c=v_all, s=s, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{self.var_name} (scatter) @ {t_str}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        else:
            raise RuntimeError("Cannot plot without map_shape and scatter fallback disabled.")
        fig.tight_layout()

    def animate(
        self,
        out_path: Union[str, Path],
        level: Optional[int] = None,
        x_range: Optional[Tuple[int, int]] = None,
        y_range: Optional[Tuple[int, int]] = None,
        t_range: Optional[Tuple[int, int]] = None,
        fps: int = 10,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (8, 6),
    ) -> None:
        if self._map_shape is None:
            raise RuntimeError("Animation requires map_shape.")
        t_start = 0 if t_range is None else max(0, int(t_range[0]))
        t_end = self._time_len if t_range is None else min(self._time_len, int(t_range[1]))
        if t_start >= t_end:
            raise ValueError("Invalid t_range: ensure t_start < t_end")

        nx_, ny_ = self._map_shape
        xmin = 0 if x_range is None else max(0, int(x_range[0]))
        xmax = nx_ - 1 if x_range is None else min(nx_ - 1, int(x_range[1]))
        ymin = 0 if y_range is None else max(0, int(y_range[0]))
        ymax = ny_ - 1 if y_range is None else min(ny_ - 1, int(y_range[1]))
        if xmin > xmax or ymin > ymax:
            raise ValueError("Invalid x_range or y_range")

        first_grid = self.get_grid(t_start, level=level)
        window = first_grid[xmin:xmax + 1, ymin:ymax + 1]
        if vmin is None:
            vmin = np.nanmin(window) if np.isfinite(window).any() else 0.0
        if vmax is None:
            vmax = np.nanmax(window) if np.isfinite(window).any() else 1.0
        if not (vmax > vmin):
            vmax = vmin + 1.0

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(window.T, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ttl = ax.set_title(f"{self.var_name} @ {self.times[t_start].isoformat()}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.tight_layout()

        def _update(frame_idx: int):
            ti = t_start + frame_idx
            grid = self.get_grid(ti, level=level)
            win = grid[xmin:xmax + 1, ymin:ymax + 1]
            im.set_data(win.T)
            ttl.set_text(f"{self.var_name} @ {self.times[ti].isoformat()}")
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

    # ----------------------------------------------------------------------------------
    # Export
    # ----------------------------------------------------------------------------------
    def export_to_cama_bin(
        self,
        out_dir: Union[str, Path],
        out_var_name: str,
        t_range: Optional[Tuple[int, int]] = None,
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
                    grid = self.get_grid(ti, level=None, fill_value=fill_value, dtype=dtype)
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
                f"levels={'yes' if info['has_levels'] else 'no'}, coord={info['coord_name'] or 'N/A'}"
            )
        print("\n".join(lines))
