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
        Manage NetCDF outputs written per-rank by StatisticsAggregator.

        Key features:
            - Auto-detect rank files: {var_name}_rank{rank}.nc.
            - Derive (x, y) locations for saved_points using exactly one of:
                • coord_source=(nx, ny) tuple
                • coord_source=NetCDF file path with nx/ny or map_shape
                • coord_source=converter function taking a single argument: converter(coord_raw) -> (x, y)
            - Basic visualization (single frame and animation).
            - Export combined grid-time data to CaMa-Flood-compatible .bin files.

        Assumptions:
            - Main variable inside each NetCDF: var_name (exactly as provided)
            - Dimensions: time (unlimited), saved_points [, levels]
            - Optional coordinate variable aligned to ('saved_points',) is used as save_coord.
        """
    
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

    # -----------------------------
    # Internal: scan and parse
    # -----------------------------
    def _scan_rank_files(self) -> List[dict]:
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
        if not self._rank_files:
            raise RuntimeError("No rank files loaded.")

        with nc.Dataset(self._rank_files[0]["path"], "r") as ds0:
            tvar = ds0.variables["time"]
            self._time_units = getattr(tvar, "units")
            self._time_calendar = getattr(tvar, "calendar", "standard")
            t0 = np.array(tvar[:])
            self._time_values_num = t0
            self._time_datetimes = nc.num2date(
                t0, units=self._time_units, calendar=self._time_calendar
            ).tolist()
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
        """
        Compute (x, y) for each rank:
          - If custom converter provided, use converter(coord_raw) -> (x, y).
          - Else, if map_shape exists and coord_raw is a linear index in [0, nx*ny), use unravel_index.
          - Else, keep None to indicate no grid projection is available.
        """
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
        time_range: Optional[Tuple[int, int]] = None,
        cache: bool = False,
    ):
        """
        Args:
          base_dir: Directory containing per-rank NetCDF files.
          var_name: Full variable name; files are matched as {var_name}_rank*.nc, and the internal variable has the same name.
          coord_name: Explicit variable name to use as save_coord (optional).
          coord_source: Provide exactly one of: (nx, ny) tuple, NetCDF path (str|Path), or a converter function(coord_raw)->(x,y).
          time_range: Optional (start, end) indices to restrict the time axis; end is exclusive.
          cache: If True, preload variable data of each rank into memory for the chosen time range.
        """
        self.base_dir = Path(base_dir)
        self.var_name = var_name
        self.coord_name = coord_name

        self._map_shape: Optional[Tuple[int, int]] = None
        self._coord_converter: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None

        # Internal state
        self._rank_files: List[dict] = []
        self._t_offset = 0
        self._cache_enabled = bool(cache)

        # Interpret coord_source
        if coord_source is not None:
            if callable(coord_source):
                self._coord_converter = coord_source
            elif isinstance(coord_source, (str, Path)):
                self.load_map_shape_from_nc(coord_source)
            else:
                nx, ny = coord_source  # type: ignore
                self.set_map_shape((int(nx), int(ny)))

        # Scan rank files
        self._rank_files = self._scan_rank_files()
        if not self._rank_files:
            raise FileNotFoundError(
                f"No files found in {self.base_dir} matching: {self.var_name}_rank*.nc"
            )

        # Read time axis
        self._read_time_axis()

        # Apply time range (half-open) if provided
        if time_range is not None:
            ts = max(0, int(time_range[0]))
            te = min(self._time_len, int(time_range[1]))
            if ts >= te:
                raise ValueError("Invalid time_range: ensure start < end within available time length.")
            self._t_offset = ts
            self._time_values_num = self._time_values_num[ts:te]
            self._time_datetimes = self._time_datetimes[ts:te]
            self._time_len = te - ts

        # Compute (x, y) after files are known
        self._compute_all_xy(force=True)

        # Preload cache if requested
        if self._cache_enabled:
            self._preload_cache()

    def _preload_cache(self) -> None:
        """Preload the variable into memory for the configured time slice."""
        for info in self._rank_files:
            if info["saved_points"] == 0:
                info["cache"] = None
                continue
            try:
                with nc.Dataset(info["path"], "r") as ds:
                    var = ds.variables[self.var_name]
                    if info["has_levels"]:
                        data = var[self._t_offset : self._t_offset + self._time_len, :, :]
                    else:
                        data = var[self._t_offset : self._t_offset + self._time_len, :]
                    info["cache"] = np.array(data, copy=True)
            except Exception as e:
                print(f"Warning: failed to cache {info['path'].name}: {e}")
                info["cache"] = None

    # -----------------------------
    # Core getters (concise names)
    # -----------------------------
    def get_vector(
        self,
        t_index: int,
        level: Optional[int] = None,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        if t_index < 0 or t_index >= self._time_len:
            raise IndexError(f"t_index out of range [0, {self._time_len - 1}]")
        parts: List[np.ndarray] = []
        for info in self._rank_files:
            if info["saved_points"] == 0:
                parts.append(np.empty((0,), dtype=dtype or np.float32))
                continue
            cache_arr = info.get("cache") if isinstance(info, dict) else None
            if cache_arr is not None:
                if info["has_levels"]:
                    data = cache_arr[t_index, :, level if level is not None else 0]
                else:
                    data = cache_arr[t_index, :]
            else:
                with nc.Dataset(info["path"], "r") as ds:
                    var = ds.variables[self.var_name]
                    ti = self._t_offset + t_index
                    if info["has_levels"]:
                        if level is None:
                            data = var[ti, :, 0]
                        else:
                            data = var[ti, :, level]
                    else:
                        data = var[ti, :]
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
        nx_, ny_ = self._map_shape
        grid = np.full((nx_, ny_), fill_value, dtype=dtype or np.float32)
        for info in self._rank_files:
            if info["saved_points"] == 0:
                continue
            x = info.get("x")
            y = info.get("y")
            if x is None or y is None:
                raise RuntimeError(f"{info['path'].name} missing (x,y); set map_shape or coord converter.")
            cache_arr = info.get("cache") if isinstance(info, dict) else None
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
                    ti = self._t_offset + t_index
                    if info["has_levels"]:
                        if level is None:
                            raise ValueError("This variable has 'levels'; please specify 'level'.")
                        vals = var[ti, :, level]
                    else:
                        vals = var[ti, :]
            vals = np.array(vals, copy=False)
            grid[x, y] = vals
        return grid

    def get_series(
        self,
        points: Union[np.ndarray, Sequence[np.ndarray]],
        level: Optional[int] = None,
        fill_value: float = np.nan,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        # Input normalization: XY (N,2) arrays or ID (N,) arrays, or lists of them
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
            xy_pairs: List[Tuple[int, int]] = []
            for a in arr_list:
                if not (a.ndim == 2 and a.shape[1] == 2):
                    raise ValueError(f"Invalid XY array shape: {a.shape}. Expected (N,2).")
                a2 = np.asarray(a, dtype=np.int64)
                xy_pairs.extend([(int(px), int(py)) for px, py in a2])
            queries = xy_pairs
        else:
            ids_all: List[int] = []
            for a in arr_list:
                if a.ndim != 1:
                    raise ValueError(f"Invalid ID array shape: {a.shape}. Expected 1D.")
                ids_all.extend([int(v) for v in np.asarray(a, dtype=np.int64).ravel()])
            queries = ids_all

        N = len(queries)
        out = np.full((self._time_len, N), fill_value, dtype=dtype or np.float32)

        # Build mapping (col -> (rank_idx, local_idx))
        col_to_hits: List[Optional[Tuple[int, int]]] = [None] * N
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

        # Group by rank and load
        rank_to_cols: dict[int, List[Tuple[int, int]]] = {}
        for col, hit in enumerate(col_to_hits):
            if hit is None:
                continue
            r_idx, li = hit
            rank_to_cols.setdefault(r_idx, []).append((col, li))

        for r_idx, pairs in rank_to_cols.items():
            info = self._rank_files[r_idx]
            cache_arr = info.get("cache") if isinstance(info, dict) else None
            if cache_arr is not None:
                if info["has_levels"]:
                    if level is None:
                        raise ValueError("This variable has 'levels'; please specify `level`.")
                    for col, li in pairs:
                        vec = cache_arr[:, li, level]
                        out[:, col] = np.asarray(vec, dtype=dtype or np.float32)
                else:
                    for col, li in pairs:
                        vec = cache_arr[:, li]
                        out[:, col] = np.asarray(vec, dtype=dtype or np.float32)
            else:
                with nc.Dataset(info["path"], "r") as ds:
                    var = ds.variables[self.var_name]
                    s = slice(self._t_offset, self._t_offset + self._time_len)
                    if info["has_levels"]:
                        if level is None:
                            raise ValueError("This variable has 'levels'; please specify `level`.")
                        for col, li in pairs:
                            vec = var[s, li, level]
                            out[:, col] = np.asarray(vec, dtype=dtype or np.float32)
                    else:
                        for col, li in pairs:
                            vec = var[s, li]
                            out[:, col] = np.asarray(vec, dtype=dtype or np.float32)
        return out

    # -----------------------------
    # Basic info
    # -----------------------------
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
        """Set map shape (nx, ny)."""
        if len(map_shape) != 2:
            raise ValueError("map_shape must be a (nx, ny) tuple.")
        self._map_shape = (int(map_shape[0]), int(map_shape[1]))
        # Recompute (x, y) only if rank files are already available
        if getattr(self, "_rank_files", None):
            self._compute_all_xy(force=True)

    def load_map_shape_from_nc(
        self,
        nc_path: Union[str, Path],
    ) -> None:
        """
        Load (nx, ny) from a NetCDF file, checking (in order):
          - global attributes 'nx' and 'ny'
          - variable 'nx' and 'ny' (scalar)
          - variable 'map_shape' (length >= 2)
          - global attribute 'map_shape' (length >= 2)
          - dimensions 'nx' and 'ny'
          - dimensions 'x' and 'y'
          - dimensions 'lon' and 'lat'
        """
        p = Path(nc_path)
        if not p.exists():
            raise FileNotFoundError(f"NetCDF file not found: {p}")

        nx = ny = None
        with nc.Dataset(p, "r") as ds:
            # Global attributes
            attrs = {a: ds.getncattr(a) for a in ds.ncattrs()}
            if "nx" in attrs and "ny" in attrs:
                nx = int(attrs["nx"])
                ny = int(attrs["ny"])

            # Variables 'nx'/'ny'
            if (nx is None or ny is None) and "nx" in ds.variables and "ny" in ds.variables:
                try:
                    nx = int(np.array(ds.variables["nx"][:]).squeeze())
                    ny = int(np.array(ds.variables["ny"][:]).squeeze())
                except Exception:
                    pass

            # Variable 'map_shape'
            if (nx is None or ny is None) and "map_shape" in ds.variables:
                arr = np.array(ds.variables["map_shape"][:]).squeeze()
                if arr.size >= 2:
                    nx = int(arr[0])
                    ny = int(arr[1])

            # Attribute 'map_shape'
            if (nx is None or ny is None) and "map_shape" in attrs:
                arr = np.array(attrs["map_shape"]).squeeze()
                if np.size(arr) >= 2:
                    nx = int(np.ravel(arr)[0])
                    ny = int(np.ravel(arr)[1])

            # Dimensions: nx/ny, x/y, lon/lat
            dim_pairs = [("nx", "ny"), ("x", "y"), ("lon", "lat")]
            if nx is None or ny is None:
                dim_names = ds.dimensions.keys()
                for a, b in dim_pairs:
                    if a in dim_names and b in dim_names:
                        nx = int(ds.dimensions[a].size)
                        ny = int(ds.dimensions[b].size)
                        break

        if nx is None or ny is None:
            raise KeyError(
                "Could not find nx/ny or map_shape (as attrs/vars/dims) in NetCDF."
            )

        self.set_map_shape((nx, ny))

    def set_coord_converter(
        self,
        converter: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """Set a custom single-argument converter: converter(coord_raw) -> (x, y)."""
        self._coord_converter = converter
        self._compute_all_xy(force=True)

    # -----------------------------
    # Data read (new concise APIs are defined above)
    # -----------------------------

    # -----------------------------
    # Visualization
    # -----------------------------
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
        """Plot a single time slice; defaults to the first time step."""
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
                    raise RuntimeError(
                        "map_shape is not set and (x, y) are not available from a converter."
                    )
                xs.append(info["x"])
                ys.append(info["y"])
                cache_arr = info.get("cache") if isinstance(info, dict) else None
                if cache_arr is not None:
                    if info["has_levels"]:
                        vv = cache_arr[t_index, :, level if level is not None else 0]
                    else:
                        vv = cache_arr[t_index, :]
                else:
                    with nc.Dataset(info["path"], "r") as ds:
                        var = ds.variables[self.var_name]
                        ti = self._t_offset + t_index
                        if info["has_levels"]:
                            vv = var[ti, :, level] if level is not None else var[ti, :, 0]
                        else:
                            vv = var[ti, :]
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
            raise RuntimeError("Cannot plot without map_shape and without scatter fallback.")

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
        """Create an animation over a given spatial and temporal range (requires map_shape)."""
        if self._map_shape is None:
            raise RuntimeError("Animation requires map_shape. Set it or provide a NetCDF shape file.")

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
        window = first_grid[xmin : xmax + 1, ymin : ymax + 1]
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
            win = grid[xmin : xmax + 1, ymin : ymax + 1]
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
                raise RuntimeError("Cannot create GIF. Install Pillow or choose .mp4 with ffmpeg.")
            ani.save(out_path, writer=writer)
        else:
            try:
                Writer = animation.writers["ffmpeg"]
                writer = Writer(fps=fps, metadata=dict(artist="MultiRankStatsReader"))
            except Exception:
                raise RuntimeError("ffmpeg writer not found. Install ffmpeg or write .gif instead.")
            ani.save(out_path, writer=writer)

        plt.close(fig)

    # -----------------------------
    # Export to CaMa-Flood binary
    # -----------------------------
    def export_to_cama_bin(
        self,
        out_dir: Union[str, Path],
        out_var_name: str,
        t_range: Optional[Tuple[int, int]] = None,
        fill_value: float = 1e20,
        dtype: np.dtype = np.float32,
        progress: bool = True,
    ) -> None:
        """
        Export grids grouped by calendar year to raw .bin files, named as "{var}{year}.bin".
        The 'out_var_name' parameter is used for filenames (does not use self.var_name).
        Variables with a 'levels' dimension are not supported.

        Each year's file contains all time steps in that year, appended in time order.
        """
        if self._map_shape is None:
            raise RuntimeError("map_shape is required to export .bin files.")

        # Disallow variables with levels
        if any(info["has_levels"] for info in self._rank_files):
            raise ValueError("export_to_cama_bin does not support variables with a 'levels' dimension.")

        if not self.times or self._time_len == 0:
            raise RuntimeError("No time axis available for export.")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        t_start = 0 if t_range is None else max(0, int(t_range[0]))
        t_end = self._time_len if t_range is None else min(self._time_len, int(t_range[1]))
        if t_start >= t_end:
            raise ValueError("Invalid t_range: ensure t_start < t_end")

        # Group indices by year
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
                    arr = np.asfortranarray(grid)
                    fw.write(arr.tobytes(order="F"))

    # -----------------------------
    # Utilities
    # -----------------------------
    def get_all_coords_xy(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return concatenated (x, y) across ranks if all ranks provide them."""
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
        lines = [
            f"Variable         : {self.var_name}",
            f"Base dir         : {self.base_dir}",
            f"Ranks            : {self.num_ranks}",
            f"Time length      : {self.time_len}",
            f"Map shape        : {self._map_shape}",
            f"Coord converter  : {'yes' if self._coord_converter is not None else 'no'}",
        ]
        for i, info in enumerate(self._rank_files):
            lines.append(
                f"  - rank[{i}]: file={info['path'].name}, saved_points={info['saved_points']}, "
                f"levels={'yes' if info['has_levels'] else 'no'}, coord={info['coord_name'] or 'N/A'}"
            )
        print("\n".join(lines))
