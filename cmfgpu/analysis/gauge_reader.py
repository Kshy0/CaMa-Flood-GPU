# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import netCDF4 as nc
import numpy as np


@dataclass
class GaugeSeries:
    """
    Simple container for gauge time series.

    Fields:
      - gauge_id: integer gauge ID
      - dates: list of datetime (naive, assumed in UTC or local as-is)
      - values: numpy array of floats (np.nan for missing)
      - units: string, e.g., "m3/s"
      - meta: optional free-form metadata
    """

    gauge_id: str
    dates: List[datetime]
    values: np.ndarray
    units: str = "m3/s"
    meta: Optional[dict] = None

    def clip(self, start: Optional[datetime], end: Optional[datetime]) -> "GaugeSeries":
        if start is None and end is None:
            return self
        dates_arr = np.array(self.dates, dtype="datetime64[ns]")
        mask = np.ones(dates_arr.shape[0], dtype=bool)
        if start is not None:
            mask &= dates_arr >= np.datetime64(start)
        if end is not None:
            mask &= dates_arr <= np.datetime64(end)
        return GaugeSeries(
            gauge_id=self.gauge_id,
            dates=[d for d, m in zip(self.dates, mask.tolist()) if m],
            values=self.values[mask],
            units=self.units,
            meta=self.meta,
        )

def default_grdc_resolver(gauge_id: str) -> str: 
    return f"{gauge_id}_Q_Day.Cmd.txt"


def parse_grdc_text(gauge_id: str, text: str) -> GaugeSeries:
    """Parse raw GRDC daily discharge text into a GaugeSeries.

    Parameters
    ----------
    gauge_id : str
        Station identifier.
    text : str
        Full content of a GRDC daily .txt file.

    Returns
    -------
    GaugeSeries
    """
    dates: List[datetime] = []
    vals: List[float] = []
    units = "m3/s"
    meta: dict = {"source": "GRDC"}

    in_data = False
    for raw in text.split("\n"):
        line = raw.strip("\n")
        if not in_data:
            if line.startswith("#"):
                if "Latitude" in line:
                    try:
                        meta["lat"] = float(line.split(":")[-1])
                    except Exception:
                        pass
                if "Longitude" in line:
                    try:
                        meta["lon"] = float(line.split(":")[-1])
                    except Exception:
                        pass
                if "Catchment area" in line:
                    try:
                        meta["area_km2"] = float(line.split(":")[-1])
                    except Exception:
                        pass
                if "Unit of measure" in line and ("m3" in line or "m?s" in line):
                    units = "m3/s"
                continue
            if line.upper().startswith("YYYY-") or line.strip().upper() == "# DATA":
                in_data = True
                continue
            else:
                continue

        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) >= 3:
            date_str, time_str, val_str = parts[:3]
        else:
            toks = line.split()
            if len(toks) < 2:
                continue
            date_str, val_str = toks[0], toks[-1]
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            try:
                dt = datetime.strptime(date_str, "%d.%m.%Y")
            except Exception:
                continue

        val_str = val_str.replace(",", ".")
        try:
            v = float(val_str)
        except Exception:
            continue
        if v in (-999.0, -999.000, -9999.0, -9999.000) or int(v) in (-999, -9999):
            v = np.nan

        dates.append(dt)
        vals.append(v)

    return GaugeSeries(
        gauge_id=gauge_id,
        dates=dates,
        values=np.array(vals, dtype=float),
        units=units,
        meta=meta if meta else None,
    )


def load_grdc(gauge_id: str, file_path: Union[str, Path]) -> GaugeSeries:
    """Load a GRDC daily discharge file → GaugeSeries."""
    file_path = Path(file_path)
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    return parse_grdc_text(gauge_id, text)


def read_mapdim(map_dir: Union[str, Path]) -> Tuple[int, int]:
    """Read (nx, ny) from CaMa-Flood mapdim.txt.

    Parameters
    ----------
    map_dir : path
        Directory containing ``mapdim.txt``.

    Returns
    -------
    (nx, ny) : tuple of int
    """
    with open(Path(map_dir) / "mapdim.txt") as f:
        nx = int(f.readline().split("!!")[0].strip())
        ny = int(f.readline().split("!!")[0].strip())
    return nx, ny


def parse_alloc_txt(alloc_path: Union[str, Path]) -> Dict[int, dict]:
    """Parse CaMa-Flood GRDC_alloc.txt → per-station allocation info.

    Parameters
    ----------
    alloc_path : path
        Path to ``GRDC_alloc.txt``.

    Returns
    -------
    dict
        ``{station_id: {ix1, iy1, ix2, iy2, area_cama, error}}``
        Grid indices are 1-based (Fortran convention, as in the file).
    """
    result: Dict[int, dict] = {}
    with open(alloc_path) as f:
        lines = f.readlines()
    for line in lines[1:]:
        tok = line.split()
        if len(tok) < 14:
            continue
        sid = int(tok[0])
        result[sid] = {
            "ix1": int(tok[8]),
            "iy1": int(tok[9]),
            "ix2": int(tok[10]),
            "iy2": int(tok[11]),
            "area_cama": float(tok[5]),
            "error": float(tok[3]),
        }
    return result

class GaugeReader:
    """
    Generic gauge time series reader with an overridable loader.

    Default loader expects GRDC station .txt format like:
      - lines starting with '#' are comments
      - header row: "YYYY-MM-DD;hh:mm; Value"
      - data rows:  "1982-10-01;--:--;      0.000"
      - missing values: -999, -999.000, -9999, -9999.000 (treated as NaN)

    You can customize:
      - file_resolver: map gauge_id -> file path
      - loader: map (gauge_id, file_path) -> GaugeSeries
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        file_resolver: Callable[[str], Union[str, Path]] = default_grdc_resolver,
        loader: Callable[[str, Union[str, Path]], GaugeSeries] = load_grdc,
    ) -> None:
        self.base_dir = Path(base_dir)
        self._file_resolver = file_resolver
        self._loader = loader

        # Gauge meta state (populated via load_meta)
        self._gauge_ids: List[str] = []
        # gauge_id -> ((ix1, iy1), (ix2, iy2) or None)
        self._gauge_xy: Dict[str, Tuple[Tuple[int, int], Optional[Tuple[int, int]]]] = {}
        # gauge_id -> (area1, area2 or None)
        self._gauge_areas: Dict[str, Tuple[float, Optional[float]]] = {}
        # gauge_id -> [catchment_id(s)]
        self._gauge_catchments: Dict[str, List[int]] = {}
        # map shape for catchment id flattening (nx, ny)
        self._map_shape: Optional[Tuple[int, int]] = None

    # -----------------------------
    # Public API
    # -----------------------------
    def set_loader(self, loader: Callable[[str, Union[str, Path]], GaugeSeries]) -> None:
        """Override the loader used to parse a single gauge file."""
        self._loader = loader

    def set_file_resolver(self, resolver: Callable[[str], Union[str, Path]]) -> None:
        """Override how gauge_id is mapped to a file path."""
        self._file_resolver = resolver

    def resolve_path(self, gauge_id: str) -> Path:
        name_or_path = Path(self._file_resolver(gauge_id))
        p = name_or_path if name_or_path.is_absolute() else (self.base_dir / name_or_path)
        if not p.exists():
            raise FileNotFoundError(f"Gauge file not found for id={gauge_id}: {p}")
        return p

    def read(self, gauge_id: str, start: Optional[datetime] = None, end: Optional[datetime] = None) -> GaugeSeries:
        """Read a gauge by id and clip to [start, end] if provided."""
        file_path = self.resolve_path(gauge_id)
        series = self._loader(gauge_id, file_path)
        return series.clip(start, end)

    def _rebuild_catchments(self) -> None:
        if self._map_shape is None:
            return
        _, ny_ = self._map_shape
        self._gauge_catchments.clear()
        for gid, (xy1, xy2) in self._gauge_xy.items():
            ct1 = int(xy1[0]) * int(ny_) + int(xy1[1])
            cts = [ct1]
            if xy2 is not None:
                ct2 = int(xy2[0]) * int(ny_) + int(xy2[1])
                cts.append(ct2)
            self._gauge_catchments[gid] = cts

    # -----------------------------
    # Gauge meta loader and helpers
    # -----------------------------
    def set_map_shape(self, nx: int, ny: int) -> None:
        """Set map shape (nx, ny) used to compute catchment_id = ix * ny + iy (C-order)."""
        self._map_shape = (int(nx), int(ny))
        # Recompute catchment ids if meta already loaded
        if self._gauge_xy:
            self._rebuild_catchments()

    def load_map_shape_from_nc(self, nc_path: Union[str, Path]) -> None:
        """Load (nx, ny) from a NetCDF file if available in attributes/variables/dimensions."""
        if nc is None:
            raise RuntimeError("netCDF4 not available to read map shape; install netCDF4 or use set_map_shape().")
        p = Path(nc_path)
        if not p.exists():
            raise FileNotFoundError(f"NetCDF not found: {p}")
        nx_val = ny_val = None
        with nc.Dataset(p, "r") as ds:
            # try global attrs
            nx_val = getattr(ds, "nx", None)
            ny_val = getattr(ds, "ny", None)
            # try variables
            if (nx_val is None or ny_val is None) and "nx" in ds.variables and "ny" in ds.variables:
                try:
                    nx_val = int(ds["nx"][()])
                    ny_val = int(ds["ny"][()])
                except Exception:
                    pass
            # try dims
            if (nx_val is None or ny_val is None) and "nx" in ds.dimensions and "ny" in ds.dimensions:
                nx_val = int(len(ds.dimensions["nx"]))
                ny_val = int(len(ds.dimensions["ny"]))
            # common alternative x/y
            if (nx_val is None or ny_val is None) and "x" in ds.dimensions and "y" in ds.dimensions:
                nx_val = int(len(ds.dimensions["x"]))
                ny_val = int(len(ds.dimensions["y"]))
        if nx_val is None or ny_val is None:
            raise ValueError("Failed to get (nx, ny) from NetCDF. Provide explicitly via set_map_shape().")
        self.set_map_shape(nx_val, ny_val)

    def load_meta(
        self,
        meta_txt: Union[str, Path],
        shape_source: Optional[
            Union[
                Tuple[int, int],
                str,
                Path,
                Callable[[], Tuple[int, int]],
            ]
        ] = None,
        err_threshold: float = 0.1,
        skip_multi_catchment: bool = False,
    ) -> None:
        """
        Load gauge metadata from a whitespace-separated text file with columns including:
          ID lat lon err area_GRDC area_CaMa diff ups_num ix1 iy1 ix2 iy2 area1 area2

        Notes:
          - ix1/iy1 is the primary catchment grid.
          - ix2/iy2 can be negative (e.g., -999 or -9999) to indicate no second catchment.
          - catchment_id is computed as ix * ny + iy (C-order) using map shape (nx, ny).
          - shape_source allows unified input for map shape similar to a "coord_source" pattern.
            Accepted forms:
              * (nx, ny) tuple
              * str/Path to a NetCDF parameter file (from which nx, ny are inferred)
          - err_threshold: filter out gauges where abs(err) > err_threshold (default 0.05)
          - skip_multi_catchment: if True, skip rows where ix2/iy2 indicate a second catchment (default False)
        """
        meta_path = Path(meta_txt)
        if not meta_path.exists():
            raise FileNotFoundError(f"Gauge meta file not found: {meta_path}")

        # Resolve map shape if not already set, using unified shape_source
        if self._map_shape is None:
            if shape_source is not None:
                # Unified resolution path
                if isinstance(shape_source, tuple) and len(shape_source) == 2:
                    sx, sy = shape_source
                    self.set_map_shape(int(sx), int(sy))
                elif isinstance(shape_source, (str, Path)):
                    self.load_map_shape_from_nc(Path(shape_source))
                else:
                    raise TypeError(
                        "Unsupported shape_source type. Expected (nx, ny) tuple or str/Path."
                    )
            else:
                raise ValueError(
                    "Map shape (nx, ny) required. Provide shape_source first."
                )

        # Parse header to locate column indices
        with meta_path.open("r", encoding="utf-8", errors="ignore") as f:
            header_idx_map: Dict[str, int] = {}
            rows: List[List[str]] = []
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                if line.lstrip().startswith("#"):
                    continue
                # split by comma and/or whitespace
                toks = [t for t in re.split(r"[\s,]+", line.strip()) if t]
                # Detect header row by presence of required labels
                if ("ID" in toks and "ix1" in toks and "iy1" in toks):
                    header_idx_map = {name: toks.index(name) for name in toks}
                    continue
                # Otherwise it's a data row
                rows.append(toks)

        if not header_idx_map:
            # Fallback: assume fixed column order
            expected = [
                "ID","lat","lon","err","area_GRDC","area_CaMa","diff","ups_num",
                "ix1","iy1","ix2","iy2","area1","area2"
            ]
            header_idx_map = {name: i for i, name in enumerate(expected)}

        def get_val(toks: List[str], key: str, cast):
            idx = header_idx_map.get(key)
            if idx is None or idx >= len(toks):
                return None
            try:
                return cast(toks[idx])
            except Exception:
                return None

        self._gauge_ids.clear()
        self._gauge_xy.clear()
        self._gauge_areas.clear()
        self._gauge_catchments.clear()

        nx_, ny_ = self._map_shape  # type: ignore[misc]
        assert nx_ is not None and ny_ is not None

        # Temp storage for gauges per catchment
        temp_gauges = {}  # ct_id -> list of (gid, err, xy1, xy2, a1, a2)

        for toks in rows:
            gid = str(get_val(toks, "ID", str))
            if gid in ("None", "nan", "NaN"):
                continue
            err = get_val(toks, "err", float)
            if abs(err) > err_threshold:
                continue
            ix1 = get_val(toks, "ix1", int) - 1
            iy1 = get_val(toks, "iy1", int) - 1
            ix2 = get_val(toks, "ix2", int) - 1
            iy2 = get_val(toks, "iy2", int) - 1
            a1 = get_val(toks, "area1", float)
            a2 = get_val(toks, "area2", float)
            if ix1 is None or iy1 is None:
                continue
            xy1 = (int(ix1), int(iy1))
            xy2: Optional[Tuple[int, int]] = None
            if ix2 is not None and iy2 is not None and (ix2 >= 0) and (iy2 >= 0):
                xy2 = (int(ix2), int(iy2))

            if skip_multi_catchment and xy2 is not None:
                continue

            # Compute catchment id for xy1
            ct1 = int(xy1[0]) * int(ny_) + int(xy1[1])
            temp_gauges.setdefault(ct1, []).append((gid, err, xy1, xy2, a1, a2))

        # Now select one gauge per catchment with min abs(err)
        for ct_id, gauges in temp_gauges.items():
            if len(gauges) == 1:
                gid, err, xy1, xy2, a1, a2 = gauges[0]
            else:
                # Sort by abs(err), select the smallest
                gauges.sort(key=lambda x: abs(x[1]) if x[1] is not None else float('inf'))
                gid, err, xy1, xy2, a1, a2 = gauges[0]

            self._gauge_ids.append(gid)
            self._gauge_xy[gid] = (xy1, xy2)
            self._gauge_areas[gid] = (
                float(a1) if a1 is not None else float("nan"),
                float(a2) if (a2 is not None and (xy2 is not None)) else None,
            )

        # Compute catchment ids now that xy are known
        self._rebuild_catchments()

    # Accessors
    @property
    def gauge_ids(self) -> List[str]:
        """List of gauge IDs loaded from meta."""
        return list(self._gauge_ids)

    def get_xy(self, gauge_id: str) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        return self._gauge_xy[gauge_id]

    def get_catchments(self, gauge_id: str) -> List[int]:
        return self._gauge_catchments[gauge_id]

    def get_areas(self, gauge_id: str) -> Tuple[float, Optional[float]]:
        return self._gauge_areas[gauge_id]

    def get_available_cids(self) -> List[int]:
        """Get list of catchment IDs for gauges where the file exists."""
        available = []
        for gid in self._gauge_ids:
            try:
                self.resolve_path(gid)
                cts = self.get_catchments(gid)
                available.extend(cts)
            except FileNotFoundError:
                continue
        return list(set(available))

    # -----------------------------
    # Metrics
    # -----------------------------
    @staticmethod
    def nse(y_obs: np.ndarray, y_sim: np.ndarray) -> float:
        """Compute Nash-Sutcliffe Efficiency ignoring NaNs; assumes aligned time axis."""
        if y_obs.size == 0 or y_sim.size == 0:
            return float("nan")
        mask = np.isfinite(y_obs) & np.isfinite(y_sim)
        if mask.sum() < 2:
            return float("nan")
        o = y_obs[mask]
        s = y_sim[mask]
        denom = np.sum((o - np.mean(o)) ** 2)
        if denom == 0:
            return float("nan")
        num = np.sum((s - o) ** 2)
        return 1.0 - (num / denom)


# ──────────────────────────────────────────────────────────────────────────────
#  Batch extraction and NetCDF writing
# ──────────────────────────────────────────────────────────────────────────────

def scan_grdc_zips(grdc_dir: Union[str, Path]) -> Dict[str, List[Tuple[int, str]]]:
    """Discover all daily-discharge station files inside GRDC zip archives.

    Scans all ``*.zip`` files in *grdc_dir* for filenames matching
    ``{station_id}_Q_Day.Cmd.txt``.

    Parameters
    ----------
    grdc_dir : path
        Directory containing ``{country}.zip`` files.

    Returns
    -------
    dict
        ``{zip_stem: [(station_id, filename_in_zip), ...]}``
        Ready to pass to :func:`extract_grdc_from_zips`.
    """
    grdc_dir = Path(grdc_dir)
    result: Dict[str, List[Tuple[int, str]]] = {}
    for zf_path in sorted(grdc_dir.glob("*.zip")):
        entries: List[Tuple[int, str]] = []
        try:
            with zipfile.ZipFile(zf_path) as zf:
                for name in zf.namelist():
                    if "_Q_Day" not in name:
                        continue
                    # Extract station ID from e.g. "6342640_Q_Day.Cmd.txt"
                    stem = Path(name).name.split("_Q_Day")[0]
                    try:
                        sid = int(stem)
                    except ValueError:
                        continue
                    entries.append((sid, name))
        except Exception:
            continue
        if entries:
            result[zf_path.stem] = entries
    n_total = sum(len(v) for v in result.values())
    print(f"Scanned {len(result)} zips, found {n_total} daily station files")
    return result


def extract_grdc_from_zips(
    grdc_dir: Union[str, Path],
    zip_file_map: Dict[str, List[Tuple[int, str]]],
    time_start: str,
    time_end: str,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, dict]]:
    """Extract daily discharge from GRDC zip archives.

    Parameters
    ----------
    grdc_dir : path
        Directory containing ``{country}.zip`` files.
    zip_file_map : dict
        ``{zip_name: [(station_id, filename_in_zip), ...]}``
    time_start, time_end : str
        ISO date strings (inclusive) for clipping, e.g. ``"1979-01-01"``.

    Returns
    -------
    dict
        ``{station_id: (dates_datetime64, values_f32, meta_dict)}``
        where *meta_dict* may contain ``lat``, ``lon``, ``area_km2``.
        Only stations with at least one non-NaN value are included.
    """
    grdc_dir = Path(grdc_dir)
    t0 = np.datetime64(time_start)
    t1 = np.datetime64(time_end)

    result: Dict[int, Tuple[np.ndarray, np.ndarray, dict]] = {}
    n_zips = len(zip_file_map)

    for zi, (zf_name, entries) in enumerate(sorted(zip_file_map.items())):
        zf_path = grdc_dir / f"{zf_name}.zip"
        if not zf_path.exists():
            continue
        try:
            with zipfile.ZipFile(zf_path) as zf:
                for sid, fname in entries:
                    try:
                        with zf.open(fname) as f:
                            raw = f.read().decode("utf-8", errors="replace")
                        gs = parse_grdc_text(str(sid), raw)
                        if len(gs.dates) == 0:
                            continue
                        dates = np.array(gs.dates, dtype="datetime64[D]")
                        vals = gs.values.astype(np.float32)
                        mask = (dates >= t0) & (dates <= t1)
                        dates, vals = dates[mask], vals[mask]
                        if len(vals) > 0 and np.any(np.isfinite(vals)):
                            result[sid] = (dates, vals, gs.meta or {})
                    except Exception:
                        pass
        except Exception:
            pass
        if (zi + 1) % 20 == 0 or zi + 1 == n_zips:
            print(f"  {zi+1}/{n_zips} zips done, {len(result)} series")

    print(f"Extracted {len(result)} time series")
    return result


def write_gauge_nc(
    output_path: Union[str, Path],
    station_ids: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    reported_area_km2: np.ndarray,
    downstream_station_id: np.ndarray,
    observations: np.ndarray,
    time_start: str,
    resolution_allocs: Optional[
        Dict[str, Dict[str, np.ndarray]]
    ] = None,
    resolution_dims: Optional[Dict[str, Tuple[int, int]]] = None,
    title: str = "GRDC daily discharge observations",
    source: str = "GRDC",
    error_threshold: float = 0.10,
) -> Path:
    """Write a standardised gauge observation NetCDF.

    Parameters
    ----------
    output_path : path
        Destination ``.nc`` file.
    station_ids : (N,) int64
    lat, lon : (N,) float32
    reported_area_km2 : (N,) float32
    downstream_station_id : (N,) int64  — ``-1`` for none.
    observations : (T, N) float32  — NaN for missing.
    time_start : str
        ISO date of first time step, e.g. ``"1979-01-01"``.
    resolution_allocs : dict, optional
        ``{res_name: {"catchment_id": (N,) i64,
                      "allocated_area_km2": (N,) f32,
                      "alloc_error": (N,) f32}}``
    resolution_dims : dict, optional
        ``{res_name: (nx, ny)}``.  Required if *resolution_allocs* given.
    title, source : str
        Global attributes.
    error_threshold : float
        Recorded as a global attribute.

    Returns
    -------
    Path
        The written file path.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_time, n_station = observations.shape
    ds = nc.Dataset(str(out), "w", format="NETCDF4")

    ds.title = title
    ds.source = source
    ds.time_start = time_start
    ds.allocation_error_threshold = error_threshold

    ds.createDimension("time", None)
    ds.createDimension("station", n_station)

    # Time
    t_var = ds.createVariable("time", "f8", ("time",))
    t_var.units = f"days since {time_start}"
    t_var.calendar = "standard"
    t_var.long_name = "Time"
    t_var[:] = np.arange(n_time, dtype=np.float64)

    # Station ID
    v = ds.createVariable("station_id", "i8", ("station",), zlib=True)
    v.long_name = "GRDC station ID"
    v[:] = station_ids

    # Lat / Lon
    v = ds.createVariable("lat", "f4", ("station",), zlib=True)
    v.units = "degrees_north"
    v.long_name = "Station latitude"
    v[:] = lat

    v = ds.createVariable("lon", "f4", ("station",), zlib=True)
    v.units = "degrees_east"
    v.long_name = "Station longitude"
    v[:] = lon

    # Reported area
    v = ds.createVariable("reported_area_km2", "f4", ("station",), zlib=True)
    v.units = "km2"
    v.long_name = "Reported upstream drainage area"
    v[:] = reported_area_km2

    # Downstream station
    v = ds.createVariable("downstream_station_id", "i8", ("station",), zlib=True)
    v.long_name = "Nearest downstream station (-1 = none)"
    v[:] = downstream_station_id

    # Observations (chunked per station time series)
    v = ds.createVariable(
        "observations", "f4", ("time", "station"),
        zlib=True, complevel=4, shuffle=True,
        fill_value=np.float32(np.nan),
        chunksizes=(n_time, 1),
    )
    v.units = "m3/s"
    v.long_name = "Daily mean discharge"
    for j in range(n_station):
        v[:, j] = observations[:, j]

    # Per-resolution allocation variables
    if resolution_allocs:
        for res, arrs in resolution_allocs.items():
            nx, ny = resolution_dims[res]
            suffix = f"_{res}"

            v = ds.createVariable(
                f"catchment_id{suffix}", "i8", ("station",), zlib=True)
            v.long_name = f"Catchment index on {res} grid (ix*ny+iy, 0-based)"
            v.setncattr("nx", int(nx))
            v.setncattr("ny", int(ny))
            v[:] = arrs["catchment_id"]

            v = ds.createVariable(
                f"allocated_area{suffix}_km2", "f4", ("station",), zlib=True)
            v.units = "km2"
            v.long_name = f"CaMa-allocated drainage area on {res} grid"
            v[:] = arrs["allocated_area_km2"]

            v = ds.createVariable(
                f"alloc_error{suffix}", "f4", ("station",), zlib=True)
            v.long_name = f"Relative area allocation error on {res} grid"
            v[:] = arrs["alloc_error"]

    ds.close()
    size_mb = out.stat().st_size / 1e6
    print(f"Created: {out} ({size_mb:.1f} MB)")
    print(f"  {n_station} stations × {n_time} days")
    return out
