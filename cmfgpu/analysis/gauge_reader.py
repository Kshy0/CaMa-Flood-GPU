# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import re
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

    # -----------------------------
    # Default GRDC Loader
    # -----------------------------
    @staticmethod
    def _load_grdc(gauge_id: str, file_path: Union[str, Path]) -> GaugeSeries:
        file_path = Path(file_path)
        dates: List[datetime] = []
        vals: List[float] = []
        units = "m3/s"

        # Try extract basic meta from header
        meta: dict = {"source": "GRDC"}
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            in_data = False
            for raw in f:
                line = raw.strip("\n")
                if not in_data:
                    if line.startswith("#"):
                        # capture a few optional fields
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
                        if "Unit of measure" in line and ("m3" in line or "m?s" in line):
                            units = "m3/s"
                        continue
                    # Switch to data when the header separator or table header has passed
                    if line.upper().startswith("YYYY-") or line.strip().upper() == "# DATA":
                        in_data = True
                        continue
                    else:
                        continue

                # Data section
                if not line or line.startswith("#"):
                    continue
                # Support both ';' separated and whitespace
                parts = [p.strip() for p in line.split(";")]
                if len(parts) >= 3:
                    date_str, time_str, val_str = parts[:3]
                else:
                    toks = line.split()
                    if len(toks) < 2:
                        continue
                    date_str, val_str = toks[0], toks[-1]
                    time_str = "--:--"
                try:
                    # Some files use --:--, ignore time and parse date only
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                except Exception:
                    # Try alternative formats
                    try:
                        dt = datetime.strptime(date_str, "%d.%m.%Y")
                    except Exception:
                        continue

                val_str = val_str.replace(",", ".")  # safety
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

    def __init__(
        self,
        base_dir: Union[str, Path],
        file_pattern: str = "{gauge_id}_Q_Day.Cmd.txt",
        file_resolver: Optional[Callable[[str], Union[str, Path]]] = None,
        loader: Optional[Callable[[str, Union[str, Path]], GaugeSeries]] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.file_pattern = file_pattern
        self._file_resolver = file_resolver
        self._loader = loader or self._load_grdc

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
        if self._file_resolver is not None:
            p = Path(self._file_resolver(gauge_id))
        else:
            name = self.file_pattern.format(gauge_id=gauge_id)
            p = self.base_dir / name
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

        for toks in rows:
            gid = str(get_val(toks, "ID", str))
            if gid in ("None", "nan", "NaN"):
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

    # -----------------------------
    # Metrics
    # -----------------------------
    @staticmethod
    def nse(y_obs: np.ndarray, y_sim: np.ndarray) -> float:
        """Compute Nash–Sutcliffe Efficiency ignoring NaNs; assumes aligned time axis."""
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
