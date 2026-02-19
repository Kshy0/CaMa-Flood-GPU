# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
High-resolution map data handler for CaMa-Flood-GPU gauge allocation.

Python re-implementation of the three Fortran allocation programs:

* ``allocate_flow_gauge.F90``  → :class:`FlowGaugeMixin`
* ``allocate_dam.F90``         → :class:`DamAllocMixin`
* ``allocate_level_gauge.F90`` → :class:`LevelGaugeAllocMixin`

Uses Pydantic v2 for configuration, NumPy for array storage, and Numba for
performance-critical inner loops.  All (x, y) pairs are converted to flat IDs
(``np.ravel_multi_index``) as early as possible.

Typical usage
-------------
>>> hires = HiResMap(map_dir="/path/to/glb_15min",
...                  gauge_list="/path/to/gauge_list.txt")
>>> hires.build()                        # flow-gauge allocation
>>> hires.build_dams("/path/to/dams.txt") # dam allocation
>>> hires.build_level_gauges()            # level-gauge allocation
>>> mapping = hires.catchment_to_gauge_mapping(kind="flow")
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Dict, List, Optional

import numpy as np
from pydantic import (BaseModel, ConfigDict, DirectoryPath, Field, FilePath,
                      model_validator)

from cmfgpu.params.allocation.alloc_dam import DamAllocMixin
from cmfgpu.params.allocation.alloc_flow_gauge import FlowGaugeMixin
from cmfgpu.params.allocation.alloc_level_gauge import LevelGaugeAllocMixin
from cmfgpu.params.allocation.hires_kernels import (build_upstream_table,
                                                    calc_outlet_pixels)
from cmfgpu.utils import binread, read_map

# ---------------------------------------------------------------------------
# Pydantic data class
# ---------------------------------------------------------------------------


class HiResMap(FlowGaugeMixin, DamAllocMixin, LevelGaugeAllocMixin, BaseModel):
    """High-resolution map handler for gauge / dam / level-gauge allocation.

    This class loads the low-resolution CaMa-Flood map, the corresponding
    high-resolution (typically 1 min or 15 sec) MERIT Hydro data, and performs
    allocation using upstream-area matching and high-resolution river tracing.

    All intermediate arrays are cached as instance attributes so that
    downstream code can inspect / reuse them.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra="allow",
    )

    # ── Input paths ──────────────────────────────────────────────────────
    map_dir: DirectoryPath = Field(
        description="Low-resolution CaMa-Flood map directory (contains params.txt, nextxy.bin, uparea.bin, etc.)"
    )

    gauge_list: Optional[FilePath] = Field(
        default=None,
        description=(
            "Path to gauge list file.  Each line (after header) must start with: "
            "ID  Lat  Lon  Uparea(km²)  <...ignored columns...>"
        ),
    )

    hires_tag: str = Field(
        default="",
        description=(
            "Sub-directory name for the hi-res map (e.g. '1min', '15sec').  "
            "Leave empty to auto-detect (tries '1min' first, then '15sec')."
        ),
    )

    mode: str = Field(
        default="multi",
        description="Allocation mode: 'multi' (up to 2 grids) or 'single' (1 grid only).",
    )

    search_radius: int = Field(
        default=3,
        description="Search radius (±nn hi-res pixels) around gauge coordinate.",
    )

    n_upstream: int = Field(
        default=8,
        description="Maximum number of upstream grids to consider per cell.",
    )

    out_file: str = Field(
        default="gauge_alloc.txt",
        description="Output file name for the allocation results.",
    )

    out_dir: Optional[Path] = Field(
        default=None,
        description="Output directory.  Defaults to map_dir if not specified.",
    )

    # ── Class-level constants ────────────────────────────────────────────
    MISSING: ClassVar[int] = -9999
    lowres_idx_precision: ClassVar[str] = "<i4"
    hires_idx_precision: ClassVar[str] = "<i2"
    map_precision: ClassVar[str] = "<f4"

    # =====================================================================
    # Loading methods
    # =====================================================================

    def _detect_hires_tag(self) -> str:
        """Auto-detect the high-resolution sub-directory."""
        for tag in ("1min", "15sec"):
            loc = self.map_dir / tag / "location.txt"
            if loc.exists():
                with open(loc) as f:
                    first = f.readline().split()
                    narea = int(first[0])
                if narea == 1:
                    return tag
        raise FileNotFoundError(
            f"No valid high-resolution sub-directory (1min/15sec) found under {self.map_dir}"
        )

    def load_lowres(self) -> None:
        """Read CaMa-Flood low-resolution map parameters and binary files."""
        params_path = self.map_dir / "params.txt"
        with open(params_path) as f:
            lines = f.readlines()
        self.nXX = int(lines[0].split()[0])
        self.nYY = int(lines[1].split()[0])
        self.gsize = float(lines[3].split()[0])
        self.west = float(lines[4].split()[0])
        self.east = float(lines[5].split()[0])
        self.south = float(lines[6].split()[0])
        self.north = float(lines[7].split()[0])
        self.is_global = abs(self.east - self.west - 360.0) < 1e-3

        print(f"Low-res grid: {self.nXX}×{self.nYY}, gsize={self.gsize}°, global={self.is_global}")

        # Upstream area (convert m² → km²)
        uparea_raw = read_map(
            self.map_dir / "uparea.bin", (self.nXX, self.nYY), precision=self.map_precision
        )
        self.uparea = np.where(uparea_raw > 0, uparea_raw * 1e-6, uparea_raw).astype(np.float32)

        # Downstream pointers (Fortran 1-based → Python 0-based)
        nextxy = binread(
            self.map_dir / "nextxy.bin", (self.nXX, self.nYY, 2), dtype_str=self.lowres_idx_precision
        )
        self.nextXX = nextxy[:, :, 0].astype(np.int32)
        self.nextYY = nextxy[:, :, 1].astype(np.int32)
        valid = self.nextXX > 0
        self.nextXX[valid] -= 1
        self.nextYY[valid] -= 1
        self.nextXX[~valid] = self.MISSING
        self.nextYY[~valid] = self.MISSING

        # Lon / lat of cell centres
        self.glon = self.west + self.gsize * (np.arange(self.nXX) + 0.5)
        self.glat = self.north - self.gsize * (np.arange(self.nYY) + 0.5)

    def load_lowres_extra(self) -> None:
        """Read optional low-res maps needed for dam / level-gauge allocation.

        Loads ``ctmare.bin`` (catchment area) and ``elevtn.bin`` (elevation).
        Call *after* :meth:`load_lowres`.
        """
        ctmare_raw = read_map(
            self.map_dir / "ctmare.bin", (self.nXX, self.nYY), precision=self.map_precision
        )
        self.ctmare = np.where(ctmare_raw > 0, ctmare_raw * 1e-6, ctmare_raw).astype(np.float32)

        self.elevtn = read_map(
            self.map_dir / "elevtn.bin", (self.nXX, self.nYY), precision=self.map_precision
        ).astype(np.float32)

        print(f"Loaded ctmare + elevtn (low-res, {self.nXX}×{self.nYY})")

    def load_hires(self) -> None:
        """Read high-resolution map data and cache in memory."""
        tag = self.hires_tag or self._detect_hires_tag()
        self.hires_tag = tag
        hires_dir = self.map_dir / tag

        with open(hires_dir / "location.txt") as f:
            lines = f.readlines()
        parts = lines[2].split()
        self.west2 = float(parts[2])
        self.east2 = float(parts[3])
        self.south2 = float(parts[4])
        self.north2 = float(parts[5])
        self.hires_nx = int(parts[6])
        self.hires_ny = int(parts[7])
        self.csize = (self.east2 - self.west2) / self.hires_nx

        print(f"Hi-res grid ({tag}): {self.hires_nx}×{self.hires_ny}, csize={self.csize:.10f}°")

        nx, ny = self.hires_nx, self.hires_ny

        self.upa1m = read_map(
            hires_dir / f"{tag}.uparea.bin", (nx, ny), precision=self.map_precision
        ).astype(np.float32)

        catmxy = read_map(
            hires_dir / f"{tag}.catmxy.bin", (nx, ny, 2), precision=self.hires_idx_precision
        )
        self.ctx1m = (catmxy[:, :, 0] - 1).astype(np.int16)
        self.cty1m = (catmxy[:, :, 1] - 1).astype(np.int16)

        downxy = read_map(
            hires_dir / f"{tag}.downxy.bin", (nx, ny, 2), precision=self.hires_idx_precision
        )
        self.dwx1m = downxy[:, :, 0].astype(np.int16)
        self.dwy1m = downxy[:, :, 1].astype(np.int16)

        self.hires_lon = self.west2 + self.csize * (np.arange(nx) + 0.5)
        self.hires_lat = self.north2 - self.csize * (np.arange(ny) + 0.5)

        mem_mb = (self.upa1m.nbytes + self.ctx1m.nbytes + self.cty1m.nbytes +
                  self.dwx1m.nbytes + self.dwy1m.nbytes) / 1e6
        print(f"Hi-res maps cached: {mem_mb:.1f} MB")

    def load_hires_elevtn(self) -> None:
        """Read high-resolution elevation map (needed for level-gauge allocation).

        Call *after* :meth:`load_hires`.
        """
        tag = self.hires_tag
        hires_dir = self.map_dir / tag
        nx, ny = self.hires_nx, self.hires_ny

        self.elv1m = read_map(
            hires_dir / f"{tag}.elevtn.bin", (nx, ny), precision=self.map_precision
        ).astype(np.float32)

        print(f"Hi-res elevation cached: {self.elv1m.nbytes / 1e6:.1f} MB")

    def build_upstream_table(self) -> None:
        """Build upstream grid look-up table (numba-accelerated)."""
        print(f"Building upstream table (n_ups={self.n_upstream}) ...")
        self.upstXX, self.upstYY = build_upstream_table(
            self.nextXX, self.nextYY, self.uparea, self.n_upstream
        )
        print("Upstream table built.")

    def calc_outlet_pixels(self) -> None:
        """Find each unit-catchment's outlet pixel on the hi-res grid."""
        print("Calculating outlet pixels ...")
        self.outx, self.outy = calc_outlet_pixels(
            self.ctx1m, self.cty1m, self.dwx1m, self.dwy1m, self.upa1m,
            self.nXX, self.nYY,
        )
        n_valid = int(np.sum(self.outx != self.MISSING))
        print(f"Outlet pixels computed: {n_valid}/{self.nXX * self.nYY} catchments have outlets.")

    def load_gauge_list(self) -> None:
        """Parse gauge list file (ID, Lat, Lon, Uparea in km²)."""
        ids: List[int] = []
        lats: List[float] = []
        lons: List[float] = []
        areas: List[float] = []

        with open(self.gauge_list) as f:
            header = f.readline()  # noqa: F841 — skip header
            sep = "," if "," in header else None
            for line in f:
                parts = line.strip().split(sep)
                if len(parts) < 4:
                    continue
                ids.append(int(parts[0]))
                lats.append(float(parts[1]))
                lons.append(float(parts[2]))
                areas.append(float(parts[3]))

        self.gauge_ids = np.array(ids, dtype=np.int64)
        self.gauge_lats = np.array(lats, dtype=np.float64)
        self.gauge_lons = np.array(lons, dtype=np.float64)
        self.gauge_areas = np.array(areas, dtype=np.float64)

        print(f"Loaded {len(ids)} gauges from {self.gauge_list}")

    # =====================================================================
    # Fundamental mapping interface
    # =====================================================================

    def catchment_to_gauge_mapping(
        self,
        kind: str = "flow",
    ) -> Dict[int, Dict]:
        """Return the fundamental catchment_id → gauge_id mapping with error.

        Parameters
        ----------
        kind : ``"flow"`` | ``"dam"`` | ``"level"``
            Which allocation result to use.

        Returns
        -------
        dict
            ``{catchment_id: {"gauge_id": int, "error": float,
            "area_gauge": float, "area_cama": float, ...}}``
        """
        mapping: Dict[int, Dict] = {}
        shape = (self.nXX, self.nYY)

        if kind == "flow":
            res = self.results_as_structured_array()
            for row in res:
                gid = int(row["id"])
                area_in = float(row["area_input"])  # already km²
                cid1 = int(row["catchment_id1"])
                cid2 = int(row["catchment_id2"])
                sn = int(row["snum"])
                if cid1 < 0:
                    continue
                if sn == 0:
                    area_cmf = float(self.uparea[row["ix1"], row["iy1"]])
                else:
                    area_cmf = float(row["area1"])
                    if sn >= 2:
                        area_cmf += float(row["area2"])
                err = (area_cmf - area_in) / area_in if area_in > 0 else 0.0
                mapping[cid1] = {
                    "gauge_id": gid, "error": err,
                    "area_gauge": area_in, "area_cama": float(row["area1"]),
                }
                if cid2 >= 0:
                    mapping[cid2] = {
                        "gauge_id": gid, "error": err,
                        "area_gauge": area_in, "area_cama": float(row["area2"]),
                    }

        elif kind == "dam":
            for i in range(len(self.dam_ids)):
                ix, iy = self.dam_staX[i], self.dam_staY[i]
                if ix == self.MISSING:
                    continue
                cid = int(np.ravel_multi_index((ix, iy), shape))
                mapping[cid] = {
                    "gauge_id": int(self.dam_ids[i]),
                    "error": float(self.dam_err_rel[i]),
                    "area_gauge": float(self.dam_areas[i]),  # already km²
                    "area_cama": float(self.dam_area_cmf[i]),
                }

        elif kind == "level":
            for i in range(len(self.gauge_ids)):
                ix, iy = self.lvl_staX[i], self.lvl_staY[i]
                if ix == self.MISSING:
                    continue
                cid = int(np.ravel_multi_index((ix, iy), shape))
                area_in_km2 = float(self.gauge_areas[i])  # already km²
                area_cmf = float(self.uparea[ix, iy])
                err = (area_cmf - area_in_km2) / area_in_km2 if area_in_km2 > 0 else 0.0
                mapping[cid] = {
                    "gauge_id": int(self.gauge_ids[i]),
                    "error": err,
                    "area_gauge": area_in_km2,
                    "area_cama": area_cmf,
                    "gauge_type": int(self.lvl_gtype[i]),
                    "dst_outlet_km": float(self.lvl_dst_outlet[i]),
                    "elv_outlet": float(self.lvl_elv_outlet[i]),
                    "elv_gauge": float(self.lvl_elv_gauge[i]),
                }
        else:
            raise ValueError(f"Unknown kind={kind!r}; expected 'flow', 'dam', or 'level'")

        return mapping

    # =====================================================================
    # Top-level pipelines
    # =====================================================================

    def build(self) -> None:
        """Full pipeline: load → build tables → allocate flow gauges → write."""
        print("=" * 60)
        print("HiResMap gauge allocation pipeline")
        print("=" * 60)
        self.load_lowres()
        self.load_hires()
        self.build_upstream_table()
        self.calc_outlet_pixels()
        self.load_gauge_list()
        self.allocate()
        self.write_alloc_file()
        print("=" * 60)
        print("Pipeline complete.")

    def build_dams(self, dam_list_path: str | Path) -> None:
        """Full pipeline for dam allocation.

        Parameters
        ----------
        dam_list_path : path-like
            Path to the dam list file (ID Lat Lon Uparea ...).
        """
        print("=" * 60)
        print("HiResMap DAM allocation pipeline")
        print("=" * 60)
        self.load_lowres()
        self.load_lowres_extra()
        self.load_hires()
        self.build_upstream_table()
        self.calc_outlet_pixels()
        self.load_dam_list(dam_list_path)
        self.allocate_dams()
        self.write_dam_alloc_file()
        print("=" * 60)
        print("Dam pipeline complete.")

    def build_level_gauges(self) -> None:
        """Full pipeline for level-gauge allocation.

        Requires ``gauge_list`` to be set at construction.
        """
        print("=" * 60)
        print("HiResMap LEVEL-GAUGE allocation pipeline")
        print("=" * 60)
        self.load_lowres()
        self.load_lowres_extra()
        self.load_hires()
        self.load_hires_elevtn()
        self.build_upstream_table()
        self.calc_outlet_pixels()
        self.load_gauge_list()
        self.allocate_level_gauges()
        self.write_level_gauge_alloc_file()
        print("=" * 60)
        print("Level-gauge pipeline complete.")

    @model_validator(mode="after")
    def _set_defaults(self) -> "HiResMap":
        if self.out_dir is None:
            object.__setattr__(self, "out_dir", Path(self.map_dir))
        else:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        return self

    # =====================================================================
    # Utility: convert CaMa-Flood official allocation output → gauge list
    # =====================================================================

    @staticmethod
    def grdc_alloc_to_gauge_list(
        alloc_path: str | Path,
        out_path: str | Path | None = None,
    ) -> Path:
        """Convert a CaMa-Flood ``GRDC_alloc.txt`` to a simple gauge list.

        The official ``GRDC_alloc.txt`` produced by CaMa-Flood's Fortran
        ``allocate_flow_gauge`` has 14 fixed-width columns per data line::

            ID  lat  lon  err  area_GRDC  area_CaMa  diff  ups_num
            ix1  iy1  ix2  iy2  area1  area2

        This method extracts ``(ID, lat, lon, area_GRDC)`` and writes a
        simple whitespace-separated file that :meth:`load_gauge_list` can
        read directly.

        Parameters
        ----------
        alloc_path : path-like
            Path to the CaMa-Flood ``GRDC_alloc.txt``.
        out_path : path-like or None
            Output file path.  Defaults to ``<alloc_dir>/GRDC_gauge_list.txt``.

        Returns
        -------
        Path
            The path to the generated gauge list file.
        """
        alloc_path = Path(alloc_path)
        if out_path is None:
            out_path = alloc_path.parent / "GRDC_gauge_list.txt"
        else:
            out_path = Path(out_path)

        records = []
        with open(alloc_path) as f:
            f.readline()  # skip header
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                gid = int(parts[0])
                lat = float(parts[1])
                lon = float(parts[2])
                area = float(parts[4])  # area_GRDC (km²)
                records.append((gid, lat, lon, area))

        with open(out_path, "w") as f:
            f.write("ID  lat  lon  uparea_km2\n")
            for gid, lat, lon, area in records:
                f.write(f"{gid}  {lat:.6f}  {lon:.6f}  {area:.1f}\n")

        print(f"Converted {len(records)} gauges: {alloc_path} → {out_path}")
        return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    map_resolution = "glb_15min"
    map_dir = f"/home/eat/cmf_v420_pkg/map/{map_resolution}"

    # Convert GRDC_alloc.txt → simple gauge list
    gauge_list = HiResMap.grdc_alloc_to_gauge_list(
        f"{map_dir}/GRDC_alloc.txt",
    )

    hires = HiResMap(
        map_dir=map_dir,
        gauge_list=gauge_list,
        hires_tag="1min",
    )
    hires.build()
