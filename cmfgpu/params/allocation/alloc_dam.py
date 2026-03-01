# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Dam allocation kernel and mixin for :class:`HiResMap`.

Python re-implementation of ``fortran/allocate_dam.F90``.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from numba import njit

from cmfgpu.params.allocation.hires_kernels import (search_best_pixel,
                                                    trace_gauge_downstream)

if TYPE_CHECKING:
    from cmfgpu.params.allocation.hires_map import HiResMap


def _find_col(header_lower: list[str], aliases: list[str]) -> int | None:
    """Return the column index matching any alias, or *None*."""
    for alias in aliases:
        if alias in header_lower:
            return header_lower.index(alias)
    return None


# ---------------------------------------------------------------------------
# Numba kernel
# ---------------------------------------------------------------------------

@njit(cache=True)
def allocate_all_dams(
    dam_ids: np.ndarray,        # (N,) int64
    dam_lats: np.ndarray,       # (N,) float64
    dam_lons: np.ndarray,       # (N,) float64
    dam_areas: np.ndarray,      # (N,) float64  — upstream area in km²
    upa1m: np.ndarray,
    ctx1m: np.ndarray,
    cty1m: np.ndarray,
    dwx1m: np.ndarray,
    dwy1m: np.ndarray,
    uparea: np.ndarray,         # km²
    ctmare: np.ndarray,         # km²  (catchment area)
    upstXX: np.ndarray,
    upstYY: np.ndarray,
    outx: np.ndarray,
    outy: np.ndarray,
    west: float, north: float, gsize: float,
    west2: float, north2: float, csize: float,
    nx: int, ny: int, nXX: int, nYY: int,
    nn: int, n_ups: int, is_global: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dam allocation — single-mode upstream, with sub-grid dam detection.

    Returns 5 arrays of length N:
    - staX, staY    : allocated grid (0-based), -9999 if unallocated
    - area_cmf      : CaMa upstream area at allocated grid (km²), -888 = sub-grid, -999 = error
    - err_rel       : relative error, -8 for sub-grid
    - snum          : 0 = outlet, 1 = upstream grid used
    """
    N = dam_ids.shape[0]
    MISS = np.int32(-9999)
    staX = np.full(N, MISS, dtype=np.int32)
    staY = np.full(N, MISS, dtype=np.int32)
    area_cmf = np.full(N, -999.0, dtype=np.float64)
    err_rel = np.full(N, -999.0, dtype=np.float64)
    snum_arr = np.zeros(N, dtype=np.int32)

    for g in range(N):
        lat0 = dam_lats[g]
        lon0 = dam_lons[g]
        area0 = dam_areas[g]   # km²

        east = west + nXX * gsize
        south = north - nYY * gsize
        if lon0 < west or lon0 > east or lat0 < south or lat0 > north:
            continue
        if area0 <= 0.0:
            continue

        area0_km2 = area0  # already in km²

        ix_c = int((lon0 - west2) / csize)
        iy_c = int((north2 - lat0) / csize)

        best_kx, best_ky, best_err1, best_area, best_rate = search_best_pixel(
            ix_c, iy_c, upa1m, area0, nn, nx, ny, is_global
        )
        if best_rate >= 1.0e20:
            continue

        ix0 = best_kx
        iy0 = best_ky
        iXX0 = int(ctx1m[ix0, iy0])
        iYY0 = int(cty1m[ix0, iy0])
        if iXX0 < 0 or iYY0 < 0 or iXX0 >= nXX or iYY0 >= nYY:
            continue

        # Sub-grid dam: area0 < 30% of catchment area
        if area0_km2 < ctmare[iXX0, iYY0] * 0.3:
            staX[g] = iXX0
            staY[g] = iYY0
            area_cmf[g] = -888.0
            err_rel[g] = -8.0
            continue

        staX[g] = iXX0
        staY[g] = iYY0
        area_cmf[g] = uparea[iXX0, iYY0]

        err1 = (uparea[iXX0, iYY0] - area0_km2) / area0_km2

        snum = 0
        if abs(err1) > 0.2:   # dam uses 20% threshold (vs 5% for gauges)
            for i_ups in range(n_ups):
                jXX = upstXX[iXX0, iYY0, i_ups]
                jYY = upstYY[iXX0, iYY0, i_ups]
                if jXX < 0:
                    break
                oix = outx[jXX, jYY]
                oiy = outy[jXX, jYY]
                if oix < 0 or oiy < 0:
                    continue
                found = trace_gauge_downstream(
                    ix0, iy0, iXX0, iYY0, oix, oiy, ctx1m, cty1m, dwx1m, dwy1m, nx
                )
                if found:
                    if uparea[jXX, jYY] < area0_km2 * 0.1:
                        continue
                    new_area = uparea[jXX, jYY]
                    err2 = (new_area - area0_km2) / area0_km2
                    if abs(err2) < abs(err1):
                        err1 = err2
                        snum = 1
                        staX[g] = jXX
                        staY[g] = jYY
                        area_cmf[g] = new_area
                        if abs(err1) < 0.1:
                            break

        # If outlet error still > 100%, mark as sub-grid
        # (Fortran: reclassify regardless of snum)
        if snum == 0:
            area_cmf[g] = uparea[iXX0, iYY0]

        diff_val = area_cmf[g] - area0_km2
        err1 = diff_val / area0_km2 if area0_km2 > 0.0 else 0.0

        if err1 > 1.0:
            staX[g] = iXX0
            staY[g] = iYY0
            area_cmf[g] = -888.0
            err_rel[g] = -8.0
            snum_arr[g] = 0
            continue

        err_rel[g] = err1
        snum_arr[g] = snum

    return staX, staY, area_cmf, err_rel, snum_arr


# ---------------------------------------------------------------------------
# Mixin class
# ---------------------------------------------------------------------------


class DamAllocMixin:
    """Methods for dam allocation on :class:`HiResMap`."""

    def load_dam_list(self: HiResMap, dam_list_path: str | Path) -> None:
        """Parse dam list file.

        Expected format (comma or space separated, 1 header line)::

            ID  Lat  Lon  Uparea(km²)  DamName  RivName  Cap_MCM  Year

        The first four numeric columns (ID, Lat, Lon, Uparea) are always
        read by position.  ``cap_mcm`` and ``year`` are detected by header
        name if present.
        """
        dam_list_path = Path(dam_list_path)
        ids: List[int] = []
        lats: List[float] = []
        lons: List[float] = []
        areas: List[float] = []
        names: List[str] = []
        cap_mcm_list: List[float] = []
        year_list: List[int] = []

        with open(dam_list_path, encoding="utf-8-sig") as f:
            header = f.readline()
            sep = "," if "," in header else None
            header_lower = [h.strip().lower() for h in header.split(sep)]

            # Detect optional columns by header name
            cap_col = _find_col(header_lower, ["cap_mcm"])
            year_col = _find_col(header_lower, ["year"])

            for line in f:
                parts = line.strip().split(sep)
                if len(parts) < 4:
                    continue
                try:
                    ids.append(int(parts[0]))
                    lats.append(float(parts[1]))
                    lons.append(float(parts[2]))
                    areas.append(float(parts[3]))
                except (ValueError, IndexError):
                    continue
                names.append(parts[4].strip() if len(parts) > 4 else "")

                if cap_col is not None and len(parts) > cap_col:
                    try:
                        cap_mcm_list.append(float(parts[cap_col]))
                    except ValueError:
                        cap_mcm_list.append(-999.0)
                else:
                    cap_mcm_list.append(-999.0)

                if year_col is not None and len(parts) > year_col:
                    try:
                        year_list.append(int(parts[year_col]))
                    except ValueError:
                        year_list.append(-99)
                else:
                    year_list.append(-99)

        self.dam_ids = np.array(ids, dtype=np.int64)
        self.dam_lats = np.array(lats, dtype=np.float64)
        self.dam_lons = np.array(lons, dtype=np.float64)
        self.dam_areas = np.array(areas, dtype=np.float64)
        self.dam_names = names
        self.dam_cap_mcm = np.array(cap_mcm_list, dtype=np.float64)
        self.dam_years = np.array(year_list, dtype=np.int64)

        extras = []
        if cap_col is not None:
            extras.append(f"cap_mcm (col {cap_col})")
        if year_col is not None:
            extras.append(f"year (col {year_col})")
        msg = f"Loaded {len(ids)} dams from {dam_list_path}"
        if extras:
            msg += f" [{', '.join(extras)}]"
        print(msg)

    def allocate_dams(self: HiResMap) -> None:
        """Run dam allocation (numba-accelerated core).

        Requires: :meth:`load_lowres`, :meth:`load_lowres_extra`,
        :meth:`load_hires`, :meth:`build_upstream_table`,
        :meth:`calc_outlet_pixels`, :meth:`load_dam_list`.
        """
        print("Allocating dams ...")
        (
            self.dam_staX,
            self.dam_staY,
            self.dam_area_cmf,
            self.dam_err_rel,
            self.dam_snum,
        ) = allocate_all_dams(
            self.dam_ids,
            self.dam_lats,
            self.dam_lons,
            self.dam_areas,
            self.upa1m,
            self.ctx1m,
            self.cty1m,
            self.dwx1m,
            self.dwy1m,
            self.uparea,
            self.ctmare,
            self.upstXX,
            self.upstYY,
            self.outx,
            self.outy,
            self.west,
            self.north,
            self.gsize,
            self.west2,
            self.north2,
            self.csize,
            self.hires_nx,
            self.hires_ny,
            self.nXX,
            self.nYY,
            self.search_radius,
            self.n_upstream,
            self.is_global,
        )

        n_river = int(np.sum((self.dam_staX != self.MISSING) & (self.dam_area_cmf > 0)))
        n_small = int(np.sum(self.dam_area_cmf == -888.0))
        n_fail = int(np.sum(self.dam_staX == self.MISSING))
        print(f"Dam allocation done: {n_river} river, {n_small} sub-grid, {n_fail} failed")

    def dam_results_as_structured_array(self: HiResMap) -> np.ndarray:
        """Pack dam allocation results into a structured NumPy array.

        Fields
        ------
        id, lat, lon, area_input : from dam list
        ix, iy                  : allocated grid (0-based), -9999 if unallocated
        catchment_id            : flat ID, -1 if unallocated
        area_cama               : CaMa upstream area (km²); -888 = sub-grid, -999 = error
        error                   : relative error; -8 for sub-grid
        snum                    : 0 = outlet, 1 = upstream grid
        cap_mcm                 : total capacity (MCM), -999 if unavailable
        year                    : construction year, -99 if unavailable
        """
        N = len(self.dam_ids)
        dtype = np.dtype([
            ("id", np.int64),
            ("lat", np.float64),
            ("lon", np.float64),
            ("area_input", np.float64),
            ("ix", np.int32),
            ("iy", np.int32),
            ("catchment_id", np.int64),
            ("area_cama", np.float64),
            ("error", np.float64),
            ("snum", np.int32),
            ("cap_mcm", np.float64),
            ("year", np.int64),
        ])
        arr = np.empty(N, dtype=dtype)
        arr["id"] = self.dam_ids
        arr["lat"] = self.dam_lats
        arr["lon"] = self.dam_lons
        arr["area_input"] = self.dam_areas
        arr["ix"] = self.dam_staX
        arr["iy"] = self.dam_staY
        arr["area_cama"] = self.dam_area_cmf
        arr["error"] = self.dam_err_rel
        arr["snum"] = self.dam_snum
        arr["cap_mcm"] = getattr(self, "dam_cap_mcm",
                                  np.full(N, -999.0, dtype=np.float64))
        arr["year"] = getattr(self, "dam_years",
                               np.full(N, -99, dtype=np.int64))

        cid = np.full(N, -1, dtype=np.int64)
        ok = (self.dam_staX >= 0) & (self.dam_staY >= 0)
        if np.any(ok):
            cid[ok] = np.ravel_multi_index(
                (self.dam_staX[ok], self.dam_staY[ok]), (self.nXX, self.nYY)
            )
        arr["catchment_id"] = cid
        return arr

    def write_dam_alloc_file(self: HiResMap, out_path: str | Path | None = None) -> Path:
        """Write dam allocation results to text file."""
        if out_path is None:
            out_path = (self.out_dir or self.map_dir) / "dam_alloc.txt"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        header = (
            "        ID       lat       lon  area_Input   area_CaMa"
            "       error  Type     ix    iy  catchment_id"
        )
        lines = [header]
        for i in range(len(self.dam_ids)):
            gid = self.dam_ids[i]
            lat = self.dam_lats[i]
            lon = self.dam_lons[i]
            area_in = self.dam_areas[i]  # already km²
            area_c = self.dam_area_cmf[i]
            err = self.dam_err_rel[i]
            ix_out = self.dam_staX[i]
            iy_out = self.dam_staY[i]

            if ix_out == self.MISSING:
                tag = "ERR"
            elif area_c == -888.0:
                tag = "SUB"
            else:
                tag = "RIV"

            cid = int(np.ravel_multi_index(
                (ix_out, iy_out), (self.nXX, self.nYY)
            )) if ix_out >= 0 else -1

            lines.append(
                f"{gid:10d}{lat:10.3f}{lon:10.3f}{area_in:12.2f}"
                f"{area_c:12.2f}{err:12.4f}  {tag:>3s}"
                f"{ix_out + 1:8d}{iy_out + 1:6d}{cid:14d}"
            )

        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Wrote dam allocation to {out_path}")
        return out_path
