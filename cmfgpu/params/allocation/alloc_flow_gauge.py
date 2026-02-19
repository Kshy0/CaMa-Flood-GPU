# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Flow-gauge allocation kernel and mixin for :class:`HiResMap`.

Python re-implementation of ``fortran/allocate_flow_gauge.F90``.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import numpy as np
from numba import njit

from cmfgpu.params.allocation.hires_kernels import (search_best_pixel,
                                                    trace_gauge_downstream)

if TYPE_CHECKING:
    from cmfgpu.params.allocation.hires_map import HiResMap


# ---------------------------------------------------------------------------
# Numba kernel
# ---------------------------------------------------------------------------

@njit(cache=True)
def allocate_all_gauges(
    gauge_ids: np.ndarray,       # (N,) int64
    gauge_lats: np.ndarray,      # (N,) float64
    gauge_lons: np.ndarray,      # (N,) float64
    gauge_areas: np.ndarray,     # (N,) float64, upstream area in m² (same unit as upa1m)
    upa1m: np.ndarray,           # (nx, ny) float32
    ctx1m: np.ndarray,           # (nx, ny) int16, 0-based
    cty1m: np.ndarray,
    dwx1m: np.ndarray,
    dwy1m: np.ndarray,
    uparea: np.ndarray,          # (nXX, nYY) float32, in km² (Fortran convention)
    upstXX: np.ndarray,          # (nXX, nYY, n_ups)
    upstYY: np.ndarray,
    outx: np.ndarray,            # (nXX, nYY)
    outy: np.ndarray,
    west: float,
    north: float,
    gsize: float,
    west2: float,
    north2: float,
    csize: float,
    nx: int,
    ny: int,
    nXX: int,
    nYY: int,
    nn: int,                     # search radius in hi-res pixels
    n_ups: int,
    is_global: bool,
    mode_single: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Allocate all gauges to CaMa-Flood grid cells.

    Returns 7 arrays of length N (one per gauge):
    - staX1, staY1  : primary grid (0-based), -9999 if unallocated
    - staX2, staY2  : secondary grid (0-based), -9999 if unused
    - staA1, staA2  : corresponding upstream areas (km²)
    - snum          : number of upstream grids used (0 = outlet itself, 1-2 = upstream)
    """
    N = gauge_ids.shape[0]
    staX1 = np.full(N, -9999, dtype=np.int32)
    staY1 = np.full(N, -9999, dtype=np.int32)
    staX2 = np.full(N, -9999, dtype=np.int32)
    staY2 = np.full(N, -9999, dtype=np.int32)
    staA1 = np.full(N, -999.0, dtype=np.float64)
    staA2 = np.full(N, -999.0, dtype=np.float64)
    snum_arr = np.zeros(N, dtype=np.int32)

    for g in range(N):
        lat0 = gauge_lats[g]
        lon0 = gauge_lons[g]
        area0 = gauge_areas[g]  # in m² (same as upa1m)

        # Domain check (in degrees, low-res)
        east = west + nXX * gsize
        south = north - nYY * gsize
        if lon0 < west or lon0 > east or lat0 < south or lat0 > north:
            continue

        if area0 <= 0.0:
            continue

        # Convert gauge lat/lon to hi-res pixel (0-based)
        ix_c = int((lon0 - west2) / csize)
        iy_c = int((north2 - lat0) / csize)

        # Search neighbourhood
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

        staX1[g] = iXX0
        staY1[g] = iYY0
        staA1[g] = uparea[iXX0, iYY0]

        area0_km2 = area0  # already in km² (same unit as upa1m and uparea)

        # Check if outlet itself is good enough (<=5% error)
        err1 = (uparea[iXX0, iYY0] - area0_km2) / area0_km2

        snum = 0
        upa_sum = 0.0

        if abs(err1) > 0.05:
            # Try upstream grids
            for i_ups in range(n_ups):
                jXX = upstXX[iXX0, iYY0, i_ups]
                jYY = upstYY[iXX0, iYY0, i_ups]
                if jXX < 0:
                    break  # no more upstream grids

                oix = outx[jXX, jYY]
                oiy = outy[jXX, jYY]
                if oix < 0 or oiy < 0:
                    continue

                found = trace_gauge_downstream(
                    ix0, iy0, iXX0, iYY0, oix, oiy, ctx1m, cty1m, dwx1m, dwy1m, nx
                )

                if found:
                    if uparea[jXX, jYY] < area0_km2 * 0.1:
                        continue  # small tributary, skip

                    new_area = upa_sum + uparea[jXX, jYY]
                    diff = new_area - area0_km2
                    err2 = diff / area0_km2

                    if abs(err2) < abs(err1):
                        err1 = err2
                        snum += 1
                        if snum == 1:
                            staX1[g] = jXX
                            staY1[g] = jYY
                            staA1[g] = uparea[jXX, jYY]
                        elif snum == 2:
                            staX2[g] = jXX
                            staY2[g] = jYY
                            staA2[g] = uparea[jXX, jYY]
                        upa_sum = new_area

                        if snum >= 2:
                            break
                        if mode_single and snum >= 1:
                            break
                        if abs(err1) < 0.1:
                            break

        # ── Final rejection: small-tributary gauge with no CaMa grid ──
        # Fortran: if( snum==0 .and. err1>1 ) → unallocated
        if snum == 0:
            upa_sum = uparea[iXX0, iYY0]
            err1_final = (upa_sum - area0_km2) / area0_km2
            if err1_final > 1.0:
                staX1[g] = np.int32(-9999)
                staY1[g] = np.int32(-9999)
                staA1[g] = -999.0
                continue

        snum_arr[g] = snum

    return staX1, staY1, staX2, staY2, staA1, staA2, snum_arr


# ---------------------------------------------------------------------------
# Mixin class
# ---------------------------------------------------------------------------


class FlowGaugeMixin:
    """Methods for flow-gauge allocation on :class:`HiResMap`."""

    def allocate(self: HiResMap) -> None:
        """Run the gauge allocation algorithm (numba-accelerated core)."""
        print("Allocating gauges ...")
        (
            self.staX1,
            self.staY1,
            self.staX2,
            self.staY2,
            self.staA1,
            self.staA2,
            self.snum,
        ) = allocate_all_gauges(
            self.gauge_ids,
            self.gauge_lats,
            self.gauge_lons,
            self.gauge_areas,
            self.upa1m,
            self.ctx1m,
            self.cty1m,
            self.dwx1m,
            self.dwy1m,
            self.uparea,
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
            self.mode == "single",
        )

        n_ok = int(np.sum(self.staX1 != self.MISSING))
        print(f"Allocation done: {n_ok}/{len(self.gauge_ids)} gauges allocated.")

    def results_as_structured_array(self: HiResMap) -> np.ndarray:
        """Pack flow-gauge results into a structured NumPy array.

        Fields
        ------
        id, lat, lon, area_input : from gauge list
        ix1, iy1, ix2, iy2      : allocated grid coords (0-based), -9999 if unallocated
        area1, area2             : upstream areas at allocated grids (km²)
        snum                     : 0 = outlet, 1-2 = upstream grids used
        catchment_id1, catchment_id2 : flat IDs via ravel_multi_index, -1 if unallocated
        """
        N = len(self.gauge_ids)
        dtype = np.dtype([
            ("id", np.int64),
            ("lat", np.float64),
            ("lon", np.float64),
            ("area_input", np.float64),
            ("ix1", np.int32),
            ("iy1", np.int32),
            ("ix2", np.int32),
            ("iy2", np.int32),
            ("area1", np.float64),
            ("area2", np.float64),
            ("snum", np.int32),
            ("catchment_id1", np.int64),
            ("catchment_id2", np.int64),
        ])
        arr = np.empty(N, dtype=dtype)
        arr["id"] = self.gauge_ids
        arr["lat"] = self.gauge_lats
        arr["lon"] = self.gauge_lons
        arr["area_input"] = self.gauge_areas
        arr["ix1"] = self.staX1
        arr["iy1"] = self.staY1
        arr["ix2"] = self.staX2
        arr["iy2"] = self.staY2
        arr["area1"] = self.staA1
        arr["area2"] = self.staA2
        arr["snum"] = self.snum

        # Compute flat catchment IDs (0-based)
        map_shape = (self.nXX, self.nYY)
        cid1 = np.full(N, -1, dtype=np.int64)
        cid2 = np.full(N, -1, dtype=np.int64)
        ok1 = (self.staX1 >= 0) & (self.staY1 >= 0)
        ok2 = (self.staX2 >= 0) & (self.staY2 >= 0)
        if np.any(ok1):
            cid1[ok1] = np.ravel_multi_index(
                (self.staX1[ok1], self.staY1[ok1]), map_shape
            )
        if np.any(ok2):
            cid2[ok2] = np.ravel_multi_index(
                (self.staX2[ok2], self.staY2[ok2]), map_shape
            )
        arr["catchment_id1"] = cid1
        arr["catchment_id2"] = cid2
        return arr

    def write_alloc_file(self: HiResMap) -> Path:
        """Write Fortran-compatible ``gauge_alloc.txt`` output."""
        out_path = (self.out_dir or self.map_dir) / self.out_file
        out_path.parent.mkdir(parents=True, exist_ok=True)

        header = (
            "        ID       lat       lon  area_Input   area_CaMa"
            "       error        diff    Type     ix1   iy1"
            "     ix2   iy2       area1       area2"
        )

        lines = [header]
        for i in range(len(self.gauge_ids)):
            gid = self.gauge_ids[i]
            lat = self.gauge_lats[i]
            lon = self.gauge_lons[i]
            area_in = self.gauge_areas[i]  # already km²

            ix1_out = self.staX1[i]
            iy1_out = self.staY1[i]
            ix2_out = self.staX2[i]
            iy2_out = self.staY2[i]

            if ix1_out == self.MISSING:
                # unallocated
                lines.append(
                    f"{gid:10d}{lat:10.3f}{lon:10.3f}{area_in:12.2f}"
                    f"{-999.0:12.2f}{-999.0:12.2f}{-999.0:12.2f}"
                    f"{-9:8d}{-999:8d}{-999:6d}{-999:8d}{-999:6d}"
                    f"{-999.0:12.1f}{-999.0:12.1f}"
                )
                continue

            sn = self.snum[i]
            if sn == 0:
                upa_sum = self.uparea[ix1_out, iy1_out]
            else:
                upa_sum = self.staA1[i]
                if sn >= 2:
                    upa_sum += self.staA2[i]

            diff = upa_sum - area_in
            err = diff / area_in if area_in > 0 else 0.0

            # Output uses **1-based** coordinates for Fortran compatibility
            lines.append(
                f"{gid:10d}{lat:10.3f}{lon:10.3f}{area_in:12.2f}"
                f"{upa_sum:12.2f}{err:12.2f}{diff:12.2f}"
                f"{sn:8d}{ix1_out + 1:8d}{iy1_out + 1:6d}"
                f"{ix2_out + 1 if ix2_out != self.MISSING else -9999:8d}"
                f"{iy2_out + 1 if iy2_out != self.MISSING else -9999:6d}"
                f"{self.staA1[i]:12.1f}{self.staA2[i]:12.1f}"
            )

        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Wrote allocation results to {out_path}")
        return out_path

    def to_merit_map_gauge_info(self: HiResMap) -> dict:
        """Convert allocation results to the dict format expected by
        ``MERITMap.gauge_info`` / ``MERITMap.load_gauge_id``.

        Returns a dict keyed by gauge ID (int) mapping to::

            {"upstream_id": [catchment_id, ...], "lat": float, "lon": float}

        Only successfully allocated gauges are included.
        """
        res = self.results_as_structured_array()
        gauge_info = {}
        gauge_id_set = set()

        for row in res:
            if row["catchment_id1"] < 0:
                continue
            cids = [int(row["catchment_id1"])]
            gauge_id_set.add(int(row["catchment_id1"]))
            if row["catchment_id2"] >= 0:
                cids.append(int(row["catchment_id2"]))
                gauge_id_set.add(int(row["catchment_id2"]))
            gauge_info[int(row["id"])] = {
                "upstream_id": cids,
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
            }

        return gauge_info
