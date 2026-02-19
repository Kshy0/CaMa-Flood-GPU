# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Level-gauge allocation kernel and mixin for :class:`HiResMap`.

Python re-implementation of ``fortran/allocate_level_gauge.F90``.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import numpy as np
from numba import njit

from cmfgpu.params.allocation.hires_kernels import (nextxy_hires, rgetlen,
                                                    search_best_pixel)

if TYPE_CHECKING:
    from cmfgpu.params.allocation.hires_map import HiResMap


# ---------------------------------------------------------------------------
# Numba kernel
# ---------------------------------------------------------------------------

@njit(cache=True)
def allocate_all_level_gauges(
    gauge_ids: np.ndarray,
    gauge_lats: np.ndarray,
    gauge_lons: np.ndarray,
    gauge_areas: np.ndarray,      # m²
    upa1m: np.ndarray,
    ctx1m: np.ndarray,
    cty1m: np.ndarray,
    dwx1m: np.ndarray,
    dwy1m: np.ndarray,
    uparea: np.ndarray,           # km²
    elevtn: np.ndarray,           # low-res elevation
    elv1m: np.ndarray,            # hi-res elevation
    hires_lon: np.ndarray,        # (nx,) pixel centre longitudes
    hires_lat: np.ndarray,        # (ny,) pixel centre latitudes
    upstXX: np.ndarray,
    upstYY: np.ndarray,
    outx: np.ndarray,
    outy: np.ndarray,
    west: float, north: float, gsize: float,
    west2: float, north2: float, csize: float,
    nx: int, ny: int, nXX: int, nYY: int,
    nn: int, n_ups: int, is_global: bool,
) -> Tuple[
    np.ndarray, np.ndarray,  # staX, staY
    np.ndarray,              # gauge_type (1=main, 2=trib, 3=small)
    np.ndarray, np.ndarray,  # upstream grid (jXX0, jYY0)
    np.ndarray, np.ndarray, np.ndarray,  # elv_outlet, elv_gauge, elv_upstream
    np.ndarray, np.ndarray,  # dst_outlet, dst_upstream
]:
    """Level-gauge allocation — determines position type, distance, elevation.

    Returns 10 arrays of length N.
    """
    N = gauge_ids.shape[0]
    MISS = np.int32(-9999)
    staX = np.full(N, MISS, dtype=np.int32)
    staY = np.full(N, MISS, dtype=np.int32)
    gtype = np.full(N, -9, dtype=np.int32)  # gauge type
    upsX = np.full(N, MISS, dtype=np.int32)
    upsY = np.full(N, MISS, dtype=np.int32)
    elv_outlet = np.full(N, -999.0, dtype=np.float64)
    elv_gauge = np.full(N, -999.0, dtype=np.float64)
    elv_upst = np.full(N, -999.0, dtype=np.float64)
    dst_outlet = np.full(N, -999.0, dtype=np.float64)
    dst_upst = np.full(N, -999.0, dtype=np.float64)

    for g in range(N):
        lat0 = gauge_lats[g]
        lon0 = gauge_lons[g]
        area0 = gauge_areas[g]

        east = west + nXX * gsize
        south = north - nYY * gsize
        if lon0 < west or lon0 > east or lat0 < south or lat0 > north:
            continue
        if area0 <= 0.0:
            continue

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

        staX[g] = iXX0
        staY[g] = iYY0
        elv_outlet[g] = elevtn[iXX0, iYY0]
        elv_gauge[g] = elv1m[ix0, iy0]

        # Distance gauge→outlet along hi-res river
        jx0 = outx[iXX0, iYY0]
        jy0 = outy[iXX0, iYY0]
        cur_ix = ix0
        cur_iy = iy0
        dst = 0.0
        max_steps = nx * 10
        for _ in range(max_steps):
            if cur_ix == jx0 and cur_iy == jy0:
                break
            if dwx1m[cur_ix, cur_iy] <= -900:
                break
            nix, niy = nextxy_hires(cur_ix, cur_iy, dwx1m, dwy1m, nx)
            if nix < 0 or nix >= nx or niy < 0 or niy >= ny:
                break
            dst += rgetlen(hires_lon[cur_ix], hires_lat[cur_iy],
                            hires_lon[nix], hires_lat[niy])
            if ctx1m[nix, niy] != iXX0 or cty1m[nix, niy] != iYY0:
                break
            cur_ix = nix
            cur_iy = niy
        dst_outlet[g] = dst

        # Determine type (1=mainstem, 2=tributary, 3=small stream)
        found_type = 0
        for i_ups in range(n_ups):
            jXX = upstXX[iXX0, iYY0, i_ups]
            jYY = upstYY[iXX0, iYY0, i_ups]
            if jXX < 0:
                continue
            oix = outx[jXX, jYY]
            oiy = outy[jXX, jYY]
            if oix < 0 or oiy < 0:
                continue

            # Follow from upstream outlet to see if we reach gauge pixel
            t_jx, t_jy = nextxy_hires(oix, oiy, dwx1m, dwy1m, nx)
            d = rgetlen(hires_lon[oix], hires_lat[oiy],
                         hires_lon[t_jx], hires_lat[t_jy])
            found_it = False
            for _ in range(max_steps):
                if t_jx < 0 or t_jx >= nx or t_jy < 0 or t_jy >= ny:
                    break
                if ctx1m[t_jx, t_jy] != iXX0 or cty1m[t_jx, t_jy] != iYY0:
                    break
                if dwx1m[t_jx, t_jy] <= -900:
                    break
                if t_jx == ix0 and t_jy == iy0:
                    found_it = True
                    break
                n_jx, n_jy = nextxy_hires(t_jx, t_jy, dwx1m, dwy1m, nx)
                d += rgetlen(hires_lon[t_jx], hires_lat[t_jy],
                              hires_lon[n_jx], hires_lat[n_jy])
                t_jx = n_jx
                t_jy = n_jy

            if found_it:
                if i_ups == 0:
                    found_type = 1  # mainstem
                else:
                    found_type = 2  # tributary
                upsX[g] = int(ctx1m[oix, oiy])
                upsY[g] = int(cty1m[oix, oiy])
                elv_upst[g] = elevtn[upsX[g], upsY[g]]
                dst_upst[g] = d
                break  # decided

        if found_type == 0:
            found_type = 3  # small stream
        gtype[g] = found_type

    return staX, staY, gtype, upsX, upsY, elv_outlet, elv_gauge, elv_upst, dst_outlet, dst_upst


# ---------------------------------------------------------------------------
# Mixin class
# ---------------------------------------------------------------------------


class LevelGaugeAllocMixin:
    """Methods for level-gauge allocation on :class:`HiResMap`."""

    def allocate_level_gauges(self: HiResMap) -> None:
        """Run level-gauge allocation (numba-accelerated core).

        Requires: :meth:`load_lowres`, :meth:`load_lowres_extra`,
        :meth:`load_hires`, :meth:`load_hires_elevtn`,
        :meth:`build_upstream_table`, :meth:`calc_outlet_pixels`,
        :meth:`load_gauge_list`.
        """
        print("Allocating level gauges ...")
        (
            self.lvl_staX,
            self.lvl_staY,
            self.lvl_gtype,
            self.lvl_upsX,
            self.lvl_upsY,
            self.lvl_elv_outlet,
            self.lvl_elv_gauge,
            self.lvl_elv_upst,
            self.lvl_dst_outlet,
            self.lvl_dst_upst,
        ) = allocate_all_level_gauges(
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
            self.elevtn,
            self.elv1m,
            self.hires_lon,
            self.hires_lat,
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

        n_ok = int(np.sum(self.lvl_staX != self.MISSING))
        n_main = int(np.sum(self.lvl_gtype == 1))
        n_trib = int(np.sum(self.lvl_gtype == 2))
        n_small = int(np.sum(self.lvl_gtype == 3))
        print(
            f"Level-gauge allocation done: {n_ok}/{len(self.gauge_ids)} allocated "
            f"(mainstem={n_main}, tributary={n_trib}, small={n_small})"
        )

    def level_gauge_results_as_structured_array(self: HiResMap) -> np.ndarray:
        """Pack level-gauge allocation results into a structured NumPy array.

        Fields
        ------
        id, lat, lon, area_input     : from gauge list
        ix, iy, catchment_id         : allocated grid
        gauge_type                   : 1=mainstem, 2=tributary, 3=small
        elv_outlet, elv_gauge, elv_upstream : elevations (m)
        dst_outlet, dst_upstream     : distances (km)
        """
        N = len(self.gauge_ids)
        dtype = np.dtype([
            ("id", np.int64),
            ("lat", np.float64),
            ("lon", np.float64),
            ("area_input", np.float64),
            ("ix", np.int32),
            ("iy", np.int32),
            ("catchment_id", np.int64),
            ("gauge_type", np.int32),
            ("elv_outlet", np.float64),
            ("elv_gauge", np.float64),
            ("elv_upstream", np.float64),
            ("dst_outlet", np.float64),
            ("dst_upstream", np.float64),
        ])
        arr = np.empty(N, dtype=dtype)
        arr["id"] = self.gauge_ids
        arr["lat"] = self.gauge_lats
        arr["lon"] = self.gauge_lons
        arr["area_input"] = self.gauge_areas
        arr["ix"] = self.lvl_staX
        arr["iy"] = self.lvl_staY
        arr["gauge_type"] = self.lvl_gtype
        arr["elv_outlet"] = self.lvl_elv_outlet
        arr["elv_gauge"] = self.lvl_elv_gauge
        arr["elv_upstream"] = self.lvl_elv_upst
        arr["dst_outlet"] = self.lvl_dst_outlet
        arr["dst_upstream"] = self.lvl_dst_upst

        cid = np.full(N, -1, dtype=np.int64)
        ok = (self.lvl_staX >= 0) & (self.lvl_staY >= 0)
        if np.any(ok):
            cid[ok] = np.ravel_multi_index(
                (self.lvl_staX[ok], self.lvl_staY[ok]), (self.nXX, self.nYY)
            )
        arr["catchment_id"] = cid
        return arr

    def write_level_gauge_alloc_file(self: HiResMap, out_path: str | Path | None = None) -> Path:
        """Write level-gauge allocation results to text file."""
        if out_path is None:
            out_path = (self.out_dir or self.map_dir) / "level_gauge_alloc.txt"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        header = (
            "        ID       lat       lon  area_Input"
            "  gtype     ix    iy  catchment_id"
            "  elv_outlet  elv_gauge  elv_upst  dst_outlet  dst_upst"
        )
        lines = [header]
        type_names = {1: "MAIN", 2: "TRIB", 3: "SMLL", -9: " ERR"}
        for i in range(len(self.gauge_ids)):
            gid = self.gauge_ids[i]
            lat = self.gauge_lats[i]
            lon = self.gauge_lons[i]
            area_in = self.gauge_areas[i]  # already km²
            gt = self.lvl_gtype[i]
            ix_out = self.lvl_staX[i]
            iy_out = self.lvl_staY[i]
            cid = int(np.ravel_multi_index(
                (ix_out, iy_out), (self.nXX, self.nYY)
            )) if ix_out >= 0 else -1

            lines.append(
                f"{gid:10d}{lat:10.3f}{lon:10.3f}{area_in:12.2f}"
                f"  {type_names.get(gt, '????'):>4s}"
                f"{ix_out + 1:8d}{iy_out + 1:6d}{cid:14d}"
                f"{self.lvl_elv_outlet[i]:12.2f}{self.lvl_elv_gauge[i]:12.2f}"
                f"{self.lvl_elv_upst[i]:12.2f}"
                f"{self.lvl_dst_outlet[i]:12.2f}{self.lvl_dst_upst[i]:12.2f}"
            )

        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Wrote level-gauge allocation to {out_path}")
        return out_path
