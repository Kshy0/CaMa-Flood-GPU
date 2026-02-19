# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Shared numba-accelerated kernels for hi-res gauge / dam / level-gauge
allocation on CaMa-Flood grids.

These are pure functions with no class dependency — they operate solely on
NumPy arrays and scalars and are imported by the allocation mixin modules.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Upstream table builder
# ---------------------------------------------------------------------------

@njit(cache=True)
def build_upstream_table(
    nextXX: np.ndarray,   # (nXX, nYY) int32, 0-based downstream X (-9999 = mouth/invalid)
    nextYY: np.ndarray,   # (nXX, nYY) int32
    uparea: np.ndarray,   # (nXX, nYY) float32, upstream area in km²
    n_ups: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build upstream grid table (up to *n_ups* largest upstream grids per cell).

    Iterates *n_ups* passes: in each pass, for every valid cell, register it at
    its downstream cell if it has not been registered before and has the largest
    remaining upstream area.

    Returns
    -------
    upstXX, upstYY : (nXX, nYY, n_ups) int32
        Upstream grid coordinates; -9999 where no upstream exists.
    """
    nXX, nYY = nextXX.shape
    MISS = np.int32(-9999)
    upstXX = np.full((nXX, nYY, n_ups), MISS, dtype=np.int32)
    upstYY = np.full((nXX, nYY, n_ups), MISS, dtype=np.int32)
    maxupa = np.zeros((nXX, nYY), dtype=np.float32)

    for i_ups in range(n_ups):
        maxupa[:, :] = 0.0
        for iYY in range(nYY):
            for iXX in range(nXX):
                jXX = nextXX[iXX, iYY]
                jYY = nextYY[iXX, iYY]
                if jXX < 0 or jYY < 0 or jXX >= nXX or jYY >= nYY:
                    continue
                # skip if already registered in a previous pass
                already = False
                for j_ups in range(i_ups):
                    if iXX == upstXX[jXX, jYY, j_ups] and iYY == upstYY[jXX, jYY, j_ups]:
                        already = True
                        break
                if already:
                    continue
                if uparea[iXX, iYY] > maxupa[jXX, jYY]:
                    maxupa[jXX, jYY] = uparea[iXX, iYY]
                    upstXX[jXX, jYY, i_ups] = iXX
                    upstYY[jXX, jYY, i_ups] = iYY
    return upstXX, upstYY


# ---------------------------------------------------------------------------
# Outlet pixel detection
# ---------------------------------------------------------------------------

@njit(cache=True)
def calc_outlet_pixels(
    ctx1m: np.ndarray,   # (nx, ny) int16, 0-based unit-catchment X for each hi-res pixel
    cty1m: np.ndarray,   # (nx, ny) int16
    dwx1m: np.ndarray,   # (nx, ny) int16, downstream pixel offset dx
    dwy1m: np.ndarray,   # (nx, ny) int16, downstream pixel offset dy
    upa1m: np.ndarray,   # (nx, ny) float32, hi-res upstream area
    nXX: int,
    nYY: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the outlet pixel (in hi-res coords) of each unit-catchment.

    A pixel is an *outlet* if either:
    - It is a river mouth (``dwx1m <= -900``), or
    - Its downstream pixel belongs to a *different* unit-catchment.

    When multiple outlets exist (possible after resampling), the one draining
    to the pixel with the largest upstream area is selected.

    Returns
    -------
    outx, outy : (nXX, nYY) int32
        Hi-res outlet pixel coordinate for each unit-catchment; -9999 if none.
    """
    nx, ny = ctx1m.shape
    MISS = np.int32(-9999)
    outx = np.full((nXX, nYY), MISS, dtype=np.int32)
    outy = np.full((nXX, nYY), MISS, dtype=np.int32)
    maxupa = np.zeros((nXX, nYY), dtype=np.float32)

    for iy in range(ny):
        for ix in range(nx):
            ciXX = ctx1m[ix, iy]
            ciYY = cty1m[ix, iy]
            if ciXX < 0 or ciYY < 0 or ciXX >= nXX or ciYY >= nYY:
                continue

            if dwx1m[ix, iy] <= -900:
                # river mouth pixel
                outx[ciXX, ciYY] = ix
                outy[ciXX, ciYY] = iy
            else:
                # compute downstream pixel
                jx = ix + dwx1m[ix, iy]
                jy = iy + dwy1m[ix, iy]
                # wrap in x (global)
                if jx < 0:
                    jx += nx
                if jx >= nx:
                    jx -= nx
                if jx < 0 or jx >= nx or jy < 0 or jy >= ny:
                    continue
                # outlet if downstream pixel belongs to a different unit-catchment
                if ctx1m[jx, jy] != ciXX or cty1m[jx, jy] != ciYY:
                    if outx[ciXX, ciYY] != MISS:
                        # multiple outlets: pick the one with larger downstream uparea
                        if upa1m[jx, jy] > maxupa[ciXX, ciYY]:
                            maxupa[ciXX, ciYY] = upa1m[jx, jy]
                            outx[ciXX, ciYY] = ix
                            outy[ciXX, ciYY] = iy
                    else:
                        outx[ciXX, ciYY] = ix
                        outy[ciXX, ciYY] = iy
                        maxupa[ciXX, ciYY] = upa1m[jx, jy]
    return outx, outy


# ---------------------------------------------------------------------------
# Hi-res pixel navigation
# ---------------------------------------------------------------------------

@njit(cache=True)
def nextxy_hires(ix: int, iy: int, dwx1m: np.ndarray, dwy1m: np.ndarray, nx: int) -> Tuple[int, int]:
    """Compute next hi-res pixel, wrapping in x for global domain."""
    jx = ix + dwx1m[ix, iy]
    jy = iy + dwy1m[ix, iy]
    if jx < 0:
        jx += nx
    if jx >= nx:
        jx -= nx
    return jx, jy


# ---------------------------------------------------------------------------
# Neighbourhood search
# ---------------------------------------------------------------------------

@njit(cache=True)
def search_best_pixel(
    ix_center: int,
    iy_center: int,
    upa1m: np.ndarray,
    area0: float,
    nn: int,
    nx: int,
    ny: int,
    is_global: bool,
) -> Tuple[int, int, float, float, float]:
    """Search ±nn neighbourhood of (ix_center, iy_center) on the hi-res grid.

    Returns (best_kx, best_ky, best_err1, best_area, best_rate).
    best_kx == -1 means no match found.
    """
    best_kx = np.int32(-1)
    best_ky = np.int32(-1)
    best_err1 = np.float64(0.0)
    best_area = np.float64(0.0)
    best_rate = np.float64(1.0e20)

    for dy in range(-nn, nn + 1):
        for dx in range(-nn, nn + 1):
            jx = ix_center + dx
            jy = iy_center + dy
            if is_global:
                if jx < 0:
                    jx += nx
                if jx >= nx:
                    jx -= nx
            if jx < 0 or jx >= nx or jy < 0 or jy >= ny:
                continue
            a = upa1m[jx, jy]
            if a <= area0 * 0.05:
                continue
            err = (a - area0) / area0
            dd = (dx * dx + dy * dy) ** 0.5
            err2 = err + 0.02 * dd if err > 0 else err - 0.02 * dd

            if err2 >= 0:
                rate = 1.0 + err2
            elif err2 > -1.0:
                rate = 1.0 / (1.0 + err2)
                if rate > 1000.0:
                    rate = 1000.0
            else:
                rate = 1000.0

            if rate < best_rate:
                best_rate = rate
                best_err1 = err
                best_kx = jx
                best_ky = jy
                best_area = a
    return best_kx, best_ky, best_err1, best_area, best_rate


# ---------------------------------------------------------------------------
# Downstream river tracing
# ---------------------------------------------------------------------------

@njit(cache=True)
def trace_gauge_downstream(
    ix0: int,
    iy0: int,
    iXX0: int,
    iYY0: int,
    out_ix: int,
    out_iy: int,
    ctx1m: np.ndarray,
    cty1m: np.ndarray,
    dwx1m: np.ndarray,
    dwy1m: np.ndarray,
    nx: int,
) -> bool:
    """Follow hi-res river from *upstream outlet* downstream until leaving the
    target unit-catchment (iXX0, iYY0) or reaching the gauge pixel (ix0, iy0).

    Returns True if the gauge pixel is found downstream of the upstream outlet.
    """
    jx, jy = nextxy_hires(out_ix, out_iy, dwx1m, dwy1m, nx)
    max_steps = nx * 10  # safety limit
    for _ in range(max_steps):
        if jx < 0 or jx >= ctx1m.shape[0] or jy < 0 or jy >= ctx1m.shape[1]:
            return False
        if ctx1m[jx, jy] != iXX0 or cty1m[jx, jy] != iYY0:
            return False
        if dwx1m[jx, jy] <= -900:
            return False  # reached mouth without hitting gauge
        if jx == ix0 and jy == iy0:
            return True
        jx, jy = nextxy_hires(jx, jy, dwx1m, dwy1m, nx)
    return False


# ---------------------------------------------------------------------------
# Geodesic distance (Fortran rgetlen equivalent)
# ---------------------------------------------------------------------------

@njit(cache=True)
def rgetlen(rlon1: float, rlat1: float, rlon2: float, rlat2: float) -> float:
    """Geodesic distance between two lon/lat points [km].

    Mirrors the Fortran ``rgetlen`` function in ``allocate_level_gauge.F90``.
    Uses the chord-length formula on an ellipsoidal Earth (WGS-84 semi-major
    axis, eccentricity²=0.006694470).
    """
    PI = 3.141592653589793
    DA = 6378137.0          # semi-major axis [m]
    DE2 = 0.006694470       # eccentricity²

    rlat1_r = rlat1 * PI / 180.0
    rlon1_r = rlon1 * PI / 180.0
    rlat2_r = rlat2 * PI / 180.0
    rlon2_r = rlon2 * PI / 180.0

    slat1 = math.sin(rlat1_r)
    clat1 = math.cos(rlat1_r)
    slon1 = math.sin(rlon1_r)
    clon1 = math.cos(rlon1_r)

    dn1 = DA / math.sqrt(1.0 - DE2 * slat1 * slat1)
    dx1 = dn1 * clat1 * clon1
    dy1 = dn1 * clat1 * slon1
    dz1 = dn1 * (1.0 - DE2) * slat1

    slat2 = math.sin(rlat2_r)
    clat2 = math.cos(rlat2_r)
    slon2 = math.sin(rlon2_r)
    clon2 = math.cos(rlon2_r)

    dn2 = DA / math.sqrt(1.0 - DE2 * slat2 * slat2)
    dx2 = dn2 * clat2 * clon2
    dy2 = dn2 * clat2 * slon2
    dz2 = dn2 * (1.0 - DE2) * slat2

    dlen = math.sqrt((dx1 - dx2) ** 2 + (dy1 - dy2) ** 2 + (dz1 - dz2) ** 2)
    drad = math.asin(min(dlen / (2.0 * DA), 1.0))
    return drad * 2.0 * DA * 0.001   # m → km
