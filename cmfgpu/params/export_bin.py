# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Export CaMa-Flood-GPU NetCDF parameter files back to CaMa-Flood v4 binary format.

This module converts:
    1. parameters.nc  →  nextxy.bin, ctmare.bin, elevtn.bin, nxtdst.bin,
                          rivlen.bin, fldhgt.bin, rivwth_gwdlr.bin, rivhgt.bin,
                          rivman.bin, lonlat.bin, width.bin,
                          levhgt.bin, levfrc.bin, bifprm.txt, dam_params.csv,
                          mapdim.txt
    2. runoff_mapping.npz  →  inpmat*.bin, diminfo*.txt

Both full-grid and cropped (POI-filtered) parameters are supported.
When ``crop_to_bbox=True`` (default for cropped NC), the output grid is
shrunk to the bounding box of the catchment subset, producing much smaller
binary files.  All Fortran-facing indices (nextxy, bifprm, inpmat RECL) are
remapped to the cropped coordinate system.

Binary conventions (matching CaMa-Flood Fortran):
    * Little-endian
    * Indices: int32 (<i4)
    * Physical fields: float32 (<f4)
    * Fortran column-major (order='F')
    * nextxy.bin uses 1-based indexing; all river mouths → (-9, -9)
    * inpmat is Fortran DIRECT ACCESS (RECL = 4*NX*NY), no record markers.
    * Manning coefficients are not stored in the NC; uniform defaults
      (river_manning=0.03, flood_manning=0.1) are used.

Usage example
-------------
>>> from cmfgpu.params.export_bin import export_to_cama_bin
>>> export_to_cama_bin(
...     nc_path="inp/glb_15min/parameters.nc",
...     out_dir="exported/glb_15min",
...     npz_path="inp/glb_15min/runoff_mapping_nc.npz",       # optional
...     diminfo_name="diminfo_test-15min_nc.txt",              # optional
...     inpmat_name="inpmat_test-15min_nc.bin",                # optional
... )

>>> # Cropped export with minimal file size:
>>> export_to_cama_bin(
...     nc_path="inp/glb_15min/parameters_mxxb.nc",
...     out_dir="exported/mxxb",
...     npz_path="inp/glb_15min/runoff_mapping_nc.npz",
...     crop_to_bbox=True,   # shrink grid to catchment bounding box
... )
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from netCDF4 import Dataset
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _compute_bbox(
    catchment_x: np.ndarray,
    catchment_y: np.ndarray,
    full_nx: int,
    full_ny: int,
    full_west: float = -180.0,
    full_north: float = 90.0,
) -> Tuple[int, int, int, int, int, int, float, float, float, float]:
    """Compute the bounding box of the catchment subset.

    Returns
    -------
    x_min, x_max, y_min, y_max : int
        0-based bounding box indices in the full grid.
    crop_nx, crop_ny : int
        Dimensions of the cropped grid.
    west, east, north, south : float
        Geographic extent of the cropped grid.
    """
    x_min = int(catchment_x.min())
    x_max = int(catchment_x.max())
    y_min = int(catchment_y.min())
    y_max = int(catchment_y.max())
    crop_nx = x_max - x_min + 1
    crop_ny = y_max - y_min + 1

    dlon = (360.0) / full_nx   # e.g. 0.25° for 1440
    dlat = (180.0) / full_ny   # e.g. 0.25° for 720
    west = full_west + x_min * dlon
    east = full_west + (x_max + 1) * dlon
    north = full_north - y_min * dlat
    south = full_north - (y_max + 1) * dlat

    return x_min, x_max, y_min, y_max, crop_nx, crop_ny, west, east, north, south

def _write_bin(path: Path, data: np.ndarray) -> None:
    """Write *data* as a Fortran-order flat binary file (no record markers)."""
    data.flatten(order="F").tofile(path)
    print(f"  Wrote {path.name}  shape={data.shape}  dtype={data.dtype}")


def _scatter_1d_to_2d(
    nx: int,
    ny: int,
    cx: np.ndarray,
    cy: np.ndarray,
    values: np.ndarray,
    fill: float = -9999.0,
    dtype: str = "<f4",
) -> np.ndarray:
    """Scatter 1-D catchment-indexed *values* back onto a 2-D (nx, ny) grid."""
    grid = np.full((nx, ny), fill, dtype=dtype)
    grid[cx, cy] = values.astype(dtype)
    return grid


def _scatter_1d_to_3d(
    nx: int,
    ny: int,
    nz: int,
    cx: np.ndarray,
    cy: np.ndarray,
    values: np.ndarray,
    fill: float = -9999.0,
    dtype: str = "<f4",
) -> np.ndarray:
    """Scatter 1-D catchment-indexed *values* (shape [n, nz]) onto (nx, ny, nz)."""
    grid = np.full((nx, ny, nz), fill, dtype=dtype)
    grid[cx, cy, :] = values.astype(dtype)
    return grid


# ---------------------------------------------------------------------------
# nextxy reconstruction
# ---------------------------------------------------------------------------

_MOUTH_OCEAN = -9     # CaMa-Flood convention for ocean outlet


def _rebuild_nextxy(
    nx: int,
    ny: int,
    catchment_id: np.ndarray,
    downstream_id: np.ndarray,
    catchment_x: np.ndarray,
    catchment_y: np.ndarray,
    missing_int: int = -9999,
) -> np.ndarray:
    """Reconstruct nextxy.bin (int32, [nx, ny, 2], 1-based).

    All river mouths are encoded as (-9, -9) (ocean outlet).  The
    distinction between ocean outlets and inland sinks has no effect on
    CaMa-Flood computation, so we use (-9, -9) uniformly.
    """
    nextxy = np.full((nx, ny, 2), missing_int, dtype="<i4")

    is_mouth = downstream_id == catchment_id  # mouths point to themselves in NC

    # Build id → (x, y) lookup
    id_to_x = dict(zip(catchment_id, catchment_x))
    id_to_y = dict(zip(catchment_id, catchment_y))

    for i in range(len(catchment_id)):
        ix, iy = int(catchment_x[i]), int(catchment_y[i])
        if is_mouth[i]:
            nextxy[ix, iy, :] = _MOUTH_OCEAN
        else:
            ds_id = int(downstream_id[i])
            if ds_id in id_to_x:
                nextxy[ix, iy, 0] = int(id_to_x[ds_id]) + 1  # 1-based
                nextxy[ix, iy, 1] = int(id_to_y[ds_id]) + 1
            else:
                # Downstream cell not in subset → treat as mouth
                nextxy[ix, iy, :] = _MOUTH_OCEAN
    return nextxy


# ---------------------------------------------------------------------------
# bifprm.txt writer
# ---------------------------------------------------------------------------

def _write_bifprm(
    path: Path,
    bif_cx: np.ndarray,
    bif_cy: np.ndarray,
    bif_dx: np.ndarray,
    bif_dy: np.ndarray,
    bif_length: np.ndarray,
    bif_elevation: np.ndarray,
    bif_width: np.ndarray,
    longitude: Optional[np.ndarray],
    latitude: Optional[np.ndarray],
    catchment_x: np.ndarray,
    catchment_y: np.ndarray,
    catchment_id: np.ndarray,
    river_mouth_id: Optional[np.ndarray],
    catchment_elevation: Optional[np.ndarray] = None,
    missing_int: int = -9999,
) -> None:
    """Write CaMa-Flood v4 ``bifprm.txt``.

    Format (per path, one printed line)::

        ix iy  jx jy  length  elevtn  depth  width1 ... widthN  lat lon basin_up basin_dn

    Coordinates are 1-based.

    **Elevation / depth recovery** — The NetCDF stores per-level ``PTH_ELV``
    (the threshold elevation at which flow activates for each level), but
    Fortran expects ``PELV`` (ground surface elevation) and ``PDPH``
    (channel depth) and **recomputes** ``PTH_ELV`` at runtime via::

        Level 0 (channel):   PTH_ELV = PELV - PDPH
        Level k (floodplain, k>=2, 1-based): PTH_ELV = PELV + k - 2

    We therefore reverse-engineer ``PELV`` from the first valid higher-level
    entry and ``PDPH = PELV - PTH_ELV[0]`` when level 0 is active.
    """
    n_paths = len(bif_cx)
    n_levels = bif_width.shape[1] if bif_width.ndim == 2 else 1

    # Build coordinate lookups for lat/lon
    if longitude is not None and latitude is not None:
        id_to_lon = dict(zip(catchment_id, longitude))
        id_to_lat = dict(zip(catchment_id, latitude))
    else:
        id_to_lon = id_to_lat = None

    # Build river-mouth lookup for basin IDs
    if river_mouth_id is not None:
        id_to_mouth = dict(zip(catchment_id, river_mouth_id))
    else:
        id_to_mouth = None

    # Map (x, y) → catchment_id
    xy_to_cid = {}
    for cid, cx, cy in zip(catchment_id, catchment_x, catchment_y):
        xy_to_cid[(int(cx), int(cy))] = int(cid)

    # Map (x, y) → ground elevation (for PELV fallback)
    xy_to_elevtn = {}
    if catchment_elevation is not None:
        for cx, cy, elv in zip(catchment_x, catchment_y, catchment_elevation):
            xy_to_elevtn[(int(cx), int(cy))] = float(elv)

    with open(path, "w") as f:
        f.write(
            f"{n_paths:8d}{n_levels:8d}  npath_new, nlev_new, "
            "(ix,iy), (jx,jy), length, elevtn, depth, "
            "(width1, width2, ... width_nlev), (lat,lon), (basins)\n"
        )
        for i in range(n_paths):
            ix = int(bif_cx[i]) + 1   # 1-based
            iy = int(bif_cy[i]) + 1
            jx = int(bif_dx[i]) + 1
            jy = int(bif_dy[i]) + 1
            length_val = float(bif_length[i])

            # ---------------------------------------------------------
            # Recover PELV (ground elevation) and PDPH (channel depth)
            # from the per-level PTH_ELV stored in the NetCDF.
            #
            # Fortran formula (1-based ILEV):
            #   ILEV=1 (channel):    PTH_ELV = PELV - PDPH
            #   ILEV>=2 (floodplain): PTH_ELV = PELV + ILEV - 2
            #
            # Recovery:
            #   PELV = PTH_ELV[k] - (k+1 - 2)   for any valid k>=1 (0-based)
            #   PDPH = PELV - PTH_ELV[0]         if level 0 active
            # ---------------------------------------------------------
            if bif_elevation.ndim == 2:
                elvs = bif_elevation[i, :]
            else:
                elvs = np.array([float(bif_elevation[i])])

            elev_val = -9999.0   # PELV
            depth_val = -9999.0  # PDPH

            # Try to recover PELV from a valid higher level (0-based k>=1)
            for _k in range(1, len(elvs)):
                _w = float(bif_width[i, _k]) if bif_width.ndim == 2 else 0.0
                _e = float(elvs[_k])
                if _w > 0 and abs(_e) < 1e10:
                    elev_val = _e - (_k + 1 - 2)   # PELV
                    break

            # Fallback: use catchment_elevation at the upstream cell
            if abs(elev_val - (-9999.0)) < 1.0:
                up_elevtn = xy_to_elevtn.get(
                    (int(bif_cx[i]), int(bif_cy[i])), None
                )
                if up_elevtn is not None and abs(up_elevtn) < 1e10:
                    elev_val = up_elevtn

            # Recover PDPH from level 0
            if bif_width.ndim == 2:
                w0 = float(bif_width[i, 0])
            else:
                w0 = float(bif_width[i])
            e0 = float(elvs[0])
            if w0 > 0 and abs(e0) < 1e10 and abs(elev_val - (-9999.0)) > 1.0:
                depth_val = elev_val - e0  # PDPH = PELV - PTH_ELV[0]

            # Widths – also clamp fill values
            if bif_width.ndim == 2:
                widths = [float(w) if abs(float(w)) < 1e10 else 0.0
                          for w in bif_width[i, :]]
            else:
                w0 = float(bif_width[i])
                widths = [w0 if abs(w0) < 1e10 else 0.0]

            # Lat / Lon from upstream cell
            up_cid = xy_to_cid.get((int(bif_cx[i]), int(bif_cy[i])), None)
            if id_to_lat is not None and up_cid is not None:
                lat_val = float(id_to_lat.get(up_cid, 0.0))
                lon_val = float(id_to_lon.get(up_cid, 0.0))
            else:
                lat_val = 0.0
                lon_val = 0.0

            # Basin IDs from river mouths
            dn_cid = xy_to_cid.get((int(bif_dx[i]), int(bif_dy[i])), None)
            if id_to_mouth is not None:
                basin_up = int(id_to_mouth.get(up_cid, missing_int)) if up_cid is not None else missing_int
                basin_dn = int(id_to_mouth.get(dn_cid, missing_int)) if dn_cid is not None else missing_int
            else:
                basin_up = missing_int
                basin_dn = missing_int

            # --- Write 2-line record ---
            # Line 1: ix iy jx jy length elevtn depth width1 ...
            parts = [f"{ix:8d}", f"{iy:8d}", f"{jx:8d}", f"{jy:8d}"]
            parts.append(f"{length_val:12.2f}")
            parts.append(f"{elev_val:12.2f}")
            parts.append(f"{depth_val:12.2f}")
            for w in widths:
                parts.append(f"{float(w):12.2f}")
            # Line 2 continuation: lat lon basin_up basin_dn
            parts.append(f"{lat_val:10.3f}")
            parts.append(f"{lon_val:10.3f}")
            parts.append(f"{basin_up:8d}")
            parts.append(f"{basin_dn:8d}")

            f.write("".join(parts) + "\n")

    print(f"  Wrote bifprm.txt  ({n_paths} paths, {n_levels} levels)")


# ---------------------------------------------------------------------------
# dam_params.csv writer
# ---------------------------------------------------------------------------

def _write_dam_params_csv(
    path: Path,
    reservoir_catchment_id: np.ndarray,
    reservoir_capacity: np.ndarray,
    conservation_volume: np.ndarray,
    emergency_volume: np.ndarray,
    normal_outflow: np.ndarray,
    flood_control_outflow: np.ndarray,
    catchment_id: np.ndarray,
    full_ny: int,
    x_min: int = 0,
    y_min: int = 0,
    out_nx: int = 0,
    out_ny: int = 0,
    longitude: Optional[np.ndarray] = None,
    latitude: Optional[np.ndarray] = None,
    upstream_area: Optional[np.ndarray] = None,
) -> None:
    """Write ``dam_params.csv`` in the format expected by CaMa-Flood Fortran.

    The CSV has the same layout as ``cmf_ctrl_damout_mod.F90``::

        Line 1  :  NDAM
        Line 2  :  header (skipped by Fortran)
        Line 3+ :  DamID  DamName  DamLat  DamLon  upreal
                    DamIX  DamIY  FldVol_mcm  ConVol_mcm  TotVol_mcm  Qn  Qf

    Only dams whose grid cell falls within the output grid ``[0, out_nx) ×
    [0, out_ny)`` are retained (relevant when ``crop_to_bbox=True``).

    Parameters
    ----------
    reservoir_catchment_id : 1-D int64
        Ravel-index of each dam in the **full** (uncropped) grid.
    reservoir_capacity, conservation_volume, emergency_volume : 1-D float
        Storage parameters in m³.  ``emergency_volume = ConVol + FldVol*0.95``.
    normal_outflow, flood_control_outflow : 1-D float
        Outflow parameters in m³ s⁻¹ (**raw**, before Yamazaki–Funato
        modification; Fortran re-derives effective values at runtime).
    catchment_id : 1-D int64
        Catchment IDs of the domain (already filtered to current NC).
    full_ny : int
        Column count of the **full** (uncropped) grid, used to unravel
        ``reservoir_catchment_id``.
    x_min, y_min : int
        Crop offset (0 when not cropping).
    out_nx, out_ny : int
        Output grid dimensions.
    longitude, latitude : 1-D float, optional
        Catchment-indexed lon/lat (same order as *catchment_id*).
    upstream_area : 1-D float, optional
        Catchment-indexed upstream drainage area in m² (same order as
        *catchment_id*).
    """
    n_total = len(reservoir_catchment_id)
    if n_total == 0:
        return

    # ---- Unravel reservoir_catchment_id → (full_x, full_y) ----
    full_x = reservoir_catchment_id // full_ny
    full_y = reservoir_catchment_id % full_ny

    # ---- Offset to output-grid coordinates ----
    dam_cx = full_x - x_min
    dam_cy = full_y - y_min

    # ---- Filter to dams inside the output grid ----
    inside = (
        (dam_cx >= 0) & (dam_cx < out_nx) &
        (dam_cy >= 0) & (dam_cy < out_ny)
    )
    idx_keep = np.nonzero(inside)[0]
    n_kept = len(idx_keep)
    if n_kept == 0:
        print("  No dams inside the output grid — skipping dam_params.csv")
        return

    # ---- Build catchment_id → array-index lookup ----
    cid_to_idx = {int(cid): i for i, cid in enumerate(catchment_id)}

    # ---- Reverse-engineer FldVol from emergency_volume and conservation_volume ----
    #      EmeVol = ConVol + FldVol * 0.95  →  FldVol = (EmeVol - ConVol) / 0.95
    flood_volume = (emergency_volume - conservation_volume) / 0.95

    # ---- Write CSV ----
    with open(path, "w") as f:
        f.write(f"{n_kept}\n")
        f.write("DamID DamName DamLat DamLon upreal "
                "DamIX DamIY FldVol_mcm ConVol_mcm TotVol_mcm Qn Qf\n")

        for rank, i in enumerate(idx_keep):
            dam_id = rank + 1                        # sequential 1-based ID
            dam_name = f"DAM_{dam_id}"

            ix_out = int(dam_cx[i]) + 1              # 1-based Fortran index
            iy_out = int(dam_cy[i]) + 1

            fld_mcm = float(flood_volume[i]) / 1.0e6
            con_mcm = float(conservation_volume[i]) / 1.0e6
            tot_mcm = float(reservoir_capacity[i]) / 1.0e6
            qn_val = float(normal_outflow[i])
            qf_val = float(flood_control_outflow[i])

            # Lat / Lon / upstream area from catchment arrays
            cid_int = int(reservoir_catchment_id[i])
            catch_idx = cid_to_idx.get(cid_int, None)
            if catch_idx is not None and latitude is not None:
                lat_val = float(latitude[catch_idx])
                lon_val = float(longitude[catch_idx])
            else:
                lat_val = 0.0
                lon_val = 0.0

            if catch_idx is not None and upstream_area is not None:
                upreal_km2 = float(upstream_area[catch_idx]) / 1.0e6  # m² → km²
            else:
                upreal_km2 = 0.0

            f.write(
                f"{dam_id} {dam_name} {lat_val:.4f} {lon_val:.4f} "
                f"{upreal_km2:.2f} {ix_out} {iy_out} "
                f"{fld_mcm:.4f} {con_mcm:.4f} {tot_mcm:.4f} "
                f"{qn_val:.4f} {qf_val:.4f}\n"
            )

    print(f"  Wrote dam_params.csv  ({n_kept} dams"
          + (f", {n_total - n_kept} outside crop" if n_kept < n_total else "")
          + ")")


# ---------------------------------------------------------------------------
# mapdim / diminfo writers
# ---------------------------------------------------------------------------

def _write_mapdim(path: Path, nx: int, ny: int, nlfp: int) -> None:
    with open(path, "w") as f:
        f.write(f"{nx:10d}    !! nXX\n")
        f.write(f"{ny:10d}    !! nYY\n")
        f.write(f"{nlfp:10d}    !! floodplain layer\n")
    print(f"  Wrote mapdim.txt  (nx={nx}, ny={ny}, nlfp={nlfp})")


def _write_crop_info(
    path: Path,
    full_nx: int,
    full_ny: int,
    x_min: int,
    y_min: int,
    crop_nx: int,
    crop_ny: int,
) -> None:
    """Write ``crop_info.txt`` — metadata for mapping between cropped and
    full-grid coordinate systems.

    This file is **essential** when the exported binaries use a cropped grid
    (``crop_to_bbox=True``).  Without it, CaMa-Flood Fortran output cannot
    be mapped back to the original GPU ``catchment_id`` values, because:

        ``catchment_id = catchment_x × full_ny + catchment_y``

    After cropping, the bin grid uses offset coordinates
    ``(crop_x, crop_y) = (catchment_x − x_min, catchment_y − y_min)``.
    To recover the original ``catchment_id`` from Fortran output:

        ``catchment_id = (crop_x + x_min) × full_ny + (crop_y + y_min)``

    Fields
    ------
    full_nx, full_ny : original global grid dimensions (e.g. 1440, 720)
    x_min, y_min     : 0-based offset of the crop window in the full grid
    crop_nx, crop_ny : cropped grid dimensions
    """
    with open(path, "w") as f:
        f.write(f"{full_nx:10d}    !! full_nx  (original global grid)\n")
        f.write(f"{full_ny:10d}    !! full_ny\n")
        f.write(f"{x_min:10d}    !! x_min    (0-based crop offset)\n")
        f.write(f"{y_min:10d}    !! y_min\n")
        f.write(f"{crop_nx:10d}    !! crop_nx  (cropped grid size)\n")
        f.write(f"{crop_ny:10d}    !! crop_ny\n")
    print(f"  Wrote crop_info.txt  (offset=({x_min},{y_min}), "
          f"full={full_nx}×{full_ny} → crop={crop_nx}×{crop_ny})")


def _write_diminfo(
    path: Path,
    nx: int,
    ny: int,
    nlfp: int,
    nxin: int,
    nyin: int,
    inpn: int,
    inpmat_name: str,
    west: float,
    east: float,
    north: float,
    south: float,
) -> None:
    with open(path, "w") as f:
        f.write(f"{nx:10d}     !! nXX\n")
        f.write(f"{ny:10d}     !! nYY\n")
        f.write(f"{nlfp:10d}     !! floodplain layer\n")
        f.write(f"{nxin:10d}     !! input nXX\n")
        f.write(f"{nyin:10d}     !! input nYY\n")
        f.write(f"{inpn:10d}     !! input num\n")
        f.write(f"{inpmat_name}\n")
        f.write(f"{west:14.3f}     !! west  edge\n")
        f.write(f"{east:14.3f}     !! east  edge\n")
        f.write(f"{north:14.3f}     !! north edge\n")
        f.write(f"{south:14.3f}     !! south edge\n")
    print(f"  Wrote {path.name}")


# ===========================================================================
# Main: export NC → binary map files
# ===========================================================================

def export_map_params(
    nc_path: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    crop_to_bbox: bool = False,
    river_manning_default: float = 0.03,
    flood_manning_default: float = 0.1,
    missing_float: float = -9999.0,
    missing_int: int = -9999,
) -> Tuple[Path, int, int, int, float, float, float, float]:
    """Convert a CaMa-Flood-GPU ``parameters.nc`` back to CaMa-Flood v4 binary files.

    Generates the following files in *out_dir*:

    * ``nextxy.bin``  – downstream connectivity (int32, 1-based)
    * ``ctmare.bin``  – unit-catchment area [m²]
    * ``elevtn.bin``  – channel-top elevation [m]
    * ``nxtdst.bin``  – downstream distance [m]
    * ``rivlen.bin``  – river channel length [m]
    * ``fldhgt.bin``  – floodplain elevation profile [m]
    * ``rivwth_gwdlr.bin`` – channel width [m]  (if ``river_width`` present)
    * ``rivhgt.bin``  – channel depth [m]  (if ``river_height`` present)
    * ``rivman.bin``  – Manning coefficient (uniform *river_manning_default*)
    * ``lonlat.bin``  – lon/lat coordinates  (if present)
    * ``width.bin``   – satellite-derived width (if ``satellite_width`` present)
    * ``levhgt.bin``  – levee crown height [m]  (if ``levee_crown_height`` present)
    * ``levfrc.bin``  – unprotected fraction [0–1]  (if ``levee_fraction`` present)
    * ``bifprm.txt``  – bifurcation channel table (if bifurcations present)
    * ``dam_params.csv`` – reservoir/dam parameter table (if reservoirs present)
    * ``mapdim.txt``  – grid dimensions

    Notes
    -----
    * When ``crop_to_bbox=True``, the output grid is shrunk to the
      tight bounding box of the catchment subset.  All index-based fields
      (nextxy, bifprm) are remapped to the smaller coordinate system.
      This can dramatically reduce file sizes for small subsets.
    * All river mouths are written as ocean outlets (-9, -9).
    * ``nxtdst`` at mouths is written as-is from the NC (typically 10000 m).
    * Manning coefficients are not stored in the NC; uniform defaults are used.

    Parameters
    ----------
    nc_path : path-like
        Input ``parameters.nc`` file.
    out_dir : path-like
        Output directory (created if needed).
    crop_to_bbox : bool
        If True, shrink the output grid to the bounding box of the
        catchment subset (default False for backward compatibility).
    river_manning_default : float
        Manning coefficient written to ``rivman.bin`` (default 0.03).
    flood_manning_default : float
        Floodplain Manning coefficient written to ``fldman.bin`` (default 0.1).
        This file is optional in CaMa-Flood; it is only written when the
        value differs from the hardcoded Fortran default.
    missing_float / missing_int : float / int
        Fill values for ocean/invalid cells.

    Returns
    -------
    (out_dir, out_nx, out_ny, nlfp, west, east, north, south) : tuple
        Output directory, grid dimensions, and geographic extent written.
    """
    nc_path = Path(nc_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Exporting {nc_path.name} → CaMa-Flood binary files ===")
    print(f"    Output directory: {out_dir}")

    with Dataset(str(nc_path), "r") as ds:
        # --- Grid dimensions (full, from NC) ---
        full_nx = int(ds.getncattr("nx"))
        full_ny = int(ds.getncattr("ny"))

        # --- Core 1-D catchment arrays ---
        catchment_id = ds.variables["catchment_id"][:].astype(np.int64)
        downstream_id = ds.variables["downstream_id"][:].astype(np.int64)
        catchment_x = ds.variables["catchment_x"][:].astype(np.int64)
        catchment_y = ds.variables["catchment_y"][:].astype(np.int64)
        n_catch = len(catchment_id)

        # Flood levels
        flood_depth_table = ds.variables["flood_depth_table"][:]
        nlfp = flood_depth_table.shape[1]

        # --- Determine output grid (full or cropped) ---
        if crop_to_bbox and n_catch < full_nx * full_ny:
            (x_min, x_max, y_min, y_max,
             out_nx, out_ny,
             out_west, out_east, out_north, out_south,
             ) = _compute_bbox(catchment_x, catchment_y, full_nx, full_ny)
            # Offset coordinates to the cropped grid
            cx = catchment_x - x_min
            cy = catchment_y - y_min
            print(f"    Full grid: {full_nx}×{full_ny}  →  "
                  f"Cropped bbox: x=[{x_min},{x_max}] y=[{y_min},{y_max}]  "
                  f"→  {out_nx}×{out_ny}")
            print(f"    Geographic extent: W={out_west:.3f} E={out_east:.3f} "
                  f"N={out_north:.3f} S={out_south:.3f}")
        else:
            out_nx, out_ny = full_nx, full_ny
            cx, cy = catchment_x, catchment_y
            x_min, y_min = 0, 0
            out_west, out_east = -180.0, 180.0
            out_north, out_south = 90.0, -90.0

        print(f"    Output grid: {out_nx}×{out_ny},  catchments: {n_catch},  "
              f"flood levels: {nlfp}")

        # ---- nextxy.bin ----
        nextxy = _rebuild_nextxy(
            out_nx, out_ny, catchment_id, downstream_id,
            cx, cy,
            missing_int=missing_int,
        )
        _write_bin(out_dir / "nextxy.bin", nextxy)

        # ---- Standard 2-D float maps (including nxtdst) ----
        _2d_fields = {
            "ctmare.bin": "catchment_area",
            "elevtn.bin": "catchment_elevation",
            "rivlen.bin": "river_length",
            "nxtdst.bin": "downstream_distance",
        }
        for fname, var_name in _2d_fields.items():
            if var_name in ds.variables:
                vals = ds.variables[var_name][:].astype("<f4")
                grid = _scatter_1d_to_2d(out_nx, out_ny, cx, cy, vals,
                                         fill=missing_float, dtype="<f4")
                _write_bin(out_dir / fname, grid)

        # ---- Optional 2-D float maps ----
        _optional_2d = {
            "rivwth_gwdlr.bin": "river_width",
            "rivhgt.bin": "river_height",
            "width.bin": "satellite_width",
        }
        for fname, var_name in _optional_2d.items():
            if var_name in ds.variables:
                vals = ds.variables[var_name][:].astype("<f4")
                grid = _scatter_1d_to_2d(out_nx, out_ny, cx, cy, vals,
                                         fill=missing_float, dtype="<f4")
                _write_bin(out_dir / fname, grid)

        # ---- rivman.bin (uniform Manning coefficient) ----
        man_vals = np.full(n_catch, river_manning_default, dtype="<f4")
        grid = _scatter_1d_to_2d(out_nx, out_ny, cx, cy, man_vals,
                                 fill=missing_float, dtype="<f4")
        _write_bin(out_dir / "rivman.bin", grid)

        # ---- fldhgt.bin (3-D) ----
        fld_grid = _scatter_1d_to_3d(
            out_nx, out_ny, nlfp, cx, cy,
            flood_depth_table.astype("<f4"),
            fill=missing_float, dtype="<f4",
        )
        _write_bin(out_dir / "fldhgt.bin", fld_grid)

        # ---- lonlat.bin ----
        if "longitude" in ds.variables and "latitude" in ds.variables:
            lon = ds.variables["longitude"][:].astype("<f4")
            lat = ds.variables["latitude"][:].astype("<f4")
            lonlat = np.full((out_nx, out_ny, 2), missing_float, dtype="<f4")
            lonlat[cx, cy, 0] = lon
            lonlat[cx, cy, 1] = lat
            _write_bin(out_dir / "lonlat.bin", lonlat)

        # ---- levhgt.bin / levfrc.bin (levee parameters) ----
        if "levee_crown_height" in ds.variables and "levee_fraction" in ds.variables:
            lev_hgt = ds.variables["levee_crown_height"][:].astype("<f4")
            lev_frc = ds.variables["levee_fraction"][:].astype("<f4")
            lev_cx = ds.variables["levee_catchment_x"][:].astype(np.int64) - x_min
            lev_cy = ds.variables["levee_catchment_y"][:].astype(np.int64) - y_min
            # Scatter onto 2-D grid (non-levee cells get fill=0)
            grid_hgt = _scatter_1d_to_2d(out_nx, out_ny, lev_cx, lev_cy, lev_hgt,
                                         fill=0.0, dtype="<f4")
            grid_frc = _scatter_1d_to_2d(out_nx, out_ny, lev_cx, lev_cy, lev_frc,
                                         fill=-1.0, dtype="<f4")
            _write_bin(out_dir / "levhgt.bin", grid_hgt)
            _write_bin(out_dir / "levfrc.bin", grid_frc)

        # ---- bifprm.txt ----
        if "bifurcation_catchment_x" in ds.variables:
            bif_cx = ds.variables["bifurcation_catchment_x"][:].astype(np.int64) - x_min
            bif_cy = ds.variables["bifurcation_catchment_y"][:].astype(np.int64) - y_min
            bif_dx = ds.variables["bifurcation_downstream_x"][:].astype(np.int64) - x_min
            bif_dy = ds.variables["bifurcation_downstream_y"][:].astype(np.int64) - y_min
            bif_len = ds.variables["bifurcation_length"][:].astype("<f4")
            bif_elv = ds.variables["bifurcation_elevation"][:]
            bif_wth = ds.variables["bifurcation_width"][:]

            lon_1d = ds.variables["longitude"][:] if "longitude" in ds.variables else None
            lat_1d = ds.variables["latitude"][:] if "latitude" in ds.variables else None
            mouth_id = (
                ds.variables["river_mouth_id"][:].astype(np.int64)
                if "river_mouth_id" in ds.variables
                else None
            )

            _write_bifprm(
                out_dir / "bifprm.txt",
                bif_cx, bif_cy, bif_dx, bif_dy,
                bif_len, bif_elv, bif_wth,
                longitude=lon_1d, latitude=lat_1d,
                catchment_x=cx, catchment_y=cy,
                catchment_id=catchment_id,
                river_mouth_id=mouth_id,
                catchment_elevation=(
                    ds.variables["catchment_elevation"][:].data
                    if "catchment_elevation" in ds.variables else None
                ),
                missing_int=missing_int,
            )

        # ---- dam_params.csv (reservoir parameters) ----
        if "reservoir_catchment_id" in ds.variables:
            res_cid = ds.variables["reservoir_catchment_id"][:].astype(np.int64)
            res_cap = ds.variables["reservoir_capacity"][:].astype("<f4")
            res_con = ds.variables["conservation_volume"][:].astype("<f4")
            res_eme = ds.variables["emergency_volume"][:].astype("<f4")
            res_qn = ds.variables["normal_outflow"][:].astype("<f4")
            res_qf = ds.variables["flood_control_outflow"][:].astype("<f4")

            up_area = (
                ds.variables["upstream_area"][:].astype("<f4")
                if "upstream_area" in ds.variables else None
            )
            lon_1d = (
                ds.variables["longitude"][:].astype("<f4")
                if "longitude" in ds.variables else None
            )
            lat_1d = (
                ds.variables["latitude"][:].astype("<f4")
                if "latitude" in ds.variables else None
            )

            _write_dam_params_csv(
                out_dir / "dam_params.csv",
                reservoir_catchment_id=res_cid,
                reservoir_capacity=res_cap,
                conservation_volume=res_con,
                emergency_volume=res_eme,
                normal_outflow=res_qn,
                flood_control_outflow=res_qf,
                catchment_id=catchment_id,
                full_ny=full_ny,
                x_min=x_min,
                y_min=y_min,
                out_nx=out_nx,
                out_ny=out_ny,
                longitude=lon_1d,
                latitude=lat_1d,
                upstream_area=up_area,
            )

        # ---- mapdim.txt ----
        _write_mapdim(out_dir / "mapdim.txt", out_nx, out_ny, nlfp)

        # ---- crop_info.txt (only when cropped) ----
        if crop_to_bbox and (out_nx != full_nx or out_ny != full_ny):
            _write_crop_info(
                out_dir / "crop_info.txt",
                full_nx, full_ny, x_min, y_min, out_nx, out_ny,
            )

    print(f"=== Map export complete → {out_dir} ===\n")
    return out_dir, out_nx, out_ny, nlfp, out_west, out_east, out_north, out_south


# ===========================================================================
# Main: export NPZ → inpmat binary + diminfo
# ===========================================================================

def export_inpmat(
    npz_path: Union[str, Path],
    nc_path: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    inpmat_name: str = "inpmat.bin",
    diminfo_name: str = "diminfo.txt",
    inpn: int = 24,
    out_nx: Optional[int] = None,
    out_ny: Optional[int] = None,
    nlfp: Optional[int] = None,
    west: float = -180.0,
    east: float = 180.0,
    north: float = 90.0,
    south: float = -90.0,
    crop_to_bbox: bool = False,
) -> Path:
    """Convert a CaMa-Flood-GPU runoff mapping ``.npz`` to a CaMa-Flood ``inpmat*.bin``.

    The ``inpmat`` binary is Fortran DIRECT ACCESS (``RECL = 4*NX*NY``) with
    ``3 * INPN`` records:

    * Records  1 … INPN      : ``INPX`` – int32 x-indices into the runoff grid (1-based)
    * Records  INPN+1 … 2·INPN : ``INPY`` – int32 y-indices into the runoff grid (1-based)
    * Records  2·INPN+1 … 3·INPN : ``INPA`` – float32 area weights

    A corresponding ``diminfo*.txt`` is also written.

    Parameters
    ----------
    npz_path : path-like
        Input ``.npz`` file produced by ``generate_runoff_mapping_table()``.
    nc_path : path-like
        ``parameters.nc`` (used for grid dimensions and catchment coordinates).
    out_dir : path-like
        Output directory.
    inpmat_name : str
        Output binary file name.
    diminfo_name : str
        Output diminfo text file name.
    inpn : int
        Maximum number of contributing runoff-grid cells per map cell
        (must match CaMa-Flood's ``diminfo``; default 24).
    out_nx, out_ny : int, optional
        Output grid dimensions.  If provided (e.g. from a prior
        ``export_map_params`` call with ``crop_to_bbox=True``), the inpmat
        uses this grid size.  Otherwise read from the NC.
    nlfp : int, optional
        Number of flood levels.  If not provided, read from NC.
    west, east, north, south : float
        Geographic extent for ``diminfo`` header.
    crop_to_bbox : bool
        If True and ``out_nx``/``out_ny`` are not given, compute the
        bounding box from NC catchment coordinates automatically.

    Returns
    -------
    Path
        Path to the written ``inpmat`` binary.
    """
    npz_path = Path(npz_path)
    nc_path = Path(nc_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Exporting {npz_path.name} → CaMa-Flood inpmat binary ===")

    # --- Load NPZ ---
    mapping_data = np.load(str(npz_path))
    npz_catchment_ids = mapping_data["catchment_ids"].astype(np.int64)
    sparse_data = mapping_data["sparse_data"]
    sparse_indices = mapping_data["sparse_indices"]
    sparse_indptr = mapping_data["sparse_indptr"]
    matrix_shape = tuple(mapping_data["matrix_shape"])
    coord_lon = mapping_data["coord_lon"]
    coord_lat = mapping_data["coord_lat"]

    nxin = len(coord_lon)
    nyin = len(coord_lat)

    print(f"    NPZ: {matrix_shape[0]} catchments, grid {nxin}×{nyin}")

    # Rebuild the sparse matrix
    full_sparse = csr_matrix(
        (sparse_data, sparse_indices, sparse_indptr),
        shape=matrix_shape,
    )

    # --- Load NC to get grid dims + catchment positions ---
    with Dataset(str(nc_path), "r") as ds:
        full_nx = int(ds.getncattr("nx"))
        full_ny = int(ds.getncattr("ny"))
        nc_cid = ds.variables["catchment_id"][:].astype(np.int64)
        nc_cx = ds.variables["catchment_x"][:].astype(np.int64)
        nc_cy = ds.variables["catchment_y"][:].astype(np.int64)
        if nlfp is None:
            nlfp = ds.variables["flood_depth_table"].shape[1]

    # --- Decide output grid dimensions ---
    x_min = y_min = 0
    if out_nx is not None and out_ny is not None:
        # Caller supplied cropped dimensions (e.g. from export_map_params)
        # We still need x_min/y_min to offset catchment coordinates
        if out_nx != full_nx or out_ny != full_ny:
            bbox = _compute_bbox(nc_cx, nc_cy, full_nx, full_ny)
            x_min = bbox[0]  # x_min
            y_min = bbox[2]  # y_min
    elif crop_to_bbox:
        bbox = _compute_bbox(nc_cx, nc_cy, full_nx, full_ny)
        x_min = bbox[0]
        y_min = bbox[2]
        out_nx = bbox[4]  # crop_nx
        out_ny = bbox[5]  # crop_ny
        west = bbox[6]
        east = bbox[7]
        north = bbox[8]
        south = bbox[9]
    else:
        out_nx, out_ny = full_nx, full_ny

    print(f"    Map grid: {out_nx}×{out_ny},  INPN={inpn}")

    # Build npz_catchment_id → row-index in sparse matrix
    npz_cid_to_row = {int(cid): r for r, cid in enumerate(npz_catchment_ids)}

    # Build catchment_id → (cx, cy) in output grid coordinates
    nc_cid_to_xy = {}
    for cid, x, y in zip(nc_cid, nc_cx, nc_cy):
        cx = int(x) - x_min
        cy = int(y) - y_min
        nc_cid_to_xy[int(cid)] = (cx, cy)

    # --- Allocate output arrays (out_nx, out_ny, INPN) ---
    INPX = np.zeros((out_nx, out_ny, inpn), dtype="<i4")
    INPY = np.zeros((out_nx, out_ny, inpn), dtype="<i4")
    INPA = np.zeros((out_nx, out_ny, inpn), dtype="<f4")

    truncated = 0  # count of cells where entries exceeded INPN

    # --- Populate per-cell mapping ---
    for cid in npz_catchment_ids:
        cid_int = int(cid)
        if cid_int not in nc_cid_to_xy:
            continue  # catchment not in this NC (e.g. different crop)
        row = npz_cid_to_row[cid_int]
        ix, iy = nc_cid_to_xy[cid_int]

        if ix < 0 or ix >= out_nx or iy < 0 or iy >= out_ny:
            continue  # outside cropped grid

        # Get non-zero entries for this catchment row
        start, end = full_sparse.indptr[row], full_sparse.indptr[row + 1]
        if start == end:
            continue

        col_indices = full_sparse.indices[start:end]
        area_weights = full_sparse.data[start:end]

        # Sort by descending weight so that truncation drops the smallest
        order = np.argsort(-area_weights)
        col_indices = col_indices[order]
        area_weights = area_weights[order]

        n_entries = len(col_indices)
        if n_entries > inpn:
            truncated += 1
            col_indices = col_indices[:inpn]
            area_weights = area_weights[:inpn]
            n_entries = inpn

        # Convert flattened column index → (ixin, iyin)
        # Column convention: col = iyin * nxin + ixin
        # NOTE: INPX/INPY are 1-based indices into the RUNOFF grid
        #       (NXIN × NYIN), NOT the map grid — they stay unchanged.
        ixin = (col_indices % nxin).astype("i4")
        iyin = (col_indices // nxin).astype("i4")

        # Store 1-based indices
        INPX[ix, iy, :n_entries] = ixin + 1
        INPY[ix, iy, :n_entries] = iyin + 1
        INPA[ix, iy, :n_entries] = area_weights.astype("<f4")

    if truncated > 0:
        print(f"    WARNING: {truncated} cells had > {inpn} contributing sources;"
              f" smallest weights were dropped.")

    # --- Write inpmat binary (Fortran DIRECT ACCESS, RECL = 4*out_nx*out_ny) ---
    inpmat_path = out_dir / inpmat_name
    record_bytes = 4 * out_nx * out_ny

    with open(inpmat_path, "wb") as f:
        for n in range(inpn):
            INPX[:, :, n].flatten(order="F").tofile(f)
        for n in range(inpn):
            INPY[:, :, n].flatten(order="F").tofile(f)
        for n in range(inpn):
            INPA[:, :, n].flatten(order="F").tofile(f)

    expected_size = 3 * inpn * record_bytes
    actual_size = inpmat_path.stat().st_size
    assert actual_size == expected_size, (
        f"inpmat size mismatch: expected {expected_size}, got {actual_size}"
    )
    print(f"  Wrote {inpmat_name}  ({actual_size / 1e6:.1f} MB,  "
          f"{3 * inpn} records of {record_bytes} bytes)")

    # --- Write diminfo ---
    diminfo_path = out_dir / diminfo_name
    _write_diminfo(
        diminfo_path, out_nx, out_ny, nlfp,
        nxin, nyin, inpn, inpmat_name,
        west, east, north, south,
    )

    print(f"=== inpmat export complete → {out_dir} ===\n")
    return inpmat_path


# ===========================================================================
# Convenience wrapper
# ===========================================================================

def export_to_cama_bin(
    nc_path: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    npz_path: Optional[Union[str, Path]] = None,
    river_manning_default: float = 0.03,
    flood_manning_default: float = 0.1,
    inpmat_name: str = "inpmat.bin",
    diminfo_name: str = "diminfo.txt",
    inpn: int = 24,
    west: float = -180.0,
    east: float = 180.0,
    north: float = 90.0,
    south: float = -90.0,
    crop_to_bbox: bool = False,
) -> Path:
    """One-shot export: ``parameters.nc`` (+ optional ``.npz``) → CaMa-Flood binary files.

    Parameters
    ----------
    nc_path : path-like
        CaMa-Flood-GPU ``parameters.nc``.
    out_dir : path-like
        Target directory for all output binary files.
    npz_path : path-like, optional
        Runoff mapping ``.npz``; if given, ``inpmat*.bin`` and ``diminfo*.txt``
        are also generated.
    river_manning_default : float
        River Manning coefficient (default 0.03).
    flood_manning_default : float
        Floodplain Manning coefficient (default 0.1).
    inpmat_name / diminfo_name : str
        File names for the input-matrix binary and dimension-info text.
    inpn : int
        Max contributing sources per cell for ``inpmat`` (default 24).
    west, east, north, south : float
        Geographic extent written into ``diminfo``.
    crop_to_bbox : bool
        If ``True``, shrink the output grid to the bounding box of
        catchments present in the NC, producing much smaller files for
        cropped / POI-filtered datasets.

    Returns
    -------
    Path
        *out_dir* path.
    """
    out_dir = Path(out_dir)

    result = export_map_params(
        nc_path, out_dir,
        river_manning_default=river_manning_default,
        flood_manning_default=flood_manning_default,
        crop_to_bbox=crop_to_bbox,
    )
    _, out_nx, out_ny, nlfp, c_west, c_east, c_north, c_south = result

    if npz_path is not None:
        export_inpmat(
            npz_path, nc_path, out_dir,
            inpmat_name=inpmat_name,
            diminfo_name=diminfo_name,
            inpn=inpn,
            out_nx=out_nx,
            out_ny=out_ny,
            nlfp=nlfp,
            west=c_west, east=c_east,
            north=c_north, south=c_south,
            crop_to_bbox=crop_to_bbox,
        )

    return out_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export CaMa-Flood-GPU NetCDF parameters to CaMa-Flood v4 binary format."
    )
    parser.add_argument("nc_path", help="Path to parameters.nc")
    parser.add_argument("out_dir", help="Output directory for binary files")
    parser.add_argument("--npz", default=None, help="Path to runoff_mapping.npz (optional)")
    parser.add_argument("--river-manning", type=float, default=0.03,
                        help="River Manning coefficient (default 0.03)")
    parser.add_argument("--flood-manning", type=float, default=0.1,
                        help="Floodplain Manning coefficient (default 0.1)")
    parser.add_argument("--inpn", type=int, default=24, help="Max input sources per cell")
    parser.add_argument("--inpmat-name", default="inpmat.bin", help="inpmat output filename")
    parser.add_argument("--diminfo-name", default="diminfo.txt", help="diminfo output filename")
    parser.add_argument("--west", type=float, default=-180.0)
    parser.add_argument("--east", type=float, default=180.0)
    parser.add_argument("--north", type=float, default=90.0)
    parser.add_argument("--south", type=float, default=-90.0)
    parser.add_argument(
        "--crop-to-bbox", action="store_true",
        help="Shrink output grid to the bounding box of catchments for minimal file size.",
    )

    args = parser.parse_args()

    export_to_cama_bin(
        nc_path=args.nc_path,
        out_dir=args.out_dir,
        npz_path=args.npz,
        river_manning_default=args.river_manning,
        flood_manning_default=args.flood_manning,
        inpmat_name=args.inpmat_name,
        diminfo_name=args.diminfo_name,
        inpn=args.inpn,
        west=args.west,
        east=args.east,
        north=args.north,
        south=args.south,
        crop_to_bbox=args.crop_to_bbox,
    )
