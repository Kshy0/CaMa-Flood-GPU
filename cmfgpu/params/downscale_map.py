# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""High-resolution downscaling parameter builder from CaMa-Flood binary maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from hydroforge.io import (
    BOOL_LOGICAL_DTYPE,
    LOGICAL_DTYPE_ATTR,
    atomic_netcdf_dataset,
)
from pydantic import BaseModel, ConfigDict, DirectoryPath, Field, FilePath

_HIRES_CELL_SIZES: dict[str, float] = {
    "1sec": 1.0 / 3600.0,
    "3sec": 1.0 / 1200.0,
    "5sec": 1.0 / 720.0,
    "15sec": 1.0 / 240.0,
    "30sec": 1.0 / 120.0,
    "1min": 1.0 / 60.0,
}

# MERIT D8 flow-direction codes (1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW)
_FLWDIR_DX = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int16)
_FLWDIR_DY = np.array([0, -1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int16)


def _convert_flwdir_to_downxy(flwdir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert 1-byte MERIT flow direction codes into Fortran (downX, downY) offsets."""
    downx = np.full(flwdir.shape, -9999, dtype=np.int16)
    downy = np.full(flwdir.shape, -9999, dtype=np.int16)
    d8 = (flwdir >= 1) & (flwdir <= 8)
    codes = flwdir[d8].astype(np.int32)
    downx[d8] = _FLWDIR_DX[codes]
    downy[d8] = _FLWDIR_DY[codes]
    mouth = flwdir == 0
    downx[mouth] = -999
    downy[mouth] = -999
    inland = flwdir == -1
    downx[inland] = -1000
    downy[inland] = -1000
    return downx, downy


def _build_tributary_and_backwater_tables(
    catm_xx: np.ndarray,
    catm_yy: np.ndarray,
    down_x: np.ndarray,
    down_y: np.ndarray,
    flddif: np.ndarray,
    next_xx: np.ndarray,
    next_yy: np.ndarray,
    uparea: np.ndarray,
    pixel_flat_to_compact: np.ndarray,
) -> dict[str, np.ndarray]:
    """Precompute exact static topology matching ``downscale_flddph_trib.F90``.

    Arrays ``catm_xx``, ``catm_yy``, ``down_x``, ``down_y``, ``flddif`` have
    Fortran-aligned shape ``(mx, my)`` over the buffered window.
    ``pixel_flat_to_compact`` maps flat ``ix * my + iy`` (or ``-1``) to the
    0-based index in ``pixel_id`` (and stores ``pixel_id`` lookup).
    """
    mx, my = catm_xx.shape
    nxx, nyy = next_xx.shape

    # 1. Compute max upstream drainage area per coarse cell (lines 405-423)
    uparea_scaled = uparea.astype(np.float32, copy=True)
    valid_coarse = next_xx != -9999
    uparea_scaled[valid_coarse] = uparea_scaled[valid_coarse] * np.float32(1.0e-6)

    maxupa = np.zeros((nxx, nyy), dtype=np.float32)
    src_x, src_y = np.where(next_xx > 0)
    dst_x = next_xx[src_x, src_y] - 1
    dst_y = next_yy[src_x, src_y] - 1
    np.maximum.at(maxupa, (dst_x, dst_y), uparea_scaled[src_x, src_y])

    # 2. Mark outlet (6), mainstem (3), tributary (1), out-of-domain (-1) (lines 613-649)
    check = np.zeros((mx, my), dtype=np.int8)
    for iy in range(my):
        for ix in range(mx):
            ixx = int(catm_xx[ix, iy])
            if ixx <= 0:
                continue
            iyy = int(catm_yy[ix, iy])
            jxx = int(next_xx[ixx - 1, iyy - 1])
            if jxx <= 0:
                continue
            jyy = int(next_yy[ixx - 1, iyy - 1])
            dx_step = int(down_x[ix, iy])
            if dx_step < -900:
                continue
            jx = ix + dx_step
            jy = iy + int(down_y[ix, iy])
            if jx < 0 or jx >= mx or jy < 0 or jy >= my:
                continue
            if int(catm_xx[jx, jy]) == jxx and int(catm_yy[jx, jy]) == jyy:
                check[ix, iy] = 6
                val = (
                    3
                    if uparea_scaled[ixx - 1, iyy - 1] == maxupa[jxx - 1, jyy - 1]
                    else 1
                )
                while int(catm_xx[jx, jy]) == jxx and int(catm_yy[jx, jy]) == jyy:
                    if val > check[jx, jy]:
                        check[jx, jy] = val
                    d_x = int(down_x[jx, jy])
                    if d_x < -900:
                        break
                    kx = jx + d_x
                    ky = jy + int(down_y[jx, jy])
                    if kx < 0 or kx >= mx or ky < 0 or ky >= my:
                        check[jx, jy] = -1
                        break
                    jx, jy = kx, ky

    # 3. Extract tributary interpolation links (lines 653-705)
    trib_target_flat: list[int] = []
    trib_up_flat: list[int] = []
    trib_dn_flat: list[int] = []
    trib_steps: list[float] = []
    trib_lengths: list[float] = []
    potentially_modified = np.zeros((nxx, nyy), dtype=bool)

    for iy in range(my):
        for ix in range(mx):
            ixx = int(catm_xx[ix, iy])
            if ixx <= 0:
                continue
            iyy = int(catm_yy[ix, iy])
            jxx = int(next_xx[ixx - 1, iyy - 1])
            if jxx <= 0:
                continue
            jyy = int(next_yy[ixx - 1, iyy - 1])
            dx_step = int(down_x[ix, iy])
            if dx_step < -900:
                continue
            jx = ix + dx_step
            jy = iy + int(down_y[ix, iy])
            if jx < 0 or jx >= mx or jy < 0 or jy >= my:
                continue
            if int(catm_xx[jx, jy]) == jxx and int(catm_yy[jx, jy]) == jyy:
                if check[jx, jy] > 1 or check[jx, jy] != 1:
                    continue
                # Trace tributary until mainstem / outlet
                chain_len = 1
                tx, ty = jx, jy
                while check[tx, ty] == 1:
                    d_x = int(down_x[tx, ty])
                    if d_x < -900:
                        break
                    d_y = int(down_y[tx, ty])
                    tx = tx + d_x
                    ty = ty + d_y
                    chain_len += 1
                if check[tx, ty] == -1 or int(catm_xx[tx, ty]) <= 0:
                    continue

                up_flat = iy * mx + ix
                dn_flat = ty * mx + tx
                step_idx = 1
                cx, cy = jx, jy
                potentially_modified[jxx - 1, jyy - 1] = True
                while check[cx, cy] == 1:
                    trib_target_flat.append(cy * mx + cx)
                    trib_up_flat.append(up_flat)
                    trib_dn_flat.append(dn_flat)
                    trib_steps.append(float(step_idx))
                    trib_lengths.append(float(chain_len))
                    d_x = int(down_x[cx, cy])
                    if d_x < -900:
                        break
                    nx_c = cx + d_x
                    ny_c = cy + int(down_y[cx, cy])
                    cx, cy = nx_c, ny_c
                    step_idx += 1

    # 4. Simulate static backwater segment traversal in exact (iy, ix) order (lines 708-754)
    check_bw = check.copy()
    # stop_parent[flat] stores immediate segment stop flat index (-1 if none)
    stop_parent = np.full(mx * my, -1, dtype=np.int64)

    for iy in range(my):
        for ix in range(mx):
            ixx = int(catm_xx[ix, iy])
            if ixx <= 0:
                continue
            iyy = int(catm_yy[ix, iy])
            if not potentially_modified[ixx - 1, iyy - 1]:
                continue
            if check_bw[ix, iy] != 0:
                continue
            dx_step = int(down_x[ix, iy])
            if dx_step < -900:
                continue
            jx = ix + dx_step
            jy = iy + int(down_y[ix, iy])
            if jx < 0 or jx >= mx or jy < 0 or jy >= my:
                continue

            while check_bw[jx, jy] == 0:
                d_x = int(down_x[jx, jy])
                if d_x < -900:
                    break
                kx = jx + d_x
                ky = jy + int(down_y[jx, jy])
                if kx < 0 or kx >= mx or ky < 0 or ky >= my:
                    check_bw[jx, jy] = -1
                    break
                jx, jy = kx, ky

            if check_bw[jx, jy] == -1:
                # Matches Fortran `if( check(jx,jy)==-1 ) exit` (exits ix loop)
                break

            stop_flat = jy * mx + jx
            cx, cy = ix, iy
            while check_bw[cx, cy] == 0:
                c_flat = cy * mx + cx
                if c_flat != stop_flat and int(catm_xx[jx, jy]) > 0:
                    stop_parent[c_flat] = stop_flat
                check_bw[cx, cy] = 10
                d_x = int(down_x[cx, cy])
                if d_x < -900:
                    break
                kx = cx + d_x
                ky = cy + int(down_y[cx, cy])
                if kx < 0 or kx >= mx or ky < 0 or ky >= my:
                    check_bw[cx, cy] = -1
                    break
                cx, cy = kx, ky

    # Path-compress stop_parent chains to terminal pixel & max intermediate flddif
    bw_candidates = np.flatnonzero(stop_parent >= 0)
    bw_pix_ids: list[int] = []
    bw_target_ids: list[int] = []
    bw_floors: list[float] = []
    flddif_row_major = flddif.T.reshape(-1)

    for flat_p in bw_candidates.tolist():
        if pixel_flat_to_compact[flat_p] < 0:
            continue
        cur = int(stop_parent[flat_p])
        floor_elv = -1.0e20
        while stop_parent[cur] >= 0:
            val = float(flddif_row_major[cur])
            if val > floor_elv:
                floor_elv = val
            cur = int(stop_parent[cur])
        if pixel_flat_to_compact[cur] < 0 or cur == flat_p:
            continue
        bw_pix_ids.append(flat_p)
        bw_target_ids.append(cur)
        bw_floors.append(floor_elv)

    trib_target_arr = np.asarray(trib_target_flat, dtype=np.int64)
    unique_trib_pixels = np.unique(trib_target_arr)

    return {
        "tributary_pixel_id": unique_trib_pixels,
        "tributary_link_id": np.arange(len(trib_target_flat), dtype=np.int64),
        "tributary_target_id": trib_target_arr,
        "tributary_upstream_pixel_id": np.asarray(trib_up_flat, dtype=np.int64),
        "tributary_downstream_pixel_id": np.asarray(trib_dn_flat, dtype=np.int64),
        "tributary_chain_step": np.asarray(trib_steps, dtype=np.float32),
        "tributary_chain_length": np.asarray(trib_lengths, dtype=np.float32),
        "backwater_pixel_id": np.asarray(bw_pix_ids, dtype=np.int64),
        "backwater_target_pixel_id": np.asarray(bw_target_ids, dtype=np.int64),
        "backwater_floor_elevation": np.asarray(bw_floors, dtype=np.float32),
    }


def scatter_pixels_to_raster(
    pixel_id: np.ndarray,
    pixel_values: np.ndarray,
    *,
    mx: int,
    my: int,
    dx: int = 0,
    dy: int = 0,
    lx: int | None = None,
    ly: int | None = None,
    undef_value: float = -9999.0,
) -> np.ndarray:
    """Scatter compact 1D pixel values back to Fortran ``(mx, my)`` or cropped ``(lx, ly)`` raster.

    ``pixel_id`` encodes row-major ``jy * mx + jx`` over the ``(mx, my)`` grid,
    returning array of shape ``(lx, ly)`` matching Fortran direct-access binary layout
    ``flood(jx, jy)``.
    """
    values = np.asarray(pixel_values, dtype=np.float32)
    ids = np.asarray(pixel_id, dtype=np.int64)
    jx = ids % mx
    jy = ids // mx

    if values.ndim == 1:
        raster = np.full((mx, my), undef_value, dtype=np.float32)
        raster[jx, jy] = values
        if lx is not None and ly is not None:
            return raster[dx : dx + lx, dy : dy + ly].copy()
        return raster

    batch = values.shape[0]
    raster = np.full((batch, mx, my), undef_value, dtype=np.float32)
    raster[:, jx, jy] = values
    if lx is not None and ly is not None:
        return raster[:, dx : dx + lx, dy : dy + ly].copy()
    return raster


class DownscaleMap(BaseModel):
    """Build NetCDF or in-memory parameter dictionaries for :class:`CaMaDownscale`.

    Supports all four original Fortran CaMa-Flood downscaling modes:
    standard depth (``downscale_flddph.F90``), levee-aware depth
    (``define_protect_pix.F90`` + ``downscale_flddph_levee.F90``), sub-grid
    tributary & backwater smoothing (``downscale_flddph_trib.F90``), and
    inundation duration (``downscale_fldprd.F90``).
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=False, extra="allow"
    )

    map_dir: DirectoryPath = Field(
        description="Coarse CaMa-Flood map directory containing params.txt, nextxy.bin, and hires subfolder"
    )
    hires_tag: str = Field(
        default="1min",
        description="High-resolution subdirectory tag (1min, 30sec, 15sec, 5sec, 3sec, 1sec)",
    )
    west: float = Field(description="Target downscaling western longitude bound")
    east: float = Field(description="Target downscaling eastern longitude bound")
    south: float = Field(description="Target downscaling southern latitude bound")
    north: float = Field(description="Target downscaling northern latitude bound")

    levee_file: FilePath | None = Field(
        default=None,
        description="Optional coarse levfrc.bin path enabling levee-protected pixel classification",
    )
    include_tributary: bool = Field(
        default=False,
        description="Precompute sub-grid D8 tributary and backwater topology with 5-cell buffer",
    )
    tributary_buffer_cells: int = Field(
        default=5,
        ge=0,
        description="Coarse-grid buffer width around target domain when include_tributary is enabled",
    )
    num_duration_bins: int | None = Field(
        default=None,
        ge=1,
        description="Optional number of depth bins to include duration_bin_id for inundation-duration mode",
    )
    coarse_params_nc: FilePath | None = Field(
        default=None,
        description="Optional MERITMap parameters.nc to inherit catchment_basin_id from",
    )
    out_nc: Path | None = Field(
        default=None,
        description="Optional output NetCDF file path for persisting downscaling parameters",
    )

    missing_value: ClassVar[float] = -9999.0

    def build(self) -> dict[str, Any]:
        """Load coarse and high-resolution map tiles and return tensors/arrays + domain metadata."""
        if self.hires_tag not in _HIRES_CELL_SIZES:
            raise ValueError(
                f"Unsupported hires_tag {self.hires_tag!r}; expected one of {sorted(_HIRES_CELL_SIZES)}"
            )
        csize = _HIRES_CELL_SIZES[self.hires_tag]

        with open(self.map_dir / "params.txt") as f:
            lines = [line.split()[0] for line in f if line.strip()]
        nxx = int(lines[0])
        nyy = int(lines[1])
        gsize = float(lines[3])

        lx = int(round((self.east - self.west) / csize))
        ly = int(round((self.north - self.south) / csize))

        if self.include_tributary and self.tributary_buffer_cells > 0:
            buf_deg = gsize * self.tributary_buffer_cells
            west_buf = self.west - buf_deg
            east_buf = self.east + buf_deg
            south_buf = self.south - buf_deg
            north_buf = self.north + buf_deg
            mx = int(round((east_buf - west_buf) / csize))
            my = int(round((north_buf - south_buf) / csize))
            dx = int(gsize * self.tributary_buffer_cells / csize + 0.01)
            dy = int(gsize * self.tributary_buffer_cells / csize + 0.01)
        else:
            west_buf, east_buf, south_buf, north_buf = (
                self.west,
                self.east,
                self.south,
                self.north,
            )
            mx, my = lx, ly
            dx, dy = 0, 0

        nextxy = np.fromfile(self.map_dir / "nextxy.bin", dtype="<i4").reshape(
            (nxx, nyy, 2), order="F"
        )
        next_xx = nextxy[:, :, 0]
        next_yy = nextxy[:, :, 1]

        catm_xx = np.full((mx, my), -9999, dtype=np.int16)
        catm_yy = np.full((mx, my), -9999, dtype=np.int16)
        flddif = np.full((mx, my), -9999.0, dtype=np.float32)
        rivwth = np.full((mx, my), -9999.0, dtype=np.float32)
        catmz100 = (
            np.zeros((mx, my), dtype=np.int8) if self.levee_file is not None else None
        )
        down_x = (
            np.full((mx, my), -9999, dtype=np.int16)
            if self.include_tributary
            else None
        )
        down_y = (
            np.full((mx, my), -9999, dtype=np.int16)
            if self.include_tributary
            else None
        )

        hires_dir = self.map_dir / self.hires_tag
        with open(hires_dir / "location.txt") as loc_f:
            loc_lines = loc_f.readlines()[2:]

        for line in loc_lines:
            parts = line.split()
            if len(parts) < 8:
                continue
            area = parts[1]
            lon_ori = float(parts[2])
            lon_end = float(parts[3])
            lat_end = float(parts[4])
            lat_ori = float(parts[5])
            nx = int(parts[6])
            ny = int(parts[7])
            tile_csize = float(parts[8]) if len(parts) >= 9 else csize

            if (
                lon_end < west_buf
                or lon_ori > east_buf
                or lat_ori < south_buf
                or lat_end > north_buf
            ):
                continue

            lon_in = lon_ori + (np.arange(1, nx + 1, dtype=np.float64) - 0.5) * tile_csize
            lon_in = np.where(lon_in >= 180.0, lon_in - 360.0, lon_in)
            lon_in = np.where(lon_in < -180.0, lon_in + 360.0, lon_in)
            lat_in = lat_ori - (np.arange(1, ny + 1, dtype=np.float64) - 0.5) * tile_csize

            x_mask = (lon_in > west_buf) & (lon_in < east_buf)
            y_mask = (lat_in > south_buf) & (lat_in < north_buf)
            if not np.any(x_mask) or not np.any(y_mask):
                continue

            src_ix = np.flatnonzero(x_mask)
            src_iy = np.flatnonzero(y_mask)
            eps = 0.001 if self.include_tributary else 0.0
            dst_jx = ((lon_in[src_ix] - west_buf) / tile_csize + eps).astype(np.int64)
            dst_jy = ((north_buf - lat_in[src_iy]) / tile_csize + eps).astype(np.int64)

            in_bounds_x = (dst_jx >= 0) & (dst_jx < mx)
            in_bounds_y = (dst_jy >= 0) & (dst_jy < my)
            src_ix, dst_jx = src_ix[in_bounds_x], dst_jx[in_bounds_x]
            src_iy, dst_jy = src_iy[in_bounds_y], dst_jy[in_bounds_y]

            catm_tile = np.memmap(
                hires_dir / f"{area}.catmxy.bin",
                dtype="<i2",
                mode="r",
                shape=(nx, ny, 2),
                order="F",
            )
            catm_xx[np.ix_(dst_jx, dst_jy)] = catm_tile[np.ix_(src_ix, src_iy, [0])][
                :, :, 0
            ]
            catm_yy[np.ix_(dst_jx, dst_jy)] = catm_tile[np.ix_(src_ix, src_iy, [1])][
                :, :, 0
            ]
            del catm_tile

            flddif_tile = np.memmap(
                hires_dir / f"{area}.flddif.bin",
                dtype="<f4",
                mode="r",
                shape=(nx, ny),
                order="F",
            )
            flddif[np.ix_(dst_jx, dst_jy)] = flddif_tile[np.ix_(src_ix, src_iy)]
            del flddif_tile

            rivwth_tile = np.memmap(
                hires_dir / f"{area}.rivwth.bin",
                dtype="<f4",
                mode="r",
                shape=(nx, ny),
                order="F",
            )
            rivwth[np.ix_(dst_jx, dst_jy)] = rivwth_tile[np.ix_(src_ix, src_iy)]
            del rivwth_tile

            if catmz100 is not None:
                catmz_path = hires_dir / f"{area}.catmz100.bin"
                catmz_tile = np.memmap(
                    catmz_path, dtype="<i1", mode="r", shape=(nx, ny), order="F"
                )
                catmz100[np.ix_(dst_jx, dst_jy)] = catmz_tile[np.ix_(src_ix, src_iy)]
                del catmz_tile

            if down_x is not None and down_y is not None:
                downxy_path = hires_dir / f"{area}.downxy.bin"
                if downxy_path.exists():
                    down_tile = np.memmap(
                        downxy_path,
                        dtype="<i2",
                        mode="r",
                        shape=(nx, ny, 2),
                        order="F",
                    )
                    down_x[np.ix_(dst_jx, dst_jy)] = down_tile[
                        np.ix_(src_ix, src_iy, [0])
                    ][:, :, 0]
                    down_y[np.ix_(dst_jx, dst_jy)] = down_tile[
                        np.ix_(src_ix, src_iy, [1])
                    ][:, :, 0]
                    del down_tile
                else:
                    flw_tile = np.memmap(
                        hires_dir / f"{area}.flwdir.bin",
                        dtype="<i1",
                        mode="r",
                        shape=(nx, ny),
                        order="F",
                    )
                    dx_sub, dy_sub = _convert_flwdir_to_downxy(
                        np.asarray(flw_tile[np.ix_(src_ix, src_iy)])
                    )
                    down_x[np.ix_(dst_jx, dst_jy)] = dx_sub
                    down_y[np.ix_(dst_jx, dst_jy)] = dy_sub
                    del flw_tile

        # Order valid pixels in row-major (jy, jx) order so pixel_id = jy * mx + jx is sorted
        catm_xx_rm = catm_xx.T
        catm_yy_rm = catm_yy.T
        flddif_rm = flddif.T
        rivwth_rm = rivwth.T

        valid_rm = (catm_xx_rm > 0) & (next_xx[np.clip(catm_xx_rm - 1, 0, nxx - 1), np.clip(catm_yy_rm - 1, 0, nyy - 1)] != -9999)
        # Note: Fortran checks `if (catmXX(ix, iy) > 0)` directly:
        valid_rm = catm_xx_rm > 0
        pixel_id = np.flatnonzero(valid_rm).astype(np.int64)

        pix_cxx = catm_xx_rm.reshape(-1)[pixel_id].astype(np.int64) - 1
        pix_cyy = catm_yy_rm.reshape(-1)[pixel_id].astype(np.int64) - 1
        pixel_catchment_id = np.ravel_multi_index((pix_cxx, pix_cyy), (nxx, nyy)).astype(
            np.int64
        )
        pixel_flood_diff = flddif_rm.reshape(-1)[pixel_id].astype(np.float32)
        pix_rivwth = rivwth_rm.reshape(-1)[pixel_id]
        pixel_is_permanent_water = (pix_rivwth != -9999.0) & (pix_rivwth != 0.0)

        catchment_id = np.unique(pixel_catchment_id)
        if self.coarse_params_nc is not None:
            from netCDF4 import Dataset

            with Dataset(self.coarse_params_nc, "r") as nc:
                full_cid = np.asarray(nc.variables["catchment_id"][:], dtype=np.int64)
                full_bid = np.asarray(
                    nc.variables["catchment_basin_id"][:], dtype=np.int64
                )
            pos = np.searchsorted(full_cid, catchment_id)
            matched = (pos < full_cid.size) & (full_cid[np.minimum(pos, full_cid.size - 1)] == catchment_id)
            catchment_basin_id = np.where(matched, full_bid[np.minimum(pos, full_cid.size - 1)], 0).astype(
                np.int64
            )
        else:
            basin_path = self.map_dir / "basin.bin"
            if basin_path.exists():
                basin_map = np.fromfile(basin_path, dtype="<i4").reshape(
                    (nxx, nyy), order="F"
                )
                cx_u, cy_u = np.unravel_index(catchment_id, (nxx, nyy))
                catchment_basin_id = np.maximum(basin_map[cx_u, cy_u], 0).astype(
                    np.int64
                )
            else:
                catchment_basin_id = np.zeros_like(catchment_id)

        # Output pixel selection (crops buffer zone when tributary buffer is active)
        if dx > 0 or dy > 0:
            pix_jx = pixel_id % mx
            pix_jy = pixel_id // mx
            in_target = (
                (pix_jx >= dx)
                & (pix_jx < dx + lx)
                & (pix_jy >= dy)
                & (pix_jy < dy + ly)
            )
            output_pixel_id = pixel_id[in_target]
        else:
            output_pixel_id = pixel_id.copy()

        data: dict[str, Any] = {
            "catchment_id": catchment_id,
            "catchment_basin_id": catchment_basin_id,
            "pixel_id": pixel_id,
            "pixel_catchment_id": pixel_catchment_id,
            "output_pixel_id": output_pixel_id,
            "pixel_flood_diff": pixel_flood_diff,
            "pixel_is_permanent_water": pixel_is_permanent_water,
        }

        if self.levee_file is not None and catmz100 is not None:
            levfrc_map = np.fromfile(self.levee_file, dtype="<f4").reshape(
                (nxx, nyy), order="F"
            )
            cx_u, cy_u = np.unravel_index(catchment_id, (nxx, nyy))
            c_levfrc = levfrc_map[cx_u, cy_u]
            has_lev = (c_levfrc >= 0.0) & (c_levfrc <= 1.0)
            levee_catchment_id = catchment_id[has_lev]
            levee_fraction = c_levfrc[has_lev].astype(np.float32)
            levee_id = np.arange(levee_catchment_id.size, dtype=np.int64)
            pixel_elevation_percentile = (
                np.clip(catmz100.T.reshape(-1)[pixel_id], 0, 100).astype(np.float32)
            )
            data.update(
                {
                    "levee_id": levee_id,
                    "levee_catchment_id": levee_catchment_id,
                    "levee_fraction": levee_fraction,
                    "pixel_elevation_percentile": pixel_elevation_percentile,
                }
            )

        if self.include_tributary and down_x is not None and down_y is not None:
            uparea = np.fromfile(self.map_dir / "uparea.bin", dtype="<f4").reshape(
                (nxx, nyy), order="F"
            )
            flat_to_compact = np.full(mx * my, -1, dtype=np.int64)
            flat_to_compact[pixel_id] = np.arange(pixel_id.size, dtype=np.int64)
            trib_data = _build_tributary_and_backwater_tables(
                catm_xx=catm_xx,
                catm_yy=catm_yy,
                down_x=down_x,
                down_y=down_y,
                flddif=flddif,
                next_xx=next_xx,
                next_yy=next_yy,
                uparea=uparea,
                pixel_flat_to_compact=flat_to_compact,
            )
            data.update(trib_data)

        if self.num_duration_bins is not None:
            data["duration_bin_id"] = np.arange(self.num_duration_bins, dtype=np.int64)

        self._last_catchment_id = catchment_id
        self.grid_metadata = {
            "coarse_nx": nxx,
            "coarse_ny": nyy,
            "mx": mx,
            "my": my,
            "lx": lx,
            "ly": ly,
            "dx": dx,
            "dy": dy,
            "csize": csize,
            "west": self.west,
            "east": self.east,
            "south": self.south,
            "north": self.north,
        }

        if self.out_nc is not None:
            self._write_netcdf(self.out_nc, data)

        return data

    def extract_coarse_field(self, coarse_grid: np.ndarray) -> np.ndarray:
        """Slice a coarse ``(nXX, nYY)`` or ``(..., nXX, nYY)`` Fortran field onto ``catchment_id``."""
        meta = self.grid_metadata
        nxx, nyy = meta["coarse_nx"], meta["coarse_ny"]
        arr = np.asarray(coarse_grid, dtype=np.float32)
        arr = np.where(arr >= 1.0e19, -9999.0, arr)
        cx, cy = np.unravel_index(self._last_catchment_id, (nxx, nyy))
        if arr.shape == (nxx, nyy):
            return arr[cx, cy]
        return arr[..., cx, cy]

    def _write_netcdf(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        dim_for_key = {
            "catchment_id": ("catchment",),
            "catchment_basin_id": ("catchment",),
            "pixel_id": ("pixel",),
            "pixel_catchment_id": ("pixel",),
            "output_pixel_id": ("output_pixel",),
            "pixel_flood_diff": ("pixel",),
            "pixel_is_permanent_water": ("pixel",),
            "levee_id": ("levee",),
            "levee_catchment_id": ("levee",),
            "levee_fraction": ("levee",),
            "pixel_elevation_percentile": ("pixel",),
            "tributary_pixel_id": ("tributary_pixel",),
            "tributary_link_id": ("tributary_link",),
            "tributary_target_id": ("tributary_link",),
            "tributary_upstream_pixel_id": ("tributary_link",),
            "tributary_downstream_pixel_id": ("tributary_link",),
            "tributary_chain_step": ("tributary_link",),
            "tributary_chain_length": ("tributary_link",),
            "backwater_pixel_id": ("backwater_pixel",),
            "backwater_target_pixel_id": ("backwater_pixel",),
            "backwater_floor_elevation": ("backwater_pixel",),
            "duration_bin_id": ("duration_bin",),
        }
        self._last_catchment_id = data["catchment_id"]
        with atomic_netcdf_dataset(path) as ds:
            for name, val in self.grid_metadata.items():
                setattr(ds, name, val)
            created_dims: set[str] = set()
            for key, arr in data.items():
                dims = dim_for_key[key]
                for d_name, length in zip(dims, arr.shape, strict=True):
                    if d_name not in created_dims:
                        ds.createDimension(d_name, int(length))
                        created_dims.add(d_name)
                if arr.dtype == np.bool_:
                    nc_dtype = "u1"
                    write_arr = arr.astype(np.uint8)
                elif arr.dtype == np.int64:
                    nc_dtype = "i8"
                    write_arr = arr
                else:
                    nc_dtype = "f4"
                    write_arr = arr.astype(np.float32)
                var = ds.createVariable(key, nc_dtype, dims, zlib=True, complevel=4)
                if arr.dtype == np.bool_:
                    var.setncattr(LOGICAL_DTYPE_ATTR, BOOL_LOGICAL_DTYPE)
                var[:] = write_arr
