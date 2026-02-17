# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
River channel width and depth estimation from runoff climatology.

Mirrors the logic of CaMa-Flood's ``calc_rivwth.F90``:

.. code-block:: text

    river_height = max(HMIN, HC * Q^HP + HO)
    river_width  = max(WMIN, WC * Q^WP + WO)

where *Q* is the mean annual discharge (m³/s) accumulated from upstream
to downstream along the river network.

**Workflow**:

1. Read catchment-level mean runoff from a climatology NetCDF produced by
   :meth:`AbstractDataset.export_runoff_climatology`.
2. Read the river network (``catchment_id``, ``downstream_id``) from
   :class:`MERITMap`'s ``parameters.nc``.
3. Accumulate local runoff from upstream to downstream to obtain mean
   discharge *Q* at every catchment.
4. Apply the power-law relationship to compute ``river_width`` and
   ``river_height``.
5. Write the results back into the parameters NetCDF (overwrite in-place
   or create a new copy).

**Unit convention**:

The ``export_runoff_climatology`` output stores the area-weighted sum of
runoff per catchment.  The stored units depend on the dataset's
``unit_factor``:

* ``unit_factor = 86400000`` (mm/day → **m/s**): climatology is already in
  **m³/s** (m/s × area_m²).  Use ``runoff_to_m3s = 1.0`` **(default)**.
* ``unit_factor = 86400`` (mm/day → **mm/s**): climatology is in mm·m²/s.
  Use ``runoff_to_m3s = 1e-3`` (``mm → m``).

Example
-------
>>> from cmfgpu.params.calc_rivwth import estimate_river_geometry
>>> estimate_river_geometry(
...     climatology_nc="output/runoff_clm.nc",
...     parameter_nc="inp/glb_15min/parameters.nc",
...     output_nc="inp/glb_15min/parameters_new.nc",   # None → overwrite
... )
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Union

import numba
import numpy as np
from netCDF4 import Dataset

from cmfgpu.params.utils import compute_init_river_depth
from cmfgpu.utils import find_indices_in


# ---------------------------------------------------------------------------
# Numba-accelerated kernels
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _accumulate_discharge(
    local_runoff: np.ndarray,
    downstream_idx: np.ndarray,
) -> np.ndarray:
    """Accumulate local runoff along the river network.

    Catchments **must** already be stored in topological (upstream-first)
    order — which is the ordering guaranteed by :class:`MERITMap`.

    Parameters
    ----------
    local_runoff : (N,) float64
        Local runoff at each catchment (m³/s after unit conversion).
    downstream_idx : (N,) int64
        Index (into this same array) of the downstream catchment.
        River mouths have ``downstream_idx[i] == i`` (self-loop) or ``< 0``.

    Returns
    -------
    discharge : (N,) float64
        Accumulated discharge at each catchment (m³/s).
    """
    n = local_runoff.shape[0]
    discharge = local_runoff.copy()
    for i in range(n):
        ds = downstream_idx[i]
        if 0 <= ds < n and ds != i:
            discharge[ds] += discharge[i]
    return discharge


@numba.njit(parallel=True, cache=True)
def _power_law(
    discharge: np.ndarray,
    HC: float,
    HP: float,
    HO: float,
    HMIN: float,
    WC: float,
    WP: float,
    WO: float,
    WMIN: float,
):
    """Compute river height and width from discharge via power law.

    Formulae (matching ``calc_rivwth.F90``):

    .. code-block:: text

        height = max(HMIN, HC * Q^HP + HO)
        width  = max(WMIN, WC * Q^WP + WO)

    Parameters
    ----------
    discharge : (N,) float64
        Mean annual discharge (m³/s).

    Returns
    -------
    width  : (N,) float32
    height : (N,) float32
    """
    n = discharge.shape[0]
    width = np.empty(n, dtype=np.float32)
    height = np.empty(n, dtype=np.float32)
    for i in numba.prange(n):
        q = abs(discharge[i])
        if q > 0.0:
            height[i] = max(HMIN, HC * q ** HP + HO)
            width[i] = max(WMIN, WC * q ** WP + WO)
        else:
            height[i] = HMIN
            width[i] = WMIN
    return width, height


@numba.njit(parallel=True, cache=True)
def _fuse_satellite_width(
    rivwth: np.ndarray,
    gwdlr: np.ndarray,
    sat_min_threshold: float = 50.0,
    lower_ratio: float = 0.5,
    upper_ratio: float = 5.0,
    max_width: float = 10000.0,
):
    """Fuse power-law width with satellite (GWD-LR) width.

    Mirrors the logic of ``set_gwdlr.F90``::

        if   gwdlr <  sat_min_threshold:   fused = max(gwdlr, rivwth)
        elif gwdlr <  rivwth * lower_ratio: fused = rivwth * lower_ratio
        else:
             if gwdlr > rivwth * upper_ratio: fused = rivwth * upper_ratio
             if fused  > max_width:            fused = max_width

    For catchments where satellite width is non-positive (no observation),
    the power-law width is kept unchanged.

    Parameters
    ----------
    rivwth : (N,) float32
        Power-law river width (m).
    gwdlr : (N,) float32
        Satellite-derived river width (m).  Non-positive values mean no data.
    sat_min_threshold : float
        Below this value, the satellite width is considered unreliable and we
        take the max of satellite and power-law (default 50 m).
    lower_ratio : float
        If satellite < rivwth * lower_ratio, clamp to rivwth * lower_ratio.
    upper_ratio : float
        If satellite > rivwth * upper_ratio, clamp to rivwth * upper_ratio.
    max_width : float
        Absolute maximum allowed width (default 10000 m).

    Returns
    -------
    fused : (N,) float32
        Fused river width.
    """
    n = rivwth.shape[0]
    fused = np.empty(n, dtype=np.float32)
    for i in numba.prange(n):
        w = rivwth[i]
        g = gwdlr[i]
        if g <= 0.0:
            # No satellite data — keep power-law
            fused[i] = w
        elif g < sat_min_threshold:
            fused[i] = max(g, w)
        elif g < w * lower_ratio:
            fused[i] = w * lower_ratio
        else:
            out = g
            if out > w * upper_ratio:
                out = w * upper_ratio
            if out > max_width:
                out = max_width
            fused[i] = out
    return fused


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_river_geometry(
    climatology_nc: Union[str, Path],
    parameter_nc: Union[str, Path],
    output_nc: Optional[Union[str, Path]] = None,
    clm_var: str = "runoff_clm",
    runoff_to_m3s: float = 1.0,
    HC: float = 0.1,
    HP: float = 0.50,
    HO: float = 0.0,
    HMIN: float = 1.0,
    WC: float = 2.50,
    WP: float = 0.60,
    WO: float = 0.0,
    WMIN: float = 5.0,
    verbose: bool = True,
) -> Path:
    """Estimate river width, height from climatology.

    Parameters
    ----------
    climatology_nc : path
        NetCDF produced by ``AbstractDataset.export_runoff_climatology``.
        Must contain ``catchment_id`` and the variable *clm_var*.
    parameter_nc : path
        ``parameters.nc`` produced by :class:`MERITMap`.  Must contain
        ``catchment_id``, ``downstream_id``, ``river_width``, ``river_height``.
        If the file also contains ``satellite_width``, the power-law width
        is automatically fused with satellite data via
        :func:`_fuse_satellite_width`.
    output_nc : path or None
        Where to write the result.

        * *None* → overwrite *parameter_nc* in-place.
        * otherwise → copy *parameter_nc* to *output_nc*, then update.
    clm_var : str
        Variable name inside *climatology_nc* (default ``"runoff_clm"``).
    runoff_to_m3s : float
        Multiplicative factor to convert the values stored in *climatology_nc*
        to **m³/s**.  Default ``1.0`` (appropriate when the dataset's
        ``unit_factor`` already converts raw data to m/s and the mapping
        matrix contains areas in m², so climatology is already in m³/s).
    HC, HP, HO, HMIN : float
        Power-law coefficients for river **height** (bank depth).
        ``height = max(HMIN, HC * Q^HP + HO)``.
    WC, WP, WO, WMIN : float
        Power-law coefficients for river **width**.
        ``width = max(WMIN, WC * Q^WP + WO)``.
    verbose : bool
        Print progress information.

    Returns
    -------
    Path
        Path to the written NetCDF file.
    """
    climatology_nc = Path(climatology_nc)
    parameter_nc = Path(parameter_nc)

    if not climatology_nc.exists():
        raise FileNotFoundError(f"Climatology file not found: {climatology_nc}")
    if not parameter_nc.exists():
        raise FileNotFoundError(f"Parameter file not found: {parameter_nc}")

    # ------------------------------------------------------------------
    # 1. Read climatology (catchment_id → mean runoff)
    # ------------------------------------------------------------------
    with Dataset(str(climatology_nc), "r") as ds:
        clm_cids = np.asarray(ds.variables["catchment_id"][:]).astype(np.int64)
        clm_vals = np.asarray(ds.variables[clm_var][:]).astype(np.float64)

    if verbose:
        print(f"[calc_rivwth] Loaded climatology: {len(clm_cids)} catchments "
              f"from {climatology_nc.name}")

    # ------------------------------------------------------------------
    # 2. Read river network from parameters.nc
    # ------------------------------------------------------------------
    with Dataset(str(parameter_nc), "r") as ds:
        param_cids = np.asarray(ds.variables["catchment_id"][:]).astype(np.int64)
        param_dsid = np.asarray(ds.variables["downstream_id"][:]).astype(np.int64)

    n_catch = len(param_cids)

    if verbose:
        print(f"[calc_rivwth] Loaded parameters: {n_catch} catchments "
              f"from {parameter_nc.name}")

    # ------------------------------------------------------------------
    # 3. Map climatology values onto the parameter catchment array
    # ------------------------------------------------------------------
    # clm_cids may be a subset or the same set as param_cids.
    clm_to_param = find_indices_in(clm_cids, param_cids)
    valid = clm_to_param >= 0
    if not np.all(valid):
        n_miss = int((~valid).sum())
        print(f"[calc_rivwth] Warning: {n_miss} climatology catchments "
              "not found in parameter file — ignored.")

    local_runoff = np.zeros(n_catch, dtype=np.float64)
    local_runoff[clm_to_param[valid]] = clm_vals[valid] * runoff_to_m3s

    # ------------------------------------------------------------------
    # 4. Build downstream index array and accumulate discharge
    # ------------------------------------------------------------------
    downstream_idx = find_indices_in(param_dsid, param_cids)
    # River mouths: downstream_id == self → downstream_idx == self index
    # They are treated as self-loops in _accumulate_discharge (no transfer).

    discharge = _accumulate_discharge(local_runoff, downstream_idx)

    if verbose:
        q_pos = discharge[discharge > 0]
        if len(q_pos) > 0:
            print(f"[calc_rivwth] Discharge stats (m³/s): "
                  f"min={q_pos.min():.4f}, median={np.median(q_pos):.4f}, "
                  f"max={q_pos.max():.4f}")
        else:
            print("[calc_rivwth] Warning: all discharge values are zero!")

    # ------------------------------------------------------------------
    # 5. Compute river width and height via power law
    # ------------------------------------------------------------------
    new_width, new_height = _power_law(
        discharge, HC, HP, HO, HMIN, WC, WP, WO, WMIN,
    )

    if verbose:
        print(f"[calc_rivwth] Height: "
              f"H = max({HMIN}, {HC}*Q^{HP}+{HO})")
        print(f"[calc_rivwth] Width:  "
              f"W = max({WMIN}, {WC}*Q^{WP}+{WO})")
        print(f"[calc_rivwth] Result height range: "
              f"[{new_height.min():.2f}, {new_height.max():.2f}] m")
        print(f"[calc_rivwth] Power-law width range: "
              f"[{new_width.min():.2f}, {new_width.max():.2f}] m")

    # ------------------------------------------------------------------
    # 5b. Fuse with satellite-derived width (if present in parameter_nc)
    # ------------------------------------------------------------------
    with Dataset(str(parameter_nc), "r") as ds:
        has_sat = "satellite_width" in ds.variables
        if has_sat:
            sat = np.asarray(ds.variables["satellite_width"][:]).astype(np.float32)

    if has_sat:
        new_width = _fuse_satellite_width(new_width, sat)
        if verbose:
            n_sat = int(np.count_nonzero(sat > 0))
            print(f"[calc_rivwth] Fused with satellite width "
                  f"({n_sat}/{n_catch} cells have satellite observations) → "
                  f"final width range [{new_width.min():.2f}, {new_width.max():.2f}] m")

    # ------------------------------------------------------------------
    # 6. Recompute river_depth and river_storage
    #    (they depend on river_height and river_width)
    # ------------------------------------------------------------------
    with Dataset(str(parameter_nc), "r") as ds:
        catchment_elevation = np.asarray(ds.variables["catchment_elevation"][:]).astype(np.float32)
        river_length = np.asarray(ds.variables["river_length"][:]).astype(np.float32)

    new_depth = compute_init_river_depth(
        catchment_elevation, new_height, downstream_idx,
    )
    new_storage = river_length * new_width * new_depth

    if verbose:
        print(f"[calc_rivwth] Recomputed river_depth  range: "
              f"[{new_depth.min():.4f}, {new_depth.max():.4f}] m")
        print(f"[calc_rivwth] Recomputed river_storage range: "
              f"[{new_storage.min():.2f}, {new_storage.max():.2f}] m³")

    # ------------------------------------------------------------------
    # 7. Optionally update bifurcation_elevation level-0
    #    Level-0 elevation = pelv - dph, where
    #    dph = clamp(log10(wth[0]) * 2.5 - 4, 0.5, max(rivhgt_up, rivhgt_dn))
    #    When river_height changes, the upper bound of the clamp changes.
    # ------------------------------------------------------------------
    new_bif_elv = None
    with Dataset(str(parameter_nc), "r") as ds:
        has_bif = "bifurcation_elevation" in ds.variables
        if has_bif:
            bif_elv = np.asarray(ds.variables["bifurcation_elevation"][:]).astype(np.float64)
            bif_wth = np.asarray(ds.variables["bifurcation_width"][:]).astype(np.float64)
            bif_cid = np.asarray(ds.variables["bifurcation_catchment_id"][:]).astype(np.int64)
            bif_did = np.asarray(ds.variables["bifurcation_downstream_id"][:]).astype(np.int64)

    if has_bif:
        new_bif_elv = _update_bifurcation_elevation(
            bif_elv, bif_wth, bif_cid, bif_did,
            param_cids, new_height,
        )
        if verbose:
            changed = np.count_nonzero(bif_elv[:, 0] != new_bif_elv[:, 0])
            print(f"[calc_rivwth] Updated {changed}/{len(bif_elv)} "
                  "bifurcation level-0 elevations")

    # ------------------------------------------------------------------
    # 8. Write results to NetCDF
    # ------------------------------------------------------------------
    if output_nc is None:
        target = parameter_nc
    else:
        target = Path(output_nc)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parameter_nc, target)
        if verbose:
            print(f"[calc_rivwth] Copied {parameter_nc.name} → {target}")

    with Dataset(str(target), "r+") as ds:
        # river_width
        if "river_width" in ds.variables:
            ds.variables["river_width"][:] = new_width
        else:
            dims = _infer_catchment_dim(ds, n_catch)
            var = ds.createVariable(
                "river_width", "f4", dims, zlib=True, complevel=4,
            )
            var[:] = new_width
            var.setncattr("units", "m")
            var.setncattr("long_name", "river channel width")

        # river_height
        if "river_height" in ds.variables:
            ds.variables["river_height"][:] = new_height
        else:
            dims = _infer_catchment_dim(ds, n_catch)
            var = ds.createVariable(
                "river_height", "f4", dims, zlib=True, complevel=4,
            )
            var[:] = new_height
            var.setncattr("units", "m")
            var.setncattr("long_name", "river bank height")

        # river_depth
        if "river_depth" in ds.variables:
            ds.variables["river_depth"][:] = new_depth
        else:
            dims = _infer_catchment_dim(ds, n_catch)
            var = ds.createVariable(
                "river_depth", "f4", dims, zlib=True, complevel=4,
            )
            var[:] = new_depth
            var.setncattr("units", "m")
            var.setncattr("long_name", "initial river water depth")

        # river_storage
        if "river_storage" in ds.variables:
            ds.variables["river_storage"][:] = new_storage
        else:
            dims = _infer_catchment_dim(ds, n_catch)
            var = ds.createVariable(
                "river_storage", "f4", dims, zlib=True, complevel=4,
            )
            var[:] = new_storage
            var.setncattr("units", "m3")
            var.setncattr("long_name", "initial river storage")

        # bifurcation_elevation
        if new_bif_elv is not None:
            ds.variables["bifurcation_elevation"][:] = new_bif_elv.astype(np.float32)

        # Store power-law metadata as global attributes
        ds.setncattr("rivwth_formula", f"max({WMIN}, {WC}*Q^{WP}+{WO})")
        ds.setncattr("rivhgt_formula", f"max({HMIN}, {HC}*Q^{HP}+{HO})")
        ds.setncattr("runoff_to_m3s", runoff_to_m3s)

    if verbose:
        print(f"[calc_rivwth] Written river_width, river_height, "
              f"river_depth, river_storage to {target}")

    return target


def _update_bifurcation_elevation(
    bif_elv: np.ndarray,
    bif_wth: np.ndarray,
    bif_cid: np.ndarray,
    bif_did: np.ndarray,
    param_cids: np.ndarray,
    new_height: np.ndarray,
) -> np.ndarray:
    """Recompute bifurcation level-0 elevation using updated river heights.

    Mirrors the Fortran ``set_bifparam`` logic:

    .. code-block:: text

        dph = clamp(log10(wth[0]) * 2.5 - 4.0, 0.5,
                    max(rivhgt_up, rivhgt_dn))
        elv_level0 = pelv - dph

    Only level 0 depends on ``river_height``; higher levels are
    ``pelv + (ilev - 1)`` and remain unchanged.
    """
    out = bif_elv.copy()
    w0 = bif_wth[:, 0]
    pos = w0 > 0.0
    if not np.any(pos):
        return out

    # Map bif endpoints to param indices
    up_idx = find_indices_in(bif_cid, param_cids)
    dn_idx = find_indices_in(bif_did, param_cids)

    # Recover pelv from the original table:
    #   original: elv[i, 0] = pelv[i] - dph_old[i]
    #   and for ilev>=1: elv[i, ilev] = pelv[i] + (ilev - 1)
    # Use level-1 if available (more reliable: pelv = elv[i,1] - 0 = elv[i,1])
    if bif_elv.shape[1] >= 2:
        # pelv = elv[:,1] when wth[:,1]>0; otherwise derive from level 0
        has_lev1 = bif_wth[:, 1] > 0.0
        pelv = np.where(has_lev1, bif_elv[:, 1], np.nan)
        # For paths without valid level-1, we cannot recover pelv reliably;
        # leave their elevation unchanged.
        can_update = pos & has_lev1 & (up_idx >= 0) & (dn_idx >= 0)
    else:
        # Only 1 level — cannot recover pelv; skip
        return out

    if not np.any(can_update):
        return out

    # Recompute dph with new heights
    dph_raw = np.log10(w0[can_update]) * 2.5 - 4.0
    dph_raw = np.maximum(dph_raw, 0.5)
    h_up = new_height[up_idx[can_update]].astype(np.float64)
    h_dn = new_height[dn_idx[can_update]].astype(np.float64)
    dph_max = np.maximum(h_up, h_dn)
    dph_new = np.minimum(dph_raw, dph_max)

    out[can_update, 0] = pelv[can_update] - dph_new
    return out


def _infer_catchment_dim(ds: Dataset, n_catch: int) -> tuple:
    """Find the dimension name matching *n_catch* in an open NetCDF dataset."""
    for dim_name, dim in ds.dimensions.items():
        if len(dim) == n_catch:
            return (dim_name,)
    raise ValueError(
        f"Cannot find a dimension of size {n_catch} in the NetCDF file."
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    map_resolution = "glb_15min"
    estimate_river_geometry(
        climatology_nc=f"/home/eat/CaMa-Flood-GPU/inp/{map_resolution}/runoff_clm.nc",
        parameter_nc=f"/home/eat/CaMa-Flood-GPU/inp/{map_resolution}/parameters.nc",
        output_nc=f"/home/eat/CaMa-Flood-GPU/inp/{map_resolution}/parameters_new.nc",
        runoff_to_m3s=1.0,  # climatology already in m³/s
    )
