# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Dam / reservoir parameter estimation from naturalized discharge.

Builds dam parameters from the standard CaMa-Flood discharge statistics
pipeline:

.. code-block:: text

    1. Extract annual-max and mean discharge at each dam grid cell.
    2. Fit a Gumbel distribution (L-moments) to obtain 100-yr return period
       flood discharge.
    3. Estimate flood-control storage from GRSAD surface-area time series
       and ReGeom bathymetry (optional; fallback = 37 % of total capacity).
    4. Merge results into per-dam Qn, Qf, FldVol, ConVol.

**Workflow** (GPU / NetCDF path):

1. Read per-dam *total capacity* from a GRanD dam-list CSV
   (``GRanD_allocated.csv``) and *catchment_id* from a dam allocation
   result (produced by :class:`HiResMap`'s ``build_dams`` pipeline).
2. Use a pre-aggregated statistics NetCDF (produced by
   :class:`StatisticsRuntime` with ``max_mean`` and ``mean_mean`` ops)
   to read annual-max peaks and mean discharge at dam cells.
3. Fit Gumbel → Q100 → ``Qf = 0.3 * Q100`` (with Qf < Qn adjustment).
4. Compute flood-control / conservation storage (GRSAD + ReGeom **or**
   37 % fallback).
5. De-duplicate multiple dams on one grid cell (keep largest capacity).
6. Write results to a CaMa-Flood-compatible CSV **and/or** append reservoir
   variables to an existing ``parameters.nc``.

Example
-------
>>> from cmfgpu.params.estimate_dam_params import estimate_dam_params
>>> estimate_dam_params(
...     dam_list="inp/GRanD_allocated.csv",
...     dam_alloc="inp/dam_alloc.txt",
...     parameter_nc="inp/glb_15min/parameters.nc",
...     outflow_stats_nc="output/total_outflow_stats_rank0.nc",
...     output_csv="output/dam_params.csv",
...     output_nc="inp/glb_15min/parameters.nc",   # update in-place
... )
"""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path

import numba
import numpy as np
from hydroforge.core import find_indices_in
from hydroforge.io import MultiRankStatsReader
from hydroforge.io.netcdf.encoding import read_netcdf_values
from netCDF4 import Dataset

# ---------------------------------------------------------------------------
# Numba-accelerated kernels
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _gumbel_100yr(annual_max: np.ndarray) -> float:
    """Estimate 100-year return-period discharge via Gumbel L-moments.

    Parameters
    ----------
    annual_max : (N,) float64
        Sorted (ascending) annual-maximum discharge values.

    Returns
    -------
    float
        100-year return-period discharge (m³/s).  Returns NaN on failure.
    """
    n = annual_max.shape[0]
    if n < 2:
        return np.nan

    # Check for constant series
    if annual_max[0] == annual_max[-1]:
        return np.nan

    # L-moment estimation (Weibull plotting position, alpha = 0)
    b0 = 0.0
    b1 = 0.0
    for i in range(n):
        b0 += annual_max[i]
        b1 += i * annual_max[i]
    b0 /= n
    b1 /= n * (n - 1)

    lam1 = b0
    lam2 = 2.0 * b1 - b0

    a = lam2 / 0.6931471805599453  # ln(2)
    c = lam1 - 0.5772156649015329 * a  # Euler-Mascheroni constant

    # 100-year return period
    prob = 1.0 - 1.0 / 100.0
    yp = c - a * np.log(-np.log(prob))

    if yp <= 0.0:
        return np.nan
    return yp


@numba.njit(cache=True, parallel=True)
def _estimate_qf_batch(
    annual_max: np.ndarray,
    qn: np.ndarray,
    qf_ratio: float,
) -> np.ndarray:
    """Numba-parallel Gumbel 100-yr Qf estimation for all dams.

    Parameters
    ----------
    annual_max : (Y, D) float64
    qn : (D,) float64
    qf_ratio : float

    Returns
    -------
    qf : (D,) float64
    """
    n_dam = annual_max.shape[1]
    qf = np.empty(n_dam, dtype=np.float64)

    for d in numba.prange(n_dam):
        # Filter NaN / sentinel values
        col = annual_max[:, d]
        count = 0
        for i in range(col.shape[0]):
            if np.isfinite(col[i]) and col[i] < 1e20:
                count += 1
        if count < 2:
            qf[d] = 2.0 * qn[d]
            continue

        am = np.empty(count, dtype=np.float64)
        k = 0
        for i in range(col.shape[0]):
            if np.isfinite(col[i]) and col[i] < 1e20:
                am[k] = col[i]
                k += 1
        am.sort()

        q100 = _gumbel_100yr(am)
        if np.isnan(q100):
            qf[d] = 2.0 * qn[d]
            continue

        qf_val = qf_ratio * q100
        if qf_val < qn[d]:
            if 0.4 * q100 >= qn[d]:
                qf_val = 0.4 * q100
            else:
                qf_val = 2.0 * qn[d]
        qf[d] = qf_val

    return qf


@numba.njit(cache=True, parallel=True)
def _extract_dam_stats(
    max_data: np.ndarray,
    mean_data: np.ndarray,
    idx_max: np.ndarray,
    idx_mean: np.ndarray,
    valid: np.ndarray,
) -> tuple:
    """Numba-parallel extraction of per-dam annual-max and mean Q.

    Parameters
    ----------
    max_data : (Y, S) float64
    mean_data : (Y, S) float64
    idx_max, idx_mean : (D,) int64   (index into S dimension, -1 = invalid)
    valid : (D,) bool

    Returns
    -------
    annual_max_all : (Y, D) float64
    qn : (D,) float64
    """
    n_years = max_data.shape[0]
    n_dam = idx_max.shape[0]
    annual_max_all = np.full((n_years, n_dam), np.nan, dtype=np.float64)
    qn = np.full(n_dam, 1e-10, dtype=np.float64)

    for d in numba.prange(n_dam):
        if not valid[d]:
            continue
        im = idx_max[d]
        for t in range(n_years):
            v = max_data[t, im]
            if not np.isfinite(v) or v >= 1e20 or v <= -999.0:
                annual_max_all[t, d] = np.nan
            else:
                annual_max_all[t, d] = v if v > 0.0 else 0.0
        # mean
        s = 0.0
        count = 0
        imn = idx_mean[d]
        for t in range(n_years):
            value = mean_data[t, imn]
            if np.isfinite(value) and value < 1e20 and value > -999.0:
                s += value
                count += 1
        avg = s / count if count > 0 else 1e-10
        qn[d] = avg if avg > 1e-10 else 1e-10

    return annual_max_all, qn


# Column-name aliases (lower-cased) for GRanD dam-list CSV
_CSV_COL_MAP: dict[str, list[str]] = {
    "ids": ["id", "grand_id", "dam_id"],
    "lats": ["lat_alloc", "lat_merit", "lat_ori", "lat"],
    "lons": ["lon_alloc", "lon_merit", "lon_ori", "lon"],
    "upareas": ["area_alloc", "area_merit", "area_ori", "uparea"],
    "names": ["damname", "dam_name", "name"],
    "cap_mcm": ["cap_mcm"],
    "years": ["year"],
}


def _resolve_col(header_lower: list[str], aliases: list[str]) -> int | None:
    """Return the column index matching any alias, or None."""
    for alias in aliases:
        if alias in header_lower:
            return header_lower.index(alias)
    return None


def _load_dam_list_csv(dam_list_path: Path) -> dict:
    """Parse comma-separated GRanD dam-list CSV."""
    ids, lats, lons, upareas, names = [], [], [], [], []
    cap_mcm_list, years_list = [], []

    with open(dam_list_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        header_lower = [h.strip().lower() for h in header]

        # Resolve column indices
        col = {}
        for key, aliases in _CSV_COL_MAP.items():
            col[key] = _resolve_col(header_lower, aliases)

        # Validate required columns
        required = ["ids", "lats", "lons", "cap_mcm"]
        missing = [k for k in required if col[k] is None]
        if missing:
            raise ValueError(
                f"Dam list CSV is missing required columns: {missing}. Header: {header}"
            )

        # After validation, required columns are guaranteed non-None
        ids_col: int = col["ids"]  # type: ignore[assignment]
        lats_col: int = col["lats"]  # type: ignore[assignment]
        lons_col: int = col["lons"]  # type: ignore[assignment]
        cap_col: int = col["cap_mcm"]  # type: ignore[assignment]

        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                record = (
                    int(row[ids_col]),
                    float(row[lats_col]),
                    float(row[lons_col]),
                    float(row[col["upareas"]]) if col["upareas"] is not None else -999.0,
                    row[col["names"]].strip()
                    if col["names"] is not None else f"dam_{row[ids_col]}",
                    float(row[cap_col]),
                    int(row[col["years"]]) if col["years"] is not None else -99,
                )
            except (ValueError, IndexError):
                continue  # malformed row, including present-but-empty year: discard all
            for column, value in zip(
                (ids, lats, lons, upareas, names, cap_mcm_list, years_list),
                record, strict=True,
            ):
                column.append(value)

    if len({len(column) for column in
            (ids, lats, lons, upareas, names, cap_mcm_list, years_list)}) != 1:
        raise RuntimeError("Dam CSV columns lost row alignment")
    return {
        "ids": np.array(ids, dtype=np.int64),
        "lats": np.array(lats, dtype=np.float64),
        "lons": np.array(lons, dtype=np.float64),
        "upareas": np.array(upareas, dtype=np.float64),
        "names": names,
        "cap_mcm": np.array(cap_mcm_list, dtype=np.float64),
        "years": np.array(years_list, dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_dam_list(dam_list_path: Path) -> dict:
    """Parse a ``GRanD_allocated.csv`` (or similar) dam-list CSV.

    Column matching is case-insensitive via :data:`_CSV_COL_MAP`.

    Returns
    -------
    dict
        Keys: ``ids``, ``lats``, ``lons``, ``upareas``, ``names``,
        ``cap_mcm``, ``years``.
    """
    return _load_dam_list_csv(dam_list_path)


def _read_alloc_file(path: Path) -> np.ndarray:
    """Read dam allocation output from :meth:`DamAllocMixin.write_dam_alloc_file`.

    Returns a structured array with fields ``id``, ``ix``, ``iy``,
    ``catchment_id``, ``area_cama``.
    """
    ids, ixs, iys, cids, areas = [], [], [], [], []
    with open(path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.split()
            if len(parts) < 10:
                continue
            ids.append(int(parts[0]))
            areas.append(float(parts[4]))  # area_CaMa
            ixs.append(int(parts[7]))  # 1-based
            iys.append(int(parts[8]))  # 1-based
            cids.append(int(parts[9]))  # catchment_id

    n = len(ids)
    dtype = np.dtype(
        [
            ("id", np.int64),
            ("ix", np.int32),
            ("iy", np.int32),
            ("catchment_id", np.int64),
            ("area_cama", np.float64),
        ]
    )
    arr = np.empty(n, dtype=dtype)
    arr["id"] = ids
    arr["ix"] = np.array(ixs, dtype=np.int32) - 1  # 1-based → 0-based
    arr["iy"] = np.array(iys, dtype=np.int32) - 1
    arr["catchment_id"] = cids
    arr["area_cama"] = areas
    return arr


def _resolve_dam_catchment_ids(
    dam_info: dict,
    dam_alloc: str | Path | np.ndarray,
) -> np.ndarray:
    """Resolve catchment_id for each dam from allocation results.

    Also updates *dam_info* in-place with ``ix``, ``iy`` (1-based),
    and ``area_cama`` from the allocation result.

    Parameters
    ----------
    dam_info : dict
        Output of :func:`_load_dam_list`.
    dam_alloc : path or structured array
        Allocation result from
        ``DamAllocMixin.dam_results_as_structured_array()`` or a path to
        the alloc output text file written by
        ``DamAllocMixin.write_dam_alloc_file()``.

    Returns
    -------
    dam_cids : (D,) int64
        Catchment ID for each dam.  ``-1`` means unallocated.
    """
    if isinstance(dam_alloc, (str, Path)):
        dam_alloc = _read_alloc_file(Path(dam_alloc))

    # Build mapping: alloc dam id → index in alloc array
    alloc_id_to_idx: dict[int, int] = {}
    for i in range(len(dam_alloc)):
        alloc_id_to_idx[int(dam_alloc["id"][i])] = i

    n_dam = len(dam_info["ids"])
    dam_cids = np.full(n_dam, -1, dtype=np.int64)
    ix_arr = np.full(n_dam, 0, dtype=np.int64)
    iy_arr = np.full(n_dam, 0, dtype=np.int64)
    area_cama = np.full(n_dam, -999.0, dtype=np.float64)

    for i in range(n_dam):
        dam_id = int(dam_info["ids"][i])
        if dam_id in alloc_id_to_idx:
            j = alloc_id_to_idx[dam_id]
            dam_cids[i] = int(dam_alloc["catchment_id"][j])
            # alloc stores 0-based; convert to 1-based for CSV output
            ix_arr[i] = int(dam_alloc["ix"][j]) + 1
            iy_arr[i] = int(dam_alloc["iy"][j]) + 1
            area_cama[i] = float(dam_alloc["area_cama"][j])

    dam_info["ix"] = ix_arr
    dam_info["iy"] = iy_arr
    dam_info["area_cama"] = area_cama

    return dam_cids


def _deduplicate_dams(
    dam_ids: np.ndarray,
    dam_cids: np.ndarray,
    cap_mcm: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """Remove smaller dams when multiple dams share the same grid cell.

    Returns
    -------
    keep_mask : (D,) bool
        True for dams to keep.
    """
    keep = np.ones(len(dam_ids), dtype=np.bool_)

    cid_to_indices = defaultdict(list)
    for i, cid in enumerate(dam_cids):
        if cid >= 0:
            cid_to_indices[int(cid)].append(i)

    n_removed = 0
    for indices in cid_to_indices.values():
        if len(indices) <= 1:
            continue
        caps = cap_mcm[indices]
        max_cap = caps.max()
        for idx in indices:
            if cap_mcm[idx] < max_cap:
                keep[idx] = False
                n_removed += 1
        # If all have the same capacity, keep first only
        if np.sum(caps == max_cap) > 1:
            first_found = False
            for idx in indices:
                if cap_mcm[idx] == max_cap:
                    if first_found:
                        keep[idx] = False
                        n_removed += 1
                    first_found = True

    if verbose and n_removed > 0:
        n_cells = sum(1 for v in cid_to_indices.values() if len(v) > 1)
        print(
            f"[dam_params] Dedup: removed {n_removed} smaller dams "
            f"across {n_cells} shared grid cells"
        )

    return keep


def _estimate_flood_storage_ratio(
    cap_mcm: np.ndarray,
    ratio: float = 0.37,
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback: estimate flood/conservation storage as a fixed ratio.

    Parameters
    ----------
    cap_mcm : (D,) float64
        Total reservoir capacity in MCM.
    ratio : float
        Fraction of total capacity allocated to flood control (default 0.37).

    Returns
    -------
    fld_vol_mcm : (D,) float64
    con_vol_mcm : (D,) float64
    """
    fld_vol = cap_mcm * ratio
    con_vol = cap_mcm - fld_vol
    return fld_vol, con_vol


def _estimate_flood_storage_grsad(
    dam_id: int,
    total_cap_mcm: float,
    grsad_dir: Path,
    regeom_dir: Path,
    percentile: float = 75.0,
) -> float:
    """Estimate flood-control storage for one dam from GRSAD + ReGeom.

    Parameters
    ----------
    dam_id : int
        GRanD dam ID.
    total_cap_mcm : float
        Total reservoir capacity (MCM) from the GRanD database.
    grsad_dir : Path
        Directory containing ``{dam_id}_intp`` time series files.
    regeom_dir : Path
        Directory containing ``{dam_id}.csv`` bathymetry files.
    percentile : float
        Percentile of GRSAD surface-area corresponding to normal water
        level (default 75).

    Returns
    -------
    float
        Flood-control storage in MCM.  Returns ``NaN`` if data is missing.
    """
    grsad_path = grsad_dir / f"{dam_id}_intp"
    regeom_path = regeom_dir / f"{dam_id}.csv"
    if not grsad_path.is_file() or not regeom_path.is_file():
        return np.nan
    # ---- Read GRSAD surface-area time series ----
    areas = []
    with grsad_path.open(newline="", encoding="utf-8") as source:
        rows = csv.reader(source, delimiter="\t")
        header = next(rows)
        if "3water_enh" not in header:
            return np.nan
        area_column = header.index("3water_enh") - 1
        for row in rows:
            if not row:
                continue
            row += [""] * (len(header) - len(row))
            # The first column is a date label; all remaining columns are
            # measurements. Preserve the former whole-row missing-data filter.
            values = np.array(
                [
                    np.nan
                    if value.strip() in {"", "NA", "N/A", "null"}
                    else float(value)
                    for value in row[1:]
                ]
            )
            if not np.isnan(values).any():
                areas.append(values[area_column])
    data = np.asarray(areas, dtype=np.float64)
    # Remove suspicious repeated values (>12 identical records).
    unique, counts = np.unique(data, return_counts=True)
    data = data[~np.isin(data, unique[counts > 12])]
    if data.size < 2:
        return np.nan

    # Normal-water-level surface area
    fld_area = float(np.percentile(data, percentile))
    area_max = float(np.max(data))

    # ---- Read ReGeom bathymetry ----
    regeom = np.genfromtxt(regeom_path, delimiter=",", skip_header=8, ndmin=2)
    if len(regeom) <= 1:
        return np.nan
    area, storage = regeom[:, 1], regeom[:, 2]

    # Adjust GRSAD area to ReGeom scale
    fld_area = fld_area * area[-1] / area_max

    # Linear interpolation of storage at normal-water-level area
    use_sto = np.nan
    for i in range(len(regeom)):
        rg_area = area[i]
        if rg_area < fld_area:
            continue
        elif rg_area == fld_area:
            use_sto = float(np.mean(storage[area == fld_area]))
            break
        else:
            if i == 0:
                use_sto = storage[0]
                break
            sto_max = storage[i]
            area_hi = area[i]
            sto_min = storage[i - 1]
            area_lo = area[i - 1]
            if area_hi == area_lo:
                use_sto = sto_min
            else:
                use_sto = sto_min + (sto_max - sto_min) * (fld_area - area_lo) / (
                    area_hi - area_lo
                )
            break

    if np.isnan(use_sto):
        return np.nan

    # Adjust to GRanD total capacity
    regeom_total = storage[-1]
    if regeom_total > 0:
        use_sto = use_sto * total_cap_mcm / regeom_total

    use_sto = min(use_sto, total_cap_mcm)
    fld_sto = total_cap_mcm - use_sto
    return max(fld_sto, 0.0)


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


def _write_dam_csv(
    path: Path,
    dam_ids: np.ndarray,
    names: list[str],
    lats: np.ndarray,
    lons: np.ndarray,
    upareas: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    fld_vol: np.ndarray,
    con_vol: np.ndarray,
    tot_vol: np.ndarray,
    qn: np.ndarray,
    qf: np.ndarray,
    years: np.ndarray,
) -> None:
    """Write dam parameters in CaMa-Flood CSV format.

    Format
    ------
    Line 1:  NDAM
    Line 2:  header
    Line 3+: GRAND_ID DamName DamLat DamLon area_CaMa DamIX DamIY
             FldVol_mcm ConVol_mcm TotalVol_mcm Qn Qf year
    """
    n_dam = len(dam_ids)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"{n_dam}\n")
        writer = csv.writer(f)
        writer.writerow(
            (
                "GRAND_ID",
                "DamName",
                "DamLat",
                "DamLon",
                "area_CaMa",
                "DamIX",
                "DamIY",
                "FldVol_mcm",
                "ConVol_mcm",
                "TotalVol_mcm",
                "Qn",
                "Qf",
                "year",
            )
        )
        for i in range(n_dam):
            writer.writerow(
                (
                    dam_ids[i],
                    names[i],
                    f"{lats[i]:.4f}",
                    f"{lons[i]:.4f}",
                    f"{upareas[i]:.2f}",
                    ix[i],
                    iy[i],
                    f"{fld_vol[i]:.4f}",
                    f"{con_vol[i]:.4f}",
                    f"{tot_vol[i]:.4f}",
                    f"{qn[i]:.4f}",
                    f"{qf[i]:.4f}",
                    years[i],
                )
            )


# ---------------------------------------------------------------------------
# NetCDF writer
# ---------------------------------------------------------------------------


def _write_dam_to_nc(
    nc_path: Path,
    param_cids: np.ndarray,
    map_shape: tuple[int, int],
    dam_cids: np.ndarray,
    dam_ids: np.ndarray,
    fld_vol_m3: np.ndarray,
    con_vol_m3: np.ndarray,
    tot_vol_m3: np.ndarray,
    qn: np.ndarray,
    qf: np.ndarray,
    res_area: np.ndarray,
    verbose: bool = True,
) -> None:
    """Append / update reservoir variables in an existing parameters.nc.

    Variables written (reservoir-indexed):
        reservoir_id, reservoir_catchment_id,
        reservoir_capacity, conservation_volume, emergency_volume,
        normal_outflow, flood_control_outflow, reservoir_area.
    """
    n_res = len(dam_ids)

    # Emergency volume = ConVol + FldVol * 0.95.
    eme_vol = con_vol_m3 + fld_vol_m3 * 0.95

    with Dataset(str(nc_path), "r+") as ds:
        # Create reservoir dimension (replace if already exists)
        if "reservoir" in ds.dimensions:
            existing = len(ds.dimensions["reservoir"])
            if existing != n_res:
                raise ValueError(
                    f"Existing reservoir dimension has size {existing}, "
                    f"but {n_res} reservoirs were provided"
                )
        else:
            ds.createDimension("reservoir", n_res)

        def _put(
            name: str,
            data: np.ndarray,
            dtype: str,
            units: str = "",
            long_name: str = "",
        ) -> None:
            if name in ds.variables:
                ds.variables[name][:] = data
            else:
                v = ds.createVariable(
                    name,
                    dtype,
                    ("reservoir",),
                    zlib=True,
                    complevel=4,
                )
                v[:] = data
                if units:
                    v.setncattr("units", units)
                if long_name:
                    v.setncattr("long_name", long_name)

        _put(
            "reservoir_id",
            np.asarray(dam_ids, dtype=np.int64),
            "i8",
            long_name="GRanD reservoir identifier",
        )
        _put(
            "reservoir_catchment_id",
            dam_cids,
            "i8",
            long_name="catchment id of this reservoir",
        )
        _put("reservoir_capacity", tot_vol_m3, "f8", "m3", "total reservoir capacity")
        _put(
            "conservation_volume",
            con_vol_m3,
            "f8",
            "m3",
            "conservation (normal-use) storage",
        )
        _put("emergency_volume", eme_vol, "f8", "m3", "emergency storage threshold")
        _put("normal_outflow", qn, "f8", "m3/s", "mean annual outflow (Qn)")
        _put("flood_control_outflow", qf, "f8", "m3/s", "flood control outflow (Qf)")
        _put("reservoir_area", res_area, "f8", "m2", "reservoir surface area")

    if verbose:
        print(f"[dam_params] Written {n_res} reservoirs to {nc_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _read_nc_catchment_ids(ds: Dataset) -> np.ndarray:
    """Read catchment IDs from a NC dataset."""
    if "catchment_id" in ds.variables:
        return np.asarray(ds.variables["catchment_id"][:]).astype(np.int64)
    raise KeyError(
        f"NC has no 'catchment_id' variable. Variables: {list(ds.variables.keys())}"
    )


def read_annual_stats(
    base_path: str | Path,
    catchment_ids: np.ndarray,
    max_name: str,
    mean_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Read standard CaMa annual output in parameter catchment order.

    The caller supplies complete annual max/mean products from the same run.
    Rank-file schema and live file checks belong to MultiRankStatsReader.
    A merged file contains both variables; rank outputs are passed as a directory.
    """
    path = Path(base_path)
    if path.is_file():
        with Dataset(path, "r") as dataset:
            indices = find_indices_in(catchment_ids, _read_nc_catchment_ids(dataset))
            if np.any(indices < 0):
                raise ValueError("parameter catchments are missing from annual output")
            return tuple(
                np.ma.asarray(
                    read_netcdf_values(dataset.variables[name]), dtype=np.float64
                ).filled(np.nan)[:, indices]
                for name in (max_name, mean_name)
            )
    series = []
    for name in (max_name, mean_name):
        reader = MultiRankStatsReader(
            base_dir=path,
            var_name=name,
            coord_name="catchment_id",
            split_by_year=any(path.glob(f"{name}_rank0_*.nc")),
        )
        try:
            series.append(reader.get_series(catchment_ids, dtype=np.float64))
        finally:
            reader.close()
    return series[0], series[1]


def compute_dam_discharge_from_timeseries(
    dam_list: str | Path,
    parameter_nc: str | Path,
    outflow_stats_nc: str | Path,
    *,
    dam_alloc: str | Path | np.ndarray | None = None,
    annual_max_var: str = "total_outflow_max_mean",
    annual_mean_var: str = "total_outflow_mean_mean",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Read pre-aggregated annual-max and mean discharge at dam cells.

    Expects NetCDF file(s) produced by :class:`StatisticsRuntime` with
    yearly ``max_mean`` and ``mean_mean`` operations, e.g.::

        variables_to_save = {
            "max_mean": ["total_outflow"],
            "mean_mean": ["total_outflow"],
        }

    Pass one complete merged file containing both variables, or a directory
    of formal rank-output files. All declared years and spatial ranks are read.
    Both series must be complete annual products from the same run, with matching
    rows. Annual aggregation is configured by the producer.

    Parameters
    ----------
    dam_list : path
        GRanD dam-list CSV (e.g. ``GRanD_allocated.csv``).
    parameter_nc : path
        ``parameters.nc``.
    outflow_stats_nc : path
        Aggregator output NetCDF **file or directory** containing yearly
        statistics.
    dam_alloc : path or structured array
        Dam allocation result.  Either a structured array from
        ``DamAllocMixin.dam_results_as_structured_array()`` or a path
        to the allocation output text file from
        ``DamAllocMixin.write_dam_alloc_file()``.
    annual_max_var : str
        Variable name for annual maximum (default ``"total_outflow_max_mean"``).
    annual_mean_var : str
        Variable name for annual mean (default ``"total_outflow_mean_mean"``).
    verbose : bool
        Print progress.

    Returns
    -------
    annual_max : (Y, D) float64
        Annual maximum outflow for each year × dam.
    qn : (D,) float64
        Long-term mean discharge at each dam (m³/s).
    dam_cids : (D,) int64
        Catchment IDs for each dam.
    dam_info : dict
        Parsed dam-list metadata.
    """
    dam_list = Path(dam_list)
    parameter_nc = Path(parameter_nc)
    outflow_stats_nc = Path(outflow_stats_nc)

    if not dam_list.exists():
        raise FileNotFoundError(f"Dam list not found: {dam_list}")
    if not parameter_nc.exists():
        raise FileNotFoundError(f"Parameter file not found: {parameter_nc}")
    if not outflow_stats_nc.exists():
        raise FileNotFoundError(f"Outflow stats file not found: {outflow_stats_nc}")

    # ---- Load dam list and grid info ----
    dam_info = _load_dam_list(dam_list)
    n_dam = len(dam_info["ids"])

    with Dataset(str(parameter_nc), "r") as ds:
        param_cids = _read_nc_catchment_ids(ds)
        nx = int(ds.getncattr("nx"))
        ny = int(ds.getncattr("ny"))

    if dam_alloc is None:
        raise ValueError(
            "dam_alloc is required: pass either a structured array from "
            "DamAllocMixin.dam_results_as_structured_array() or a path to "
            "the alloc output file from DamAllocMixin.write_dam_alloc_file()."
        )
    dam_cids = _resolve_dam_catchment_ids(dam_info, dam_alloc)

    if verbose:
        n_ok = int((dam_cids >= 0).sum())
        print(f"[dam_params] Loaded {n_dam} dams, grid ({nx}×{ny}), {n_ok} allocated")

    # Read standard annual products in parameter catchment order.
    max_data, mean_data = read_annual_stats(
        outflow_stats_nc, param_cids, annual_max_var, annual_mean_var,
    )
    n_years = max_data.shape[0]

    # ---- Map dams to aggregator catchment indices ----
    dam_indices = find_indices_in(dam_cids, param_cids).astype(np.int64, copy=False)
    valid = dam_indices >= 0

    # ---- Extract at dam cells (Numba-parallel) ----
    annual_max_all, qn = _extract_dam_stats(
        max_data,
        mean_data,
        dam_indices,
        dam_indices,
        valid,
    )

    if verbose:
        n_valid = int(valid.sum())
        print(
            f"[dam_params] Matched {n_valid}/{n_dam} dams "
            f"({n_years} years of aggregated statistics)"
        )
        if n_valid > 0:
            qn_pos = qn[valid]
            print(
                f"[dam_params] Qn stats (m³/s): "
                f"min={qn_pos.min():.2f}, "
                f"median={np.median(qn_pos):.2f}, "
                f"max={qn_pos.max():.2f}"
            )

    return annual_max_all, qn, dam_cids, dam_info


def estimate_flood_discharge(
    annual_max: np.ndarray,
    qn: np.ndarray,
    *,
    qf_ratio: float = 0.3,
    verbose: bool = True,
) -> np.ndarray:
    """Compute flood-control discharge Qf via Gumbel 100-yr fitting.

    Parameters
    ----------
    annual_max : (Y, D) float64
        Annual-maximum outflow per year per dam.
    qn : (D,) float64
        Mean annual outflow per dam.
    qf_ratio : float
        Qf = qf_ratio × Q100 (default 0.3).
    verbose : bool
        Print progress.

    Returns
    -------
    qf : (D,) float64
        Flood-control outflow for each dam (m³/s).
    """
    # ---- Numba-parallel Gumbel batch ----
    qf = _estimate_qf_batch(annual_max, qn, qf_ratio)
    n_dam = annual_max.shape[1]

    if verbose:
        valid = np.isfinite(qf)
        n_ok = int(valid.sum())
        print(f"[dam_params] Gumbel 100-yr fitting: {n_ok}/{n_dam} dams successful")
        if n_ok > 0:
            print(
                f"[dam_params] Qf stats (m³/s): "
                f"min={qf[valid].min():.2f}, "
                f"median={np.median(qf[valid]):.2f}, "
                f"max={qf[valid].max():.2f}"
            )

    return qf


def estimate_dam_params(
    dam_list: str | Path,
    parameter_nc: str | Path,
    outflow_stats_nc: str | Path,
    *,
    dam_alloc: str | Path | np.ndarray | None = None,
    output_csv: str | Path | None = None,
    output_nc: str | Path | None = None,
    annual_max_var: str = "total_outflow_max_mean",
    annual_mean_var: str = "total_outflow_mean_mean",
    qf_ratio: float = 0.3,
    flood_storage_ratio: float = 0.37,
    min_uparea: float = 0.0,
    grsad_dir: str | Path | None = None,
    regeom_dir: str | Path | None = None,
    grsad_percentile: float = 75.0,
    verbose: bool = True,
) -> Path:
    """Estimate dam / reservoir parameters and write CSV and/or NetCDF.

    Reads pre-aggregated yearly outflow statistics from a
    :class:`StatisticsRuntime` output NetCDF and runs the Gumbel 100-yr
    pipeline to compute Qn (mean discharge) and Qf (flood-control
    discharge) at each dam grid cell.

    The aggregator should be configured with yearly output and::

        variables_to_save = {
            "max_mean": ["total_outflow"],
            "mean_mean": ["total_outflow"],
        }

    Parameters
    ----------
    dam_list : path
        GRanD dam-list CSV (e.g. ``GRanD_allocated.csv``).
    parameter_nc : path
        ``parameters.nc`` with ``catchment_id``, ``nx``, ``ny``.
    outflow_stats_nc : path
        Aggregator output NetCDF with yearly ``max_mean`` and ``mean_mean``
        statistics.
    dam_alloc : path or structured array
        Dam allocation result.  Either a structured array from
        ``DamAllocMixin.dam_results_as_structured_array()`` or a path
        to the allocation output text file from
        ``DamAllocMixin.write_dam_alloc_file()``.
    output_csv : path or None
        CaMa-Flood-compatible CSV output path.
    output_nc : path or None
        NetCDF to update with reservoir variables.  If same as
        *parameter_nc*, updates in-place; otherwise copies first.
    annual_max_var : str
        Variable name for annual maximum in *outflow_stats_nc*.
    annual_mean_var : str
        Variable name for annual mean in *outflow_stats_nc*.
    qf_ratio : float
        ``Qf = qf_ratio × Q100`` (default 0.3).
    flood_storage_ratio : float
        Fraction of total capacity assigned to flood storage when GRSAD/
        ReGeom data are not available (default 0.37).
    min_uparea : float
        Minimum upstream area (km²) to keep a dam (default 0 = keep all).
    grsad_dir : path or None
        GRSAD surface-area time-series directory.
    regeom_dir : path or None
        ReGeom reservoir bathymetry directory.
    grsad_percentile : float
        Percentile for normal water level (default 75).
    verbose : bool
        Print progress.

    Returns
    -------
    Path
        Path to the primary output file (CSV or NC).
    """
    if output_csv is None and output_nc is None:
        raise ValueError(
            "At least one of 'output_csv' or 'output_nc' must be provided."
        )

    parameter_nc = Path(parameter_nc)

    # ------------------------------------------------------------------
    # 1. Extract discharge at dam cells
    # ------------------------------------------------------------------
    annual_max, qn, dam_cids, dam_info = compute_dam_discharge_from_timeseries(
        dam_list=dam_list,
        parameter_nc=parameter_nc,
        outflow_stats_nc=outflow_stats_nc,
        dam_alloc=dam_alloc,
        annual_max_var=annual_max_var,
        annual_mean_var=annual_mean_var,
        verbose=verbose,
    )
    # Gumbel 100-yr → Qf
    qf = estimate_flood_discharge(
        annual_max,
        qn,
        qf_ratio=qf_ratio,
        verbose=verbose,
    )

    n_dam = len(dam_info["ids"])

    # ------------------------------------------------------------------
    # 2. Estimate flood / conservation storage
    # ------------------------------------------------------------------
    cap_mcm = dam_info["cap_mcm"]
    fld_vol_mcm = np.full(n_dam, np.nan, dtype=np.float64)

    if grsad_dir is not None and regeom_dir is not None:
        grsad_dir = Path(grsad_dir)
        regeom_dir = Path(regeom_dir)
        if verbose:
            print("[dam_params] Estimating flood storage from GRSAD + ReGeom")
        for d in range(n_dam):
            fld_vol_mcm[d] = _estimate_flood_storage_grsad(
                dam_id=int(dam_info["ids"][d]),
                total_cap_mcm=float(cap_mcm[d]),
                grsad_dir=grsad_dir,
                regeom_dir=regeom_dir,
                percentile=grsad_percentile,
            )

    # Fallback for NaN / missing
    missing = np.isnan(fld_vol_mcm) | (fld_vol_mcm < 0)
    n_fallback = int(missing.sum())
    fld_vol_mcm[missing] = cap_mcm[missing] * flood_storage_ratio
    con_vol_mcm = cap_mcm - fld_vol_mcm

    if verbose and n_fallback > 0:
        print(
            f"[dam_params] Used {flood_storage_ratio:.0%} fallback "
            f"for {n_fallback}/{n_dam} dams"
        )

    # ------------------------------------------------------------------
    # 3. Filter by minimum upstream area
    # ------------------------------------------------------------------
    area_col = dam_info.get("area_cama", dam_info["upareas"])
    keep = area_col >= min_uparea
    if not keep.all() and verbose:
        print(
            f"[dam_params] Filtered {int((~keep).sum())} dams "
            f"with upstream area < {min_uparea} km²"
        )

    # ------------------------------------------------------------------
    # 4. De-duplicate dams on same grid cell
    # ------------------------------------------------------------------
    keep_dedup = _deduplicate_dams(
        dam_info["ids"],
        dam_cids,
        cap_mcm,
        verbose=verbose,
    )
    keep = keep & keep_dedup & (dam_cids >= 0)

    # Remove NaN discharge
    keep = keep & np.isfinite(qn) & np.isfinite(qf)

    # ------------------------------------------------------------------
    # 5. Filter dams outside the local parameter domain
    # ------------------------------------------------------------------
    with Dataset(str(parameter_nc), "r") as ds:
        param_cids = np.asarray(ds.variables["catchment_id"][:]).astype(np.int64)

    in_domain = np.isin(dam_cids, param_cids)
    n_out = int((keep & ~in_domain).sum())
    keep = keep & in_domain
    if verbose and n_out > 0:
        print(
            f"[dam_params] Filtered {n_out} dams outside local domain "
            f"({len(param_cids)} catchments)"
        )

    n_kept = int(keep.sum())
    if verbose:
        print(f"[dam_params] Keeping {n_kept}/{n_dam} dams after filtering")

    if n_kept == 0:
        raise RuntimeError("No dams remaining after filtering.")

    # Apply filter
    dam_ids_k = dam_info["ids"][keep]
    names_k = [dam_info["names"][i] for i in range(n_dam) if keep[i]]
    lats_k = dam_info["lats"][keep]
    lons_k = dam_info["lons"][keep]
    upareas_k = area_col[keep]
    ix_k = dam_info["ix"][keep]  # 1-based
    iy_k = dam_info["iy"][keep]  # 1-based
    fld_k = fld_vol_mcm[keep]
    con_k = con_vol_mcm[keep]
    tot_k = cap_mcm[keep]
    qn_k = qn[keep]
    qf_k = qf[keep]
    years_k = dam_info["years"][keep]
    cids_k = dam_cids[keep]

    # ------------------------------------------------------------------
    # 5. Write CSV output
    # ------------------------------------------------------------------
    primary_output = None

    if output_csv is not None:
        csv_path = Path(output_csv)
        _write_dam_csv(
            csv_path,
            dam_ids_k,
            names_k,
            lats_k,
            lons_k,
            upareas_k,
            ix_k,
            iy_k,
            fld_k,
            con_k,
            tot_k,
            qn_k,
            qf_k,
            years_k,
        )
        if verbose:
            print(f"[dam_params] Wrote CSV: {csv_path} ({n_kept} dams)")
        primary_output = csv_path

    # ------------------------------------------------------------------
    # 6. Write NetCDF output
    # ------------------------------------------------------------------
    if output_nc is not None:
        nc_path = Path(output_nc)
        if nc_path != parameter_nc:
            nc_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(parameter_nc, nc_path)
            if verbose:
                print(f"[dam_params] Copied {parameter_nc.name} → {nc_path}")

        with Dataset(str(parameter_nc), "r") as ds:
            param_cids = np.asarray(ds.variables["catchment_id"][:]).astype(np.int64)
            nx = int(ds.getncattr("nx"))
            ny = int(ds.getncattr("ny"))

        # Convert MCM → m³
        fld_m3 = fld_k * 1.0e6
        con_m3 = con_k * 1.0e6
        tot_m3 = tot_k * 1.0e6
        res_area = np.zeros(n_kept, dtype=np.float64)

        _write_dam_to_nc(
            nc_path,
            param_cids,
            (nx, ny),
            cids_k,
            dam_ids_k,
            fld_m3,
            con_m3,
            tot_m3,
            qn_k,
            qf_k,
            res_area,
            verbose=verbose,
        )
        if primary_output is None:
            primary_output = nc_path

    if primary_output is None:
        raise RuntimeError(
            "No output file was generated (neither CSV nor NC was requested)."
        )

    return primary_output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example: estimate dam parameters from aggregated outflow statistics
    estimate_dam_params(
        dam_list="inp/GRanD_allocated.csv",
        dam_alloc="inp/dam_alloc.txt",
        parameter_nc="inp/glb_15min/parameters.nc",
        outflow_stats_nc="out/total_outflow_max_mean_rank0.nc",
        output_csv="output/dam_params.csv",
        output_nc="inp/glb_15min/parameters.nc",
    )
