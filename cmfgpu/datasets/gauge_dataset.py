"""Gauge observation dataset for CaMa-Flood-GPU inflow injection.

Loads a station-level gauge NetCDF written by ``s01_prepare_intervals.py``
with dims ``(time, station)`` and variables ``catchment_id_{resolution}``,
``reported_area_km2`` and ``observations`` (NaN for missing).  The only
downstream consumer is ``qualify_inflow`` which collapses stations to one
series per catchment and finds the first contiguous valid segment.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import numpy as np
import torch
from netCDF4 import Dataset as NCDataset
from netCDF4 import num2date


class GaugeDataset(torch.utils.data.Dataset):
    """Strict-schema gauge NetCDF loader.

    Required variables in *path*:

    - ``time`` (CF units) with dim ``time``.
    - ``catchment_id_{resolution}`` with dim ``station`` (int64).
    - ``reported_area_km2`` with dim ``station`` (float).
    - ``observations`` with dims ``(time, station)`` (float32, NaN = missing).

    No fallbacks are attempted.  Stations whose ``catchment_id`` is < 0
    are dropped as unallocated.
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        start_date: datetime,
        end_date: datetime,
        time_interval: timedelta,
        resolution: str,
    ):
        super().__init__()
        path = Path(path)

        cid_var = f"catchment_id_{resolution}"
        obs_var = "observations"
        area_var = "reported_area_km2"

        with NCDataset(str(path), "r") as ds:
            for v in (cid_var, obs_var, area_var, "time"):
                if v not in ds.variables:
                    raise ValueError(
                        f"{path}: missing required variable '{v}'. "
                        f"Available: {list(ds.variables)}"
                    )
            cids_all = np.asarray(ds.variables[cid_var][:], dtype=np.int64)
            area_all = np.asarray(
                ds.variables[area_var][:], dtype=np.float64)

            time_var = ds.variables["time"]
            raw_times = num2date(
                time_var[:], units=time_var.units,
                calendar=getattr(time_var, "calendar", "standard"),
            )
            times = np.array([
                datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
                for t in raw_times
            ])

            mask = (times >= start_date) & (times <= end_date)
            if not np.any(mask):
                raise ValueError(
                    f"No data in [{start_date}, {end_date}] in {path}")
            time_idx = np.where(mask)[0]

            raw = ds.variables[obs_var][time_idx, :]
            obs = raw.filled(np.nan) if isinstance(raw, np.ma.MaskedArray) \
                else np.asarray(raw, dtype=np.float32)
            obs = obs.astype(np.float32)

        allocated = cids_all >= 0
        n_total = len(cids_all)
        n_alloc = int(allocated.sum())
        print(f"[GaugeDataset] {n_alloc}/{n_total} stations allocated "
              f"on {resolution}")

        self._gauge_catchment_ids = cids_all[allocated]
        self._data = obs[:, allocated]
        self._gauge_area_km2 = area_all[allocated]
        self._time_interval = time_interval
        self._start_date = start_date
        self._end_date = end_date

    # ------------------------------------------------------------------
    @property
    def gauge_catchment_ids(self) -> np.ndarray:
        return self._gauge_catchment_ids

    @property
    def num_gauges(self) -> int:
        return len(self._gauge_catchment_ids)

    # ------------------------------------------------------------------
    def interp_small_gaps(self, max_gap: int) -> None:
        """Linearly interpolate NaN runs of length ``<= max_gap`` in-place.

        Gaps longer than ``max_gap`` or touching either end are left as NaN.
        """
        max_gap = int(max_gap)
        if max_gap <= 0:
            return
        data = self._data
        is_nan_any = np.isnan(data)
        if not is_nan_any.any():
            return
        n_t = data.shape[0]
        for g in range(data.shape[1]):
            col = data[:, g]
            is_nan = np.isnan(col)
            if not is_nan.any():
                continue
            diff = np.diff(
                np.concatenate(([False], is_nan, [False])).astype(np.int8))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for s, e in zip(starts, ends):
                if e - s > max_gap or s == 0 or e >= n_t:
                    continue
                left, right = col[s - 1], col[e]
                col[s:e] = np.linspace(left, right, e - s + 2)[1:-1]

    # ------------------------------------------------------------------
    def qualify_inflow(self, max_gap: int) -> dict:
        """Merge stations per catchment and qualify first contiguous window.

        Pipeline:

        1. :meth:`interp_small_gaps` in place.
        2. Area-weighted mm/day merge per catchment id, strict NaN: for
           every day where *any* contributor is NaN the merged value is
           NaN.  ``y_c(t) = Σ_i q_i(t) / A_i * 86400 * 1000`` (mm/day)
           and ``q_c(t) = y_c(t) * A_total_c / (86400 * 1000)`` recovers
           the m³/s aggregate.  A_total_c := Σ A_i.  When a catchment
           has only one contributor this reduces to the identity.
        3. Per merged column, take the **longest** contiguous non-NaN
           segment; ``basin_shift_days[c]`` = # leading invalid days before
           that segment, ``valid_length_days[c]`` = segment length.  Ties
           resolve to the earliest segment.

        Returns
        -------
        dict with keys
            ``catchment_ids`` : (N,) int64
            ``basin_shift_days`` : (N,) int64
                Number of leading NaN days before the first contiguous
                valid segment on the native observation axis.  Consumers
                feed this to :meth:`~cmfgpu.modules.base.BaseModule.
                catchment_shift_days` so that ExportedDataset / the
                inflow overlay read the gauge series at
                ``raw[t + shift[c], c]`` for simulation step *t*.
            ``valid_length_days`` : (N,) int64
                Length of the first contiguous valid segment.
            ``data`` : (T, N) float32
                Merged gauge discharge on the **native observation
                axis** (row *t* == calendar day *t*), NaN replaced by 0.
                Not pre-shifted; ExportedDataset is responsible for
                applying ``basin_shift_days`` at read time via the same
                ``catchment_shift_days`` used for runoff.
        """
        self.interp_small_gaps(int(max_gap))

        cids = self._gauge_catchment_ids.astype(np.int64)
        obs = self._data
        area = self._gauge_area_km2 * 1e6  # km² → m²

        unique_cid, inv = np.unique(cids, return_inverse=True)
        T = obs.shape[0]
        N = unique_cid.size

        obs64 = obs.astype(np.float64)
        nan_mask = np.isnan(obs64)

        area_total = np.zeros(N, dtype=np.float64)
        np.add.at(area_total, inv, area)
        if np.any(area_total <= 0):
            bad = unique_cid[area_total <= 0]
            raise ValueError(
                f"qualify_inflow: non-positive A_total for catchments "
                f"{bad.tolist()[:5]}")

        # Per-station mm/day (zero on NaN days; NaN flag tracked separately).
        yield_mm = np.where(
            nan_mask, 0.0, obs64 * 86400.0 * 1000.0 / area[None, :])
        merged_mm = np.zeros((T, N), dtype=np.float64)
        nan_any = np.zeros((T, N), dtype=bool)
        np.add.at(merged_mm, (slice(None), inv), yield_mm)
        np.logical_or.at(nan_any, (slice(None), inv), nan_mask)

        merged_q = merged_mm * area_total[None, :] / (86400.0 * 1000.0)

        shift = np.zeros(N, dtype=np.int64)
        length = np.zeros(N, dtype=np.int64)
        keep = np.ones(N, dtype=bool)
        for c in range(N):
            valid = ~nan_any[:, c]
            if not valid.any():
                keep[c] = False
                continue
            # Find longest run of True in `valid`; ties → earliest.
            padded = np.concatenate(([False], valid, [False]))
            diff = np.diff(padded.astype(np.int8))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            lengths = ends - starts
            k = int(np.argmax(lengths))
            shift[c] = int(starts[k])
            length[c] = int(lengths[k])

        if not keep.all():
            raise ValueError(
                f"qualify_inflow: {(~keep).sum()} catchments have no valid "
                f"data in [{self._start_date}, {self._end_date}]: "
                f"{unique_cid[~keep].tolist()[:10]}")

        # Replace NaN with 0 on native axis; ExportedDataset.attach_inflow
        # will apply the shift.
        data = np.where(nan_any, 0.0, merged_q).astype(np.float32)

        return {
            "catchment_ids": unique_cid,
            "basin_shift_days": shift,
            "valid_length_days": length,
            "data": data,
        }
