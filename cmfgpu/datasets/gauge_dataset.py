"""Gauge observation dataset for CaMa-Flood-GPU inflow injection.

Loads a station-level gauge NetCDF written by ``s01_prepare_intervals.py``
with dims ``(time, station)`` and variables ``catchment_id_{resolution}``,
``reported_area_km2`` and ``discharge`` (NaN for missing).  The only
downstream consumer is ``qualify_inflow`` which collapses stations to one
series per catchment and finds the longest contiguous valid segment.
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
    - ``discharge`` with dims ``(time, station)`` (float32, NaN = missing).

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
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            raise TypeError("start_date and end_date must be datetime values")
        if end_date < start_date:
            raise ValueError("end_date must not precede start_date")
        if not isinstance(time_interval, timedelta) or time_interval <= timedelta(0):
            raise ValueError("time_interval must be a positive timedelta")
        duration = end_date - start_date
        if duration % time_interval != timedelta(0):
            raise ValueError(
                "[start_date, end_date] must align exactly to time_interval"
            )

        cid_var = f"catchment_id_{resolution}"
        obs_var = "discharge"
        area_var = "reported_area_km2"

        with NCDataset(str(path), "r") as ds:
            for v in (cid_var, obs_var, area_var, "time"):
                if v not in ds.variables:
                    raise ValueError(
                        f"{path}: missing required variable '{v}'. "
                        f"Available: {list(ds.variables)}"
                    )
            raw_cids = ds.variables[cid_var][:]
            cids_all = np.asarray(
                raw_cids.filled(-1)
                if isinstance(raw_cids, np.ma.MaskedArray) else raw_cids,
                dtype=np.int64,
            )
            raw_area = ds.variables[area_var][:]
            area_all = np.asarray(
                raw_area.filled(np.nan)
                if isinstance(raw_area, np.ma.MaskedArray) else raw_area,
                dtype=np.float64,
            )
            if cids_all.ndim != 1 or area_all.shape != cids_all.shape:
                raise ValueError(
                    f"{path}: catchment IDs and reported areas must be "
                    "one-dimensional arrays with identical shape"
                )

            time_var = ds.variables["time"]
            if time_var.dimensions != ("time",):
                raise ValueError(
                    f"{path}: time must have dimensions ('time',), got "
                    f"{time_var.dimensions}"
                )
            discharge_var = ds.variables[obs_var]
            if discharge_var.dimensions != ("time", "station"):
                raise ValueError(
                    f"{path}: discharge must have dimensions "
                    f"('time', 'station'), got {discharge_var.dimensions}"
                )
            if discharge_var.shape[1] != cids_all.size:
                raise ValueError(
                    f"{path}: discharge has {discharge_var.shape[1]} stations "
                    f"but {cid_var} has {cids_all.size}"
                )
            raw_times = num2date(
                time_var[:], units=time_var.units,
                calendar=getattr(time_var, "calendar", "standard"),
            )
            times = np.array([
                datetime(
                    t.year, t.month, t.day, t.hour, t.minute, t.second,
                    getattr(t, "microsecond", 0),
                )
                for t in raw_times
            ])

            mask = (times >= start_date) & (times <= end_date)
            if not np.any(mask):
                raise ValueError(
                    f"No data in [{start_date}, {end_date}] in {path}")
            time_idx = np.where(mask)[0]

            expected_count = duration // time_interval + 1
            expected_times = np.array(
                [start_date + i * time_interval for i in range(expected_count)]
            )
            selected_times = times[time_idx]
            if not np.array_equal(selected_times, expected_times):
                raise ValueError(
                    f"{path}: time axis in [{start_date}, {end_date}] must be "
                    f"continuous, unique, and spaced by {time_interval}; got "
                    f"{selected_times.size} samples, expected {expected_count}"
                )

            raw = discharge_var[time_idx, :]
            obs = raw.filled(np.nan) if isinstance(raw, np.ma.MaskedArray) \
                else np.asarray(raw, dtype=np.float32)
            obs = obs.astype(np.float32)
            obs[~np.isfinite(obs)] = np.nan

        allocated = cids_all >= 0
        n_total = len(cids_all)
        n_alloc = int(allocated.sum())
        if n_alloc == 0:
            raise ValueError(f"{path}: no stations are allocated on {resolution}")
        allocated_area = area_all[allocated]
        invalid_area = ~np.isfinite(allocated_area) | (allocated_area <= 0)
        if np.any(invalid_area):
            bad = np.flatnonzero(allocated)[invalid_area]
            raise ValueError(
                f"{path}: allocated stations must have finite positive "
                f"reported_area_km2; invalid station indices "
                f"{bad[:5].tolist()}"
            )
        print(f"[GaugeDataset] {n_alloc}/{n_total} stations allocated "
              f"on {resolution}")

        self._gauge_catchment_ids = cids_all[allocated]
        self._data = obs[:, allocated]
        self._gauge_area_km2 = allocated_area
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
            for s, e in zip(starts, ends, strict=True):
                if e - s > max_gap or s == 0 or e >= n_t:
                    continue
                left, right = col[s - 1], col[e]
                col[s:e] = np.linspace(left, right, e - s + 2)[1:-1]

    # ------------------------------------------------------------------
    def qualify_inflow(self, max_gap: int) -> dict:
        """Merge stations per catchment and qualify the longest valid window.

        Pipeline:

        1. :meth:`interp_small_gaps` in place.
        2. Sum discharge per catchment id, strict NaN: for every day where
           *any* contributor is NaN the merged value is NaN.  This is
           equivalent to taking an area-weighted mean of station runoff
           yields and multiplying by total contributing area, and therefore
           conserves ``q_c(t) = Σ_i q_i(t)``.
        3. Per merged column, take the **longest** contiguous non-NaN
           segment; ``basin_shift_days[c]`` = # leading invalid days before
           that segment, ``valid_length_days[c]`` = segment length.  Ties
           resolve to the earliest segment.

        Returns
        -------
        dict with keys
            ``catchment_ids`` : (N,) int64
            ``basin_shift_days`` : (N,) int64
                Number of leading NaN days before the longest contiguous
                valid segment on the native observation axis.  Consumers
                feed this to ``InflowModule.basin_shift_days`` so that
                ExportedDataset / the
                inflow overlay read the gauge series at
                ``raw[t + shift[c], c]`` for simulation step *t*.
            ``valid_length_days`` : (N,) int64
                Length of the longest contiguous valid segment.
            ``data`` : (T, N) float32
                Merged gauge discharge on the **native observation
                axis** (row *t* == calendar day *t*), retaining NaN for
                missing observations.  Keeping the validity marker allows
                ``ExportedDataset.attach_inflow_overlay`` to derive the same
                shift and valid length without a separate side channel.
        """
        self.interp_small_gaps(int(max_gap))

        cids = self._gauge_catchment_ids.astype(np.int64)
        obs = self._data
        area = self._gauge_area_km2
        invalid_area = ~np.isfinite(area) | (area <= 0)
        if np.any(invalid_area):
            raise ValueError(
                "qualify_inflow: every station area must be finite and positive; "
                f"invalid station indices {np.flatnonzero(invalid_area)[:5].tolist()}"
            )

        unique_cid, inv = np.unique(cids, return_inverse=True)
        T = obs.shape[0]
        N = unique_cid.size

        obs64 = obs.astype(np.float64)
        nan_mask = ~np.isfinite(obs64)

        merged_q = np.zeros((T, N), dtype=np.float64)
        nan_any = np.zeros((T, N), dtype=bool)
        np.add.at(merged_q, (slice(None), inv), np.where(nan_mask, 0.0, obs64))
        np.logical_or.at(nan_any, (slice(None), inv), nan_mask)

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

        # Preserve NaN until the overlay has derived its validity interval.
        data = np.where(nan_any, np.nan, merged_q).astype(np.float32)

        return {
            "catchment_ids": unique_cid,
            "basin_shift_days": shift,
            "valid_length_days": length,
            "data": data,
        }
