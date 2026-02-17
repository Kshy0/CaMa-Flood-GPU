# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from datetime import datetime, timedelta
from typing import Callable, Optional, Union

import cftime
import numpy as np

from cmfgpu.datasets.netcdf_dataset import NetCDFDataset
from cmfgpu.datasets.utils import monthly_time_to_key


class ERA5LandDataset(NetCDFDataset):
    """
    ERA5-Land dataset for accumulated (cumulative) variables such as hourly runoff.

    The caller supplies physical start_date and end_date.  Each time point t
    represents the runoff for the interval [t, t + Δt).  For example:
        start_date = datetime(2000, 1, 1)            # first interval: [00:00, 01:00)
        end_date   = datetime(2000, 12, 31, 23, 0)   # last  interval: [23:00, 00:00 next day)

    time_iter() and get_time_by_index() always return these physical (unshifted)
    times so that downstream code (e.g. current_time.hour == 0 for daily
    statistics) works correctly.

    Why we shift internally by +time_interval:
    ERA5-Land accumulated variables (e.g., hourly runoff `ro`) are time-stamped
    at the END of the accumulation period. Many preprocessed hourly files also
    store values as "cumulative since 00:00 UTC of the same day," with an
    important caveat:
      - At 00:00, the record stores the previous day's total (24h) accumulation.
      - The value at 01:00 represents the accumulation over [00:00, 01:00) of the new day.
      - The value at 02:00 represents the accumulation over [00:00, 02:00), and so on.

    When we want per-interval (hourly) increments aligned to [t, t+Δt), we need
    the cumulative value at (t+Δt). Therefore, internally the data-reading
    window is shifted forward by one time_interval. This shift is transparent
    to the caller.

    Note: because of the +Δt shift, reading the last physical time step may
    require a data file one interval beyond end_date. For example, with hourly
    data and end_date = datetime(2000, 12, 31, 23), the file runoff_2001_01.nc
    must exist and contain at least the 00:00 record.

    Example (Δt = 1 hour, units in mm):
      Cumulative (00:00 holds the previous day's 24h total):
        23:00 -> 10.0   (covers [00:00, 23:00) of the same day)
        00:00 -> 12.0   (yesterday's 24h total)
        01:00 -> 1.0    (new day: covers [00:00, 01:00))
      Desired hourly increments:
        [23:00, 00:00) -> 12.0 - 10.0 = 2.0
        [00:00, 01:00) -> 1.0

    Implementation outline (_transform_cumulative_to_incremental):
      1) inc[0] = arr[0]  -- first record is used as-is.
      2) inc[1:] = max(arr[1:] - arr[:-1], 0)  -- positive finite differences.
      3) At every day boundary (0::steps_per_day), the cumulative resets, so
         the raw value is the increment and is kept as-is.

    This keeps the output aligned with the physical interval [t, t+Δt) and avoids
    off-by-one mistakes caused by end-of-period time stamps and the 00:00 daily total.
    """
    def __init__(
        self,
        base_dir: str,
        start_date: datetime,
        end_date: datetime,
        time_interval: timedelta = timedelta(hours=1),
        chunk_len: int = 24,
        var_name: str = "ro",
        prefix: str = "runoff_",
        suffix: str = ".nc",
        time_to_key: Optional[Callable[[datetime], str]] = monthly_time_to_key,
        spin_up_start_date: Optional[datetime] = None,
        spin_up_end_date: Optional[datetime] = None,
        clip_incremental_negative: bool = True,
        *args,
        **kwargs,
    ):
        # Whether to clip negative increments to zero during cumulative-to-incremental
        # conversion.  True (default) is correct for non-negative fluxes like runoff.
        # Set to False for fluxes that can legitimately be negative.
        self.clip_incremental_negative = clip_incremental_negative

        # Configure time resolution first to derive daily step constraint
        self.num_daily_steps = int(86400 / time_interval.total_seconds())
        if int(chunk_len) <= 0 or (int(chunk_len) % self.num_daily_steps) != 0:
            raise ValueError(
                f"length must be a positive multiple of num_daily_steps ({self.num_daily_steps}), got {chunk_len}"
            )

        # ---- Store the original physical dates for time reporting ----
        self._physical_start_date = start_date
        self._physical_end_date = end_date
        self._physical_spin_up_start_date = spin_up_start_date
        self._physical_spin_up_end_date = spin_up_end_date
        self._era5_time_shift = time_interval   # the +Δt shift applied to data reading

        # Shift spin-up dates if provided, similar to main simulation dates
        if spin_up_start_date is not None:
            spin_up_start_date += time_interval
        if spin_up_end_date is not None:
            spin_up_end_date += time_interval

        super().__init__(
            base_dir=base_dir,
            start_date=start_date + time_interval,
            end_date=end_date + time_interval,
            time_interval=time_interval,
            chunk_len=chunk_len,
            var_name=var_name,
            prefix=prefix,
            suffix=suffix,
            time_to_key=time_to_key,
            spin_up_start_date=spin_up_start_date,
            spin_up_end_date=spin_up_end_date,
            *args,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Time reporting: return physical (unshifted) times
    # ------------------------------------------------------------------
    def get_time_by_index(self, idx: int) -> Union[datetime, cftime.datetime]:
        """Return the physical simulation time for step `idx`.

        The base class stores shifted dates (start_date + Δt) for internal
        data-reading purposes.  We subtract the shift here so that callers
        always see the original physical time.
        """
        shifted_time = super().get_time_by_index(idx)
        return shifted_time - self._era5_time_shift

    def get_index_by_time(self, dt: Union[datetime, cftime.datetime]) -> int:
        """Return the step index for a physical datetime."""
        return super().get_index_by_time(dt + self._era5_time_shift)

    def _transform_cumulative_to_incremental(self, arr: np.ndarray) -> np.ndarray:
        # Convert cumulative-per-day to hourly increments along time axis
        # Implement in NumPy to keep return type consistent
        steps_per_day = int(86400 // self.time_interval.total_seconds())
        if arr.shape[0] % steps_per_day != 0:
            raise ValueError(f"Data length {arr.shape[0]} is not a multiple of steps_per_day {steps_per_day}")
        inc = np.empty_like(arr)
        # First row as-is
        inc[0] = arr[0]
        diff = arr[1:] - arr[:-1]
        if self.clip_incremental_negative:
            np.maximum(diff, 0, out=diff)
        inc[1:] = diff
        # Reset at the start of each day to cumulative value
        inc[0::steps_per_day, :] = arr[0::steps_per_day, :]
        return inc

    def get_data(self, current_time: datetime, chunk_len: int) -> np.ndarray:
        arr = super().get_data(current_time, chunk_len)
        return self._transform_cumulative_to_incremental(arr)

    def read_chunk(self, idx: int) -> np.ndarray:
        arr = super().read_chunk(idx)
        return self._transform_cumulative_to_incremental(arr)


if __name__ == "__main__":
    resolution = "glb_06min"
    dataset = ERA5LandDataset(
        base_dir="/home/eat/ERA5_Runoff",
        start_date=datetime(2000, 1, 1),
        end_date=datetime(2000, 2, 1),
        prefix="runoff_",
        suffix=".nc",
        var_name="ro",
    )
    dataset.generate_runoff_mapping_table(
        map_dir=f"/home/eat/cmf_v420_pkg/map/{resolution}",
        out_dir=f"/home/eat/CaMa-Flood-GPU/inp/{resolution}",
        npz_file="runoff_mapping_era5.npz",
        parameter_nc=f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/parameters.nc",
    )
