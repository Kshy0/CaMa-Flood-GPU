# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from datetime import datetime, timedelta
from typing import Callable, Optional

import numpy as np

from cmfgpu.datasets.netcdf_dataset import NetCDFDataset


def monthly_time_to_key(dt: datetime) -> str:
    dt = dt
    return dt.strftime("%Y_%m")

class ERA5LandDataset(NetCDFDataset):
    """
    Why we use `current_time + self.time_interval`:
    ERA5-Land accumulated variables (e.g., hourly runoff `ro`) are time-stamped at the
    END of the accumulation period. Many preprocessed hourly files also store values as
    "cumulative since 00:00 UTC of the same day," with an important caveat:
      - At 00:00, the record stores the previous day's total (24h) accumulation.
      - The value at 01:00 represents the accumulation over [00:00, 01:00) of the new day.
      - The value at 02:00 represents the accumulation over [00:00, 02:00), and so on.

    When we want per-interval (hourly) increments aligned to [t, t+Δt), we need the
    cumulative at (t+Δt). Therefore, we shift the request by one step:
        current_time = current_time + self.time_interval

    Example (Δt = 1 hour, units in mm):
      Cumulative (00:00 holds the previous day's 24h total):
        23:00 -> 10.0   (covers [00:00, 23:00) of the same day)
        00:00 -> 12.0   (yesterday's 24h total)
        01:00 -> 1.0    (new day: covers [00:00, 01:00))
      Desired hourly increments:
        [23:00, 00:00) -> 12.0 - 10.0 = 2.0
        [00:00, 01:00) -> 1.0

    Implementation outline:
      1) Read cumulative values starting from t+Δt, so the first returned row already
         corresponds to [t, t+Δt). We set inc[0] = arr[0].
      2) For subsequent steps, use positive difference: inc[1:] = max(arr[1:] - arr[:-1], 0).
      3) Because 00:00 holds the previous day's total while the new day restarts from small
         values, explicitly set day-start increments from the first hour of the day so the
         output aligns with [t, t+Δt).

    This keeps the output aligned with the physical interval [t, t+Δt) and avoids
    off-by-one mistakes caused by end-of-period time stamps and the 00:00 daily total.
    """
    def __init__(
        self,
        base_dir: str,
        start_date: datetime,
        end_date: datetime,
        unit_factor: float = 1.0, # mm/day divided by unit_factor to get m/s
        time_interval: timedelta = timedelta(hours=1),
        chunk_len: int = 24,
        var_name: str = "ro",
        prefix: str = "runoff_",
        suffix: str = ".nc",
        out_dtype: str = "float32",
        time_to_key: Optional[Callable[[datetime], str]] = monthly_time_to_key,
        *args,
        **kwargs,
    ):
        # Configure time resolution first to derive daily step constraint
        self.num_daily_steps = int(86400 / time_interval.total_seconds())
        if int(chunk_len) <= 0 or (int(chunk_len) % self.num_daily_steps) != 0:
            raise ValueError(
                f"length must be a positive multiple of num_daily_steps ({self.num_daily_steps}), got {chunk_len}"
            )

        super().__init__(
            base_dir=base_dir,
            start_date=start_date + time_interval,
            end_date=end_date + time_interval,
            unit_factor=unit_factor,
            time_interval=time_interval,
            var_name=var_name,
            prefix=prefix,
            suffix=suffix,
            out_dtype=out_dtype,
            chunk_len=chunk_len,
            time_to_key=time_to_key,
            *args, **kwargs,
        )

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
    )
