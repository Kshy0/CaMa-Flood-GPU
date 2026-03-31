"""NetCDF-based gauge inflow dataset."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import numpy as np
from netCDF4 import Dataset as NCDataset
from netCDF4 import num2date

from cmfgpu.datasets.abstract_gauge import AbstractGaugeDataset


class NetCDFGaugeDataset(AbstractGaugeDataset):
    r"""Gauge dataset backed by a NetCDF file.

    Expected NetCDF structure:

    * Dimensions: ``time``, ``gauge``
    * Variables:

      - ``time`` *(time,)* — numeric with ``units`` and ``calendar`` attrs
      - ``catchment_id`` *(gauge,)* — integer catchment IDs
      - ``discharge`` *(time, gauge)* — discharge in **m³ s⁻¹**

    Variable and coordinate names are configurable.

    Parameters
    ----------
    path : str or Path
        Path to the NetCDF file.
    start_date, end_date : datetime
        Inclusive time range to extract.
    time_interval : timedelta
        Expected time step (must match the runoff dataset).
    chunk_len : int
        Chunk length (must match the runoff dataset).
    num_prepend_chunks : int
        Zero-filled chunks to prepend (set to
        ``runoff_dataset.num_spin_up_chunks``).
    var_name : str
        Name of the discharge variable.
    coord_name : str
        Name of the catchment-ID coordinate variable.
    """

    # ------------------------------------------------------------------
    @staticmethod
    def _load_nc(
        path: Path,
        start_date: datetime,
        end_date: datetime,
        time_interval: timedelta,
        var_name: str,
        coord_name: str,
    ):
        with NCDataset(str(path), "r") as ds:
            # --- catchment IDs ---
            if coord_name not in ds.variables:
                raise ValueError(
                    f"Coordinate variable '{coord_name}' not found in {path}. "
                    f"Available: {list(ds.variables.keys())}"
                )
            cids_raw = ds.variables[coord_name][:]
            cids = (
                cids_raw.filled(0) if isinstance(cids_raw, np.ma.MaskedArray)
                else np.asarray(cids_raw)
            ).astype(np.int64)

            # --- time axis ---
            time_var = ds.variables["time"]
            units = time_var.units
            calendar = getattr(time_var, "calendar", "standard")
            raw_times = num2date(time_var[:], units=units, calendar=calendar)
            times = np.array(
                [
                    datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
                    for t in raw_times
                ]
            )

            # Filter to [start_date, end_date]
            mask = (times >= start_date) & (times <= end_date)
            if not np.any(mask):
                raise ValueError(
                    f"No gauge data found in [{start_date}, {end_date}] in {path}"
                )
            time_idx = np.where(mask)[0]

            # --- discharge ---
            if var_name not in ds.variables:
                raise ValueError(
                    f"Variable '{var_name}' not found in {path}. "
                    f"Available: {list(ds.variables.keys())}"
                )
            raw = ds.variables[var_name][time_idx, :]
            if isinstance(raw, np.ma.MaskedArray):
                raw = raw.filled(0.0)
            data = np.asarray(raw, dtype=np.float32)

        expected_steps = int(
            (end_date - start_date).total_seconds()
            / time_interval.total_seconds()
        ) + 1
        if data.shape[0] != expected_steps:
            print(
                f"[NetCDFGaugeDataset] Warning: expected {expected_steps} steps "
                f"but file has {data.shape[0]} in the requested range"
            )

        return cids, data

    def __init__(
        self,
        path: Union[str, Path],
        *,
        start_date: datetime,
        end_date: datetime,
        time_interval: timedelta,
        chunk_len: int = 24,
        num_prepend_chunks: int = 0,
        var_name: str = "discharge",
        coord_name: str = "catchment_id",
    ):
        ids, data = self._load_nc(
            Path(path), start_date, end_date, time_interval, var_name, coord_name
        )
        super().__init__(
            ids, data,
            chunk_len=chunk_len,
            num_prepend_chunks=num_prepend_chunks,
        )
