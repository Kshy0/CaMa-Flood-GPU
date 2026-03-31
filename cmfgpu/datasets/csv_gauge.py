"""CSV-based gauge inflow dataset."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from cmfgpu.datasets.abstract_gauge import AbstractGaugeDataset


class CSVGaugeDataset(AbstractGaugeDataset):
    r"""Gauge dataset backed by a wide-format CSV file.

    Expected CSV layout::

        datetime,12345,67890,11111
        2000-01-01,100.5,200.3,50.0
        2000-01-02,105.2,198.7,52.1
        ...

    * First column (configurable via *datetime_col*) holds timestamps.
    * Remaining column headers are **integer catchment IDs**.
    * Values are discharge in **m³ s⁻¹**.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    start_date, end_date : datetime
        Inclusive time range to extract.
    time_interval : timedelta
        Expected time step (must match the runoff dataset).
    chunk_len : int
        Chunk length (must match the runoff dataset).
    num_prepend_chunks : int
        Zero-filled chunks to prepend (set to
        ``runoff_dataset.num_spin_up_chunks``).
    datetime_col : str
        Name of the datetime column.
    """

    # ------------------------------------------------------------------
    @staticmethod
    def _load_csv(
        path: Path,
        start_date: datetime,
        end_date: datetime,
        time_interval: timedelta,
        datetime_col: str,
    ):
        df = pd.read_csv(path, parse_dates=[datetime_col])
        df = df.sort_values(datetime_col).reset_index(drop=True)

        # Filter to [start_date, end_date]
        mask = (df[datetime_col] >= pd.Timestamp(start_date)) & (
            df[datetime_col] <= pd.Timestamp(end_date)
        )
        df = df.loc[mask].reset_index(drop=True)

        if df.empty:
            raise ValueError(
                f"No gauge data found in [{start_date}, {end_date}] in {path}"
            )

        catchment_cols = [c for c in df.columns if c != datetime_col]
        catchment_ids = np.array([int(c) for c in catchment_cols], dtype=np.int64)
        data = df[catchment_cols].values.astype(np.float32)

        expected_steps = int(
            (end_date - start_date).total_seconds()
            / time_interval.total_seconds()
        ) + 1
        if data.shape[0] != expected_steps:
            print(
                f"[CSVGaugeDataset] Warning: expected {expected_steps} steps "
                f"but CSV has {data.shape[0]} rows in the requested range"
            )

        return catchment_ids, data

    def __init__(
        self,
        path: Union[str, Path],
        *,
        start_date: datetime,
        end_date: datetime,
        time_interval: timedelta,
        chunk_len: int = 24,
        num_prepend_chunks: int = 0,
        datetime_col: str = "datetime",
    ):
        ids, data = self._load_csv(
            Path(path), start_date, end_date, time_interval, datetime_col
        )
        super().__init__(
            ids, data,
            chunk_len=chunk_len,
            num_prepend_chunks=num_prepend_chunks,
        )
