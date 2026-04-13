"""Unified gauge observation dataset for CaMa-Flood-GPU inflow injection.

Reads a gauge / inflow NetCDF file and caches the full observation matrix
in memory.  Two usage modes:

1. **Station-level gauge NC** (from ``s00_build_gauge_nc.py``):
   Dimensions ``(time, station)``, with ``catchment_id_{resolution}``
   per station and ``observations`` that may contain NaN.

2. **Inflow NC** (from ``t01_prepare_cama.py``):
   Dimensions ``(time, gauge)``, with ``catchment_id`` per inject point
   and ``discharge`` with NaN already filled to 0.

In both cases, inflow stations are selected at ``build_inflow_mapping``
time.

Data flow::

    gauge / inflow NC
            │
      GaugeDataset.__init__()
      ├── read catchment IDs, keep allocated (>= 0)
      ├── optionally fill NaN → 0 (fill_nan=True)
      └── cache in memory
            │
      build_inflow_mapping(desired_cids, full_cids, device)
      ├── pick columns matching desired inflow catchment IDs
      ├── validate: no NaN, no negatives, length check
      └── cache reorder + injection indices
            │
      shard_forcing(batch_gauge, batch_runoff)
      └── batch_runoff[:, inflow_idx] += gauge_flat
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from hydroforge.modeling.distributed import find_indices_in
from netCDF4 import Dataset as NCDataset
from netCDF4 import num2date


class GaugeDataset(torch.utils.data.Dataset):
    """Gauge observation dataset backed by a NetCDF file.

    Parameters
    ----------
    path : str or Path
        Path to gauge / inflow NetCDF.
    start_date, end_date : datetime
        Inclusive time range to extract.
    time_interval : timedelta
        Expected time step (must match the runoff dataset).
    resolution : str or None
        Map resolution tag (e.g. ``"glb_15min"``).  When set, reads
        ``catchment_id_{resolution}`` and ``observations``.
        When *None*, reads ``catchment_id`` and ``discharge``
        (inflow NC convention).
    fill_nan : bool
        If *True*, replace NaN with 0 after loading.  Useful for inflow
        injection where missing observations should inject nothing.
    chunk_len : int
        Chunk length (must match the runoff dataset).
    num_prepend_chunks : int
        Zero-filled chunks to prepend for spin-up alignment.
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        start_date: datetime,
        end_date: datetime,
        time_interval: timedelta,
        resolution: str | None = "glb_15min",
        fill_nan: bool = False,
        chunk_len: int = 24,
        num_prepend_chunks: int = 0,
    ):
        super().__init__()
        path = Path(path)

        # Determine variable names from resolution
        if resolution is not None:
            cid_var = f"catchment_id_{resolution}"
            obs_var = "observations"
        else:
            cid_var = "catchment_id"
            obs_var = "discharge"

        with NCDataset(str(path), "r") as ds:
            if cid_var not in ds.variables:
                raise ValueError(
                    f"Variable '{cid_var}' not found in {path}. "
                    f"Available: {list(ds.variables.keys())}"
                )
            cids_all = np.asarray(ds.variables[cid_var][:], dtype=np.int64)

            # Time axis
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
                    f"No data in [{start_date}, {end_date}] in {path}"
                )
            time_idx = np.where(mask)[0]

            # Read full observations for the time window
            raw = ds.variables[obs_var][time_idx, :]
            if isinstance(raw, np.ma.MaskedArray):
                obs = raw.filled(np.nan)
            else:
                obs = np.asarray(raw, dtype=np.float32)
            obs = obs.astype(np.float32)

        # Keep only stations allocated for this resolution
        allocated = cids_all >= 0
        n_total = len(cids_all)
        n_alloc = int(allocated.sum())
        res_label = resolution if resolution is not None else "inflow"
        print(
            f"[GaugeDataset] {n_alloc}/{n_total} stations allocated "
            f"on {res_label}"
        )

        self._gauge_catchment_ids = cids_all[allocated]
        self._data = obs[:, allocated]

        if fill_nan:
            n_nan = int(np.isnan(self._data).sum())
            if n_nan > 0:
                self._data = np.nan_to_num(self._data, nan=0.0)
                print(f"[GaugeDataset] Filled {n_nan} NaN values with 0")
        self.chunk_len = chunk_len
        self._num_prepend_chunks = num_prepend_chunks
        self._time_interval = time_interval
        self._start_date = start_date
        self._end_date = end_date

        # Set after build_inflow_mapping
        self._local_indices: Optional[np.ndarray] = None
        self._inflow_catchment_idx: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def gauge_catchment_ids(self) -> np.ndarray:
        """All allocated catchment IDs in this dataset, shape ``(G,)``."""
        return self._gauge_catchment_ids

    @property
    def num_gauges(self) -> int:
        return len(self._gauge_catchment_ids)

    # ------------------------------------------------------------------
    # Inflow mapping + validation
    # ------------------------------------------------------------------
    def build_inflow_mapping(
        self,
        desired_catchment_ids: np.ndarray,
        catchment_id: np.ndarray,
        device: torch.device,
    ) -> None:
        """Select inflow stations from the gauge pool and validate.

        Parameters
        ----------
        desired_catchment_ids : np.ndarray
            Inflow catchment IDs from ``parameters.nc``
            (``model.base.inflow_catchment_id.numpy()``).
        catchment_id : np.ndarray
            Full model catchment ID array
            (``model.base.catchment_id.numpy()``).
        device : torch.device
            Target device for the injection index tensor.

        Raises
        ------
        ValueError
            If any desired ID is missing, or selected columns contain
            NaN / negative values, or time length does not match.
        """
        desired = np.asarray(desired_catchment_ids, dtype=np.int64)

        # Column reorder: gauge-file order → desired order
        col_pos = find_indices_in(desired, self._gauge_catchment_ids)
        if np.any(col_pos == -1):
            missing = desired[col_pos == -1].tolist()
            raise ValueError(f"Gauge data missing catchment IDs: {missing}")

        # Validate selected columns
        selected = self._data[:, col_pos]

        n_nan = int(np.isnan(selected).sum())
        if n_nan > 0:
            raise ValueError(
                f"Selected inflow stations contain {n_nan} NaN values. "
                f"Only stations with complete coverage can be used for inflow."
            )

        n_neg = int((selected < 0).sum())
        if n_neg > 0:
            raise ValueError(
                f"Selected inflow stations contain {n_neg} negative values. "
                f"Discharge must be >= 0."
            )

        expected_steps = int(
            (self._end_date - self._start_date).total_seconds()
            / self._time_interval.total_seconds()
        ) + 1
        if self._data.shape[0] != expected_steps:
            print(
                f"[GaugeDataset] Warning: expected {expected_steps} time "
                f"steps but got {self._data.shape[0]}"
            )

        self._local_indices = col_pos.astype(np.int64)

        # Injection indices: desired → position in the full catchment array
        inflow_idx = find_indices_in(
            desired, np.asarray(catchment_id, dtype=np.int64)
        )
        if np.any(inflow_idx == -1):
            missing = desired[inflow_idx == -1].tolist()
            raise ValueError(
                f"Inflow catchment IDs not in model catchment array: {missing}"
            )
        self._inflow_catchment_idx = torch.from_numpy(
            inflow_idx.astype(np.int64)
        ).to(device)

        print(
            f"[GaugeDataset] Mapped {len(desired)} inflow gauges "
            f"from {self.num_gauges} in file"
        )

    # ------------------------------------------------------------------
    # Forcing injection
    # ------------------------------------------------------------------
    def shard_forcing(
        self,
        batch_data: torch.Tensor,
        batch_runoff: torch.Tensor,
    ) -> torch.Tensor:
        """Inject gauge discharge into the runoff tensor.

        Parameters
        ----------
        batch_data : torch.Tensor
            Gauge discharge from ``DataLoader``, shape ``(B, T, G)``.
        batch_runoff : torch.Tensor
            Runoff tensor, shape ``(B*T, C)``.

        Returns
        -------
        torch.Tensor
            Modified *batch_runoff* with gauge discharge added at
            inflow catchment locations.
        """
        if self._inflow_catchment_idx is None:
            raise RuntimeError(
                "build_inflow_mapping must be called before shard_forcing"
            )

        if batch_data.dim() == 3:
            B, T, G = batch_data.shape
            gauge_flat = batch_data.reshape(B * T, G).contiguous()
        else:
            raise ValueError(f"Expected 3D gauge tensor, got {batch_data.dim()}D")

        batch_runoff[:, self._inflow_catchment_idx] += gauge_flat
        return batch_runoff

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0:
            idx += len(self)

        G = (
            len(self._local_indices)
            if self._local_indices is not None
            else self.num_gauges
        )

        # Prepend chunks are all-zero (spin-up: no injection)
        if idx < self._num_prepend_chunks:
            return np.zeros((self.chunk_len, G), dtype=np.float32)

        real_idx = idx - self._num_prepend_chunks
        start = real_idx * self.chunk_len
        end = min(start + self.chunk_len, self._data.shape[0])
        chunk = self._data[start:end]

        if self._local_indices is not None:
            chunk = chunk[:, self._local_indices]

        T = chunk.shape[0]
        if T < self.chunk_len:
            pad = np.zeros((self.chunk_len - T, G), dtype=np.float32)
            chunk = np.vstack([chunk, pad])

        return np.ascontiguousarray(chunk)

    def __len__(self) -> int:
        real_chunks = (self._data.shape[0] + self.chunk_len - 1) // self.chunk_len
        return self._num_prepend_chunks + real_chunks
