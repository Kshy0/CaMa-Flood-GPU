"""Base class for gauge inflow injection datasets.

Gauge datasets hold per-gauge (catchment-level) discharge time series.
``shard_forcing`` injects gauge discharge into a runoff tensor at the
corresponding catchment indices so that no changes to the physics kernel
or model class are required.
"""

from typing import Optional

import numpy as np
import torch
from hydroforge.modeling.distributed import find_indices_in


class AbstractGaugeDataset(torch.utils.data.Dataset):
    """Base class for gauge inflow datasets.

    Subclasses load data from a specific file format and pass the resulting
    arrays to this constructor.

    Parameters
    ----------
    gauge_catchment_ids : np.ndarray
        Integer catchment IDs read from the gauge file, shape ``(G,)``.
    data : np.ndarray
        Discharge time series in **m³ s⁻¹**, shape ``(T, G)``.
    chunk_len : int
        Number of time steps per chunk (must match the runoff dataset).
    num_prepend_chunks : int
        Zero-filled chunks to prepend for spin-up alignment.  Set to
        ``runoff_dataset.num_spin_up_chunks`` when the runoff dataset
        has spin-up enabled.
    """

    def __init__(
        self,
        gauge_catchment_ids: np.ndarray,
        data: np.ndarray,
        *,
        chunk_len: int = 24,
        num_prepend_chunks: int = 0,
    ):
        super().__init__()
        self._gauge_catchment_ids = np.asarray(gauge_catchment_ids, dtype=np.int64)
        self._data = np.asarray(data, dtype=np.float32)
        self.chunk_len = chunk_len
        self._num_prepend_chunks = num_prepend_chunks
        self._local_indices: Optional[np.ndarray] = None
        self._inflow_catchment_idx: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def gauge_catchment_ids(self) -> np.ndarray:
        """Catchment IDs from the gauge data file, shape ``(G,)``."""
        return self._gauge_catchment_ids

    @property
    def num_gauges(self) -> int:
        return len(self._gauge_catchment_ids)

    # ------------------------------------------------------------------
    # Mapping / injection
    # ------------------------------------------------------------------
    def build_local_mapping(
        self,
        desired_catchment_ids: np.ndarray,
        catchment_id: np.ndarray,
        device: torch.device,
    ) -> None:
        """Set up column reorder and compute injection indices.

        Parameters
        ----------
        desired_catchment_ids : np.ndarray
            Inflow catchment IDs from *parameters.nc*
            (``model.base.inflow_catchment_id.numpy()``).
        catchment_id : np.ndarray
            Full model catchment ID array
            (``model.base.catchment_id.numpy()``).
        device : torch.device
            Target device for the injection index tensor.
        """
        desired = np.asarray(desired_catchment_ids, dtype=np.int64)

        # Column reorder: gauge-file order → desired order
        col_pos = find_indices_in(desired, self.gauge_catchment_ids)
        if np.any(col_pos == -1):
            missing = desired[col_pos == -1].tolist()
            raise ValueError(f"Gauge data missing catchment IDs: {missing}")
        self._local_indices = col_pos.astype(np.int64)

        # Injection indices: desired → position in the full catchment array
        inflow_idx = find_indices_in(desired, np.asarray(catchment_id, dtype=np.int64))
        if np.any(inflow_idx == -1):
            missing = desired[inflow_idx == -1].tolist()
            raise ValueError(
                f"Inflow catchment IDs not found in model catchment array: {missing}"
            )
        self._inflow_catchment_idx = torch.from_numpy(
            inflow_idx.astype(np.int64)
        ).to(device)

        print(
            f"[GaugeDataset] Mapped {len(desired)} inflow gauges "
            f"from {self.num_gauges} in file"
        )

    def shard_forcing(
        self,
        batch_data: torch.Tensor,
        batch_runoff: torch.Tensor,
    ) -> torch.Tensor:
        """Inject gauge discharge into the runoff tensor.

        Parameters
        ----------
        batch_data : torch.Tensor
            Gauge discharge from :class:`DataLoader`, shape ``(B, T, G)``.
        batch_runoff : torch.Tensor
            Runoff already processed by the runoff dataset's
            ``shard_forcing``, shape ``(B*T, C)``.

        Returns
        -------
        torch.Tensor
            The modified *batch_runoff* with gauge discharge added at
            inflow catchment locations.
        """
        if self._inflow_catchment_idx is None:
            raise RuntimeError(
                "build_local_mapping must be called before shard_forcing"
            )

        if batch_data.dim() == 3:
            B, T, G = batch_data.shape
            gauge_flat = batch_data.reshape(B * T, G).contiguous()
        else:
            raise ValueError(f"Expected 3D gauge tensor, got {batch_data.dim()}D")

        batch_runoff[:, self._inflow_catchment_idx] += gauge_flat
        return batch_runoff

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0:
            idx += len(self)

        G = len(self._local_indices) if self._local_indices is not None else self.num_gauges

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

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        real_chunks = (self._data.shape[0] + self.chunk_len - 1) // self.chunk_len
        return self._num_prepend_chunks + real_chunks
