# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cftime
import numpy as np

from cmfgpu.datasets.abstract_dataset import AbstractDataset
from cmfgpu.datasets.utils import daily_time_to_key, single_file_key


class DailyBinDataset(AbstractDataset):
    """
    Dataset class that reads daily binary files.

    By default each binary file contains one day's data, with filenames
    ``{prefix}{YYYYMMDD}{suffix}``.  The ``time_to_key`` callable controls
    the mapping from date to filename key:

    * **One file per day** (default): ``time_to_key = daily_time_to_key``
      → every date gets a unique key, each file has one frame.
    * **Single file** (e.g. 365-day climatology): use a constant key
      ``time_to_key = lambda dt: ""``  (or any constant string).
      Consecutive frames are read from the file by offset.
    """

    def _build_file_mapping(self):
        """Map each simulation date to ``(file_key, frame_index)``."""
        self._dt_to_loc = {}
        key_frame_count: dict = {}
        t = self.start_date
        while t <= self.end_date:
            key = self.time_to_key(t)
            if key not in key_frame_count:
                key_frame_count[key] = 0
            self._dt_to_loc[t] = (key, key_frame_count[key])
            key_frame_count[key] += 1
            t += self.time_interval

        # Also handle spin-up dates
        if self.spin_up_cycles > 0 and self.spin_up_start_date and self.spin_up_end_date:
            t = self.spin_up_start_date
            while t <= self.spin_up_end_date:
                if t not in self._dt_to_loc:
                    key = self.time_to_key(t)
                    if key not in key_frame_count:
                        key_frame_count[key] = 0
                    self._dt_to_loc[t] = (key, key_frame_count[key])
                    key_frame_count[key] += 1
                t += self.time_interval

    def _validate_files_exist(self):
        """Validate that all required files exist."""
        unique_files = set()
        for key, _ in self._dt_to_loc.values():
            unique_files.add(Path(self.base_dir) / f"{self.prefix}{key}{self.suffix}")
        self.validate_files_exist(list(unique_files))

    def __init__(self,
                 base_dir: str,
                 shape: List[int],
                 start_date: datetime,
                 end_date: datetime,
                 unit_factor: float = 1.0, # mm/day divided by unit_factor to get m/s
                 bin_dtype: str = "float32",
                 prefix: str = "Roff____",
                 suffix: str = ".one",
                 out_dtype: str = "float32",
                 calendar: str = "standard",
                 lat_south_to_north: bool = False,  # If True, latitude goes from south to north
                 time_to_key: Optional[Callable[[Union[datetime, cftime.datetime]], str]] = daily_time_to_key,
                 *args, **kwargs):

        self.base_dir = base_dir
        self.shape = tuple(shape)
        self.unit_factor = unit_factor
        self.bin_dtype = bin_dtype
        self.prefix = prefix
        self.suffix = suffix
        self.lat_south_to_north = lat_south_to_north
        self.time_to_key = time_to_key if time_to_key is not None else single_file_key
        super().__init__(out_dtype=out_dtype, chunk_len=1, time_interval=timedelta(days=1), start_date=start_date, end_date=end_date, calendar=calendar, *args, **kwargs)
        self._build_file_mapping()
        self._validate_files_exist()

    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lon, lat) coordinate arrays.
        
        Note: shape is (ny, nx) = (lat, lon), so shape[0] is lat size, shape[1] is lon size.
        Coordinates are cell centers, computed from shape assuming global coverage.
        
        If lat_south_to_north is True, latitude goes from -90 to 90 (south to north).
        Otherwise, latitude goes from 90 to -90 (north to south, default).
        """
        ny, nx = self.shape
        # Resolution in degrees
        res_lat = 180.0 / ny
        res_lon = 360.0 / nx
        # Cell centers
        if self.lat_south_to_north:
            lat = np.linspace(-90 + res_lat / 2, 90 - res_lat / 2, ny)
        else:
            lat = np.linspace(90 - res_lat / 2, -90 + res_lat / 2, ny)
        lon = np.linspace(-180 + res_lon / 2, 180 - res_lon / 2, nx)
        return lon, lat

    def get_data(self, current_time: datetime, chunk_len: int) -> np.ndarray:
        """Read one day's data from binary file.
        
        Returns:
        - If _local_runoff_indices is set: (1, N) compressed array
        - If _local_runoff_indices is None: (1, Y, X) full grid array
        
        Spatial convention: (Y, X) = (lat, lon), C-order flatten (lon varies fastest)
        """
        if chunk_len != 1:
            raise ValueError("DailyBinDataset only supports chunk_len=1 (one day per file)")

        key, frame_idx = self._dt_to_loc[current_time]
        filename = f"{self.prefix}{key}{self.suffix}"
        file_path = Path(self.base_dir) / filename

        ny, nx = self.shape
        frame_size = ny * nx
        element_size = np.dtype(self.bin_dtype).itemsize

        data = np.fromfile(
            file_path, dtype=self.bin_dtype,
            count=frame_size, offset=frame_idx * frame_size * element_size,
        )
        data[~(data >= 0)] = 0.0
        data = data.astype(self.out_dtype) / self.unit_factor
        
        if self._local_runoff_indices is not None:
            return data[self._local_runoff_indices][None, :]
        else:
            return data.reshape(1, ny, nx)
    
    def close(self):
        pass

    def __len__(self):
        """
        Returns the total number of samples in the dataset based on the time range.
        """
        return super().__len__()

if __name__ == "__main__":
    resolution = "glb_15min"
    dataset = DailyBinDataset(
        base_dir="/home/eat/cmf_v420_pkg/inp/test_1deg/runoff",
        shape=[180, 360],
        start_date=datetime(2000, 1, 1),
        end_date=datetime(2000, 12, 31),
    )
    dataset.generate_runoff_mapping_table(
        map_dir=f"/home/eat/cmf_v420_pkg/map/{resolution}",
        out_dir=f"/home/eat/CaMa-Flood-GPU/inp/{resolution}",
        npz_file="runoff_mapping_bin.npz",
    )
