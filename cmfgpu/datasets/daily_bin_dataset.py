# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np

from cmfgpu.datasets.abstract_dataset import AbstractDataset


class DailyBinDataset(AbstractDataset):
    """
    Example of a Dataset class that reads daily binary files.
    Each bin file contains one day's data.
    """

    def _validate_files_exist(self):
        """
        Validates that all expected files between start_date and end_date exist.
        """
        def file_path_gen(dt: datetime) -> Path:
            filename = f"{self.prefix}{dt:%Y%m%d}{self.suffix}"
            return Path(self.base_dir) / filename
            
        self.validate_files_in_range(file_path_gen)
        
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
                 *args, **kwargs):

        self.base_dir = base_dir
        self.shape = tuple(shape)
        self.unit_factor = unit_factor
        self.bin_dtype = bin_dtype
        self.prefix = prefix
        self.suffix = suffix
        super().__init__(out_dtype=out_dtype, chunk_len=1, time_interval=timedelta(days=1), start_date=start_date, end_date=end_date, calendar=calendar, *args, **kwargs)
        self._validate_files_exist()

    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        lat = np.arange(89.5, -89.5 - 1, -1)
        lon = np.arange(-179.5, 179.5 + 1, 1)
        return lon, lat

    def get_data(self, current_time: datetime, chunk_len: int) -> np.ndarray:
        """Read one day's data from binary file.
        
        If _local_runoff_indices is set (after build_local_runoff_matrix),
        extracts only active columns for memory efficiency.
        Otherwise returns full grid data (1, Y*X).
        """
        if chunk_len != 1:
            raise ValueError("DailyBinDataset only supports chunk_len=1 (one day per file)")
        filename = f"{self.prefix}{current_time:%Y%m%d}{self.suffix}"
        file_path = Path(self.base_dir) / filename
        data = np.fromfile(file_path, dtype=self.bin_dtype)
        data[~(data >= 0)] = 0.0
        
        # Extract active columns if set
        if self._local_runoff_indices is not None:
            data = data[self._local_runoff_indices]
        
        return (data.astype(self.out_dtype) / self.unit_factor)[None, :]
    
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
