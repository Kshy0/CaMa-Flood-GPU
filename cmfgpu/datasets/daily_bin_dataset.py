from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import List, Tuple

import numpy as np

from cmfgpu.datasets.abstract_dataset import AbstractDataset


class DailyBinDataset(AbstractDataset):
    """
    Example of a Dataset class that reads daily binary files.
    Each bin file contains one day's data.
    """

    def __len__(self):
        """
        Returns the total number of samples in the dataset based on the time range.
        """
        return (self.end_date - self.start_date).days + 1
    
    def _validate_files_exist(self):
        """
        Validates that all expected files between start_date and end_date exist.
        """
        missing_files = []
        for idx in range(self.__len__()):
            date = self.get_time_by_index(idx)
            filename = f"{self.prefix}{date:%Y%m%d}{self.suffix}"
            file_path = Path(self.base_dir) / filename
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            raise FileNotFoundError(
                f"The following required data files are missing:\n" +
                "\n".join(missing_files)
            )
        
    def __init__(self,
                 base_dir: str,
                 shape: List[int],
                 start_date: datetime,
                 end_date: datetime,
                 unit_factor: float = 1.0,
                 bin_dtype: str = "float32",
                 prefix: str = "Roff____",
                 suffix: str = ".one",
                 *args, **kwargs):
        
        self.base_dir = base_dir
        self.shape = tuple(shape)
        self.start_date = start_date
        self.end_date = end_date
        self.unit_factor = unit_factor
        self.bin_dtype = bin_dtype
        self.prefix = prefix
        self.suffix = suffix
        self._validate_files_exist()
        super().__init__(*args, **kwargs)

    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        lat = np.arange(89.5, -89.5 - 1, -1)
        lon = np.arange(-179.5, 179.5 + 1, 1)
        return lon, lat

    def get_data(self, current_time: datetime) -> np.ndarray:
        filename = f"{self.prefix}{current_time:%Y%m%d}{self.suffix}"
        file_path = Path(self.base_dir) / filename

        data = np.fromfile(file_path, dtype=self.bin_dtype)
        # data = data.reshape(self.shape, order='C')
        data[~(data >= 0)] = 0.0

        return data.astype(self.out_dtype) / self.unit_factor

    @cached_property
    def data_mask(self):
        return np.ones(np.prod(self.shape), dtype=bool)
    
    def get_time_by_index(self, idx: int) -> datetime:
        """
        Returns the datetime corresponding to the given index.
        """
        return self.start_date + timedelta(days=idx)
    
    def close(self):
        pass

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
