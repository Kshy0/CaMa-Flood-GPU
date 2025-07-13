from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
from cmfgpu.datasets.abstract_dataset import DefaultDataset
import numpy as np
from datetime import timedelta

class DailyBinDataset(DefaultDataset):
    """
    Example of a Dataset class that reads daily binary files.
    Each bin file contains one day's data.
    """
    def __init__(self,
                 base_dir: str,
                 shape: List[int],
                 start_date: datetime,
                 end_date: datetime,
                 unit_factor: float = 1.0,
                 out_dtype: str = "float32",
                 bin_dtype: str = "float32",
                 prefix: str = "Roff____",
                 suffix: str = ".one",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_dir = base_dir
        self.shape = tuple(shape)
        self.start_date = start_date
        self.end_date = end_date
        self.unit_factor = unit_factor
        self.out_dtype = out_dtype
        self.bin_dtype = bin_dtype
        self.prefix = prefix
        self.suffix = suffix
        self._validate_files_exist()
    
    

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

    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        lat = np.arange(89.5, -89.5 - 1, -1)
        lon = np.arange(-179.5, 179.5 + 1, 1)
        return lon, lat

    def get_data(self, current_time: datetime) -> np.ndarray:
        filename = f"{self.prefix}{current_time:%Y%m%d}{self.suffix}"
        file_path = Path(self.base_dir) / filename

        data = np.fromfile(file_path, dtype=self.bin_dtype)
        # data = data.reshape(self.shape, order='C')
        data = data[self.local_runoff_indices] if self.local_runoff_indices is not None else data
        data[~(data >= 0)] = 0.0
        
        return data.astype(self.out_dtype) / self.unit_factor
    
    @property
    def data_mask(self):
        return None
    
    def get_time_by_index(self, idx: int) -> datetime:
        """
        Returns the datetime corresponding to the given index.
        """
        return self.start_date + timedelta(days=idx)
    
    def close(self):
        pass