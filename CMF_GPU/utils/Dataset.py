from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
from netCDF4 import Dataset as NcHandler
from datetime import timedelta

# TODO: add rank for multiple GPUs
class RunoffDataset(torch.utils.data.Dataset, ABC):
    """
    Custom abstract class that inherits from PyTorch Dataset.
    Defines a common interface for accessing data.
    """
    def __init__(self, start_date: str, end_date: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.num_samples = self.calculate_num_samples()

    def calculate_num_samples(self) -> int:
        """
        Calculate the number of samples based on start_date and end_date.
        Assuming data is daily, this will return the number of days in the range.
        """
        return (self.end_date - self.start_date).days + 1

    def __len__(self):
        """
        Returns the total number of samples in the dataset based on the time range.
        """
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Fetches data for a given index.
        """
        current_time = self.get_time_by_index(idx)
        data, seconds = self.get_data(current_time)
        return data, seconds
    
    @abstractmethod
    def get_data(self, current_time: datetime) -> Tuple[np.ndarray, int]:
        """
        To be implemented by subclasses, reads data of the specified date and time point from storage and returns (data, seconds).
        """
        pass
    
    @abstractmethod
    def get_mask(self) -> np.ndarray:
        """
        Returns the mask of the dataset.
        """
        pass

    @abstractmethod
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        To be implemented by subclasses, returns the coordinates of the dataset.
        """
        pass

    @abstractmethod
    def get_time_by_index(self, idx: int) -> datetime:
        """
        Returns the datetime corresponding to the given index.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Closes any open resources or files.
        """
        pass

class DailyBinDataset(RunoffDataset):
    """
    Example of a Dataset class that reads daily binary files.
    Each bin file contains one day's data.
    """
    def __init__(self,
                 base_dir: str,
                 shape: List[int],
                 start_date: str,
                 end_date: str,
                 unit_factor: float = 1.0,
                 dtype: str = 'float32',
                 prefix: str = "Roff____",
                 suffix: str = ".one"):
        super().__init__(start_date, end_date)
        self.base_dir = base_dir
        self.shape = tuple(shape)
        self.unit_factor = unit_factor
        self.dtype = dtype
        self.prefix = prefix
        self.suffix = suffix
    
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        lat = np.arange(89.5, -89.5 - 1, -1)
        lon = np.arange(-179.5, 179.5 + 1, 1)
        return lon, lat

    def get_data(self, current_time: datetime) -> Tuple[np.ndarray, int]:
        filename = f"{self.prefix}{current_time:%Y%m%d}{self.suffix}"
        file_path = Path(self.base_dir) / filename

        data = np.fromfile(file_path, dtype=self.dtype)
        # data = data.reshape(self.shape, order='C')
        data[~(data > 0)] = 0.0
        return data / self.unit_factor, 86400
    
    def get_mask(self) -> np.ndarray:
        return None
    
    def get_time_by_index(self, idx: int) -> datetime:
        """
        Returns the datetime corresponding to the given index.
        """
        return self.start_date + timedelta(days=idx)
    
    def close(self):
        pass

class YearlyNetCDFDataset(RunoffDataset):
    def __init__(self,
                 base_dir: str,
                 start_date: str,
                 end_date: str,
                 unit_factor: float = 1.0,
                 var_name: str = "Runoff",
                 prefix: str = "e2o_ecmwf_wrr2_glob15_day_Runoff_",
                 suffix: str = ".nc"):
        super().__init__(start_date, end_date)
        self.base_dir = base_dir
        self.unit_factor = unit_factor
        self.var_name = var_name
        self.prefix = prefix
        self.suffix = suffix
        self._cached_dataset: Optional[NcHandler] = None
        self._cached_year: Optional[int] = None
        self._dim_order: Optional[Tuple[int, int, int]] = None  # (time, lat, lon)

    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        filename = f"{self.prefix}{2000}{self.suffix}"
        file_path = Path(self.base_dir) / filename

        with NcHandler(file_path, 'r') as dataset:
            lat_var = (
                dataset.variables.get('lat')
                or dataset.variables.get('latitude')
            )
            lon_var = (
                dataset.variables.get('lon')
                or dataset.variables.get('longitude')
                or dataset.variables.get('long')
            )

            if lat_var is None or lon_var is None:
                raise ValueError("Unable to find lat/lon variables in the dataset.")

            lat = np.array(lat_var[:])
            lon = np.array(lon_var[:])
            return lon, lat
        
    def _init_dims(self, var) -> None:
        if self._dim_order is None:
            dim_names = var.dimensions
            dim_mapping = {name.lower(): i for i, name in enumerate(dim_names)}
            try:
                time_idx = dim_mapping['time']
                lat_idx = dim_mapping.get('lat') or dim_mapping.get('latitude')
                lon_idx = (
                    dim_mapping.get('lon')
                    or dim_mapping.get('longitude')
                    or dim_mapping.get('long')
                )
                if lat_idx is None or lon_idx is None:
                    raise ValueError("Unable to recognize the dimension for lat/lon")
                self._dim_order = (time_idx, lat_idx, lon_idx)
            except Exception:
                raise ValueError(f"The dimensions of variable {self.var_name} do not contain information about time/lat/lon: {dim_names}")

    def _read_and_process_var(self, var, time_index: int, fill_missing:bool = True) -> np.ndarray:
        self._init_dims(var)
        data = var[time_index, :, :]

        if fill_missing:
            data = data.filled(0.0)

        if self._dim_order != (0, 1, 2):
            transpose_order = [self._dim_order.index(i) for i in (0, 1, 2)]
            data = np.transpose(data, axes=[transpose_order[1], transpose_order[2]])

        return data

    def get_data(self, current_time: datetime, fill_missing:bool = True) -> Tuple[np.ndarray, int]:
        year = current_time.year
        if year != self._cached_year:
            self._close_dataset()
            year_str = current_time.strftime("%Y")
            filename = f"{self.prefix}{year_str}{self.suffix}"
            file_path = Path(self.base_dir) / filename
            self._cached_dataset = NcHandler(file_path, 'r')
            self._cached_year = year

        time_index = current_time.timetuple().tm_yday - 1
        var = self._cached_dataset.variables[self.var_name]
        data = self._read_and_process_var(var, time_index, fill_missing)
        return data / self.unit_factor, 86400
    
    def get_mask(self):
        example_data, _ = self.get_data(datetime(2000, 1, 1), False)
        return ~example_data.mask
    
    def get_time_by_index(self, idx: int) -> datetime:
        """
        Returns the datetime corresponding to the given index.
        """
        return self.start_date + timedelta(days=idx)

    def _close_dataset(self) -> None:
        if self._cached_dataset is not None:
            self._cached_dataset.close()
            self._cached_dataset = None
            self._cached_year = None
            self._dim_order = None

    def close(self) -> None:
        self._close_dataset()

if __name__ == "__main__":
    # Example usage
    from torch.utils.data import DataLoader
    dataset = DailyBinDataset(
        base_dir="/home/eat/cmf_v420_pkg/inp/test_1deg/runoff",
        shape=[180, 360],
        start_date=datetime(2000, 1, 1),
        end_date=datetime(2000, 1, 31)
    )
    
    lon, lat = dataset.get_coordinates()
    print("Coordinates:", lon.shape, lat.shape)
    
    data, seconds = dataset.get_data(datetime(2000, 1, 1))
    print("Data shape:", data.shape, "Seconds:", seconds)
    dataset.close()
    batch_size = 1  
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for i, (data_batch, _) in enumerate(data_loader):
        print(f"Batch {i + 1} data shape:", data_batch.shape)

    dataset.close()