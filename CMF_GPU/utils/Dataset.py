from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from netCDF4 import Dataset as NcHandler

class RunoffDataset(ABC):
    """
    Custom abstract class, defines a common interface for accessing data.
    """

    @abstractmethod
    def get_data(self, time_start: datetime) -> Tuple[np.ndarray, int]:
        """
        To be implemented by subclasses, reads data of the specified date and time point from storage and returns (data, seconds).
        """
        pass
    @abstractmethod
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        To be implemented by subclasses, returns the coordinates of the dataset.
        """
        pass

    def get_mask(self) -> Optional[np.ndarray]:
        """
        Optional method, returns mask data.
        """
        return None

class DailyBinDataset(RunoffDataset):
    """
    Example of a Dataset class that reads daily binary files.
    Each bin file contains one day's data.
    """
    def __init__(self,
                 base_dir: str,
                 shape: List[int],
                 dtype: str = 'float32',
                 prefix: str = "Roff____",
                 suffix: str = ".one"):
        self.base_dir = base_dir
        self.shape = tuple(shape)
        self.dtype = dtype
        self.prefix = prefix
        self.suffix = suffix
    
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        lat = np.arange(89.5, -89.5 - 1, -1)
        lon = np.arange(-179.5, 179.5 + 1, 1)
        return lon, lat

    def get_data(self, time_start: datetime) -> Tuple[np.ndarray, int]:
        filename = f"{self.prefix}{time_start:%Y%m%d}{self.suffix}"
        file_path = Path(self.base_dir) / filename

        data = np.fromfile(file_path, dtype=self.dtype)
        data = data.reshape(self.shape, order='C')
        data[~(data > 0)] = 0.0
        return data, 86400

class YearlyNetCDFDataset(RunoffDataset):
    # lat = np.arange(89.875, -59.875 - 0.25, -0.25)    
    # lon = np.arange(-179.875, 179.875 + 0.25, 0.25)
    def __init__(self,
                 base_dir: str,
                 var_name: str = "Runoff",
                 prefix: str = "e2o_ecmwf_wrr2_glob15_day_Runoff_",
                 suffix: str = ".nc"):
        self.base_dir = base_dir
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

    def get_data(self, time_start: datetime, fill_missing:bool = True) -> Tuple[np.ndarray, int]:
        year = time_start.year
        if year != self._cached_year:
            self._close_dataset()
            year_str = time_start.strftime("%Y")
            filename = f"{self.prefix}{year_str}{self.suffix}"
            file_path = Path(self.base_dir) / filename
            self._cached_dataset = NcHandler(file_path, 'r')
            self._cached_year = year

        time_index = time_start.timetuple().tm_yday - 1
        var = self._cached_dataset.variables[self.var_name]
        data = self._read_and_process_var(var, time_index, fill_missing)
        return data, 86400
    
    def get_mask(self):
        example_data, _ = self.get_data(datetime(2000, 1, 1), False)
        return ~example_data.mask
    
    def _close_dataset(self) -> None:
        if self._cached_dataset is not None:
            self._cached_dataset.close()
            self._cached_dataset = None
            self._cached_year = None
            self._dim_order = None

    def close(self) -> None:
        self._close_dataset()