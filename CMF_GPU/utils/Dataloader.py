import torch
import torch.multiprocessing as mp
import numpy as np
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import OrderedDict
from netCDF4 import Dataset as NcHandler

class RunoffDataset(ABC):
    """
    Custom abstract class, defines a common interface for accessing data.
    """
    @property
    @abstractmethod
    def lat(self) -> np.ndarray:
        pass
    
    @property
    @abstractmethod
    def lon(self) -> np.ndarray:
        pass    

    @abstractmethod
    def get_data(self, time_start: datetime) -> Tuple[np.ndarray, int]:
        """
        To be implemented by subclasses, reads data of the specified date and time point from storage and returns (data, seconds).
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
    lat = np.arange(89.5, -89.5 - 1, -1)
    lon = np.arange(-179.5, 179.5 + 1, 1)
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

    def get_data(self, time_start: datetime) -> Tuple[np.ndarray, int]:
        filename = f"{self.prefix}{time_start:%Y%m%d}{self.suffix}"
        file_path = Path(self.base_dir) / filename

        data = np.fromfile(file_path, dtype=self.dtype)
        data = data.reshape(self.shape, order='C')
        data[~(data > 0)] = 0.0
        return data, 86400

class YearlyNetCDFDataset(RunoffDataset):
    lat = np.arange(89.875, -59.875 - 0.25, -0.25)    
    lon = np.arange(-179.875, 179.875 + 0.25, 0.25)
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
        self._dim_order: Optional[Tuple[int, int, int]] = None  # (time, lat, lon) 顺序

    def _init_dims_if_needed(self, var) -> None:
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
        self._init_dims_if_needed(var)
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

def _worker_read_data_wrapper(idx, t, dataset, precision, unit_factor, runoff_mask_list=None):
    """Modified worker function to include the index for ordering"""
    try:
        data, seconds = dataset.get_data(t)
        if precision == "float64":
            data = data.astype(np.float64)
        else:
            data = data.astype(np.float32)
        data = data / unit_factor
        if runoff_mask_list is not None:
            data = data.ravel(order='C')
            splits = [torch.tensor(data[mask]) for mask in runoff_mask_list]
        else:
            splits = torch.tensor(data)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise 

    return idx, t, splits, seconds

class DataLoader:
    """
    A DataLoader that guarantees data is returned in the exact order of time_starts.
    Data is cleared from memory once it has been retrieved.
    """
    def __init__(self,
                 time_starts: List[datetime],
                 dataset,
                 unit_factor: float,
                 precision: str,
                 num_workers: int = 1,
                 max_cache_steps: int = 5,
                 runoff_mask: List[np.ndarray] = None):
        mp.set_start_method("spawn", force=True)
        self.time_list = sorted(time_starts)
        self.dataset = dataset
        self.unit_factor = unit_factor
        self.precision = precision
        self.max_cache_steps = max_cache_steps
        self.runoff_mask = runoff_mask
        self.num_workers = num_workers

        # Use an ordered dictionary to store processed results by index
        self._results: OrderedDict[int, Tuple[datetime, List[torch.Tensor], int]] = OrderedDict()
        self._next_idx = 0  # Next index to process
        self._next_return_idx = 0  # Next index to return
        self._processing_indices = set()  # Track indices that are currently being processed
        
        self.pool = mp.Pool(processes=num_workers)
        self._fill_pipeline()

    def _fill_pipeline(self):
        """Fill the processing pipeline with new tasks up to max_cache_steps."""
        while (self._next_idx < len(self.time_list) and 
               len(self._processing_indices) + len(self._results) < self.max_cache_steps):
            idx = self._next_idx
            t = self.time_list[idx]
            args = (idx, t,
                    self.dataset,
                    self.precision,
                    self.unit_factor,
                    self.runoff_mask)
            self.pool.apply_async(
                _worker_read_data_wrapper,
                args=args,
                callback=self._on_task_complete
            )
            self._processing_indices.add(idx)
            self._next_idx += 1

    def _on_task_complete(self, result):
        """Handle completed task and remove it from processing set."""
        idx, t, data_splits, seconds = result
        self._results[idx] = (t, data_splits, seconds)
        self._processing_indices.remove(idx)
        # Try to fill the pipeline with more tasks
        self._fill_pipeline()

    def get_data(self, t: datetime) -> Tuple[List[torch.Tensor], int]:
        """
        Returns data for the specified time.
        If earlier time points haven't been processed yet, waits for them first
        to maintain strict ordering. Clears data from memory once retrieved.
        """
        # Find the index of the requested time
        if t not in self.time_list:
            raise ValueError(f"Time {t} not in time_list")
        request_idx = self.time_list.index(t)
        
        # Wait for all data up to and including the requested index
        while self._next_return_idx <= request_idx:
            # Wait for the next item to be available
            while self._next_return_idx not in self._results:
                time.sleep(0.01)  # Wait a bit before checking again
            
            # If we've reached the requested index, return the data
            if self._next_return_idx == request_idx:
                result_t, data_splits, seconds = self._results[request_idx]
                # Remove this data from the results since it's no longer needed
                del self._results[request_idx]
                self._next_return_idx += 1  # Move to the next index
                # Try to fill the pipeline now that we've removed an item
                self._fill_pipeline()
                return data_splits, seconds
            else:
                # Skip earlier data points, remove them, and advance the return index
                del self._results[self._next_return_idx]
                self._next_return_idx += 1
                # Try to fill the pipeline now that we've removed an item
                self._fill_pipeline()
        
        # This should not be reached if the logic above is correct
        raise RuntimeError("Logic error in get_data")

    def close(self):
        if hasattr(self.dataset, "close") and callable(self.dataset.close):
            self.dataset.close()
        self.pool.close()
        self.pool.join()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    time_starts = [datetime(2000, 1, i + 1) for i in range(10)]

    # 1) DailyBinDataset test
    daily_bin_ds = DailyBinDataset(
        base_dir=r"/home/eat/cmf_v420_pkg/inp/test_1deg/runoff",
        shape=(180, 360),
        dtype='float32',
        prefix="Roff____",
        suffix=".one"
    )
    loader_bin = DataLoader(
        time_starts,
        daily_bin_ds,
        unit_factor=1,
        precision="float32",
        num_workers=1,
    )
    data_0102_bin, seconds_bin = loader_bin.get_data(datetime(2000, 1, 2))
    print("DailyBin shape =", data_0102_bin.shape)
    print("DailyBin seconds =", seconds_bin)
    plt.imshow(data_0102_bin.cpu().numpy() > 0, cmap='viridis', aspect='auto')
    plt.show()
    loader_bin.close()

    # 2) YearlyNcDataset test
    yearly_nc_ds = YearlyNetCDFDataset(
        base_dir=r"/home/eat/cmf_v420_pkg/inp/test_15min_nc",
        var_name="Runoff",
        prefix="e2o_ecmwf_wrr2_glob15_day_Runoff_",
        suffix=".nc"
    )
    loader_nc = DataLoader(
        time_starts,
        yearly_nc_ds,
        unit_factor=1,
        precision="float32",
        num_workers=1,
    )
    data_0102_nc, seconds_nc = loader_nc.get_data(datetime(2000, 1, 2))
    plt.imshow(data_0102_nc.cpu().numpy() > 0, cmap='viridis', aspect='auto')
    plt.show()
    print("YearlyNc shape =", data_0102_nc.shape)
    print("YearlyNc seconds =", seconds_nc)
    loader_nc.close()