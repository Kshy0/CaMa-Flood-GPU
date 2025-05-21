import torch
import torch.multiprocessing as mp
import numpy as np
import time
from datetime import datetime
from typing import List, Tuple
from collections import OrderedDict


def _worker_read_data_wrapper(idx, t, dataset, precision):
    """Modified worker function to include the index for ordering"""
    try:
        data, seconds = dataset.get_data(t)
        if precision == "float64":
            data = data.astype(np.float64)
        else:
            data = data.astype(np.float32)
        data = torch.tensor(data.ravel(order='C'))

    except Exception as e:
        import traceback; traceback.print_exc()
        raise 

    return idx, t, data, seconds

class DataLoader:
    """
    A DataLoader that guarantees data is returned in the exact order of time_starts.
    Data is cleared from memory once it has been retrieved.
    """
    def __init__(self,
                 time_starts: List[datetime],
                 dataset,
                 precision: str,
                 num_workers: int = 1,
                 max_cache_steps: int = 5):
        mp.set_start_method("spawn", force=True)
        self.time_list = sorted(time_starts)
        self.dataset = dataset
        self.precision = precision
        self.max_cache_steps = max_cache_steps
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
                    self.precision,)
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
    from CMF_GPU.utils.Dataset import DailyBinDataset, YearlyNetCDFDataset
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
    lon, lat = yearly_nc_ds.get_coordinates()
    loader_nc = DataLoader(
        time_starts,
        yearly_nc_ds,
        precision="float32",
        num_workers=1,
    )
    data_0102_nc, seconds_nc = loader_nc.get_data(datetime(2000, 1, 2))
    plt.imshow(data_0102_nc.cpu().numpy() > 0, cmap='viridis', aspect='auto')
    plt.show()
    print("YearlyNc shape =", data_0102_nc.shape)
    print("YearlyNc seconds =", seconds_nc)
    loader_nc.close()