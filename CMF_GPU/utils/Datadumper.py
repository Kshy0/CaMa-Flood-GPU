import os
import multiprocessing as mp
import numpy as np
import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from netCDF4 import Dataset
from typing import Dict, List, Tuple


def open_file(file_path: Path, file_format: str, var_name: str, data: np.ndarray):
    """
    Opens or creates a file/netCDF resource for writing.
    """
    if file_format == "bin":
        return open(file_path, 'ab')
    elif file_format == "nc":
        mode = 'a' if file_path.exists() else 'w'
        ds = Dataset(file_path, mode, format='NETCDF4')
        # For netCDF, create dimensions/variables if not existing
        if 'catchment' not in ds.dimensions:
            ds.createDimension('catchment', data.shape[0])
        if data.ndim > 1 and 'feature_dim' not in ds.dimensions:
            ds.createDimension('feature_dim', data.shape[1])
        if 'time' not in ds.dimensions:
            ds.createDimension('time', None)
        if 'time' not in ds.variables:
            time_var = ds.createVariable('time', 'f8', ('time',))
            time_var.units = "days since 1970-01-01"
            time_var.calendar = "standard"
        if var_name not in ds.variables:
            dims = ('time', 'catchment') if data.ndim == 1 else ('time', 'catchment', 'feature_dim')
            ds.createVariable(var_name, data.dtype.str, dims, zlib=True, shuffle=True)
        return ds
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def write_data(handler, file_format: str, var_name: str, data: np.ndarray, timestamp: datetime = None):
    """
    Writes data into the open file or netCDF handler and flushes immediately.
    """
    if file_format == "bin":
        handler.seek(0, os.SEEK_END)
        data.tofile(handler)
        handler.flush()  # 强制刷新文件缓冲区
    elif file_format == "nc":
        var = handler.variables[var_name]
        idx = var.shape[0]
        if data.ndim == 1:
            var[idx, :] = data
        else:
            var[idx, :, :] = data

        if timestamp is not None and 'time' in handler.variables:
            base = datetime(1970, 1, 1)
            delta_days = (timestamp - base).total_seconds() / 86400.0
            handler.variables['time'][idx] = delta_days

        handler.sync()  # netCDF 数据写入磁盘
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def worker_multi_key(
    base_dir: Path,
    file_format: str,
    save_by: str,
    key_queues: Dict,
    aggregator_keys: List[str]
):
    """
    A worker function that handles writing data for multiple aggregator keys.
    Supports safe termination via sentinel values.
    """
    current_periods: Dict[str, Tuple[int, int]] = {}
    handlers: Dict[str, object] = {}
    timestamp_caches: Dict[str, datetime] = {}
    done_keys = set()

    for agg_key in aggregator_keys:
        current_periods[agg_key] = None
        handlers[agg_key] = None
        timestamp_caches[agg_key] = None

    while True:
        all_done = True
        for agg_key in aggregator_keys:
            if agg_key in done_keys:
                continue
            try:
                timestamp, data = key_queues[agg_key].get(timeout=0.1)
                if timestamp is None and data is None:
                    done_keys.add(agg_key)
                    continue
                all_done = False
            except:
                continue  # skip if nothing available right now

            # Decide current file period
            period = (timestamp.year, timestamp.month) if save_by == "month" else (timestamp.year, None)

            # Change file if needed
            if current_periods[agg_key] != period:
                if handlers[agg_key]:
                    handlers[agg_key].close()
                current_periods[agg_key] = period
                file_name = f"{agg_key}_{period[0]}"
                if period[1]:
                    file_name += f"-{period[1]:02d}"
                file_name += f".{file_format}"
                file_path = base_dir / file_name
                handlers[agg_key] = open_file(file_path, file_format, agg_key, data)

            # Update timestamp cache
            if file_format == "nc":
                timestamp_caches[agg_key] = timestamp
            else:
                timestamp_caches[agg_key] = None

            # Write data
            write_data(
                handler=handlers[agg_key],
                file_format=file_format,
                var_name=agg_key,
                data=data,
                timestamp=timestamp_caches[agg_key]
            )

        if len(done_keys) == len(aggregator_keys):
            break

    # Final cleanup
    for agg_key in aggregator_keys:
        if handlers[agg_key]:
            handlers[agg_key].close()
        print(f"[{agg_key}] Writing completed.")


class DataDumper:
    """
    Modified DataDumper that:
    1) Does not require var_names.
    2) Accepts an aggregator dict (e.g., from update_stats_aggregator).
    3) Creates separate files & worker processes for each aggregator.
    4) Supports netCDF or binary output modes, saving by year or month.
    5) Additional param num_workers that distributes aggregator keys to avoid handle conflicts.
    """

    def __init__(
        self,
        base_dir: str,
        stats_config: Dict[str, list],
        file_format: str = "nc",
        save_by: str = "year",
        max_queue_size: int = 100,
        num_workers: int = 1
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.file_format = file_format
        self.save_by = save_by
        self.stats_config = stats_config
        self.num_workers = max(num_workers, 1)

        # Use spawn method with torch.multiprocessing
        mp.set_start_method('spawn', force=True)

        # Build aggregator keys (e.g., "river_storage_mean")
        all_aggregator_keys = []
        for var_name, stat_list in stats_config.items():
            for stat_type in stat_list:
                all_aggregator_keys.append(f"{var_name}_{stat_type}")

        # If we have fewer aggregator keys than requested workers, reduce worker count
        num_workers_actual = min(self.num_workers, len(all_aggregator_keys)) if all_aggregator_keys else 0
        self.aggregator_keys = all_aggregator_keys
        # Prepare queues for each aggregator key
        self.task_queues = {
            agg_key: mp.Queue(maxsize=max_queue_size) for agg_key in all_aggregator_keys
        }

        # Partition aggregator keys among workers
        self._workers = []
        if num_workers_actual > 0:
            # Create subsets
            for i in range(num_workers_actual):
                # Slice aggregator keys in a round-robin or simple step
                subset_keys = all_aggregator_keys[i::num_workers_actual]
                if not subset_keys:
                    continue
                worker = mp.Process(
                    target=worker_multi_key,
                    args=(self.base_dir, self.file_format, self.save_by, self.task_queues, subset_keys),
                )
                self._workers.append(worker)

        for worker in self._workers:
            worker.start()

    def submit_data(self, timestamp: datetime, aggregator: Dict[str, Dict[str, np.ndarray]]):
        for agg_key, array_data in aggregator.items():
            self.task_queues[agg_key].put((timestamp, array_data))

    def close(self):
        """
        Wait for writers to finish and close all resources.
        """
        SENTINEL = (None, None)
        for agg_key in self.aggregator_keys:
            self.task_queues[agg_key].put(SENTINEL)  # 向每个队列发哨兵

        for worker in self._workers:
            worker.join()


def test_data_dumper_nc():
    stats_config = {
        "river_storage": ["mean", "max"],
        "soil_moisture": ["min", "mean"]
    }
    base_dir = "./test_nc_output_mod"
    dumper = DataDumper(
        base_dir=base_dir,
        stats_config=stats_config,
        file_format="nc",
        save_by="month",
        max_queue_size=10,
        num_workers=2
    )

    t_start = datetime(2025, 1, 1)
    # Simulate aggregator data for multiple days
    for i in range(5):
        t_now = t_start + timedelta(days=i)
        aggregator_sample = {
            "river_storage_mean": np.random.rand(3).astype('float32'),
            "river_storage_max": np.random.rand(3).astype('float32'),
            "soil_moisture_min": np.random.rand(3).astype('float32'),
            "soil_moisture_mean": np.random.rand(3).astype('float32')
        }
        dumper.submit_data(t_now, aggregator_sample)

    # Switch to a new month
    t_new_month = datetime(2025, 2, 1)
    dumper.submit_data(t_new_month, {
        "river_storage_mean": np.random.rand(3).astype('float32'),
        "river_storage_max": np.random.rand(3).astype('float32'),
        "soil_moisture_min": np.random.rand(3).astype('float32'),
        "soil_moisture_mean": np.random.rand(3).astype('float32')
    })

    dumper.close()

    print("NetCDF test completed for modified DataDumper with multiple workers.")
    shutil.rmtree(base_dir)
    print("Temporary netCDF directory removed.\n")


def test_data_dumper_bin():
    stats_config = {
        "river_storage": ["mean", "max"],
        "soil_moisture": ["min", "mean"]
    }
    base_dir = "./test_bin_output_mod"
    dumper = DataDumper(
        base_dir=base_dir,
        stats_config=stats_config,
        file_format="bin",
        save_by="month",
        max_queue_size=10,
        num_workers=2
    )

    t_start = datetime(2025, 1, 1)
    # Simulate aggregator data for multiple days
    for i in range(5):
        t_now = t_start + timedelta(days=i)
        aggregator_sample = {
            "river_storage_mean": np.random.rand(3).astype('float32'),
            "river_storage_max": np.random.rand(3).astype('float32'),
            "soil_moisture_min": np.random.rand(3).astype('float32'),
            "soil_moisture_mean": np.random.rand(3).astype('float32')
        }   
        dumper.submit_data(t_now, aggregator_sample)

    # Switch to new month
    t_new_month = datetime(2025, 2, 1)
    dumper.submit_data(t_new_month, {
        "river_storage_mean": np.random.rand(3).astype('float32'),
        "river_storage_max": np.random.rand(3).astype('float32'),
        "soil_moisture_min": np.random.rand(3).astype('float32'),
        "soil_moisture_mean": np.random.rand(3).astype('float32')
    })

    time.sleep(1)
    dumper.close()
    print("Binary test completed for modified DataDumper with multiple workers.")
    shutil.rmtree(base_dir)
    print("Temporary BIN directory removed.\n")


if __name__ == "__main__":
    test_data_dumper_nc()
    test_data_dumper_bin()
