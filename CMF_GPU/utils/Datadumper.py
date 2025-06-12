import os
import multiprocessing as mp
import numpy as np
import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from netCDF4 import Dataset
from typing import Dict, List, Tuple
from CMF_GPU.utils.utils import check_enabled, get_global_rank, gather_device_dicts
from CMF_GPU.utils.Aggregator import LEGAL_AGG_ARRAYS, DIM_INFO

def open_file(
    file_path: Path,
    file_format: str,
    var_name: str,
    data: np.ndarray,
    dims: Tuple[str, ...],
    dim_info: Dict[str, np.ndarray]
):
    """
    Opens or creates a file/netCDF resource for writing.
    Uses 'dim_info' to translate dimension names into coordinate names (if any).
    """
    if file_format == "bin":
        return open(file_path, 'ab')

    elif file_format == "nc":
        mode = 'a' if file_path.exists() else 'w'
        ds = Dataset(file_path, mode, format='NETCDF4')

        # Create dimension(s) and coordinate variable(s) if indicated in dim_info
        # For example, for dims = ("num_catchments_to_save",), check dim_info["num_catchments_to_save"].
        # If it's not None, we create a dimension with that name and a coordinate variable.
        for i_dim, dim_name in enumerate(dims):
            coord_name = DIM_INFO[dim_name]  # e.g. DIM_INFO["num_catchments_to_save"] = "catchment_save_idx"
            size = data.shape[i_dim]
            if coord_name not in ds.dimensions:
                ds.createDimension(coord_name, size)
                # Create coordinate variable
                coord_var = ds.createVariable(coord_name, 'i8', (coord_name,))
                coord_var[:] = dim_info.get(dim_name, np.arange(size))
            
        # Also create/ensure time dimension
        if 'time' not in ds.dimensions:
            ds.createDimension('time', None)
        if 'time' not in ds.variables:
            time_var = ds.createVariable('time', 'f8', ('time',))
            time_var.units = "days since 1970-01-01"
            time_var.calendar = "standard"

        # Now build the dimension tuple for the variable
        # e.g. dims=('num_catchments_to_save',) => could become ('time','catchment_save_idx')
        var_dims = ['time']
        for i_dim, dim_name in enumerate(dims):
            coord_name = DIM_INFO[dim_name]
            var_dims.append(coord_name)

        if var_name not in ds.variables:
            ds.createVariable(var_name, data.dtype.str, tuple(var_dims), zlib=True, shuffle=True)
        return ds

    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def write_data(
    handler,
    file_format: str,
    var_name: str,
    data: np.ndarray,
    timestamp: datetime = None
):
    """
    Writes data into the open file or netCDF handler and flushes immediately.
    """
    if file_format == "bin":
        handler.seek(0, os.SEEK_END)
        data.tofile(handler)
        handler.flush()
    elif file_format == "nc":
        var = handler.variables[var_name]
        idx = var.shape[0]  # new record along time dimension
        # Destination slices -> var[idx, ...]
        if data.ndim == 1:
            var[idx, :] = data
        elif data.ndim == 2:
            var[idx, :, :] = data
        else:
            # Possibly more dimensions, adapt accordingly
            var[idx] = data

        if timestamp is not None and 'time' in handler.variables:
            base = datetime(1970, 1, 1)
            delta_days = (timestamp - base).total_seconds() / 86400.0
            handler.variables['time'][idx] = delta_days

        handler.sync()
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def worker_multi_key(
    base_dir: Path,
    file_format: str,
    save_by: str,
    key_queues: Dict,
    aggregator_keys: List[str],
    var_dims_map: Dict[str, Tuple[str, ...]],
    dim_info: Dict[str, np.ndarray],
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
                continue

            # Decide current file period
            period = (timestamp.year, timestamp.month) if save_by == "month" else (timestamp.year, None)

            if current_periods[agg_key] != period:
                if handlers[agg_key]:
                    handlers[agg_key].close()
                current_periods[agg_key] = period

                file_name = f"{agg_key}_{period[0]}_rank{get_global_rank()}"
                if period[1]:
                    file_name += f"-{period[1]:02d}"
                file_name += f".{file_format}"
                file_path = base_dir / file_name
                
                # Retrieve dims for this aggregator key
                dims_for_var = var_dims_map[agg_key]
                handlers[agg_key] = open_file(
                    file_path, file_format, agg_key, data, dims_for_var, dim_info
                )

            # Write data
            write_data(
                handler=handlers[agg_key],
                file_format=file_format,
                var_name=agg_key,
                data=data,
                timestamp=timestamp
            )

        if len(done_keys) == len(aggregator_keys):
            break

    for agg_key in aggregator_keys:
        if handlers[agg_key]:
            handlers[agg_key].close()
        print(f"[{agg_key}] Writing completed.")


class DataDumper:
    def __init__(
        self,
        base_dir: str,
        statistics: Dict[str, List[str]],
        dim_info: Dict[str, np.ndarray],
        file_format: str = "nc",
        save_by: str = "year",
        max_queue_size: int = 100,
        num_workers: int = 1,
        disabled=False
    ):
        self.disabled = disabled
        if self.disabled:
            return
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.dim_info = dim_info
        self.file_format = file_format
        self.save_by = save_by
        self.num_workers = max(num_workers, 1)

        all_aggregator_keys = []
        self.var_dims_map = {}
        if statistics is None:
            self.disabled = True
            if get_global_rank() == 0:
                print("No statistics provided. No data will be dumped.")
            return
        
        for stat_name, vars_ in statistics.items():
            if vars_ is not None:
                for var in vars_:
                    agg_key = f"{var}_{stat_name}"
                    all_aggregator_keys.append(agg_key)
                    # Find the shape from LEGAL_AGG_ARRAYS if var in it
                    found = False
                    for shape_key, shape_vars in LEGAL_AGG_ARRAYS.items():
                        dims_tuple = shape_key[1]
                        if var in shape_vars:
                            self.var_dims_map[agg_key] = dims_tuple
                            found = True
                            break
                    if not found:
                        raise ValueError(f"Variable '{var}' not found in LEGAL_AGG_ARRAYS. Please check your statistics configuration.")
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
                    args=(
                        self.base_dir,
                        self.file_format,
                        self.save_by,
                        self.task_queues,
                        subset_keys,
                        self.var_dims_map,
                        self.dim_info,
                    ),
                )
                self._workers.append(worker)
        else:
            self.disabled = True
            print("No workers created. No data will be dumped.")

        for worker in self._workers:
            worker.start()

    @check_enabled
    def gather_stats(self, states: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        return gather_device_dicts(states, keys=self.aggregator_keys)

    @check_enabled
    def submit_data(self, timestamp: datetime, aggregator: Dict[str, Dict[str, np.ndarray]]):
        for agg_key, array_data in aggregator.items():
            self.task_queues[agg_key].put((timestamp, array_data))

    @check_enabled
    def close(self):
        """
        Wait for writers to finish and close all resources.
        """
        print("Closing DataDumper and waiting for workers to finish...")
        SENTINEL = (None, None)
        for agg_key in self.aggregator_keys:
            self.task_queues[agg_key].put(SENTINEL)
        for worker in self._workers:
            worker.join()


def test_data_dumper_nc():
    statistics = {
        "min":  ["bifurcation_cross_section_depth"],
        "max":  ["river_storage"],
        "mean": ["river_storage", "bifurcation_outflow"]
    }
    base_dir = "./test_nc_output_mod"
    dumper = DataDumper(
        base_dir=base_dir,
        statistics=statistics,
        dim_info={
            "catchment_save_idx": np.arange(3),
            "bifurcation_path_save_idx": np.arange(2),
        },
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
            "bifurcation_outflow_mean": np.random.rand(2, 4).astype('float32'),
            "bifurcation_cross_section_depth_min": np.random.rand(2, 4).astype('float32')
        }
        dumper.submit_data(t_now, aggregator_sample)

    # Switch to a new month
    t_new_month = datetime(2025, 2, 1)
    dumper.submit_data(t_new_month, {
        "river_storage_mean": np.random.rand(3).astype('float32'),
        "river_storage_max": np.random.rand(3).astype('float32'),
        "bifurcation_outflow_mean": np.random.rand(2, 4).astype('float32'),
        "bifurcation_cross_section_depth_min": np.random.rand(2, 4).astype('float32')
    })

    dumper.close()

    print("NetCDF test completed for modified DataDumper with multiple workers.")
    shutil.rmtree(base_dir)
    print("Temporary netCDF directory removed.\n")

def test_data_dumper_bin():
    statistics = {
        "min":  ["bifurcation_cross_section_depth"],
        "max":  ["river_storage"],
        "mean": ["river_storage", "bifurcation_outflow"]
    }
    base_dir = "./test_bin_output_mod"
    dumper = DataDumper(
        base_dir=base_dir,
        statistics=statistics,
        dim_info={
            "catchment_save_idx": np.arange(3),
            "bifurcation_path_save_idx": np.arange(2),
        },
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
            "bifurcation_outflow_mean": np.random.rand(2, 4).astype('float32'),
            "bifurcation_cross_section_depth_min": np.random.rand(2, 4).astype('float32')
        }   
        dumper.submit_data(t_now, aggregator_sample)

    # Switch to new month
    t_new_month = datetime(2025, 2, 1)
    dumper.submit_data(t_new_month, {
        "river_storage_mean": np.random.rand(3).astype('float32'),
        "river_storage_max": np.random.rand(3).astype('float32'),
        "bifurcation_outflow_mean": np.random.rand(2, 4).astype('float32'),
        "bifurcation_cross_section_depth_min": np.random.rand(2, 4).astype('float32')
    })

    time.sleep(1)
    dumper.close()
    print("Binary test completed for modified DataDumper with multiple workers.")
    shutil.rmtree(base_dir)
    print("Temporary BIN directory removed.\n")


if __name__ == "__main__":
    test_data_dumper_nc()
    test_data_dumper_bin()
