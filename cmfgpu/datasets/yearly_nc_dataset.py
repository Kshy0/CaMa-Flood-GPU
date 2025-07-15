from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import Tuple

import numpy as np
from netCDF4 import Dataset

from cmfgpu.datasets.abstract_dataset import DefaultDataset


class YearlyNetCDFDataset(DefaultDataset):
    def __init__(self,
                 base_dir: str,
                 start_date: datetime,
                 end_date: datetime,
                 unit_factor: float = 1.0,
                 out_dtype: str = "float32",
                 var_name: str = "Runoff",
                 prefix: str = "e2o_ecmwf_wrr2_glob15_day_Runoff_",
                 suffix: str = ".nc",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_dir = base_dir
        self.start_date = start_date
        self.end_date = end_date
        self.unit_factor = unit_factor
        self.out_dtype = out_dtype
        self.var_name = var_name
        self.prefix = prefix
        self.suffix = suffix
    
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        filename = f"{self.prefix}{2000}{self.suffix}"
        file_path = Path(self.base_dir) / filename

        with Dataset(file_path, 'r') as dataset:
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
        
    def _init_dims(self, var) -> Tuple[int, int, int]:
        """
        Initialize and return dimension order for the variable.
        Returns: (time_idx, lat_idx, lon_idx)
        """
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
            return (time_idx, lat_idx, lon_idx)
        except Exception:
            raise ValueError(f"The dimensions of variable {self.var_name} do not contain information about time/lat/lon: {dim_names}")

    def _read_and_process_var(self, var, time_index: int) -> np.ndarray:
        dim_order = self._init_dims(var)
        data = var[time_index, :, :]

        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(0.0)
        else:
            data = np.nan_to_num(data, nan=0.0)
        
        if dim_order != (0, 1, 2):
            transpose_order = [dim_order.index(i) for i in (0, 1, 2)]
            data = np.transpose(data, axes=[transpose_order[1], transpose_order[2]])
        
        data = data[self.data_mask][self.local_runoff_indices] if self.local_runoff_indices is not None else data
        return data.astype(self.out_dtype)
    
    def get_data(self, current_time: datetime) -> np.ndarray:
        year_str = current_time.strftime("%Y")
        filename = f"{self.prefix}{year_str}{self.suffix}"
        file_path = Path(self.base_dir) / filename
        
        with Dataset(file_path, 'r') as dataset:
            time_index = current_time.timetuple().tm_yday - 1
            var = dataset.variables[self.var_name]
            data = self._read_and_process_var(var, time_index)
            return data / self.unit_factor
    
    @cached_property
    def data_mask(self) -> np.ndarray:
        """
        Returns a boolean mask where True indicates invalid/masked values.
        It uses the first day of the first year as a reference.
        """
        year = self.start_date.year
        filename = f"{self.prefix}{year}{self.suffix}"
        file_path = Path(self.base_dir) / filename

        with Dataset(file_path, 'r') as dataset:
            var = dataset.variables[self.var_name]
            dim_order = self._init_dims(var)
            data = var[0, :, :] 

            if isinstance(data, np.ma.MaskedArray):
                mask = np.array(~data.mask)
            else:
                mask = ~np.isnan(data)

            if dim_order != (0, 1, 2):
                transpose_order = [dim_order.index(i) for i in (0, 1, 2)]
                mask = np.transpose(mask, axes=[transpose_order[1], transpose_order[2]])

            return mask
    
    def get_time_by_index(self, idx: int) -> datetime:
        """
        Returns the datetime corresponding to the given index.
        """
        return self.start_date + timedelta(days=idx)

    def close(self) -> None:
        pass

    def __len__(self):
        """
        Returns the total number of samples in the dataset based on the time range.
        """
        return (self.end_date - self.start_date).days + 1
    
if __name__ == "__main__":
    dataset = YearlyNetCDFDataset(
        base_dir="/home/eat/cmf_v420_pkg/inp/test_15min_nc",
        start_date=datetime(2000, 1, 1),
        end_date=datetime(2000, 12, 31),
        prefix="e2o_ecmwf_wrr2_glob15_day_Runoff_",
        suffix=".nc",
        var_name="Runoff",
    )
    dataset.generate_runoff_mapping_table(
        map_dir="/home/eat/cmf_v420_pkg/map/glb_15min",
        out_dir="/home/eat/CaMa-Flood-GPU/inp/glb_15min",
        npz_file="runoff_mapping_nc.npz",
    )
