# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.ma as ma
import torch
from netCDF4 import Dataset


class InputProxy:
    """
    A proxy class for NetCDF input/output.
    Stores data in CPU memory (numpy arrays or torch tensors).
    """

    def __init__(
        self,
        data: Dict[str, Union[np.ndarray, torch.Tensor, float, int]],
        attrs: Optional[Dict[str, Any]] = None,
        dims: Optional[Dict[str, int]] = None,
    ):
        self.data = data
        self.attrs = attrs or {}
        self.dims = dims or {}

    @classmethod
    def from_nc(cls, file_path: Union[str, Path]) -> InputProxy:
        """
        Create an InputProxy from a NetCDF file.
        Reads all variables, dimensions, and attributes into memory.
        """
        data = {}
        attrs = {}
        dims = {}

        try:
            with Dataset(file_path, "r") as ds:
                # Read global attributes
                for attr_name in ds.ncattrs():
                    attrs[attr_name] = ds.getncattr(attr_name)

                # Read dimensions
                for dim_name, dim in ds.dimensions.items():
                    dims[dim_name] = dim.size

                # Read variables
                for var_name, var in ds.variables.items():
                    v = var[:]
                    if ma.isMaskedArray(v):
                        # Fill masked values conservatively
                        if np.issubdtype(v.dtype, np.floating):
                            data[var_name] = np.asarray(v.filled(np.nan))
                        else:
                            data[var_name] = np.asarray(v.filled(-1))
                    else:
                        data[var_name] = np.asarray(v)

        except Exception as e:
            raise RuntimeError(f"Error loading data from NetCDF {file_path}: {e}")

        return cls(data, attrs, dims)

    def to_nc(self, file_path: Union[str, Path], output_complevel: int = 4) -> None:
        """
        Write the stored data to a NetCDF file.
        """
        with Dataset(file_path, "w") as ds:
            # Write global attributes
            ds.setncatts(self.attrs)

            # Helper to ensure dimension exists
            def _ensure_dim(name: str, size: Optional[int], unlimited: bool = False) -> None:
                if name in ds.dimensions:
                    return
                ds.createDimension(name, None if unlimited else size)

            # Helper to infer and write variable
            def _infer_and_write_var(name: str, data: Any) -> None:
                # Convert to numpy if tensor
                if isinstance(data, torch.Tensor):
                    arr = data.detach().cpu().numpy()
                else:
                    arr = np.asarray(data)

                # Handle bool
                if arr.dtype == np.bool_:
                    vtype = "u1"
                    arr_to_write = arr.astype("u1")
                else:
                    vtype = arr.dtype
                    arr_to_write = arr

                # Define dimensions
                if arr.ndim == 0:
                    dims = ()
                else:
                    dims = []
                    for ax, sz in enumerate(arr.shape):
                        dim_name = f"{name}_dim{ax}"
                        _ensure_dim(dim_name, sz, unlimited=False)
                        dims.append(dim_name)

                # Create variable
                var = ds.createVariable(
                    name, vtype, dims, zlib=(output_complevel > 0), complevel=output_complevel
                )
                var[:] = arr_to_write

            # Write variables
            for name, val in self.data.items():
                _infer_and_write_var(name, val)

    @staticmethod
    def merge(
        output_path: Union[str, Path],
        rank_paths: List[Union[str, Path]],
        variable_group_mapping: Dict[str, str],
        output_complevel: int = 4,
    ) -> None:
        """
        Merge multiple per-rank NetCDF files into a single file.
        """
        offsets: Dict[str, int] = {}

        with Dataset(output_path, "w", format="NETCDF4") as merged_ds:
            merged_ds.title = "CaMa-Flood-GPU Model State (merged)"
            merged_ds.source = "InputProxy.merge"

            for r, rank_path in enumerate(rank_paths):
                if not Path(rank_path).exists():
                    raise FileNotFoundError(f"Missing file: {rank_path}")

                with Dataset(rank_path, "r") as rank_ds:
                    for var_name, var_in in rank_ds.variables.items():
                        is_distributed = var_name in variable_group_mapping
                        data = np.asarray(var_in[:])

                        # Define/create dims and variable in merged file
                        if var_name not in merged_ds.variables:
                            # Build dims
                            if data.ndim == 0:
                                dims = ()
                            else:
                                dims = []
                                for ax, sz in enumerate(data.shape):
                                    if is_distributed and ax == 0:
                                        dname = f"{var_name}_n"
                                        # Ensure dim exists
                                        if dname not in merged_ds.dimensions:
                                            merged_ds.createDimension(dname, None) # Unlimited
                                    else:
                                        dname = f"{var_name}_dim{ax}"
                                        if dname not in merged_ds.dimensions:
                                            merged_ds.createDimension(dname, sz)
                                    dims.append(dname)

                            # Dtype handling
                            if data.dtype == np.bool_:
                                vtype = "u1"
                            else:
                                vtype = data.dtype

                            kwargs = {}
                            if len(dims) > 0:
                                kwargs = dict(
                                    zlib=True, complevel=output_complevel, shuffle=True
                                )
                            merged_var = merged_ds.createVariable(
                                var_name, vtype, tuple(dims), **kwargs
                            )
                        else:
                            merged_var = merged_ds.variables[var_name]

                        # Write/append
                        if data.ndim == 0:
                            # Only copy from rank 0 for non-distributed scalars
                            if r == 0:
                                if data.dtype == np.bool_:
                                    merged_var.assignValue(data.astype("u1"))
                                else:
                                    merged_var.assignValue(data)
                        else:
                            if is_distributed:
                                off = offsets.get(var_name, 0)
                                n = data.shape[0]
                                if data.dtype == np.bool_:
                                    data = data.astype("u1")
                                merged_var[off : off + n, ...] = data
                                offsets[var_name] = off + n
                            else:
                                # Only copy non-distributed arrays from rank 0
                                if r == 0:
                                    if data.dtype == np.bool_:
                                        data = data.astype("u1")
                                    merged_var[:] = data

    def set_variable(self, name: str, value: Any, indices: Optional[Any] = None) -> None:
        """
        Set or update a variable.
        
        Args:
            name: Name of the variable.
            value: New value.
            indices: Optional indices to update specific elements. 
                     If None, replaces the entire variable.
        """
        if indices is not None:
            if name not in self.data:
                raise KeyError(f"Variable '{name}' not found in InputProxy, cannot update indices.")
            
            target = self.data[name]
            
            # Ensure target is mutable (numpy array or torch tensor)
            if not isinstance(target, (np.ndarray, torch.Tensor)):
                 raise TypeError(f"Variable '{name}' is of type {type(target)}, which does not support indexed assignment.")

            target[indices] = value
        else:
            self.data[name] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.data
