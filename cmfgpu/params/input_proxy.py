# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Tuple

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
        lazy: bool = True,
        file_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        visible_vars: Optional[Set[str]] = None,
        file_indices: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.data = data
        self.attrs = attrs or {}
        self.dims = dims or {}
        self.lazy = lazy
        self.file_path = file_path
        self.visible_vars = visible_vars or set(data.keys())
        self.file_map: Dict[str, str] = {}
        self.file_indices = file_indices or {}

    @staticmethod
    def _read_var_from_ds(ds: Dataset, var_name: str, indices: Any = None) -> np.ndarray:
        var = ds.variables[var_name]
        if indices is None:
            v = var[:]
        else:
            v = var[indices]

        if ma.isMaskedArray(v):
            # Fill masked values conservatively
            if np.issubdtype(v.dtype, np.floating):
                return np.asarray(v.filled(np.nan))
            else:
                return np.asarray(v.filled(-1))
        else:
            return np.asarray(v)

    @classmethod
    def from_nc(
        cls,
        file_path: Union[str, Path, List[Union[str, Path]]],
        lazy: bool = False,
        visible_vars: Optional[Union[List[str], Set[str]]] = None,
        align_on: Optional[str] = None,
    ) -> InputProxy:
        """
        Create an InputProxy from one or multiple NetCDF files.
        If multiple files are provided, checks for naming conflicts.
        Reads variables, dimensions, and attributes into memory or sets up lazy loading.

        Args:
            file_path: Path(s) to NetCDF file(s).
            lazy: If True, data is loaded on demand.
            visible_vars: Optional list/set of variable names to include. Others are ignored.
            align_on: Variable name to use for alignment. 
                      The FIRST file encountered containing this variable serves as the REFERENCE.
                      Subsequent files will be reordered to match the order of this variable in the reference file.
        """
        data = {}
        attrs = {}
        dims = {}
        found_vars = set()
        file_map = {}
        file_indices = {}

        # Normalize visible_vars
        if visible_vars is not None:
            visible_vars = set(visible_vars)

        # Normalize to list
        if isinstance(file_path, (str, Path)):
            file_paths = [file_path]
        else:
            file_paths = file_path

        reference_keys = None

        for fp in file_paths:
            path_str = str(fp)
            try:
                with Dataset(path_str, "r") as ds:
                    # Alignment logic
                    alignment_idx = None
                    if align_on:
                        # Only consider alignment if the key variable is "visible"
                        is_align_key_visible = (visible_vars is None) or (align_on in visible_vars)
                        
                        if is_align_key_visible and align_on in ds.variables:
                            current_keys = cls._read_var_from_ds(ds, align_on)
                            
                            if reference_keys is None:
                                # First file with the key sets the reference order
                                reference_keys = current_keys
                            else:
                                if len(current_keys) != len(reference_keys):
                                     raise ValueError(f"Alignment error: Variable '{align_on}' in '{path_str}' has different length ({len(current_keys)}) than reference ({len(reference_keys)}).")

                                # Subsequent files are aligned to the reference
                                sorter = np.argsort(current_keys)
                                sorted_keys = current_keys[sorter]

                                # Match keys strictly
                                insert_idx = np.searchsorted(sorted_keys, reference_keys)
                                
                                # Check range and equality
                                if np.any(insert_idx >= len(current_keys)):
                                     raise ValueError(f"Alignment failed: Key variable '{align_on}' in '{path_str}' mismatches reference keys (indices out of bounds).")
                                
                                matched = sorted_keys[insert_idx]
                                if not np.array_equal(matched, reference_keys):
                                     raise ValueError(f"Alignment failed: Key variable '{align_on}' in '{path_str}' does not strictly match reference keys.")
                                
                                # alignment_idx maps: index in Ref -> index in Current
                                alignment_idx = sorter[insert_idx]
                                file_indices[path_str] = alignment_idx

                    # Merge attributes
                    for attr_name in ds.ncattrs():
                        attrs[attr_name] = ds.getncattr(attr_name)

                    # Merge dimensions
                    for dim_name, dim in ds.dimensions.items():
                        dims[dim_name] = dim.size
                    
                    # Merge variables and check for conflicts
                    for var_name in ds.variables:
                        # Visibility check
                        if visible_vars is not None and var_name not in visible_vars:
                            continue

                        if var_name in found_vars:
                             prev_file = file_map.get(var_name)
                             # Skip conflict check for align key (we use the first one encountered)
                             if align_on and var_name == align_on:
                                 continue
                             
                             if prev_file != path_str:
                                 raise ValueError(f"Naming conflict: Variable '{var_name}' exists in both '{prev_file}' and '{path_str}'")
                        
                        found_vars.add(var_name)
                        file_map[var_name] = path_str

                        if not lazy:
                            val = cls._read_var_from_ds(ds, var_name)
                            # Apply alignment if eager loading
                            if alignment_idx is not None and val.ndim > 0 and val.shape[0] == len(alignment_idx):
                                val = val[alignment_idx]
                            data[var_name] = val

            except Exception as e:
                # Re-raise if it's our error
                if "Naming conflict" in str(e) or "Alignment failed" in str(e):
                    raise
                raise RuntimeError(f"Error loading data from NetCDF {path_str}: {e}")

        instance = cls(
            data, attrs, dims, lazy=lazy, file_path=file_path, visible_vars=found_vars, file_indices=file_indices
        )
        instance.file_map = file_map
        return instance

    def _resolve_target_path(self, key: str) -> str:
        target_path = None
        
        # Priority 1: Check internal file map (populated if from_nc with multiple files)
        if self.file_map and key in self.file_map:
            target_path = self.file_map[key]
        
        # Priority 2: If single file path is stored, use it
        elif self.file_path and isinstance(self.file_path, (str, Path)):
             target_path = self.file_path
             
        # Priority 3: If file_path is a list and map failed
        elif self.file_path and isinstance(self.file_path, list):
             raise RuntimeError(f"Variable '{key}' not found in file map, and multiple files provided. Cannot disambiguate source.")

        if not target_path:
             # Last resort check: if we only have one file in list?
             if isinstance(self.file_path, list) and len(self.file_path) == 1:
                 target_path = self.file_path[0]
             else:
                raise RuntimeError(f"Cannot lazy load variable '{key}': file source not mapped and file_path is ambiguous.")
        
        return str(target_path)

    def _load_var(self, key: str, indices: Any = None) -> np.ndarray:
        target_path = self._resolve_target_path(key)
        alignment_idx = self.file_indices.get(target_path)

        try:
            with Dataset(target_path, "r") as ds:
                if key not in ds.variables:
                     raise KeyError(f"Variable '{key}' not found in {target_path}")

                final_indices = indices
                if alignment_idx is not None:
                     v_shape = ds.variables[key].shape
                     if len(v_shape) > 0 and v_shape[0] == len(alignment_idx):
                         if indices is None:
                             final_indices = alignment_idx
                         elif isinstance(indices, tuple):
                             remapped = alignment_idx[indices[0]]
                             final_indices = (remapped, *indices[1:])
                         else:
                             final_indices = alignment_idx[indices]

                return self._read_var_from_ds(ds, key, indices=final_indices)
        except Exception as e:
            raise RuntimeError(f"Error lazy loading variable '{key}' from {target_path}: {e}")

    def get_subset(self, key: str, indices: Any) -> Any:
        """
        Get a subset of a variable.
        If the variable is in memory, slices it.
        If lazy, reads only the requested indices from the file.
        """
        if key in self.data:
            return self.data[key][indices]
        
        if self.lazy and key in self.visible_vars:
            return self._load_var(key, indices=indices)
            
        raise KeyError(f"Variable '{key}' not found in InputProxy.")

    def get_var_shape(self, key: str) -> Tuple[int, ...]:
        """
        Get the shape of a variable without loading it fully if possible.
        """
        # If in memory, return shape
        if key in self.data:
            val = self.data[key]
            # Handle list/scalar or other types if necessary, though data usually is ndarray/tensor
            if hasattr(val, "shape"):
                return tuple(val.shape)
            return ()

        # If lazy, peek at file
        if self.lazy and key in self.visible_vars:
            target_path = self._resolve_target_path(key)
            with Dataset(target_path, "r") as ds:
                if key not in ds.variables:
                     raise KeyError(f"Variable '{key}' not found in {target_path}")
                return tuple(ds.variables[key].shape)
        
        raise KeyError(f"Variable '{key}' not found in InputProxy.")


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
            for name in self.keys():
                val = self[name]
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
            # If lazy and not in memory yet, try to load it first so we can update it
            if name not in self.data and self.lazy and name in self.visible_vars:
                self.data[name] = self._load_var(name)

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
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key: str) -> Any:
        if key in self.data:
            return self.data[key]
        
        if self.lazy and key in self.visible_vars:
            return self._load_var(key)
            
        raise KeyError(f"Variable '{key}' not found in InputProxy.")

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.data or (self.lazy and key in self.visible_vars)

    def keys(self) -> Set[str]:
        return self.visible_vars.union(self.data.keys())
