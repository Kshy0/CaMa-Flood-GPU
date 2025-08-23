"""
MERIT-based map parameter generation using Pydantic v2.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar, List, Optional

import numpy as np
from netCDF4 import Dataset
from pydantic import (BaseModel, ConfigDict, DirectoryPath, Field, FilePath,
                      model_validator)

from cmfgpu.params.utils import (compute_init_river_depth,
                                 min_cuts_for_balance, read_bifori,
                                 reorder_by_basin_size, topological_sort,
                                 trace_outlets_dict)
from cmfgpu.utils import binread, find_indices_in, read_map


class MERITMap(BaseModel):
    """
    MERIT-based map parameter generation class using netCDF4 (.nc).

    This class handles the loading and processing of MERIT Hydro global hydrography data
    for CaMa-Flood-GPU simulations. It follows the AbstractModule design pattern with
    proper field validation and type safety.

    Default configuration is for 15-minute resolution global maps.
    """

    # Pydantic configuration
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra='allow'
    )

    # === Input Configuration Fields ===
    map_dir: DirectoryPath = Field(
        description="Directory containing map files (nextxy.bin, rivlen.bin, etc.)"
    )

    out_dir: Path = Field(
        description="Output directory for generated input files"
    )

    out_file: str = Field(
        description="Name of the output NetCDF file for storing map parameters",
        default="parameters.nc"
    )

    bifori_file: Optional[FilePath] = Field(
        default=None,
        description="Path to original bifurcation table (bifori.txt)."
    )

    bif_levels_to_keep: int = Field(
        default=5,
        description="Keep first N levels from bifori; filter out paths with all zero widths in [1..N]"
    )
    
    gauge_file: Optional[FilePath] = Field(
        default=None,
        description="Path to gauge information file"
    )

    # === Physical Parameters ===
    gravity: float = Field(
        default=9.8,
        description="Gravitational acceleration [m/s²]"
    )

    river_mouth_distance: float = Field(
        default=10000.0,
        description="Distance to river mouth [m]"
    )

    visualized: bool = Field(
        default=True,
        description="Generate basin visualization"
    )

    simulate_gauged_basins_only: bool = Field(
        default=False,
        description="If True, only include basins that contain at least one gauge"
    )

    target_gpus: int = Field(
        default=4,
        description="Desired number of GPUs (MPI ranks) for load-balanced assignment"
    )

    mpi_balance_tolerance: float = Field(
        default=0.10,
        description="Allowed relative load deviation per GPU after LPT, e.g., 0.10 means ±10%"
    )

    # === File Mapping ===
    missing_value: ClassVar[float] = -9999.0
    output_required: ClassVar[List[str]] = [
        "catchment_id",
        "downstream_id",
        "nx",
        "ny",
        "catchment_x",
        "catchment_y",
        "catchment_basin_id",
        "basin_sizes",
        "num_basins",
        "gauge_mask",
        "river_depth",
        "river_width",
        "river_length",
        "river_height",
        "flood_depth_table",
        "catchment_elevation",
        "catchment_area",
        "downstream_distance",
        "river_storage",
        "river_mouth_id",
        "is_river_mouth",
        "is_reservoir",
    ]

    output_optional: ClassVar[List[str]] = [
        "bifurcation_path_id",
        "bifurcation_catchment_id",
        "bifurcation_downstream_id",
        "bifurcation_manning",
        "bifurcation_catchment_x",
        "bifurcation_downstream_x",
        "bifurcation_catchment_y",
        "bifurcation_downstream_y",
        "bifurcation_basin_id",
        "bifurcation_width",
        "bifurcation_length",
        "bifurcation_elevation",
    ]
    # === Data Type Configuration ===
    idx_precision: ClassVar[str] = "<i4"
    map_precision: ClassVar[str] = "<f4"
    numpy_precision: ClassVar[str] = "float32"

    def load_map_info(self) -> None:
        """Load map dimensions from mapdim.txt file."""
        mapdim_path = self.map_dir / "mapdim.txt"

        with open(mapdim_path, "r") as f:
            lines = f.readlines()
            self.nx = int(lines[0].split('!!')[0].strip())
            self.ny = int(lines[1].split('!!')[0].strip())
            self.num_flood_levels = int(lines[2].split('!!')[0].strip())

        print(f"Loaded map dimensions: nx={self.nx}, ny={self.ny}, num_flood_levels={self.num_flood_levels}")

    def load_catchment_id(self) -> None:
        """Load catchment IDs and connectivity from nextxy.bin."""
        nextxy_path = self.map_dir / "nextxy.bin"

        nextxy_data = binread(
            nextxy_path,
            (self.nx, self.ny, 2),
            dtype_str=self.idx_precision
        )

        self.map_shape = nextxy_data.shape[:2]

        # Find valid catchments
        catchment_x, catchment_y = np.where(nextxy_data[:, :, 0] != self.missing_value)
        next_catchment_x = nextxy_data[catchment_x, catchment_y, 0] - 1
        next_catchment_y = nextxy_data[catchment_x, catchment_y, 1] - 1

        # Create catchment IDs
        catchment_id = np.ravel_multi_index((catchment_x, catchment_y), self.map_shape)

        # Create downstream connectivity
        downstream_id = np.full_like(next_catchment_x, -1, dtype=np.int64)
        valid_next = (next_catchment_x >= 0) & (next_catchment_y >= 0)
        downstream_id[valid_next] = np.ravel_multi_index(
            (next_catchment_x[valid_next], next_catchment_y[valid_next]),
            self.map_shape
        )

        # Trace to river mouths
        river_mouth_id = trace_outlets_dict(catchment_id, downstream_id)

        # Store results
        self.catchment_x = catchment_x
        self.catchment_y = catchment_y
        self.catchment_id = catchment_id
        self.downstream_id = downstream_id
        self.river_mouth_id = river_mouth_id
        self.is_river_mouth = find_indices_in(downstream_id, catchment_id) < 0
        self.is_reservoir = np.zeros_like(catchment_id, dtype=bool)  # placeholder; set from data if available
        self.num_catchments = len(catchment_id)

        print(f"Loaded {len(catchment_id)} catchments")

    def load_gauge_id(self) -> None:
        """Load gauge information from gauge file."""
        if self.gauge_file is None:
            self.num_gauges = 0
            self.gauge_id = np.array([], dtype=np.int64)
            return
        gauge_id_set = set()
        self.gauge_info = {}

        with open(self.gauge_file, "r") as f:
            lines = f.readlines()

        # Skip header and process gauge data
        for line in lines[1:]:
            data = line.split()
            if len(data) < 14:
                raise ValueError(f"Invalid gauge data line: {line.strip()}")

            gauge_name = int(data[0])
            lat = float(data[1])
            lon = float(data[2])
            type_num = int(data[7])
            ix1, iy1 = int(data[8]) - 1, int(data[9]) - 1
            ix2, iy2 = int(data[10]) - 1, int(data[11]) - 1

            catchment_ids = []

            # Primary catchment
            if ix1 >= 0 and iy1 >= 0:
                catchment1 = np.ravel_multi_index((ix1, iy1), self.map_shape)
                catchment_ids.append(catchment1)
                gauge_id_set.add(catchment1)

            # Secondary catchment for type 2 gauges
            if type_num == 2 and ix2 >= 0 and iy2 >= 0:
                catchment2 = np.ravel_multi_index((ix2, iy2), self.map_shape)
                catchment_ids.append(catchment2)
                gauge_id_set.add(catchment2)

            if catchment_ids:
                self.gauge_info[gauge_name] = {
                    "upstream_id": catchment_ids,
                    "lat": lat,
                    "lon": lon
                }

        if len(self.gauge_info) == 0:
            print("Warning: No valid gauges found in the gauge file")

        self.gauge_id = np.array(sorted(gauge_id_set), dtype=np.int64)
        self.num_gauges = len(self.gauge_id)

        print(f"Loaded {len(self.gauge_info)} gauges covering {self.num_gauges} catchments")

    def _slice_arr(self, name: str, mask: np.ndarray):
        arr = getattr(self, name)
        setattr(self, name, arr[mask])

    def _finalize_connectivity(self, root_mouth: np.ndarray) -> None:
        """Finalize ordering and connectivity regardless of bifurcation availability.

        This performs:
        - Topological sort and basin-size reordering (stable assumptions for check_flow_direction).
        - Rebuild of downstream indices and river mouth flags.
        - Consistent remapping of per-catchment arrays.
        - Basin id creation and gauge mask refresh.
        - Optional alignment for bifurcation arrays if present.
        """
        # 1) Topological ordering and reordering by basin size
        topo_idx = topological_sort(self.catchment_id, self.downstream_id)
        sorted_idx, basin_sizes = reorder_by_basin_size(topo_idx, root_mouth)

        # 2) Apply ordering to per-catchment arrays
        self.catchment_id   = self.catchment_id[sorted_idx]
        self.downstream_id  = self.downstream_id[sorted_idx]
        self.catchment_x    = self.catchment_x[sorted_idx]
        self.catchment_y    = self.catchment_y[sorted_idx]
        self.river_mouth_id = self.river_mouth_id[sorted_idx]
        self.root_mouth     = root_mouth[sorted_idx]

        # 3) Basin stats and ids
        self.basin_sizes = basin_sizes
        self.num_basins  = len(basin_sizes)
        _, inverse_indices = np.unique(self.root_mouth, return_inverse=True)
        self.catchment_basin_id = inverse_indices.astype(np.int64)

        # 4) Downstream indices and mouth fix-up
        self.downstream_idx = find_indices_in(self.downstream_id, self.catchment_id)
        self.is_river_mouth = (self.downstream_idx < 0)
        # Mouths should point to themselves
        self.downstream_id[self.is_river_mouth] = self.catchment_id[self.is_river_mouth]

        # 5) Reservoir placeholder and gauge mask refresh
        self.is_reservoir = np.zeros_like(self.catchment_id, dtype=bool)

        self.gauge_mask = np.zeros(self.catchment_id.shape[0], dtype=bool)
        if self.num_gauges > 0:
            gi = find_indices_in(self.gauge_id, self.catchment_id)
            gi = gi[gi >= 0]
            self.gauge_mask[gi] = True

        # 6) If bifurcation arrays exist, align their basin ids with the new catchment ordering
        if self.num_bifurcation_paths > 0:
            temp_idx = find_indices_in(self.bifurcation_catchment_id, self.catchment_id)
            if np.any(temp_idx == -1):
                raise ValueError("Bifurcation catchment IDs do not match catchment IDs after reordering.")
            self.bifurcation_basin_id = self.catchment_basin_id[temp_idx]

    def load_bifurcation_parameters(self) -> None:
        if self.bifori_file is None:
            self.num_bifurcation_paths = 0
            self.root_mouth = self.river_mouth_id.copy()
            self._finalize_connectivity(root_mouth=self.root_mouth)
            return

        # Read maps needed by bifurcation parsing
        rivhgt_2d = read_map(self.map_dir / "rivhgt.bin", (self.nx, self.ny), precision=self.map_precision)
        pth_upst, pth_down, pth_dst, pth_wth, pth_elv = read_bifori(self.bifori_file, rivhgt_2d, self.bif_levels_to_keep)

        # Initialize arrays
        self.num_bifurcation_paths = len(pth_upst)
        self.bifurcation_manning = np.tile([0.03] + [0.1] * (self.bif_levels_to_keep - 1), (self.num_bifurcation_paths, 1)).astype(self.numpy_precision)
        pth_upst = np.array(pth_upst, dtype=np.int64)
        pth_down = np.array(pth_down, dtype=np.int64)
        self.bifurcation_catchment_x = pth_upst[:, 0]
        self.bifurcation_catchment_y = pth_upst[:, 1]
        self.bifurcation_downstream_x = pth_down[:, 0]
        self.bifurcation_downstream_y = pth_down[:, 1]
        self.bifurcation_width = np.array(pth_wth, dtype=self.numpy_precision)
        self.bifurcation_length = np.array(pth_dst, dtype=self.numpy_precision)
        self.bifurcation_elevation = np.array(pth_elv, dtype=self.numpy_precision)
        self.bifurcation_path_id = np.arange(self.num_bifurcation_paths, dtype=np.int64)
        self.bifurcation_catchment_id = np.ravel_multi_index((self.bifurcation_catchment_x, self.bifurcation_catchment_y), self.map_shape)
        self.bifurcation_downstream_id = np.ravel_multi_index((self.bifurcation_downstream_x, self.bifurcation_downstream_y), self.map_shape)

        # === Mouth-level planning (full-union → binary search minimal breaks if needed) BEFORE basin_sizes
        tmp_idx_up = find_indices_in(self.bifurcation_catchment_id, self.catchment_id)
        tmp_idx_dn = find_indices_in(self.bifurcation_downstream_id, self.catchment_id)
        ori_river_mouth_id = self.river_mouth_id[tmp_idx_up]
        bif_river_mouth_id = self.river_mouth_id[tmp_idx_dn]

        root_mouth, kept_mouth_pairs, union_report = min_cuts_for_balance(
            river_mouth_id=self.river_mouth_id,
            bif_from_mouth=ori_river_mouth_id,
            bif_to_mouth=bif_river_mouth_id,
            n_ranks=self.target_gpus,
            tol=self.mpi_balance_tolerance,
        )
        print(
            f"Cut {union_report['n_edges_removed']}/{union_report['n_edges_initial']} "
            f"inter-basin bifurcation links ({union_report['removed_ratio']*100:.2f}%)."
        )
        print(
            f"GPU loads (LPT over merged basins, ranks={self.target_gpus}, tol={self.mpi_balance_tolerance:.0%}): "
            f"{union_report['loads']}  "
            f"Balanced={union_report['balanced']}"
        )

        # === Cut the bifurcation paths that should be cut based on kept mouth pairs
        same_mouth = (ori_river_mouth_id == bif_river_mouth_id)

        mouths_all = np.unique(self.river_mouth_id)
        mouth_to_idx = {int(m): i for i, m in enumerate(mouths_all)}
        M = mouths_all.size

        ai = np.array([mouth_to_idx.get(int(x), -1) for x in ori_river_mouth_id], dtype=np.int64)
        bi = np.array([mouth_to_idx.get(int(x), -1) for x in bif_river_mouth_id], dtype=np.int64)
        valid_ab = (ai >= 0) & (bi >= 0)
        uu = np.minimum(ai, bi)
        vv = np.maximum(ai, bi)
        key_path = (uu.astype(np.int64) * np.int64(M)) + vv.astype(np.int64)

        if kept_mouth_pairs.shape[0] > 0:
            ku_full = np.array([mouth_to_idx.get(int(x), -1) for x in kept_mouth_pairs[:, 0]], dtype=np.int64)
            kv_full = np.array([mouth_to_idx.get(int(x), -1) for x in kept_mouth_pairs[:, 1]], dtype=np.int64)
            valid_k = (ku_full >= 0) & (kv_full >= 0)
            ku = np.minimum(ku_full[valid_k], kv_full[valid_k])
            kv = np.maximum(ku_full[valid_k], kv_full[valid_k])
            key_kept = (ku.astype(np.int64) * np.int64(M)) + kv.astype(np.int64)
        else:
            key_kept = np.zeros((0,), dtype=np.int64)

        keep_inter = np.isin(key_path, key_kept)
        keep_mask = same_mouth | (valid_ab & keep_inter)

        # Persist keep/cut info for visualization before pruning arrays
        n_before = int(self.num_bifurcation_paths)
        self.bifurcation_keep_mask_full = keep_mask.copy()
        removed_mask = ~keep_mask
        self.removed_bifurcation_catchment_x = self.bifurcation_catchment_x[removed_mask]
        self.removed_bifurcation_catchment_y = self.bifurcation_catchment_y[removed_mask]
        self.removed_bifurcation_downstream_x = self.bifurcation_downstream_x[removed_mask]
        self.removed_bifurcation_downstream_y = self.bifurcation_downstream_y[removed_mask]

        # Apply pruning
        if keep_mask.shape[0] != n_before:
            raise ValueError("Internal error: keep_mask length mismatch with bifurcation paths")

        for key in [
            "bifurcation_path_id",
            "bifurcation_catchment_id",
            "bifurcation_downstream_id",
            "bifurcation_manning",
            "bifurcation_catchment_x",
            "bifurcation_downstream_x",
            "bifurcation_catchment_y",
            "bifurcation_downstream_y",
            "bifurcation_width",
            "bifurcation_length",
            "bifurcation_elevation",
        ]:
            self._slice_arr(key, keep_mask)

        self.num_bifurcation_paths = int(np.sum(keep_mask))
        self.bifurcation_path_id = np.arange(self.num_bifurcation_paths, dtype=np.int64)
        n_cut = n_before - self.num_bifurcation_paths
        if n_cut > 0:
            print(f"Pruned {n_cut}/{n_before} bifurcation paths according to union planning ({(n_cut/n_before)*100:.2f}%).")

        # === Unify the code path: always finalize after planning/pruning
        self._finalize_connectivity(root_mouth=root_mouth)

    def filter_to_gauged_basins(self) -> None:
        """
        Optionally keep only basins that contain at least one gauge; update all dependent arrays.
        """
        if not self.simulate_gauged_basins_only:
            return

        if self.num_gauges ==0:
            raise ValueError("simulate_gauged_basins_only=True but no gauges were loaded.")

        # Build gauge mask in current ordering
        gauge_mask = np.zeros(self.catchment_id.shape[0], dtype=bool)
        gi = find_indices_in(self.gauge_id, self.catchment_id)
        gi = gi[gi >= 0]
        gauge_mask[gi] = True

        # Determine basins to keep, preserving their original relative order via first occurrence
        kept_basin_ids, kept_first_idx = np.unique(self.catchment_basin_id[gauge_mask], return_index=True)
        order_first = np.argsort(kept_first_idx)
        kept_basin_ids = kept_basin_ids[order_first]

        keep_mask = np.isin(self.catchment_basin_id, kept_basin_ids)

        # Slice catchment-level arrays that depend on catchment dimension
        for key in [
            "catchment_id",
            "downstream_id",
            "is_river_mouth",
            "catchment_x",
            "catchment_y",
            "root_mouth",
            "catchment_basin_id",
            "is_reservoir",
            "river_mouth_id",
        ]:
            self._slice_arr(key, keep_mask)

        # Update count
        self.num_catchments = int(self.catchment_id.shape[0])

        # Reindex basin IDs to contiguous [0..num_basins-1], preserving kept basin relative order.
        old = kept_basin_ids  # desired order
        order_val = np.argsort(old)        # order to sort by value for searchsorted
        old_sorted = old[order_val]
        pos = np.empty_like(order_val)     # map from value-sorted index -> desired-order index
        pos[order_val] = np.arange(order_val.size)
        idx_in_sorted = np.searchsorted(old_sorted, self.catchment_basin_id)
        self.catchment_basin_id = pos[idx_in_sorted].astype(np.int64)

        self.basin_sizes = np.bincount(self.catchment_basin_id, minlength=old.size).astype(np.int64)
        self.num_basins = int(self.basin_sizes.shape[0])
        self.downstream_idx = find_indices_in(self.downstream_id, self.catchment_id)
        self.downstream_idx[self.is_river_mouth] = -1

        # Refresh gauge_mask after filtering
        self.gauge_mask = np.zeros(self.num_catchments, dtype=bool)
        gi = find_indices_in(self.gauge_id, self.catchment_id)
        gi = gi[gi >= 0]
        self.gauge_mask[gi] = True

        # Filter bifurcation paths to kept basins and remap their basin ids to the new contiguous ids
        if self.num_bifurcation_paths > 0:
            # kept_basin_ids are in old numbering; select bifurcations within kept basins first
            keep_bif = np.isin(self.bifurcation_basin_id, kept_basin_ids)

            for key in [
                "bifurcation_path_id",
                "bifurcation_catchment_id",
                "bifurcation_downstream_id",
                "bifurcation_manning",
                "bifurcation_catchment_x",
                "bifurcation_downstream_x",
                "bifurcation_catchment_y",
                "bifurcation_downstream_y",
                "bifurcation_basin_id",
                "bifurcation_width",
                "bifurcation_length",
                "bifurcation_elevation",
            ]:
                self._slice_arr(key, keep_bif)

            self.num_bifurcation_paths = int(np.sum(keep_bif))
            self.bifurcation_path_id = np.arange(self.num_bifurcation_paths, dtype=np.int64)
            idx_in_sorted_bif = np.searchsorted(old_sorted, self.bifurcation_basin_id)
            self.bifurcation_basin_id = pos[idx_in_sorted_bif].astype(np.int64)

    def load_parameters(self) -> None:

        def _read_2d_map(filename: str) -> np.ndarray:
            """Read a 2D map variable and extract catchment values."""
            file_path = self.map_dir / filename
            data = read_map(file_path, (self.nx, self.ny), precision=self.map_precision)
            return data.astype(self.numpy_precision)[self.catchment_x, self.catchment_y]
        self.river_length = _read_2d_map("rivlen.bin")
        self.river_width = _read_2d_map("rivwth_gwdlr.bin")
        self.river_height = _read_2d_map("rivhgt.bin")
        self.catchment_elevation = _read_2d_map("elevtn.bin")
        self.catchment_area = _read_2d_map("ctmare.bin")
        self.downstream_distance = _read_2d_map("nxtdst.bin")
        self.downstream_distance[self.is_river_mouth] = self.river_mouth_distance

        data = read_map(self.map_dir / "fldhgt.bin", (self.nx, self.ny, self.num_flood_levels), precision=self.map_precision)
        flood_table = data.astype(self.numpy_precision)[self.catchment_x, self.catchment_y, :]
        self.flood_depth_table = np.hstack([
            np.zeros((self.num_catchments, 1), dtype=self.numpy_precision),
            flood_table
        ])

    def check_flow_direction(self) -> None:
        """Validate flow direction consistency."""
        non_mouth = ~self.is_river_mouth
        if not (self.downstream_idx[non_mouth] > np.flatnonzero(non_mouth)).all():
            raise ValueError("Flow direction error: downstream catchment should have higher index")

        if not (self.downstream_idx[self.is_river_mouth] == -1).all():
            raise ValueError("Flow direction error: river mouths should point to themselves")

    def init_river_depth(self) -> None:
        """Initialize river depth based on elevation gradients."""
        self.river_depth = compute_init_river_depth(
            self.catchment_elevation,
            self.river_height,
            self.downstream_idx
        ).astype(self.numpy_precision)

        self.river_storage = self.river_length * self.river_width * self.river_depth

    def create_nc_file(self) -> None:
        """Create a NetCDF4 file and store parameters."""
        nc_path = self.out_dir / self.out_file
        with Dataset(nc_path, 'w', format='NETCDF4') as ds:
            # Global attributes
            ds.title = "MERIT-based map parameters for CaMa-Flood-GPU"
            ds.history = "Created by CaMa-Flood-GPU netCDF4 writer"

            # Define dimensions
            ds.createDimension("catchment", self.num_catchments)
            ds.createDimension("basin", self.num_basins)
            ds.createDimension("bifurcation_path", self.num_bifurcation_paths)

            # Flood level dimension (includes the leading zero column)
            ds.createDimension("flood_level", self.num_flood_levels + 1)

            # Bifurcation level dimension
            ds.createDimension("bifurcation_level", self.bif_levels_to_keep)

            # Helper to map data shapes to dims
            def infer_dims(arr: np.ndarray):
                shape = arr.shape
                if shape == ():  # scalar
                    return ()
                if len(shape) == 1:
                    if shape[0] == self.num_catchments:
                        return ("catchment",)
                    if shape[0] == self.num_basins:
                        return ("basin",)
                    if shape[0] == self.num_bifurcation_paths:
                        return ("bifurcation_path",)
                    if shape[0] == self.flood_depth_table.shape[1]:
                        return ("flood_level",)
                elif len(shape) == 2:
                    if shape[0] == self.num_catchments and shape[1] == self.flood_depth_table.shape[1]:
                        return ("catchment", "flood_level")
                    if shape[0] == self.num_bifurcation_paths and shape[1] == self.bifurcation_width.shape[1]:
                        return ("bifurcation_path", "bifurcation_level")
                # Fallback: create anonymous dims for unexpected shapes
                dims = []
                for i, n in enumerate(shape):
                    dim_name = f"dim_{i}_{n}"
                    if dim_name not in ds.dimensions:
                        ds.createDimension(dim_name, n)
                    dims.append(dim_name)
                return tuple(dims)

            # Write variables
            def _vars_to_write():
                for key in self.output_required:
                    if not hasattr(self, key):
                        raise ValueError(f"Missing required field: {key}")
                    yield key
                for key in getattr(self, "output_optional", []):
                    if hasattr(self, key):
                        yield key

            for key in _vars_to_write():
                data = getattr(self, key)
                arr = np.array(data)

                dims = infer_dims(arr)

                # Choose dtype; convert booleans to unsigned byte (u1) for compatibility
                if arr.dtype == np.bool_:
                    vdtype = 'u1'
                    arr_to_write = arr.astype('u1')
                else:
                    vdtype = arr.dtype
                    arr_to_write = arr

                kwargs = {}
                if len(dims) > 0:
                    kwargs = dict(zlib=True, complevel=4, shuffle=True)

                var = ds.createVariable(key, vdtype, dims, **kwargs)
                var[:] = arr_to_write

            # Save some useful scalar attributes as global too
            ds.setncattr("nx", int(self.nx))
            ds.setncattr("ny", int(self.ny))
            ds.setncattr("num_basins", int(self.num_basins))

    def visualize_basins(self) -> None:
        """Generate basin visualization if requested, including removed bifurcation paths."""
        if self.visualized is False:
            return
        try:
            import matplotlib.pyplot as plt
            from matplotlib.collections import LineCollection
            from matplotlib.colors import ListedColormap
        except ImportError:
            print("matplotlib not available, skipping basin visualization")
            return
        if self.num_catchments > 1e7:
            print("Too many catchments to visualize, skipping.")
            return
        def generate_random_colors(N, avoid_rgb_colors):
            colors = []
            avoid_rgb_colors = np.array(avoid_rgb_colors)

            while len(colors) < N:
                color = np.random.rand(3)
                if np.all(np.linalg.norm(avoid_rgb_colors - color, axis=1) > 0.7):
                    colors.append(color)

            return np.array(colors)

        special_colors = [
            (0, 1, 0),      # gauges pure green
            (0, 0, 1),      # bifurcations pure blue
            (1, 0, 0) # removed paths red
        ]

        basin_map = np.full(self.map_shape, fill_value=-1, dtype=int)
        unique_roots = np.unique(self.root_mouth)
        root_to_basin = {root: i for i, root in enumerate(unique_roots)}
        basin_ids = np.array([root_to_basin[r] for r in self.root_mouth])
        basin_map[self.catchment_x, self.catchment_y] = basin_ids

        num_basins = len(unique_roots)
        basin_colors = generate_random_colors(num_basins, avoid_rgb_colors=special_colors)
        all_colors = np.vstack(([1, 1, 1], basin_colors))
        cmap = ListedColormap(all_colors)

        masked_map = np.ma.masked_where(basin_map == -1, basin_map + 1)

        plt.figure(figsize=(12, 10))
        plt.imshow(masked_map.T, origin='upper', cmap=cmap, interpolation='nearest')
        plt.title(f"MERIT Global Basins with Bifurcation Paths")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(False)

        # Plot gauges if available
        if self.num_gauges > 0:
            gauge_catchment_ids = []
            for info in self.gauge_info.values():
                gauge_catchment_ids.extend(info["upstream_id"])
            gauge_catchment_ids = np.unique(gauge_catchment_ids)

            catchment_id_to_idx = {cid: idx for idx, cid in enumerate(self.catchment_id)}
            gauge_indices = [catchment_id_to_idx[cid] for cid in gauge_catchment_ids if cid in catchment_id_to_idx]

            if gauge_indices:
                gauge_x = self.catchment_x[gauge_indices]
                gauge_y = self.catchment_y[gauge_indices]
                plt.scatter(gauge_x, gauge_y, c='#00FF00', s=0.5, label='Gauges')

        # Plot kept bifurcation paths (after pruning, arrays contain only kept)
        if self.num_bifurcation_paths > 0 and self.num_bifurcation_paths < 3e6:
            x1 = self.bifurcation_catchment_x
            y1 = self.bifurcation_catchment_y
            x2 = self.bifurcation_downstream_x
            y2 = self.bifurcation_downstream_y
            # avoid wrap-around across the dateline
            mask_keep = np.abs(x1 - x2) <= self.nx / 2
            x1k = x1[mask_keep]
            y1k = y1[mask_keep]
            x2k = x2[mask_keep]
            y2k = y2[mask_keep]
            if x1k.size > 0:
                line_segments_keep = np.array([[[x1k[i], y1k[i]], [x2k[i], y2k[i]]] for i in range(len(x1k))])
                kept_lines = LineCollection(line_segments_keep, colors='#0000FF', linestyles='--', linewidths=0.5, alpha=0.6)
                plt.gca().add_collection(kept_lines)
                plt.plot([], [], color='#0000FF', linestyle='--', linewidth=0.5, alpha=0.6, label='Bifurcation Paths')

        # Plot removed bifurcation paths, if any were pruned
        if hasattr(self, "removed_bifurcation_catchment_x") and self.removed_bifurcation_catchment_x.size > 0:
            rx1 = self.removed_bifurcation_catchment_x
            ry1 = self.removed_bifurcation_catchment_y
            rx2 = self.removed_bifurcation_downstream_x
            ry2 = self.removed_bifurcation_downstream_y
            mask_cut = np.abs(rx1 - rx2) <= self.nx / 2
            rx1 = rx1[mask_cut]
            ry1 = ry1[mask_cut]
            rx2 = rx2[mask_cut]
            ry2 = ry2[mask_cut]
            if rx1.size > 0:
                line_segments_removed = np.array([[[rx1[i], ry1[i]], [rx2[i], ry2[i]]] for i in range(len(rx1))])
                removed_lines = LineCollection(line_segments_removed, colors='#FF0000', linestyles=':', linewidths=1, alpha=0.5)
                plt.gca().add_collection(removed_lines)
                plt.plot([], [], color='#FF0000', linestyle=':', linewidth=1, alpha=0.7, label='Bifurcation Paths (removed)')

        handles, labels = plt.gca().get_legend_handles_labels()
        if labels:  
            plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(self.out_dir / f"basin_map.png", dpi=600, bbox_inches='tight')
        plt.close()

        print(f"Saved basin visualization to {self.out_dir / f'basin_map.png'}")

    def print_summary(self) -> None:
        print(f"MERIT Map Processing Summary:")
        print(f"Grid dimensions          : {self.nx} x {self.ny}")
        print(f"Number of basins         : {self.num_basins}")
        print(f"Number of catchments     : {self.num_catchments}")
        print(f"Number of reservoirs     : {self.is_reservoir.sum()}")
        print(f"Number of bifurcation paths : {self.num_bifurcation_paths}")
        print(f"Number of gauges         : {self.num_gauges}")
        print(f"Gauge catchments         : {len(self.gauge_id)}")
        print(f"Output directory         : {self.out_dir}")

    def build_input(self) -> None:
        print(f"Starting MERIT Map processing pipeline")
        print(f"Map directory: {self.map_dir}")
        print(f"Output file: {self.out_dir / self.out_file}")

        # Core processing steps
        self.load_map_info()
        self.load_catchment_id()
        self.load_gauge_id()
        self.load_bifurcation_parameters()
        self.filter_to_gauged_basins()
        self.load_parameters()
        self.check_flow_direction()
        self.init_river_depth()
        self.create_nc_file()
        self.visualize_basins()
        self.print_summary()

        print("MERIT Map processing pipeline completed successfully")

    @model_validator(mode="after")
    def validate_out_dir(self) -> MERITMap:
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True, exist_ok=True)
        return self

if __name__ == "__main__":
    map_resolution = "glb_15min"
    merit_map = MERITMap(
        map_dir=f"/home/eat/cmf_v420_pkg/map/{map_resolution}",
        out_dir=Path(f"/home/eat/CaMa-Flood-GPU/inp/{map_resolution}"),
        bifori_file=f"/home/eat/cmf_v420_pkg/map/{map_resolution}/bifori.txt",
        gauge_file=f"/home/eat/cmf_v420_pkg/map/{map_resolution}/GRDC_alloc.txt",
        visualized=True,
        simulate_gauged_basins_only=False,
        bif_levels_to_keep=5,
        target_gpus=1,
        out_file="parameters.nc",
    )
    merit_map.build_input()
