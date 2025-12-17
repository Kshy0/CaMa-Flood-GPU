# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
MERIT-based map parameter generation using Pydantic v2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence

import numpy as np
from netCDF4 import Dataset
from pydantic import (BaseModel, ConfigDict, DirectoryPath, Field, FilePath,
                      model_validator)

from cmfgpu.params.utils import (compute_init_river_depth, read_bifori,
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
    
    basin_use_file: bool = Field(
        default=False,
        description="If True, use basin.bin file to cut bifurcations crossing basin boundaries."
    )

    levee_flag: bool = Field(
        default=False,
        description="If True, merge levee data into map parameters."
    )
    
    gauge_file: Optional[FilePath] = Field(
        default=None,
        description="Path to gauge information file"
    )

    skip_secondary_gauges: bool = Field(
        default=False,
        description=(
            "If True, skip any gauge file row whose secondary coordinates (ix2, iy2) "
            "are valid. This removes multi-catchment (type 2) gauge entries entirely instead "
            "of registering their primary cell."
        )
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

    only_save_pois: bool = Field(
        default=False,
        description="If True, only save catchments that are points of interest (POI); otherwise save all catchments."
    )
    # Allow selecting a minimal subset of basins via points of interest (POI)
    # Structure example:
    # {
    #   "gauges": "all" | ["1234", "5678"],  # gauge IDs as strings; "all" keeps basins with any loaded gauge
    #   "coords": [(x, y), ...],               # 0-based grid indices; will be validated and mapped to catchment IDs
    #   "catchments": [int, int, ...]          # explicit catchment_id list
    # }
    points_of_interest: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional POI selector to reduce simulated area: gauges ('all' or string IDs), "
            "coords as (x,y) pairs, and explicit catchment IDs."
        ),
    )

    target_gpus: int = Field(
        default=4,
        description="Desired number of GPUs (MPI ranks) for load-balanced assignment"
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
        "catchment_save_mask",
        "river_depth",
        "river_width",
        "river_length",
        "river_height",
        "flood_depth_table",
        "catchment_elevation",
        "catchment_area",
        "upstream_area",
        "downstream_distance",
        "river_storage",
        "river_mouth_id",
        "is_river_mouth",
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
        "levee_id",
        "levee_catchment_id",
        "levee_catchment_x",
        "levee_catchment_y",
        "levee_fraction",
        "levee_crown_height",
        "levee_basin_id",
        "longitude",
        "latitude",
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
        self.num_catchments = len(catchment_id)
        self.river_mouth_id = river_mouth_id
        self.is_river_mouth = (self.downstream_id < 0)
        self.catchment_save_mask = np.ones(self.num_catchments, dtype=bool)
        

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

            # Option: exclude entire line if secondary coords exist
            if self.skip_secondary_gauges and (ix2 >= 0 and iy2 >= 0):
                continue

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

        # 5) Mask
        self.catchment_save_mask = self.catchment_save_mask[sorted_idx]

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

        # Handle basin-based pruning if basin_use_file is True
        n_before = int(self.num_bifurcation_paths)
        if self.basin_use_file:
            basin_file = self.map_dir / "basin.bin"
            # Read basin.bin
            basin_data = read_map(basin_file, (self.nx, self.ny), precision=self.idx_precision)
            # Get basin IDs for bifurcation endpoints
            bifurcation_up_basin = basin_data[self.bifurcation_catchment_x, self.bifurcation_catchment_y]
            bifurcation_down_basin = basin_data[self.bifurcation_downstream_x, self.bifurcation_downstream_y]
            # Keep only intra-basin bifurcations
            keep_mask = (bifurcation_up_basin == bifurcation_down_basin)
            # Set removed for visualization
            removed_mask = ~keep_mask
            self.removed_bifurcation_catchment_x = self.bifurcation_catchment_x[removed_mask]
            self.removed_bifurcation_catchment_y = self.bifurcation_catchment_y[removed_mask]
            self.removed_bifurcation_downstream_x = self.bifurcation_downstream_x[removed_mask]
            self.removed_bifurcation_downstream_y = self.bifurcation_downstream_y[removed_mask]
        else:
            # Calculate river mouths for bifurcation endpoints to allow basin merging
            tmp_idx_up = find_indices_in(self.bifurcation_catchment_id, self.catchment_id)
            tmp_idx_dn = find_indices_in(self.bifurcation_downstream_id, self.catchment_id)
            
            # Filter valid indices (though they should be valid if loaded correctly)
            valid_bif = (tmp_idx_up >= 0) & (tmp_idx_dn >= 0)
            
            if np.any(valid_bif):
                ori_river_mouth_id = self.river_mouth_id[tmp_idx_up[valid_bif]]
                bif_river_mouth_id = self.river_mouth_id[tmp_idx_dn[valid_bif]]
                
                # Merge basins using Union-Find
                parent = {m: m for m in np.unique(self.river_mouth_id)}
                
                def find(x):
                    if parent[x] != x:
                        parent[x] = find(parent[x])
                    return parent[x]
                
                def union(x, y):
                    rootX = find(x)
                    rootY = find(y)
                    if rootX != rootY:
                        parent[rootX] = rootY
                
                for m1, m2 in zip(ori_river_mouth_id, bif_river_mouth_id):
                    union(m1, m2)
                
                # Update root_mouth based on merged results
                self.root_mouth = np.array([find(m) for m in self.river_mouth_id], dtype=np.int64)
            else:
                self.root_mouth = self.river_mouth_id.copy()

            keep_mask = np.ones(self.num_bifurcation_paths, dtype=bool)
            # No removed
            self.removed_bifurcation_catchment_x = np.array([], dtype=np.int64)
            self.removed_bifurcation_catchment_y = np.array([], dtype=np.int64)
            self.removed_bifurcation_downstream_x = np.array([], dtype=np.int64)
            self.removed_bifurcation_downstream_y = np.array([], dtype=np.int64)

        # Apply pruning
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
            print(f"Pruned {n_cut}/{n_before} bifurcation paths crossing basin boundaries ({(n_cut/n_before)*100:.2f}%).")

        # Finalize connectivity
        self._finalize_connectivity(root_mouth=self.root_mouth)

        # Always compute and report load distribution (LPT over basins -> simulated GPU assignment).
        # If `basin_use_file` was used earlier we already printed pruning info; regardless, always
        # produce the per-rank loads and warn on imbalance (>10%).
        def _lpt_schedule(sizes, n_bins):
            bins = [0] * max(1, int(n_bins))
            for size in sorted(list(sizes), reverse=True):
                min_bin = min(range(len(bins)), key=lambda i: bins[i])
                bins[min_bin] += int(size)
            return bins

        loads = _lpt_schedule(self.basin_sizes, self.target_gpus)
        print(f"GPU loads (LPT over basins, ranks={self.target_gpus}): {loads}")

        # Check imbalance relative to average; warn if exceed 10%
        total_load = sum(loads)
        if total_load > 0:
            avg_load = total_load / max(1, int(self.target_gpus))
            max_dev = max(abs(l - avg_load) / avg_load for l in loads)
        else:
            avg_load = 0.0
            max_dev = 0.0

        imbalance_threshold = 0.10
        if max_dev > imbalance_threshold:
            print(
                f"Warning: Load imbalance detected across {self.target_gpus} ranks. "
                f"Max deviation = {max_dev:.2%}, loads = {loads}, average = {avg_load:.2f}"
            )

        # If basin_use_file is True, also compute and report the unpruned case
        if self.basin_use_file:
            # Compute unpruned basin sizes (allowing bifurcations to merge basins)
            def compute_merged_basin_sizes(river_mouth_id, bifurcation_catchment_id, bifurcation_downstream_id, catchment_id):
                from collections import defaultdict
                parent = {}
                def find(x):
                    if x not in parent:
                        parent[x] = x
                    if parent[x] != x:
                        parent[x] = find(parent[x])
                    return parent[x]
                def union(x, y):
                    px = find(x)
                    py = find(y)
                    if px != py:
                        parent[px] = py
                catchment_to_basin = {cid: rid for cid, rid in zip(catchment_id, river_mouth_id)}
                for up_cid, down_cid in zip(bifurcation_catchment_id, bifurcation_downstream_id):
                    if up_cid in catchment_to_basin and down_cid in catchment_to_basin:
                        union(catchment_to_basin[up_cid], catchment_to_basin[down_cid])
                basin_to_size = defaultdict(int)
                for cid in catchment_id:
                    root = find(catchment_to_basin[cid])
                    basin_to_size[root] += 1
                return list(basin_to_size.values())

            # Use original bifurcation arrays before pruning
            orig_bif_catchment_id = np.ravel_multi_index((pth_upst[:, 0], pth_upst[:, 1]), self.map_shape)
            orig_bif_downstream_id = np.ravel_multi_index((pth_down[:, 0], pth_down[:, 1]), self.map_shape)
            unpruned_basin_sizes = compute_merged_basin_sizes(self.river_mouth_id, orig_bif_catchment_id, orig_bif_downstream_id, self.catchment_id)
            unpruned_loads = _lpt_schedule(unpruned_basin_sizes, self.target_gpus)
            print(f"Unpruned GPU loads (LPT over merged basins, ranks={self.target_gpus}): {unpruned_loads}")

            # Check imbalance for unpruned
            total_unpruned_load = sum(unpruned_loads)
            if total_unpruned_load > 0:
                avg_unpruned_load = total_unpruned_load / max(1, int(self.target_gpus))
                max_unpruned_dev = max(abs(l - avg_unpruned_load) / avg_unpruned_load for l in unpruned_loads)
            else:
                avg_unpruned_load = 0.0
                max_unpruned_dev = 0.0

            if max_unpruned_dev > imbalance_threshold:
                print(
                    f"Warning: Load imbalance detected in unpruned case across {self.target_gpus} ranks. "
                    f"Max deviation = {max_unpruned_dev:.2%}, loads = {unpruned_loads}, average = {avg_unpruned_load:.2f}"
                )

    def filter_to_poi_basins(self) -> None:
        """
        Restrict simulation to user-provided points of interest (POI) to minimize regions.

        Behavior:
        - If points_of_interest is not provided: no filtering (keep all basins).
        - Else: build a target catchment set from POI: gauge IDs (strings or 'all'), coords, and/or catchment IDs.
          Keep the union of basins that contain any loaded gauge (when gauges='all') and/or contain any target catchment.
        """
        poi = self.points_of_interest
        if not poi:
            return

        # POI-driven selection only
        target_cids: List[int] = []

        # 1) Gauges (strings) -> use self.gauge_info keys (parsed as int when loading)
        gauges_val = poi.get("gauges")
        if gauges_val is not None:
            if isinstance(gauges_val, (str, bytes)):
                if str(gauges_val).lower() != "all":
                    raise ValueError("If gauges is a string, it must be 'all'.")
                # Use all loaded gauge catchments as targets; basins will be computed later from target_cids
                if getattr(self, "num_gauges", 0) == 0:
                    raise ValueError("gauges='all' specified but no gauges were loaded; provide a gauge_file.")
                target_cids.extend(self.gauge_id.tolist())
            else:
                if not isinstance(gauges_val, Sequence):
                    raise ValueError("points_of_interest['gauges'] must be a sequence of string IDs or 'all'.")
                if not hasattr(self, "gauge_info") or len(self.gauge_info) == 0:
                    raise ValueError("Gauge list provided but no gauge_info available; ensure gauge_file is set and loaded.")
                for g in gauges_val:
                    try:
                        gid_int = int(str(g))
                    except Exception:
                        raise ValueError(f"Gauge id '{g}' is not numeric; expected stringified integer.")
                    if gid_int not in self.gauge_info:
                        raise ValueError(f"Gauge id '{g}' not found in loaded gauge_info.")
                    target_cids.extend(self.gauge_info[gid_int]["upstream_id"])  # may include 1-2 catchments

        # 2) Coordinates -> map (x,y) to catchment_id and validate
        coords_val = poi.get("coords")
        if coords_val is not None:
            if not isinstance(coords_val, Sequence) or (len(coords_val) > 0 and not isinstance(coords_val[0], (list, tuple))):
                raise ValueError("points_of_interest['coords'] must be a sequence of (x, y) pairs.")
            valid_cid_set = set(map(int, self.catchment_id.tolist()))
            for pair in coords_val:
                if len(pair) != 2:
                    raise ValueError(f"Invalid coord {pair}; expected (x, y).")
                x, y = int(pair[0]), int(pair[1])
                if not (0 <= x < self.nx and 0 <= y < self.ny):
                    raise ValueError(f"Coord {(x, y)} out of bounds for grid ({self.nx}, {self.ny}).")
                cid_xy = int(np.ravel_multi_index((x, y), self.map_shape))
                if cid_xy not in valid_cid_set:
                    raise ValueError(f"Coord {(x, y)} is not a valid catchment cell in current map (masked/missing).")
                target_cids.append(cid_xy)

        # 3) Explicit catchment IDs
        catches_val = poi.get("catchments")
        if catches_val is not None:
            if not isinstance(catches_val, Sequence) or (len(catches_val) > 0 and isinstance(catches_val, (str, bytes))):
                raise ValueError("points_of_interest['catchments'] must be a sequence of integers.")
            valid_cid_set = set(map(int, self.catchment_id.tolist()))
            for c in catches_val:
                cid = int(c)
                if cid not in valid_cid_set:
                    raise ValueError(f"catchment_id {cid} not present in current map.")
                target_cids.append(cid)

        # Cross-check: if both coords and catchments given, every coord's mapped cid must be present in provided catchments
        if coords_val is not None and catches_val is not None:
            cid_from_coords: List[int] = []
            for pair in coords_val:
                x, y = int(pair[0]), int(pair[1])
                cid_from_coords.append(int(np.ravel_multi_index((x, y), self.map_shape)))
            provided_cids = set(int(c) for c in catches_val)
            for cc in cid_from_coords:
                if cc not in provided_cids:
                    raise ValueError(
                        f"Coord-derived catchment_id {cc} is not in provided points_of_interest['catchments']."
                    )

        # Collect basins from explicit target catchments (must be non-empty)
        self.target_cids = target_cids = np.unique(np.array(target_cids, dtype=np.int64))
        if len(self.target_cids) == 0:
            raise ValueError("points_of_interest produced an empty target set; nothing to keep.")
        target_idx = find_indices_in(self.target_cids, self.catchment_id)
        if np.any(target_idx < 0):
            missing = self.target_cids[target_idx < 0]
            raise ValueError(f"Internal error: target catchment IDs {missing} not found in current catchment_id array.")
        mask_target = np.zeros(self.catchment_id.shape[0], dtype=bool)
        mask_target[target_idx] = True
        kept_basin_ids_t, kept_first_idx_t = np.unique(self.catchment_basin_id[mask_target], return_index=True)
        order_first_t = np.argsort(kept_first_idx_t)
        kept_basin_ids = kept_basin_ids_t[order_first_t]



        # Apply common filtering given kept_basin_ids
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

        # Create catchment_save_mask after filtering (mark target catchments)
        if self.only_save_pois:
            self.catchment_save_mask = np.zeros(self.num_catchments, dtype=bool)
            ti = find_indices_in(self.target_cids, self.catchment_id)
            self.catchment_save_mask[ti] = True

        # Filter bifurcation paths to kept basins and remap their basin ids to the new contiguous ids
        if self.num_bifurcation_paths > 0:
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
        self.upstream_area = _read_2d_map("uparea.bin")
        self.downstream_distance = _read_2d_map("nxtdst.bin")
        self.downstream_distance[self.is_river_mouth] = self.river_mouth_distance

        lonlat_path = self.map_dir / "lonlat.bin"
        if lonlat_path.exists():
            print(f"Loading lon/lat from {lonlat_path}")
            lonlat_data = binread(
                lonlat_path,
                (self.nx, self.ny, 2),
                dtype_str=self.map_precision
            )
            self.longitude = lonlat_data[self.catchment_x, self.catchment_y, 0].astype(self.numpy_precision)
            self.latitude = lonlat_data[self.catchment_x, self.catchment_y, 1].astype(self.numpy_precision)

        if self.levee_flag:
            levee_crown_height = _read_2d_map("levhgt.bin")
            levee_fraction = _read_2d_map("levfrc.bin")
            levee_mask = (levee_fraction >= 0) & (levee_fraction < 1.0) & (levee_crown_height > 0)
            self.num_levees = int(np.sum(levee_mask))
            self.levee_id = np.arange(self.num_levees, dtype=np.int64)
            self.levee_catchment_id = self.catchment_id[levee_mask]
            self.levee_catchment_x = self.catchment_x[levee_mask]
            self.levee_catchment_y = self.catchment_y[levee_mask]
            self.levee_crown_height = levee_crown_height[levee_mask]
            self.levee_fraction = levee_fraction[levee_mask]
            self.levee_basin_id = self.catchment_basin_id[levee_mask]

        data = read_map(self.map_dir / "fldhgt.bin", (self.nx, self.ny, self.num_flood_levels), precision=self.map_precision)
        self.flood_depth_table = data.astype(self.numpy_precision)[self.catchment_x, self.catchment_y, :]

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

            # Flood level dimension
            ds.createDimension("flood_level", self.num_flood_levels)

            # Bifurcation level dimension
            ds.createDimension("bifurcation_level", self.bif_levels_to_keep)

            if self.levee_flag:
                ds.createDimension("levee", self.num_levees)

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
                    if self.levee_flag and shape[0] == self.num_levees:
                        return ("levee",)
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

    def visualize_basins(
        self,
        interactive_basin_picker: bool = False,
        visualize_gauges: bool = True,
        visualize_bifurcations: bool = True,
        visualize_removed_bifurcations: bool = True,
        visualize_levees: bool = True
    ) -> None:
        """Generate basin visualization if requested, including removed bifurcation paths.

        Behavior changes:
        - Automatically crops the plot to the minimal bounding box of valid catchments
          (excludes empty/NaN background), improving readability and interaction speed.
        - In interactive mode, prints both gauge IDs (names) and gauge catchment_ids
          of the clicked basin to the console.
        - Supports plotting in Longitude/Latitude if available.
        """
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

        # Check if we can use Lat/Lon
        use_lonlat = hasattr(self, 'longitude') and hasattr(self, 'latitude')
        
        if use_lonlat:
            # Estimate grid parameters: lon = slope * x + intercept
            if len(self.catchment_x) > 1:
                slope_x, intercept_x = np.polyfit(self.catchment_x, self.longitude, 1)
                slope_y, intercept_y = np.polyfit(self.catchment_y, self.latitude, 1)
            else:
                use_lonlat = False
        
        if use_lonlat:
            # Calculate extent for imshow [left, right, bottom, top]
            # Pixel centers are at integer indices. Edges are +/- 0.5.
            left = intercept_x + slope_x * (-0.5)
            right = intercept_x + slope_x * (self.nx - 0.5)
            # For Y, index 0 is usually top (max lat), so slope_y is negative.
            # bottom (min lat) corresponds to max index (ny-0.5)
            bottom = intercept_y + slope_y * (self.ny - 0.5)
            top = intercept_y + slope_y * (-0.5)
            
            extent = [left, right, bottom, top]
            xlabel = "Longitude"
            ylabel = "Latitude"
            
            def idx_to_lon(x): return intercept_x + slope_x * x
            def idx_to_lat(y): return intercept_y + slope_y * y
        else:
            extent = None
            xlabel = "X Index"
            ylabel = "Y Index"
            def idx_to_lon(x): return x
            def idx_to_lat(y): return y

        def generate_random_colors(N, avoid_rgb_colors):
            colors = []
            avoid_rgb_colors = np.array(avoid_rgb_colors)

            while len(colors) < N:
                color = np.random.rand(3)
                if np.all(np.linalg.norm(avoid_rgb_colors - color, axis=1) > 0.7):
                    colors.append(color)

            return np.array(colors)

        special_colors = []
        if visualize_gauges:
            special_colors.append((0, 1, 0))      # gauges pure green
        if visualize_bifurcations:
            special_colors.append((0, 0, 1))      # bifurcations pure blue
        if visualize_removed_bifurcations:
            special_colors.append((1, 0, 0))      # removed paths red
        if visualize_levees:
            special_colors.append((0.5, 0, 0.5))  # levees purple

        # Build a float basin map; use NaN as background so it can be rendered transparent
        basin_map = np.full(self.map_shape, fill_value=np.nan, dtype=float)
        unique_roots = np.unique(self.root_mouth)
        root_to_basin = {root: i for i, root in enumerate(unique_roots)}
        basin_ids = np.array([root_to_basin[r] for r in self.root_mouth])
        basin_map[self.catchment_x, self.catchment_y] = basin_ids

        num_basins = len(unique_roots)
        basin_colors = generate_random_colors(num_basins, avoid_rgb_colors=special_colors)
        cmap = ListedColormap(basin_colors)
        cmap.set_bad(alpha=0.0)

        plt.figure(figsize=(12, 10))
        # Use vmin/vmax for discrete mapping of integer basin ids
        plt.imshow(np.ma.masked_invalid(basin_map).T, origin='upper', cmap=cmap, interpolation='nearest',
                   vmin=-0.5, vmax=num_basins - 0.5, extent=extent)
        plt.title(f"MERIT Global Basins with Bifurcation Paths")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(False)

        # --- Auto-crop to non-empty region ---
        margin = 1
        x0_idx = max(0, int(self.catchment_x.min()) - margin)
        x1_idx = min(self.nx - 1, int(self.catchment_x.max()) + margin)
        y0_idx = max(0, int(self.catchment_y.min()) - margin)
        y1_idx = min(self.ny - 1, int(self.catchment_y.max()) + margin)

        if use_lonlat:
            # Convert indices to coordinates for limits
            # Note: y0_idx is min index (North), y1_idx is max index (South)
            xlim_min = idx_to_lon(x0_idx - 0.5)
            xlim_max = idx_to_lon(x1_idx + 0.5)
            ylim_min = idx_to_lat(y1_idx + 0.5)
            ylim_max = idx_to_lat(y0_idx - 0.5)
            
            plt.xlim(xlim_min, xlim_max)
            plt.ylim(ylim_min, ylim_max)
            
            def within_extent(xv, yv):
                return (xv >= min(xlim_min, xlim_max)) & (xv <= max(xlim_min, xlim_max)) & \
                       (yv >= min(ylim_min, ylim_max)) & (yv <= max(ylim_min, ylim_max))
        else:
            plt.xlim(x0_idx - 0.5, x1_idx + 0.5)
            plt.ylim(y1_idx + 0.5, y0_idx - 0.5)
            def within_extent(xv, yv):
                return (xv >= x0_idx) & (xv <= x1_idx) & (yv >= y0_idx) & (yv <= y1_idx)

        # Plot gauges if available
        if visualize_gauges and self.num_gauges > 0:
            gauge_catchment_ids = []
            for info in self.gauge_info.values():
                gauge_catchment_ids.extend(info["upstream_id"])
            gauge_catchment_ids = np.unique(gauge_catchment_ids)

            catchment_id_to_idx = {cid: idx for idx, cid in enumerate(self.catchment_id)}
            gauge_indices = [catchment_id_to_idx[cid] for cid in gauge_catchment_ids if cid in catchment_id_to_idx]

            if gauge_indices:
                if use_lonlat:
                    gauge_x = self.longitude[gauge_indices]
                    gauge_y = self.latitude[gauge_indices]
                else:
                    gauge_x = self.catchment_x[gauge_indices]
                    gauge_y = self.catchment_y[gauge_indices]
                
                m = within_extent(gauge_x, gauge_y)
                if np.any(m):
                    plt.scatter(gauge_x[m], gauge_y[m], c='#00FF00', s=0.5, label='Gauges', zorder=5)

        # Plot levees if available
        if visualize_levees and hasattr(self, 'levee_catchment_x') and self.num_levees > 0:
            if use_lonlat:
                levee_x = idx_to_lon(self.levee_catchment_x)
                levee_y = idx_to_lat(self.levee_catchment_y)
            else:
                levee_x = self.levee_catchment_x
                levee_y = self.levee_catchment_y
            
            m = within_extent(levee_x, levee_y)
            if np.any(m):
                plt.scatter(levee_x[m], levee_y[m], c='#800080', s=0.2, label='Levees', zorder=4)

        # Plot user points of interest (POIs)
        if hasattr(self, "target_cids") and isinstance(getattr(self, "target_cids"), np.ndarray) and self.target_cids.size > 0:
            poi_idx = find_indices_in(self.target_cids, self.catchment_id)
            poi_idx = poi_idx[poi_idx >= 0]
            if poi_idx.size > 0:
                if use_lonlat:
                    poi_x = self.longitude[poi_idx]
                    poi_y = self.latitude[poi_idx]
                else:
                    poi_x = self.catchment_x[poi_idx]
                    poi_y = self.catchment_y[poi_idx]
                
                m = within_extent(poi_x, poi_y)
                if np.any(m):
                    plt.scatter(poi_x[m], poi_y[m], c="#C10000", s=0.3, label='Points of Interest', zorder=6)

        if not interactive_basin_picker:
            # Plot kept bifurcation paths
            if visualize_bifurcations and self.num_bifurcation_paths > 0 and self.num_bifurcation_paths < 3e6:
                if use_lonlat:
                    x1a = idx_to_lon(self.bifurcation_catchment_x)
                    y1a = idx_to_lat(self.bifurcation_catchment_y)
                    x2a = idx_to_lon(self.bifurcation_downstream_x)
                    y2a = idx_to_lat(self.bifurcation_downstream_y)
                    mask_keep = (np.abs(x1a - x2a) <= 180.0)
                else:
                    x1a = self.bifurcation_catchment_x
                    y1a = self.bifurcation_catchment_y
                    x2a = self.bifurcation_downstream_x
                    y2a = self.bifurcation_downstream_y
                    mask_keep = (np.abs(x1a - x2a) <= self.nx / 2)
                
                mask_keep &= (within_extent(x1a, y1a) | within_extent(x2a, y2a))
                if np.any(mask_keep):
                    x1k = x1a[mask_keep]; y1k = y1a[mask_keep]; x2k = x2a[mask_keep]; y2k = y2a[mask_keep]
                    line_segments_keep = np.array([[[x1k[i], y1k[i]], [x2k[i], y2k[i]]] for i in range(len(x1k))])
                    kept_lines = LineCollection(line_segments_keep, colors='#0000FF', linestyles='--', linewidths=0.5, alpha=0.6, zorder=3)
                    plt.gca().add_collection(kept_lines)
                    plt.plot([], [], color='#0000FF', linestyle='--', linewidth=0.5, alpha=0.6, label='Bifurcation Paths')

            # Plot removed bifurcation paths
            if visualize_removed_bifurcations and hasattr(self, "removed_bifurcation_catchment_x") and self.removed_bifurcation_catchment_x.size > 0:
                if use_lonlat:
                    rx1a = idx_to_lon(self.removed_bifurcation_catchment_x)
                    ry1a = idx_to_lat(self.removed_bifurcation_catchment_y)
                    rx2a = idx_to_lon(self.removed_bifurcation_downstream_x)
                    ry2a = idx_to_lat(self.removed_bifurcation_downstream_y)
                    mask_cut = (np.abs(rx1a - rx2a) <= 180.0)
                else:
                    rx1a = self.removed_bifurcation_catchment_x
                    ry1a = self.removed_bifurcation_catchment_y
                    rx2a = self.removed_bifurcation_downstream_x
                    ry2a = self.removed_bifurcation_downstream_y
                    mask_cut = (np.abs(rx1a - rx2a) <= self.nx / 2)
                
                mask_cut &= (within_extent(rx1a, ry1a) | within_extent(rx2a, ry2a))
                if np.any(mask_cut):
                    rx1 = rx1a[mask_cut]; ry1 = ry1a[mask_cut]; rx2 = rx2a[mask_cut]; ry2 = ry2a[mask_cut]
                    line_segments_removed = np.array([[[rx1[i], ry1[i]], [rx2[i], ry2[i]]] for i in range(len(rx1))])
                    removed_lines = LineCollection(line_segments_removed, colors='#FF0000', linestyles=':', linewidths=1, alpha=0.5, zorder=3)
                    plt.gca().add_collection(removed_lines)
                    plt.plot([], [], color='#FF0000', linestyle=':', linewidth=1, alpha=0.7, label='Bifurcation Paths (removed)')

        # --- Interactive basin -> gauge ids picker ---
        if interactive_basin_picker:
            ax = plt.gca()

            catchment_id_to_idx = {cid: idx for idx, cid in enumerate(self.catchment_id)}
            basin_to_gauges: Dict[int, set] = {}
            basin_to_gauge_cids: Dict[int, set] = {}
            if getattr(self, "num_gauges", 0) > 0 and hasattr(self, "gauge_info"):
                for gname, info in self.gauge_info.items():
                    for cid in info.get("upstream_id", []):
                        idx = catchment_id_to_idx.get(cid, -1)
                        if idx >= 0:
                            b = int(self.catchment_basin_id[idx])
                            basin_to_gauges.setdefault(b, set()).add(int(gname))
                            basin_to_gauge_cids.setdefault(b, set()).add(int(cid))

            ann = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(5, 5),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6),
                fontsize=7,
                visible=False,
            )

            def on_click(event):
                if event.inaxes is not ax or event.xdata is None or event.ydata is None:
                    return

                if use_lonlat:
                    # Convert back to indices
                    xi_f = (event.xdata - intercept_x) / slope_x
                    yi_f = (event.ydata - intercept_y) / slope_y
                    xi = int(np.clip(round(xi_f), 0, self.nx - 1))
                    yi = int(np.clip(round(yi_f), 0, self.ny - 1))
                    
                    if not within_extent(event.xdata, event.ydata):
                        ann.set_visible(False)
                        plt.draw()
                        return
                else:
                    xi = int(np.clip(round(event.xdata), 0, self.nx - 1))
                    yi = int(np.clip(round(event.ydata), 0, self.ny - 1))
                    if not within_extent(event.xdata, event.ydata):
                        ann.set_visible(False)
                        plt.draw()
                        return

                bval = basin_map[xi, yi]
                if np.isnan(bval):
                    ann.set_visible(False)
                    plt.draw()
                    return
                b = int(bval)
                gids = sorted(basin_to_gauges.get(b, []))
                gcids = sorted(basin_to_gauge_cids.get(b, []))
                msg = f"basin={b}, gauges={gids if len(gids) > 0 else '[]'}, gauge_catchment_ids={gcids if len(gcids) > 0 else '[]'}, click=(x={xi}, y={yi})"
                print(msg)
                log_path = self.out_dir / f"basin_{b}.txt"
                if len(gids) > 0:
                    with open(log_path, "a", encoding="utf-8") as fp:
                        for gid in gids:
                            fp.write(f"{gid}\n")
                ann.set_text(f"basin={b}\ngauges={gids if len(gids) > 0 else '[]'}")
                ann.xy = (event.xdata, event.ydata)
                ann.set_visible(True)
                plt.draw()

            plt.gcf().canvas.mpl_connect("button_press_event", on_click)
            plt.title(plt.gca().get_title() + "(click to identify basin/gauges)")
            handles, labels = plt.gca().get_legend_handles_labels()
            if labels:
                plt.legend(loc='lower right')
            plt.tight_layout()
            plt.show()
            return
        # --- end interactive ---

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
        print(f"Number of levees         : {getattr(self, 'num_levees', 0)}")
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
        self.filter_to_poi_basins()
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
        levee_flag=True,
        visualized=True,
        bif_levels_to_keep=5,
        basin_use_file=False,
        target_gpus=1,
        out_file="parameters.nc",
    )
    merit_map.build_input()

    
