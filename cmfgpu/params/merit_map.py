"""
MERIT-based map parameter generation using Pydantic v2.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import ClassVar, List, Optional

import h5py
import numpy as np
from numba import njit
from pydantic import BaseModel, ConfigDict, DirectoryPath, Field, FilePath, model_validator

from cmfgpu.utils import binread, find_indices_in, read_map


@njit
def trace_outlets_dict(catchment_id, downstream_id):
    """Numba-optimized function to trace outlets."""
    id_to_down = dict()
    for i in range(len(catchment_id)):
        id_to_down[catchment_id[i]] = downstream_id[i]

    result = np.zeros(len(catchment_id), dtype=np.int32)

    for i in range(len(catchment_id)):
        current = catchment_id[i]
        while id_to_down[current] >= 0:
            current = id_to_down[current]
        result[i] = current
    return result


@njit
def topological_sort(catchment_id, downstream_id):
    """Numba-optimized topological sorting."""
    id_to_index = dict()
    for i in range(len(catchment_id)):
        id_to_index[catchment_id[i]] = i

    n = len(catchment_id)
    indegree = np.zeros(n, dtype=np.int32)

    for i in range(n):
        d_id = downstream_id[i]
        if d_id >= 0 and d_id in id_to_index:
            d_idx = id_to_index[d_id]
            indegree[d_idx] += 1

    result_idx = np.empty(n, dtype=np.int32)
    q = np.empty(n, dtype=np.int32)
    front = 0
    back = 0

    for i in range(n):
        if indegree[i] == 0:
            q[back] = i
            back += 1

    count = 0
    while front < back:
        u = q[front]
        front += 1
        result_idx[count] = u
        count += 1

        d_id = downstream_id[u]
        if d_id >= 0 and d_id in id_to_index:
            d_idx = id_to_index[d_id]
            indegree[d_idx] -= 1
            if indegree[d_idx] == 0:
                q[back] = d_idx
                back += 1

    if count != n:
        raise ValueError("Rings exist and topological sorting is not possible")

    return result_idx

@njit
def compute_init_river_depth(catchment_elevation, river_height, downstream_idx):
    num_catchments = len(catchment_elevation)
    river_depth = np.zeros(num_catchments, dtype=np.float32)
    river_elevation = catchment_elevation - river_height

    for i in range(num_catchments - 1, -1, -1):
        j = downstream_idx[i]
        if i == j or j < 0:
            river_depth[i] = river_height[i]
        else:
            river_depth[i] = max(
                river_depth[j] + river_elevation[j] - river_elevation[i],
                0.0
            )
        river_depth[i] = min(river_depth[i], river_height[i])

    return river_depth

def reorder_by_basin_size(topo_idx: np.ndarray, basin_id: np.ndarray):
    """Reorder by basin size."""
    groups = defaultdict(list)
    for idx in topo_idx:                      
        groups[basin_id[idx]].append(idx)

    ordered_basins = sorted(groups.keys(),
                            key=lambda b: len(groups[b]),
                            reverse=True)

    new_order = []
    basin_sizes = np.empty(len(ordered_basins), dtype=np.int64)
    for k, b in enumerate(ordered_basins):
        new_order.extend(groups[b])
        basin_sizes[k] = len(groups[b])

    return (np.asarray(new_order, dtype=np.int64), basin_sizes)


def read_bifparam(filename):
    """Read bifurcation parameters from file."""
    with open(filename, 'r') as f:
        first_line = f.readline().strip().split()
        num_paths = int(first_line[0])
        num_levels = int(first_line[1])

        pth_upst = []
        pth_down = []
        pth_dst = []
        pth_wth = []
        pth_elv = []

        for ipth in range(num_paths):
            line = f.readline().strip().split()
            ix = int(line[0]) - 1
            iy = int(line[1]) - 1
            jx = int(line[2]) - 1
            jy = int(line[3]) - 1
            dst = float(line[4])
            pelv = float(line[5])
            pdph = float(line[6])
            
            pth_upst.append([ix, iy])
            pth_down.append([jx, jy])
            pth_dst.append(dst)

            pth_elv_row = []
            pth_wth_row = []
            for ilev in range(num_levels):
                pwth = float(line[7 + ilev])
                if ilev == 0:
                    if pwth > 0:
                        pth_elv_tmp = pelv - pdph
                    else:
                        pth_elv_tmp = 1.0E20
                else:
                    if pwth > 0:
                        pth_elv_tmp = pelv + ilev - 1.0
                    else:
                        pth_elv_tmp = 1.0E20
                pth_elv_row.append(pth_elv_tmp)
                pth_wth_row.append(pwth)
            pth_wth.append(pth_wth_row)
            pth_elv.append(pth_elv_row)

    return num_levels, pth_upst, pth_down, pth_dst, pth_wth, pth_elv


class MERITMap(BaseModel):
    """
    MERIT-based map parameter generation class using Pydantic v2.
    
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

    h5_file: str = Field(
        description="Name of the output HDF5 file for storing map parameters",
        default="parameters.h5"
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

    visualize_basins: bool = Field(
        default=True,
        description="Generate basin visualization"
    )
    
    # === File Mapping ===
    missing_value: ClassVar[float] = -9999.0
    h5_save_keys: ClassVar[List[str]] = [
        "catchment_id",
        "downstream_id",
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
    
    
    def _load_map_info(self) -> None:
        """Load map dimensions from mapdim.txt file."""
        mapdim_path = self.map_dir / "mapdim.txt"
        
        with open(mapdim_path, "r") as f:
            lines = f.readlines()
            self.nx = int(lines[0].split('!!')[0].strip())
            self.ny = int(lines[1].split('!!')[0].strip())
            self.num_flood_levels  = int(lines[2].split('!!')[0].strip())

        print(f"Loaded map dimensions: nx={self.nx}, ny={self.ny}, num_flood_levels={self.num_flood_levels}")

    def _load_catchment_id(self) -> None:
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
        self.num_catchments = len(catchment_id)
        
        print(f"Loaded {len(catchment_id)} catchments")

    def _load_gauge_id(self) -> None:
        """Load gauge information from gauge file."""
        gauge_id_set = set()
        self.gauge_info = {}
        
        if self.gauge_file is None:
            print("No gauge file provided, skipping gauge loading")
            self.gauge_id = np.array([], dtype=np.int64)
            return
            
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
        self.gauge_mask = np.zeros(self.num_catchments, dtype=bool)
        temp_idx = find_indices_in(self.gauge_id, self.catchment_id)
        self.gauge_mask[temp_idx] = True
        
        print(f"Loaded {len(self.gauge_info)} gauges covering {len(self.gauge_id)} catchments")

    def _load_bifurcation_parameters(self) -> None:
        """Load bifurcation channel parameters."""
        bif_file = self.map_dir / "bifprm.txt"

        num_bifurcation_levels, pth_upst, pth_down, pth_dst, pth_wth, pth_elv = read_bifparam(bif_file)
        self.num_bifurcation_paths = len(pth_upst)
        self.bifurcation_manning = np.tile(
            [0.03] + [0.1] * (num_bifurcation_levels - 1),
            (self.num_bifurcation_paths, 1)
        ).astype(self.numpy_precision)
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

        # Handle basin integration for bifurcations
        unique_river_mouth_id = np.unique(self.river_mouth_id)
        temp_idx = find_indices_in(self.bifurcation_catchment_id, self.catchment_id)
        ori_river_mouth_id = self.river_mouth_id[temp_idx]
        temp_idx = find_indices_in(self.bifurcation_downstream_id, self.catchment_id)
        bif_river_mouth_id = self.river_mouth_id[temp_idx]
        id2idx = {id_: idx for idx, id_ in enumerate(unique_river_mouth_id)}

        # Union-find for basin integration
        parent = np.arange(len(unique_river_mouth_id))
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        
        for a, b in zip(ori_river_mouth_id, bif_river_mouth_id):
            union(id2idx[a], id2idx[b])

        # Rebuild the root river mouth IDs
        root_mouth = np.array([unique_river_mouth_id[find(id2idx[m])] for m in self.river_mouth_id])

        # Topological sorting and basin reordering
        topo_idx = topological_sort(self.catchment_id, self.downstream_id)
        sorted_idx, basin_sizes = reorder_by_basin_size(topo_idx, root_mouth)
        
        # Apply sorting
        self.catchment_id = self.catchment_id[sorted_idx]
        self.downstream_id = self.downstream_id[sorted_idx]
        self.catchment_x = self.catchment_x[sorted_idx]
        self.catchment_y = self.catchment_y[sorted_idx]
        self.root_mouth = root_mouth[sorted_idx]
        self.basin_sizes = basin_sizes
        self.num_basins = len(basin_sizes)
        _, inverse_indices = np.unique(self.root_mouth, return_inverse=True)
        self.catchment_basin_id = inverse_indices.astype(np.int64)
        self.bifurcation_basin_id = np.zeros_like(self.bifurcation_catchment_id, dtype=np.int64)
        temp_idx = find_indices_in(self.bifurcation_catchment_id, self.catchment_id)
        if np.any(temp_idx == -1):
            raise ValueError("Bifurcation catchment IDs do not match catchment IDs")
        self.bifurcation_basin_id = self.catchment_basin_id[temp_idx]
        
        # Create downstream indices
        self.downstream_idx = find_indices_in(self.downstream_id, self.catchment_id)
        self.is_river_mouth = (self.downstream_idx < 0)
        self.downstream_id[self.is_river_mouth] = self.catchment_id[self.is_river_mouth]
        # Initialize reservoir mask TODO: this should be set based on actual reservoir data
        self.is_reservoir = np.zeros_like(self.catchment_id, dtype=bool)
        
            

    def _load_parameters(self) -> None:

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

    def _check_flow_direction(self) -> None:
        """Validate flow direction consistency."""
        non_mouth = ~self.is_river_mouth
        if not (self.downstream_idx[non_mouth] > np.flatnonzero(non_mouth)).all():
            raise ValueError("Flow direction error: downstream catchment should have higher index")
            
        if not (self.downstream_idx[self.is_river_mouth] == -1).all():
            raise ValueError("Flow direction error: river mouths should point to themselves")

    def _init_river_depth(self) -> None:
        """Initialize river depth based on elevation gradients."""
        self.river_depth = compute_init_river_depth(
            self.catchment_elevation,
            self.river_height,
            self.downstream_idx
        ).astype(self.numpy_precision)

        self.river_storage = self.river_length * self.river_width * self.river_depth

    def _create_h5_file(self) -> None:
        with h5py.File(self.out_dir / self.h5_file, 'w') as f:
            for key in self.h5_save_keys:
                if not hasattr(self, key):
                    raise ValueError(f"Missing required field: {key}")
                f.create_dataset(key, data=getattr(self, key))

    def _visualize_basins(self) -> None:
        """Generate basin visualization if requested."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.collections import LineCollection
            from matplotlib.colors import ListedColormap
        except ImportError:
            print("matplotlib not available, skipping basin visualization")
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
            (1, 0, 0),      # gauges pure red
            (0, 0, 1),      # bifurcations pure blue
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
        if len(self.gauge_info) > 0:
            gauge_catchment_ids = []
            for info in self.gauge_info.values():
                gauge_catchment_ids.extend(info["upstream_id"])
            gauge_catchment_ids = np.unique(gauge_catchment_ids)

            catchment_id_to_idx = {cid: idx for idx, cid in enumerate(self.catchment_id)}
            gauge_indices = [catchment_id_to_idx[cid] for cid in gauge_catchment_ids if cid in catchment_id_to_idx]
            
            if gauge_indices:
                gauge_x = self.catchment_x[gauge_indices]
                gauge_y = self.catchment_y[gauge_indices]
                plt.scatter(gauge_x, gauge_y, c='#FF0000', s=0.5, label='Gauges')

        # Plot bifurcation paths if available
        if len(self.bifurcation_catchment_id) > 0:
            x1 = self.bifurcation_catchment_x
            y1 = self.bifurcation_catchment_y
            x2 = self.bifurcation_downstream_x
            y2 = self.bifurcation_downstream_y
            mask = np.abs(x1 - x2) <= self.nx / 2
            x1 = x1[mask]
            y1 = y1[mask]
            x2 = x2[mask]
            y2 = y2[mask]
            line_segments = np.array([[[x1[i], y1[i]], [x2[i], y2[i]]] for i in range(len(x1))])

            saved_lines = LineCollection(line_segments, colors='#0000FF', linestyles='dashed', linewidths=0.5, alpha=0.5)
            plt.gca().add_collection(saved_lines)
            plt.plot([], [], color='#0000FF', linestyle='--', linewidth=0.5, alpha=0.5, label=f'Bifurcation Paths')

        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(self.out_dir / f"basin_map.png", dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"Saved basin visualization to {self.out_dir / f'basin_map.png'}")

    def _print_summary(self) -> None:
        print(f"MERIT Map Processing Summary:")
        print(f"Grid dimensions          : {self.nx} x {self.ny}")
        print(f"Number of basins         : {self.num_basins}")
        print(f"Number of catchments     : {self.num_catchments}")
        print(f"Number of reservoirs     : {self.is_reservoir.sum()}")
        print(f"Number of bifurcation paths : {self.num_bifurcation_paths}")
        print(f"Number of gauges         : {len(self.gauge_info)}")
        print(f"Gauge catchments         : {len(self.gauge_id)}")
        print(f"Output directory         : {self.out_dir}")


    def build_model_input_pipeline(self) -> None:
        print(f"Starting MERIT Map processing pipeline")
        print(f"Map directory: {self.map_dir}")
        print(f"Output file: {self.out_dir / self.h5_file}")
        
        # Core processing steps
        self._load_map_info()
        self._load_catchment_id()
        self._load_gauge_id()
        self._load_bifurcation_parameters()
        self._load_parameters()
        self._check_flow_direction()
        self._init_river_depth()
        self._create_h5_file()
        if self.visualize_basins:
            self._visualize_basins()
        self._print_summary()
        
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
        out_dir=f"/home/eat/CaMa-Flood-GPU/inp/{map_resolution}",
        gauge_file=f"/home/eat/cmf_v420_pkg/map/{map_resolution}/GRDC_alloc.txt",
        visualize_basins=True
    )
    merit_map.build_model_input_pipeline()
