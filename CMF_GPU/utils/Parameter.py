import numpy as np
import importlib
import os
import sys
import pickle
from collections import defaultdict
from pathlib import Path
from omegaconf import OmegaConf
from numba import njit
from scipy.sparse import csr_matrix, save_npz
from CMF_GPU.utils.Variables import MODULES_INFO
from CMF_GPU.utils.Checker import CONFIG_REQUIRED_KEYS
from CMF_GPU.utils.utils import snapshot_to_h5, gather_all_keys

def binread(filename, shape, dtype_str):
    """
    Reads a binary file and reshapes it to a specified shape.
    shape is a tuple, like (nx*ny, 2), etc.
    dtype_str can be 'int32', 'float32', 'float64', etc.
    """
    count = 1
    for s in shape:
        count *= s
    arr = np.fromfile(filename, dtype=dtype_str, count=count)
    return arr.reshape(shape, order='F')


def read_map(filename, map_shape, precision):
    """
    Used to read spatial mapping files (such as rivlen.bin / rivhgt.bin etc.) and filter based on map_idx.
    map_shape: (nx, ny) or (nx, ny, NLFP), etc.
    When map_shape is 2D, returns data[map_idx]
    When map_shape is 3D, returns data in the form of [(nx*ny), NLFP] and then indexes it.
    """
    if len(map_shape) == 2:
        nx, ny = map_shape
        data = binread(filename, (nx, ny), dtype_str=precision)
    elif len(map_shape) == 3:
        nx, ny, nlfp = map_shape
        # First read into shape [nx*ny, nlfp]
        data = binread(filename, (nx, ny, nlfp), dtype_str=precision)
    else:
        raise ValueError("Unsupported map_shape dimension.")

    return data

@njit
def trace_outlets_dict(catchment_id, downstream_id):
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

def reorder_by_basin_size(topo_idx: np.ndarray,
                          basin_id: np.ndarray):

    groups = defaultdict(list)
    for idx in topo_idx:                      
        groups[basin_id[idx]].append(idx)

    ordered_basins = sorted(groups.keys(),
                            key=lambda b: len(groups[b]),
                            reverse=True)

    new_order   = []
    basin_sizes = np.empty(len(ordered_basins), dtype=np.int64)
    for k, b in enumerate(ordered_basins):
        new_order.extend(groups[b])
        basin_sizes[k] = len(groups[b])

    return (np.asarray(new_order, dtype=np.int64), basin_sizes)            

def read_bifparam(filename):
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

def find_indices_in(a, b):
    """
    Returns the index position of each element of a in b. The elements of b must be unique.
    """

    order = np.argsort(b)
    sorted_b = b[order]
    pos_in_sorted = np.searchsorted(sorted_b, a)
    valid_mask = pos_in_sorted < len(sorted_b)
    hit_mask = np.zeros_like(a, dtype=bool)
    hit_mask[valid_mask] = (sorted_b[pos_in_sorted[valid_mask]] == a[valid_mask])
    index = np.full_like(pos_in_sorted, -1, dtype=int)
    index[hit_mask] = order[pos_in_sorted[hit_mask]]

    # assert (index != -1).all(), "Error: Some elements in 'a' are not found in 'b'."

    return index


def compute_runoff_id(runoff_lon, runoff_lat, hires_lon, hires_lat):
    """
    Calculates runoff grid IDs considering ascending or descending order of coordinates.
    """
    lon_ascending = runoff_lon[1] > runoff_lon[0]
    lat_ascending = runoff_lat[1] > runoff_lat[0]

    gsize_lon = abs(runoff_lon[1] - runoff_lon[0])
    gsize_lat = abs(runoff_lat[1] - runoff_lat[0])

    if lon_ascending:
        westin = runoff_lon[0] - 0.5 * gsize_lon
        ixin = np.floor((hires_lon - westin) / gsize_lon).astype(int)
    else:
        westin = runoff_lon[0] + 0.5 * gsize_lon
        ixin = np.floor((westin - hires_lon) / gsize_lon).astype(int)

    if lat_ascending:
        southin = runoff_lat[0] - 0.5 * gsize_lat
        iyin = np.floor((hires_lat - southin) / gsize_lat).astype(int)
    else:
        northin = runoff_lat[0] + 0.5 * gsize_lat
        iyin = np.floor((northin - hires_lat) / gsize_lat).astype(int)

    nxin = len(runoff_lon)
    nyin = len(runoff_lat)

    assert np.all((ixin >= 0) & (ixin < nxin)), "Some hires_lon points fall outside the runoff grid (longitude)"
    assert np.all((iyin >= 0) & (iyin < nyin)), "Some hires_lat points fall outside the runoff grid (latitude)"

    runoff_id = iyin * nxin + ixin

    return runoff_id

class DefaultGlobalCatchment:
    num_flood_levels = 10
    
    gravity = 9.8
    river_manning = 0.03 # not used, load from rivman.bin (may be same?)
    flood_manning = 0.1
    river_mouth_distance = 10000.0
    log_buffer_size = 800
    adaptation_factor = 0.7
    missing_value = -9999
    possible_modules = ["base", "adaptive_time_step", "log", "bifurcation"]
    
    var_maps = {
    "river_length": "rivlen.bin",
    "river_width": "rivwth_gwdlr.bin",
    "river_height": "rivhgt.bin",
    "river_manning": "rivman.bin",
    "catchment_elevation": "elevtn.bin",
    "catchment_area": "ctmare.bin",
    "downstream_distance": "nxtdst.bin",
    "flood_depth_table": "fldhgt.bin",
    }
    zero_init_states = [
    "flood_storage",
    "river_outflow",
    "flood_depth",
    "flood_outflow",
    "river_cross_section_depth",
    "flood_cross_section_depth",
    "flood_cross_section_area",
    ]
    map_info = "mapinfo.txt"
    bif_info = "bifprm.txt"

    idx_precision = "<i4"
    map_precision = "<f4"
    hires_map_tag = "1min"
    hires_idx_precision = "<i2"
    hires_map_precision = "<f4"
    numpy_precision = np.float32

    def __init__(self, config):
        self._check_config(config)
        parameter_config = config["parameter_config"]
        runoff_config = config["runoff_config"]
        self.map_dir = Path(parameter_config["map_dir"])
        self.hires_map_dir = Path(parameter_config["hires_map_dir"])
        self.inp_dir = Path(parameter_config["inp_dir"])
        self.gauge_file = Path(parameter_config["gauge_file"]) if parameter_config["gauge_file"] is not None else None
        self.save_gauge_only = parameter_config["save_gauge_only"]
        self.simulate_gauge_only = parameter_config["simulate_gauge_only"]
        self.runoff_config = runoff_config
        self.modules = self.possible_modules
        self.params = {}
        self.states = {}
        self.gauge_info = {}
        self.parameters_loaded = False

    def _check_config(self, config):
        for key in config:
            if key not in CONFIG_REQUIRED_KEYS:
                raise ValueError(f"Unrecognized config key: {key}")           
        for key in CONFIG_REQUIRED_KEYS:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
            CONFIG_REQUIRED_KEYS[key](config[key])

    def _load_map_info(self):
        """
        Loads map dimensions (nx, ny) from the basin.ctl file.
        """
        mapinfo_path = self.map_dir / "basin.ctl"
        with open(mapinfo_path, "r") as f:
            lines = f.readlines()

        # Example assumes nx, ny are on line 3, like in many .ctl files
        for line in lines:
            if line.strip().startswith("xdef"):
                parts = line.strip().split()
                self.nx = int(parts[1])
            elif line.strip().startswith("ydef"):
                parts = line.strip().split()
                self.ny = int(parts[1])
        
        print(f"Loaded map dimensions: nx={self.nx}, ny={self.ny}")


    def _load_catchment_id(self):
        """
        Load catchment IDs and next catchment IDs from binary files.
        """
        nextxy_data = binread(self.map_dir / "nextxy.bin", (self.nx, self.ny, 2), dtype_str=self.idx_precision)
        self.map_shape = nextxy_data.shape[:2]
        catchment_x, catchment_y = np.where(nextxy_data[:, :, 0] != self.missing_value)
        next_catchment_x, next_catchment_y = nextxy_data[catchment_x, catchment_y, 0] - 1, nextxy_data[catchment_x, catchment_y, 1] - 1
        catchment_id = np.ravel_multi_index((catchment_x, catchment_y), self.map_shape)
        downstream_id = np.full_like(next_catchment_x, -1, dtype=np.int64)
        valid_next = (next_catchment_x >= 0) & (next_catchment_y >= 0)
        downstream_id[valid_next] = np.ravel_multi_index(
            (next_catchment_x[valid_next], next_catchment_y[valid_next]),
            self.map_shape
        )
        river_mouth_id = trace_outlets_dict(catchment_id, downstream_id)
        self.catchment_x = catchment_x
        self.catchment_y = catchment_y
        self.catchment_id = catchment_id
        self.downstream_id = downstream_id
        self.river_mouth_id = river_mouth_id

    def _load_gauge_id(self):
        gauge_id_set = set()
        if self.gauge_file is not None:
            with open(self.gauge_file, "r") as f:
                lines = f.readlines()

            # skip header
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

                if ix1 >= 0 and iy1 >= 0:
                    catchment1 = np.ravel_multi_index((ix1, iy1), self.map_shape)
                    catchment_ids.append(catchment1)
                    gauge_id_set.add(catchment1)

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
                raise ValueError("No valid gauges found in the gauge file.")
        
        self.gauge_id = np.array(sorted(gauge_id_set), dtype=np.int64)

    def _load_bifurcation_parameters(self):
        num_bifurcation_levels, pth_upst, pth_down, pth_dst, pth_wth, pth_elv = read_bifparam(self.map_dir / self.bif_info)
        bifurcation_manning = [self.river_manning] + [self.flood_manning] * (num_bifurcation_levels - 1)
        pth_upst = np.array(pth_upst, dtype=np.int64)
        pth_down = np.array(pth_down, dtype=np.int64)
        bifurcation_catchment_id = np.ravel_multi_index((pth_upst[:, 0], pth_upst[:, 1]), self.map_shape)
        bifurcation_downstream_id = np.ravel_multi_index((pth_down[:, 0], pth_down[:, 1]), self.map_shape)
        num_bifurcation_paths = len(bifurcation_catchment_id)
        unique_river_mouth_id = np.unique(self.river_mouth_id)
        temp_idx = find_indices_in(bifurcation_catchment_id, self.catchment_id)
        ori_river_mouth_id = self.river_mouth_id[temp_idx]
        temp_idx = find_indices_in(bifurcation_downstream_id, self.catchment_id)
        bif_river_mouth_id = self.river_mouth_id[temp_idx]
        id2idx = {id_: idx for idx, id_ in enumerate(unique_river_mouth_id)}

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

        # Rebuild the root river mouth IDs based on union-find
        root_mouth = np.array([unique_river_mouth_id[find(id2idx[m])] for m in self.river_mouth_id])
        if self.simulate_gauge_only:
            gauge_idx = find_indices_in(self.gauge_id, self.catchment_id)
            gauge_basin_roots = np.unique(root_mouth[gauge_idx])
            basin_keep_mask = np.isin(root_mouth, gauge_basin_roots)
            self.catchment_id = self.catchment_id[basin_keep_mask]
            self.downstream_id = self.downstream_id[basin_keep_mask]
            self.catchment_x = self.catchment_x[basin_keep_mask]
            self.catchment_y = self.catchment_y[basin_keep_mask]
            self.river_mouth_id = self.river_mouth_id[basin_keep_mask]
            root_mouth = root_mouth[basin_keep_mask]
            bif_keep_mask = np.isin(bifurcation_catchment_id, self.catchment_id)
            bifurcation_catchment_id = bifurcation_catchment_id[bif_keep_mask]
            bifurcation_downstream_id = bifurcation_downstream_id[bif_keep_mask]
            num_bifurcation_paths = len(bifurcation_catchment_id)
            assert np.isin(bifurcation_catchment_id, self.catchment_id).all(), "Bifurcation catchment ID must be in the catchment ID list."
            assert np.isin(bifurcation_downstream_id, self.catchment_id).all(), "Bifurcation next catchment ID must be in the catchment ID list."

        topo_idx = topological_sort(self.catchment_id, self.downstream_id)
        sorted_idx, basin_sizes = reorder_by_basin_size(topo_idx, root_mouth)
        self.catchment_id      = self.catchment_id[sorted_idx]
        self.downstream_id = self.downstream_id[sorted_idx]
        self.catchment_x       = self.catchment_x[sorted_idx]
        self.catchment_y       = self.catchment_y[sorted_idx]
        self.root_mouth    = root_mouth[sorted_idx]
        self.downstream_idx = find_indices_in(self.downstream_id, self.catchment_id)
        self.is_river_mouth = (self.downstream_idx == -1)
        self.downstream_idx[self.is_river_mouth] = np.flatnonzero(self.is_river_mouth)
        self.num_basins = len(basin_sizes)
        self.num_catchments_per_basin = basin_sizes
        self.is_reservoir = np.zeros_like(self.catchment_id, dtype=bool)
        self.num_catchments = int(basin_sizes.sum())
        self.bifurcation_catchment_idx = find_indices_in(bifurcation_catchment_id, self.catchment_id)
        self.bifurcation_downstream_idx = find_indices_in(bifurcation_downstream_id, self.catchment_id)
        self.params["is_reservoir"] = self.is_reservoir
        self.params["is_river_mouth"] = self.is_river_mouth
        self.params["downstream_idx"] = self.downstream_idx
        self.params["num_catchments"] = self.num_catchments
        self.params["num_basins"] = self.num_basins
        self.params["num_catchments_per_basin"] = self.num_catchments_per_basin
        self.params["num_bifurcation_paths"] = num_bifurcation_paths
        self.params["num_bifurcation_levels"] = num_bifurcation_levels
        self.params["bifurcation_catchment_idx"] = self.bifurcation_catchment_idx
        self.params["bifurcation_downstream_idx"] = self.bifurcation_downstream_idx
        self.params["bifurcation_manning"] = np.tile(np.asarray(bifurcation_manning, dtype=self.numpy_precision), (num_bifurcation_paths, 1))
        self.params["bifurcation_width"] = np.asarray(pth_wth, dtype=self.numpy_precision)
        self.params["bifurcation_length"] = np.asarray(pth_dst, dtype=self.numpy_precision)
        self.params["bifurcation_elevation"] = np.asarray(pth_elv, dtype=self.numpy_precision)
        self.states["bifurcation_outflow"] = np.zeros((num_bifurcation_paths, num_bifurcation_levels), dtype=self.numpy_precision)
        self.states["bifurcation_cross_section_depth"] = np.zeros((num_bifurcation_paths, num_bifurcation_levels), dtype=self.numpy_precision)
        if self.save_gauge_only:
            catchment_save_idx = find_indices_in(self.gauge_id, self.catchment_id)
        else:
            catchment_save_idx = np.arange(self.num_catchments, dtype=np.int64)
        self.params["catchment_save_idx"] = catchment_save_idx
        self.params["num_catchments_to_save"] = len(catchment_save_idx)
        self.params["bifurcation_path_save_idx"] = np.arange(num_bifurcation_paths, dtype=np.int64) # cannot be modified currently, as bifurcation paths are not saved in the same way as catchments


    def _visualize_basin(self):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.collections import LineCollection

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
        plt.title("Global basins with Bifurcation Paths")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(False)
        

        gauge_catchment_ids = []
        for info in self.gauge_info.values():
            gauge_catchment_ids.extend(info["upstream_id"])
        gauge_catchment_ids = np.unique(gauge_catchment_ids)

        catchment_id_to_idx = {cid: idx for idx, cid in enumerate(self.catchment_id)}
        gauge_indices = [catchment_id_to_idx[cid] for cid in gauge_catchment_ids if cid in catchment_id_to_idx]
        gauge_x = self.catchment_x[gauge_indices]
        gauge_y = self.catchment_y[gauge_indices]

        plt.scatter(gauge_x, gauge_y, c='#FF0000', s=0.5, label='Gauges')

        upstream_x = self.catchment_x[self.bifurcation_catchment_idx]
        upstream_y = self.catchment_y[self.bifurcation_catchment_idx]
        downstream_x = self.catchment_x[self.bifurcation_downstream_idx]
        downstream_y = self.catchment_y[self.bifurcation_downstream_idx]

        line_segments = np.array([[[upstream_x[i], upstream_y[i]], 
                                [downstream_x[i], downstream_y[i]]] 
                                for i in range(len(upstream_x))])



        saved_lines = LineCollection(line_segments, colors='#0000FF', linestyles='dashed', linewidths=0.5, alpha=0.5)
        plt.gca().add_collection(saved_lines)
        plt.plot([], [], color='#0000FF', linestyle='--', linewidth=0.5, alpha=0.5, label=f'Bifurcation Paths')
        plt.legend(loc='lower right')
        plt.tight_layout()
        # plt.show()
        plt.savefig(self.inp_dir / "basin_map.png", dpi=600, bbox_inches='tight')

    def _check_flow_direction(self):
        assert (self.downstream_idx[~self.is_river_mouth] > np.flatnonzero(~self.is_river_mouth)).all(), "Flow direction is not correct! Downstream catchment ID should be greater than upstream catchment ID."
        assert (self.downstream_idx[self.is_river_mouth] == np.flatnonzero(self.is_river_mouth)).all(), "Flow direction is not correct! Downstream catchment ID should be equal to the river mouth catchment ID."
        
    def _load_parameters(self):
        for param_name, filename in self.var_maps.items():
            file_path = self.map_dir / filename
            if param_name == "flood_depth_table":
                data = read_map(file_path, (self.nx, self.ny, self.num_flood_levels), precision=self.map_precision)[self.catchment_x, self.catchment_y, :]
                data = np.hstack([
                    np.zeros((self.num_catchments,1)).astype(self.numpy_precision),
                    data,
                    ])
            else:
                data = read_map(file_path, (self.nx, self.ny), precision=self.map_precision)[self.catchment_x, self.catchment_y]
            
            self.params[param_name] = data
        
        self.params["gravity"] = self.gravity
        self.params["flood_manning"] = 0.1 * np.ones(self.num_catchments, dtype=self.numpy_precision)
        self.params["log_buffer_size"] = self.log_buffer_size
        self.params["adaptation_factor"] = self.adaptation_factor
        self.params["downstream_distance"][self.is_river_mouth] = self.river_mouth_distance
        self.params["num_flood_levels"] = self.num_flood_levels

        self.parameters_loaded = True
    
    def _init_river_depth(self):
        assert self.parameters_loaded, "init_river_depth should be called after load_parameters!"
        river_depth_init = np.zeros(self.num_catchments, dtype=self.numpy_precision)
        river_elevation = self.params["catchment_elevation"] - self.params["river_height"]

        for ii, jj in zip(reversed(range(self.num_catchments)), reversed(self.downstream_idx)):
            if ii == jj or jj < 0:
                river_depth_init[ii] = self.params["river_height"][ii]
            else:
                river_depth_init[ii] = max(
                    river_depth_init[jj] + river_elevation[jj] - river_elevation[ii],
                    0.0
                )
            river_depth_init[ii] = min(river_depth_init[ii], self.params["river_height"][ii])

        self.states["river_depth"] = river_depth_init
        self.states["river_storage"] = (
            self.params["river_width"] * self.states["river_depth"] * self.params["river_length"]
        )
        
    def _simple_init_other_states(self):
        for state in self.zero_init_states:
            self.states[state] = np.zeros(self.num_catchments, dtype=self.numpy_precision)
    
    def _create_input_matrix(self):
        location_file = Path(self.hires_map_dir) / "location.txt"
        with open(location_file, "r") as f:
            lines = f.readlines()

        data = lines[2].split()
        Nx, Ny = int(data[6]), int(data[7])
        West, East = float(data[2]), float(data[3])
        South, North = float(data[4]), float(data[5])
        Csize = float(data[8])
        hires_lon = np.linspace(West  + 0.5 * Csize, East  - 0.5 * Csize, Nx)
        hires_lat = np.linspace(North - 0.5 * Csize, South + 0.5 * Csize, Ny)
        lon2D, lat2D = np.meshgrid(hires_lon, hires_lat)  
        hires_lon_2D = lon2D.T
        hires_lat_2D = lat2D.T

        HighResGridArea = read_map(self.hires_map_dir / f"{self.hires_map_tag}.grdare.bin", (Nx, Ny), precision=self.hires_map_precision) * 1E6
        HighResCatchmentId = read_map(self.hires_map_dir / f"{self.hires_map_tag}.catmxy.bin", (Nx, Ny, 2), precision=self.hires_idx_precision)

        valid_mask = HighResCatchmentId[:, :, 0] > 0
        x_indices, y_indices = np.where(valid_mask)
        HighResCatchmentId = HighResCatchmentId -1 # 1-based to 0-based
        valid_x = HighResCatchmentId[x_indices, y_indices, 0]
        valid_y = HighResCatchmentId[x_indices, y_indices, 1]
        valid_areas = HighResGridArea[x_indices, y_indices]

        catchment_id_hires = np.ravel_multi_index((valid_x, valid_y), (self.nx, self.ny))
        ds_cls = getattr(importlib.import_module("CMF_GPU.utils.Dataset"), self.runoff_config.class_name)
        example_ds = ds_cls(
            **self.runoff_config.params
        )
        ro_lon, ro_lat = example_ds.get_coordinates()
        valid_lon = hires_lon_2D[x_indices, y_indices]
        valid_lat = hires_lat_2D[x_indices, y_indices]

        row_indices = find_indices_in(catchment_id_hires, self.catchment_id)
        col_indices = compute_runoff_id(ro_lon, ro_lat, valid_lon, valid_lat)

        col_mask = example_ds.get_mask()
        if col_mask is not None:
            col_mask = np.ravel(col_mask, order="C")
        else :
            col_mask = np.ones(len(ro_lat)*len(ro_lon), dtype=bool)
        # remap the col_indices based on the col_mask
        col_mapping = -np.ones_like(col_mask, dtype=np.int64)
        col_mapping[np.flatnonzero(col_mask)] = np.arange(col_mask.sum())
        col_indices = col_mapping[col_indices]
        row_mask = (row_indices != -1) # catchment_id not found in the hires map

        valid_count = len(np.unique(row_indices[row_mask]))

        if valid_count != self.num_catchments:
            print(
                f"Warning: { self.num_catchments - valid_count} catchment(s) will never receive valid runoff data "
                "because all their associated grid cells are invalid; their runoff input will always be 0. "
                "If there are many such catchments, this may indicate an issue with the input data or code logic."
            )
        self.runoff_matrix = csr_matrix((valid_areas[row_mask], (row_indices[row_mask], col_indices[row_mask])), shape=(len(self.catchment_id), col_mask.sum()), dtype=self.numpy_precision)

    def _create_files(self):
        os.makedirs(self.inp_dir, exist_ok=True)
        save_npz(self.inp_dir / "runoff_input_matrix.npz", self.runoff_matrix, compressed=True)
        snapshot_to_h5(self.inp_dir / "parameters.h5", self.params, "param", self.modules, np.dtype(self.numpy_precision).name, omit_hidden=True)
        snapshot_to_h5(self.inp_dir / "init_states.h5", self.states, "state", self.modules, np.dtype(self.numpy_precision).name, omit_hidden=True)
        with open(self.inp_dir / "gauge_info.pkl", "wb") as f:
            pickle.dump(self.gauge_info, f)
        np.savez_compressed(self.inp_dir / "catchment_id.npz", 
                            catchment_id=self.catchment_id, 
                            downstream_id=self.downstream_id,
                            catchment_x=self.catchment_x, 
                            catchment_y=self.catchment_y, 
                            bifurcation_catchment_idx=self.bifurcation_catchment_idx,
                            bifurcation_downstream_idx=self.bifurcation_downstream_idx,)

    def _summary(self):
        # ---------- basic stats ----------
        print(f"Number of (Integrated) Basins : {self.num_basins}")
        print(f"Number of Reservoirs         : {self.is_reservoir.sum()}")
        print(f"Number of Catchments        : {self.num_catchments}")
        print(f"Number of Bifurcation Paths :  {self.params['num_bifurcation_paths']}")
        print(f"Number of gauges              : {len(self.gauge_info)}")
        # ---------- 1) completeness check ----------

        errors = []          
        for mod in self.modules:
            cfg       = MODULES_INFO.get(mod, {})
            req_p     = set(cfg.get("params", [])) | set(cfg.get("scalar_params", []))
            req_s     = set(cfg.get("states", []))

            missing_p = [p for p in req_p if p not in self.params]
            missing_s = [s for s in req_s if s not in self.states]

            if missing_p:
                errors.append(f"{mod}: missing parameters {missing_p}")
            if missing_s:
                errors.append(f"{mod}: missing states {missing_s}")

        if errors:
            raise ValueError(
                "Required variables are missing:\n  " + "\n  ".join(errors)
            )
        # Build the full catalogue of *recognized* variables from all possible modules
        known_p, _, known_scalar = gather_all_keys("param", self.modules)
        known_s, _, _ = gather_all_keys("state", self.modules)
        known_p = set(known_p) | set(known_scalar)
        known_s = set(known_s)

        # Anything not in the catalogue is truly unrecognized → error out
        extra_p = [p for p in self.params if p not in known_p]
        extra_s = [s for s in self.states if s not in known_s]

        extra_errors = []
        if extra_p:
            extra_errors.append(f"unrecognized parameters {sorted(extra_p)}")
        if extra_s:
            extra_errors.append(f"unrecognized states {sorted(extra_s)}")

        if extra_errors:
            raise ValueError(
                "Unrecognized variables detected:\n  " + "\n  ".join(extra_errors)
            )
    
    def build_model_input_pipeline(self):
        """
        Main pipeline to load and process catchment data.
        """
        self._load_map_info()
        self._load_catchment_id()
        self._load_gauge_id()
        self._load_bifurcation_parameters()
        self._load_parameters()
        self._check_flow_direction()
        self._init_river_depth()
        self._simple_init_other_states()
        self._create_input_matrix()
        self._create_files()
        self._visualize_basin()
        self._summary()
        print("Model input pipeline built successfully.")
        
        
if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("./configs/glb_15min.yaml")
    default_global_catchment = DefaultGlobalCatchment(config)
    default_global_catchment.build_model_input_pipeline()