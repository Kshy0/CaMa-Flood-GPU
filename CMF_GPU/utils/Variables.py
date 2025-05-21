import numpy as np
import torch
import os
from datetime import datetime
from typing import Optional
from omegaconf import DictConfig, ListConfig

#    catchment_id should be ordered to satisfy the following:
# 1. Catchments belonging to the same basin are grouped together;
# 2. Basins with bifurcation connections are placed adjacently;
# 3. Within each basin, catchments are ordered from upstream to downstream according to flow direction.
MODULES_INFO = {
    "base": {
        "params": [
            "downstream_idx", # equal to itself if the catchment is river mouth
            "num_catchments_per_basin", # size of associated basins due to bifurcation
            "is_river_mouth",
            "is_reservoir",
            "river_width",
            "river_length",
            "river_height",
            "catchment_elevation",
            "downstream_distance",
            "flood_depth_table",
            "catchment_area",
            "river_manning",
            "flood_manning",
            "runoff_input_matrix",
        ],
        "hidden_params": [
            "river_elevation",
            "river_area",
            "river_max_storage",
            "max_flood_area",
            "total_width_table",
            "total_storage_table",
            "flood_gradient_table",
        ],
        "scalar_params": [
            "num_gpus",
            "num_basins",
            "num_catchments",
            "num_flood_levels",
            "gravity", 
        ],
        "states": [
            "river_storage",
            "flood_storage",
            "river_depth",
            "river_outflow",
            "flood_depth",
            "flood_outflow",
            "river_cross_section_depth",
            "flood_cross_section_depth",
            "flood_cross_section_area",
        ],
        "hidden_states": [
            "total_storage",
            "outgoing_storage",
            "water_surface_elevation",
            "limit_rate",
            "river_inflow",
            "flood_area",
            "flood_fraction",
            "flood_inflow",
        ],

    },
    "adaptive_time_step": {
        "scalar_params": ["adaptation_factor"],
        "hidden_states": ["min_time_step"],
    },
    "log": {
        "scalar_params": ["log_buffer_size"],
        "hidden_states": [
            "total_storage_pre_sum",
            "total_storage_next_sum",
            "total_storage_new",
            "total_inflow",
            "total_outflow",
            "total_storage_stage_new",
            "total_river_storage",
            "total_flood_storage",
            "total_flood_area",
            "total_inflow_error",
            "total_stage_error",
        ],
    },
    "bifurcation": {
        "params": [
            "bifurcation_catchment_idx",
            "bifurcation_downstream_idx",
            "bifurcation_manning",
            "bifurcation_width",
            "bifurcation_length",
            "bifurcation_elevation",
        ],
        "scalar_params": [
            "num_bifurcation_paths",
            "num_bifurcation_levels"
        ],
        "states": [
            "bifurcation_outflow",
            "bifurcation_cross_section_depth",
        ],
        "hidden_states": [],
    },
    "reservoir": {
        "params": [
            "reservoir_catchment_idx",
            "conservation_volume",
            "emergency_volume",
            "normal_outflow",
            "flood_control_outflow",
        ],
        "hidden_params": [
            "reservoir_upstream_idx",
            "adjustment_volume",
            "adjustment_outflow",
        ],
        "scalar_params": [
            "num_reservoirs",
            "num_upstreams",
            "min_slope",
        ],
    },
}

LEGAL_STATS = [
    "min",
    "max",
    "mean",
]

LEGAL_AGG_ARRAYS = {
    ("num_catchments",): [
        "river_storage",
        "flood_storage",
        "river_depth",
        "river_outflow",
        "flood_depth",
        "flood_outflow",
        "river_cross_section_depth",
        "flood_cross_section_depth",
        "flood_cross_section_area",
        "total_storage",
        "water_surface_elevation",
        "limit_rate",
        "flood_area",
        "flood_fraction",
    ],
    ("num_bifurcation_paths", "num_bifurcation_levels"): [
        "bifurcation_outflow",
        "bifurcation_cross_section_depth",
    ],
    # ("num_bifurcation_paths",): [
    #     "bifurcation_outflow_sum",
    # ],
}

def _check_statistics(stats):
    if stats is None:
        return True
    if not isinstance(stats, (dict, DictConfig)):
        raise ValueError(f"CONFIG ERROR: statistics must be a dict or DictConfig: {stats}")

    legal_vars = []
    for _, variables in LEGAL_AGG_ARRAYS.items():
        legal_vars.extend(variables)

    for stat_type, variables in stats.items():
        if stat_type not in LEGAL_STATS:
            raise ValueError(f"CONFIG ERROR: Invalid statistic type: {stat_type}")
        if variables is not None:
            for var in variables:
                if var not in legal_vars:
                    raise ValueError(f"CONFIG ERROR: Invalid variable '{var}' in statistics[{stat_type}]")
    return True

def _check_device_indices(device_indices):
    if isinstance(device_indices, int):
        device_indices = [device_indices]
    elif isinstance(device_indices, (ListConfig, list)):
        if not all(isinstance(i, int) for i in device_indices):
            raise ValueError(f"CONFIG ERROR: device_indices must be a list of integers: {device_indices}")
    else:
        raise ValueError(f"CONFIG ERROR: Invalid device_indices type: {type(device_indices)}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")

    for i in device_indices:
        if i >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {i} is out of range. "
                f"Only {torch.cuda.device_count()} devices available."
            )
        try:
            _ = torch.cuda.get_device_properties(i)
        except RuntimeError as e:
            raise RuntimeError(f"CUDA device {i} is not usable: {e}")
    return True

def _check_parameter_config(cfg):
    if not isinstance(cfg, (dict, DictConfig)):
        raise ValueError(f"CONFIG ERROR: parameter_config must be a dict or DictConfig: {cfg}")

    required = [
        "experiment_name", "map_dir", "hires_map_dir",
        "working_dir", "inp_dir", "precision"
    ]
    for key in required:
        if key not in cfg:
            raise ValueError(f"CONFIG ERROR: parameter_config missing required key '{key}'")

    if not isinstance(cfg["experiment_name"], str) or not cfg["experiment_name"]:
        raise ValueError(f"CONFIG ERROR: Invalid experiment_name: {cfg['experiment_name']}")

    for key in ["map_dir", "hires_map_dir", "working_dir"]:
        if not isinstance(cfg[key], str):
            raise ValueError(f"CONFIG ERROR: {key} must be a string path, got {type(cfg[key])}")
        if not os.path.isdir(cfg[key]):
            raise ValueError(f"CONFIG ERROR: {key} does not exist or is not a directory: {cfg[key]}")

    inp_dir_parent = os.path.dirname(cfg["inp_dir"])
    if not os.path.isdir(inp_dir_parent):
        raise ValueError(f"CONFIG ERROR: inp_dir parent does not exist or is not a directory: {inp_dir_parent}")

    if cfg["precision"] not in ["float32", "float64"]:
        raise ValueError(f"CONFIG ERROR: Invalid precision: {cfg['precision']}")
    
    return True

def _check_runoff_config(cfg):
    if not isinstance(cfg, (dict, DictConfig)):
        raise ValueError(f"CONFIG ERROR: runoff_config must be a dict or DictConfig: {cfg}")
    if "class_name" not in cfg:
        raise ValueError("CONFIG ERROR: runoff_config must contain class_name")
    if "params" not in cfg:
        raise ValueError("CONFIG ERROR: runoff_config must contain params")
    return True

def _check_simulation_config(cfg):
    if not isinstance(cfg, (dict, DictConfig)):
        raise ValueError(f"CONFIG ERROR: simulation_config must be a dict or DictConfig: {cfg}")
    modules = cfg.get("modules", None)
    if not isinstance(modules, (ListConfig, list)):
        raise ValueError(f"CONFIG ERROR: simulation_config must contain a list of modules: {cfg}")
    for module in modules:
        if module not in MODULES_INFO:
            raise ValueError(f"CONFIG ERROR: Invalid module in simulation_config: {module}")
    required = [
        "experiment_name", "working_dir", "out_dir", "inp_dir", "states_file", "precision",
        "start_date", "end_date", "device", "device_indices", "time_step", "default_num_sub_steps",
    ]
    for key in required:
        if key not in cfg:
            raise ValueError(f"CONFIG ERROR: simulation_config missing required key '{key}'")
        
    if cfg["precision"] not in ["float32", "float64"]:
        raise ValueError(f"CONFIG ERROR: Invalid precision: {cfg['precision']}")
    
    if not isinstance(cfg["experiment_name"], str) or not cfg["experiment_name"]:
        raise ValueError(f"CONFIG ERROR: Invalid experiment_name: {cfg['experiment_name']}")
    
    for key in ["working_dir"]:
        if not isinstance(cfg[key], str):
            raise ValueError(f"CONFIG ERROR: {key} must be a string path, got {type(cfg[key])}")
        if not os.path.isdir(cfg[key]):
            raise ValueError(f"CONFIG ERROR: {key} does not exist or is not a directory: {cfg[key]}")
        
    for key in ["out_dir", "inp_dir"]:
        parent_dir = os.path.dirname(cfg[key])
        if not os.path.isdir(parent_dir):
            raise ValueError(f"CONFIG ERROR: {key} parent does not exist or is not a directory: {parent_dir}")

    try:
        datetime.strptime(cfg["start_date"], "%Y-%m-%d")
        datetime.strptime(cfg["end_date"], "%Y-%m-%d")
    except Exception:
        raise ValueError("CONFIG ERROR: start_date and end_date must be in YYYY-MM-DD format")

    if cfg["device"] != "gpu":
        raise ValueError(f"CONFIG ERROR: Invalid device (only 'gpu' supported): {cfg['device']}")

    _check_device_indices(cfg["device_indices"])

    if not isinstance(cfg["time_step"], (int, float)) or cfg["time_step"] <= 0:
        raise ValueError(f"CONFIG ERROR: runtime_config time_step must be positive number")
    if not isinstance(cfg["default_num_sub_steps"], int) or cfg["default_num_sub_steps"] <= 0:
        raise ValueError(f"CONFIG ERROR: runtime_config default_num_sub_steps must be positive integer")
    _check_statistics(cfg["statistics"])
    return True

CONFIG_REQUIRED_KEYS = {
    "parameter_config": _check_parameter_config,
    "runoff_config": _check_runoff_config,
    "simulation_config": _check_simulation_config,
}

SCALAR_TYPES = {
    "num_gpus": int,
    "num_basins": int,
    "num_catchments": int,
    "num_flood_levels": int,
    "gravity": float,
    "adaptation_factor": float,
    "log_buffer_size": int,
    "num_bifurcation_paths": int,
    "num_bifurcation_levels": int,
    "num_reservoirs": int,
    "num_upstreams": int,
    "min_slope": float,
}

SPECIAL_ARRAY_TYPES = {
    "downstream_idx": np.int64,
    "num_catchments_per_basin": np.int64,
    "is_river_mouth": np.bool,
    "is_reservoir": np.bool,
    "bifurcation_catchment_idx": np.int64,
    "bifurcation_downstream_idx": np.int64,
}



def _compute_greedy_partition(num_gpus: int, basin_sizes: np.ndarray):
    """
    Greedily assign contiguous basins to GPUs.

    Args
    ----
    num_gpus      : int
    basin_sizes   : 1-D int array, number of catchments in each basin
                    (basins are contiguous in the original ordering)

    Returns
    -------
    catchment_order : 1-D int array
        Permutation that maps GPU-major order → original catchment id.
    split_indices   : 1-D int array, length = num_gpus + 1
        Slice boundaries for every GPU **starting with 0** and
        ending with total number of catchments, i.e.  
        slice(i) = catchment_order[ split[i] : split[i+1] ]
    """
    n_basins = basin_sizes.size

    # pre-compute starting offsets of each basin
    basin_starts = np.empty(n_basins, dtype=np.int64)
    offset = 0
    for i in range(n_basins):
        basin_starts[i] = offset
        offset += basin_sizes[i]
    total_catch = offset

    # arrays to track assignment + loads
    loads = np.zeros(num_gpus, dtype=np.int64)           # current load per GPU
    assignment = -np.ones(n_basins, dtype=np.int64)      # basin → GPU

    # indices sorted by basin size (largest first)
    sorted_idx = np.argsort(basin_sizes)[::-1]

    # greedy fill
    for b in sorted_idx:
        gpu = np.argmin(loads)           # GPU with the least load
        assignment[b] = gpu
        loads[gpu] += basin_sizes[b]

    # build permutation & split indices in GPU-major order
    catchment_order = np.empty(total_catch, dtype=np.int64)
    split_indices = np.empty(num_gpus + 1, dtype=np.int64)
    split_indices[0] = 0

    pos = 0
    for g in range(num_gpus):
        for b in range(n_basins):
            if assignment[b] == g:
                start = basin_starts[b]
                size = basin_sizes[b]
                for k in range(size):
                    catchment_order[pos] = start + k
                    pos += 1
        split_indices[g + 1] = pos      # end-exclusive for GPU g
    inverse_order = np.argsort(catchment_order)              # original → GPU-major
    return catchment_order, inverse_order, split_indices

def _compute_sub_order(
    inverse_order: np.ndarray,
    split_indices: np.ndarray,
    input_idx: Optional[np.ndarray],
):
    if input_idx is None:
        return None, None
    positions     = inverse_order[input_idx]                 # GPU-major positions

    sub_order = np.argsort(positions)                        # permutation

    num_gpus = split_indices.size - 1
    counts   = np.zeros(num_gpus, dtype=np.int64)
    for g in range(num_gpus):
        start, end = split_indices[g], split_indices[g + 1]
        counts[g]  = np.sum((positions >= start) & (positions < end))

    # ── 关键改动：首位补 0，形如 [0, …, total] ────────────────────────────
    sub_split_indices        = np.empty(num_gpus + 1, dtype=np.int64)
    sub_split_indices[0]     = 0
    sub_split_indices[1:]    = np.cumsum(counts)
    # （此处 np.cumsum(counts)[-1] 一定等于 len(input_idx)）

    return sub_order, sub_split_indices

# ---------------------------------------------------------------------
# 3. Map downstream catchments → local indices
# ---------------------------------------------------------------------
def _compute_local_idx(
    inverse_order: np.ndarray,
    split_indices: np.ndarray,
    input_idx: Optional[np.ndarray],
):
    if input_idx is None:
        return None
    positions = inverse_order[input_idx]            # GPU-major positions

    # GPU owning each downstream catchment
    gpu_ids = np.searchsorted(split_indices, positions, side='right') - 1

    # local index = global position − slice start
    local_idx = positions - split_indices[gpu_ids]

    return local_idx


ORDERS = {
    # Base catchment partitioning
    ("catchment_order", "inverse_order", "split_indices"): lambda p, o: 
        _compute_greedy_partition(p["num_gpus"], p["num_catchments_per_basin"]),
    
    "downstream_idx": lambda p, o: 
        _compute_local_idx(o["inverse_order"], o["split_indices"], p["downstream_idx"]),

    # Bifurcation-related mappings
    ("bifurcation_order", "bifurcation_split_indices"): lambda p, o: 
        _compute_sub_order(o["inverse_order"], o["split_indices"], p.get("bifurcation_catchment_idx", None)),

    "bifurcation_idx": lambda p, o: 
        _compute_local_idx(o["inverse_order"], o["split_indices"], p.get("bifurcation_catchment_idx", None)),

    "bifurcation_downstream_idx": lambda p, o: 
        _compute_local_idx(o["inverse_order"], o["split_indices"], p.get("bifurcation_downstream_idx", None)),

    # Reservoir-related mappings
    ("reservoir_order", "reservoir_split_indices"): lambda p, o: 
        _compute_sub_order(o["inverse_order"], o["split_indices"], p.get("reservoir_catchment_idx", None)),

    "reservoir_idx": lambda p, o: 
        _compute_local_idx(o["inverse_order"], o["split_indices"], p.get("reservoir_catchment_idx", None)),
}

def _delete(x, orders, devices):
    return [None] * len(devices)

def default_array_split(x, orders, devices):
    splits = np.split(x, orders["split_indices"][1:-1])
    return [torch.tensor(arr, device=devices[i]) for i, arr in enumerate(splits)]

def default_scalar_split(x, orders, devices):
    return [x] * len(devices)

def _split_runoff_input_matrix(x, orders, devices):
    x = x[orders["catchment_order"]]  

    split_indices = orders["split_indices"]
    splits = [x[split_indices[i]:split_indices[i+1]] for i in range(len(split_indices) - 1)]

    result = []
    dtype = torch.float32 if x.dtype == np.float32 else torch.float64

    for i, split in enumerate(splits):
        crow = torch.tensor(split.indptr, dtype=torch.int64)
        cols = torch.tensor(split.indices, dtype=torch.int64)
        vals = torch.tensor(split.data, dtype=dtype)
        shape = split.shape

        torch_sparse = torch.sparse_csr_tensor(crow, cols, vals, size=shape).to(devices[i])
        result.append(torch_sparse)

    return result

def _update_downstream_idx(x, orders, devices):
    splits = np.split(orders["downstream_idx"][orders["catchment_order"]], orders["split_indices"][1:-1])
    return [torch.tensor(arr, device=devices[i]) for i, arr in enumerate(splits)]

def _update_bifurcation_catchment_idx(x, orders, devices):
    splits = np.split(orders["bifurcation_catchment_idx"][orders["bifurcation_order"]], orders["bifurcation_split_indices"][1:-1])
    return [torch.tensor(arr, device=devices[i]) for i, arr in enumerate(splits)]

def _update_bifurcation_downstream_idx(x, orders, devices):
    splits = np.split(orders["bifurcation_downstream_idx"][orders["bifurcation_order"]], orders["bifurcation_split_indices"][1:-1])
    return [torch.tensor(arr, device=devices[i]) for i, arr in enumerate(splits)]

def _update_reservoir_catchment_idx(x, orders, devices):
    splits = np.split(orders["reservoir_catchment_idx"][orders["reservoir_order"]], orders["reservoir_split_indices"][1:-1])
    return [torch.tensor(arr, device=devices[i]) for i, arr in enumerate(splits)]

def _split_bifurcation_array(x, orders, devices):
    splits = np.split(x[orders["bifurcation_order"]], orders["bifurcation_split_indices"][1:-1])
    return [torch.tensor(arr, device=devices[i]) for i, arr in enumerate(splits)]

def _split_reservoir_array(x, orders, devices):
    splits = np.split(x[orders["reservoir_order"]], orders["reservoir_split_indices"][1:-1])
    return [torch.tensor(arr, device=devices[i]) for i, arr in enumerate(splits)]

# default: np.split(ARRAY, o["split_indices"][1::-1])
# None means that the array will not be used in the simulation
SPECIAL_SPLIT_ARRAYS = {
    "runoff_input_matrix": _split_runoff_input_matrix,
    "downstream_idx": _update_downstream_idx,
    "num_catchments_per_basin": _delete,
    "bifurcation_catchment_idx": _update_bifurcation_catchment_idx,
    "bifurcation_downstream_idx": _update_bifurcation_downstream_idx,
    "bifurcation_manning": _split_bifurcation_array,
    "bifurcation_width": _split_bifurcation_array,
    "bifurcation_length": _split_bifurcation_array,
    "bifurcation_elevation": _split_bifurcation_array,
    "bifurcation_outflow": _split_bifurcation_array,
    "reservoir_catchment_idx": _update_reservoir_catchment_idx,
    "conservation_volume": _split_reservoir_array,
    "emergency_volume": _split_reservoir_array,
    "normal_outflow": _split_reservoir_array,
    "flood_control_outflow": _split_reservoir_array,
}

SPECIAL_SPLIT_SCALARS = {
    "num_catchments": lambda p, o, d: [int(i) for i in np.diff(o["split_indices"])],
    "num_bifurcation_paths": lambda p, o, d: [int(i) for i in np.diff(o["bifurcation_split_indices"])],
    "num_reservoirs": lambda p, o, d: [int(i) for i in np.diff(o["reservoir_split_indices"])],
}

HIDDEN_PARAMS = {
    # base
    "river_elevation": lambda p: p["catchment_elevation"] - p["river_height"],
    "river_area": lambda p: p["river_length"] * p["river_width"],
    "river_max_storage": lambda p: p["river_area"] * p["river_height"],
    "max_flood_area": lambda p: p["river_area"] + p["catchment_area"],
    "total_width_table": lambda p: torch.cat([
        p["river_width"][:, None],
        p["river_width"][:, None] + torch.linspace(
            0, 1, p["num_flood_levels"] + 1, device=p["river_width"].device
        )[None, :] * p["catchment_area"][:, None] / p["river_length"][:, None],
        (p["river_width"] + p["catchment_area"] / p["river_length"])[:, None]
    ], dim=1),
    "total_storage_table": lambda p: torch.cat([
        torch.zeros((p["num_catchments"], 1), device=p["river_length"].device),
        torch.cumsum(
            p["river_length"][:, None] *
            0.5 * (p["total_width_table"][:, :-1] + p["total_width_table"][:, 1:]) *
            (p["flood_depth_table"][:, 1:] - p["flood_depth_table"][:, :-1]),
            dim=1
        )
    ], dim=1),
    "flood_gradient_table": lambda p: torch.where(
        (p["total_width_table"][:, 1:] - p["total_width_table"][:, :-1]) != 0,
        (p["flood_depth_table"][:, 1:] - p["flood_depth_table"][:, :-1]) /
        (p["total_width_table"][:, 1:] - p["total_width_table"][:, :-1]),
        torch.zeros_like(p["flood_depth_table"][:, 1:])
    ),
    # reservoir
    "adjustment_volume": lambda p: p["conservation_volume"] + p["emergency_volume"] * 0.1,
    "adjustment_outflow": lambda p: (
        torch.minimum(p["normal_outflow"], (
            (p["conservation_volume"] * 0.7 + p["normal_outflow"] * (365. * 24 * 60 * 60) / 4)
            / (180. * 24 * 60 * 60)
        )) * 1.5 + p["flood_control_outflow"]
    ) * 0.5,
}

def default_shape(params):
    return (params["num_catchments"],)

def _log_shape(params):
    return (params["log_buffer_size"],)

def _bifurcation_1D_shape(params):
    return (params["num_bifurcation_paths"],)

def _bifurcation_2D_shape(params):
    return (params["num_bifurcation_paths"], params["num_bifurcation_levels"])

def _reservoir_shape(params):
    return (params["num_reservoirs"],)

MODULE_DEPENDENT_ARRAYS = {
    "is_reservoir": lambda p, m: p["is_reservoir"] if "reservoir" in m else np.zeros_like(p["is_reservoir"]),
}


# all zeros
SPECIAL_HIDDEN_STATES = {
    # adaptive_time_step
    "min_time_step": lambda p: (1,),
    # log
    "total_storage_pre_sum": _log_shape,
    "total_storage_next_sum": _log_shape,
    "total_storage_new": _log_shape,
    "total_inflow": _log_shape,
    "total_outflow": _log_shape,
    "total_storage_stage_new": _log_shape,
    "total_river_storage": _log_shape,
    "total_flood_storage": _log_shape,
    "total_flood_area": _log_shape,
    "total_inflow_error": _log_shape,
    "total_stage_error": _log_shape,
    # bifurcation
}
    
SPECIAL_INPUT_SHAPES = {
    # base
    "runoff_input_matrix": lambda p: (p["num_catchments"], p["runoff_input_matrix"].shape[1]), # only check the first dimension, TODO: trim the second dimension by the mask
    "flood_depth_table": lambda p: (p["num_catchments"], p["num_flood_levels"] + 3),
    "num_catchments_per_basin": lambda p: (p["num_basins"],),
    # bifurcation
    "bifurcation_catchment_id": _bifurcation_1D_shape,
    "bifurcation_downstream_id": _bifurcation_1D_shape,
    "bifurcation_length": _bifurcation_1D_shape,
    "bifurcation_manning": _bifurcation_2D_shape,
    "bifurcation_width": _bifurcation_2D_shape,
    "bifurcation_elevation": _bifurcation_2D_shape,
    "bifurcation_outflow": _bifurcation_2D_shape,
    # reservoir
    "reservoir_catchment_idx": _reservoir_shape,
    "conservation_volume": _reservoir_shape,
    "emergency_volume": _reservoir_shape,
    "normal_outflow": _reservoir_shape,
    "flood_control_outflow": _reservoir_shape,
}


