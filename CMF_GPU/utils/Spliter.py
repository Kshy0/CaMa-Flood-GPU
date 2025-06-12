import numpy as np
import torch
from scipy.sparse import load_npz
from numba import njit

@njit
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

@njit
def _compute_sub_order(
    inverse_order: np.ndarray,
    split_indices: np.ndarray,
    input_idx: np.ndarray,
):
    positions     = inverse_order[input_idx]                 # GPU-major positions

    sub_order = np.argsort(positions)                        # permutation

    num_gpus = split_indices.size - 1
    counts   = np.zeros(num_gpus, dtype=np.int64)
    for g in range(num_gpus):
        start, end = split_indices[g], split_indices[g + 1]
        counts[g]  = np.sum((positions >= start) & (positions < end))

    sub_split_indices        = np.empty(num_gpus + 1, dtype=np.int64)
    sub_split_indices[0]     = 0
    sub_split_indices[1:]    = np.cumsum(counts)

    return sub_order, sub_split_indices

# ---------------------------------------------------------------------
# 3. Map downstream catchments → local indices
# ---------------------------------------------------------------------

@njit
def _compute_local_idx(
    inverse_order: np.ndarray,
    split_indices: np.ndarray,
    input_idx: np.ndarray,
):

    global_idx = inverse_order[input_idx]            # GPU-major positions

    # GPU owning each downstream catchment
    gpu_idx = np.searchsorted(split_indices, global_idx, side='right') - 1

    # local index = global position − slice start
    local_idx = global_idx - split_indices[gpu_idx]

    return local_idx

ORDERS = {
    "base": {
        ("catchment_order", "inverse_order", "split_indices"): lambda p, o, n: 
            _compute_greedy_partition(n, p["num_catchments_per_basin"][()]),

        ("save_order", "save_split_indices"): lambda p, o, n:
            _compute_sub_order(o["inverse_order"][()], o["split_indices"][()], p["catchment_save_idx"][()]),

        ("catchment_save_idx"): lambda p, o, n:
            _compute_local_idx(o["inverse_order"][()], o["split_indices"][()], p["catchment_save_idx"][()]),

        ("downstream_idx"): lambda p, o, n: 
            _compute_local_idx(o["inverse_order"][()], o["split_indices"][()], p["downstream_idx"][()]),
    },
    
    "bifurcation": {
        ("bifurcation_order", "bifurcation_split_indices"): lambda p, o, n: 
            _compute_sub_order(o["inverse_order"][()], o["split_indices"][()], p["bifurcation_catchment_idx"][()]),

        ("bifurcation_catchment_idx"): lambda p, o, n: 
            _compute_local_idx(o["inverse_order"][()], o["split_indices"][()], p["bifurcation_catchment_idx"][()]),

        ("bifurcation_downstream_idx"): lambda p, o, n: 
            _compute_local_idx(o["inverse_order"][()], o["split_indices"][()], p["bifurcation_downstream_idx"][()]),
    },

    "reservoir": {
        ("reservoir_order", "reservoir_split_indices"): lambda p, o, n: 
            _compute_sub_order(o["inverse_order"][()], o["split_indices"][()], p["reservoir_catchment_idx"][()]),

        ("reservoir_idx"): lambda p, o, n: 
            _compute_local_idx(o["inverse_order"][()], o["split_indices"][()], p["reservoir_catchment_idx"][()]),
    },
}

def _delete(ds, orders, rank):
    return ds[()] # to be modified

def _split_array(x, orders, rank, indices_name, order_name):
    """
    Split an array based on the provided indices and order.
    
    Args:
        ds: The dataset containing the array to be split.
        orders: The orders dictionary containing split indices and catchment order.
        rank: The rank of the current process.
        indices_name: The name of the split indices in the orders dictionary.
        order_name: The name of the catchment order in the orders dictionary.
    
    Returns:
        A tensor containing the sliced data for the specified rank.
    """
    split_indices = orders[indices_name]
    start_idx = split_indices[rank]
    end_idx = split_indices[rank + 1]
    sliced_data = x[orders[order_name][start_idx:end_idx]]
    return torch.tensor(sliced_data)

def _update_idx(ds, orders, rank, indices_name, idx_name, order_name):
    """
    Update an array based on the provided indices and order.
    
    Args:
        ds: The dataset containing the array to be updated.
        orders: The orders dictionary containing split indices and catchment order.
        rank: The rank of the current process.
        indices_name: The name of the split indices in the orders dictionary.
        order_name: The name of the catchment order in the orders dictionary.
    
    Returns:
        A tensor containing the updated data for the specified rank.
    """
    split_indices = orders[indices_name]
    start_idx = split_indices[rank]
    end_idx = split_indices[rank + 1]
    sliced_data = orders[idx_name][orders[order_name][start_idx:end_idx]]
    return torch.tensor(sliced_data)

def default_array_split(x, orders, rank, indices_name="split_indices", order_name="catchment_order"):
    return _split_array(x, orders, rank, indices_name, order_name)

def default_scalar_split(x, orders, rank):
    return x[()]

def split_runoff_input_matrix(file, orders, rank):
    x = load_npz(file)
    split_indices = orders["split_indices"]
    start_idx = split_indices[rank]
    end_idx = split_indices[rank + 1]
    sliced_data = x[orders["catchment_order"][start_idx:end_idx]].T.tocoo()
    torch_sparse = torch.sparse_coo_tensor(
        indices=torch.tensor(np.array([sliced_data.row, sliced_data.col]), dtype=torch.long),
        values=torch.tensor(sliced_data.data, dtype=torch.float32),
        size=sliced_data.shape
    )
    return torch_sparse

def _split_is_reservoir(ds, orders, rank):
    if "reservoir" not in orders["modules"]:
        sliced_data = np.zeros_like(ds[()])
    else:
        split_indices = orders["reservoir_split_indices"]
        start_idx = split_indices[rank]
        end_idx = split_indices[rank + 1]
        sliced_data = ds[()][orders["reservoir_order"][start_idx:end_idx]]
    return torch.tensor(sliced_data)


# default: np.split(ARRAY, o["split_indices"][1::-1])
# None means that the array will not be used in the simulation
SPECIAL_SPLIT_ARRAYS = {
    "catchment_save_idx": lambda x, o, r: _update_idx(x, o, r, "save_split_indices", "catchment_save_idx", "save_order"),
    "downstream_idx": lambda x, o, r: _update_idx(x, o, r, "split_indices", "downstream_idx", "catchment_order"),
    "num_catchments_per_basin": _delete,
    "bifurcation_catchment_idx": lambda x, o, r: _update_idx(x, o, r, "bifurcation_split_indices", "bifurcation_catchment_idx", "bifurcation_order"),
    "bifurcation_downstream_idx": lambda x, o, r: _update_idx(x, o, r, "bifurcation_split_indices", "bifurcation_downstream_idx", "bifurcation_order"),
    "bifurcation_manning": lambda x, o, r: _split_array(x, o, r, "bifurcation_split_indices", "bifurcation_order"),
    "bifurcation_width": lambda x, o, r: _split_array(x, o, r, "bifurcation_split_indices", "bifurcation_order"),
    "bifurcation_length": lambda x, o, r: _split_array(x, o, r, "bifurcation_split_indices", "bifurcation_order"),
    "bifurcation_elevation": lambda x, o, r: _split_array(x, o, r, "bifurcation_split_indices", "bifurcation_order"),
    "bifurcation_outflow": lambda x, o, r: _split_array(x, o, r, "bifurcation_split_indices", "bifurcation_order"),
    "is_reservoir": _split_is_reservoir,
    "reservoir_catchment_idx": lambda x, o, r: _update_idx(x, o, r, "reservoir_split_indices", "reservoir_catchment_idx", "reservoir_order"),
    "conservation_volume": lambda x, o, r: _split_array(x, o, r, "reservoir_split_indices", "reservoir_order"),
    "emergency_volume": lambda x, o, r: _split_array(x, o, r, "reservoir_split_indices", "reservoir_order"),
    "normal_outflow": lambda x, o, r: _split_array(x, o, r, "reservoir_split_indices", "reservoir_order"),
    "flood_control_outflow": lambda x, o, r: _split_array(x, o, r, "reservoir_split_indices", "reservoir_order"),
}

SPECIAL_SPLIT_SCALARS = {
    "num_basins": _delete,
    "num_catchments": lambda ds, o, r: int(np.diff(o["split_indices"])[r]),
    "num_bifurcation_paths": lambda ds, o, r: int(np.diff(o["bifurcation_split_indices"])[r]),
    "num_reservoirs": lambda ds, o, r: int(np.diff(o["reservoir_split_indices"])[r]),
}

