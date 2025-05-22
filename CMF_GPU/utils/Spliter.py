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

    positions = inverse_order[input_idx]            # GPU-major positions

    # GPU owning each downstream catchment
    gpu_ids = np.searchsorted(split_indices, positions, side='right') - 1

    # local index = global position − slice start
    local_idx = positions - split_indices[gpu_ids]

    return local_idx


ORDERS = {
    # Base catchment partitioning
    ("base", ("catchment_order", "inverse_order", "split_indices")): lambda p, o, n: 
        _compute_greedy_partition(n, p["num_catchments_per_basin"][()]),
    
    ("base",("downstream_idx")): lambda p, o, n: 
        _compute_local_idx(o["inverse_order"][()], o["split_indices"][()], p["downstream_idx"][()]),

    # Bifurcation-related mappings
    ("bifurcation",("bifurcation_order", "bifurcation_split_indices")): lambda p, o, n: 
        _compute_sub_order(o["inverse_order"][()], o["split_indices"][()], p["bifurcation_catchment_idx"][()]),

    ("bifurcation",("bifurcation_idx")): lambda p, o, n: 
        _compute_local_idx(o["inverse_order"][()], o["split_indices"][()], p["bifurcation_catchment_idx"][()]),

    ("bifurcation",("bifurcation_downstream_idx")): lambda p, o, n: 
        _compute_local_idx(o["inverse_order"][()], o["split_indices"][()], p["bifurcation_downstream_idx"][()]),

    # Reservoir-related mappings
    ("reservoir",("reservoir_order", "reservoir_split_indices")): lambda p, o, n: 
        _compute_sub_order(o["inverse_order"][()], o["split_indices"][()], p["reservoir_catchment_idx"][()]),

    ("reservoir",("reservoir_idx")): lambda p, o, n: 
        _compute_local_idx(o["inverse_order"][()], o["split_indices"][()], p["reservoir_catchment_idx"][()]),
}

def _delete(ds, orders, rank):
    return ds[()] # to be modified

def default_array_split(ds, orders, rank):
    split_indices = orders["split_indices"]
    start_idx = split_indices[rank]
    end_idx = split_indices[rank + 1]
    sliced_data = ds[()][orders["catchment_order"][start_idx:end_idx]]
    return torch.tensor(sliced_data) 

def default_scalar_split(x, orders, rank):
    return x[()]

def split_runoff_input_matrix(file, orders, rank):
    x = load_npz(file)
    split_indices = orders["split_indices"]
    start_idx = split_indices[rank]
    end_idx = split_indices[rank + 1]
    sliced_data = x[orders["catchment_order"][start_idx:end_idx]]

    dtype = torch.float32 if x.dtype == np.float32 else torch.float64

    crow = torch.tensor(sliced_data.indptr, dtype=torch.int64)
    cols = torch.tensor(sliced_data.indices, dtype=torch.int64)
    vals = torch.tensor(sliced_data.data, dtype=dtype)
    shape = sliced_data.shape

    torch_sparse = torch.sparse_csr_tensor(crow, cols, vals, size=shape)

    return torch_sparse

def _update_downstream_idx(ds, orders, rank):
    split_indices = orders["split_indices"]
    start_idx = split_indices[rank]
    end_idx = split_indices[rank + 1]
    sliced_data = orders["downstream_idx"][orders["catchment_order"][start_idx:end_idx]]
    return torch.tensor(sliced_data)

def _update_bifurcation_catchment_idx(ds, orders, rank):
    split_indices = orders["bifurcation_split_indices"]
    start_idx = split_indices[rank]
    end_idx = split_indices[rank + 1]
    sliced_data = orders["bifurcation_catchment_idx"][orders["bifurcation_order"][start_idx:end_idx]]
    return torch.tensor(sliced_data)

def _update_bifurcation_downstream_idx(ds, orders, rank):
    split_indices = orders["bifurcation_split_indices"]
    start_idx = split_indices[rank]
    end_idx = split_indices[rank + 1]
    sliced_data = orders["bifurcation_downstream_idx"][orders["bifurcation_order"][start_idx:end_idx]]
    return torch.tensor(sliced_data)

def _update_reservoir_catchment_idx(ds, orders, rank):
    split_indices = orders["reservoir_split_indices"]
    start_idx = split_indices[rank]
    end_idx = split_indices[rank + 1]
    sliced_data = orders["reservoir_catchment_idx"][orders["reservoir_order"][start_idx:end_idx]]
    return torch.tensor(sliced_data)

def _split_bifurcation_array(ds, orders, rank):
    split_indices = orders["bifurcation_split_indices"]
    start_idx = split_indices[rank]
    end_idx = split_indices[rank + 1]
    sliced_data = ds[()][orders["bifurcation_order"][start_idx:end_idx]]
    return torch.tensor(sliced_data)

def _split_is_reservoir(ds, orders, rank):
    if "reservoir" not in orders["modules"]:
        sliced_data = np.zeros_like(ds[()])
    else:
        split_indices = orders["reservoir_split_indices"]
        start_idx = split_indices[rank]
        end_idx = split_indices[rank + 1]
        sliced_data = ds[()][orders["reservoir_order"][start_idx:end_idx]]
    return torch.tensor(sliced_data)

def _split_reservoir_array(ds, orders, rank):
    split_indices = orders["reservoir_split_indices"]
    start_idx = split_indices[rank]
    end_idx = split_indices[rank + 1]
    sliced_data = ds[()][orders["reservoir_order"][start_idx:end_idx]]
    return torch.tensor(sliced_data)

# default: np.split(ARRAY, o["split_indices"][1::-1])
# None means that the array will not be used in the simulation
SPECIAL_SPLIT_ARRAYS = {
    "downstream_idx": _update_downstream_idx,
    "num_catchments_per_basin": _delete,
    "bifurcation_catchment_idx": _update_bifurcation_catchment_idx,
    "bifurcation_downstream_idx": _update_bifurcation_downstream_idx,
    "bifurcation_manning": _split_bifurcation_array,
    "bifurcation_width": _split_bifurcation_array,
    "bifurcation_length": _split_bifurcation_array,
    "bifurcation_elevation": _split_bifurcation_array,
    "bifurcation_outflow": _split_bifurcation_array,
    "is_reservoir": _split_is_reservoir,
    "reservoir_catchment_idx": _update_reservoir_catchment_idx,
    "conservation_volume": _split_reservoir_array,
    "emergency_volume": _split_reservoir_array,
    "normal_outflow": _split_reservoir_array,
    "flood_control_outflow": _split_reservoir_array,
}

SPECIAL_SPLIT_SCALARS = {
    "num_basins": _delete,
    "num_catchments": lambda ds, o, r: int(np.diff(o["split_indices"])[r]),
    "num_bifurcation_paths": lambda ds, o, r: int(np.diff(o["bifurcation_split_indices"])[r]),
    "num_reservoirs": lambda ds, o, r: int(np.diff(o["reservoir_split_indices"])[r]),
}

