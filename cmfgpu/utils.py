import os

import numpy as np
import torch
from torch import distributed as dist


def get_global_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0

def get_local_rank():
    if dist.is_available() and dist.is_initialized():
        return int(os.environ.get("LOCAL_RANK", 0))
    else:
        return 0

def is_rank_zero():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    else:
        return True 

def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1

def setup_distributed():
    torch.multiprocessing.set_start_method("spawn", force=True)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = get_local_rank()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    return local_rank, rank, world_size


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

def find_indices_in_torch(a, b):
    
    sorted_b, order = torch.sort(b)
    pos = torch.bucketize(a, sorted_b, right=False)
    valid_mask = pos < len(sorted_b)
    hit_mask = torch.zeros_like(a, dtype=torch.bool)
    hit_mask[valid_mask] = (sorted_b[pos[valid_mask]] == a[valid_mask])
    index = torch.full_like(pos, -1, dtype=torch.int64)
    index[hit_mask] = order[pos[hit_mask]]
    
    return index
