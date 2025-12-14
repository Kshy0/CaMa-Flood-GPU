# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import numpy as np
import torch
from numba import njit


def torch_to_numpy_dtype(torch_dtype):
    dtype_mapping = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.float16: np.float16,
    }
    if torch_dtype not in dtype_mapping:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
    return dtype_mapping[torch_dtype]

@njit
def compute_group_to_rank(world_size: int, group_assignments: np.ndarray):
    """
    Compute a mapping from each original group ID to a rank, using a greedy load balance.

    Returns:
      full_map: array of length (max_original_id+1), where
                full_map[original_group_id] = assigned_rank (or -1 if absent)
    """
    # Handle edge cases early
    if world_size <= 0 or group_assignments.size == 0:
        max_gid = int(group_assignments.max()) if group_assignments.size > 0 else -1
        return np.full(max_gid + 1, -1, np.int64)

    # 1) Compress original IDs → 0..n_groups-1
    # unique_ids: sorted unique original IDs
    # inv: for each entry in group_assignments, its compressed ID
    unique_ids, inv = np.unique(group_assignments), None
    # compute inv via a dense id_map:
    max_gid = int(unique_ids[-1]) if unique_ids.size > 0 else -1
    id_map = np.full(max_gid + 1, -1, np.int64)
    id_map[unique_ids] = np.arange(unique_ids.size, dtype=np.int64)

    inv = id_map[group_assignments]
    n_groups = unique_ids.size

    # 2) Count sizes of each compressed group
    group_sizes = np.bincount(inv, minlength=n_groups).astype(np.int64)

    # 3) Greedy assignment: largest groups first → argmin(rank_loads)
    order = np.argsort(group_sizes)          # ascending
    rank_loads = np.zeros(world_size, np.int64)
    comp_to_rank = np.empty(n_groups, np.int64)

    for i in range(order.size - 1, -1, -1):  # iterate from largest to smallest
        g = order[i]
        r = int(np.argmin(rank_loads))       # lightest rank
        comp_to_rank[g] = r
        rank_loads[r] += group_sizes[g]

    # 4) Expand back to the full original ID space (fill -1 for IDs not present)
    full_map = np.full(max_gid + 1, -1, np.int64)
    # unique_ids are the only valid original IDs; assign directly
    full_map[unique_ids] = comp_to_rank

    return full_map
