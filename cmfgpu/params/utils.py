# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from collections import defaultdict
from pathlib import Path

import numpy as np
from numba import njit


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

def read_bifori(bifori_file: Path, rivhgt_2d: np.ndarray, bif_levels_to_keep: int):
    """
    Vectorized reader for bifori.txt replicating the core logic of Fortran set_bifparam:
      - Keep only paths where any width>0 within the first keepN levels
      - Compute dph from wth(1) with an empirical formula and clamp to [0.5, max(rivhgt(up), rivhgt(dn))]
      - Keep widths only for the first keepN levels
      - Elevation table: level 0 -> pelv - dph (if width>0); other levels -> pelv + ilev - 1 (if width>0); else 1e20
    Returns:
      (keepN, pth_upst, pth_down, p_len, wth_keep, elv)
    Note:
      up/down indices are 0-based; wth/elv have shape (npath_kept, keepN)
    """
    # Read header to get npth, nlev
    with open(bifori_file, "r") as f:
        head = f.readline().split()
        if len(head) < 2:
            raise ValueError("Invalid bifori header")
        npth = int(head[0])
        nlev = int(head[1])

    keepN = int(bif_levels_to_keep)
    if keepN > nlev:
        raise ValueError(f"Invalid keepN: {keepN} > {nlev}")

    # Columns per row: ix iy jx jy len elv (nlev widths) lat lon
    ncols = 4 + 2 + nlev + 2

    # Load as (npth, ncols); ignore trailing extras by usecols
    data = np.loadtxt(
        bifori_file,
        skiprows=1,
        usecols=range(ncols),
        dtype=np.float64,
        ndmin=2,
    )
    if data.shape[0] != npth:
        raise ValueError(f"Expected {npth} paths, but got {data.shape[0]} rows in bifori")

    # Unpack and shift to 0-based indices
    ij = data[:, 0:4].astype(np.int64) - 1
    ix, iy, jx, jy = ij[:, 0], ij[:, 1], ij[:, 2], ij[:, 3]
    p_len = data[:, 4].astype(np.float64)
    p_elv = data[:, 5].astype(np.float64)
    wth_all = data[:, 6 : 6 + nlev].astype(np.float64)

    # 1) Keep paths with any width>0 in the first keepN levels
    keep_mask = (wth_all[:, :keepN] > 0.0).any(axis=1)
    if not np.any(keep_mask):
        raise ValueError("No bifurcation paths with width>0 in the first keepN levels")

    ix, iy, jx, jy = ix[keep_mask], iy[keep_mask], jx[keep_mask], jy[keep_mask]
    p_len = p_len[keep_mask]
    p_elv = p_elv[keep_mask]
    wth_all = wth_all[keep_mask]

    # 2) dph from wth(1), clamped to [0.5, max(rivhgt(up), rivhgt(dn))]
    w1 = wth_all[:, 0]
    dph = np.full(w1.shape, -9999.0, dtype=np.float64)
    pos = w1 > 0.0
    if np.any(pos):
        dph_pos = np.log10(w1[pos]) * 2.5 - 4.0
        dph_pos = np.maximum(dph_pos, 0.5)
        dph0 = np.maximum(rivhgt_2d[ix[pos], iy[pos]], rivhgt_2d[jx[pos], jy[pos]])
        dph[pos] = np.minimum(dph_pos, dph0)

    # 3) Truncate widths to keepN; force wth(1)=0 when w1<=0
    wth_keep = wth_all[:, :keepN].copy()
    wth_keep[~pos, 0] = 0.0

    # 4) Build elevation table
    elv = np.full((wth_keep.shape[0], keepN), 1.0e20, dtype=np.float64)

    # Level 0: pelv - dph when width>0
    w0_pos = wth_keep[:, 0] > 0.0
    if np.any(w0_pos):
        elv[w0_pos, 0] = p_elv[w0_pos] - dph[w0_pos]

    # Levels 1..keepN-1: pelv + ilev - 1 when width>0
    for ilev in range(1, keepN):
        maskL = wth_keep[:, ilev] > 0.0
        if np.any(maskL):
            elv[maskL, ilev] = p_elv[maskL] + (ilev - 1.0)

    # Assemble upstream/downstream index pairs
    pth_upst = np.stack([ix, iy], axis=1).astype(np.int64)
    pth_down = np.stack([jx, jy], axis=1).astype(np.int64)

    return (
        pth_upst,
        pth_down,
        p_len,
        wth_keep,
        elv,
    )
