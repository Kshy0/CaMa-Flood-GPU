# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

import fnmatch
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from hydroforge.modeling.distributed import find_indices_in
from netCDF4 import Dataset
from numba import njit


@njit(cache=True)
def trace_outlets(catchment_id, downstream_id):
    """Array-based outlet tracing with path compression.

    Uses a flat lookup array instead of a typed dict for O(1) access.
    Path compression ensures each cell is traced at most once.
    """
    n = len(catchment_id)

    # Determine array size from max grid id
    max_id = np.int64(0)
    for i in range(n):
        if catchment_id[i] > max_id:
            max_id = catchment_id[i]
        if downstream_id[i] > max_id:
            max_id = downstream_id[i]
    total = max_id + 1

    # Downstream lookup array (replaces typed dict)
    #   >= 0  : downstream grid id (unresolved)
    #   -1    : river mouth (unresolved)
    #   -2    : not a valid cell
    #   <= -3 : resolved, outlet = -(value + 3)
    down = np.full(total, np.int64(-2))
    for i in range(n):
        down[catchment_id[i]] = downstream_id[i]

    result = np.empty(n, dtype=np.int64)

    for i in range(n):
        cid = catchment_id[i]
        v = down[cid]

        # Already resolved via path compression
        if v <= -3:
            result[i] = -(v + 3)
            continue

        # Trace downstream to mouth or resolved cell
        current = cid
        while down[current] >= 0:
            current = down[current]

        # current is a mouth (down==-1) or already resolved (down<=-3)
        v = down[current]
        if v <= -3:
            mouth = -(v + 3)
        else:
            mouth = current

        # Path compression: mark all cells along the path
        current = cid
        while down[current] >= 0:
            nxt = down[current]
            down[current] = -(mouth + 3)
            current = nxt
        down[current] = -(mouth + 3)

        result[i] = mouth

    return result


@njit(cache=True)
def topological_sort(catchment_id, downstream_id):
    """Array-based topological sorting."""
    n = len(catchment_id)

    # Determine array size from max grid id
    max_id = np.int64(0)
    for i in range(n):
        if catchment_id[i] > max_id:
            max_id = catchment_id[i]
    total = max_id + 1

    # Grid-id to index lookup (replaces typed dict)
    grid_to_idx = np.full(total, np.int64(-1))
    for i in range(n):
        grid_to_idx[catchment_id[i]] = np.int64(i)

    indegree = np.zeros(n, dtype=np.int32)

    for i in range(n):
        d_id = downstream_id[i]
        if d_id >= 0 and d_id < total:
            d_idx = grid_to_idx[d_id]
            if d_idx >= 0:
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
        if d_id >= 0 and d_id < total:
            d_idx = grid_to_idx[d_id]
            if d_idx >= 0:
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


@njit(cache=True)
def build_upstream_csr(catchment_id, downstream_id):
    """Build CSR upstream adjacency in index space.

    Returns (indptr, indices) where for node at index *i*,
    its upstream neighbours are indices[indptr[i]:indptr[i+1]].
    """
    n = len(catchment_id)

    # Array-based grid-id to index lookup (replaces typed dict)
    max_id = np.int64(0)
    for i in range(n):
        if catchment_id[i] > max_id:
            max_id = catchment_id[i]
    total = max_id + 1
    grid_to_idx = np.full(total, np.int64(-1))
    for i in range(n):
        grid_to_idx[catchment_id[i]] = np.int64(i)

    # Pass 1: count upstream neighbours per node
    count = np.zeros(n, dtype=np.int32)
    for i in range(n):
        did = downstream_id[i]
        if did >= 0 and did < total and did != catchment_id[i]:
            d_idx = grid_to_idx[did]
            if d_idx >= 0:
                count[d_idx] += 1

    # Pass 2: build indptr
    indptr = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        indptr[i + 1] = indptr[i] + count[i]

    # Pass 3: fill indices
    indices = np.empty(indptr[n], dtype=np.int32)
    offset = np.zeros(n, dtype=np.int32)
    for i in range(n):
        did = downstream_id[i]
        if did >= 0 and did < total and did != catchment_id[i]:
            d_idx = grid_to_idx[did]
            if d_idx >= 0:
                indices[indptr[d_idx] + offset[d_idx]] = i
                offset[d_idx] += 1

    return indptr, indices


@njit(cache=True)
def trace_upstream_bfs_csr(start_idx, indptr, indices, n, stop_mask):
    """BFS upstream from *start_idx* over a CSR adjacency.

    *stop_mask*: boolean array of length *n*.  Nodes where stop_mask is True
    are **included** in the result but their upstream neighbours are **not**
    expanded (except when they are the *start_idx* itself).

    Returns a boolean visited array of length *n*.
    """
    visited = np.zeros(n, dtype=np.bool_)
    queue = np.empty(n, dtype=np.int32)
    front = 0
    back = 0

    queue[back] = start_idx
    back += 1
    visited[start_idx] = True

    while front < back:
        curr = queue[front]
        front += 1
        # Stop nodes are included but not expanded (except start)
        if stop_mask[curr] and curr != start_idx:
            continue
        for j in range(indptr[curr], indptr[curr + 1]):
            nb = indices[j]
            if not visited[nb]:
                visited[nb] = True
                queue[back] = nb
                back += 1

    return visited


@njit(cache=True)
def _uf_find_nb(parent, x):
    """Find root with path halving (numba)."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True)
def _bif_merge_one_round(
    bif_up_bidx, bif_dn_bidx, bsize, parent,
    grid_thrs, thrs2, num_basins, reverse_paths, reverse_resolve,
):
    """One round of proposal + chain-resolution + apply.

    Faithfully replicates one iteration of *set_bif_basin_mpi.F90*:
      1. Scan paths (forward or reverse) → collect bnew proposals
      2. Resolve chains (forward or reverse) → apply with size checks
      3. Return whether any merge happened
    """
    num_paths = len(bif_up_bidx)
    bnew = np.full(num_basins, -9999, dtype=np.int64)

    # --- Phase 1: collect proposals ---
    if reverse_paths:
        for ipth in range(num_paths - 1, -1, -1):
            ib_up = _uf_find_nb(parent, bif_up_bidx[ipth])
            ib_dn = _uf_find_nb(parent, bif_dn_bidx[ipth])
            if ib_up == ib_dn:
                continue
            ibsn = min(ib_up, ib_dn)
            jbsn = max(ib_up, ib_dn)
            sa, sb = bsize[ibsn], bsize[jbsn]
            if sa + sb >= grid_thrs and sa >= thrs2 and sb >= thrs2:
                continue
            if bnew[jbsn] == -9999 or bnew[jbsn] > ibsn:
                bnew[jbsn] = ibsn
    else:
        for ipth in range(num_paths):
            ib_up = _uf_find_nb(parent, bif_up_bidx[ipth])
            ib_dn = _uf_find_nb(parent, bif_dn_bidx[ipth])
            if ib_up == ib_dn:
                continue
            ibsn = min(ib_up, ib_dn)
            jbsn = max(ib_up, ib_dn)
            sa, sb = bsize[ibsn], bsize[jbsn]
            if sa + sb >= grid_thrs and sa >= thrs2 and sb >= thrs2:
                continue
            if bnew[jbsn] == -9999 or bnew[jbsn] > ibsn:
                bnew[jbsn] = ibsn

    # --- Phase 2: resolve chains + apply ---
    changed = False
    if reverse_resolve:
        for ibsn in range(num_basins - 1, -1, -1):
            if bnew[ibsn] == -9999:
                continue
            jbsn = bnew[ibsn]
            while bnew[jbsn] >= 0:  # >= 0: basin 0 is valid (0-based)
                jbsn = bnew[jbsn]
            sa, sb = bsize[ibsn], bsize[jbsn]
            if sa + sb >= grid_thrs and sa >= thrs2 and sb >= thrs2:
                bnew[ibsn] = -9999  # clear rejected proposal
                continue
            bnew[ibsn] = jbsn
            bsize[jbsn] += bsize[ibsn]
            bsize[ibsn] = 0
            parent[ibsn] = jbsn
            changed = True
    else:
        for ibsn in range(num_basins):
            if bnew[ibsn] == -9999:
                continue
            jbsn = bnew[ibsn]
            while bnew[jbsn] >= 0:  # >= 0: basin 0 is valid (0-based)
                jbsn = bnew[jbsn]
            sa, sb = bsize[ibsn], bsize[jbsn]
            if sa + sb >= grid_thrs and sa >= thrs2 and sb >= thrs2:
                bnew[ibsn] = -9999  # clear rejected proposal
                continue
            bnew[ibsn] = jbsn
            bsize[jbsn] += bsize[ibsn]
            bsize[ibsn] = 0
            parent[ibsn] = jbsn
            changed = True

    return changed


@njit(cache=True)
def _build_river_only_paths(bif_up_bidx, bif_dn_bidx, bif_is_river):
    """Extract river-channel-only path arrays (shared pre-computation)."""
    n_river = 0
    for i in range(len(bif_is_river)):
        if bif_is_river[i]:
            n_river += 1
    river_up = np.empty(n_river, dtype=np.int64)
    river_dn = np.empty(n_river, dtype=np.int64)
    j = 0
    for i in range(len(bif_is_river)):
        if bif_is_river[i]:
            river_up[j] = bif_up_bidx[i]
            river_dn[j] = bif_dn_bidx[i]
            j += 1
    return river_up, river_dn


@njit(cache=True)
def _merge_and_count(river_up, river_dn, bif_up, bif_dn,
                     init_sizes, grid_thrs, thrs2):
    """Run two-step merge at given thresholds; return (parent, bsize, count)."""
    N = len(init_sizes)
    bsize = init_sizes.copy()
    parent = np.arange(N, dtype=np.int64)
    # STEP1: river forward scan, forward resolve
    for _ in range(200):
        if not _bif_merge_one_round(river_up, river_dn, bsize, parent,
                                     grid_thrs, thrs2, N, False, False):
            break
    # STEP2: all paths forward scan, reverse resolve
    for _ in range(200):
        if not _bif_merge_one_round(bif_up, bif_dn, bsize, parent,
                                     grid_thrs, thrs2, N, False, True):
            break
    # Flatten
    for i in range(N):
        _uf_find_nb(parent, i)
    # Count roots
    count = np.int64(0)
    for i in range(N):
        if bsize[i] > 0:
            count += 1
    return parent, bsize, count


@njit(cache=True)
def merge_basins_bifurcation_thresholds(
    bif_up_bidx,
    bif_dn_bidx,
    bif_is_river,
    initial_basin_sizes,
    rate,
):
    """Two-step basin merger with size thresholds (MPI-style).

    Replicates *set_bif_basin_mpi.F90* logic:
      STEP 1 — merge river-channel bifurcations (wth[0]>0),
               forward path scan, forward chain resolution.
      STEP 2 — merge ALL bifurcation paths (including overland),
               reverse path scan, reverse chain resolution.
    Both steps iterate until convergence.
    Size thresholds prevent creating basins larger than ``rate * total_grid``.

    Parameters
    ----------
    bif_up_bidx, bif_dn_bidx : int64 arrays, shape (num_paths,)
        Basin index (into *initial_basin_sizes*) for each bifurcation endpoint.
    bif_is_river : bool array, shape (num_paths,)
        True where the path has river-channel width > 0 (wth[0] > 0).
    initial_basin_sizes : int64 array, shape (num_basins,)
        Grid count per basin before merging.
    rate : float
        Threshold ratio.  0.06 = normal (≤16 MPI), 0.03 = MaxMPI (≤30).

    Returns
    -------
    parent : int64 array, shape (num_basins,)
        parent[i] = root basin index after merging.
    bsize : int64 array, shape (num_basins,)
        Updated basin sizes (only roots have non-zero sizes).
    """
    num_basins = len(initial_basin_sizes)

    allgrid = np.int64(0)
    for i in range(num_basins):
        allgrid += initial_basin_sizes[i]

    grid_thrs = np.int64(allgrid * rate)
    thrs2 = np.int64(allgrid * 0.001)

    river_up, river_dn = _build_river_only_paths(bif_up_bidx, bif_dn_bidx, bif_is_river)
    parent, bsize, _ = _merge_and_count(river_up, river_dn, bif_up_bidx, bif_dn_bidx,
                                         initial_basin_sizes, grid_thrs, thrs2)
    return parent, bsize


@njit(cache=True)
def search_optimal_merge_rate(bif_up_bidx, bif_dn_bidx, bif_is_river,
                              initial_basin_sizes, target_basins,
                              max_iter=25):
    """Binary-search rate to get basin count closest to target_basins.

    All computation stays inside numba; river-only path arrays are built once.

    Returns (best_rate, parent, bsize, final_count).
    """
    N = len(initial_basin_sizes)
    allgrid = np.int64(0)
    for i in range(N):
        allgrid += initial_basin_sizes[i]
    thrs2 = np.int64(allgrid * 0.001)

    river_up, river_dn = _build_river_only_paths(
        bif_up_bidx, bif_dn_bidx, bif_is_river)

    lo = 0.001
    hi = 1.0
    best_rate = lo
    best_parent = np.arange(N, dtype=np.int64)
    best_bsize = initial_basin_sizes.copy()
    best_count = np.int64(N)
    best_diff = abs(N - target_basins)

    for _ in range(max_iter):
        mid = (lo + hi) * 0.5
        grid_thrs = np.int64(allgrid * mid)

        parent, bsize, count = _merge_and_count(
            river_up, river_dn, bif_up_bidx, bif_dn_bidx,
            initial_basin_sizes, grid_thrs, thrs2)

        diff = abs(count - target_basins)
        if diff < best_diff:
            best_diff = diff
            best_rate = mid
            best_parent = parent
            best_bsize = bsize
            best_count = count

        if count == target_basins:
            break
        elif count > target_basins:
            lo = mid   # too many basins → increase rate → more merging
        else:
            hi = mid   # too few → decrease rate → less merging

    return best_rate, best_parent, best_bsize, best_count


def reorder_by_basin_size(topo_idx: np.ndarray, basin_id: np.ndarray):
    """Reorder by basin size (vectorized to avoid Python-object memory overhead)."""
    topo_basin = basin_id[topo_idx]

    # Compute basin sizes via bincount
    unique_basins, inverse = np.unique(topo_basin, return_inverse=True)
    counts = np.bincount(inverse).astype(np.int64)

    # Sort basins by descending size
    size_order = np.argsort(-counts)
    basin_sizes = counts[size_order]

    # Build a mapping: old basin label → new rank (0 = largest)
    rank = np.empty_like(size_order)
    rank[size_order] = np.arange(len(size_order))
    topo_rank = rank[inverse]

    # Stable sort topo_idx by basin rank (preserves topological order within basin)
    new_order = topo_idx[np.argsort(topo_rank, kind='stable')]

    return (new_order.astype(np.int64), basin_sizes)

def read_bifori(bifori_file: Path, rivhgt_2d: Optional[np.ndarray], bif_levels_to_keep: int):
    """
    Vectorized reader for bifori.txt replicating the core logic of Fortran set_bifparam:
      - Keep only paths where any width>0 within the first keepN levels
      - Compute dph from wth(1) with an empirical formula and clamp to [0.5, max(rivhgt(up), rivhgt(dn))]
        (rivhgt clamping is skipped when rivhgt_2d is None)
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
        if rivhgt_2d is not None:
            dph0 = np.maximum(rivhgt_2d[ix[pos], iy[pos]], rivhgt_2d[jx[pos], jy[pos]])
            dph[pos] = np.minimum(dph_pos, dph0)
        else:
            dph[pos] = dph_pos

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

def resolve_target_cids_from_poi(
    poi: Dict[str, Any],
    catchment_id: np.ndarray,
    catchment_x: np.ndarray,
    catchment_y: np.ndarray,
    gauge_info: Optional[Dict[str, Any]] = None,
    # nc_src: Optional[Dataset] = None # Optional: if looking up gauges from NC
) -> np.ndarray:
    """
    Resolve Points of Interest (POI) dict to a list of unique target catchment IDs.
    
    Logic mirrors MERITMap.filter_to_poi_basins but is functional.
    """
    target_cids: List[int] = []

    # 1) Gauges
    gauges_val = poi.get("gauges")
    if gauges_val is not None:
        if gauge_info is not None:
            if isinstance(gauges_val, str) and gauges_val.lower() == "all":
                # Add all gauges
                for info in gauge_info.values():
                    target_cids.extend(info.get("upstream_id", []))
            elif isinstance(gauges_val, (list, tuple)):
                # Specific gauges or wildcards
                for g_id_pattern in gauges_val:
                    pattern = str(g_id_pattern)
                    # Check if it contains wildcard characters
                    if '*' in pattern or '?' in pattern or '[' in pattern:
                         matched = False
                         for g_key, info in gauge_info.items():
                              if fnmatch.fnmatch(g_key, pattern):
                                   target_cids.extend(info.get("upstream_id", []))
                                   matched = True
                         if not matched:
                              raise ValueError(f"No gauges matched pattern '{pattern}'.")
                    else:
                        if pattern in gauge_info:
                            target_cids.extend(gauge_info[pattern].get("upstream_id", []))
                        else:
                            raise ValueError(f"Gauge '{pattern}' not found in loaded gauge info.")
        else:
             raise ValueError("'gauges' POI requested but gauge_info not provided or available.")

    # 2) Coordinates
    coords_cids: List[int] = []
    coords_val = poi.get("coords")
    if coords_val is not None:
        for val in coords_val:
             if len(val) == 2:
                  x, y = val
                  mask = (catchment_x == x) & (catchment_y == y)
                  cids_found = catchment_id[mask]
                  if cids_found.size > 0:
                       cid = cids_found[0]
                       coords_cids.append(cid)
                       target_cids.append(cid)
                  else:
                       raise ValueError(f"No catchment found at coords ({x}, {y}). "
                                        f"Use interactive mode to find valid coordinates.")

    # 3) Explicit Catchment IDs
    catches_val = poi.get("catchments")
    if catches_val is not None:
         target_cids.extend(catches_val)
    
    # Cross-check: If both Coords and Catchments are provided, ensure strict consistency.
    # Every coordinate MUST map to a catchment ID that is present in the provided catchments list.
    if coords_val is not None and catches_val is not None:
        catches_set = set(catches_val)
        for cid in coords_cids:
            if cid not in catches_set:
                raise ValueError(
                    f"Consistency Check Failed: Coordinate mapped to CID {cid}, "
                    f"but this CID is not in the provided 'catchments' list."
                )
    
    return np.unique(np.array(target_cids, dtype=np.int64))

def get_kept_basin_ids(
    target_cids: np.ndarray,
    catchment_id: np.ndarray,
    catchment_basin_id: np.ndarray
) -> np.ndarray:
    """
    Given target catchment IDs, find which basins they belong to.
    """
    if len(target_cids) == 0:
        return np.array([], dtype=np.int64)

    target_idx = find_indices_in(target_cids, catchment_id)
    # Filter out -1 (not found)
    valid_mask = target_idx >= 0
    if not np.any(valid_mask):
         return np.array([], dtype=np.int64)
    
    kept_basin_ids = np.unique(catchment_basin_id[target_idx[valid_mask]])
    return kept_basin_ids


def _build_upstream_adj(catchment_id, downstream_id):
    """Build upstream adjacency (CSR + mapping).

    Returns ``(indptr, indices, grid_to_idx, cid_arr)`` where *indptr* / *indices*
    are the CSR arrays in index-space, *grid_to_idx* is a flat array mapping
    CID→index (−1 for absent IDs), and *cid_arr* is the catchment_id array (int64).
    """
    cid_arr = np.asarray(catchment_id, dtype=np.int64)
    did_arr = np.asarray(downstream_id, dtype=np.int64)
    indptr, indices = build_upstream_csr(cid_arr, did_arr)
    grid_to_idx = np.full(int(cid_arr.max()) + 1, -1, dtype=np.int64)
    grid_to_idx[cid_arr] = np.arange(len(cid_arr), dtype=np.int64)
    return indptr, indices, grid_to_idx, cid_arr

def plot_basins_common(
    map_shape: Tuple[int, int],
    catchment_x: np.ndarray,
    catchment_y: np.ndarray,
    catchment_basin_id: np.ndarray,
    save_path: Optional[Path] = None,
    # Optional overlays
    gauges_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    levees_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    bifurcations: Optional[Dict[str, np.ndarray]] = None, # keys: x1, y1, x2, y2
    removed_bifurcations: Optional[Dict[str, np.ndarray]] = None,
    pois_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None, # Points of interest markers
    river_mouths_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None, # River mouths markers
    dams_xyc: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None, # (x, y, capacity_mcm)
    # Configuration
    longitude: Optional[np.ndarray] = None,
    latitude: Optional[np.ndarray] = None,
    title: str = "Basin Visualization",
    interactive: bool = False,
    basin_extra_text: Optional[Dict[int, str]] = None,
    upstream_area: Optional[np.ndarray] = None,
    catchment_id: Optional[np.ndarray] = None,
    downstream_id: Optional[np.ndarray] = None,
    color_by_upstream_area: bool = False
) -> None:
    """
    Shared plotting logic for basins.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import ListedColormap, LogNorm
    except ImportError:
        print("matplotlib not available")
        return

    nx, ny = map_shape
    
    # Coordinate system setup
    use_lonlat = (longitude is not None) and (latitude is not None)
    
    if use_lonlat and len(catchment_x) > 1:
        slope_x, intercept_x = np.polyfit(catchment_x, longitude, 1)
        slope_y, intercept_y = np.polyfit(catchment_y, latitude, 1)
        
        left = intercept_x + slope_x * (-0.5)
        right = intercept_x + slope_x * (nx - 0.5)
        bottom = intercept_y + slope_y * (ny - 0.5)
        top = intercept_y + slope_y * (-0.5)
        extent = (left, right, bottom, top)
        xlabel, ylabel = "Longitude", "Latitude"
        def idx_to_lon(x): return intercept_x + slope_x * x
        def idx_to_lat(y): return intercept_y + slope_y * y
        def invert_x(v): return (v - intercept_x) / slope_x if abs(slope_x) > 1e-9 else 0
        def invert_y(v): return (v - intercept_y) / slope_y if abs(slope_y) > 1e-9 else 0
    else:
        use_lonlat = False
        extent = None
        xlabel, ylabel = "X Index", "Y Index"
        def idx_to_lon(x): return x
        def idx_to_lat(y): return y
        def invert_x(v): return v
        def invert_y(v): return v
        slope_x = slope_y = intercept_x = intercept_y = 0

    # Basin Map
    basin_map = np.full(map_shape, fill_value=np.nan, dtype=float)
    uparea_map = None
    if len(catchment_x) > 0:
        basin_map[catchment_x, catchment_y] = catchment_basin_id
        num_basins = int(catchment_basin_id.max()) + 1
        
        if upstream_area is not None:
            uparea_map = np.full(map_shape, fill_value=np.nan, dtype=float)
            uparea_map[catchment_x, catchment_y] = upstream_area
    else:
        num_basins = 0

    # Colors
    def generate_random_colors(N, avoid_rgb_colors):
        colors = []
        avoid_rgb_colors = np.array(avoid_rgb_colors)
        rng = np.random.RandomState(42)
        while len(colors) < N:
            color = rng.rand(3)
            if len(avoid_rgb_colors) == 0 or np.all(np.linalg.norm(avoid_rgb_colors - color, axis=1) > 0.7):
                colors.append(color)
        return np.array(colors)

    special_colors: list[tuple[float, float, float]] = []
    if gauges_xy:
        special_colors.append((0.0, 1.0, 0.0))
    if bifurcations:
        special_colors.append((0.0, 0.0, 1.0))
    if levees_xy:
        special_colors.append((0.5, 0.0, 0.5))

    basin_colors = generate_random_colors(num_basins, special_colors)
    default_cmap = ListedColormap(basin_colors)
    default_cmap.set_bad(alpha=0.0)

    # Plot
    display_dpi = 150
    plt.figure(figsize=(12, 10), dpi=display_dpi)
    
    if color_by_upstream_area and uparea_map is not None:
         # Use LogNorm for better visualization of large ranges in upstream area
         valid_uparea = uparea_map[uparea_map > 0]
         vmin = np.nanmin(valid_uparea) if valid_uparea.size > 0 else 1.0
         vmax = np.nanmax(uparea_map) if np.nanmax(uparea_map) > vmin else vmin + 1.0
         
         norm = LogNorm(vmin=vmin, vmax=vmax)
         img_data = np.ma.masked_invalid(uparea_map).T
         
         plt.imshow(img_data, origin='upper', cmap='plasma', interpolation='nearest',
                   norm=norm, extent=extent)
         plt.colorbar(label='Upstream Area ($m^2$)', shrink=0.5)
    else:
        plt.imshow(np.ma.masked_invalid(basin_map).T, origin='upper', cmap=default_cmap, interpolation='nearest',
                vmin=-0.5, vmax=num_basins - 0.5, extent=extent)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(False)

    # Auto-crop logic
    margin = 1
    if len(catchment_x) > 0:
        x0_idx = max(0, int(catchment_x.min()) - margin)
        x1_idx = min(nx - 1, int(catchment_x.max()) + margin)
        y0_idx = max(0, int(catchment_y.min()) - margin)
        y1_idx = min(ny - 1, int(catchment_y.max()) + margin)
        
        if use_lonlat:
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
    else:
         def within_extent(xv, yv):
             return np.zeros_like(xv, dtype=bool)

    # Overlays
    if gauges_xy:
        gx, gy = gauges_xy
        if use_lonlat:
            gx, gy = idx_to_lon(gx), idx_to_lat(gy)
        m = within_extent(gx, gy)
        if np.any(m):
            plt.scatter(gx[m], gy[m], c='#00FF00', s=0.5, label='Gauges', zorder=5)

    if levees_xy:
        lx, ly = levees_xy
        if use_lonlat:
            lx, ly = idx_to_lon(lx), idx_to_lat(ly)
        m = within_extent(lx, ly)
        if np.any(m):
            plt.scatter(lx[m], ly[m], c='#800080', s=0.2, label='Levees', zorder=4)
            
    if pois_xy:
        px, py = pois_xy
        if use_lonlat:
            px, py = idx_to_lon(px), idx_to_lat(py)
        m = within_extent(px, py)
        if np.any(m):
            plt.scatter(px[m], py[m], c="#C10000", s=5.0, label='Points of Interest', zorder=6)

    if river_mouths_xy:
        rmx, rmy = river_mouths_xy
        if use_lonlat:
            rmx, rmy = idx_to_lon(rmx), idx_to_lat(rmy)
        m = within_extent(rmx, rmy)
        if np.any(m):
             plt.scatter(rmx[m], rmy[m], c='red', edgecolors='black', s=20.0, marker='*', linewidths=0.3, label='River Mouth', zorder=7)

    if dams_xyc is not None:
        dx, dy, dcap = dams_xyc
        if use_lonlat:
            dx, dy = idx_to_lon(dx), idx_to_lat(dy)
        m = within_extent(dx, dy)
        if np.any(m):
            cap_m = dcap[m]
            # Map capacity (MCM) to marker size via log scale
            cap_pos = np.maximum(cap_m, 1.0)
            sizes = np.clip(np.log10(cap_pos) * 3.0, 1.0, 30.0)
            plt.scatter(
                dx[m], dy[m], s=sizes, c='#FF6600', marker='^',
                edgecolors='#993300', linewidths=0.3, alpha=0.8,
                label='Dams', zorder=8,
            )

    def plot_bifs(bifs, color, linestyle, label):
        if not bifs:
            return
        x1, y1 = bifs['x1'], bifs['y1']
        x2, y2 = bifs['x2'], bifs['y2']
        if use_lonlat:
            x1, y1 = idx_to_lon(x1), idx_to_lat(y1)
            x2, y2 = idx_to_lon(x2), idx_to_lat(y2)
            limit = 180.0
        else:
            limit = nx / 2
        
        mask = (np.abs(x1 - x2) <= limit) & (within_extent(x1, y1) | within_extent(x2, y2))
        
        if np.any(mask):
            x1k, y1k = x1[mask], y1[mask]
            x2k, y2k = x2[mask], y2[mask]
            segs = np.array([[[x1k[i], y1k[i]], [x2k[i], y2k[i]]] for i in range(len(x1k))])
            lines = LineCollection(segs, colors=color, linestyles=linestyle, linewidths=0.5, alpha=0.6, zorder=3)
            plt.gca().add_collection(lines)
            plt.plot([], [], color=color, linestyle=linestyle, linewidth=0.5, alpha=0.6, label=label)

    plot_bifs(bifurcations, '#0000FF', '--', 'Bifurcation Paths')
    plot_bifs(removed_bifurcations, '#FF0000', ':', 'Removed Paths')

    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend(loc='lower right')
    plt.tight_layout()

    # Interactive
    if interactive:
        from matplotlib.widgets import TextBox
        print("Interactive mode enabled. Click / type lon,lat / arrow keys to navigate.")
        ax = plt.gca()
        fig = plt.gcf()
        ann = ax.annotate(
            "", xy=(0,0), xytext=(10,10), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5),
            fontsize=8, visible=False, zorder=30
        )

        # Cursor marker for current selection
        cursor_marker, = ax.plot([], [], 'r+', markersize=12, markeredgewidth=2, zorder=25)

        highlight_im = None
        cid_map = None
        _csr_indptr = None
        _csr_indices = None
        grid_to_idx = None
        upstream_cache = {}
        # Mutable state for current selection (xi, yi in index space)
        sel = {'xi': None, 'yi': None}

        if catchment_id is not None and downstream_id is not None:
             highlight_data = np.zeros((*map_shape, 4), dtype=float)
             highlight_im = ax.imshow(highlight_data.transpose((1, 0, 2)), origin='upper', extent=extent, zorder=20, interpolation='nearest')

             cid_map = np.full(map_shape, fill_value=-1, dtype=np.int64)
             cid_map[catchment_x, catchment_y] = catchment_id

             _adj = _build_upstream_adj(catchment_id, downstream_id)
             _csr_indptr, _csr_indices, grid_to_idx, _ = _adj

        def _select_point(xi, yi):
            """Core selection logic shared by click, text input, and arrow keys."""
            if not (0 <= xi < nx and 0 <= yi < ny):
                return

            val = basin_map[xi, yi]
            if np.isnan(val):
                ann.set_visible(False)
                cursor_marker.set_data([], [])
                if highlight_im:
                     highlight_data[:] = 0.0
                     highlight_im.set_data(highlight_data.transpose((1, 0, 2)))
                fig.canvas.draw_idle()
                return

            sel['xi'], sel['yi'] = xi, yi

            # Update cursor marker
            cx_display = idx_to_lon(xi) if use_lonlat else xi
            cy_display = idx_to_lat(yi) if use_lonlat else yi
            cursor_marker.set_data([cx_display], [cy_display])

            basin_id = int(val)
            cid_text = ""
            if cid_map is not None:
                cid_val = cid_map[xi, yi]
                if cid_val != -1:
                    cid_text = f"\nCID: {cid_val}"
            text = f"Basin: {basin_id}\nIdx: ({xi},{yi}){cid_text}"

            if basin_extra_text and basin_id in basin_extra_text:
                text += f"\n{basin_extra_text[basin_id]}"

            if use_lonlat:
                text += f"\nLon: {cx_display:.3f}, Lat: {cy_display:.3f}"

            if uparea_map is not None:
                val_uparea = uparea_map[xi, yi]
                if not np.isnan(val_uparea):
                    text += f"\nUpArea: {val_uparea/1e6:.2f} km²"

            if highlight_im and cid_map is not None:
                 clicked_cid = cid_map[xi, yi]
                 if clicked_cid != -1:
                      highlight_data[:] = 0.0
                      if 0 <= basin_id < len(basin_colors):
                          bg_color = basin_colors[basin_id]
                          L = 0.2126 * bg_color[0] + 0.7152 * bg_color[1] + 0.0722 * bg_color[2]
                          if L > 0.5:
                              contrast_color = [0.0, 0.0, 0.0, 0.8]
                          else:
                              contrast_color = [1.0, 1.0, 1.0, 0.8]
                      else:
                          contrast_color = [1.0, 0.0, 1.0, 0.7]

                      if clicked_cid in upstream_cache:
                          current_set = upstream_cache[clicked_cid]
                      else:
                          _n = len(catchment_id)
                          _stop = np.zeros(_n, dtype=np.bool_)
                          _vis = trace_upstream_bfs_csr(
                              grid_to_idx[clicked_cid], _csr_indptr, _csr_indices, _n, _stop
                          )
                          current_set = set(
                              (int(catchment_x[i]), int(catchment_y[i]))
                              for i in range(_n) if _vis[i]
                          )
                          upstream_cache[clicked_cid] = current_set

                      for cx, cy in current_set:
                           highlight_data[cx, cy] = contrast_color
                      highlight_im.set_data(highlight_data.transpose((1, 0, 2)))

            print(f"Selected: {text.replace(chr(10), ', ')}")
            ann.set_text(text)
            ann.xy = (cx_display, cy_display)
            ann.set_visible(True)
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes is not ax:
                return
            if use_lonlat:
                xi_f = invert_x(event.xdata)
                yi_f = invert_y(event.ydata)
            else:
                xi_f, yi_f = event.xdata, event.ydata
            _select_point(int(np.round(xi_f)), int(np.round(yi_f)))

        def on_key(event):
            if sel['xi'] is None or sel['yi'] is None:
                return
            xi, yi = sel['xi'], sel['yi']
            # Arrow keys move in index space (x=lon-axis, y=lat-axis)
            if event.key == 'right':
                xi += 1
            elif event.key == 'left':
                xi -= 1
            elif event.key == 'up':
                yi -= 1
            elif event.key == 'down':
                yi += 1
            else:
                return
            _select_point(xi, yi)

        def on_submit(text):
            text = text.strip()
            if not text:
                return
            parts = text.replace(' ', ',').split(',')
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) != 2:
                print("Input format: lon,lat  or  x,y")
                return
            try:
                v1, v2 = float(parts[0]), float(parts[1])
            except ValueError:
                print("Invalid number format.")
                return
            if use_lonlat:
                xi = int(np.round(invert_x(v1)))
                yi = int(np.round(invert_y(v2)))
            else:
                xi, yi = int(np.round(v1)), int(np.round(v2))
            _select_point(xi, yi)

        # Lon/Lat input box — lower-right corner
        plt.tight_layout()
        ax_pos = ax.get_position()
        box_width = 0.18
        box_height = 0.03
        box_left = ax_pos.x1 - box_width
        box_bottom = ax_pos.y0 - box_height - 0.03
        ax_box = fig.add_axes((box_left, box_bottom, box_width, box_height))
        label = "Lon,Lat:" if use_lonlat else "X,Y:"
        text_box = TextBox(ax_box, label, initial="", textalignment="center")
        text_box.on_submit(on_submit)

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()
        if save_path:
             print(f"Warning: Interactive mode is on. Image will not be saved to {save_path} automatically. Use the UI to save.")
        return

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
        return
    plt.close()

def visualize_nc_basins(
    nc_path: Union[str, Path],
    save_path: Optional[Union[str, Path]] = None,
    visualize_gauges: bool = True,
    visualize_bifurcations: bool = True,
    visualize_levees: bool = True,
    visualize_dams: bool = True,
    visualize_river_mouths: bool = False,
    interactive: bool = False,
    pois_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    color_by_upstream_area: bool = False
) -> None:
    """
    Visualize basins from a generated NetCDF parameter file using shared plotting logic.
    """
    with Dataset(nc_path, 'r') as ds:
        # Load necessary arrays
        catchment_x = ds['catchment_x'][:]
        catchment_y = ds['catchment_y'][:]
        catchment_basin_id = ds['catchment_basin_id'][:]
        
        # Determine map shape
        if hasattr(ds, 'nx') and hasattr(ds, 'ny'):
            nx, ny = ds.nx, ds.ny
        else:
            nx = int(catchment_x.max() + 1)
            ny = int(catchment_y.max() + 1)
        map_shape = (nx, ny)

        # Optional LAT/LON
        longitude = None
        latitude = None
        if 'longitude' in ds.variables and 'latitude' in ds.variables:
            longitude = ds['longitude'][:]
            latitude = ds['latitude'][:]
        
        # Gather Optional Data
        gauges_xy = None
        if visualize_gauges and 'gauge_catchment_id' in ds.variables and 'catchment_id' in ds.variables:
            g_cids = ds['gauge_catchment_id'][:]
            c_ids = ds['catchment_id'][:]
            idx = find_indices_in(g_cids, c_ids)
            idx = idx[idx >= 0]
            if idx.size > 0:
                gauges_xy = (catchment_x[idx], catchment_y[idx])

        bifurcations = None
        if visualize_bifurcations and 'bifurcation_catchment_x' in ds.variables:
            bifurcations = {
                'x1': ds['bifurcation_catchment_x'][:],
                'y1': ds['bifurcation_catchment_y'][:],
                'x2': ds['bifurcation_downstream_x'][:],
                'y2': ds['bifurcation_downstream_y'][:]
            }
        
        levees_xy = None
        if visualize_levees and 'levee_catchment_x' in ds.variables:
            levees_xy = (ds['levee_catchment_x'][:], ds['levee_catchment_y'][:])

        upstream_area = None
        if 'upstream_area' in ds.variables:
            upstream_area = ds['upstream_area'][:]

        dams_xyc = None
        if visualize_dams and 'reservoir_catchment_id' in ds.variables and 'catchment_id' in ds.variables:
            res_cids = np.asarray(ds['reservoir_catchment_id'][:]).astype(np.int64)
            all_cids = np.asarray(ds['catchment_id'][:]).astype(np.int64)
            res_idx = find_indices_in(res_cids, all_cids)
            valid = res_idx >= 0
            if np.any(valid):
                rx = catchment_x[res_idx[valid]]
                ry = catchment_y[res_idx[valid]]
                if 'reservoir_capacity' in ds.variables:
                    cap_m3 = np.asarray(ds['reservoir_capacity'][:])[valid]
                    cap_mcm = cap_m3 / 1.0e6
                else:
                    cap_mcm = np.ones(int(valid.sum()), dtype=np.float64)
                dams_xyc = (rx, ry, cap_mcm)

        river_mouths_xy = None
        if visualize_river_mouths:
            # User requested identification by catchment_id == downstream_id
            if 'catchment_id' in ds.variables and 'downstream_id' in ds.variables:
                cid = ds['catchment_id'][:]
                did = ds['downstream_id'][:]
                # Mouth condition: downstream is self or invalid (-1)
                mask = (cid == did) | (did < 0)
                if np.any(mask):
                    river_mouths_xy = (catchment_x[mask], catchment_y[mask])
            elif 'is_river_mouth' in ds.variables:
                is_mouth = ds['is_river_mouth'][:]
                if np.any(is_mouth):
                     # is_river_mouth is a boolean mask or 1/0
                     mask = is_mouth == 1
                     river_mouths_xy = (catchment_x[mask], catchment_y[mask])

        catchment_id = None
        downstream_id = None
        if interactive:
            if 'catchment_id' in ds.variables:
                catchment_id = ds['catchment_id'][:]
            if 'downstream_id' in ds.variables:
                downstream_id = ds['downstream_id'][:]

        plot_basins_common(
            map_shape=map_shape,
            catchment_x=catchment_x,
            catchment_y=catchment_y,
            catchment_basin_id=catchment_basin_id,
            save_path=Path(save_path) if save_path else None,
            gauges_xy=gauges_xy,
            levees_xy=levees_xy,
            bifurcations=bifurcations,
            longitude=longitude,
            latitude=latitude,
            title=f"Basins (from {Path(nc_path).name})",
            interactive=interactive,
            pois_xy=pois_xy,
            river_mouths_xy=river_mouths_xy,
            dams_xyc=dams_xyc,
            upstream_area=upstream_area,
            catchment_id=catchment_id,
            downstream_id=downstream_id,
            color_by_upstream_area=color_by_upstream_area
        )


def _trace_upstream_bfs(start_cid, upstream_adj, stop_at=None):
    """BFS upstream from *start_cid*.  Returns a **set of CIDs**.

    *upstream_adj* must be the tuple returned by ``_build_upstream_adj``.
    If *stop_at* is a set of CIDs, those nodes are included but not expanded.
    """
    indptr, indices, id_to_idx, cid_arr = upstream_adj
    n = len(cid_arr)
    start_idx = id_to_idx[int(start_cid)]
    stop_mask = np.zeros(n, dtype=np.bool_)
    if stop_at is not None:
        id_to_idx_len = len(id_to_idx)
        for s in stop_at:
            si = int(s)
            if 0 <= si < id_to_idx_len and id_to_idx[si] >= 0:
                stop_mask[id_to_idx[si]] = True
    visited = trace_upstream_bfs_csr(start_idx, indptr, indices, n, stop_mask)
    return set(cid_arr[visited].tolist())


def _check_poi_overlap(sorted_pois, poi_upstream_list):
    """Raise ValueError if any POI is in the upstream set of another.
    
    Parameters
    ----------
    sorted_pois : array-like of int
        Sorted POI catchment IDs.
    poi_upstream_list : list of set
        ``poi_upstream_list[i]`` is the upstream set for ``sorted_pois[i]``.
    """
    overlapping = []
    for ia, cid_a in enumerate(sorted_pois):
        for ib, cid_b in enumerate(sorted_pois):
            if ia != ib and int(cid_a) in poi_upstream_list[ib]:
                overlapping.append((int(cid_a), int(cid_b)))
    if overlapping:
        pairs_str = "; ".join(f"POI {a} is upstream of POI {b}" for a, b in overlapping)
        raise ValueError(
            f"Overlapping POIs detected (one is upstream of another): {pairs_str}. "
            f"Please remove redundant POIs so that no POI is in the upstream set of another."
        )


def crop_parameters_nc(
    input_nc: Union[str, Path],
    output_nc: Union[str, Path],
    points_of_interest: Optional[Dict[str, Any]] = None,
    visualize: bool = False,
    only_save_pois: bool = False,
    crop_upstream: bool = False,
    crop_downstream: bool = False,
    crop_interval: bool = False,
    # Visualization options (forwarded to visualize_nc_basins)
    visualize_gauges: bool = True,
    visualize_bifurcations: bool = True,
    visualize_levees: bool = True,
    visualize_dams: bool = True,
    visualize_river_mouths: bool = False,
    color_by_upstream_area: bool = False,
    interactive: bool = False,
) -> None:
    """
    Crops an existing parameter NetCDF to a subset of basins covering specific points of interest.
    
    If crop_upstream=True, only the upstream catchments of each POI are kept, and the POI
    catchments are turned into river mouths (downstream_id = self). Upstream tracing follows
    main-stem downstream_id links only; bifurcation paths whose both endpoints survive the
    crop are preserved in the output.
    
    If crop_downstream=True, the upstream catchments of each POI are removed (excluding the
    POI itself). POIs become headwaters with no upstream inflow; their downstream_id is
    unchanged so they route normally downstream. This is useful for removing upstream
    tributaries and injecting prescribed inflow at the POI locations.
    
    If crop_interval=True, each POI acts as a gauge defining an interval sub-basin. The
    upstream BFS from each POI stops when it encounters another POI (included but not
    traversed further). Each POI's downstream_id is set to self (river mouth), creating
    independent interval basins. Catchments not reachable from any POI are removed.
    POIs on the same flow path are allowed (this is the primary use case).
    
    crop_upstream, crop_downstream, and crop_interval are mutually exclusive.
    For crop_upstream and crop_downstream, overlapping POIs are not allowed.
    """
    input_nc = Path(input_nc)
    output_nc = Path(output_nc)
    if sum([crop_upstream, crop_downstream, crop_interval]) > 1:
        raise ValueError("Only one of crop_upstream, crop_downstream, crop_interval can be True.")
    if interactive:
        visualize = True

    with Dataset(input_nc, 'r') as src:
        # Load connectivity
        catchment_id = src['catchment_id'][:]
        catchment_x = src['catchment_x'][:]
        catchment_y = src['catchment_y'][:]
        catchment_basin_id = src['catchment_basin_id'][:]
        downstream_id = src['downstream_id'][:] if 'downstream_id' in src.variables else None

        # Resolve target CIDs using shared logic
        if points_of_interest:
            target_cids = resolve_target_cids_from_poi(
                points_of_interest, 
                catchment_id, 
                catchment_x, 
                catchment_y,
                gauge_info=None 
            )
            
            if len(target_cids) == 0:
                raise ValueError("No valid target catchments found from points_of_interest.")

            # Find basins containing these catchments
            kept_basin_ids = get_kept_basin_ids(target_cids, catchment_id, catchment_basin_id)
            
            if len(kept_basin_ids) == 0:
                 raise ValueError("Target catchments not found in map — basin lookup failed.")
        else:
            print("No points_of_interest provided or empty. Keeping all basins.")
            target_cids = np.array([], dtype=np.int64)
            kept_basin_ids = np.unique(catchment_basin_id)
            if only_save_pois:
                 print("Warning: only_save_pois=True but no POIs provided. Switching to saving all catchments.")
                 only_save_pois = False

        # ── crop_upstream mode: keep only upstream of POIs, turn POIs into outlets ──
        outlet_cids = np.array([], dtype=np.int64)  # CIDs that will become river mouths
        if crop_upstream and len(target_cids) > 0:
            if downstream_id is None:
                print("Error: crop_upstream requires 'downstream_id' in the input NC. Aborting.")
                return

            upstream_adj = _build_upstream_adj(catchment_id, downstream_id)
            grid_to_idx = upstream_adj[2]

            effective_outlets = sorted(int(c) for c in target_cids)
            poi_upstream_list = [_trace_upstream_bfs(int(cid), upstream_adj) for cid in effective_outlets]

            _check_poi_overlap(effective_outlets, poi_upstream_list)

            print(f"crop_upstream: Effective outlet POIs: {effective_outlets}")

            # Collect all upstream catchments of effective outlets
            kept_cid_set = set()
            for us_set in poi_upstream_list:
                kept_cid_set.update(us_set)

            # Override keep_mask: only keep these catchments
            keep_mask_upstream = np.isin(catchment_id, np.array(sorted(kept_cid_set), dtype=np.int64))
            
            # Recompute basin info for the kept subset
            # Reassign basins: each outlet defines its own basin
            outlet_cids = np.array(effective_outlets, dtype=np.int64)
            
            # Assign each kept catchment to the outlet it flows to
            new_basin_assignment = np.full(len(catchment_id), -1, dtype=np.int64)
            for basin_idx, us_set in enumerate(poi_upstream_list):
                for upstream_cid in us_set:
                    if grid_to_idx[upstream_cid] >= 0:
                        new_basin_assignment[grid_to_idx[upstream_cid]] = basin_idx

            # For catchments that belong to multiple outlet upstream sets (shouldn't happen
            # after dedup, but just in case from bifurcation), assign to the first.
            
            kept_basin_ids_upstream = np.arange(len(effective_outlets), dtype=np.int64)
            catchment_basin_id = new_basin_assignment  # Override basin assignment globally
            keep_mask = keep_mask_upstream
            kept_basin_ids = kept_basin_ids_upstream
            num_kept_catchments = int(np.sum(keep_mask))

            # Mark effective outlets for downstream_id modification later
            target_cids = outlet_cids
            
            print(f"crop_upstream: Keeping {num_kept_catchments} upstream catchments across {len(effective_outlets)} sub-basins")

        # ── crop_downstream mode: remove upstream of POIs, keep POIs as headwaters ──
        removed_cid_set_dn = set()
        if crop_downstream and len(target_cids) > 0:
            if downstream_id is None:
                print("Error: crop_downstream requires 'downstream_id' in the input NC. Aborting.")
                return

            upstream_adj = _build_upstream_adj(catchment_id, downstream_id)

            sorted_pois_dn = sorted(int(c) for c in target_cids)
            poi_upstream_list_dn = [_trace_upstream_bfs(int(cid), upstream_adj) for cid in sorted_pois_dn]

            _check_poi_overlap(sorted_pois_dn, poi_upstream_list_dn)

            # Remove upstream of each POI (excluding the POI itself — it becomes a headwater)
            for i, cid in enumerate(sorted_pois_dn):
                removed_cid_set_dn.update(poi_upstream_list_dn[i] - {cid})

            # Recompute kept_basin_ids from remaining catchments
            if removed_cid_set_dn:
                remaining_mask = ~np.isin(catchment_id, np.array(sorted(removed_cid_set_dn), dtype=np.int64))
            else:
                remaining_mask = np.ones(len(catchment_id), dtype=bool)
            kept_basin_ids = np.unique(catchment_basin_id[remaining_mask])

            print(f"crop_downstream: Removing {len(removed_cid_set_dn)} upstream catchments above {len(target_cids)} POIs")

        # ── crop_interval mode: split river network into interval sub-basins at POIs ──
        if crop_interval and len(target_cids) > 0:
            if downstream_id is None:
                print("Error: crop_interval requires 'downstream_id' in the input NC. Aborting.")
                return

            upstream_adj = _build_upstream_adj(catchment_id, downstream_id)
            indptr, indices, grid_to_idx, cid_arr = upstream_adj
            n = len(cid_arr)
            poi_set = set(int(c) for c in target_cids)

            # Pre-build stop_mask ONCE (shared across all BFS calls)
            stop_mask = np.zeros(n, dtype=np.bool_)
            id_to_idx_len = len(grid_to_idx)
            for s in poi_set:
                si = int(s)
                if 0 <= si < id_to_idx_len and grid_to_idx[si] >= 0:
                    stop_mask[grid_to_idx[si]] = True

            # Interval BFS: trace upstream from each POI, stop at other POIs
            sorted_pois = sorted(poi_set)
            sorted_pois_arr = np.array(sorted_pois, dtype=np.int64)

            # Use vectorised assignment: run BFS per POI, assign directly via numpy
            cid_to_poi_basin = np.full(int(catchment_id.max()) + 1, -1, dtype=np.int64)
            # First pass: non-POI catchments assigned to their downstream POI
            for pi, poi_cid in enumerate(sorted_pois):
                start_idx = grid_to_idx[int(poi_cid)]
                visited = trace_upstream_bfs_csr(start_idx, indptr, indices, n, stop_mask)
                member_cids = cid_arr[visited]
                # Assign non-POI members (POIs overwritten in second pass)
                cid_to_poi_basin[member_cids] = pi
            # Second pass: each POI belongs to its own interval
            for pi, poi_cid in enumerate(sorted_pois):
                cid_to_poi_basin[poi_cid] = pi

            # Kept catchments = all assigned catchments
            kept_cid_mask = cid_to_poi_basin[catchment_id] >= 0

            # Assign basin IDs (vectorised)
            new_basin_assignment = cid_to_poi_basin[catchment_id].copy()

            # Set up variables like crop_upstream
            outlet_cids = sorted_pois_arr
            keep_mask = kept_cid_mask
            catchment_basin_id = new_basin_assignment
            kept_basin_ids = np.arange(len(sorted_pois), dtype=np.int64)
            num_kept_catchments = int(np.sum(keep_mask))
            target_cids = outlet_cids

            print(f"crop_interval: {len(sorted_pois)} interval basins, {num_kept_catchments} catchments kept")

        if crop_upstream or crop_interval:
            # In crop_upstream mode, basins are already reassigned above.
            # No bifurcation expansion or union-find needed — each outlet defines its own basin.
            num_merged_basins = len(kept_basin_ids)
            old_unique_basins = np.sort(kept_basin_ids)
            roots = list(kept_basin_ids)
            # Identity mapping: old_to_new_id[b] = b for all kept basins
            max_basin_id = int(kept_basin_ids.max()) if len(kept_basin_ids) > 0 else 0
            old_to_new_id = np.full(max_basin_id + 1, -1, dtype=np.int64)
            old_to_new_id[kept_basin_ids] = kept_basin_ids.astype(np.int64)
            # keep_mask and num_kept_catchments were already set above
            num_kept_catchments = int(np.sum(keep_mask))
            print(f"Cropping from {len(catchment_id)} to {num_kept_catchments} catchments (Merged Basins: {num_merged_basins})")
        else:
            if 'bifurcation_catchment_id' in src.variables and 'bifurcation_downstream_id' in src.variables:
                bif_up_cid = src['bifurcation_catchment_id'][:]
                bif_dn_cid = src['bifurcation_downstream_id'][:]
                
                # Flat array: catchment_id → basin_id
                grid_to_basin = np.full(int(catchment_id.max()) + 1, -1, dtype=np.int64)
                grid_to_basin[catchment_id] = catchment_basin_id
                
                basin_adj = defaultdict(set)
                for u_cid, d_cid in zip(bif_up_cid, bif_dn_cid):
                    u_basin = grid_to_basin[int(u_cid)]
                    d_basin = grid_to_basin[int(d_cid)]
                    
                    if u_basin >= 0 and d_basin >= 0 and u_basin != d_basin:
                        basin_adj[u_basin].add(d_basin)
                        basin_adj[d_basin].add(u_basin)
                
                queue = list(kept_basin_ids)
                visited = set(kept_basin_ids)
                
                while queue:
                    curr = queue.pop(0)
                    for neighbor in basin_adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                kept_basin_ids = np.array(sorted(list(visited)), dtype=kept_basin_ids.dtype)

            # Union-Find with flat array
            max_basin_id = int(kept_basin_ids.max()) if len(kept_basin_ids) > 0 else 0
            parent_arr = np.full(max_basin_id + 1, -1, dtype=np.int64)
            for b in kept_basin_ids:
                parent_arr[b] = b
            def find_set(x):
                while parent_arr[x] != x:
                    parent_arr[x] = parent_arr[parent_arr[x]]  # path compression
                    x = parent_arr[x]
                return x
            def union_sets(x, y):
                rootX, rootY = find_set(x), find_set(y)
                if rootX != rootY:
                    parent_arr[rootX] = rootY

            if 'bifurcation_catchment_id' in src.variables and 'bifurcation_downstream_id' in src.variables:
                 for b in kept_basin_ids:
                     for neighbor in basin_adj[b]:
                         if parent_arr[neighbor] >= 0:
                             union_sets(b, neighbor)
            
            roots = sorted(list(set(find_set(b) for b in kept_basin_ids)))
            
            # Flat arrays for root→new_id and old→new_id
            root_to_new = np.full(max_basin_id + 1, -1, dtype=np.int64)
            for i, r in enumerate(roots):
                root_to_new[r] = i
            
            old_to_new_id = np.full(max_basin_id + 1, -1, dtype=np.int64)
            for b in kept_basin_ids:
                old_to_new_id[b] = root_to_new[find_set(b)]
            
            num_merged_basins = len(roots)
            
            keep_mask = np.isin(catchment_basin_id, kept_basin_ids)
            # In crop_downstream, additionally exclude the removed upstream catchments
            if crop_downstream and removed_cid_set_dn:
                keep_mask = keep_mask & ~np.isin(catchment_id, np.array(sorted(removed_cid_set_dn), dtype=np.int64))
            num_kept_catchments = int(np.sum(keep_mask))
            print(f"Cropping from {len(catchment_id)} to {num_kept_catchments} catchments (Merged Basins: {num_merged_basins})")
        
        # Prepare catchment_save_id and catchment_save_basin_id based on only_save_pois logic
        # We need to compute it for the KEPT catchments only.
        kept_catchment_ids = catchment_id[keep_mask]
        kept_catchment_basin_ids = catchment_basin_id[keep_mask]
        
        if only_save_pois:
            # Filter target_cids to those in kept catchments, preserving order
            save_mask = np.isin(target_cids, kept_catchment_ids)
            new_save_ids = target_cids[save_mask]
            # Compute basin IDs for saved catchments
            ti = find_indices_in(new_save_ids, kept_catchment_ids)
            new_save_basin_ids = kept_catchment_basin_ids[ti]
        else:
            # Save all catchments
            new_save_ids = kept_catchment_ids.copy()
            new_save_basin_ids = kept_catchment_basin_ids.copy()

        with Dataset(output_nc, 'w') as dst:
            dst.setncatts(src.__dict__)
            
            old_unique_basins = np.sort(kept_basin_ids)
            map_idx_to_new = old_to_new_id[old_unique_basins]

            # Precompute new basin sizes
            kept_catchment_basin_ids = catchment_basin_id[keep_mask]
            idx_in_kept = np.searchsorted(old_unique_basins, kept_catchment_basin_ids)
            mapped_basin_ids = map_idx_to_new[idx_in_kept]
            new_basin_sizes = np.bincount(mapped_basin_ids, minlength=num_merged_basins).astype(np.int64)

            for name, dim in src.dimensions.items():
                if name == 'catchment':
                    dst.createDimension(name, num_kept_catchments)
                elif name == 'basin':
                    dst.createDimension(name, num_merged_basins)
                elif name == 'bifurcation_path':
                    pass
                elif name == 'levee':
                    pass
                elif name == 'gauge':
                    pass
                elif name == 'saved_catchment':
                    pass  # Will be created below
                else:
                    dst.createDimension(name, len(dim) if not dim.isunlimited() else None)
            
            # Create saved_catchment dimension (always present now)
            dst.createDimension('saved_catchment', len(new_save_ids))
            
            bif_mask = None
            if 'bifurcation_basin_id' in src.variables:
                 if crop_upstream or crop_downstream or crop_interval:
                     # Filter by both endpoints being in the kept catchment set
                     bif_up = src['bifurcation_catchment_id'][:]
                     bif_dn = src['bifurcation_downstream_id'][:]
                     # Flat array for kept-catchment lookup
                     is_kept_cid = np.zeros(int(catchment_id.max()) + 1, dtype=np.bool_)
                     is_kept_cid[kept_catchment_ids] = True
                     bif_up_i = bif_up.astype(np.int64)
                     bif_dn_i = bif_dn.astype(np.int64)
                     if crop_interval:
                         # For interval mode, both endpoints must also be in the SAME basin
                         grid_to_new_basin = np.full(int(catchment_id.max()) + 1, -1, dtype=np.int64)
                         grid_to_new_basin[catchment_id[keep_mask]] = catchment_basin_id[keep_mask]
                         bif_mask = (
                             is_kept_cid[bif_up_i] & is_kept_cid[bif_dn_i]
                             & (grid_to_new_basin[bif_up_i] == grid_to_new_basin[bif_dn_i])
                         )
                     else:
                         bif_mask = is_kept_cid[bif_up_i] & is_kept_cid[bif_dn_i]
                 else:
                     bif_basin_id = src['bifurcation_basin_id'][:]
                     bif_mask = np.isin(bif_basin_id, kept_basin_ids)
                 if 'bifurcation_path' not in dst.dimensions:
                      dst.createDimension('bifurcation_path', int(np.sum(bif_mask)))
            
            lev_mask = None
            if 'levee_basin_id' in src.variables:
                 if crop_upstream or crop_downstream or crop_interval:
                     lev_cids = src['levee_catchment_id'][:] if 'levee_catchment_id' in src.variables else None
                     if lev_cids is not None:
                         lev_mask = np.isin(lev_cids, kept_catchment_ids)
                     else:
                         lev_basin_id = src['levee_basin_id'][:]
                         lev_mask = np.zeros(len(lev_basin_id), dtype=bool)
                 else:
                     lev_basin_id = src['levee_basin_id'][:]
                     lev_mask = np.isin(lev_basin_id, kept_basin_ids)
                 if 'levee' not in dst.dimensions:
                      dst.createDimension('levee', int(np.sum(lev_mask)))

            gauge_mask = None
            if 'gauge_catchment_id' in src.variables:
                gauge_cids = src['gauge_catchment_id'][:]
                gauge_mask = np.isin(gauge_cids, kept_catchment_ids)
                if 'gauge' not in dst.dimensions:
                    dst.createDimension('gauge', int(np.sum(gauge_mask)))

            for name, var in src.variables.items():
                dims = var.dimensions
                data = var[:] 
                primary_dim = dims[0] if dims else None
                
                if name == 'num_basins':
                     dst.createVariable(name, var.dtype, dims, zlib=True)
                     dst[name][:] = np.array(num_merged_basins, dtype=var.dtype)
                     continue

                # Skip old catchment_save_mask if present
                if name == 'catchment_save_mask':
                    continue
                
                # Skip old catchment_save_id/catchment_save_basin_id - we'll write our own
                if name in ('catchment_save_id', 'catchment_save_basin_id'):
                    continue

                if primary_dim == 'catchment':
                     new_data = data[keep_mask]
                     if crop_upstream or crop_interval:
                         if name == 'catchment_basin_id':
                             # Already reassigned in new_basin_assignment
                             new_data = catchment_basin_id[keep_mask].astype(new_data.dtype)
                         elif name == 'downstream_id':
                             # Set outlet POIs' downstream_id to self (river mouth)
                             # Also set any catchment whose downstream is not in kept set to self
                             is_kept_arr = np.zeros(int(catchment_id.max()) + 1, dtype=np.bool_)
                             is_kept_arr[kept_catchment_ids] = True
                             is_outlet = np.zeros(int(catchment_id.max()) + 1, dtype=np.bool_)
                             if len(outlet_cids) > 0:
                                 is_outlet[outlet_cids] = True
                             new_cids = catchment_id[keep_mask]
                             # Vectorized: set to self where outlet or downstream not kept
                             need_self = is_outlet[new_cids] | ~is_kept_arr[new_data.astype(np.int64)]
                             new_data[need_self] = new_cids[need_self]
                     else:
                         if name == 'catchment_basin_id':
                             idx_in_kept = np.searchsorted(old_unique_basins, new_data)
                             new_data = map_idx_to_new[idx_in_kept].astype(new_data.dtype)
                     dst.createVariable(name, var.dtype, dims, zlib=True)
                     dst[name][:] = new_data
                     
                elif primary_dim == 'basin':
                     if name == 'basin_sizes':
                          new_data = new_basin_sizes.astype(data.dtype)
                     elif crop_upstream or crop_interval:
                          # In crop_upstream/interval, basins are newly defined; skip old basin-dim vars
                          # that don't have a meaningful mapping (e.g. basin_start_offsets).
                          continue
                     else:
                          new_data = data[roots]
                          
                     dst.createVariable(name, var.dtype, dims, zlib=True)
                     dst[name][:] = new_data

                elif primary_dim == 'bifurcation_path' and bif_mask is not None:
                     new_data = data[bif_mask]
                     if name == 'bifurcation_basin_id':
                          if crop_upstream or crop_interval:
                              # Remap using grid_to_new_basin lookup from upstream catchment
                              bif_up_filtered = src['bifurcation_catchment_id'][:][bif_mask]
                              if 'grid_to_new_basin' not in dir():
                                  grid_to_new_basin = np.full(int(catchment_id.max()) + 1, -1, dtype=np.int64)
                                  grid_to_new_basin[catchment_id[keep_mask]] = catchment_basin_id[keep_mask]
                              bif_basin_mapped = grid_to_new_basin[bif_up_filtered.astype(np.int64)]
                              bif_basin_mapped[bif_basin_mapped < 0] = 0
                              new_data = bif_basin_mapped.astype(new_data.dtype)
                          else:
                              idx_in_kept = np.searchsorted(old_unique_basins, new_data)
                              new_data = map_idx_to_new[idx_in_kept].astype(new_data.dtype)
                     dst.createVariable(name, var.dtype, dims, zlib=True)
                     dst[name][:] = new_data
                     
                elif primary_dim == 'levee' and lev_mask is not None:
                     new_data = data[lev_mask]
                     if name == 'levee_basin_id':
                          if crop_upstream or crop_interval:
                              lev_cids_filtered = src['levee_catchment_id'][:][lev_mask] if 'levee_catchment_id' in src.variables else None
                              if lev_cids_filtered is not None:
                                  if 'grid_to_new_basin' not in dir():
                                      grid_to_new_basin = np.full(int(catchment_id.max()) + 1, -1, dtype=np.int64)
                                      grid_to_new_basin[catchment_id[keep_mask]] = catchment_basin_id[keep_mask]
                                  lev_basin_mapped = grid_to_new_basin[lev_cids_filtered.astype(np.int64)]
                                  lev_basin_mapped[lev_basin_mapped < 0] = 0
                                  new_data = lev_basin_mapped.astype(new_data.dtype)
                              else:
                                  new_data = np.zeros(np.sum(lev_mask), dtype=new_data.dtype)
                          else:
                              idx_in_kept = np.searchsorted(old_unique_basins, new_data)
                              new_data = map_idx_to_new[idx_in_kept].astype(new_data.dtype)
                     dst.createVariable(name, var.dtype, dims, zlib=True)
                     dst[name][:] = new_data
                
                elif primary_dim == 'gauge' and gauge_mask is not None:
                    new_data = data[gauge_mask]
                    dst.createVariable(name, var.dtype, dims, zlib=True)
                    dst[name][:] = new_data

                else:
                     # Skip variables whose dimensions were not created in dst
                     if any(d not in dst.dimensions for d in dims):
                         continue
                     dst.createVariable(name, var.dtype, dims, zlib=True)
                     dst[name][:] = data
            
            # Write catchment_save_id and catchment_save_basin_id (always present)
            # Remap basin IDs for saved catchments
            idx_in_kept = np.searchsorted(old_unique_basins, new_save_basin_ids)
            remapped_save_basin_ids = map_idx_to_new[idx_in_kept].astype(np.int64)
            
            var = dst.createVariable('catchment_save_id', np.int64, ('saved_catchment',), zlib=True)
            var[:] = new_save_ids
            
            var = dst.createVariable('catchment_save_basin_id', np.int64, ('saved_catchment',), zlib=True)
            var[:] = remapped_save_basin_ids
            
            # Update global attr
            dst.setncattr("num_basins", int(num_merged_basins))

    if visualize:
         pois_xy = None
         # Calculate POI coords using kept_catchment_ids and catchment_x/y (which are still loaded in memory from src)
         # Note: catchment_x/y are the full arrays.
         if len(target_cids) > 0:
              poi_idx = find_indices_in(target_cids, catchment_id)
              poi_idx = poi_idx[poi_idx >= 0]
              if poi_idx.size > 0:
                   pois_xy = (catchment_x[poi_idx], catchment_y[poi_idx])

         img_path = output_nc.parent / (output_nc.stem + ".png")
         visualize_nc_basins(
             output_nc,
             save_path=img_path,
             pois_xy=pois_xy,
             visualize_gauges=visualize_gauges,
             visualize_bifurcations=visualize_bifurcations,
             visualize_levees=visualize_levees,
             visualize_dams=visualize_dams,
             visualize_river_mouths=visualize_river_mouths,
             color_by_upstream_area=color_by_upstream_area,
             interactive=interactive,
         )


def visualize_runoff_mapping(
    npz_path: Union[str, Path],
    parameter_nc: Union[str, Path],
    save_path: Optional[Union[str, Path]] = None,
    interactive: bool = False,
) -> None:
    """
    Visualize a runoff mapping table (npz).

    Panel 1: mapped area per source grid cell (km²).
    Panel 2: coverage ratio per catchment (mapped_area / catchment_area).
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, Normalize
    except ImportError:
        print("matplotlib not available")
        return

    from scipy.sparse import csr_matrix

    npz_path = Path(npz_path)
    parameter_nc = Path(parameter_nc)

    # --- Load mapping ---
    d = np.load(npz_path)
    mapping_cids = d["catchment_ids"]
    shape = tuple(d["matrix_shape"])
    coord_lon = d["coord_lon"]
    coord_lat = d["coord_lat"]
    nlon = len(coord_lon)
    nlat = len(coord_lat)

    mat = csr_matrix(
        (d["sparse_data"], d["sparse_indices"], d["sparse_indptr"]),
        shape=shape,
    )

    # --- Panel 1: Source grid mapped area ---
    col_sum = np.array(mat.sum(axis=0)).ravel()  # m²
    nz_cols = np.where(col_sum > 0)[0]
    ix_nz = nz_cols % nlon
    iy_nz = nz_cols // nlon

    # Determine zoom region with a small margin
    margin = 2
    ix_lo, ix_hi = max(ix_nz.min() - margin, 0), min(ix_nz.max() + margin + 1, nlon)
    iy_lo, iy_hi = max(iy_nz.min() - margin, 0), min(iy_nz.max() + margin + 1, nlat)

    src_map = np.full((nlat, nlon), np.nan)
    src_map[iy_nz, ix_nz] = col_sum[nz_cols] / 1e6  # m² → km²
    src_crop = src_map[iy_lo:iy_hi, ix_lo:ix_hi]

    # Coordinate edges for pcolormesh
    dlon = abs(coord_lon[1] - coord_lon[0]) if nlon > 1 else 1.0
    dlat = abs(coord_lat[1] - coord_lat[0]) if nlat > 1 else 1.0
    lon_edges = np.append(coord_lon[ix_lo:ix_hi] - 0.5 * dlon,
                          coord_lon[ix_hi - 1] + 0.5 * dlon)
    lat_ascending = coord_lat[1] > coord_lat[0] if nlat > 1 else True
    if lat_ascending:
        lat_edges = np.append(coord_lat[iy_lo:iy_hi] - 0.5 * dlat,
                              coord_lat[iy_hi - 1] + 0.5 * dlat)
    else:
        lat_edges = np.append(coord_lat[iy_lo:iy_hi] + 0.5 * dlat,
                              coord_lat[iy_hi - 1] - 0.5 * dlat)

    # --- Panel 2: Catchment coverage ratio ---
    row_sum = np.array(mat.sum(axis=1)).ravel()  # m²

    with Dataset(str(parameter_nc), "r") as ds:
        param_cids = np.asarray(ds["catchment_id"][:]).astype(np.int64)
        cx = ds["catchment_x"][:]
        cy = ds["catchment_y"][:]
        catchment_area = np.asarray(ds["catchment_area"][:])  # m²
        lon_param = np.asarray(ds["longitude"][:]) if "longitude" in ds.variables else None
        lat_param = np.asarray(ds["latitude"][:]) if "latitude" in ds.variables else None
        nx_attr = getattr(ds, "nx", None)
        ny_attr = getattr(ds, "ny", None)

    nx = int(nx_attr) if nx_attr is not None else int(cx.max()) + 1
    ny = int(ny_attr) if ny_attr is not None else int(cy.max()) + 1

    idx = find_indices_in(mapping_cids, param_cids)
    valid = idx >= 0
    param_row_sum = np.zeros(len(param_cids), dtype=np.float64)
    param_row_sum[idx[valid]] = row_sum[np.where(valid)[0]]

    ratio = np.where(catchment_area > 0, param_row_sum / catchment_area, 0.0)

    ratio_map = np.full((nx, ny), np.nan)
    ratio_map[cx, cy] = ratio

    # Coordinate setup for catchment grid
    use_lonlat = lon_param is not None and lat_param is not None and len(cx) > 1
    if use_lonlat:
        sx, ix_c = np.polyfit(cx, lon_param, 1)
        sy, iy_c = np.polyfit(cy, lat_param, 1)
        cama_extent = (
            ix_c + sx * (-0.5), ix_c + sx * (nx - 0.5),
            iy_c + sy * (ny - 0.5), iy_c + sy * (-0.5),
        )
        cama_xlabel, cama_ylabel = "Longitude", "Latitude"
    else:
        cama_extent = None
        cama_xlabel, cama_ylabel = "X Index", "Y Index"

    # --- Common bounding box (lon/lat) from non-zero data ---
    # Source grid bbox
    src_lon_min = float(coord_lon[ix_nz.min()] - 0.5 * dlon)
    src_lon_max = float(coord_lon[ix_nz.max()] + 0.5 * dlon)
    if lat_ascending:
        src_lat_min = float(coord_lat[iy_nz.min()] - 0.5 * dlat)
        src_lat_max = float(coord_lat[iy_nz.max()] + 0.5 * dlat)
    else:
        src_lat_min = float(coord_lat[iy_nz.max()] - 0.5 * dlat)
        src_lat_max = float(coord_lat[iy_nz.min()] + 0.5 * dlat)

    # Catchment bbox (all catchments, so unmapped ones are also visible)
    if use_lonlat:
        cat_lon_min = float(lon_param.min())
        cat_lon_max = float(lon_param.max())
        cat_lat_min = float(lat_param.min())
        cat_lat_max = float(lat_param.max())
    else:
        cat_lon_min, cat_lon_max = src_lon_min, src_lon_max
        cat_lat_min, cat_lat_max = src_lat_min, src_lat_max

    # Union of both bboxes + margin
    bbox_margin = max(dlon, dlat) * 3
    common_lon_min = min(src_lon_min, cat_lon_min) - bbox_margin
    common_lon_max = max(src_lon_max, cat_lon_max) + bbox_margin
    common_lat_min = min(src_lat_min, cat_lat_min) - bbox_margin
    common_lat_max = max(src_lat_max, cat_lat_max) + bbox_margin

    # --- Plot ---
    dpi = 150 if interactive else 300
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=dpi)

    # Panel 1: source grid mapped area
    pos_vals = col_sum[nz_cols] / 1e6
    vmin1 = pos_vals.min()
    vmax1 = pos_vals.max()
    if vmin1 == vmax1:
        vmin1, vmax1 = vmin1 * 0.5, vmax1 * 2.0
    im1 = ax1.pcolormesh(
        lon_edges, lat_edges, np.ma.masked_invalid(src_crop),
        cmap="YlOrRd", norm=LogNorm(vmin=vmin1, vmax=vmax1), shading="flat",
    )
    fig.colorbar(im1, ax=ax1, label="Mapped area (km²)", shrink=0.7)
    ax1.set_title("Source grid: mapped area per cell")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_xlim(common_lon_min, common_lon_max)
    ax1.set_ylim(common_lat_min, common_lat_max)
    ax1.set_aspect("equal")

    # Panel 2: catchment coverage ratio
    masked_ratio = np.ma.masked_invalid(ratio_map.T)
    pos_r = ratio[ratio > 0]
    if pos_r.size > 0:
        vmin2, vmax2 = pos_r.min(), min(pos_r.max(), 2.0)
        if vmin2 >= vmax2:
            vmin2, vmax2 = 0.0, 2.0
        norm2 = Normalize(vmin=vmin2, vmax=vmax2)
    else:
        norm2 = None
    im2 = ax2.imshow(
        masked_ratio, origin="upper", cmap="RdYlGn_r", interpolation="nearest",
        norm=norm2, extent=cama_extent,
    )
    fig.colorbar(im2, ax=ax2, label="Mapped / Catchment area", shrink=0.7)
    ax2.set_title("Catchment: coverage ratio")
    ax2.set_xlabel(cama_xlabel)
    ax2.set_ylabel(cama_ylabel)
    ax2.set_xlim(common_lon_min, common_lon_max)
    ax2.set_ylim(common_lat_min, common_lat_max)

    mapped = int((param_row_sum > 0).sum())
    total = len(param_cids)
    fig.suptitle(
        f"Runoff Mapping: {npz_path.name}  "
        f"({mapped}/{total} catchments, {int(mat.nnz)} entries, "
        f"{len(nz_cols)} source cells)",
        fontsize=11,
    )
    plt.tight_layout()

    if interactive:
        plt.show()
    elif save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved mapping visualization to {save_path}")
        plt.close()
    else:
        save_path = npz_path.with_suffix(".png")
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved mapping visualization to {save_path}")
        plt.close()
