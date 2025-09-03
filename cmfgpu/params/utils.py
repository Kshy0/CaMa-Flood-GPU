# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from collections import defaultdict
from pathlib import Path
from typing import Dict

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
def _lpt_is_balanced(basin_sizes: np.ndarray, n_ranks: int, tol: float):
    """
    LPT (Largest Processing Time first) load balance check.
    We model "work" as the number of catchments per basin.
    Returns:
        balanced: bool
        loads: int64[n_ranks]
        lo: int64 (min load)
        hi: int64 (max load)
        avg: float64 (average load)
    """
    M = basin_sizes.size
    if n_ranks <= 0 or M == 0:
        return True, np.zeros(0, np.int64), np.int64(0), np.int64(0), 0.0

    n_eff = n_ranks if n_ranks <= M else M 
    sizes = basin_sizes.astype(np.int64)
    order = np.argsort(sizes)  # ascending
    loads = np.zeros(n_eff, np.int64)

    # LPT: largest first -> argmin(load)
    for k in range(order.size - 1, -1, -1):
        b = order[k]
        best_r = np.argmin(loads)
        loads[best_r] += sizes[b]

    total = np.sum(loads)
    avg = total / float(n_eff) if n_eff > 0 else 0.0
    lo = np.min(loads) if loads.size > 0 else np.int64(0)
    hi = np.max(loads) if loads.size > 0 else np.int64(0)

    T_star = (1.0 + tol) * avg
    balanced = (float(hi) <= T_star + 1e-9)

    return balanced, loads, lo, hi, avg

@njit
def _components_sizes(n_nodes: int, edges_u: np.ndarray, edges_v: np.ndarray, alive_edge: np.ndarray, node_weight: np.ndarray):
    """
    Compute connected components and their weighted sizes on an undirected graph using CSR + BFS.
    Parameters:
        n_nodes     : number of nodes
        edges_u/v   : arrays of undirected edges (each edge once with u <= v)
        alive_edge  : boolean mask per edge indicating if the edge is active
        node_weight : weight (e.g., catchment count) per node (int64)
    Returns:
        comp_id     : int64[n_nodes], component id per node (0..C-1)
        comp_sizes  : int64[C], sum of node_weight per component
    """
    # Compute degree per node to size CSR
    deg = np.zeros(n_nodes, np.int64)
    m = edges_u.size
    for e in range(m):
        if alive_edge[e]:
            u = edges_u[e]
            v = edges_v[e]
            if u != v:
                deg[u] += 1
                deg[v] += 1

    # Build CSR adjacency
    ofs = np.zeros(n_nodes + 1, np.int64)
    for i in range(n_nodes):
        ofs[i + 1] = ofs[i] + deg[i]
    adj = np.zeros(ofs[n_nodes], np.int64)

    fill = np.zeros(n_nodes, np.int64)
    for e in range(m):
        if alive_edge[e]:
            u = edges_u[e]
            v = edges_v[e]
            if u != v:
                pos = ofs[u] + fill[u]
                adj[pos] = v
                fill[u] += 1
                pos = ofs[v] + fill[v]
                adj[pos] = u
                fill[v] += 1

    # BFS over components
    comp = -np.ones(n_nodes, np.int64)
    comp_sizes_tmp = np.empty(n_nodes, np.int64)  # upper bound; trimmed after
    C = 0
    q = np.empty(n_nodes, np.int64)

    for s in range(n_nodes):
        if comp[s] >= 0:
            continue
        head = 0
        tail = 0
        q[tail] = s
        tail += 1
        comp[s] = C
        total = node_weight[s]

        while head < tail:
            u = q[head]
            head += 1
            for jj in range(ofs[u], ofs[u + 1]):
                w = adj[jj]
                if comp[w] < 0:
                    comp[w] = C
                    q[tail] = w
                    tail += 1
                    total += node_weight[w]

        comp_sizes_tmp[C] = total
        C += 1

    comp_sizes = np.zeros(C, np.int64)
    for i in range(C):
        comp_sizes[i] = comp_sizes_tmp[i]

    return comp, comp_sizes

def min_cuts_for_balance(
    river_mouth_id: np.ndarray,
    bif_from_mouth: np.ndarray,
    bif_to_mouth: np.ndarray,
    n_ranks: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, Dict]:
    """
    Plan inter-basin unions with minimal breaks using a binary search on how many
    inter-basin edges to keep. Reuses lpt_is_balanced for load checking.

    Returns:
        root_mouth: representative mouth id per catchment (same shape as river_mouth_id)
        report: dictionary with balance report information
        kept_mouth_pairs: int64[K, 2], the kept undirected mouth-to-mouth pairs (as mouth ids)
    """
    mouths, inv = np.unique(river_mouth_id, return_inverse=True)
    M = mouths.size

    # Weight per mouth = number of catchments draining to that mouth
    node_wt = np.bincount(inv, minlength=M).astype(np.int64)

    # Build unique undirected inter-basin edges from bifurcations (from_mouth != to_mouth)
    a = bif_from_mouth
    b = bif_to_mouth
    mask = (a != b)
    a = a[mask]
    b = b[mask]
    if a.size == 0:
        raise ValueError("No valid bifurcation connections found.")

    idx_of = {int(m): i for i, m in enumerate(mouths)}
    ai = np.array([idx_of.get(int(x), -1) for x in a], dtype=np.int64)
    bi = np.array([idx_of.get(int(x), -1) for x in b], dtype=np.int64)
    valid = (ai >= 0) & (bi >= 0)
    ai = ai[valid]
    bi = bi[valid]
    if ai.size == 0:
        raise ValueError("No valid bifurcation connections found after filtering.")

    uu = np.minimum(ai, bi)
    vv = np.maximum(ai, bi)
    edges = np.unique(np.stack([uu, vv], axis=1), axis=0)

    edges_u = edges[:, 0].astype(np.int64)
    edges_v = edges[:, 1].astype(np.int64)
    E = edges_u.size

    def rep_root_from_comp(comp_arr: np.ndarray):
        rep_mouth = np.zeros(comp_arr.max() + 1, dtype=mouths.dtype)
        seen = set()
        for i, c in enumerate(comp_arr):
            ci = int(c)
            if ci not in seen:
                seen.add(ci)
                rep_mouth[ci] = mouths[i]
        return rep_mouth

    if E == 0:
        # No inter-basin edges; components are single mouths.
        comp = np.arange(M, dtype=np.int64)
        sizes = node_wt.copy()
        ok, loads_chk, lo, hi, avg = _lpt_is_balanced(sizes, int(n_ranks), float(tol))
        rep_mouth = rep_root_from_comp(comp)
        root_out = rep_mouth[comp[inv]]
        report = {
            "balanced": bool(ok),
            "loads": loads_chk.tolist(),
            "lo": int(lo),
            "hi": float(hi),
            "avg": float(avg),
            "n_components": int(sizes.size),
            "n_edges_initial": 0,
            "n_edges_alive": 0,
            "n_edges_removed": 0,
            "removed_ratio": 0.0,
        }
        kept_mouth_pairs = np.zeros((0, 2), dtype=mouths.dtype)
        return root_out, kept_mouth_pairs, report

    # Edge order: add merges of small total size first
    edge_weight = node_wt[edges_u] + node_wt[edges_v]
    order = np.argsort(edge_weight, kind="stable")
    edges_u_ord = edges_u[order]
    edges_v_ord = edges_v[order]

    alive = np.zeros(E, dtype=np.bool_)

    def compute_components_with_K(K: int):
        # keep first K edges in the chosen order
        alive[:] = False
        if K > 0:
            alive[:K] = True
        comp, sizes = _components_sizes(M, edges_u_ord, edges_v_ord, alive, node_wt)
        return comp, sizes

    # Feasibility probe: K = 0 (all inter-basin links cut)
    comp0, sizes0 = compute_components_with_K(0)
    ok0, loads0, lo0, hi0, avg0 = _lpt_is_balanced(sizes0, int(n_ranks), float(tol))
    if not ok0:
        rep0 = rep_root_from_comp(comp0)
        root_out = rep0[comp0[inv]]
        n_edges_initial = int(E)
        n_edges_alive = 0
        n_edges_removed = n_edges_initial - n_edges_alive
        removed_ratio = float(n_edges_removed) / float(max(1, n_edges_initial))
        report = {
            "balanced": False,
            "loads": loads0.tolist(),
            "lo": int(lo0),
            "hi": float(hi0),
            "avg": float(avg0),
            "n_components": int(sizes0.size),
            "n_edges_initial": n_edges_initial,
            "n_edges_alive": n_edges_alive,
            "n_edges_removed": n_edges_removed,
            "removed_ratio": removed_ratio,
        }
        kept_mouth_pairs = np.zeros((0, 2), dtype=mouths.dtype)
        return root_out, kept_mouth_pairs, report

    # Binary search K in [0, E] to maximize kept edges while keeping balance
    loK = 0
    hiK = E
    best = {
        "K": 0,
        "comp": comp0,
        "sizes": sizes0,
        "loads": loads0,
        "lo": lo0,
        "hi": hi0,
        "avg": avg0,
    }

    while loK <= hiK:
        mid = (loK + hiK) // 2
        comp_mid, sizes_mid = compute_components_with_K(mid)
        ok_mid, loads_mid, lo_mid, hi_mid, avg_mid = _lpt_is_balanced(sizes_mid, int(n_ranks), float(tol))
        if ok_mid:
            best.update({
                "K": mid,
                "comp": comp_mid,
                "sizes": sizes_mid,
                "loads": loads_mid,
                "lo": lo_mid,
                "hi": hi_mid,
                "avg": avg_mid,
            })
            loK = mid + 1
        else:
            hiK = mid - 1

    # Build final result from best K
    K_star = int(best["K"])
    comp_fin = best["comp"]
    sizes_fin = best["sizes"]
    loads_fin = best["loads"]
    lo_fin = best["lo"]
    hi_fin = best["hi"]
    avg_fin = best["avg"]
    ok_fin = True  # feasible by construction

    rep = rep_root_from_comp(comp_fin)
    root_out = rep[comp_fin[inv]]

    # NEW: Only consider an edge "removed" if it spans different final components.
    # Any edge whose endpoints are within the same component is safe to keep
    # (it does not change connectivity) and should not be marked as removed.
    same_comp_mask = (comp_fin[edges_u_ord] == comp_fin[edges_v_ord])

    # Kept edges = all edges that lie within the same final component
    kept_idx = np.nonzero(same_comp_mask)[0]
    kept_u = edges_u_ord[kept_idx]
    kept_v = edges_v_ord[kept_idx]

    n_edges_initial = int(E)
    n_edges_alive = int(kept_idx.size)
    n_edges_removed = n_edges_initial - n_edges_alive
    removed_ratio = float(n_edges_removed) / float(max(1, n_edges_initial))

    if n_edges_alive > 0:
        kept_mouth_pairs = np.stack([mouths[kept_u], mouths[kept_v]], axis=1)
    else:
        kept_mouth_pairs = np.zeros((0, 2), dtype=mouths.dtype)

    report = {
        "balanced": bool(ok_fin),
        "loads": loads_fin.tolist(),
        "lo": int(lo_fin),
        "hi": float(hi_fin),
        "avg": float(avg_fin),
        "n_components": int(sizes_fin.size),
        "n_edges_initial": n_edges_initial,
        "n_edges_alive": n_edges_alive,
        "n_edges_removed": n_edges_removed,
        "removed_ratio": removed_ratio,
        # Optional: also expose K* that defines connectivity and how many extra edges were auto-kept
        "K_star_defining": K_star,
        "extra_kept_intra_component": int(n_edges_alive - min(K_star, n_edges_initial)),
    }
    return root_out, kept_mouth_pairs, report

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
