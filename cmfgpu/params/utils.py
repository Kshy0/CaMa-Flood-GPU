# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

from netCDF4 import Dataset
from cmfgpu.utils import find_indices_in

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
                # Specific gauges
                for g_id_str in gauges_val:
                    if str(g_id_str) in gauge_info:
                        target_cids.extend(gauge_info[str(g_id_str)].get("upstream_id", []))
                    else:
                        print(f"Warning: Gauge {g_id_str} not found in loaded gauge info.")
        else:
             # Fallback or warning if gauge_info not provided (e.g. from NC without gauge vars)
             print("Warning: 'gauges' POI requested but gauge_info not provided or available.")

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
                       print(f"Warning: No catchment found at coords ({x}, {y})")

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
    # Configuration
    longitude: Optional[np.ndarray] = None,
    latitude: Optional[np.ndarray] = None,
    title: str = "Basin Visualization",
    interactive: bool = False,
    basin_extra_text: Optional[Dict[int, str]] = None
) -> None:
    """
    Shared plotting logic for basins.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import ListedColormap
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
        extent = [left, right, bottom, top]
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
    if len(catchment_x) > 0:
        basin_map[catchment_x, catchment_y] = catchment_basin_id
        num_basins = int(catchment_basin_id.max()) + 1
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

    special_colors = []
    if gauges_xy: special_colors.append((0, 1, 0))
    if bifurcations: special_colors.append((0, 0, 1))
    if levees_xy: special_colors.append((0.5, 0, 0.5))

    basin_colors = generate_random_colors(num_basins, special_colors)
    cmap = ListedColormap(basin_colors)
    cmap.set_bad(alpha=0.0)

    # Plot
    plt.figure(figsize=(12, 10))
    plt.imshow(np.ma.masked_invalid(basin_map).T, origin='upper', cmap=cmap, interpolation='nearest',
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
         def within_extent(xv, yv): return np.zeros_like(xv, dtype=bool)

    # Overlays
    if gauges_xy:
        gx, gy = gauges_xy
        if use_lonlat: gx, gy = idx_to_lon(gx), idx_to_lat(gy)
        m = within_extent(gx, gy)
        if np.any(m):
            plt.scatter(gx[m], gy[m], c='#00FF00', s=0.5, label='Gauges', zorder=5)

    if levees_xy:
        lx, ly = levees_xy
        if use_lonlat: lx, ly = idx_to_lon(lx), idx_to_lat(ly)
        m = within_extent(lx, ly)
        if np.any(m):
            plt.scatter(lx[m], ly[m], c='#800080', s=0.2, label='Levees', zorder=4)
            
    if pois_xy:
        px, py = pois_xy
        if use_lonlat: px, py = idx_to_lon(px), idx_to_lat(py)
        m = within_extent(px, py)
        if np.any(m):
            plt.scatter(px[m], py[m], c="#C10000", s=0.3, label='Points of Interest', zorder=6)

    def plot_bifs(bifs, color, linestyle, label):
        if not bifs: return
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

    # Interactive
    if interactive:
        print("Interactive mode enabled. Click on the map to identify basins.")
        ax = plt.gca()
        ann = ax.annotate(
            "", xy=(0,0), xytext=(10,10), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.9),
            fontsize=8, visible=False, zorder=10
        )
        def on_click(event):
            if event.inaxes is not ax: return
             # Calc indices
            if use_lonlat:
                xi_f = invert_x(event.xdata)
                yi_f = invert_y(event.ydata)
            else:
                xi_f, yi_f = event.xdata, event.ydata
            
            xi, yi = int(np.round(xi_f)), int(np.round(yi_f))
            if not (0 <= xi < nx and 0 <= yi < ny): return

            val = basin_map[xi, yi]
            if np.isnan(val):
                ann.set_visible(False); plt.draw(); return
            
            basin_id = int(val)
            text = f"Basin: {basin_id}\nIdx: ({xi},{yi})"
            
            if basin_extra_text and basin_id in basin_extra_text:
                text += f"\n{basin_extra_text[basin_id]}"

            if use_lonlat: text += f"\nLoc: {event.xdata:.3f},{event.ydata:.3f}"
            print(f"Selected: {text.replace(chr(10), ', ')}")
            ann.set_text(text)
            ann.xy = (event.xdata, event.ydata)
            ann.set_visible(True)
            plt.draw()
            
        plt.gcf().canvas.mpl_connect("button_press_event", on_click)
        plt.show()
        return

    handles, labels = plt.gca().get_legend_handles_labels()
    if labels: plt.legend(loc='lower right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    plt.close()

def visualize_nc_basins(
    nc_path: Union[str, Path],
    save_path: Optional[Union[str, Path]] = None,
    visualize_gauges: bool = True,
    visualize_bifurcations: bool = True,
    visualize_levees: bool = True,
    interactive: bool = False,
    pois_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None
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
            pois_xy=pois_xy
        )

def crop_parameters_nc(
    input_nc: Union[str, Path],
    output_nc: Union[str, Path],
    points_of_interest: Dict[str, Any],
    visualize: bool = False,
    only_save_pois: bool = False
) -> None:
    """
    Crops an existing parameter NetCDF to a subset of basins covering specific points of interest.
    """
    input_nc = Path(input_nc)
    output_nc = Path(output_nc)

    with Dataset(input_nc, 'r') as src:
        # Load connectivity
        catchment_id = src['catchment_id'][:]
        catchment_x = src['catchment_x'][:]
        catchment_y = src['catchment_y'][:]
        catchment_basin_id = src['catchment_basin_id'][:]
        
        # Resolve target CIDs using shared logic
        target_cids = resolve_target_cids_from_poi(
            points_of_interest, 
            catchment_id, 
            catchment_x, 
            catchment_y,
            gauge_info=None 
        )
        
        if len(target_cids) == 0:
            print("No valid target catchments found to crop to. Aborting.")
            return

        # Find basins containing these catchments
        kept_basin_ids = get_kept_basin_ids(target_cids, catchment_id, catchment_basin_id)
        
        if len(kept_basin_ids) == 0:
             print("Target catchments not found in map. Aborting.")
             return

        if 'bifurcation_catchment_id' in src.variables and 'bifurcation_downstream_id' in src.variables:
            bif_up_cid = src['bifurcation_catchment_id'][:]
            bif_dn_cid = src['bifurcation_downstream_id'][:]
            
            cid_to_basin = dict(zip(catchment_id, catchment_basin_id))
            
            basin_adj = defaultdict(set)
            for u_cid, d_cid in zip(bif_up_cid, bif_dn_cid):
                u_basin = cid_to_basin.get(u_cid)
                d_basin = cid_to_basin.get(d_cid)
                
                if u_basin is not None and d_basin is not None and u_basin != d_basin:
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

        parent_map = {b: b for b in kept_basin_ids}
        def find_set(x):
            if parent_map[x] != x:
                parent_map[x] = find_set(parent_map[x])
            return parent_map[x]
        def union_sets(x, y):
            rootX, rootY = find_set(x), find_set(y)
            if rootX != rootY:
                parent_map[rootX] = rootY

        if 'bifurcation_catchment_id' in src.variables:
             for b in kept_basin_ids:
                 for neighbor in basin_adj[b]:
                     if neighbor in parent_map:
                         union_sets(b, neighbor)
        
        roots = sorted(list(set(find_set(b) for b in kept_basin_ids)))
        root_to_new_id = {r: i for i, r in enumerate(roots)}
        
        old_to_new_id = {b: root_to_new_id[find_set(b)] for b in kept_basin_ids}
        
        num_merged_basins = len(roots)
        
        keep_mask = np.isin(catchment_basin_id, kept_basin_ids)
        num_kept_catchments =  np.sum(keep_mask)
        print(f"Cropping from {len(catchment_id)} to {num_kept_catchments} catchments (Merged Basins: {num_merged_basins})")
        
        # Prepare catchment_save_mask based on only_save_pois logic
        # We need to compute it for the KEPT catchments only.
        kept_catchment_ids = catchment_id[keep_mask]
        
        if only_save_pois:
            new_save_mask = np.isin(kept_catchment_ids, target_cids)
        else:
            new_save_mask = np.ones(num_kept_catchments, dtype=np.bool_)

        with Dataset(output_nc, 'w') as dst:
            dst.setncatts(src.__dict__)
            
            old_unique_basins = np.sort(kept_basin_ids)
            map_idx_to_new = np.array([old_to_new_id[b] for b in old_unique_basins], dtype=np.int64)

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
                elif name == 'bifurcation_path': pass
                elif name == 'levee': pass
                else:
                    dst.createDimension(name, len(dim) if not dim.isunlimited() else None)
            
            bif_mask = None
            if 'bifurcation_basin_id' in src.variables:
                 bif_basin_id = src['bifurcation_basin_id'][:]
                 bif_mask = np.isin(bif_basin_id, kept_basin_ids)
                 if 'bifurcation_path' not in dst.dimensions:
                      dst.createDimension('bifurcation_path', np.sum(bif_mask))
            
            lev_mask = None
            if 'levee_basin_id' in src.variables:
                 lev_basin_id = src['levee_basin_id'][:]
                 lev_mask = np.isin(lev_basin_id, kept_basin_ids)
                 if 'levee' not in dst.dimensions:
                      dst.createDimension('levee', np.sum(lev_mask))

            for name, var in src.variables.items():
                dims = var.dimensions
                data = var[:] 
                primary_dim = dims[0] if dims else None
                
                if name == 'num_basins':
                     dst.createVariable(name, var.dtype, dims, zlib=True)
                     dst[name][:] = np.array(num_merged_basins, dtype=var.dtype)
                     continue

                if name == 'catchment_save_mask':
                    # Always overwrite with our computed mask
                    dst.createVariable(name, 'u1', dims, zlib=True) # force bool/u1
                    dst[name][:] = new_save_mask.astype('u1')
                    continue

                if primary_dim == 'catchment':
                     new_data = data[keep_mask]
                     if name == 'catchment_basin_id':
                         idx_in_kept = np.searchsorted(old_unique_basins, new_data)
                         new_data = map_idx_to_new[idx_in_kept].astype(new_data.dtype)
                     dst.createVariable(name, var.dtype, dims, zlib=True)
                     dst[name][:] = new_data
                     
                elif primary_dim == 'basin':
                     if name == 'basin_sizes':
                          new_data = new_basin_sizes.astype(data.dtype)
                     else:
                          new_data = data[roots]
                          
                     dst.createVariable(name, var.dtype, dims, zlib=True)
                     dst[name][:] = new_data

                elif primary_dim == 'bifurcation_path' and bif_mask is not None:
                     new_data = data[bif_mask]
                     if name == 'bifurcation_basin_id':
                          idx_in_kept = np.searchsorted(old_unique_basins, new_data)
                          new_data = map_idx_to_new[idx_in_kept].astype(new_data.dtype)
                     dst.createVariable(name, var.dtype, dims, zlib=True)
                     dst[name][:] = new_data
                     
                elif primary_dim == 'levee' and lev_mask is not None:
                     new_data = data[lev_mask]
                     if name == 'levee_basin_id':
                          idx_in_kept = np.searchsorted(old_unique_basins, new_data)
                          new_data = map_idx_to_new[idx_in_kept].astype(new_data.dtype)
                     dst.createVariable(name, var.dtype, dims, zlib=True)
                     dst[name][:] = new_data
                else:
                     dst.createVariable(name, var.dtype, dims, zlib=True)
                     dst[name][:] = data
            
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
         visualize_nc_basins(output_nc, save_path=img_path, pois_xy=pois_xy)
