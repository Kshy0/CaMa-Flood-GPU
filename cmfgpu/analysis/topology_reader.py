# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np


class TopologyReader:
    """
    Reader for analyzing catchment topology and parameters from CaMa-Flood-GPU parameters.nc.
    
    Features:
    - Lookup catchment by ID or (x, y) coordinates.
    - Retrieve upstream/downstream relationships.
    - Visualize basin structure and catchment location.
    - Print ASCII topology tree.
    """

    def __init__(self, params_path: Union[str, Path]):
        self.params_path = Path(params_path)
        if not self.params_path.exists():
            raise FileNotFoundError(f"Parameters file not found: {self.params_path}")
            
        self.ds = None
        self._load_data()

    def _load_data(self):
        """Load necessary variables into memory and build indices."""
        print(f"Loading topology from {self.params_path}...")
        self.catchment_data = {}
        
        with nc.Dataset(self.params_path, 'r') as ds:
            # Load dimensions
            if 'nx' in ds.variables:
                self.nx = ds.variables['nx'][:].item() if ds.variables['nx'].shape == () else ds.variables['nx'][0]
            else:
                self.nx = None
            
            if 'ny' in ds.variables:
                self.ny = ds.variables['ny'][:].item() if ds.variables['ny'].shape == () else ds.variables['ny'][0]
            else:
                self.ny = None

            # Load all 1D catchment variables
            for var_name, var in ds.variables.items():
                if var.dimensions == ('catchment',):
                    self.catchment_data[var_name] = var[:]
            
        # Ensure essential variables are present
        if 'catchment_id' not in self.catchment_data:
             raise ValueError("catchment_id not found in parameters.nc")
        
        self.c_ids = self.catchment_data['catchment_id']
        self.d_ids = self.catchment_data.get('downstream_id')
        self.b_ids = self.catchment_data.get('catchment_basin_id')
        self.c_x = self.catchment_data.get('catchment_x')
        self.c_y = self.catchment_data.get('catchment_y')

        # Build ID -> Index map
        self.id_to_idx = {cid: i for i, cid in enumerate(self.c_ids)}
        
        # Build Upstream Adjacency (Downstream ID -> List of Upstream IDs)
        self.rev_adj: Dict[int, List[int]] = {}
        
        if self.d_ids is not None:
            for i, did in enumerate(self.d_ids):
                if did != -9999: # Valid downstream
                    if did not in self.rev_adj:
                        self.rev_adj[did] = []
                    self.rev_adj[did].append(self.c_ids[i])
                
        print(f"Loaded {len(self.c_ids)} catchments.")

    def get_id_from_xy(self, x: int, y: int) -> Optional[int]:
        """Find catchment ID from (x, y) coordinates."""
        if self.c_x is None or self.c_y is None:
            return None
        matches = np.where((self.c_x == x) & (self.c_y == y))[0]
        if len(matches) > 0:
            return self.c_ids[matches[0]]
        return None

    def get_catchment_info(self, cid: Optional[int] = None, xy: Optional[Tuple[int, int]] = None) -> Optional[Dict]:
        """Get detailed information for a catchment."""
        if cid is None and xy is None:
            raise ValueError("Must provide either cid or xy.")
            
        if cid is None:
            cid = self.get_id_from_xy(*xy)
            if cid is None:
                return None
        
        if cid not in self.id_to_idx:
            return None
            
        idx = self.id_to_idx[cid]
        
        info = {
            "index": int(idx),
            "upstream_ids": self.rev_adj.get(cid, [])
        }
        
        # Add all loaded catchment variables
        for var_name, data in self.catchment_data.items():
            val = data[idx]
            if isinstance(val, (np.integer, np.floating)):
                val = val.item()
            info[var_name] = val
            
        return info

    def visualize_basin(self, cid: Optional[int] = None, xy: Optional[Tuple[int, int]] = None, figsize=(12, 10)):
        """
        Visualize the basin containing the specified catchment using a grid view.
        Highlights the mainstream (longest/largest) and the path from the target.
        """
        from matplotlib.collections import LineCollection

        info = self.get_catchment_info(cid, xy)
        if info is None:
            print("Catchment not found.")
            return
            
        target_cid = info['catchment_id']
        basin_id = info.get('catchment_basin_id')
        
        if basin_id is None or self.b_ids is None:
            print("Basin ID information not available.")
            return

        # Find all catchments in this basin
        basin_mask = (self.b_ids == basin_id)
        basin_indices = np.where(basin_mask)[0]
        
        if len(basin_indices) == 0:
            print(f"No catchments found for basin {basin_id}")
            return
            
        # Extract data for basin
        bx = self.c_x[basin_indices]
        by = self.c_y[basin_indices]
        b_cids = self.c_ids[basin_indices]
        
        # Determine bounding box
        min_x, max_x = bx.min(), bx.max()
        min_y, max_y = by.min(), by.max()
        margin = 2
        x0, x1 = max(0, min_x - margin), min(self.nx, max_x + margin + 1)
        y0, y1 = max(0, min_y - margin), min(self.ny, max_y + margin + 1)
        
        # Create grid for background (Elevation)
        grid_shape = (x1 - x0, y1 - y0)
        grid = np.full(grid_shape, np.nan)
        
        # Local coordinates
        lx = bx - x0
        ly = by - y0
        
        if 'catchment_elevation' in self.catchment_data:
            elev = self.catchment_data['catchment_elevation'][basin_indices]
            grid[lx, ly] = elev
            label = 'Elevation (m)'
            cmap = 'terrain'
        else:
            grid[lx, ly] = 1
            label = 'Basin Mask'
            cmap = 'Blues'

        # Check for Lat/Lon
        use_lonlat = 'longitude' in self.catchment_data and 'latitude' in self.catchment_data
        
        if use_lonlat:
            lons = self.catchment_data['longitude'][basin_indices]
            lats = self.catchment_data['latitude'][basin_indices]
            
            # Calculate dlon/dlat based on basin boundaries (User request: rough range, equal interval)
            min_x_basin, max_x_basin = bx.min(), bx.max()
            min_y_basin, max_y_basin = by.min(), by.max()
            
            min_lon_basin, max_lon_basin = lons.min(), lons.max()
            min_lat_basin, max_lat_basin = lats.min(), lats.max()
            
            if max_x_basin > min_x_basin:
                dlon = (max_lon_basin - min_lon_basin) / (max_x_basin - min_x_basin)
            else:
                dlon = 360.0 / self.nx if self.nx else 0.1
                
            if max_y_basin > min_y_basin:
                # Standard orientation: Y increases Southwards (Lat decreases)
                dlat = (min_lat_basin - max_lat_basin) / (max_y_basin - min_y_basin)
            else:
                dlat = -180.0 / self.ny if self.ny else -0.1
                
            # Calculate extent for imshow
            # x0, x1 are indices. Pixel center is at index. Edge at index +/- 0.5
            left_lon = min_lon_basin + (x0 - 0.5 - min_x_basin) * dlon
            right_lon = min_lon_basin + (x1 - 0.5 - min_x_basin) * dlon
            
            top_lat = max_lat_basin + (y0 - 0.5 - min_y_basin) * dlat
            bottom_lat = max_lat_basin + (y1 - 0.5 - min_y_basin) * dlat
            
            extent = [left_lon, right_lon, bottom_lat, top_lat]
            xlabel = "Longitude"
            ylabel = "Latitude"
            
            # Recalculate plot coordinates for all catchments
            c_x_plot = min_lon_basin + (self.c_x - min_x_basin) * dlon
            c_y_plot = max_lat_basin + (self.c_y - min_y_basin) * dlat
            
            target_x = min_lon_basin + (info['catchment_x'] - min_x_basin) * dlon
            target_y = max_lat_basin + (info['catchment_y'] - min_y_basin) * dlat
        else:
            # Use indices, centered
            extent = [x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5]
            xlabel = "X Index"
            ylabel = "Y Index"
            c_x_plot = self.c_x
            c_y_plot = self.c_y
            target_x = info['catchment_x']
            target_y = info['catchment_y']

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(grid.T, origin='upper', extent=extent, cmap=cmap, alpha=0.6)
        plt.colorbar(im, ax=ax, label=label)
        
        # --- Identify Mainstream ---
        # 1. Find Mouth of Basin (Max Upstream Area)
        if 'upstream_area' in self.catchment_data:
            up_areas = self.catchment_data['upstream_area'][basin_indices]
            mouth_local_idx = np.argmax(up_areas)
            mouth_idx = basin_indices[mouth_local_idx]
            mouth_id = self.c_ids[mouth_idx]
        else:
            # Fallback: find node with no downstream in basin
            basin_cids_set = set(b_cids)
            mouth_id = target_cid # Default
            for idx in basin_indices:
                did = self.d_ids[idx]
                if did not in basin_cids_set:
                    mouth_id = self.c_ids[idx]
                    break
        
        # 2. Trace Mainstream (Upstream from Mouth)
        mainstream_path = [mouth_id]
        visited_main = {mouth_id}
        curr = mouth_id
        while True:
            ups = self.rev_adj.get(curr, [])
            if not ups:
                break
            # Choose up with max area
            best_up = None
            max_area = -1.0
            
            for up in ups:
                if up in self.id_to_idx:
                    u_idx = self.id_to_idx[up]
                    # Check if in basin
                    if self.b_ids[u_idx] == basin_id:
                        area = self.catchment_data['upstream_area'][u_idx] if 'upstream_area' in self.catchment_data else 0
                        if area > max_area:
                            max_area = area
                            best_up = up
            
            if best_up is not None and best_up not in visited_main:
                mainstream_path.append(best_up)
                visited_main.add(best_up)
                curr = best_up
            else:
                break
        
        # 3. Trace Target Path (Downstream from Target)
        target_path = []
        visited_target = set()
        curr = target_cid
        while curr in self.id_to_idx:
            if curr in visited_target:
                print(f"Warning: Cycle detected at {curr}")
                break
            visited_target.add(curr)
            target_path.append(curr)
            idx = self.id_to_idx[curr]
            did = self.d_ids[idx]
            if did == -9999 or did == curr:
                break
            if did in self.id_to_idx and self.b_ids[self.id_to_idx[did]] == basin_id:
                curr = did
            else:
                break
                
        # --- Prepare Segments ---
        mainstream_set = set(mainstream_path)
        target_path_set = set(target_path)
        
        seg_normal = []
        seg_main = []
        seg_target = []
        
        for idx in basin_indices:
            cid = self.c_ids[idx]
            did = self.d_ids[idx]
            
            if did != -9999 and did in self.id_to_idx:
                d_idx = self.id_to_idx[did]
                
                # Coords
                x1_c, y1_c = c_x_plot[idx], c_y_plot[idx]
                x2_c, y2_c = c_x_plot[d_idx], c_y_plot[d_idx]
                
                # Add segment (x, y)
                segment = [(x1_c, y1_c), (x2_c, y2_c)]
                
                is_target = (cid in target_path_set and did in target_path_set)
                is_main = (cid in mainstream_set and did in mainstream_set)
                
                if is_target:
                    seg_target.append(segment)
                elif is_main:
                    seg_main.append(segment)
                else:
                    seg_normal.append(segment)

        # Add collections
        # Normal: Thin gray
        lc_norm = LineCollection(seg_normal, colors='gray', linewidths=0.5, alpha=0.3)
        # Mainstream: Thick Blue
        lc_main = LineCollection(seg_main, colors='blue', linewidths=2.0, alpha=0.8)
        # Target: Thick Red (drawn last to be on top)
        lc_target = LineCollection(seg_target, colors='red', linewidths=2.5, alpha=0.9)
        
        ax.add_collection(lc_norm)
        ax.add_collection(lc_main)
        ax.add_collection(lc_target)
        
        # Plot Target Point
        ax.scatter(target_x, target_y, c='red', s=200, marker='*', label='Target', zorder=100, edgecolors='black')
        
        # Plot Outlet Point
        if mouth_id in self.id_to_idx:
            mouth_idx_global = self.id_to_idx[mouth_id]
            mouth_x = c_x_plot[mouth_idx_global]
            mouth_y = c_y_plot[mouth_idx_global]
            ax.scatter(mouth_x, mouth_y, c='cyan', s=150, marker='s', label='Outlet', zorder=100, edgecolors='black')
        
        ax.set_title(f"Basin {basin_id} Flow Network\nTarget: {target_cid} (Mainstream: Blue, Target Path: Red)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Dummy legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='Mainstream'),
            Line2D([0], [0], color='red', lw=2, label='Target Path'),
            Line2D([0], [0], color='gray', lw=0.5, label='Tributaries'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markeredgecolor='black', markersize=15, label='Target'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='cyan', markeredgecolor='black', markersize=10, label='Outlet')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.show()

    def print_topology(self, cid: Optional[int] = None, xy: Optional[Tuple[int, int]] = None):
        """
        Print the flow path passing through the catchment: Source -> Target -> Outlet.
        """
        info = self.get_catchment_info(cid, xy)
        if info is None:
            print("Catchment not found.")
            return
            
        target_cid = info['catchment_id']
        
        # 1. Trace Downstream
        down_path = []
        curr = target_cid
        visited = {curr}
        while True:
            if curr not in self.id_to_idx:
                break
            idx = self.id_to_idx[curr]
            did = self.d_ids[idx]
            
            if did == -9999:
                break
            if did in visited:
                break
            
            down_path.append(did)
            visited.add(did)
            curr = did
            
        # 2. Trace Upstream (Longest)
        up_path = []
        curr = target_cid
        visited_up = {curr}
        
        while True:
            ups = self.rev_adj.get(curr, [])
            if not ups:
                break
            
            # Find upstream with max area
            best_up = None
            max_area = -1.0
            
            for up in ups:
                if up in self.id_to_idx:
                    u_idx = self.id_to_idx[up]
                    area = self.catchment_data['upstream_area'][u_idx] if 'upstream_area' in self.catchment_data else 0
                    if area > max_area:
                        max_area = area
                        best_up = up
            
            if best_up and best_up not in visited_up:
                up_path.append(best_up)
                visited_up.add(best_up)
                curr = best_up
            else:
                break
                
        # Combine: Reverse Up -> Target -> Down
        full_path = list(reversed(up_path)) + [target_cid] + down_path
        
        # Print
        print(f"Flow Path for Catchment {target_cid}:")
        prefix = ""
        for i, node_id in enumerate(full_path):
            connector = "└── " if i > 0 else ""
            
            # Determine label
            label = f"ID: {node_id}"
            if node_id == target_cid:
                label += " (Current)"
            elif i == 0:
                label += " (Source)"
            elif i == len(full_path) - 1:
                label += " (Outlet)"
                
            print(f"{prefix}{connector}{label}")
            
            if i > 0:
                prefix += "    "

