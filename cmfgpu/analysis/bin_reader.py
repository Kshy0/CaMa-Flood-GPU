# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


class BinReader:
    """
    Reader for CaMa-Flood CPU version binary outputs (e.g., rivoutYYYY.bin).
    Assumes global binary files use column-major (nx, ny) frame layout.
    
    Provides the same public reader interface as MultiRankStatsReader.
    """

    def _load_dims(self):
        # Look for mapdim.txt in base_dir or parent directories
        explicit_nx = self.nx
        explicit_ny = self.ny
        candidates = [
            self.base_dir / "mapdim.txt",
            self.base_dir.parent / "mapdim.txt",
            self.base_dir.parent.parent / "mapdim.txt"
        ]
        
        mapdim_path = None
        for p in candidates:
            if p.exists():
                mapdim_path = p
                break
        
        if mapdim_path is None:
            return

        positional: list[int] = []
        try:
            with mapdim_path.open("r") as f:
                for line in f:
                    head = line.split("!!", 1)[0].strip()
                    if not head:
                        continue
                    parts = head.replace("=", " ").split()
                    key = parts[0].lower()
                    if (
                        key in {"nx", "nxx"}
                        and len(parts) >= 2
                        and self.nx is None
                    ):
                        self.nx = int(parts[1])
                    elif (
                        key in {"ny", "nyy"}
                        and len(parts) >= 2
                        and self.ny is None
                    ):
                        self.ny = int(parts[1])
                    elif len(parts) == 1:
                        try:
                            positional.append(int(parts[0]))
                        except ValueError:
                            continue
            if self.nx is None and positional:
                self.nx = positional[0]
            if self.ny is None and len(positional) >= 2:
                self.ny = positional[1]
        except (OSError, ValueError) as exc:
            raise ValueError(f"Failed to read {mapdim_path}: {exc}") from exc
        if self.nx is None or self.ny is None or self.nx <= 0 or self.ny <= 0:
            raise ValueError(
                f"{mapdim_path} does not define positive nx/ny dimensions"
            )
        # Explicit constructor dimensions take precedence over metadata while
        # still allowing the missing partner to be discovered from mapdim.
        if explicit_nx is not None:
            self.nx = explicit_nx
        if explicit_ny is not None:
            self.ny = explicit_ny
        print(f"Loaded dimensions from {mapdim_path}: nx={self.nx}, ny={self.ny}")

    def _scan_files(self) -> List[dict]:
        """
        Scan for files named {var_name}YYYY.bin
        Returns a list of dicts with file info.
        """
        # Pattern for rivoutYYYY.bin
        pattern = re.compile(rf"^{re.escape(self.var_name)}(\d{{4}})\.bin$")
        files = []
        
        if not self.base_dir.exists():
            return []

        for p in sorted(self.base_dir.glob(f"{self.var_name}*.bin")):
            match = pattern.match(p.name)
            if match:
                year = int(match.group(1))
                size = p.stat().st_size
                if self.nx and self.ny:
                    frame_size = self.nx * self.ny * 4 # float32
                    if size % frame_size:
                        raise ValueError(
                            f"{p} size {size} is not a whole number of "
                            f"{self.nx}x{self.ny} float32 frames"
                        )
                    n_frames = size // frame_size
                else:
                    n_frames = 0 # Unknown
                
                files.append({
                    "path": p,
                    "year": year,
                    "n_frames": n_frames,
                    "start_idx": 0, # To be filled
                    "end_idx": 0    # To be filled
                })
        
        # Sort by year
        files.sort(key=lambda x: x["year"])
        
        # Calculate global indices
        current_idx = 0
        for f in files:
            f["start_idx"] = current_idx
            f["end_idx"] = current_idx + f["n_frames"]
            current_idx += f["n_frames"]
            
        return files

    def _build_time_axis(self):
        self.times = []
        if self.start_datetime is not None:
            current = self.start_datetime
            for f in self.files:
                if f["n_frames"] and current.year != f["year"]:
                    raise ValueError(
                        f"Explicit start_datetime and frame counts place "
                        f"{f['path'].name} at {current:%Y-%m-%d}, which does "
                        f"not match its filename year {f['year']}"
                    )
                for _ in range(f["n_frames"]):
                    self.times.append(current)
                    current += timedelta(seconds=self.dt)
            self._time_len = len(self.times)
            return

        for f in self.files:
            year = f["year"]
            
            # Check if it looks like daily data
            is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
            days_in_year = 366 if is_leap else 365
            
            start_date = datetime(year, 1, 1)
            seconds_in_year = days_in_year * 86400
            complete_frames = (
                seconds_in_year // self.dt
                if seconds_in_year % self.dt == 0
                else None
            )
            if f["n_frames"] != complete_frames:
                warnings.warn(
                    f"{f['path'].name} has {f['n_frames']} frames, not a "
                    f"complete {year} at dt={self.dt}s; timestamps are being "
                    "anchored at January 1. Pass start_datetime for a "
                    "partial-year run.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            for i in range(f["n_frames"]):
                self.times.append(
                    start_date + timedelta(seconds=i * self.dt)
                )
                     
        self._time_len = len(self.times)
    def __init__(
        self,
        base_dir: Union[str, Path],
        var_name: str,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
        dt: int = 86400,  # Default daily (seconds)
        unit: str = "m3/s",
        start_datetime: Optional[datetime] = None,
    ):
        self.base_dir = Path(base_dir)
        self.var_name = var_name
        if type(dt) is not int or dt <= 0:
            raise ValueError("dt must be a positive integer number of seconds")
        self.dt = dt
        self.unit = unit
        if start_datetime is not None and not isinstance(
            start_datetime, datetime
        ):
            raise TypeError("start_datetime must be a datetime or None")
        self.start_datetime = start_datetime
        self.nx = nx
        self.ny = ny

        for name, value in (("nx", nx), ("ny", ny)):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer")

        # Try to load dimensions if not provided
        if nx is None or ny is None:
            self._load_dims()

        if self.nx is None or self.ny is None:
            raise ValueError(
                "Map dimensions not found; provide nx and ny or a valid mapdim.txt"
            )
        if (
            type(self.nx) is not int
            or type(self.ny) is not int
            or self.nx <= 0
            or self.ny <= 0
        ):
            raise ValueError("nx and ny must be positive integers")
        
        self.map_shape = (self.nx, self.ny) if (self.nx and self.ny) else None
        self.files = self._scan_files()
        
        if not self.files:
             print(f"Warning: No files found for variable {var_name} in {base_dir}")
             self.times = []
             self._time_len = 0
        else:
            self._build_time_axis()

    @property
    def time_len(self) -> int:
        return self._time_len
    
    @property
    def num_ranks(self) -> int:
        return 1 # CPU model is effectively 1 rank for reading purposes

    def get_grid(
        self,
        t_index: int,
        fill_value: float = np.nan,
    ) -> np.ndarray:
        if t_index < 0 or t_index >= self._time_len:
            raise IndexError(f"Time index {t_index} out of range [0, {self._time_len})")
            
        if self.nx is None or self.ny is None:
            raise RuntimeError("Map dimensions not set")

        # Find which file contains t_index
        target_file = None
        local_idx = 0
        
        for f in self.files:
            if f["start_idx"] <= t_index < f["end_idx"]:
                target_file = f
                local_idx = t_index - f["start_idx"]
                break
        
        if target_file is None:
            raise RuntimeError(f"Could not find file for index {t_index}")
            
        # Read data
        offset = local_idx * self.nx * self.ny * 4
        with open(target_file["path"], "rb") as fh:
            fh.seek(offset)
            data = np.fromfile(fh, dtype="<f4", count=self.nx * self.ny)
            
        if data.size != self.nx * self.ny:
            raise EOFError(
                f"Short read from {target_file['path']}: got {data.size} values, "
                f"expected {self.nx * self.ny}"
            )
        grid = data.reshape((self.nx, self.ny), order='F')
        missing = (grid == np.float32(-9999.0)) | (grid >= np.float32(1.0e19))
        if np.any(missing):
            grid = grid.copy()
            grid[missing] = fill_value
        return grid

    def get_vector(self, t_index: int) -> np.ndarray:
        grid = self.get_grid(t_index)
        return grid.flatten(order='F')

    def get_series(
        self,
        points: Union[np.ndarray, Sequence[Tuple[int, int]]],
        fill_value: float = np.nan,
    ) -> np.ndarray:
        """
        Get time series for specific points (x, y).
        points: list of (x, y) tuples or (N, 2) array. 0-based indices.
        """
        pts = np.asarray(points)
        if pts.ndim != 2 or pts.shape[1] != 2:
             raise ValueError("Points must be (N, 2) array of (x, y) coordinates")
        if not np.issubdtype(pts.dtype, np.integer):
            if not np.issubdtype(pts.dtype, np.number) or not np.all(
                np.isfinite(pts) & (pts == np.floor(pts))
            ):
                raise TypeError("Point coordinates must be finite integers")
        pts = pts.astype(np.int64, copy=False)
             
        if self.nx is None or self.ny is None:
            raise RuntimeError("Map dimensions not set")

        n_points = len(pts)
        out = np.full((self._time_len, n_points), fill_value, dtype=np.float32)
        
        # pts is (x, y). 
        # In memmap (n_frames, ny, nx) [C-order], this corresponds to [t, y, x]
        ys = pts[:, 1]
        xs = pts[:, 0]
        invalid = (xs < 0) | (xs >= self.nx) | (ys < 0) | (ys >= self.ny)
        if np.any(invalid):
            raise IndexError(
                f"Point coordinates outside [0,{self.nx}) x [0,{self.ny}): "
                f"{pts[invalid][:5].tolist()}"
            )
        
        current_t = 0
        for f_info in self.files:
            path = f_info["path"]
            n_frames = f_info["n_frames"]
            
            # Memmap as (n_frames, ny, nx) C-order over column-major frames.
            mmap = np.memmap(path, dtype="<f4", mode="r", shape=(n_frames, self.ny, self.nx))
            
            # Extract points
            # mmap[:, ys, xs] uses fancy indexing on the last two dimensions
            chunk_data = mmap[:, ys, xs]
            missing = (
                (chunk_data == np.float32(-9999.0))
                | (chunk_data >= np.float32(1.0e19))
            )
            if np.any(missing):
                chunk_data = np.array(chunk_data, copy=True)
                chunk_data[missing] = fill_value
            out[current_t : current_t + n_frames, :] = chunk_data
            
            current_t += n_frames
            
            # Close memmap
            del mmap
            
        return out

    def plot_single_time(
        self,
        t_index: int = 0,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None,
    ) -> None:
        # Plotting is an optional analysis feature; keep matplotlib out of the
        # reader's import path so the base package works without the ``dev``
        # extra (and without loading a second C++ runtime during I/O tests).
        import matplotlib.pyplot as plt

        grid = self.get_grid(t_index)
        
        # Mask missing values
        grid = np.ma.masked_equal(grid, -9999.0)
        grid = np.ma.masked_greater(grid, 1e19)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if vmin is None:
            vmin = np.nanmin(grid)
        if vmax is None:
            vmax = np.nanmax(grid)
        
        # Plot transposed grid (ny, nx) so x-axis is longitude (nx), y-axis is latitude (ny)
        im = ax.imshow(grid.T, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
        fig.colorbar(im, ax=ax, label=self.unit)
        
        t_str = self.times[t_index].strftime("%Y-%m-%d %H:%M") if self.times else f"Index {t_index}"
        ax.set_title(title or f"{self.var_name} at {t_str}")
        ax.set_xlabel("Longitude Index")
        ax.set_ylabel("Latitude Index")
        
        plt.show()

    def animate(
        self,
        out_path: Union[str, Path],
        t_range: Optional[Tuple[int, int]] = None,
        fps: int = 10,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (10, 6),
    ) -> None:
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        t_start = 0 if t_range is None else max(0, int(t_range[0]))
        t_end = self._time_len if t_range is None else min(self._time_len, int(t_range[1]))
        
        if t_start >= t_end:
            print("Invalid time range")
            return

        # Setup figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Initial frame
        grid = self.get_grid(t_start)
        grid = np.ma.masked_equal(grid, -9999.0)
        grid = np.ma.masked_greater(grid, 1e19)
        
        if vmin is None:
            vmin = np.nanmin(grid)
        if vmax is None:
            vmax = np.nanmax(grid)
        
        im = ax.imshow(grid.T, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
        title = ax.set_title(f"{self.var_name}")
        fig.colorbar(im, ax=ax, label=self.unit)
        
        def update(frame_idx):
            t = t_start + frame_idx
            grid = self.get_grid(t)
            grid = np.ma.masked_equal(grid, -9999.0)
            grid = np.ma.masked_greater(grid, 1e19)
            im.set_data(grid.T)
            t_str = self.times[t].strftime("%Y-%m-%d") if self.times else f"{t}"
            title.set_text(f"{self.var_name} @ {t_str}")
            return im, title

        ani = animation.FuncAnimation(fig, update, frames=t_end - t_start, interval=1000/fps, blit=False)
        
        out_path = Path(out_path)
        if out_path.suffix == '.gif':
            ani.save(out_path, writer='pillow', fps=fps)
        else:
            ani.save(out_path, writer='ffmpeg', fps=fps)
            
        plt.close(fig)
        print(f"Animation saved to {out_path}")

    def summary(self) -> str:
        lines = [
            f"Variable         : {self.var_name}",
            f"Base dir         : {self.base_dir}",
            f"Map shape        : {self.map_shape}",
            f"Time len         : {self.time_len}",
            f"First time       : {self.times[0] if self.times else 'N/A'}",
            f"Last time        : {self.times[-1] if self.times else 'N/A'}",
            f"Files found      : {len(self.files)}",
        ]
        return "\n".join(lines)
