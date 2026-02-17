# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Example: estimate river width / height from ELSE_GPCC daily climatology.

1. Build a ``DailyBinDataset`` with a constant ``time_to_key`` (single
   binary file containing 365 daily frames) and construct the local-runoff
   mapping matrix via the Dataset API.
2. Call ``export_runoff_climatology`` to produce a catchment-level
   mean-runoff NetCDF (analogous to ``calc_outclm.F90``).
3. Call ``estimate_river_geometry`` to compute river width, height,
   depth, storage, and update bifurcation elevation
   (analogous to ``calc_rivwth.F90``).
"""

from datetime import datetime

import numpy as np
from netCDF4 import Dataset

from cmfgpu.datasets.daily_bin_dataset import DailyBinDataset
from cmfgpu.params import estimate_river_geometry


def main():
    ### Configuration Start ###
    resolution = "glb_15min"
    input_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/parameters.nc"

    # Pre-computed 365-day runoff climatology (360×180, float32, mm/day)
    # Single file → use time_to_key=constant so all 365 days read from one file
    runoff_clm_dir = "/home/eat/cmf_v420_pkg/map/data"
    runoff_clm_prefix = "ELSE_GPCC_dayclm-1981-2010"
    runoff_clm_suffix = ".one"
    runoff_mapping_file = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/runoff_mapping_bin.npz"
    runoff_shape = [180, 360]       # (ny, nx)
    unit_factor = 86400000          # mm/day → m/s

    # Output paths
    climatology_nc = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/runoff_clm.nc"
    output_nc = f"/home/eat/CaMa-Flood-GPU/inp/{resolution}/parameters_new.nc"
    ### Configuration End ###

    # ------------------------------------------------------------------
    # 1. Read catchment IDs from parameters.nc
    # ------------------------------------------------------------------
    with Dataset(input_file, "r") as ds:
        catchment_ids = np.asarray(ds.variables["catchment_id"][:]).astype(np.int64)

    # ------------------------------------------------------------------
    # 2. Build dataset and mapping matrix
    # ------------------------------------------------------------------
    dataset = DailyBinDataset(
        base_dir=runoff_clm_dir,
        shape=runoff_shape,
        start_date=datetime(2001, 1, 1),  # any non-leap year with 365 days
        end_date=datetime(2001, 12, 31),
        unit_factor=unit_factor,
        prefix=runoff_clm_prefix,
        suffix=runoff_clm_suffix,
        time_to_key=None,              # single file mode
    )

    local_runoff_matrix = dataset.build_local_runoff_matrix(
        runoff_mapping_file=runoff_mapping_file,
        desired_catchment_ids=catchment_ids,
        device="cpu",
    )

    # ------------------------------------------------------------------
    # 3. Export runoff climatology
    # ------------------------------------------------------------------
    dataset.export_runoff_climatology(
        out_path=climatology_nc,
        local_runoff_matrix=local_runoff_matrix,
        device="cpu",
    )

    # ------------------------------------------------------------------
    # 4. Estimate river geometry and write updated parameters
    # ------------------------------------------------------------------
    estimate_river_geometry(
        climatology_nc=climatology_nc,
        parameter_nc=input_file,
        output_nc=output_nc,
    )

    print(f"\nDone. New parameters written to: {output_nc}")


if __name__ == "__main__":
    main()
