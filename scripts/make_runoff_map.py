# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Script to generate runoff mapping tables for input datasets.
"""

from datetime import datetime, timedelta
from pathlib import Path

from hydroforge.data.datasets import DailyBinDataset, generate_mapping_table


def main():
    print("\n=== Generating Runoff Mapping Table ===")

    # --- Configuration Start ---
    map_resolution = "glb_15min"
    # Windows path example: map_dir = fr"C:\Users\YourName\cmf_v420_pkg\map\{map_resolution}"
    map_dir = f"/home/eat/cmf_v420_pkg/map/{map_resolution}"
    out_dir = f"/home/eat/CaMa-Flood-GPU/inp/{map_resolution}"

    # Runoff data configuration
    runoff_base_dir = "/home/eat/cmf_v420_pkg/inp/test_1deg/runoff"
    runoff_shape = (180, 360)  # (lat, lon)
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2000, 12, 31)
    # --- Configuration End ---

    dataset = DailyBinDataset(
        prefix="Roff____",
        suffix=".one",
        base_dir=runoff_base_dir,
        shape=runoff_shape,
        start_date=start_date,
        end_date=end_date,
        time_interval=timedelta(days=1),
        model_step=timedelta(days=1),
    )

    generate_mapping_table(dataset, map_dir, Path(out_dir) / "runoff_mapping_bin.npz")


if __name__ == "__main__":
    main()

# from hydroforge.data.datasets import NetCDFDataset
# dataset = NetCDFDataset(
#     base_dir="/home/eat/cmf_v420_pkg/inp/test_15min_nc",
#     start_date=datetime(2000, 1, 1),
#     end_date=datetime(2000, 12, 31),
#     time_interval=timedelta(days=1),
#     prefix="e2o_ecmwf_wrr2_glob15_day_Runoff_",
#     suffix=".nc",
#     var_name="Runoff",
#     chunk_len=24,
# )
# generate_mapping_table(
#     dataset,
#     f"/home/eat/cmf_v420_pkg/map/{map_resolution}",
#     f"/home/eat/CaMa-Flood-GPU/inp/{map_resolution}/runoff_mapping_nc.npz",
# )
