# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Script to generate runoff mapping tables for input datasets.
"""
from datetime import datetime

from hydroforge.io.datasets import DailyBinDataset

print("\n=== Generating Runoff Mapping Table ===")

# --- Configuration Start ---
map_resolution = "glb_15min"
map_dir = f"/home/eat/cmf_v420_pkg/map/{map_resolution}"
out_dir = f"/home/eat/CaMa-Flood-GPU/inp/{map_resolution}"

# Runoff data configuration
runoff_base_dir = "/home/eat/cmf_v420_pkg/inp/test_1deg/runoff"
runoff_shape = [180, 360] # [lat, lon]
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
)

dataset.generate_mapping_table(
    map_dir=map_dir,
    out_dir=out_dir,
    npz_file="runoff_mapping_bin.npz",
)

# from hydroforge.io.datasets import NetCDFDataset
# dataset = NetCDFDataset(
#     base_dir="/home/eat/cmf_v420_pkg/inp/test_15min_nc",
#     start_date=datetime(2000, 1, 1),
#     end_date=datetime(2000, 12, 31),
#     prefix="e2o_ecmwf_wrr2_glob15_day_Runoff_",
#     suffix=".nc",
#     var_name="Runoff",
#     chunk_len=24,
# )
# dataset.generate_mapping_table(
#     map_dir=f"/home/eat/cmf_v420_pkg/map/{map_resolution}",
#     out_dir=f"/home/eat/CaMa-Flood-GPU/inp/{map_resolution}",
#     npz_file="runoff_mapping_nc.npz",
# )
