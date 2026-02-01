"""
Script to generate runoff mapping tables for input datasets.
"""
from datetime import datetime

from cmfgpu.datasets.daily_bin_dataset import DailyBinDataset

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
    base_dir=runoff_base_dir,
    shape=runoff_shape,
    start_date=start_date,
    end_date=end_date,
)

dataset.generate_runoff_mapping_table(
    map_dir=map_dir,
    out_dir=out_dir,
    npz_file="runoff_mapping_bin.npz",
)

# from cmfgpu.datasets.daily_bin_dataset import NetCDFDataset
# dataset = NetCDFDataset(
#     base_dir="/home/eat/cmf_v420_pkg/inp/test_15min_nc",
#     start_date=datetime(2000, 1, 1),
#     end_date=datetime(2000, 12, 31),
#     prefix="e2o_ecmwf_wrr2_glob15_day_Runoff_",
#     suffix=".nc",
#     var_name="Runoff",
#     chunk_len=24,
# )
# dataset.generate_runoff_mapping_table(
#     map_dir=f"/home/eat/cmf_v420_pkg/map/{map_resolution}",
#     hires_map_tag="1min",
#     out_dir=f"/home/eat/CaMa-Flood-GPU/inp/{map_resolution}",
#     npz_file="runoff_mapping_nc.npz",
# )
