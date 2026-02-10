"""
Script to generate model parameters from CaMa-Flood map data.
"""


from cmfgpu.params.merit_map import MERITMap

print("=== Generating Map Parameters ===")

# --- Configuration Start ---
map_resolution = "glb_15min"
map_dir = f"/home/eat/cmf_v420_pkg/map/{map_resolution}"
out_dir = f"/home/eat/CaMa-Flood-GPU/inp/{map_resolution}"

# Optional files
bifori_file = f"{map_dir}/bifori.txt"
gauge_file = f"{map_dir}/GRDC_alloc.txt"

# Settings
target_gpus = 1
visualized = True
basin_use_file = False
# --- Configuration End ---

merit_map = MERITMap(
    map_dir=map_dir,
    out_dir=out_dir,
    bifori_file=bifori_file, # Set to None if not available
    gauge_file=gauge_file,   # Set to None if not available
    visualized=visualized,
    bif_levels_to_keep=5,
    basin_use_file=basin_use_file,
    target_gpus=target_gpus,
    out_file="parameters.nc",
)
merit_map.build_input()
