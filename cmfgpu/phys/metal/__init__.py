# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Native Metal backend for CaMa-Flood-GPU physics kernels (Apple Silicon).

HydroForge's Objective-C++ runtime compiles and specializes the ``.metal``
sources, then exposes launchers compatible with the unified kernel kwargs
convention. Fixed sub-step sequences may be captured into a persistent ICB.

Note: Metal only supports float32 storage.  Set ``precision='float32'``
and ``mixed_precision=False`` when using this backend.
"""

from pathlib import Path

from hydroforge.runtime.backend import make_metal_dispatcher

# ── Shader paths ──────────────────────────────────────────────────────────

_DIR = Path(__file__).parent

_OUTFLOW_TEMPLATE_DEFAULTS = {
    "HAS_BIFURCATION": False,
    "HAS_SEA_LEVEL": False,
    "HAS_RESERVOIR": False,
    "batched_river_manning": False,
    "batched_flood_manning": False,
    "batched_river_width": False,
    "batched_river_length": False,
    "batched_river_height": False,
    "batched_catchment_elevation": False,
}

_OUTFLOW_FUNCTION_CONSTANTS = {
    "HAS_BIFURCATION": 0,
    "HAS_SEA_LEVEL": 1,
}

_INFLOW_FUNCTION_CONSTANTS = {"HAS_RESERVOIR": 2}

_BATCHED_OUTFLOW_FUNCTION_CONSTANTS = {
    **_OUTFLOW_FUNCTION_CONSTANTS,
    "batched_river_manning": 3,
    "batched_flood_manning": 4,
    "batched_river_width": 5,
    "batched_river_length": 6,
    "batched_river_height": 7,
    "batched_catchment_elevation": 8,
}

# ── Outflow / Inflow ─────────────────────────────────────────────────────

compute_outflow_kernel = make_metal_dispatcher(
    _DIR / "outflow.metal", "compute_outflow",
    args=(
        "downstream_idx_ptr", "river_inflow_ptr", "river_outflow_ptr",
        "river_manning_ptr", "river_depth_ptr", "river_width_ptr",
        "river_length_ptr", "river_height_ptr", "river_storage_ptr",
        "flood_inflow_ptr", "flood_outflow_ptr", "flood_manning_ptr",
        "flood_depth_ptr", "protected_depth_ptr", "catchment_elevation_ptr",
        "downstream_distance_ptr", "flood_storage_ptr", "protected_storage_ptr",
        "river_cross_section_depth_ptr", "flood_cross_section_depth_ptr",
        "flood_cross_section_area_ptr", "global_bifurcation_outflow_ptr",
        "total_storage_ptr", "outgoing_storage_ptr",
        "water_surface_elevation_ptr", "protected_water_surface_elevation_ptr",
        "gravity", "time_step_ptr", "num_catchments",
        "sea_surface_elevation_ptr", "catchment_sea_level_idx_ptr",
    ),
    buffer_access={
        'downstream_idx_ptr': 'read',
        'river_inflow_ptr': 'read_write',
        'river_outflow_ptr': 'read_write',
        'river_manning_ptr': 'read',
        'river_depth_ptr': 'read',
        'river_width_ptr': 'read',
        'river_length_ptr': 'read',
        'river_height_ptr': 'read',
        'river_storage_ptr': 'read',
        'flood_inflow_ptr': 'read_write',
        'flood_outflow_ptr': 'read_write',
        'flood_manning_ptr': 'read',
        'flood_depth_ptr': 'read',
        'protected_depth_ptr': 'read',
        'catchment_elevation_ptr': 'read',
        'downstream_distance_ptr': 'read',
        'flood_storage_ptr': 'read',
        'protected_storage_ptr': 'read',
        'river_cross_section_depth_ptr': 'read_write',
        'flood_cross_section_depth_ptr': 'read_write',
        'flood_cross_section_area_ptr': 'read_write',
        'global_bifurcation_outflow_ptr': 'read_write',
        'total_storage_ptr': 'read_write',
        'outgoing_storage_ptr': 'read_write',
        'water_surface_elevation_ptr': 'read_write',
        'protected_water_surface_elevation_ptr': 'read_write',
        'time_step_ptr': 'read',
        'sea_surface_elevation_ptr': 'read',
        'catchment_sea_level_idx_ptr': 'read',
    },
    function_constants=_OUTFLOW_FUNCTION_CONSTANTS,
    scalar_types={"gravity": "float32", "num_catchments": "int32"},
    optional_buffers={
        "global_bifurcation_outflow_ptr": "HAS_BIFURCATION",
        "sea_surface_elevation_ptr": "HAS_SEA_LEVEL",
        "catchment_sea_level_idx_ptr": "HAS_SEA_LEVEL",
    },
    arg_defaults={**_OUTFLOW_TEMPLATE_DEFAULTS, "HAS_BIFURCATION": True},
)

compute_inflow_kernel = make_metal_dispatcher(
    _DIR / "outflow.metal", "compute_inflow",
    args=(
        "downstream_idx_ptr", "river_outflow_ptr", "flood_outflow_ptr",
        "river_storage_ptr", "flood_storage_ptr", "outgoing_storage_ptr",
        "river_inflow_ptr", "flood_inflow_ptr", "limit_rate_ptr",
        "reservoir_total_inflow_ptr", "is_reservoir_ptr",
        "num_catchments",
    ),
    buffer_access={
        'downstream_idx_ptr': 'read',
        'river_outflow_ptr': 'read_write',
        'flood_outflow_ptr': 'read_write',
        'river_storage_ptr': 'read',
        'flood_storage_ptr': 'read',
        'outgoing_storage_ptr': 'read',
        'river_inflow_ptr': 'read_write',
        'flood_inflow_ptr': 'read_write',
        'limit_rate_ptr': 'read_write',
        'reservoir_total_inflow_ptr': 'read_write',
        'is_reservoir_ptr': 'read',
    },
    function_constants=_INFLOW_FUNCTION_CONSTANTS,
    scalar_types={"num_catchments": "int32"},
    optional_buffers={
        "reservoir_total_inflow_ptr": "HAS_RESERVOIR",
        "is_reservoir_ptr": "HAS_RESERVOIR",
    },
    arg_defaults={**_OUTFLOW_TEMPLATE_DEFAULTS, "HAS_RESERVOIR": False},
)

compute_outflow_batched_kernel = make_metal_dispatcher(
    _DIR / "outflow.metal", "compute_outflow_batched",
    args=(
        "downstream_idx_ptr", "river_inflow_ptr", "river_outflow_ptr",
        "river_manning_ptr", "river_depth_ptr", "river_width_ptr",
        "river_length_ptr", "river_height_ptr", "river_storage_ptr",
        "flood_inflow_ptr", "flood_outflow_ptr", "flood_manning_ptr",
        "flood_depth_ptr", "protected_depth_ptr", "catchment_elevation_ptr",
        "downstream_distance_ptr", "flood_storage_ptr", "protected_storage_ptr",
        "river_cross_section_depth_ptr", "flood_cross_section_depth_ptr",
        "flood_cross_section_area_ptr", "global_bifurcation_outflow_ptr",
        "total_storage_ptr", "outgoing_storage_ptr",
        "water_surface_elevation_ptr", "protected_water_surface_elevation_ptr",
        "time_step_ptr", "_config_ptr",
        "sea_surface_elevation_ptr", "catchment_sea_level_idx_ptr",
        "num_sea_level_boundaries",
    ),
    buffer_access={
        'downstream_idx_ptr': 'read',
        'river_inflow_ptr': 'read_write',
        'river_outflow_ptr': 'read_write',
        'river_manning_ptr': 'read',
        'river_depth_ptr': 'read',
        'river_width_ptr': 'read',
        'river_length_ptr': 'read',
        'river_height_ptr': 'read',
        'river_storage_ptr': 'read',
        'flood_inflow_ptr': 'read_write',
        'flood_outflow_ptr': 'read_write',
        'flood_manning_ptr': 'read',
        'flood_depth_ptr': 'read',
        'protected_depth_ptr': 'read',
        'catchment_elevation_ptr': 'read',
        'downstream_distance_ptr': 'read',
        'flood_storage_ptr': 'read',
        'protected_storage_ptr': 'read',
        'river_cross_section_depth_ptr': 'read_write',
        'flood_cross_section_depth_ptr': 'read_write',
        'flood_cross_section_area_ptr': 'read_write',
        'global_bifurcation_outflow_ptr': 'read_write',
        'total_storage_ptr': 'read_write',
        'outgoing_storage_ptr': 'read_write',
        'water_surface_elevation_ptr': 'read_write',
        'protected_water_surface_elevation_ptr': 'read_write',
        'time_step_ptr': 'read',
        '_config_ptr': 'read',
        'sea_surface_elevation_ptr': 'read',
        'catchment_sea_level_idx_ptr': 'read',
    },
    size_key=("num_catchments", "num_trials"),
    packed_args={
        "_config_ptr": ("<fii", ["gravity", "num_catchments", "num_trials"]),
    },
    function_constants=_BATCHED_OUTFLOW_FUNCTION_CONSTANTS,
    scalar_types={"num_sea_level_boundaries": "int32"},
    optional_buffers={
        "global_bifurcation_outflow_ptr": "HAS_BIFURCATION",
        "sea_surface_elevation_ptr": "HAS_SEA_LEVEL",
        "catchment_sea_level_idx_ptr": "HAS_SEA_LEVEL",
    },
    arg_defaults={**_OUTFLOW_TEMPLATE_DEFAULTS,
        "HAS_BIFURCATION": True,
        "num_sea_level_boundaries": 0,
    },
)

compute_inflow_batched_kernel = make_metal_dispatcher(
    _DIR / "outflow.metal", "compute_inflow_batched",
    args=(
        "downstream_idx_ptr", "river_outflow_ptr", "flood_outflow_ptr",
        "river_storage_ptr", "flood_storage_ptr", "outgoing_storage_ptr",
        "river_inflow_ptr", "flood_inflow_ptr", "limit_rate_ptr",
        "reservoir_total_inflow_ptr", "is_reservoir_ptr",
        "num_catchments", "num_trials",
    ),
    buffer_access={
        'downstream_idx_ptr': 'read',
        'river_outflow_ptr': 'read_write',
        'flood_outflow_ptr': 'read_write',
        'river_storage_ptr': 'read',
        'flood_storage_ptr': 'read',
        'outgoing_storage_ptr': 'read',
        'river_inflow_ptr': 'read_write',
        'flood_inflow_ptr': 'read_write',
        'limit_rate_ptr': 'read_write',
        'reservoir_total_inflow_ptr': 'read_write',
        'is_reservoir_ptr': 'read',
    },
    size_key=("num_catchments", "num_trials"),
    function_constants=_INFLOW_FUNCTION_CONSTANTS,
    scalar_types={"num_catchments": "int32", "num_trials": "int32"},
    optional_buffers={
        "reservoir_total_inflow_ptr": "HAS_RESERVOIR",
        "is_reservoir_ptr": "HAS_RESERVOIR",
    },
    arg_defaults={**_OUTFLOW_TEMPLATE_DEFAULTS, "HAS_RESERVOIR": False},
)

# ── Flood stage ───────────────────────────────────────────────────────────

compute_flood_stage_kernel = make_metal_dispatcher(
    _DIR / "storage.metal", "compute_flood_stage",
    args=(
        "river_inflow_ptr", "flood_inflow_ptr", "river_outflow_ptr",
        "flood_outflow_ptr", "global_bifurcation_outflow_ptr", "runoff_ptr",
        "outgoing_storage_ptr", "river_storage_ptr", "flood_storage_ptr",
        "protected_storage_ptr", "river_depth_ptr", "flood_depth_ptr",
        "protected_depth_ptr", "flood_fraction_ptr", "river_height_ptr",
        "flood_depth_table_ptr", "catchment_area_ptr", "river_width_ptr",
        "river_length_ptr", "time_step_ptr", "num_catchments",
        "inflow_ptr", "catchment_inflow_idx_ptr",
    ),
    buffer_access={
        'river_inflow_ptr': 'read',
        'flood_inflow_ptr': 'read',
        'river_outflow_ptr': 'read',
        'flood_outflow_ptr': 'read',
        'global_bifurcation_outflow_ptr': 'read',
        'runoff_ptr': 'read',
        'outgoing_storage_ptr': 'read_write',
        'river_storage_ptr': 'read_write',
        'flood_storage_ptr': 'read_write',
        'protected_storage_ptr': 'read_write',
        'river_depth_ptr': 'read_write',
        'flood_depth_ptr': 'read_write',
        'protected_depth_ptr': 'read_write',
        'flood_fraction_ptr': 'read_write',
        'river_height_ptr': 'read',
        'flood_depth_table_ptr': 'read',
        'catchment_area_ptr': 'read',
        'river_width_ptr': 'read',
        'river_length_ptr': 'read',
        'time_step_ptr': 'read',
        'inflow_ptr': 'read',
        'catchment_inflow_idx_ptr': 'read',
    },
    template_vars={"__NUM_FLOOD_LEVELS__": "num_flood_levels"},
    function_constants={"HAS_BIFURCATION": 0, "HAS_INFLOW": 1},
    scalar_types={"num_catchments": "int32"},
    optional_buffers={
        "global_bifurcation_outflow_ptr": "HAS_BIFURCATION",
        "inflow_ptr": "HAS_INFLOW",
        "catchment_inflow_idx_ptr": "HAS_INFLOW",
    },
    arg_defaults={"HAS_BIFURCATION": True, "HAS_INFLOW": False},
)

compute_flood_stage_log_kernel = make_metal_dispatcher(
    _DIR / "storage.metal", "compute_flood_stage_log",
    args=(
        "river_inflow_ptr", "flood_inflow_ptr", "river_outflow_ptr",
        "flood_outflow_ptr", "global_bifurcation_outflow_ptr", "runoff_ptr",
        "outgoing_storage_ptr", "river_storage_ptr", "flood_storage_ptr",
        "protected_storage_ptr", "river_depth_ptr", "flood_depth_ptr",
        "protected_depth_ptr", "flood_fraction_ptr", "river_height_ptr",
        "flood_depth_table_ptr", "catchment_area_ptr", "river_width_ptr",
        "river_length_ptr", "is_levee_ptr",
        "log_sums_ptr",
        "time_step_ptr", "num_catchments",
        "current_step_ptr", "log_buffer_size",
        "inflow_ptr", "catchment_inflow_idx_ptr",
    ),
    buffer_access={
        'river_inflow_ptr': 'read',
        'flood_inflow_ptr': 'read',
        'river_outflow_ptr': 'read',
        'flood_outflow_ptr': 'read',
        'global_bifurcation_outflow_ptr': 'read',
        'runoff_ptr': 'read',
        'outgoing_storage_ptr': 'read_write',
        'river_storage_ptr': 'read_write',
        'flood_storage_ptr': 'read_write',
        'protected_storage_ptr': 'read_write',
        'river_depth_ptr': 'read_write',
        'flood_depth_ptr': 'read_write',
        'protected_depth_ptr': 'read_write',
        'flood_fraction_ptr': 'read_write',
        'river_height_ptr': 'read',
        'flood_depth_table_ptr': 'read',
        'catchment_area_ptr': 'read',
        'river_width_ptr': 'read',
        'river_length_ptr': 'read',
        'is_levee_ptr': 'read_write',
        'log_sums_ptr': 'read_write',
        'time_step_ptr': 'read',
        'current_step_ptr': 'read',
        'inflow_ptr': 'read',
        'catchment_inflow_idx_ptr': 'read',
    },
    template_vars={"__NUM_FLOOD_LEVELS__": "num_flood_levels"},
    function_constants={
        "HAS_BIFURCATION": 0, "HAS_INFLOW": 1, "HAS_LEVEE": 2,
    },
    scalar_types={"num_catchments": "int32", "log_buffer_size": "int32"},
    optional_buffers={
        "global_bifurcation_outflow_ptr": "HAS_BIFURCATION",
        "inflow_ptr": "HAS_INFLOW",
        "catchment_inflow_idx_ptr": "HAS_INFLOW",
        "is_levee_ptr": "HAS_LEVEE",
    },
    arg_defaults={"HAS_BIFURCATION": True, "HAS_LEVEE": False, "HAS_INFLOW": False},
)
compute_flood_stage_batched_kernel = make_metal_dispatcher(
    _DIR / "storage.metal", "compute_flood_stage_batched",
    args=(
        "river_inflow_ptr", "flood_inflow_ptr", "river_outflow_ptr",
        "flood_outflow_ptr", "global_bifurcation_outflow_ptr", "runoff_ptr",
        "outgoing_storage_ptr", "river_storage_ptr", "flood_storage_ptr",
        "protected_storage_ptr", "river_depth_ptr", "flood_depth_ptr",
        "protected_depth_ptr", "flood_fraction_ptr", "river_height_ptr",
        "flood_depth_table_ptr", "catchment_area_ptr", "river_width_ptr",
        "river_length_ptr", "time_step_ptr", "num_catchments", "num_trials",
        "inflow_ptr", "catchment_inflow_idx_ptr",
        "num_inflow_gauges",
    ),
    buffer_access={
        'river_inflow_ptr': 'read',
        'flood_inflow_ptr': 'read',
        'river_outflow_ptr': 'read',
        'flood_outflow_ptr': 'read',
        'global_bifurcation_outflow_ptr': 'read',
        'runoff_ptr': 'read',
        'outgoing_storage_ptr': 'read_write',
        'river_storage_ptr': 'read_write',
        'flood_storage_ptr': 'read_write',
        'protected_storage_ptr': 'read_write',
        'river_depth_ptr': 'read_write',
        'flood_depth_ptr': 'read_write',
        'protected_depth_ptr': 'read_write',
        'flood_fraction_ptr': 'read_write',
        'river_height_ptr': 'read',
        'flood_depth_table_ptr': 'read',
        'catchment_area_ptr': 'read',
        'river_width_ptr': 'read',
        'river_length_ptr': 'read',
        'time_step_ptr': 'read',
        'inflow_ptr': 'read',
        'catchment_inflow_idx_ptr': 'read',
    },
    template_vars={"__NUM_FLOOD_LEVELS__": "num_flood_levels"},
    function_constants={
        "HAS_BIFURCATION": 0, "HAS_INFLOW": 1,
        "batched_runoff": 3, "batched_river_height": 4,
        "batched_flood_depth_table": 5, "batched_catchment_area": 6,
        "batched_river_width": 7, "batched_river_length": 8,
    },
    scalar_types={
        "num_catchments": "int32", "num_trials": "int32",
        "num_inflow_gauges": "int32",
    },
    optional_buffers={
        "global_bifurcation_outflow_ptr": "HAS_BIFURCATION",
        "inflow_ptr": "HAS_INFLOW",
        "catchment_inflow_idx_ptr": "HAS_INFLOW",
    },
    arg_defaults={
        "HAS_BIFURCATION": True, "HAS_INFLOW": False,
        "batched_runoff": False, "batched_river_height": False,
        "batched_flood_depth_table": False,
        "batched_catchment_area": False, "batched_river_width": False,
        "batched_river_length": False, "num_inflow_gauges": 0,
    },
)

# ── Adaptive time step ────────────────────────────────────────────────────

compute_adaptive_time_step_kernel = make_metal_dispatcher(
    _DIR / "adaptive_time.metal", "compute_adaptive_time_step",
    args=(
        "river_depth_ptr", "downstream_distance_ptr", "is_dam_related_ptr",
        "max_sub_steps_ptr", "time_step", "adaptive_time_factor", "gravity",
        "num_catchments",
    ),
    buffer_access={
        'river_depth_ptr': 'read',
        'downstream_distance_ptr': 'read',
        'is_dam_related_ptr': 'read',
        'max_sub_steps_ptr': 'read_write',
    },
    function_constants={"HAS_RESERVOIR": 0},
    scalar_types={
        "time_step": "float32", "adaptive_time_factor": "float32",
        "gravity": "float32", "num_catchments": "int32",
    },
    optional_buffers={"is_dam_related_ptr": "HAS_RESERVOIR"},
    arg_defaults={"HAS_RESERVOIR": False},
)

compute_adaptive_time_step_batched_kernel = make_metal_dispatcher(
    _DIR / "adaptive_time.metal", "compute_adaptive_time_step_batched",
    args=(
        "river_depth_ptr", "downstream_distance_ptr", "is_dam_related_ptr",
        "max_sub_steps_ptr", "time_step", "adaptive_time_factor", "gravity",
        "num_catchments", "num_trials",
    ),
    buffer_access={
        'river_depth_ptr': 'read',
        'downstream_distance_ptr': 'read',
        'is_dam_related_ptr': 'read',
        'max_sub_steps_ptr': 'read_write',
    },
    size_key=("num_catchments", "num_trials"),
    function_constants={
        "HAS_RESERVOIR": 0, "batched_downstream_distance": 1,
    },
    scalar_types={
        "time_step": "float32", "adaptive_time_factor": "float32",
        "gravity": "float32", "num_catchments": "int32",
        "num_trials": "int32",
    },
    optional_buffers={"is_dam_related_ptr": "HAS_RESERVOIR"},
    arg_defaults={
        "HAS_RESERVOIR": False, "batched_downstream_distance": False,
    },
)

# ── Bifurcation ───────────────────────────────────────────────────────────

compute_bifurcation_outflow_kernel = make_metal_dispatcher(
    _DIR / "bifurcation.metal", "compute_bifurcation_outflow",
    args=(
        "bifurcation_catchment_idx_ptr", "bifurcation_downstream_idx_ptr",
        "bifurcation_manning_ptr", "bifurcation_outflow_ptr",
        "bifurcation_width_ptr", "bifurcation_length_ptr",
        "bifurcation_elevation_ptr", "bifurcation_cross_section_depth_ptr",
        "water_surface_elevation_ptr", "total_storage_ptr",
        "outgoing_storage_ptr", "gravity", "time_step_ptr",
        "num_bifurcation_paths",
    ),
    buffer_access={
        'bifurcation_catchment_idx_ptr': 'read',
        'bifurcation_downstream_idx_ptr': 'read',
        'bifurcation_manning_ptr': 'read',
        'bifurcation_outflow_ptr': 'read_write',
        'bifurcation_width_ptr': 'read',
        'bifurcation_length_ptr': 'read',
        'bifurcation_elevation_ptr': 'read',
        'bifurcation_cross_section_depth_ptr': 'read_write',
        'water_surface_elevation_ptr': 'read',
        'total_storage_ptr': 'read',
        'outgoing_storage_ptr': 'read_write',
        'time_step_ptr': 'read',
    },
    scalar_types={'gravity': 'float32', 'num_bifurcation_paths': 'int32'},
    size_key="num_bifurcation_paths",
    template_vars={"__NUM_BIF_LEVELS__": "num_bifurcation_levels"},
)

compute_bifurcation_inflow_kernel = make_metal_dispatcher(
    _DIR / "bifurcation.metal", "compute_bifurcation_inflow",
    args=(
        "bifurcation_catchment_idx_ptr", "bifurcation_downstream_idx_ptr",
        "limit_rate_ptr", "bifurcation_outflow_ptr",
        "global_bifurcation_outflow_ptr", "num_bifurcation_paths",
    ),
    buffer_access={
        'bifurcation_catchment_idx_ptr': 'read',
        'bifurcation_downstream_idx_ptr': 'read',
        'limit_rate_ptr': 'read',
        'bifurcation_outflow_ptr': 'read_write',
        'global_bifurcation_outflow_ptr': 'read_write',
    },
    scalar_types={'num_bifurcation_paths': 'int32'},
    size_key="num_bifurcation_paths",
    template_vars={"__NUM_BIF_LEVELS__": "num_bifurcation_levels"},
)

compute_bifurcation_outflow_batched_kernel = make_metal_dispatcher(
    _DIR / "bifurcation.metal", "compute_bifurcation_outflow_batched",
    args=(
        "bifurcation_catchment_idx_ptr", "bifurcation_downstream_idx_ptr",
        "bifurcation_manning_ptr", "bifurcation_outflow_ptr",
        "bifurcation_width_ptr", "bifurcation_length_ptr",
        "bifurcation_elevation_ptr", "bifurcation_cross_section_depth_ptr",
        "water_surface_elevation_ptr", "total_storage_ptr",
        "outgoing_storage_ptr", "gravity", "time_step_ptr",
        "num_bifurcation_paths", "num_catchments", "num_trials",
    ),
    buffer_access={
        'bifurcation_catchment_idx_ptr': 'read',
        'bifurcation_downstream_idx_ptr': 'read',
        'bifurcation_manning_ptr': 'read',
        'bifurcation_outflow_ptr': 'read_write',
        'bifurcation_width_ptr': 'read',
        'bifurcation_length_ptr': 'read',
        'bifurcation_elevation_ptr': 'read',
        'bifurcation_cross_section_depth_ptr': 'read_write',
        'water_surface_elevation_ptr': 'read',
        'total_storage_ptr': 'read',
        'outgoing_storage_ptr': 'read_write',
        'time_step_ptr': 'read',
    },
    size_key=("num_bifurcation_paths", "num_trials"),
    template_vars={"__NUM_BIF_LEVELS__": "num_bifurcation_levels"},
    function_constants={
        "batched_bifurcation_manning": 0,
        "batched_bifurcation_width": 1,
        "batched_bifurcation_length": 2,
        "batched_bifurcation_elevation": 3,
    },
    scalar_types={
        "gravity": "float32", "num_bifurcation_paths": "int32",
        "num_catchments": "int32", "num_trials": "int32",
    },
    arg_defaults={
        "batched_bifurcation_manning": False,
        "batched_bifurcation_width": False,
        "batched_bifurcation_length": False,
        "batched_bifurcation_elevation": False,
    },
)

compute_bifurcation_inflow_batched_kernel = make_metal_dispatcher(
    _DIR / "bifurcation.metal", "compute_bifurcation_inflow_batched",
    args=(
        "bifurcation_catchment_idx_ptr", "bifurcation_downstream_idx_ptr",
        "limit_rate_ptr", "bifurcation_outflow_ptr",
        "global_bifurcation_outflow_ptr",
        "num_bifurcation_paths", "num_catchments", "num_trials",
    ),
    buffer_access={
        'bifurcation_catchment_idx_ptr': 'read',
        'bifurcation_downstream_idx_ptr': 'read',
        'limit_rate_ptr': 'read',
        'bifurcation_outflow_ptr': 'read_write',
        'global_bifurcation_outflow_ptr': 'read_write',
    },
    scalar_types={'num_bifurcation_paths': 'int32', 'num_catchments': 'int32', 'num_trials': 'int32'},
    size_key=("num_bifurcation_paths", "num_trials"),
    template_vars={"__NUM_BIF_LEVELS__": "num_bifurcation_levels"},
)

# ── Reservoir ─────────────────────────────────────────────────────────────

compute_reservoir_outflow_kernel = make_metal_dispatcher(
    _DIR / "reservoir.metal", "compute_reservoir_outflow",
    args=(
        "reservoir_catchment_idx_ptr", "downstream_idx_ptr",
        "reservoir_total_inflow_ptr", "river_outflow_ptr",
        "flood_outflow_ptr", "river_storage_ptr", "flood_storage_ptr",
        "conservation_volume_ptr", "emergency_volume_ptr",
        "adjustment_volume_ptr", "normal_outflow_ptr",
        "adjustment_outflow_ptr", "flood_control_outflow_ptr",
        "runoff_ptr", "total_storage_ptr", "outgoing_storage_ptr",
        "time_step_ptr", "num_reservoirs",
    ),
    buffer_access={
        'reservoir_catchment_idx_ptr': 'read',
        'downstream_idx_ptr': 'read',
        'reservoir_total_inflow_ptr': 'read_write',
        'river_outflow_ptr': 'read_write',
        'flood_outflow_ptr': 'read_write',
        'river_storage_ptr': 'read',
        'flood_storage_ptr': 'read',
        'conservation_volume_ptr': 'read',
        'emergency_volume_ptr': 'read',
        'adjustment_volume_ptr': 'read',
        'normal_outflow_ptr': 'read',
        'adjustment_outflow_ptr': 'read',
        'flood_control_outflow_ptr': 'read',
        'runoff_ptr': 'read',
        'total_storage_ptr': 'read_write',
        'outgoing_storage_ptr': 'read_write',
        'time_step_ptr': 'read',
    },
    scalar_types={'num_reservoirs': 'int32'},
    size_key="num_reservoirs",
)

compute_reservoir_outflow_batched_kernel = make_metal_dispatcher(
    _DIR / "reservoir.metal", "compute_reservoir_outflow_batched",
    args=(
        "reservoir_catchment_idx_ptr", "downstream_idx_ptr",
        "reservoir_total_inflow_ptr", "river_outflow_ptr",
        "flood_outflow_ptr", "river_storage_ptr", "flood_storage_ptr",
        "conservation_volume_ptr", "emergency_volume_ptr",
        "adjustment_volume_ptr", "normal_outflow_ptr",
        "adjustment_outflow_ptr", "flood_control_outflow_ptr",
        "runoff_ptr", "total_storage_ptr", "outgoing_storage_ptr",
        "time_step_ptr", "num_reservoirs",
        "num_catchments", "num_trials",
    ),
    buffer_access={
        'reservoir_catchment_idx_ptr': 'read',
        'downstream_idx_ptr': 'read',
        'reservoir_total_inflow_ptr': 'read_write',
        'river_outflow_ptr': 'read_write',
        'flood_outflow_ptr': 'read_write',
        'river_storage_ptr': 'read',
        'flood_storage_ptr': 'read',
        'conservation_volume_ptr': 'read',
        'emergency_volume_ptr': 'read',
        'adjustment_volume_ptr': 'read',
        'normal_outflow_ptr': 'read',
        'adjustment_outflow_ptr': 'read',
        'flood_control_outflow_ptr': 'read',
        'runoff_ptr': 'read',
        'total_storage_ptr': 'read_write',
        'outgoing_storage_ptr': 'read_write',
        'time_step_ptr': 'read',
    },
    scalar_types={'num_trials': 'int32', 'num_catchments': 'int32', 'num_reservoirs': 'int32'},
    size_key=("num_reservoirs", "num_trials"),
)

# ── Levee ──────────────────────────────────────────────────────────────────

compute_levee_stage_kernel = make_metal_dispatcher(
    _DIR / "levee.metal", "compute_levee_stage",
    args=(
        "levee_catchment_idx_ptr", "river_storage_ptr", "flood_storage_ptr",
        "protected_storage_ptr", "river_depth_ptr", "flood_depth_ptr",
        "protected_depth_ptr", "river_height_ptr", "flood_depth_table_ptr",
        "catchment_area_ptr", "river_width_ptr", "river_length_ptr",
        "levee_base_height_ptr", "levee_crown_height_ptr",
        "levee_fraction_ptr", "flood_fraction_ptr", "num_levees",
    ),
    buffer_access={
        'levee_catchment_idx_ptr': 'read',
        'river_storage_ptr': 'read_write',
        'flood_storage_ptr': 'read_write',
        'protected_storage_ptr': 'read_write',
        'river_depth_ptr': 'read_write',
        'flood_depth_ptr': 'read_write',
        'protected_depth_ptr': 'read_write',
        'river_height_ptr': 'read',
        'flood_depth_table_ptr': 'read',
        'catchment_area_ptr': 'read',
        'river_width_ptr': 'read',
        'river_length_ptr': 'read',
        'levee_base_height_ptr': 'read',
        'levee_crown_height_ptr': 'read',
        'levee_fraction_ptr': 'read',
        'flood_fraction_ptr': 'read_write',
    },
    scalar_types={'num_levees': 'int32'},
    size_key="num_levees",
    template_vars={"__NUM_FLOOD_LEVELS__": "num_flood_levels"},
)

compute_levee_stage_log_kernel = make_metal_dispatcher(
    _DIR / "levee.metal", "compute_levee_stage_log",
    args=(
        "levee_catchment_idx_ptr", "river_storage_ptr", "flood_storage_ptr",
        "protected_storage_ptr", "river_depth_ptr", "flood_depth_ptr",
        "protected_depth_ptr", "river_height_ptr", "flood_depth_table_ptr",
        "catchment_area_ptr", "river_width_ptr", "river_length_ptr",
        "levee_base_height_ptr", "levee_crown_height_ptr",
        "levee_fraction_ptr", "flood_fraction_ptr",
        "log_sums_ptr",
        "num_levees", "current_step_ptr", "log_buffer_size",
    ),
    buffer_access={
        'levee_catchment_idx_ptr': 'read',
        'river_storage_ptr': 'read_write',
        'flood_storage_ptr': 'read_write',
        'protected_storage_ptr': 'read_write',
        'river_depth_ptr': 'read_write',
        'flood_depth_ptr': 'read_write',
        'protected_depth_ptr': 'read_write',
        'river_height_ptr': 'read',
        'flood_depth_table_ptr': 'read',
        'catchment_area_ptr': 'read',
        'river_width_ptr': 'read',
        'river_length_ptr': 'read',
        'levee_base_height_ptr': 'read',
        'levee_crown_height_ptr': 'read',
        'levee_fraction_ptr': 'read',
        'flood_fraction_ptr': 'read_write',
        'log_sums_ptr': 'read_write',
        'current_step_ptr': 'read',
    },
    scalar_types={'log_buffer_size': 'int32', 'num_levees': 'int32'},
    size_key="num_levees",
    template_vars={"__NUM_FLOOD_LEVELS__": "num_flood_levels"},
)

compute_levee_bifurcation_outflow_kernel = make_metal_dispatcher(
    _DIR / "levee_bifurcation_outflow.metal", "compute_levee_bifurcation_outflow",
    args=(
        "bifurcation_catchment_idx_ptr", "bifurcation_downstream_idx_ptr",
        "bifurcation_manning_ptr", "bifurcation_outflow_ptr",
        "bifurcation_width_ptr", "bifurcation_length_ptr",
        "bifurcation_elevation_ptr", "bifurcation_cross_section_depth_ptr",
        "water_surface_elevation_ptr", "protected_water_surface_elevation_ptr",
        "total_storage_ptr", "outgoing_storage_ptr",
        "gravity", "time_step_ptr", "num_bifurcation_paths",
    ),
    buffer_access={
        'bifurcation_catchment_idx_ptr': 'read',
        'bifurcation_downstream_idx_ptr': 'read',
        'bifurcation_manning_ptr': 'read',
        'bifurcation_outflow_ptr': 'read_write',
        'bifurcation_width_ptr': 'read',
        'bifurcation_length_ptr': 'read',
        'bifurcation_elevation_ptr': 'read',
        'bifurcation_cross_section_depth_ptr': 'read_write',
        'water_surface_elevation_ptr': 'read',
        'protected_water_surface_elevation_ptr': 'read',
        'total_storage_ptr': 'read',
        'outgoing_storage_ptr': 'read_write',
        'time_step_ptr': 'read',
    },
    scalar_types={'gravity': 'float32', 'num_bifurcation_paths': 'int32'},
    size_key="num_bifurcation_paths",
    template_vars={"__NUM_BIF_LEVELS__": "num_bifurcation_levels"},
)

compute_levee_stage_batched_kernel = make_metal_dispatcher(
    _DIR / "levee.metal", "compute_levee_stage_batched",
    args=(
        "levee_catchment_idx_ptr", "river_storage_ptr", "flood_storage_ptr",
        "protected_storage_ptr", "river_depth_ptr", "flood_depth_ptr",
        "protected_depth_ptr", "river_height_ptr", "flood_depth_table_ptr",
        "catchment_area_ptr", "river_width_ptr", "river_length_ptr",
        "levee_base_height_ptr", "levee_crown_height_ptr",
        "levee_fraction_ptr", "flood_fraction_ptr",
        "num_levees", "num_catchments", "num_trials",
    ),
    buffer_access={
        'levee_catchment_idx_ptr': 'read',
        'river_storage_ptr': 'read_write',
        'flood_storage_ptr': 'read_write',
        'protected_storage_ptr': 'read_write',
        'river_depth_ptr': 'read_write',
        'flood_depth_ptr': 'read_write',
        'protected_depth_ptr': 'read_write',
        'river_height_ptr': 'read',
        'flood_depth_table_ptr': 'read',
        'catchment_area_ptr': 'read',
        'river_width_ptr': 'read',
        'river_length_ptr': 'read',
        'levee_base_height_ptr': 'read',
        'levee_crown_height_ptr': 'read',
        'levee_fraction_ptr': 'read',
        'flood_fraction_ptr': 'read_write',
    },
    size_key="num_levees",
    template_vars={"__NUM_FLOOD_LEVELS__": "num_flood_levels"},
    function_constants={
        "batched_river_length": 0, "batched_river_width": 1,
        "batched_river_height": 2, "batched_catchment_area": 3,
        "batched_levee_crown_height": 4, "batched_levee_fraction": 5,
        "batched_levee_base_height": 6,
        "batched_flood_depth_table": 7,
    },
    scalar_types={
        "num_levees": "int32", "num_catchments": "int32",
        "num_trials": "int32",
    },
    arg_defaults={
        "batched_river_length": False, "batched_river_width": False,
        "batched_river_height": False, "batched_catchment_area": False,
        "batched_levee_crown_height": False,
        "batched_levee_fraction": False,
        "batched_levee_base_height": False,
        "batched_flood_depth_table": False,
    },
)

compute_levee_bifurcation_outflow_batched_kernel = make_metal_dispatcher(
    _DIR / "levee_bifurcation_outflow.metal", "compute_levee_bifurcation_outflow_batched",
    args=(
        "bifurcation_catchment_idx_ptr", "bifurcation_downstream_idx_ptr",
        "bifurcation_manning_ptr", "bifurcation_outflow_ptr",
        "bifurcation_width_ptr", "bifurcation_length_ptr",
        "bifurcation_elevation_ptr", "bifurcation_cross_section_depth_ptr",
        "water_surface_elevation_ptr", "protected_water_surface_elevation_ptr",
        "total_storage_ptr", "outgoing_storage_ptr",
        "gravity", "time_step_ptr", "num_bifurcation_paths",
        "num_catchments", "num_trials",
    ),
    buffer_access={
        'bifurcation_catchment_idx_ptr': 'read',
        'bifurcation_downstream_idx_ptr': 'read',
        'bifurcation_manning_ptr': 'read',
        'bifurcation_outflow_ptr': 'read_write',
        'bifurcation_width_ptr': 'read',
        'bifurcation_length_ptr': 'read',
        'bifurcation_elevation_ptr': 'read',
        'bifurcation_cross_section_depth_ptr': 'read_write',
        'water_surface_elevation_ptr': 'read',
        'protected_water_surface_elevation_ptr': 'read',
        'total_storage_ptr': 'read',
        'outgoing_storage_ptr': 'read_write',
        'time_step_ptr': 'read',
    },
    size_key=("num_bifurcation_paths", "num_trials"),
    template_vars={"__NUM_BIF_LEVELS__": "num_bifurcation_levels"},
    function_constants={
        "batched_bifurcation_manning": 0,
        "batched_bifurcation_width": 1,
        "batched_bifurcation_length": 2,
        "batched_bifurcation_elevation": 3,
    },
    scalar_types={
        "gravity": "float32", "num_bifurcation_paths": "int32",
        "num_catchments": "int32", "num_trials": "int32",
    },
    arg_defaults={
        "batched_bifurcation_manning": False,
        "batched_bifurcation_width": False,
        "batched_bifurcation_length": False,
        "batched_bifurcation_elevation": False,
    },
)
