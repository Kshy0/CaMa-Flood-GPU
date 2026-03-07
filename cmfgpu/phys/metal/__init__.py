# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Metal shader backend for CaMa-Flood-GPU physics kernels (Apple Silicon).

JIT-compiles ``.metal`` MSL sources in the ``metal/`` subdirectory via
``torch.mps.compile_shader()`` on first use, then exposes each launcher
as a callable compatible with the unified hydroforge kwargs convention.

Note: Metal only supports float32 storage.  Set ``precision='float32'``
and ``mixed_precision=False`` when using this backend.
"""

from pathlib import Path

from hydroforge.runtime.backend import make_metal_dispatcher

# ── Shader paths ──────────────────────────────────────────────────────────

_DIR = Path(__file__).parent

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
        "gravity", "time_step", "num_catchments", "HAS_BIFURCATION",
    ),
    arg_defaults={"HAS_BIFURCATION": True},
)

compute_inflow_kernel = make_metal_dispatcher(
    _DIR / "outflow.metal", "compute_inflow",
    args=(
        "downstream_idx_ptr", "river_outflow_ptr", "flood_outflow_ptr",
        "river_storage_ptr", "flood_storage_ptr", "outgoing_storage_ptr",
        "river_inflow_ptr", "flood_inflow_ptr", "limit_rate_ptr",
        "reservoir_total_inflow_ptr", "is_reservoir_ptr",
        "num_catchments", "HAS_RESERVOIR",
    ),
    arg_defaults={"HAS_RESERVOIR": False},
)

compute_outflow_batched_kernel = None
compute_inflow_batched_kernel = None

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
        "river_length_ptr", "time_step", "num_catchments", "HAS_BIFURCATION",
    ),
    template_vars={"__NUM_FLOOD_LEVELS__": "num_flood_levels"},
    arg_defaults={"HAS_BIFURCATION": True},
)

compute_flood_stage_log_kernel = None
compute_flood_stage_batched_kernel = None

# ── Adaptive time step ────────────────────────────────────────────────────

compute_adaptive_time_step_kernel = make_metal_dispatcher(
    _DIR / "adaptive_time.metal", "compute_adaptive_time_step",
    args=(
        "river_depth_ptr", "downstream_distance_ptr", "is_dam_related_ptr",
        "max_sub_steps_ptr", "time_step", "adaptive_time_factor", "gravity",
        "num_catchments", "HAS_RESERVOIR",
    ),
    arg_defaults={"HAS_RESERVOIR": False},
)

compute_adaptive_time_step_batched_kernel = None

# ── Bifurcation ───────────────────────────────────────────────────────────

compute_bifurcation_outflow_kernel = make_metal_dispatcher(
    _DIR / "bifurcation_outflow.metal", "compute_bifurcation_outflow",
    args=(
        "bifurcation_catchment_idx_ptr", "bifurcation_downstream_idx_ptr",
        "bifurcation_manning_ptr", "bifurcation_outflow_ptr",
        "bifurcation_width_ptr", "bifurcation_length_ptr",
        "bifurcation_elevation_ptr", "bifurcation_cross_section_depth_ptr",
        "water_surface_elevation_ptr", "total_storage_ptr",
        "outgoing_storage_ptr", "gravity", "time_step",
        "num_bifurcation_paths",
    ),
    size_key="num_bifurcation_paths",
    template_vars={"__NUM_BIF_LEVELS__": "num_bifurcation_levels"},
)

compute_bifurcation_inflow_kernel = make_metal_dispatcher(
    _DIR / "bifurcation_inflow.metal", "compute_bifurcation_inflow",
    args=(
        "bifurcation_catchment_idx_ptr", "bifurcation_downstream_idx_ptr",
        "limit_rate_ptr", "bifurcation_outflow_ptr",
        "global_bifurcation_outflow_ptr", "num_bifurcation_paths",
    ),
    size_key="num_bifurcation_paths",
    template_vars={"__NUM_BIF_LEVELS__": "num_bifurcation_levels"},
)

compute_bifurcation_outflow_batched_kernel = None
compute_bifurcation_inflow_batched_kernel = None

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
        "time_step", "num_reservoirs",
    ),
    size_key="num_reservoirs",
)

# ── Levee ──────────────────────────────────────────────────────────────────

compute_levee_stage_kernel = make_metal_dispatcher(
    _DIR / "levee_stage.metal", "compute_levee_stage",
    args=(
        "levee_catchment_idx_ptr", "river_storage_ptr", "flood_storage_ptr",
        "protected_storage_ptr", "river_depth_ptr", "flood_depth_ptr",
        "protected_depth_ptr", "river_height_ptr", "flood_depth_table_ptr",
        "catchment_area_ptr", "river_width_ptr", "river_length_ptr",
        "levee_base_height_ptr", "levee_crown_height_ptr",
        "levee_fraction_ptr", "flood_fraction_ptr", "num_levees",
    ),
    size_key="num_levees",
    template_vars={"__NUM_FLOOD_LEVELS__": "num_flood_levels"},
)

compute_levee_stage_log_kernel = None

compute_levee_bifurcation_outflow_kernel = make_metal_dispatcher(
    _DIR / "levee_bifurcation_outflow.metal", "compute_levee_bifurcation_outflow",
    args=(
        "bifurcation_catchment_idx_ptr", "bifurcation_downstream_idx_ptr",
        "bifurcation_manning_ptr", "bifurcation_outflow_ptr",
        "bifurcation_width_ptr", "bifurcation_length_ptr",
        "bifurcation_elevation_ptr", "bifurcation_cross_section_depth_ptr",
        "water_surface_elevation_ptr", "protected_water_surface_elevation_ptr",
        "total_storage_ptr", "outgoing_storage_ptr",
        "gravity", "time_step", "num_bifurcation_paths",
    ),
    size_key="num_bifurcation_paths",
    template_vars={"__NUM_BIF_LEVELS__": "num_bifurcation_levels"},
)

compute_levee_stage_batched_kernel = None
compute_levee_bifurcation_outflow_batched_kernel = None
