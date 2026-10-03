"""Canonical backend-neutral ABIs for CaMa-Flood physics kernels."""

from hydroforge.kernels import (
    KernelSpec,
    KernelWorkspace,
    constant,
    module_enabled,
    output_requested,
)

HAS_BIFURCATION = module_enabled("bifurcation")
HAS_LEVEE = module_enabled("levee")
HAS_RESERVOIR = module_enabled("reservoir")
HAS_SEA_LEVEL = module_enabled("sea_level")
HAS_INFLOW = module_enabled("inflow")
HAS_TOTAL_STORAGE_OUTPUT = output_requested("base", "total_storage")

OUTFLOW = KernelSpec(
    name="compute_outflow",
    size="num_catchments",
    parameters={
        "downstream_idx_ptr": "read",
        "river_inflow_ptr": "write",
        "river_outflow_ptr": "read_write",
        "river_manning_ptr": "read",
        "river_depth_ptr": "read",
        "river_width_ptr": "read",
        "river_length_ptr": "read",
        "river_height_ptr": "read",
        "river_storage_ptr": "read",
        "flood_inflow_ptr": "write",
        "flood_outflow_ptr": "read_write",
        "flood_manning_ptr": "read",
        "flood_depth_ptr": "read",
        "catchment_elevation_ptr": "read",
        "downstream_distance_ptr": "read",
        "flood_storage_ptr": "read",
        "protected_storage_ptr": "read",
        "river_cross_section_depth_ptr": "read_write",
        "flood_cross_section_depth_ptr": "read_write",
        "flood_cross_section_area_ptr": "read_write",
        "global_bifurcation_outflow_ptr": "write",
        "outgoing_storage_ptr": "atomic_add",
        "gravity": "precision",
        "time_step_ptr": "read",
        "num_catchments": "index",
        "HAS_BIFURCATION": HAS_BIFURCATION,
        "HAS_LEVEE": HAS_LEVEE,
        "is_dam_upstream_ptr": "read",
        "HAS_RESERVOIR": HAS_RESERVOIR,
        "min_kinematic_slope": constant("precision"),
        "sea_surface_elevation_ptr": "read",
        "catchment_sea_level_idx_ptr": "read",
        "HAS_SEA_LEVEL": HAS_SEA_LEVEL,
        "ensemble_size": "index",
        **dict.fromkeys(
            (
                "batched_river_manning",
                "batched_flood_manning",
                "batched_river_width",
                "batched_river_length",
                "batched_river_height",
                "batched_catchment_elevation",
                "batched_downstream_distance",
                "batched_sea_surface_elevation",
            ),
            constant("bool"),
        ),
        "num_sea_level_boundaries": "index",
    },
    optional={
        "global_bifurcation_outflow_ptr": "HAS_BIFURCATION",
        "protected_storage_ptr": "HAS_LEVEE",
        "is_dam_upstream_ptr": "HAS_RESERVOIR",
        "sea_surface_elevation_ptr": "HAS_SEA_LEVEL",
        "catchment_sea_level_idx_ptr": "HAS_SEA_LEVEL",
    },
    optional_values={"num_sea_level_boundaries": ("HAS_SEA_LEVEL", 0)},
)

INFLOW = KernelSpec(
    name="compute_inflow",
    size="num_catchments",
    parameters={
        "downstream_idx_ptr": "read",
        "river_outflow_ptr": "read_write",
        "flood_outflow_ptr": "read_write",
        "river_storage_ptr": "read",
        "flood_storage_ptr": "read",
        "outgoing_storage_ptr": "read",
        "time_step_ptr": "read",
        "river_inflow_ptr": "atomic_add",
        "flood_inflow_ptr": "atomic_add",
        "limit_rate_ptr": "write",
        "reservoir_total_inflow_ptr": "atomic_add",
        "is_reservoir_ptr": "read",
        "num_catchments": "index",
        "HAS_BIFURCATION": HAS_BIFURCATION,
        "HAS_RESERVOIR": HAS_RESERVOIR,
        "ensemble_size": "index",
    },
    optional={
        "limit_rate_ptr": "HAS_BIFURCATION",
        "reservoir_total_inflow_ptr": "HAS_RESERVOIR",
        "is_reservoir_ptr": "HAS_RESERVOIR",
    },
)

ADAPTIVE_TIME = KernelSpec(
    name="compute_adaptive_time_step",
    size="num_catchments",
    parameters={
        "river_depth_ptr": "read",
        "downstream_distance_ptr": "read",
        "is_dam_related_ptr": "read",
        "max_sub_steps_ptr": "atomic_max",
        "outer_time_step_ptr": "read",
        "adaptive_time_factor": "precision",
        "gravity": "precision",
        "num_catchments": "index",
        "HAS_RESERVOIR": HAS_RESERVOIR,
        "ensemble_size": "index",
        "batched_downstream_distance": constant("bool"),
    },
    optional={"is_dam_related_ptr": "HAS_RESERVOIR"},
    block_sizes={"metal": 256},
)

BIFURCATION_OUTFLOW = KernelSpec(
    name="compute_bifurcation_outflow",
    size="num_bifurcation_paths",
    parameters={
        "bifurcation_catchment_idx_ptr": "read",
        "bifurcation_downstream_idx_ptr": "read",
        "bifurcation_manning_ptr": "read",
        "bifurcation_outflow_ptr": "read_write",
        "bifurcation_width_ptr": "read",
        "bifurcation_length_ptr": "read",
        "bifurcation_elevation_ptr": "read",
        "bifurcation_cross_section_depth_ptr": "read_write",
        "river_depth_ptr": "read",
        "river_height_ptr": "read",
        "catchment_elevation_ptr": "read",
        "river_storage_ptr": "read",
        "flood_storage_ptr": "read",
        "outgoing_storage_ptr": "atomic_add",
        "gravity": "precision",
        "time_step_ptr": "read",
        "num_bifurcation_paths": "index",
        "num_bifurcation_levels": constant("int32"),
        "ensemble_size": "index",
        "num_catchments": "index",
        **dict.fromkeys(
            (
                "batched_bifurcation_manning",
                "batched_bifurcation_width",
                "batched_bifurcation_length",
                "batched_bifurcation_elevation",
                "batched_river_height",
                "batched_catchment_elevation",
            ),
            constant("bool"),
        ),
    },
)

BIFURCATION_INFLOW = KernelSpec(
    name="compute_bifurcation_inflow",
    size="num_bifurcation_paths",
    parameters={
        "bifurcation_catchment_idx_ptr": "read",
        "bifurcation_downstream_idx_ptr": "read",
        "limit_rate_ptr": "read",
        "bifurcation_outflow_ptr": "read_write",
        "global_bifurcation_outflow_ptr": "atomic_add",
        "num_bifurcation_paths": "index",
        "num_bifurcation_levels": constant("int32"),
        "ensemble_size": "index",
        "num_catchments": "index",
    },
)

RESERVOIR_OUTFLOW = KernelSpec(
    name="compute_reservoir_outflow",
    size="num_reservoirs",
    parameters={
        "reservoir_catchment_idx_ptr": "read",
        "downstream_idx_ptr": "read",
        "reservoir_total_inflow_ptr": "read_write",
        "river_outflow_ptr": "read_write",
        "flood_outflow_ptr": "read_write",
        "river_storage_ptr": "read",
        "flood_storage_ptr": "read",
        "protected_storage_ptr": "read",
        "conservation_volume_ptr": "read",
        "emergency_volume_ptr": "read",
        "adjustment_volume_ptr": "read",
        "effective_normal_outflow_ptr": "read",
        "adjustment_outflow_ptr": "read",
        "flood_control_outflow_ptr": "read",
        "runoff_ptr": "read",
        "outgoing_storage_ptr": "atomic_add",
        "time_step_ptr": "read",
        "num_reservoirs": "int32",
        "num_catchments": "index",
        "HAS_LEVEE": HAS_LEVEE,
        "ensemble_size": "index",
        **dict.fromkeys(
            (
                "batched_runoff",
                "batched_conservation_volume",
                "batched_emergency_volume",
                "batched_adjustment_volume",
                "batched_effective_normal_outflow",
                "batched_adjustment_outflow",
                "batched_flood_control_outflow",
            ),
            constant("bool"),
        ),
    },
    optional={"protected_storage_ptr": "HAS_LEVEE"},
)

FLOOD_STAGE = KernelSpec(
    name="compute_flood_stage",
    size="num_catchments",
    parameters={
        "river_inflow_ptr": "read",
        "flood_inflow_ptr": "read",
        "river_outflow_ptr": "read",
        "flood_outflow_ptr": "read",
        "global_bifurcation_outflow_ptr": "read",
        "runoff_ptr": "read",
        "inflow_ptr": "read",
        "catchment_inflow_idx_ptr": "read",
        "time_step_ptr": "read",
        "outgoing_storage_ptr": "write",
        "river_storage_ptr": "read_write",
        "flood_storage_ptr": "read_write",
        "protected_storage_ptr": "read_write",
        "total_storage_ptr": "write",
        "river_depth_ptr": "write",
        "flood_depth_ptr": "write",
        "protected_depth_ptr": "write",
        "flood_fraction_ptr": "read_write",
        "river_height_ptr": "read",
        "flood_depth_table_ptr": "read",
        "catchment_area_ptr": "read",
        "river_width_ptr": "read",
        "river_length_ptr": "read",
        "num_catchments": "index",
        "num_flood_levels": constant("int32"),
        "HAS_BIFURCATION": HAS_BIFURCATION,
        "HAS_INFLOW": HAS_INFLOW,
        "HAS_LEVEE": HAS_LEVEE,
        "HAS_TOTAL_STORAGE_OUTPUT": HAS_TOTAL_STORAGE_OUTPUT,
        "ensemble_size": "index",
        **dict.fromkeys(
            (
                "batched_runoff",
                "batched_inflow",
                "batched_river_height",
                "batched_flood_depth_table",
                "batched_catchment_area",
                "batched_river_width",
                "batched_river_length",
            ),
            constant("bool"),
        ),
        "num_inflow_gauges": "int32",
    },
    optional={
        "global_bifurcation_outflow_ptr": "HAS_BIFURCATION",
        "inflow_ptr": "HAS_INFLOW",
        "catchment_inflow_idx_ptr": "HAS_INFLOW",
        "protected_storage_ptr": "HAS_LEVEE",
        "protected_depth_ptr": "HAS_LEVEE",
        "total_storage_ptr": None,
    },
    optional_values={"num_inflow_gauges": ("HAS_INFLOW", 0)},
)

FLOOD_STAGE_LOG = KernelSpec(
    name="compute_flood_stage_log",
    size="num_catchments",
    parameters={
        "river_inflow_ptr": "read",
        "flood_inflow_ptr": "read",
        "river_outflow_ptr": "read",
        "flood_outflow_ptr": "read",
        "global_bifurcation_outflow_ptr": "read",
        "runoff_ptr": "read",
        "inflow_ptr": "read",
        "catchment_inflow_idx_ptr": "read",
        "time_step_ptr": "read",
        "outgoing_storage_ptr": "write",
        "river_storage_ptr": "read_write",
        "flood_storage_ptr": "read_write",
        "protected_storage_ptr": "read_write",
        "total_storage_ptr": "write",
        "river_depth_ptr": "write",
        "flood_depth_ptr": "write",
        "protected_depth_ptr": "write",
        "flood_fraction_ptr": "read_write",
        "river_height_ptr": "read",
        "flood_depth_table_ptr": "read",
        "catchment_area_ptr": "read",
        "river_width_ptr": "read",
        "river_length_ptr": "read",
        "is_levee_ptr": "read",
        "total_storage_pre_sum_ptr": "atomic_add",
        "total_storage_next_sum_ptr": "atomic_add",
        "total_storage_new_sum_ptr": "atomic_add",
        "total_inflow_sum_ptr": "atomic_add",
        "total_outflow_sum_ptr": "atomic_add",
        "total_storage_stage_sum_ptr": "atomic_add",
        "river_storage_sum_ptr": "atomic_add",
        "flood_storage_sum_ptr": "atomic_add",
        "flood_area_sum_ptr": "atomic_add",
        "total_inflow_error_sum_ptr": "atomic_add",
        "total_stage_error_sum_ptr": "atomic_add",
        "current_step_ptr": "read",
        "num_catchments": "index",
        "num_flood_levels": constant("int32"),
        "HAS_BIFURCATION": HAS_BIFURCATION,
        "HAS_INFLOW": HAS_INFLOW,
        "HAS_LEVEE": HAS_LEVEE,
        "HAS_TOTAL_STORAGE_OUTPUT": HAS_TOTAL_STORAGE_OUTPUT,
    },
    optional={
        "global_bifurcation_outflow_ptr": "HAS_BIFURCATION",
        "inflow_ptr": "HAS_INFLOW",
        "catchment_inflow_idx_ptr": "HAS_INFLOW",
        "protected_storage_ptr": "HAS_LEVEE",
        "protected_depth_ptr": "HAS_LEVEE",
        "is_levee_ptr": "HAS_LEVEE",
        "total_storage_ptr": None,
    },
    block_sizes={"metal": 256},
)

LEVEE_STAGE = KernelSpec(
    name="compute_levee_stage",
    size="num_levees",
    parameters={
        "levee_catchment_idx_ptr": "read",
        "levee_river_max_storage_ptr": "read",
        "levee_base_storage_ptr": "read",
        "levee_top_storage_ptr": "read",
        "levee_fill_storage_ptr": "read",
        "levee_layer_top_storage_ptr": "read",
        "river_storage_ptr": "read_write",
        "flood_storage_ptr": "read_write",
        "protected_storage_ptr": "write",
        "river_depth_ptr": "read_write",
        "flood_depth_ptr": "read_write",
        "protected_depth_ptr": "write",
        "river_height_ptr": "read",
        "flood_depth_table_ptr": "read",
        "catchment_area_ptr": "read",
        "river_width_ptr": "read",
        "river_length_ptr": "read",
        "levee_base_height_ptr": "read",
        "levee_crown_height_ptr": "read",
        "levee_fraction_ptr": "read",
        "flood_fraction_ptr": "read_write",
        "num_levees": "int32",
        "num_flood_levels": constant("int32"),
        "ensemble_size": "index",
        "num_catchments": "index",
        **dict.fromkeys(
            (
                "batched_river_length",
                "batched_river_width",
                "batched_river_height",
                "batched_catchment_area",
                "batched_levee_crown_height",
                "batched_levee_fraction",
                "batched_levee_base_height",
                "batched_flood_depth_table",
            ),
            constant("bool"),
        ),
    },
)

LEVEE_STAGE_LOG = KernelSpec(
    name="compute_levee_stage_log",
    size="num_levees",
    parameters={
        "levee_catchment_idx_ptr": "read",
        "levee_river_max_storage_ptr": "read",
        "levee_base_storage_ptr": "read",
        "levee_top_storage_ptr": "read",
        "levee_fill_storage_ptr": "read",
        "levee_layer_top_storage_ptr": "read",
        "river_storage_ptr": "read_write",
        "flood_storage_ptr": "read_write",
        "protected_storage_ptr": "write",
        "river_depth_ptr": "read_write",
        "flood_depth_ptr": "read_write",
        "protected_depth_ptr": "write",
        "river_height_ptr": "read",
        "flood_depth_table_ptr": "read",
        "catchment_area_ptr": "read",
        "river_width_ptr": "read",
        "river_length_ptr": "read",
        "levee_base_height_ptr": "read",
        "levee_crown_height_ptr": "read",
        "levee_fraction_ptr": "read",
        "flood_fraction_ptr": "read_write",
        "total_storage_stage_sum_ptr": "atomic_add",
        "river_storage_sum_ptr": "atomic_add",
        "flood_storage_sum_ptr": "atomic_add",
        "flood_area_sum_ptr": "atomic_add",
        "total_stage_error_sum_ptr": "atomic_add",
        "current_step_ptr": "read",
        "num_levees": "int32",
        "num_flood_levels": constant("int32"),
    },
    block_sizes={"metal": 256},
)

LEVEE_BIFURCATION_OUTFLOW = KernelSpec(
    name="compute_levee_bifurcation_outflow",
    size="num_bifurcation_paths",
    parameters={
        "bifurcation_catchment_idx_ptr": "read",
        "bifurcation_downstream_idx_ptr": "read",
        "bifurcation_manning_ptr": "read",
        "bifurcation_outflow_ptr": "read_write",
        "bifurcation_width_ptr": "read",
        "bifurcation_length_ptr": "read",
        "bifurcation_elevation_ptr": "read",
        "bifurcation_cross_section_depth_ptr": "read_write",
        "river_depth_ptr": "read",
        "protected_depth_ptr": "read",
        "river_height_ptr": "read",
        "catchment_elevation_ptr": "read",
        "is_levee_ptr": "read",
        "river_storage_ptr": "read",
        "flood_storage_ptr": "read",
        "protected_storage_ptr": "read",
        "outgoing_storage_ptr": "atomic_add",
        "gravity": "precision",
        "time_step_ptr": "read",
        "num_bifurcation_paths": "index",
        "num_bifurcation_levels": constant("int32"),
        "ensemble_size": "index",
        "num_catchments": "index",
        **dict.fromkeys(
            (
                "batched_bifurcation_manning",
                "batched_bifurcation_width",
                "batched_bifurcation_length",
                "batched_bifurcation_elevation",
                "batched_river_height",
                "batched_catchment_elevation",
            ),
            constant("bool"),
        ),
    },
)

# Metal's double-FP32 accumulators cannot use scatter atomics. The flow cache
# is [member, catchment, channel], with channel 0 = river and 1 = flood.
# Its provisional values remain immutable throughout the supply-limit kernel.
_UNLIMITED_FLOW = KernelWorkspace(
    key="cmfgpu.metal.unlimited_outflow",
    dtype="float32",
    shape=("ensemble_size", "num_catchments", 2),
    emulated_only=True,
)
_PATH_FLOW = KernelWorkspace(
    key="cmfgpu.metal.bifurcation_path_flow",
    dtype="float32",
    shape=("ensemble_size", "num_bifurcation_paths"),
    module="bifurcation",
    emulated_only=True,
)
_ROUTING_WORKSPACE = {
    "routing_edge_start_ptr": KernelWorkspace(
        key="cmfgpu.metal.routing_edge_start",
        dtype="int32",
        shape=("num_catchments",),
        initialize="csr_offsets",
        source="downstream_idx",
        target_count="num_catchments",
        emulated_only=True,
    ),
    "routing_edge_source_ptr": KernelWorkspace(
        key="cmfgpu.metal.routing_edge_source",
        dtype="int32",
        shape=("num_catchments",),
        initialize="csr_order",
        source="downstream_idx",
        target_count="num_catchments",
        emulated_only=True,
    ),
    "bifurcation_path_flow_ptr": _PATH_FLOW,
    **{
        f"bifurcation_{endpoint}_{suffix}_ptr": KernelWorkspace(
            key=f"cmfgpu.metal.bifurcation_{endpoint}_{suffix}",
            dtype="int32",
            shape=("num_catchments" if suffix == "start" else "num_bifurcation_paths",),
            initialize="csr_offsets" if suffix == "start" else "csr_order",
            source=f"bifurcation_{endpoint}_idx",
            target_count="num_catchments",
            module="bifurcation",
            emulated_only=True,
        )
        for endpoint in ("catchment", "downstream")
        for suffix in ("start", "path")
    },
}

_ROUTING_PARAMETERS = {
    "routing_edge_start_ptr": "read",
    "routing_edge_source_ptr": "read",
    "bifurcation_catchment_start_ptr": "read",
    "bifurcation_catchment_path_ptr": "read",
    "bifurcation_downstream_start_ptr": "read",
    "bifurcation_downstream_path_ptr": "read",
    "bifurcation_path_flow_ptr": "read",
    "num_bifurcation_paths": "index",
}
_ROUTING_OPTIONAL = {
    name: None for name in _ROUTING_PARAMETERS if name.endswith("_ptr")
}

METAL_OUTFLOW = KernelSpec(
    **{
        **OUTFLOW.model_dump(),
        "workspace": {"unlimited_outflow_ptr": _UNLIMITED_FLOW},
        "parameters": {**OUTFLOW.parameters, "unlimited_outflow_ptr": "write"},
        "optional": {**OUTFLOW.optional, "unlimited_outflow_ptr": None},
    }
)
METAL_RESERVOIR_OUTFLOW = KernelSpec(
    **{
        **RESERVOIR_OUTFLOW.model_dump(),
        "workspace": {"unlimited_outflow_ptr": _UNLIMITED_FLOW},
        "parameters": {
            **RESERVOIR_OUTFLOW.parameters,
            "unlimited_outflow_ptr": "write",
        },
        "optional": {**RESERVOIR_OUTFLOW.optional, "unlimited_outflow_ptr": None},
    }
)
METAL_INFLOW = KernelSpec(
    **{
        **INFLOW.model_dump(),
        "workspace": {**_ROUTING_WORKSPACE, "unlimited_outflow_ptr": _UNLIMITED_FLOW},
        "parameters": {
            **INFLOW.parameters,
            **_ROUTING_PARAMETERS,
            "unlimited_outflow_ptr": "read",
            "outgoing_storage_ptr": "read_write",
        },
        "optional": {
            **INFLOW.optional,
            **_ROUTING_OPTIONAL,
            "unlimited_outflow_ptr": None,
        },
        "optional_values": {
            **INFLOW.optional_values,
            "num_bifurcation_paths": ("HAS_BIFURCATION", 0),
        },
    }
)
METAL_BIFURCATION_OUTFLOW = KernelSpec(
    **{
        **BIFURCATION_OUTFLOW.model_dump(),
        "workspace": {"bifurcation_path_flow_ptr": _PATH_FLOW},
        "parameters": {
            **BIFURCATION_OUTFLOW.parameters,
            "bifurcation_path_flow_ptr": "write",
        },
        "optional": {**BIFURCATION_OUTFLOW.optional, "bifurcation_path_flow_ptr": None},
    }
)
METAL_BIFURCATION_INFLOW = KernelSpec(
    **{
        **BIFURCATION_INFLOW.model_dump(),
        "workspace": {"bifurcation_path_flow_ptr": _PATH_FLOW},
        "parameters": {
            **BIFURCATION_INFLOW.parameters,
            "bifurcation_path_flow_ptr": "write",
        },
        "optional": {**BIFURCATION_INFLOW.optional, "bifurcation_path_flow_ptr": None},
    }
)
METAL_LEVEE_BIFURCATION_OUTFLOW = KernelSpec(
    **{
        **LEVEE_BIFURCATION_OUTFLOW.model_dump(),
        "workspace": {"bifurcation_path_flow_ptr": _PATH_FLOW},
        "parameters": {
            **LEVEE_BIFURCATION_OUTFLOW.parameters,
            "bifurcation_path_flow_ptr": "write",
        },
        "optional": {
            **LEVEE_BIFURCATION_OUTFLOW.optional,
            "bifurcation_path_flow_ptr": None,
        },
    }
)
_METAL_STAGE_PARAMETERS = {
    **_ROUTING_PARAMETERS,
    "river_inflow_ptr": "read_write",
    "flood_inflow_ptr": "read_write",
    "global_bifurcation_outflow_ptr": "read_write",
    "reservoir_total_inflow_ptr": "read_write",
    "is_reservoir_ptr": "read",
    "HAS_RESERVOIR": HAS_RESERVOIR,
}
_METAL_STAGE_OPTIONAL = {
    **_ROUTING_OPTIONAL,
    "reservoir_total_inflow_ptr": "HAS_RESERVOIR",
    "is_reservoir_ptr": "HAS_RESERVOIR",
}
METAL_FLOOD_STAGE = KernelSpec(
    **{
        **FLOOD_STAGE.model_dump(),
        "workspace": _ROUTING_WORKSPACE,
        "parameters": {**FLOOD_STAGE.parameters, **_METAL_STAGE_PARAMETERS},
        "optional": {**FLOOD_STAGE.optional, **_METAL_STAGE_OPTIONAL},
        "optional_values": {
            **FLOOD_STAGE.optional_values,
            "num_bifurcation_paths": ("HAS_BIFURCATION", 0),
        },
    }
)
METAL_FLOOD_STAGE_LOG = KernelSpec(
    **{
        **FLOOD_STAGE_LOG.model_dump(),
        "workspace": _ROUTING_WORKSPACE,
        "parameters": {**FLOOD_STAGE_LOG.parameters, **_METAL_STAGE_PARAMETERS},
        "optional": {**FLOOD_STAGE_LOG.optional, **_METAL_STAGE_OPTIONAL},
        "optional_values": {
            **FLOOD_STAGE_LOG.optional_values,
            "num_bifurcation_paths": ("HAS_BIFURCATION", 0),
        },
    }
)
