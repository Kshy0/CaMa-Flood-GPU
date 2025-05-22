import numpy as np
import torch

#    catchment_id should be ordered to satisfy the following:
# 1. Catchments belonging to the same basin are grouped together;
# 2. Basins with bifurcation connections are placed adjacently;
# 3. Within each basin, catchments are ordered from upstream to downstream according to flow direction.
MODULES_INFO = {
    "base": {
        "params": [
            "downstream_idx", # equal to itself if the catchment is river mouth
            "num_catchments_per_basin", # size of associated basins due to bifurcation
            "is_river_mouth",
            "is_reservoir",
            "river_width",
            "river_length",
            "river_height",
            "catchment_elevation",
            "downstream_distance",
            "flood_depth_table",
            "catchment_area",
            "river_manning",
            "flood_manning",
        ],
        "hidden_params": [
            "river_elevation",
            "river_area",
            "river_max_storage",
            "max_flood_area",
            "total_width_table",
            "total_storage_table",
            "flood_gradient_table",
        ],
        "scalar_params": [
            "num_basins",
            "num_catchments",
            "num_flood_levels",
            "gravity", 
        ],
        "states": [
            "river_storage",
            "flood_storage",
            "river_depth",
            "river_outflow",
            "flood_depth",
            "flood_outflow",
            "river_cross_section_depth",
            "flood_cross_section_depth",
            "flood_cross_section_area",
        ],
        "hidden_states": [
            "total_storage",
            "outgoing_storage",
            "water_surface_elevation",
            "limit_rate",
            "river_inflow",
            "flood_area",
            "flood_fraction",
            "flood_inflow",
        ],

    },
    "adaptive_time_step": {
        "scalar_params": ["adaptation_factor"],
        "hidden_states": ["min_time_step"],
    },
    "log": {
        "scalar_params": ["log_buffer_size"],
        "hidden_states": [
            "total_storage_pre_sum",
            "total_storage_next_sum",
            "total_storage_new",
            "total_inflow",
            "total_outflow",
            "total_storage_stage_new",
            "total_river_storage",
            "total_flood_storage",
            "total_flood_area",
            "total_inflow_error",
            "total_stage_error",
        ],
    },
    "bifurcation": {
        "params": [
            "bifurcation_catchment_idx",
            "bifurcation_downstream_idx",
            "bifurcation_manning",
            "bifurcation_width",
            "bifurcation_length",
            "bifurcation_elevation",
        ],
        "scalar_params": [
            "num_bifurcation_paths",
            "num_bifurcation_levels"
        ],
        "states": [
            "bifurcation_outflow",
            "bifurcation_cross_section_depth",
        ],
        "hidden_states": [],
    },
    "reservoir": {
        "params": [
            "reservoir_catchment_idx",
            "conservation_volume",
            "emergency_volume",
            "normal_outflow",
            "flood_control_outflow",
        ],
        "hidden_params": [
            "reservoir_upstream_idx",
            "adjustment_volume",
            "adjustment_outflow",
        ],
        "scalar_params": [
            "num_reservoirs",
            "num_upstreams",
            "min_slope",
        ],
    },
}

SCALAR_TYPES = {
    "num_basins": int,
    "num_catchments": int,
    "num_flood_levels": int,
    "gravity": float,
    "log_buffer_size": int,
    "adaptation_factor": float,
    "num_bifurcation_paths": int,
    "num_bifurcation_levels": int,
    "num_reservoirs": int,
    "num_upstreams": int,
    "min_slope": float,
}

SPECIAL_ARRAY_TYPES = {
    "downstream_idx": np.int64,
    "num_catchments_per_basin": np.int64,
    "is_river_mouth": np.bool,
    "is_reservoir": np.bool,
    "bifurcation_catchment_idx": np.int64,
    "bifurcation_downstream_idx": np.int64,
}

HIDDEN_PARAMS = {
    # base
    "river_elevation": lambda p: p["catchment_elevation"] - p["river_height"],
    "river_area": lambda p: p["river_length"] * p["river_width"],
    "river_max_storage": lambda p: p["river_area"] * p["river_height"],
    "max_flood_area": lambda p: p["river_area"] + p["catchment_area"],
    "total_width_table": lambda p: torch.cat([
        p["river_width"][:, None],
        p["river_width"][:, None] + torch.linspace(
            0, 1, p["num_flood_levels"] + 1
        )[None, :] * p["catchment_area"][:, None] / p["river_length"][:, None],
        (p["river_width"] + p["catchment_area"] / p["river_length"])[:, None]
    ], dim=1),
    "total_storage_table": lambda p: torch.cat([
        torch.zeros((p["num_catchments"], 1)),
        torch.cumsum(
            p["river_length"][:, None] *
            0.5 * (p["total_width_table"][:, :-1] + p["total_width_table"][:, 1:]) *
            (p["flood_depth_table"][:, 1:] - p["flood_depth_table"][:, :-1]),
            dim=1
        )
    ], dim=1),
    "flood_gradient_table": lambda p: torch.where(
        (p["total_width_table"][:, 1:] - p["total_width_table"][:, :-1]) != 0,
        (p["flood_depth_table"][:, 1:] - p["flood_depth_table"][:, :-1]) /
        (p["total_width_table"][:, 1:] - p["total_width_table"][:, :-1]),
        torch.zeros_like(p["flood_depth_table"][:, 1:])
    ),
    # reservoir
    "adjustment_volume": lambda p: p["conservation_volume"] + p["emergency_volume"] * 0.1,
    "adjustment_outflow": lambda p: (
        torch.minimum(p["normal_outflow"], (
            (p["conservation_volume"] * 0.7 + p["normal_outflow"] * (365. * 24 * 60 * 60) / 4)
            / (180. * 24 * 60 * 60)
        )) * 1.5 + p["flood_control_outflow"]
    ) * 0.5,
}

