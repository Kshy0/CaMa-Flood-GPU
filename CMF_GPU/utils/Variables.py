import numpy as np
np.seterr(divide='ignore', invalid='ignore')

MODULES = [
    "base",
    "adaptive_time_step",
    "aggregator",
    "log",
    "bifurcation",
#    "reservoir",
]

MODULES_CONFIG = {
    "base": {
        "params": [
            "is_river_mouth",
            "is_reservoir",
            "downstream_idx",
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
        "hidden_params": [],
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
            "is_reservoir",
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
    "num_catchments": int,
    "num_flood_levels": int,
    "gravity": float,
    "adaptation_factor": float,
    "log_buffer_size": int,
    "num_bifurcation_paths": int,
    "num_bifurcation_levels": int,
    "num_reservoirs": int,
    "num_upstreams": int,
    "min_slope": float,
}

SPECIAL_ARRAY_TYPES = {
    "is_river_mouth": np.bool,
    "is_reservoir": np.bool,
    "downstream_idx": np.int64,
    "bifurcation_catchment_idx": np.int64,
    "bifurcation_downstream_idx": np.int64,
}

SPECIAL_HIDDEN_PARAMS = {
    "river_elevation": lambda p: p["catchment_elevation"] - p["river_height"],
    "river_area": lambda p: p["river_length"] * p["river_width"],
    "river_max_storage": lambda p: p["river_area"] * p["river_height"],
    "max_flood_area": lambda p: p["river_area"] + p["catchment_area"],
    "total_width_table": lambda p: np.concatenate([
        p["river_width"][:, None],
        p["river_width"][:, None] + np.linspace(0, 1, p["num_flood_levels"] + 1)[None, :] * p["catchment_area"][:, None] / p["river_length"][:, None],
        (p["river_width"] + p["catchment_area"] / p["river_length"])[:, None]
    ], axis=1),
    "total_storage_table": lambda p: np.concatenate([
        np.zeros((p["num_catchments"], 1)),
        np.cumsum(
            p["river_length"][:, None] *
            0.5 * (p["total_width_table"][:, :-1] + p["total_width_table"][:, 1:]) *
            (p["flood_depth_table"][:, 1:] - p["flood_depth_table"][:, :-1]),
            axis=1
        )
    ], axis=1),
    "flood_gradient_table": lambda p: np.where(
        (p["total_width_table"][:, 1:] - p["total_width_table"][:, :-1]) != 0,
        (p["flood_depth_table"][:, 1:] - p["flood_depth_table"][:, :-1]) /
        (p["total_width_table"][:, 1:] - p["total_width_table"][:, :-1]),
        0
    ),
    "adjustment_volume": lambda p: p["conservation_volume"] + p["emergency_volume"] * 0.1,
    "adjustment_outflow": lambda p: (
        np.minimum(p["normal_outflow"], (
            (p["conservation_volume"] * 0.7 + p["normal_outflow"] * (365. * 24 * 60 * 60) / 4)
            / (180. * 24 * 60 * 60)
        )) * 1.5 + p["flood_control_outflow"]
    ) * 0.5,
}

SPECIAL_ARRAY_SHAPES = {
    "total_storage_table": lambda p: (p["num_catchments"], p["num_flood_levels"] + 3),
    "flood_depth_table": lambda p: (p["num_catchments"], p["num_flood_levels"] + 3),
    "total_width_table": lambda p: (p["num_catchments"], p["num_flood_levels"] + 3),
    "flood_gradient_table": lambda p: (p["num_catchments"], p["num_flood_levels"] + 2),
    "bifurcation_catchment_idx": lambda p: (p["num_bifurcation_paths"],),
    "bifurcation_downstream_idx": lambda p: (p["num_bifurcation_paths"],),
    "bifurcation_manning": lambda p: (p["num_bifurcation_paths"], p["num_bifurcation_levels"]),
    "bifurcation_width": lambda p: (p["num_bifurcation_paths"], p["num_bifurcation_levels"]),
    "bifurcation_length": lambda p: (p["num_bifurcation_paths"],),
    "bifurcation_elevation": lambda p: (p["num_bifurcation_paths"], p["num_bifurcation_levels"]),
    "bifurcation_outflow": lambda p: (p["num_bifurcation_paths"], p["num_bifurcation_levels"]),
    "bifurcation_cross_section_depth": lambda p: (p["num_bifurcation_paths"], p["num_bifurcation_levels"]),
    "reservoir_catchment_idx": lambda p: (p["num_reservoirs"],),
    "conservation_volume": lambda p: (p["num_reservoirs"],),
    "emergency_volume": lambda p: (p["num_reservoirs"],),
    "normal_outflow": lambda p: (p["num_reservoirs"],),
    "flood_control_outflow": lambda p: (p["num_reservoirs"],),
    "adjustment_volume": lambda p: (p["num_reservoirs"],),
    "adjustment_outflow": lambda p: (p["num_reservoirs"],),
}

RUNTIME_FLAGS_REQUIRED_KEYS = [
    "precision",
    "modules",
    "time_step",
    "unit_factor", # mm/day divided by unit_factor to get m/s
    "default_sub_iters",
    "device",
    "device_indices",
    "split_indices",
    "statistics"
]


LEGAL_AGG_ARRAYS = {
    ("num_catchments",): [
        "river_storage",
        "flood_storage",
        "river_depth",
        "river_outflow",
        "flood_depth",
        "flood_outflow",
        "river_cross_section_depth",
        "flood_cross_section_depth",
        "flood_cross_section_area",
        "total_storage",
        "water_surface_elevation",
        "limit_rate",
        "flood_area",
        "flood_fraction",
    ],
    ("num_bifurcation_paths", "num_bifurcation_levels"): [
        "bifurcation_outflow",
        "bifurcation_cross_section_depth",
    ],
    # ("num_bifurcation_paths",): [
    #     "bifurcation_outflow_sum",
    # ],
}

