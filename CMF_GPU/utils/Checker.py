from datetime import datetime
from CMF_GPU.utils.Variables import MODULES_INFO
from CMF_GPU.utils.Aggregator import LEGAL_AGG_ARRAYS, LEGAL_STATS
from omegaconf import DictConfig, ListConfig
from pathlib import Path


def _check_statistics(stats):
    if stats is None:
        return True
    if not isinstance(stats, (dict, DictConfig)):
        raise ValueError(f"CONFIG ERROR: statistics must be a dict or DictConfig: {stats}")

    legal_vars = []
    for _, variables in LEGAL_AGG_ARRAYS.items():
        legal_vars.extend(variables)

    for stat_type, variables in stats.items():
        if stat_type not in LEGAL_STATS:
            raise ValueError(f"CONFIG ERROR: Invalid statistic type: {stat_type}")
        if variables is not None:
            for var in variables:
                if var not in legal_vars:
                    raise ValueError(f"CONFIG ERROR: Invalid variable '{var}' in statistics[{stat_type}]")
    return True

def _check_parameter_config(cfg):
    if not isinstance(cfg, (dict, DictConfig)):
        raise ValueError(f"CONFIG ERROR: parameter_config must be a dict or DictConfig: {cfg}")

    required = [
        "experiment_name", "map_dir", "hires_map_dir",
        "working_dir", "inp_dir", "precision", "gauge_file", "save_gauge_only", "simulate_gauge_only"
    ]
    for key in required:
        if key not in cfg:
            raise ValueError(f"CONFIG ERROR: parameter_config missing required key '{key}'")

    if not isinstance(cfg["experiment_name"], str) or not cfg["experiment_name"]:
        raise ValueError(f"CONFIG ERROR: Invalid experiment_name: {cfg['experiment_name']}")

    for key in ["map_dir", "hires_map_dir", "working_dir"]:
        if not isinstance(cfg[key], str):
            raise ValueError(f"CONFIG ERROR: {key} must be a string path, got {type(cfg[key])}")
        if not Path(cfg[key]).is_dir():
            raise ValueError(f"CONFIG ERROR: {key} does not exist or is not a directory: {cfg[key]}")

    inp_dir_parent = Path(cfg["inp_dir"]).parent
    if not inp_dir_parent.is_dir():
        raise ValueError(f"CONFIG ERROR: inp_dir parent does not exist or is not a directory: {inp_dir_parent}")

    if cfg["precision"] not in ["float32", "float64"]:
        raise ValueError(f"CONFIG ERROR: Invalid precision: {cfg['precision']}")
    
    if cfg["gauge_file"] == "None":
        raise ValueError("CONFIG ERROR: in yaml, gauge_file should be null, not 'None' to indicate no gauge file.")
    if cfg["gauge_file"] is not None:
        if not isinstance(cfg["gauge_file"], str):
            raise ValueError(f"CONFIG ERROR: gauge_file must be a string path, got {type(cfg['gauge_file'])}")
        gauge_path = Path(cfg["gauge_file"])
        if not gauge_path.is_file():
            raise ValueError(f"CONFIG ERROR: gauge_file does not exist or is not a file: {gauge_path}")
    
    for key in ["save_gauge_only", "simulate_gauge_only"]:
        if not isinstance(cfg[key], bool):
            raise ValueError(f"CONFIG ERROR: {key} must be a boolean value, got {type(cfg[key])}")
        
    save_gauge_only = cfg["save_gauge_only"]
    simulate_gauge_only = cfg["simulate_gauge_only"]
    if save_gauge_only and not simulate_gauge_only:
        print("[CONFIG] save_gauge_only is True, but simulate_gauge_only is False.\n"
              " - Only gauge results will be saved, but all catchments will be simulated.\n"
              " - Water balance can be fully reflected in logs.\n"
              " - This setting may slow down calibration.")
    elif not save_gauge_only and simulate_gauge_only:
        print("[CONFIG] simulate_gauge_only is True, but save_gauge_only is False.\n"
              " - Only gauge basins will be simulated, but all available results will be saved.\n"
              " - Logs will reflect water balance for gauge basins only.")
    elif save_gauge_only and simulate_gauge_only:
        print("[CONFIG] save_gauge_only and simulate_gauge_only are both True.\n"
              " - Only gauge basins will be simulated and saved.\n"
              " - This can speed up parameter calibration.")
    else:
        print("[CONFIG] save_gauge_only and simulate_gauge_only are both False.\n"
              " - All catchments will be simulated and all results will be saved.\n")
    
    if cfg["gauge_file"] is None and save_gauge_only:
        raise ValueError("CONFIG ERROR: gauge_file is None, but save_gauge_only is True")
    if cfg["gauge_file"] is None and simulate_gauge_only:
        raise ValueError("CONFIG ERROR: gauge_file is None, but simulate_gauge_only is True")
    return True

def _check_runoff_config(cfg):
    if not isinstance(cfg, (dict, DictConfig)):
        raise ValueError(f"CONFIG ERROR: runoff_config must be a dict or DictConfig: {cfg}")
    if "class_name" not in cfg:
        raise ValueError("CONFIG ERROR: runoff_config must contain class_name")
    if "params" not in cfg:
        raise ValueError("CONFIG ERROR: runoff_config must contain params")
    return True

def _check_simulation_config(cfg):
    if not isinstance(cfg, (dict, DictConfig)):
        raise ValueError(f"CONFIG ERROR: simulation_config must be a dict or DictConfig: {cfg}")
    modules = cfg.get("modules", None)
    if not isinstance(modules, (ListConfig, list)):
        raise ValueError(f"CONFIG ERROR: simulation_config must contain a list of modules: {cfg}")
    for module in modules:
        if module not in MODULES_INFO:
            raise ValueError(f"CONFIG ERROR: Invalid module in simulation_config: {module}")
    required = [
        "experiment_name", "working_dir", "out_dir", "inp_dir", "states_file", 
        "start_date", "end_date", "time_step", "default_num_sub_steps",
    ]
    for key in required:
        if key not in cfg:
            raise ValueError(f"CONFIG ERROR: simulation_config missing required key '{key}'")
        
    if not isinstance(cfg["experiment_name"], str) or not cfg["experiment_name"]:
        raise ValueError(f"CONFIG ERROR: Invalid experiment_name: {cfg['experiment_name']}")
    
    for key in ["working_dir"]:
        if not isinstance(cfg[key], str):
            raise ValueError(f"CONFIG ERROR: {key} must be a string path, got {type(cfg[key])}")
        if not Path(cfg[key]).is_dir():
            raise ValueError(f"CONFIG ERROR: {key} does not exist or is not a directory: {cfg[key]}")
        
    for key in ["out_dir", "inp_dir"]:
        parent_dir = Path(cfg[key]).parent
        if not parent_dir.is_dir():
            raise ValueError(f"CONFIG ERROR: {key} parent does not exist or is not a directory: {parent_dir}")

    if cfg["states_file"] == "None":
        raise ValueError("CONFIG ERROR: in yaml, states_file should be null, not 'None' to indicate no states file.")
    if cfg["states_file"] is not None:
        if not isinstance(cfg["states_file"], str):
            raise ValueError(f"CONFIG ERROR: states_file must be a string path, got {type(cfg['states_file'])}")
        states_path = Path(cfg["states_file"])
        if not states_path.is_file():
            raise ValueError(f"CONFIG ERROR: states_file does not exist or is not a file: {states_path}")
    
    try:
        datetime.strptime(cfg["start_date"], "%Y-%m-%d")
        datetime.strptime(cfg["end_date"], "%Y-%m-%d")
    except Exception:
        raise ValueError("CONFIG ERROR: start_date and end_date must be in YYYY-MM-DD format")

    if not isinstance(cfg["time_step"], (int, float)) or cfg["time_step"] <= 0:
        raise ValueError(f"CONFIG ERROR: runtime_config time_step must be positive number")
    if not isinstance(cfg["default_num_sub_steps"], int) or cfg["default_num_sub_steps"] <= 0:
        raise ValueError(f"CONFIG ERROR: runtime_config default_num_sub_steps must be positive integer")
    _check_statistics(cfg["statistics"])
    return True

CONFIG_REQUIRED_KEYS = {
    "parameter_config": _check_parameter_config,
    "runoff_config": _check_runoff_config,
    "simulation_config": _check_simulation_config,
}

def default_shape(params):
    if isinstance(params["num_catchments"], int):
        return (params["num_catchments"],)
    else :
        return (params["num_catchments"][()],)

def _log_shape(params):
    if isinstance(params["log_buffer_size"], int):
        return (params["log_buffer_size"],)
    else:
        return (params["log_buffer_size"][()],)

def _bifurcation_1D_shape(params):
    if isinstance(params["num_bifurcation_paths"], int):
        return (params["num_bifurcation_paths"],)
    else:
        return (params["num_bifurcation_paths"][()],)

def _bifurcation_2D_shape(params):
    if isinstance(params["num_bifurcation_paths"], int):
        return (params["num_bifurcation_paths"], params["num_bifurcation_levels"])
    else:
        return (params["num_bifurcation_paths"][()], params["num_bifurcation_levels"][()])

def _reservoir_shape(params):
    if isinstance(params["num_reservoirs"], int):
        return (params["num_reservoirs"],)
    else:
        return (params["num_reservoirs"][()],)

# all zeros
SPECIAL_HIDDEN_STATE_SHAPES = {
    # adaptive_time_step
    "min_time_step": lambda p: (1,),
    # log
    "total_storage_pre_sum": _log_shape,
    "total_storage_next_sum": _log_shape,
    "total_storage_new": _log_shape,
    "total_inflow": _log_shape,
    "total_outflow": _log_shape,
    "total_storage_stage_new": _log_shape,
    "total_river_storage": _log_shape,
    "total_flood_storage": _log_shape,
    "total_flood_area": _log_shape,
    "total_inflow_error": _log_shape,
    "total_stage_error": _log_shape,
    # bifurcation
}
    
SPECIAL_INPUT_SHAPES = {
    # base
    "save_idx": lambda p: (p["num_catchments_to_save"][()],),
    "runoff_input_matrix": lambda p: p["runoff_input_matrix"][()].shape, # not check, TODO: trim the second dimension by the mask
    "flood_depth_table": lambda p: (p["num_catchments"][()], p["num_flood_levels"][()] + 3),
    "num_catchments_per_basin": lambda p: (p["num_basins"][()],),
    # bifurcation
    "bifurcation_catchment_id": _bifurcation_1D_shape,
    "bifurcation_downstream_id": _bifurcation_1D_shape,
    "bifurcation_length": _bifurcation_1D_shape,
    "bifurcation_manning": _bifurcation_2D_shape,
    "bifurcation_width": _bifurcation_2D_shape,
    "bifurcation_elevation": _bifurcation_2D_shape,
    "bifurcation_outflow": _bifurcation_2D_shape,
    # reservoir
    "reservoir_catchment_idx": _reservoir_shape,
    "conservation_volume": _reservoir_shape,
    "emergency_volume": _reservoir_shape,
    "normal_outflow": _reservoir_shape,
    "flood_control_outflow": _reservoir_shape,
}
