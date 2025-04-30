import torch
import numpy as np

###############################################################################
# 0. Module Configurations (unchanged)
###############################################################################
MODULES_CONFIG = {
    "base": {
        "required_params": [
            "is_river_mouth",
            "downstream_idx",
            "river_width",
            "river_length",
            "river_height",
            "river_elevation",
            "catchment_elevation",
            "downstream_distance",
            "river_max_storage",
            "river_area",
            "max_flood_area",
            "total_storage_table",
            "flood_depth_table",
            "total_width_table",
            "flood_gradient_table",
            "catchment_area",
        ],
        "optional_params": {
            "river_manning": 0.03,
            "flood_manning": 0.1,
        },
        "required_states": [
            "river_storage",
        ],
        "optional_states": {
            "flood_storage": 0.0,
            "total_storage": 0.0,
            "outgoing_storage": 0.0, #
            "water_surface_elevation": 0.0, #
            "limit_rate": 0.0, #
            "river_depth": 0.0,
            "river_inflow": 0.0, #
            "river_outflow": 0.0,
            "flood_depth": 0.0,
            "flood_area": 0.0, #
            "flood_fraction": 0.0, #
            "flood_inflow": 0.0, #
            "flood_outflow": 0.0,
            "river_cross_section_depth": 0.0,
            "flood_cross_section_depth": 0.0,
            "flood_cross_section_area": 0.0,
        },
        "scalar_params": {
            "gravity": 9.81,
            "adaptation_factor": 0.7,
        }
    },
    "foo_module": {
        "required_params": [],
        "optional_params": {
            "foo_param": 0.0,
            "bar_param": 0.0,
        },
        "required_states": [],
        "optional_states": {
            "foo_state": 0.0,
            "bar_state": 0.0,
        },
        "scalar_params": {
            "foo": 0.85,
            "minimum_release_rate": 0.05
        }
    },
}

###############################################################################
# Global required runtime flags keys
###############################################################################
RUNTIME_FLAGS_REQUIRED_KEYS = [
    "precision",
    "modules",
    "time_step",
    "unit_factor", # mm/day divided by unit_factor to get m/s
    "default_sub_iters",
    "enable_adaptive_time_step",
    "device",
    "device_indices",
    "split_indices",
]

def gather_all_keys_and_defaults(input_type="param"):
    all_required = set()
    all_optional = set()
    all_defaults = {}
    required_key = "required_params" if input_type == "param" else "required_states"
    optional_key = "optional_params" if input_type == "param" else "optional_states"
    default_sources = {}
    for mod_name, mod_cfg in MODULES_CONFIG.items():
        for rk in mod_cfg.get(required_key, []):
            all_required.add(rk)
        for ok, default_val in mod_cfg.get(optional_key, {}).items():
            all_optional.add(ok)
            if ok in all_defaults and all_defaults[ok] != default_val:
                raise ValueError(
                    f"Conflicting default values for {input_type} '{ok}': "
                    f"{all_defaults[ok]} from '{default_sources[ok]}' and "
                    f"{default_val} from '{mod_name}'"
                )
            all_defaults[ok] = default_val
            default_sources[ok] = mod_name
    return all_required, all_optional, all_defaults

def gather_all_scalar_params():
    merged_scalar_params = {}
    param_sources = {}
    for mod_name, mod_cfg in MODULES_CONFIG.items():
        for k, v in mod_cfg.get("scalar_params", {}).items():
            if k in merged_scalar_params and merged_scalar_params[k] != v:
                raise ValueError(
                    f"Conflicting scalar parameter '{k}': "
                    f"{merged_scalar_params[k]} from '{param_sources[k]}' and "
                    f"{v} from '{mod_name}'"
                )
            merged_scalar_params[k] = v
            param_sources[k] = mod_name
    return merged_scalar_params

def get_modules_using_key(key, key_type="param"):
    used_by = []
    required_key = "required_params" if key_type == "param" else "required_states"
    optional_key = "optional_params" if key_type == "param" else "optional_states"
    for mod_name, mod_cfg in MODULES_CONFIG.items():
        if key in mod_cfg.get(required_key, []) or key in mod_cfg.get(optional_key, {}):
            used_by.append(mod_name)
    return used_by

###############################################################################
# 3. Runtime Flags Validation
###############################################################################
def validate_runtime_flags(runtime_flags):
    # Check for unexpected keys
    unexpected_keys = set(runtime_flags.keys()) - set(RUNTIME_FLAGS_REQUIRED_KEYS)
    if unexpected_keys:
        raise ValueError(f"Unexpected runtime flag(s): {unexpected_keys}")
    
    for rk in RUNTIME_FLAGS_REQUIRED_KEYS:
        if rk not in runtime_flags:
            raise ValueError(f"Missing required runtime flag: {rk}")

    if "base" not in runtime_flags["modules"]:
        raise ValueError("'base' module is required and must be included in modules")

    if runtime_flags["precision"] not in ["float32", "float64"]:
        raise ValueError("precision must be 'float32' or 'float64'")

    for m in runtime_flags["modules"]:
        if m not in MODULES_CONFIG:
            raise ValueError(f"Unknown module '{m}' in runtime_flags['modules']")

    # Check types of time_step, unit_factor, default_sub_iters
    for key in ["time_step", "unit_factor", "default_sub_iters"]:
        if key not in runtime_flags:
            raise ValueError(f"Missing required runtime flag: {key}")
        if isinstance(runtime_flags[key], np.generic):
            raise ValueError(f"{key} must be a native Python object, not a NumPy type")

    if not isinstance(runtime_flags["default_sub_iters"], int):
        raise ValueError("default_sub_iters must be an integer")

    if runtime_flags["device"] != "gpu":
        raise NotImplementedError("Only 'gpu' device is supported")

    di = runtime_flags["device_indices"]
    if not (isinstance(di, int) or (isinstance(di, list) and all(isinstance(x, int) for x in di))):
        raise ValueError("device_indices must be an int or a list of int")
    
    if isinstance(di, list):
        for device_id in di:
            if device_id >= torch.cuda.device_count() or device_id < 0:
                raise ValueError(f"Invalid GPU index {device_id}. Available GPUs: {torch.cuda.device_count()}")
    else:
        # If it's a single int, check if it's valid
        if di >= torch.cuda.device_count() or di < 0:
            raise ValueError(f"Invalid GPU index {di}. Available GPUs: {torch.cuda.device_count()}") 
           
    si = runtime_flags["split_indices"]
    if np.isscalar(si):
        si = [int(si)]  # Convert scalar to list with one integer
    elif not isinstance(si, (list, np.ndarray)):
        raise ValueError("split_indices must be array-like (list or np.ndarray)")
    
    runtime_flags["split_indices"] = si

###############################################################################
# 4. check_and_convert_inputs (unchanged, still produces NumPy arrays)
###############################################################################
def check_and_convert_inputs(user_inputs, enabled_modules, precision, input_type, num_catchments=None):
    special_types = {
        "is_river_mouth": np.bool,
        "downstream_idx": np.int32,
    }
    if input_type == "param":
        all_required_keys, all_optional_keys, all_optional_defaults = gather_all_keys_and_defaults("param")
        all_scalar_params = gather_all_scalar_params()
        recognized_keys = (
            all_required_keys
            | all_optional_keys
            | set(all_scalar_params.keys())
        )
        if num_catchments is None:
            raise ValueError("num_catchments must be provided when checking parameters")
    else:
        all_required_keys, all_optional_keys, all_optional_defaults = gather_all_keys_and_defaults("state")
        all_scalar_params = {}
        recognized_keys = all_required_keys | all_optional_keys
    for k in user_inputs.keys():
        if k not in recognized_keys:
            raise ValueError(f"Unrecognized {input_type} key: {k}")
    def is_truly_required(key):
        required_by = []
        for mod_name in enabled_modules:
            mod_cfg = MODULES_CONFIG.get(mod_name, {})
            if input_type == "param":
                if key in mod_cfg.get("required_params", []):
                    required_by.append(mod_name)
            else:
                if key in mod_cfg.get("required_states", []):
                    required_by.append(mod_name)
        return len(required_by) > 0

    dtype = np.float32 if precision == "float32" else np.float64
    final_outputs = {}
    if input_type == "param":
        for sk, default_val in gather_all_scalar_params().items():
            final_outputs[sk] = user_inputs.get(sk, default_val)
    array_keys = (all_required_keys | all_optional_keys) - set(gather_all_scalar_params().keys())
    for key in array_keys:
        if key in user_inputs:
            arr = user_inputs[key]
            if key in special_types:
                arr = np.array(arr, dtype=special_types[key])
            else:
                if isinstance(arr, (list, tuple)):
                    arr = np.array(arr)
                if not isinstance(arr, (np.ndarray,)):
                    raise TypeError(f"{input_type.capitalize()} {key} must be array-like.")
                arr = np.array(arr, dtype=dtype)
            if arr.shape[0] != num_catchments:
                raise ValueError(
                    f"{input_type.capitalize()} {key} leading dimension {arr.shape[0]} != {num_catchments}."
                )
            final_outputs[key] = arr
        else:
            if is_truly_required(key):
                raise ValueError(f"Missing required {input_type}: {key}")
            default_val = all_optional_defaults.get(key, 0.0)
            final_outputs[key] = np.full((num_catchments,), default_val, dtype=dtype)
    if input_type == "state":
        if "river_storage" in final_outputs and "flood_storage" in final_outputs:
            final_outputs["total_storage"] = final_outputs["river_storage"] + final_outputs["flood_storage"]
    return final_outputs

###############################################################################
# 5. split_dict_arrays (unchanged)
###############################################################################
def split_dict_arrays(d, split_indices):
    splits = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            splits[k] = np.split(v, split_indices[:-1])
        else:
            splits[k] = [v] * len(split_indices)
    out = []
    for i in range(len(split_indices)):
        this_dict = {k: splits[k][i] for k in d}
        out.append(this_dict)
    return out

###############################################################################
# 6. gather_device_dicts 
###############################################################################

def gather_device_dicts(list_of_dicts):
    merged = {}
    for k in list_of_dicts[0].keys():
        vals = []
        for d in list_of_dicts:
            v = d[k]
            if isinstance(v, torch.Tensor):
                vals.append(v.cpu().numpy())  # Convert tensor to numpy array
            else:
                vals.append(v)
        first = vals[0]
        if isinstance(first, torch.Tensor) and first.numel() > 1:
            merged[k] = np.concatenate(vals, axis=0)  # Use numpy concatenate instead of torch.cat
        else:
            merged[k] = first
    return merged

###############################################################################
# 7. Model Preparation & Function Construction
###############################################################################
def prepare_model_and_function(user_params, user_states, step_fn, user_runtime_flags):
    validate_runtime_flags(user_runtime_flags)
    final_runtime_flags = user_runtime_flags
    device_type = final_runtime_flags["device"]
    device_indices = final_runtime_flags["device_indices"]
    split_indices = final_runtime_flags["split_indices"]
    enabled_modules = final_runtime_flags["modules"]
    precision = final_runtime_flags["precision"]

    if "river_storage" not in user_states:
        raise ValueError("river_storage is required in user_states")
    river_storage = user_states["river_storage"]
    num_catchments = (
        len(river_storage)
        if isinstance(river_storage, (list, tuple, np.ndarray))
        else river_storage.shape[0]
    )

    final_params_np = check_and_convert_inputs(user_params, enabled_modules, precision, "param", num_catchments)
    final_states_np = check_and_convert_inputs(user_states, enabled_modules, precision, "state", num_catchments)

    final_params_np_list = split_dict_arrays(final_params_np, split_indices)
    final_states_np_list = split_dict_arrays(final_states_np, split_indices)

    if isinstance(device_indices, int):
        device_indices = [device_indices]
    assert len(device_indices) == len(split_indices), "device_indices 和 split_indices 长度需匹配"

    final_params_list = []
    final_states_list = []
    for i, dev_idx in enumerate(device_indices):
        if device_type == "gpu":
            dev = torch.device(f"cuda:{dev_idx}")
        else:
            dev = torch.device("cpu")
        params_t = {}
        states_t = {}
        for k, v in final_params_np_list[i].items():
            if isinstance(v, np.ndarray):
                params_t[k] = torch.as_tensor(v, device=dev)
            else:
                params_t[k] = v
        for k, v in final_states_np_list[i].items():
            if isinstance(v, np.ndarray):
                states_t[k] = torch.as_tensor(v, device=dev)
            else:
                states_t[k] = v
        final_params_list.append(params_t)
        final_states_list.append(states_t)

    def parallel_fn(runtime_flags, params_list, init_states_list, forcing_list, input_matrix_list, dT):
        num_devices = len(params_list)
        if device_type == "gpu":
            streams = [torch.cuda.Stream(device=f"cuda:{dev}") for dev in device_indices]
            aggregator = [None] * num_devices
            for dev, stream in zip(device_indices, streams):
                with torch.cuda.stream(stream):
                    aggregator[dev] = step_fn(runtime_flags, params_list[dev], init_states_list[dev], input_matrix_list[dev] @ forcing_list[dev].to(device=f"cuda:{dev}"), dT)
            for s in streams:
                s.synchronize()
        else:
            raise NotImplementedError("Only GPU device is supported")
            # new_s, stats = step_fn(runtime_flags, params_list[0], states_list[0], forcing_list[0], dT)
            # outs_new_states[0] = new_s
            # outs_stats[0] = stats
        return aggregator

    return final_runtime_flags, final_params_list, final_states_list, parallel_fn

###############################################################################
# 8. Example Usage
###############################################################################
if __name__ == "__main__":
    user_runtime_flags = {
        "time_step": 86400,
        "unit_factor": 1000.0,
        "enable_adaptive_time_step": False,
        "precision": "float32",
        "modules": ["base", "foo_module"],
        "default_sub_iters": 24,
        "device": "gpu",
        "device_indices": [0],
        "split_indices": [6],  # means 2 devices, each gets 3
    }
    user_params = {
        "is_river_mouth": [True, False, True, False, True, False],
        "downstream_idx": [1, 2, 0, 4, 5, 3],
        "river_width": [100.0, 80.0, 60.0, 70.0, 65.0, 62.0],
        "river_length": [5000, 4000, 3000, 6000, 3200, 4500],
        "river_height": [15, 12, 10, 16, 13, 9],
        "river_elevation": [10, 9, 8, 11, 9.5, 8.2],
        "catchment_elevation": [12, 10, 9, 14, 11, 10],
        "downstream_distance": [1000, 1000, 0, 1100, 950, 1200],
        "river_max_storage": [10000, 8000, 6000, 9000, 8500, 9100],
        "river_area": [200, 180, 150, 170, 165, 168],
        "max_flood_area": [300, 250, 200, 350, 210, 215],
        "total_storage_table": [
            [0.0, 0.1, 0.2, 0.3],
            [0.0, 0.2, 0.4, 0.6],
            [0.0, 0.3, 0.6, 0.9],
            [0.0, 0.15, 0.3, 0.45],
            [0.0, 0.12, 0.24, 0.36],
            [0.0, 0.18, 0.36, 0.54],
        ],
        "flood_depth_table": [
            [0.0, 0.5, 1.0, 1.5],
            [0.0, 0.4, 0.8, 1.2],
            [0.0, 0.3, 0.6, 0.9],
            [0.0, 0.35, 0.7, 1.05],
            [0.0, 0.25, 0.5, 0.75],
            [0.0, 0.28, 0.56, 0.84],
        ],
        "total_width_table": [
            [10, 20, 30, 40],
            [15, 25, 35, 45],
            [20, 30, 40, 50],
            [12, 22, 32, 42],
            [13, 23, 33, 43],
            [14, 24, 34, 44],
        ],
        "flood_gradient_table": [
            [0.01, 0.02, 0.03, 0.04],
            [0.02, 0.03, 0.04, 0.05],
            [0.03, 0.04, 0.05, 0.06],
            [0.04, 0.05, 0.06, 0.07],
            [0.05, 0.06, 0.07, 0.08],
            [0.06, 0.07, 0.08, 0.09],
        ],
        "catchment_area": [100, 90, 80, 120, 105, 95],
        "foo_param": [1.0, 2.0, 3.0, 4.0, 2.0, 1.0],
    }
    user_states = {
        "river_storage": [10.0, 5.0, 2.0, 8.0, 7.0, 3.0],
        "foo_state": [999, 888, 777, 555, 444, 333],
    }
    
    def dummy_step_fn(runtime_flags, params, states, forcing, dT):
        new_states = {
            k: (v + 1.0 if isinstance(v, torch.Tensor) else v)
            for k, v in states.items()
        }
        scaling = dT / 3600.0
        new_states["river_storage"] = new_states["river_storage"] + forcing * scaling
        stats_out = {"river_storage_mean": states["river_storage"]}
        return new_states, stats_out

    final_rt, final_ps, final_ss, parallel_step = prepare_model_and_function(
        user_params, user_states, dummy_step_fn, user_runtime_flags
    )

    forcing_array = np.array([0.5,0.2,1.0,0.1,0.3,1.2], dtype=np.float32)
    spl = final_rt["split_indices"]
    forcing_np_list = np.split(forcing_array, spl[:-1])
    forcing_list = []
    for i, dev in enumerate(final_rt["device_indices"]):
        dev_t = torch.device(f"cuda:{dev}" if final_rt["device"]=="gpu" else "cpu")
        forcing_list.append(torch.as_tensor(forcing_np_list[i], dtype=torch.float32, device=dev_t))

    dT = 3600.0
    new_states_list, stats_list = parallel_step(final_rt, final_ps, final_ss, forcing_list, dT)

    new_states_cpu = gather_device_dicts(new_states_list)
    stats_cpu = gather_device_dicts(stats_list)
    print("New States CPU:", {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in new_states_cpu.items()})
    print("Stats Out CPU:", {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in stats_cpu.items()})