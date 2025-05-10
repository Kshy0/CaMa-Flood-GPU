import torch
import numpy as np
from CMF_GPU.utils.Variables import MODULES, MODULES_CONFIG, RUNTIME_FLAGS_REQUIRED_KEYS, SCALAR_TYPES, SPECIAL_ARRAY_SHAPES, SPECIAL_ARRAY_TYPES, SPECIAL_HIDDEN_PARAMS
from CMF_GPU.utils.utils import gather_all_keys_and_defaults, split_dict_arrays

###############################################################################
# Runtime Flags Validation
###############################################################################
def validate_runtime_flags(runtime_flags):
    # Check for unexpected keys
    unexpected_keys = set(runtime_flags.keys()) - set(RUNTIME_FLAGS_REQUIRED_KEYS)
    if unexpected_keys:
        raise ValueError(f"Unexpected runtime flag(s): {unexpected_keys}")
    
    for rk in RUNTIME_FLAGS_REQUIRED_KEYS:
        if rk not in runtime_flags:
            raise ValueError(f"Missing required runtime flag: {rk}")

    for md in runtime_flags["modules"]:
        if md not in MODULES:
            raise ValueError(f"Unknown module '{md}' in runtime_flags['modules']")

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
# check_and_convert_inputs
###############################################################################
def check_and_convert_inputs(user_inputs, enabled_modules, precision, input_type, shape_info=None):
    all_params_or_states, all_hidden_keys, all_scalar_keys = gather_all_keys_and_defaults(input_type, MODULES)
    recognized_keys = all_params_or_states + all_hidden_keys + all_scalar_keys

    params_or_states, hidden_keys, scalar_keys = gather_all_keys_and_defaults(input_type, enabled_modules)
    hidden_keys_list = []
    unused_keys_list = []

    for k in user_inputs.keys():
        if k in hidden_keys:
            hidden_keys_list.append(k)
        else:
            if k not in params_or_states and k not in scalar_keys:
                if k not in recognized_keys:
                    raise ValueError(f"Unrecognized {input_type} key: {k}")
                else:
                    unused_keys_list.append(k)

    if hidden_keys_list:
        print(f"Warning: The following {input_type} keys are hidden and will be ignored: {', '.join(hidden_keys_list)}")
    if unused_keys_list:
        print(f"Warning: The following {input_type} keys are recognized but not used in the model: {', '.join(unused_keys_list)}")

    # Collect missing required keys
    missing_keys = []
    for mod_name in enabled_modules:
        mod_cfg = MODULES_CONFIG.get(mod_name, {})
        if input_type == "param":
            required_keys = set(mod_cfg.get("params", [])) | set(mod_cfg.get("scalar_params", []))
        else:
            required_keys = set(mod_cfg.get("states", []))

        for key in required_keys:
            if key not in user_inputs and key not in hidden_keys:
                missing_keys.append((mod_name, key))

    if missing_keys:
        missing_desc = "\n".join([f"  - {key} (required by module '{mod_name}')" for mod_name, key in missing_keys])
        raise ValueError(f"Missing required {input_type} keys:\n{missing_desc}")

    dtype = np.float32 if precision == "float32" else np.float64
    final_outputs = {}

    # Process scalar values
    for key in scalar_keys:
        val = user_inputs[key]
        if not isinstance(val, (int, float)):
            raise TypeError(f"Expected a scalar value for '{key}', but got {type(val).__name__}.")
        final_outputs[key] = SCALAR_TYPES[key](val)

    if input_type == "param":
        shape_info = final_outputs

    def check_array_shape(key, val):
        if key in SPECIAL_ARRAY_SHAPES:
            expected_shape = SPECIAL_ARRAY_SHAPES[key](shape_info)
            if val.shape != expected_shape:
                raise ValueError(
                    f"{input_type.capitalize()} '{key}': expected shape {expected_shape}, but got shape {val.shape}."
                )
        else:
            if not (val.shape[0] == shape_info["num_catchments"] and len(val.shape) == 1):
                raise ValueError(
                    f"{input_type.capitalize()} '{key}': expected leading dimension = {shape_info['num_catchments']} "
                    f"and array to be 1D, but got shape {val.shape}."
                )
    
    def convert_array_type(key, val):
        if key in SPECIAL_ARRAY_TYPES:
            if val.dtype != SPECIAL_ARRAY_TYPES[key]:
                raise TypeError(
                    f"'{key}': Expected a NumPy array of type {SPECIAL_ARRAY_TYPES[key]}, but got {val.dtype}."
                )
        else:
            val = val.astype(dtype)

        return val

    # Process array values
    for key in params_or_states:
        val = user_inputs[key]
        if not isinstance(val, np.ndarray):
            raise TypeError(f"Expected a NumPy array, but got {type(val).__name__}.")  
             
        check_array_shape(key, val)
        final_outputs[key] = convert_array_type(key, val)

    # Init hidden values
    for key in hidden_keys:
        if key in SPECIAL_HIDDEN_PARAMS:
            final_outputs[key] = SPECIAL_HIDDEN_PARAMS[key](final_outputs)
        else:
            final_outputs[key] = np.zeros((shape_info["num_catchments"],), dtype=dtype)
        
        check_array_shape(key, final_outputs[key])
        final_outputs[key] = convert_array_type(key, final_outputs[key])

    
    return final_outputs



###############################################################################
# Model Preparation & Function Construction
###############################################################################
def prepare_model_and_function(user_params, user_states, step_fn, user_runtime_flags):
    validate_runtime_flags(user_runtime_flags)
    final_runtime_flags = user_runtime_flags
    device_type = final_runtime_flags["device"]
    device_indices = final_runtime_flags["device_indices"]
    split_indices = final_runtime_flags["split_indices"]
    enabled_modules = final_runtime_flags["modules"]
    precision = final_runtime_flags["precision"]

    final_params_np = check_and_convert_inputs(user_params, enabled_modules, precision, "param")
    final_states_np = check_and_convert_inputs(user_states, enabled_modules, precision, "state", shape_info=final_params_np)

    final_params_np_list = split_dict_arrays(final_params_np, split_indices)
    final_states_np_list = split_dict_arrays(final_states_np, split_indices)

    if isinstance(device_indices, int):
        device_indices = [device_indices]

    if len(device_indices) != len(split_indices):
        raise ValueError("The lengths of device_indices and split_indices need to match.")

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

    def parallel_fn(runtime_flags, params_list, init_states_list, forcing_list, input_matrix_list, dT, logger):
        if device_type == "gpu":
            streams = [torch.cuda.Stream(device=f"cuda:{dev}") for dev in device_indices]
            for dev, stream in zip(device_indices, streams):
                with torch.cuda.stream(stream):
                    step_fn(runtime_flags, params_list[dev], init_states_list[dev], input_matrix_list[dev] @ forcing_list[dev].to(device=f"cuda:{dev}"), dT, logger)
            for s in streams:
                s.synchronize()
        else:
            raise NotImplementedError("Only GPU device is supported")
            # new_s, stats = step_fn(runtime_flags, params_list[0], states_list[0], forcing_list[0], dT)
            # outs_new_states[0] = new_s
            # outs_stats[0] = stats

    return final_runtime_flags, final_params_list, final_states_list, parallel_fn

###############################################################################
# Example Usage
###############################################################################
if __name__ == "__main__":
    from CMF_GPU.utils.Variables import gather_device_dicts
    user_runtime_flags = {
        "time_step": 86400,
        "unit_factor": 1000.0,
        "enable_logging": False,
        "precision": "float32",
        "modules": ["base", "adaptive_time_step"],
        "default_sub_iters": 24,
        "device": "gpu",
        "device_indices": [0],
        "split_indices": [6],  # means 2 devices, each gets 3
    }
    user_params = {
        "num_catchments": 6,
        "num_flood_levels": 1,
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
            [0.01, 0.02, 0.03],
            [0.02, 0.03, 0.04],
            [0.03, 0.04, 0.05],
            [0.04, 0.05, 0.06],
            [0.05, 0.06, 0.07],
            [0.06, 0.07, 0.08],
        ],
        "catchment_area": [100, 90, 80, 120, 105, 95],
        "river_manning": [0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
        "flood_manning": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "gravity": 9.81,              
        "adaptation_factor": 0.7,              
    }
    user_states = {
        "river_storage": [10.0, 5.0, 2.0, 8.0, 7.0, 3.0],
        "flood_storage": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "river_depth": [1.0, 0.8, 0.5, 0.9, 0.7, 0.6],
        "river_outflow": [0.5, 0.4, 0.2, 0.6, 0.5, 0.3],
        "flood_depth": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "flood_area": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "flood_outflow": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "river_cross_section_depth": [0.8, 0.7, 0.4, 0.8, 0.6, 0.5],
        "flood_cross_section_depth": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "flood_cross_section_area": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    for key, value in user_params.items():
        if not isinstance(value, (int, float)):
            user_params[key] = np.array(value)
    for key, value in user_states.items():
        if not isinstance(value, (int, float)):
            user_states[key] = np.array(value)            
    def dummy_step_fn(runtime_flags, params, states, forcing, dT, logger):
        new_states = {
            k: (v + 1.0 if isinstance(v, torch.Tensor) else v)
            for k, v in states.items()
        }
        scaling = dT / 3600.0
        new_states["river_storage"] = new_states["river_storage"] + forcing * scaling
        agg_out = {"river_storage_mean": states["river_storage"]}
        return agg_out

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
    parallel_step(final_rt, final_ps, final_ss, forcing_list, torch.rand(6, 6).to(device="cuda:0"), dT, 0)
    states_cpu = gather_device_dicts(final_ss)

    print("Stats Out CPU:", final_ss)