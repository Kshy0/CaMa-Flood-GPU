import torch
import numpy as np
from scipy.sparse import csr_matrix
from CMF_GPU.utils.Variables import MODULES_INFO, SCALAR_TYPES, SPECIAL_INPUT_SHAPES, SPECIAL_ARRAY_TYPES, MODULE_DEPENDENT_ARRAYS, HIDDEN_PARAMS, SPECIAL_HIDDEN_STATES, ORDERS, SPECIAL_SPLIT_ARRAYS, SPECIAL_SPLIT_SCALARS, LEGAL_AGG_ARRAYS
from CMF_GPU.utils.Variables import default_array_split, default_scalar_split, default_shape
from CMF_GPU.utils.utils import gather_all_keys

def check_input(user_params, user_states, enabled_modules, precision):
    dtype = np.float32 if precision == "float32" else np.float64
    # 1. check modules
    for user_inputs, input_type in [(user_params, "param"), (user_states, "state")]:
        all_keys, all_hidden_keys, all_scalar_keys = gather_all_keys(input_type, MODULES_INFO.keys())
        used_keys, hidden_keys, scalar_keys = gather_all_keys(input_type, enabled_modules)

        recognized_keys = set(all_keys + all_hidden_keys + all_scalar_keys)

        hidden_keys_list = [k for k in user_inputs if k in hidden_keys]
        unused_keys_list = [
            k for k in user_inputs
            if (k not in hidden_keys) and (k not in used_keys) and (k not in scalar_keys) and (k in recognized_keys)
        ]
        unrecognized_keys = [
            k for k in user_inputs
            if (k not in hidden_keys) and (k not in used_keys) and (k not in scalar_keys) and (k not in recognized_keys)
        ]
        if unrecognized_keys:
            raise ValueError(f"Unrecognized {input_type} key(s): {', '.join(unrecognized_keys)}")

        missing_keys = []
        for mod_name in enabled_modules:
            mod_cfg = MODULES_INFO[mod_name]
            if input_type == "param":
                required_keys = mod_cfg.get("params", []) + mod_cfg.get("scalar_params", [])
            else:
                required_keys = mod_cfg.get("states", [])

            for key in required_keys:
                if key not in user_inputs and key not in hidden_keys:
                    missing_keys.append((mod_name, key))
        
        if hidden_keys_list:
            print(f"Warning: The following {input_type} keys are hidden and will be ignored: {', '.join(hidden_keys_list)}")
            for key in hidden_keys_list:
                del user_inputs[key]
        if unused_keys_list:
            print(f"Warning: The following {input_type} keys are recognized but not used in the model: {', '.join(unused_keys_list)}")
            for key in unused_keys_list:
                del user_inputs[key]
        
        if missing_keys:
            missing_desc = "\n".join([f"  - {key} (required by module '{mod_name}')" for mod_name, key in missing_keys])
            raise ValueError(f"Missing required {input_type} keys:\n{missing_desc}")

        
    # 2. check types
    for user_inputs, input_type in [(user_params, "param"), (user_states, "state")]:
        used_keys, _, scalar_keys = gather_all_keys(input_type, enabled_modules)

        for key in scalar_keys:
            val = user_inputs[key]
            if not isinstance(val, (int, float)):
                raise TypeError(f"Expected a scalar value for '{key}', but got {type(val).__name__}.")
            user_inputs[key] = SCALAR_TYPES[key](val)

        for key in used_keys:
            val = user_inputs[key]
            if not isinstance(val, (np.ndarray, csr_matrix)):
                raise TypeError(f"Expected a NumPy array or CSR matrix, but got {type(val).__name__}.")

            if key in SPECIAL_INPUT_SHAPES:
                expected_shape = SPECIAL_INPUT_SHAPES[key](user_params)
            else:
                expected_shape = default_shape(user_params)

            if val.shape != expected_shape:
                raise ValueError(
                    f"{input_type.capitalize()} '{key}': expected shape {expected_shape}, but got shape {val.shape}."
                )

        expected_dtype = SPECIAL_ARRAY_TYPES.get(key, dtype)
        if val.dtype != expected_dtype:
            raise TypeError(
                f"'{key}': Expected a NumPy array of type {expected_dtype}, but got {val.dtype}."
            )
        for key in used_keys:
            if key in MODULE_DEPENDENT_ARRAYS:
                user_inputs[key] = MODULE_DEPENDENT_ARRAYS[key](user_params, enabled_modules)

def split_and_generate_hidden_arrays(user_params, user_states, enabled_modules, statistics, precision, devices):
    dtype = torch.float32 if precision == "float32" else torch.float64
    num_gpu = len(devices)

    orders = {}
    for key, func in ORDERS.items():
        if isinstance(key, tuple):
            results = func(user_params, orders)
            if all(r is not None for r in results):
                orders.update(dict(zip(key, results)))
        else:
            result = func(user_params, orders)
            if result is not None:
                orders[key] = result

    def process_input(input_data, input_type, ref_params_list=None):
        final_list = [dict() for _ in range(num_gpu)]
        used_keys, hidden_keys, scalar_keys = gather_all_keys(input_type, enabled_modules)

        for key in used_keys:
            splitter = SPECIAL_SPLIT_ARRAYS.get(key, default_array_split)
            pieces = splitter(input_data[key], orders, devices)
            for i in range(num_gpu):
                final_list[i][key] = pieces[i]
        for key in scalar_keys:
            splitter = SPECIAL_SPLIT_SCALARS.get(key, default_scalar_split)
            pieces = splitter(input_data[key], orders, devices)
            for i in range(num_gpu):
                final_list[i][key] = pieces[i]

        for i in range(num_gpu):
            for key in hidden_keys:
                if input_type == "param":
                    final_list[i][key] = HIDDEN_PARAMS[key](final_list[i])
                else:
                    ref_params = ref_params_list[i] if ref_params_list else {}
                    size = SPECIAL_HIDDEN_STATES[key](ref_params) if key in SPECIAL_HIDDEN_STATES else ref_params.get("num_catchments", 0)
                    final_list[i][key] = torch.zeros(size, dtype=dtype, device=devices[i])

        if input_type == "state" and statistics:
            for stat_type, keys in statistics.items():
                for key in keys:
                    full_key = f"{key}_{stat_type}"
                    shape_keys = LEGAL_AGG_ARRAYS.get((key,), ("num_catchments",))
                    size = [ref_params.get(k, 0) for k in shape_keys]
                    for i in range(num_gpu):
                        final_list[i][full_key] = torch.zeros(size, dtype=dtype, device=devices[i])


        return final_list

    final_params_list = process_input(user_params, "param")
    final_states_list = process_input(user_states, "state", ref_params_list=final_params_list)

    return final_params_list, final_states_list

def prepare_model_and_function(user_params, user_states, step_fn, simulation_config):

    device_type = simulation_config["device"]
    device_indices = simulation_config["device_indices"]
    precision = simulation_config["precision"]
    enabled_modules = simulation_config["modules"]
    statistics = simulation_config["statistics"]
    user_params["num_gpus"] = len(device_indices)
    devices = []
    if isinstance(device_indices, int):
        device_indices = [device_indices]
    for i in device_indices:
        devices.append(torch.device(f"cuda:{i}"))

    check_input(user_params, user_states, enabled_modules, precision)
    user_params, user_states = split_and_generate_hidden_arrays(user_params, user_states, enabled_modules, statistics, precision, devices)
    streams = [torch.cuda.Stream(device=dev) for dev in devices]
    def parallel_fn(simulation_config, params_list, states_list, forcing, dT, logger, agg_fn):
        if device_type == "gpu":
            for dev, stream, params, states in zip(devices, streams, params_list, states_list):
                with torch.cuda.stream(stream):
                    runoff = params["runoff_input_matrix"] @ forcing.to(device=dev)
                    step_fn(simulation_config, params, states, runoff, dT, logger, agg_fn)
            for s in streams:
                s.synchronize()
        else:
            raise NotImplementedError("Currently, only GPU device is supported")

    return user_params, user_states, parallel_fn