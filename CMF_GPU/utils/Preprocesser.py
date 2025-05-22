import h5py
import numpy as np
import torch
from CMF_GPU.utils.Variables import MODULES_INFO, HIDDEN_PARAMS, SPECIAL_ARRAY_TYPES, SCALAR_TYPES
from CMF_GPU.utils.Aggregator import LEGAL_AGG_ARRAYS, LEGAL_STATS
from CMF_GPU.utils.Checker import SPECIAL_INPUT_SHAPES, SPECIAL_HIDDEN_STATE_SHAPES, default_shape
from CMF_GPU.utils.Spliter import ORDERS, SPECIAL_SPLIT_ARRAYS, SPECIAL_SPLIT_SCALARS, default_scalar_split, default_array_split
from CMF_GPU.utils.utils import gather_all_keys

def check_input_h5(params_filename, states_filename, enabled_modules, precision):
    dtype = np.float32 if precision == "float32" else np.float64
    # 1. check modules
    params_h5 = h5py.File(params_filename, 'r', swmr=True)
    states_h5 = h5py.File(states_filename, 'r', swmr=True)
    for input_type, data_h5 in [("param", params_h5), ("state", states_h5)]:
        all_keys, all_hidden_keys, all_scalar_keys = gather_all_keys(input_type, MODULES_INFO.keys())
        used_keys, hidden_keys, scalar_keys = gather_all_keys(input_type, enabled_modules)

        recognized_keys = set(all_keys + all_hidden_keys + all_scalar_keys)

        # Open the file and read only the keys and shapes/dtypes
        hidden_keys_list = []
        missing_keys = []
        unused_keys_list = [
            k for k in data_h5
            if (k not in hidden_keys) and (k not in used_keys) and (k not in scalar_keys) and (k in recognized_keys)
        ]
        unrecognized_keys = [
            k for k in data_h5
            if (k not in hidden_keys) and (k not in used_keys) and (k not in scalar_keys) and (k not in recognized_keys)
        ]
        
        if unrecognized_keys:
            raise ValueError(f"Unrecognized {input_type} key(s): {', '.join(unrecognized_keys)}")
        for key in data_h5.keys():
            if key in hidden_keys:
                hidden_keys_list.append(key)
            elif key in used_keys:
                if key in scalar_keys:
                    shape = data_h5[key].shape
                    if shape != ():
                        raise ValueError(f"expected a scalar for '{key}', but found shape {shape}.")
                    if isinstance(data_h5[key], SCALAR_TYPES.get(key, dtype)):
                        raise TypeError(f"'{key}': expected a scalar type {SCALAR_TYPES.get(key, dtype)}, but got {type(data_h5[key])}.")
                else:
                    shape = data_h5[key].shape
                    expected_shape = SPECIAL_INPUT_SHAPES.get(key, default_shape)(params_h5)
                    if shape != expected_shape:
                        raise ValueError(
                            f"{input_type.capitalize()} '{key}': expected shape {expected_shape}, but got shape {shape}."
                        )

                    dtype_in_file = data_h5[key].dtype
                    expected_dtype = SPECIAL_ARRAY_TYPES.get(key, dtype)
                    if dtype_in_file != expected_dtype:
                        raise TypeError(
                            f"'{key}': Expected a NumPy array of type {expected_dtype}, but got {dtype_in_file}."
                        )
            elif key not in recognized_keys:
                unrecognized_keys.append(key)

        for mod_name in enabled_modules:
            mod_cfg = MODULES_INFO[mod_name]
            if input_type == "param":
                required_keys = mod_cfg.get("params", []) + mod_cfg.get("scalar_params", [])
            else:
                required_keys = mod_cfg.get("states", [])

            for key in required_keys:
                if key not in data_h5.keys() and key not in hidden_keys:
                    missing_keys.append((mod_name, key))

        # Issue warnings for hidden or unused keys
        if hidden_keys_list:
            print(f"Warning: The following {input_type} keys are hidden and will be ignored: {', '.join(hidden_keys_list)}")

        if unused_keys_list:
            print(f"Warning: The following {input_type} keys are recognized but not used in the model: {', '.join(unused_keys_list)}")

        if unrecognized_keys:
            raise ValueError(f"Unrecognized {input_type} key(s): {', '.join(unrecognized_keys)}")

        if missing_keys:
            missing_desc = "\n".join([f"  - {key} (required by module '{mod_name}')" for mod_name, key in missing_keys])
            raise ValueError(f"Missing required {input_type} keys:\n{missing_desc}")

def make_order(params_filename, enabled_modules, statistics, num_gpus):
    params_h5 = h5py.File(params_filename, 'r', swmr=True)
    orders = {}
    orders["statistics"] = statistics
    orders["modules"] = enabled_modules

    for (mod, keys), func in ORDERS.items():
        if mod in enabled_modules:
            results = func(params_h5, orders, num_gpus)
            if isinstance(keys, tuple):                  
                orders.update(dict(zip(keys, results)))
            else:
                orders[keys] = results
    return orders

def load_input_h5(params_filename, states_filename, orders, rank):
    # Load the input h5 files
    params_h5 = h5py.File(params_filename, 'r', swmr=True)
    states_h5 = h5py.File(states_filename, 'r', swmr=True)
    statistics = orders["statistics"]
    params = {}
    states = {}
    for input_type, data_h5, data_dict in [("param", params_h5, params), ("state", states_h5, states)]:
        used_keys, hidden_keys, scalar_keys = gather_all_keys(input_type, orders["modules"])
        for key in scalar_keys:
            type_fcn = SCALAR_TYPES[key]
            if key in SPECIAL_SPLIT_SCALARS:
                data_dict[key] = type_fcn(SPECIAL_SPLIT_SCALARS[key](data_h5[key], orders, rank))
            else:
                data_dict[key] = type_fcn(default_scalar_split(data_h5[key], orders, rank))
        for key in used_keys:
            if key in SPECIAL_SPLIT_ARRAYS:
                data_dict[key] = SPECIAL_SPLIT_ARRAYS[key](data_h5[key], orders, rank)
            else:
                data_dict[key] = default_array_split(data_h5[key], orders, rank)
        for key in hidden_keys:
            if input_type == "param":
                data_dict[key] = HIDDEN_PARAMS[key](params) # data_dict
            else:
                if key in SPECIAL_HIDDEN_STATE_SHAPES:
                    data_dict[key] = torch.zeros(SPECIAL_HIDDEN_STATE_SHAPES[key](params)) # TODO: ensure dtype
                else:
                    data_dict[key] = torch.zeros(default_shape(params))
        # agg keys
        if input_type == "state" and statistics:
            for stat_type, keys in statistics.items():
                for key in keys:
                    full_key = f"{key}_{stat_type}"
                    shape_keys = LEGAL_AGG_ARRAYS.get((key,), ("num_catchments",))
                    size = [params.get(k, 0) for k in shape_keys]
                    data_dict[full_key] = torch.zeros(size)

    return params, states


if __name__ == "__main__":
    from pathlib import Path
    input_dir = Path("/home/eat/CaMa-Flood-GPU/inp/glb_06min")
    check_input_h5(input_dir / "parameters.h5", input_dir / "init_states.h5", ["base"], "float32")
    orders = make_order(input_dir / "parameters.h5", ["base"], [],2)
    params, states = load_input_h5(input_dir / "parameters.h5", input_dir / "init_states.h5", orders, 0)
    pass