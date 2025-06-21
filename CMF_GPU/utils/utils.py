import torch
import h5py
import functools
import os
from torch import distributed as dist
from CMF_GPU.utils.Variables import MODULES_INFO, SCALAR_TYPES

def get_global_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0

def get_local_rank():
    if dist.is_available() and dist.is_initialized():
        return int(os.environ.get("LOCAL_RANK", 0))
    else:
        return 0

def is_rank_zero():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    else:
        return True 

def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1

def check_enabled(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.disabled:
            return None
        return func(self, *args, **kwargs)
    return wrapper

def only_rank_zero(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not dist.is_initialized() or get_global_rank() == 0:
            return func(*args, **kwargs)
    return wrapper

def gather_all_keys(input_type, modules):
    assert input_type in ["param", "state"], "input_type must be 'param' or 'state'"
    all_params_or_states = []
    all_hidden = []
    all_scalar = []
    
    # Determine which keys to use based on input_type
    if input_type == "param":
        params_or_states_key = "params"
        hidden_key = "hidden_params"
        scalar_key = "scalar_params"
    else:  # state
        params_or_states_key = "states"
        hidden_key = "hidden_states"
    
    for mod_name, mod_cfg in MODULES_INFO.items():
        # Process params or states
        if mod_name in modules:
            for k in mod_cfg.get(params_or_states_key, []):
                all_params_or_states.append(k)
            
            # Process hidden keys
            for hk in mod_cfg.get(hidden_key, []):
                all_hidden.append(hk)
            
            if input_type == "param":
                for sk in mod_cfg.get(scalar_key, []):
                    all_scalar.append(sk)
    
    return all_params_or_states, all_hidden, all_scalar

def gather_device_dicts(data_dict, keys=None):
    merged = {}

    for k in data_dict:
        if keys is not None and k not in keys:
            continue
        value = data_dict[k]
        if isinstance(value, torch.Tensor):
            merged[k] = value.detach().cpu().numpy()
        else:
            merged[k] = value

    return merged

def snapshot_to_h5(filename, data_dict, input_type, modules, precision, omit_hidden=True):
    """
    Save a snapshot of params or states to an .h5 file, excluding hidden entries.

    Args:
        filename (str): Output .h5 file path.
        data_dict (dict): The params or states dictionary to snapshot. (numpy arrays)
        input_type (str): Either "param" or "state".
        modules (list): Enabled modules, used to determine hidden keys.
        omit_hidden (bool): If True, hidden keys will be omitted.
    """
    # Get hidden keys to exclude
    params_or_states, hidden_keys, scalar_keys = gather_all_keys(input_type, modules)

    saved_keys = []

    with h5py.File(filename, 'w') as h5f:
        h5f.attrs['default_dtype'] = precision
        for k in params_or_states:
            if k in data_dict:
                h5f.create_dataset(k, data=data_dict[k])
                saved_keys.append(k)
            else:
                raise ValueError(f"Key '{k}' not found in data_dict.")

        if not omit_hidden:
            for k in hidden_keys:
                if k in data_dict:
                    h5f.create_dataset(k, data=data_dict[k])
                    saved_keys.append(k)
                else:
                    raise ValueError(f"Key '{k}' not found in data_dict.")

        for k in scalar_keys:
            if k in data_dict:
                value = data_dict[k]
                h5f.create_dataset(k, data=value) 
                saved_keys.append(k)
            else:
                raise ValueError(f"Key '{k}' not found in data_dict.")


    print(f"Saved keys ({len(saved_keys)}): {saved_keys}")
    
def load_from_h5(filename, input_type, modules, omit_hidden=True):
    """
    Load a snapshot from a .h5 file.

    Args:
        filename (str): Input .h5 file path.
        input_type (str): Either "param" or "state".
        modules (list): Enabled modules, used to determine hidden keys.
        omit_hidden (bool): If True, hidden keys will be omitted.

    Returns:
        dict: Loaded data.
    """
    params_or_states, hidden_keys, scalar_keys = gather_all_keys(input_type, modules)
    loaded_data = {}

    with h5py.File(filename, 'r') as h5f:
        for k in params_or_states:
            if k in h5f:
                loaded_data[k] = h5f[k][()]
            else:
                raise ValueError(f"Key '{k}' not found in file.")

        if not omit_hidden:
            for k in hidden_keys:
                if k in h5f:
                    loaded_data[k] = h5f[k][()]
                else:
                    raise ValueError(f"Key '{k}' not found in file.")

        for k in scalar_keys:
            if k in h5f:
                loaded_data[k] = SCALAR_TYPES[k](h5f[k][()])
            else:
                raise ValueError(f"Key '{k}' not found in file.")

    print(f"Loaded keys ({len(loaded_data)}): {list(loaded_data.keys())}")
    return loaded_data
