import numpy as np
import torch
from CMF_GPU.utils.Variables import MODULES_INFO, SCALAR_TYPES

def check_enabled(func):
    def wrapper(self, *args, **kwargs):
        if self.disabled:
            return None
        return func(self, *args, **kwargs)
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

def gather_device_dicts(list_of_dicts, keys=None):
    merged = {}
    all_keys = list_of_dicts[0].keys()

    for k in all_keys:
        if keys is not None and k not in keys:
            continue
        values = [d[k] for d in list_of_dicts]

        if isinstance(values[0], torch.Tensor):
            if len(values) == 1:
                merged[k] = values[0].detach().cpu().numpy()
            else:
                merged[k] = torch.cat([v.detach().cpu() for v in values], dim=0).numpy()
        else:
            merged[k] = values[0]

    return merged

def snapshot_to_npz(filename, data_dict, input_type, modules, omit_hidden=True):
    """
    Save a snapshot of params or states to a .npz file, excluding hidden entries.

    Args:
        data_dict (dict): The params or states dictionary to snapshot. (numpy arrays)
        input_type (str): Either "param" or "state".
        modules (list): Enabled modules, used to determine hidden keys.
        filename (str): Output .npz file path.
    """
    # Get hidden keys to exclude
    params_or_states, hidden_keys, scalar_keys = gather_all_keys(input_type, modules)

    snapshot = {}
    saved_keys = []

    for k in params_or_states:
        if k in data_dict:
            snapshot[k] = data_dict[k]
            saved_keys.append(k)
        else:
            raise ValueError(f"Key '{k}' not found in data_dict.")
    if not omit_hidden:
        for k in hidden_keys:
            if k in data_dict:
                snapshot[k] = data_dict[k]
                saved_keys.append(k)
            else:
                raise ValueError(f"Key '{k}' not found in data_dict.")
    for k in scalar_keys:
        if k in data_dict:
            snapshot[k] = data_dict[k]
            saved_keys.append(k)
        else:
            raise ValueError(f"Key '{k}' not found in data_dict.")
    np.savez(filename, **snapshot)

    print(f"Saved keys ({len(saved_keys)}): {saved_keys}")
    
def load_from_npz(filename, input_type, modules, omit_hidden=True):
    """
    Load a snapshot from a .npz file.

    Args:
        filename (str): Input .npz file path.

    Returns:
        dict: Loaded data.
    """
    params_or_states, hidden_keys, scalar_keys = gather_all_keys(input_type, modules)
    loaded_data = {}
    with np.load(filename, allow_pickle=True) as data:
        for k in params_or_states:
            if k in data:
                loaded_data[k] = data[k]
            else:
                raise ValueError(f"Key '{k}' not found in data.")
        if not omit_hidden:
            for k in hidden_keys:
                if k in data and not omit_hidden:
                    loaded_data[k] = data[k]
                else:
                    raise ValueError(f"Key '{k}' not found in data.")
        for k in scalar_keys:
            if k in data:
                loaded_data[k] = SCALAR_TYPES[k](data[k])
            else:
                raise ValueError(f"Key '{k}' not found in data.")
    print(f"Loaded keys ({len(loaded_data)}): {list(loaded_data.keys())}")

    return loaded_data