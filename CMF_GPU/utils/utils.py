import numpy as np
import torch
import pickle
from CMF_GPU.utils.Variables import MODULES_CONFIG

def check_enabled(func):
    def wrapper(self, *args, **kwargs):
        if self.disabled:
            return None
        return func(self, *args, **kwargs)
    return wrapper


def gather_all_keys_and_defaults(input_type, modules):
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
    
    for mod_name, mod_cfg in MODULES_CONFIG.items():
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

def gather_device_dicts(list_of_dicts, keys=None):
    merged = {}
    for k in list_of_dicts[0].keys():
        if keys is not None and k not in keys:
            continue
        if isinstance(list_of_dicts[0][k], torch.Tensor):
            merged[k] = np.concatenate([d[k].cpu().detach().numpy() for d in list_of_dicts], axis=0)
        else:
            merged[k] = list_of_dicts[0][k]
            
    return merged

def load_csr_list_from_pkl(filename, device_indices):
    """
    Load sparse tensors from a pickle file as COO format and convert them to CSR format.
    
    Args:
        filename: Input pickle filename
        device_indices: List of device indices to place tensors on
    
    Returns:
        List of CSR sparse tensors
    """
    with open(filename, 'rb') as f:
        serializable_list = pickle.load(f)
    
    csr_list = []
    for data, device in zip(serializable_list, device_indices):
        # First load as COO tensor
        coo = torch.sparse_coo_tensor(
            data['indices'], 
            data['values'], 
            data['shape'], 
            dtype=data['dtype'], 
            device=f"cuda:{device}"
        )
        # Convert to CSR format
        csr = coo.to_sparse_csr()
        csr_list.append(csr)
    
    return csr_list

def snapshot_to_pkl(data_dict, input_type, modules, filename, omit_hidden=True):
    """
    Save a snapshot of params or states to a .pkl file, excluding hidden entries.

    Args:
        data_dict (dict): The params or states dictionary to snapshot. (numpy arrays)
        input_type (str): Either "param" or "state".
        modules (list): Enabled modules, used to determine hidden keys.
        filename (str): Output .pkl file path.
    """
    
    # Get hidden keys to exclude
    params_or_states, hidden_keys, scalar_keys = gather_all_keys_and_defaults(input_type, modules)

    snapshot = {}
    saved_keys = []
    ignored_keys = []

    for k in data_dict.keys():
        if k in hidden_keys and omit_hidden:
            ignored_keys.append(k)
            continue
        if k in scalar_keys or k in params_or_states or (k in hidden_keys and not omit_hidden):
            snapshot[k] = data_dict[k]
            saved_keys.append(k)
        else:
            raise ValueError(f"Key '{k}' not found in params or states.")
        
    with open(filename, "wb") as f:
        pickle.dump(snapshot, f)

    print(f"Saved keys ({len(saved_keys)}): {saved_keys}")
    if omit_hidden:
        print(f"Ignored hidden keys ({len(ignored_keys)}): {ignored_keys}")