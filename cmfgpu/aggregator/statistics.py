# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

import torch

from cmfgpu.models.utils import torch_to_numpy_dtype


class StatisticsMixin:
    """Mixin providing statistics initialization, kernel state preparation, and update logic."""

    def _prepare_kernel_states(self) -> None:
        """Pre-compute and cache all tensors required for kernel execution."""
        required_tensors: Dict[str, torch.Tensor] = {}

        def get_dependencies(expr: str) -> Set[str]:
            tokens = set(re.findall(r'\b[a-zA-Z_]\w*\b', expr))
            deps = set()
            for token in tokens:
                if token in self._tensor_registry:
                    deps.add(token)
                elif token in self._field_registry:
                    f_info = self._field_registry[token]
                    cat = getattr(f_info, 'json_schema_extra', {}).get('category')
                    if cat == 'virtual':
                        sub = getattr(f_info, 'json_schema_extra', {}).get('expr')
                        if sub:
                            deps.update(get_dependencies(sub))
            return deps

        # Add original variables and their output buffers
        for var_name, ops in self._variable_ops.items():
            field_info = self._field_registry.get(var_name)
            json_extra = getattr(field_info, 'json_schema_extra', {})
            category = json_extra.get('category', 'param')
            
            if category == 'virtual':
                expr = json_extra.get('expr')
                if expr:
                    deps = get_dependencies(expr)
                    for dep in deps:
                        if dep in self._tensor_registry:
                            required_tensors[dep] = self._tensor_registry[dep]
            elif var_name in self._tensor_registry:
                required_tensors[var_name] = self._tensor_registry[var_name]

            for op in ops:
                out_name = f"{var_name}_{op}"
                required_tensors[out_name] = self._storage[out_name]
                
                # For explicit argmax/argmin operations, add their auxiliary storage
                op_parts = op.split('_')
                outer_op = op_parts[0]
                arg_match = re.match(r'arg(max|min)(\d*)$', outer_op)
                if arg_match:
                    arg_type = arg_match.group(1)
                    arg_k_str = arg_match.group(2)
                    aux_name = f"{var_name}_{arg_type}{arg_k_str or ''}_aux"
                    if aux_name in self._storage:
                        required_tensors[aux_name] = self._storage[aux_name]
                
                # Add inner states for compound ops
                if '_' in op:
                    parts = op.split('_')
                    inner = parts[1]
                    # 'last' inner op doesn't need cross-step state
                    if inner != 'last':
                        inner_name = f"{var_name}_{inner}_inner_state"
                        if inner_name in self._storage:
                            required_tensors[inner_name] = self._storage[inner_name]
                        if inner == 'mean':
                            w_name = f"{var_name}_{inner}_weight_state"
                            if w_name in self._storage:
                                required_tensors[w_name] = self._storage[w_name]

        # Collect required dimensions and save indices
        required_dims: Set[str] = set()
        required_save_indices: Set[str] = set()
        for var_name in self._variables:
            field_info = self._field_registry[var_name]
            json_schema_extra = getattr(field_info, 'json_schema_extra', {})
            tensor_shape = json_schema_extra.get('tensor_shape', ())
            save_idx = json_schema_extra.get('save_idx')
            if save_idx:
                required_save_indices.add(save_idx)
            for dim_name in tensor_shape:
                if isinstance(dim_name, str):
                    required_dims.add(dim_name)

        # Add save_idx tensors
        for save_idx in required_save_indices:
            if save_idx in self._tensor_registry:
                required_tensors[save_idx] = self._tensor_registry[save_idx]
            else:
                raise RuntimeError(f"Save index tensor '{save_idx}' not registered")

        # Add dimension tensors/scalars
        for dim_name in required_dims:
            if dim_name in self._tensor_registry:
                tensor = self._tensor_registry[dim_name]
                if isinstance(tensor, (int, float)):
                    required_tensors[dim_name] = torch.tensor(tensor, device=self.device)
                else:
                    required_tensors[dim_name] = tensor

        self._kernel_states = required_tensors

    

    def initialize_statistics(self, variable_ops: Dict[str, List[str]]) -> None:
        """Initialize aggregation tensors and metadata for provided variables and ops."""
        # Reset generic state
        self._variables = set()
        # Normalize to lower-case list for each variable
        self._variable_ops = {}
        for var, ops in variable_ops.items():
            if ops is None:
                ops_list = ["mean"]
            elif isinstance(ops, str):
                ops_list = [ops]
            else:
                ops_list = list(ops)
            self._variable_ops[var] = [str(o).lower() for o in ops_list]
        self._storage.clear()
        self._output_keys = []
        self._metadata.clear()
        self._output_is_outer: Dict[str, bool] = {}
        
        self._aggregator_function = None
        self._aggregator_generated = False
        self._kernel_states = None
        self._current_macro_step_count = 0.0

        # Clean up old temporary files
        self._cleanup_temp_files()

        # Validate and setup each variable
        for var_name, ops in self._variable_ops.items():
            # Note: argmax/argmin cannot be user-specified operations.
            # They are automatically created as auxiliary storage when max/min is requested.
            # The argmax/argmin values are stored as integer indices (macro_step_index),
            # which are converted to NC time values (days since epoch) when written to the
            # output files. This creates a time-series-like output where each extreme value
            # record also has its corresponding occurrence time.
            import re

            # Sort ops to ensure consistent processing order
            ops.sort()
            
            tensor = None
            field_info = self._field_registry[var_name]
            json_schema_extra = getattr(field_info, 'json_schema_extra', {})
            category = json_schema_extra.get('category', 'param')
            is_virtual = category == 'virtual'

            if var_name in self._tensor_registry:
                tensor = self._tensor_registry[var_name]
            elif is_virtual:
                tensor = None
            else:
                raise ValueError(f"Variable '{var_name}' not registered. Call register_tensor() first.")
            
            target_dtype = tensor.dtype if tensor is not None else torch.float32

            tensor_shape = json_schema_extra.get('tensor_shape', ())
            save_idx = json_schema_extra.get('save_idx')
            description = getattr(field_info, 'description', f"Variable {var_name}")
            save_coord = json_schema_extra.get('save_coord')

            if not save_idx:
                raise ValueError(f"Variable '{var_name}' must have save_idx in json_schema_extra")

            if save_idx in self._tensor_registry:
                ref_save_idx = self._tensor_registry[save_idx]
                if tensor is not None:
                     # Real tensor shape/dim logic
                     tensor_ndim = tensor.ndim
                     tensor_base_shape = tensor.shape
                else:
                     # Virtual tensor logic
                     # Infer ndim/shape from tensor_shape or dependencies
                     tensor_ndim = 1 + (1 if self.num_trials > 1 else 0) # Base guess
                     if len(tensor_shape) > 1: # Has extra dims
                          tensor_ndim += (len(tensor_shape) - 1)
                     
                     # Construct hypothetical shape for allocation size
                     tensor_base_shape = (len(ref_save_idx),) # minimum
                     if len(tensor_shape) > 1:
                           # Try to resolve dimensions from dependencies or registry
                           expr = json_schema_extra.get('expr')
                           toks = re.findall(r'\b[a-zA-Z_]\w*\b', expr)
                           found_dep = False
                           for t in toks:
                                if t in self._tensor_registry:
                                     dep = self._tensor_registry[t]
                                     tensor_ndim = dep.ndim
                                     tensor_base_shape = dep.shape
                                     target_dtype = dep.dtype
                                     found_dep = True
                                     break
                           if not found_dep:
                                # Try to resolve dimensions from registry
                                try:
                                    extra_dims = []
                                    # tensor_shape[0] is the grid dimension, skip it
                                    # tensor_shape[1:] are the extra dimensions (e.g. levels)
                                    for dim_name in tensor_shape[1:]:
                                         if dim_name in self._tensor_registry:
                                              d_val = self._tensor_registry[dim_name]
                                              if d_val.numel() == 1:
                                                   extra_dims.append(int(d_val.item()))
                                              else:
                                                   raise ValueError
                                         else:
                                              raise ValueError
                                    
                                    # Construct a fake base shape that satisfies the slicing logic below
                                    # The logic below uses [2:] (if trials) or [1:] (if no trials)
                                    # to get the EXTRA dims.
                                    prefix_len = 2 if self.num_trials > 1 else 1
                                    tensor_base_shape = (1,) * prefix_len + tuple(extra_dims)
                                    target_dtype = torch.float32

                                except ValueError:
                                     raise ValueError(f"Virtual variable '{var_name}' has multi-dimensional shape {tensor_shape}. Dependencies not found in registry, and dimensions could not be resolved directly.")

                if self.num_trials > 1:
                    actual_shape = (self.num_trials, len(ref_save_idx)) + tensor_base_shape[2:] if tensor_ndim > 1 else (self.num_trials, len(ref_save_idx))
                else:
                    actual_shape = (len(ref_save_idx),) + tensor_base_shape[1:]
            else:
                raise ValueError(f"Save index '{save_idx}' not registered in tensor registry")
            
            actual_ndim = tensor_ndim
            max_ndim = 3 if self.num_trials > 1 else 2
            if actual_ndim > max_ndim:
                raise ValueError(f"Variable '{var_name}' has {actual_ndim} actual dimensions. Only up to {max_ndim}D variables are supported.")

            is_2d = (self.num_trials > 1 and actual_ndim == 3) or (self.num_trials == 1 and actual_ndim == 2)
            if is_2d and any(op.split('_')[0] in ['max', 'min'] or re.match(r'(max|min)\d+$', op.split('_')[0]) for op in ops):
                raise ValueError(f"max/min operations are not supported for 2D variable '{var_name}' (with levels).")

            # Track
            self._variables.add(var_name)

            for op in ops:
                out_name = f"{var_name}_{op}"
                
                # Parse op parts
                op_parts = op.split('_')
                outer_op = op_parts[0]
                
                # Check for K in max/min ops (e.g., max3, min3)
                k_val = 1
                match_k = re.match(r'(max|min)(\d+)$', outer_op)
                if match_k:
                    outer_base = match_k.group(1)
                    k_val = int(match_k.group(2))
                    outer_op = outer_base # normalize for allocation logic below (mostly)
                
                # Check for explicit argmax/argmin operators (e.g., argmax, argmax3)
                arg_match = re.match(r'arg(max|min)(\d*)$', outer_op)
                arg_k_val = 1  # Default for arg ops
                if arg_match:
                    arg_k_str = arg_match.group(2)
                    arg_k_val = int(arg_k_str) if arg_k_str else 1
                
                # Allocate storage by op
                if k_val > 1:
                    alloc_shape = actual_shape + (k_val,)
                else:
                    alloc_shape = actual_shape

                if arg_match:
                    # Explicit argmax/argmin operator - store integer indices only
                    arg_type = arg_match.group(1)  # 'max' or 'min'
                    arg_k_str = arg_match.group(2)  # '' or '3' etc
                    # arg_k_val already computed above
                    
                    if arg_k_val > 1:
                        arg_alloc_shape = actual_shape + (arg_k_val,)
                    else:
                        arg_alloc_shape = actual_shape
                    
                    # Store integer indices (macro step index within the window)
                    init_tensor = torch.zeros(arg_alloc_shape, dtype=torch.int32, device=self.device)
                    # Also need to track the corresponding extreme values for comparison
                    aux_name = f"{var_name}_{arg_type}{arg_k_str or ''}_aux"
                    if arg_type == 'max':
                        self._storage[aux_name] = torch.full(arg_alloc_shape, -torch.inf, dtype=target_dtype, device=self.device)
                    else:
                        self._storage[aux_name] = torch.full(arg_alloc_shape, torch.inf, dtype=target_dtype, device=self.device)
                    # aux is not an output, just internal state
                elif outer_op == 'max':
                    # max or maxK - NO automatic argmax
                    init_tensor = torch.full(alloc_shape, -torch.inf, dtype=target_dtype, device=self.device)
                elif outer_op == 'min':
                    # min or minK - NO automatic argmin
                    init_tensor = torch.full(alloc_shape, torch.inf, dtype=target_dtype, device=self.device)
                elif outer_op == 'first':
                    # Similar to 'last', we just need storage. Zero initialization is fine as it will be overwritten on is_first.
                    init_tensor = torch.zeros(alloc_shape, dtype=target_dtype, device=self.device)
                else:
                    init_tensor = torch.zeros(alloc_shape, dtype=target_dtype, device=self.device)
                self._storage[out_name] = init_tensor
                self._output_keys.append(out_name)

                # For compound ops, allocate inner state
                if len(op_parts) > 1:
                    inner_op = op_parts[1]
                    # 'last' inner op doesn't need cross-step state - it directly uses current value
                    needs_inner_state = inner_op != 'last'
                    inner_state_name = f"{var_name}_{inner_op}_inner_state"
                    
                    if needs_inner_state and inner_state_name not in self._storage:
                         # Initialize inner state
                         if inner_op == 'mean':
                             init_inner = torch.zeros(actual_shape, dtype=target_dtype, device=self.device)
                         elif inner_op == 'max':
                             init_inner = torch.full(actual_shape, -torch.inf, dtype=target_dtype, device=self.device)
                         elif inner_op == 'min':
                             init_inner = torch.full(actual_shape, torch.inf, dtype=target_dtype, device=self.device)
                         elif inner_op == 'sum':
                             init_inner = torch.zeros(actual_shape, dtype=target_dtype, device=self.device)
                         elif inner_op in ('first', 'mid'):
                             init_inner = torch.zeros(actual_shape, dtype=target_dtype, device=self.device)
                         else:
                             # Should be caught by validator, but safe fallback
                             init_inner = torch.zeros(actual_shape, dtype=target_dtype, device=self.device)
                         self._storage[inner_state_name] = init_inner
                         
                         # Allocate weight state only for inner ops that need it (mean)
                         if inner_op == 'mean':
                             weight_state_name = f"{var_name}_{inner_op}_weight_state"
                             if weight_state_name not in self._storage:
                                 self._storage[weight_state_name] = torch.zeros(actual_shape, dtype=target_dtype, device=self.device)

                if save_coord and save_coord not in self._coord_cache:
                    coord_tensor = self._tensor_registry[save_coord]
                    self._coord_cache[save_coord] = coord_tensor.detach().cpu().numpy()
                
                out_dtype = torch_to_numpy_dtype(target_dtype)
                
                # Check if this is an argmax/argmin op and determine the k value
                is_arg_op = arg_match is not None
                # For arg ops, use arg_k_val; otherwise use k_val
                effective_k = arg_k_val if is_arg_op else k_val

                meta = {
                    'original_variable': var_name,
                    'op': op,
                    'save_idx': save_idx,
                    'tensor_shape': tensor_shape,
                    'dtype': 'i4' if is_arg_op else out_dtype,  # int32 for arg ops
                    'actual_shape': actual_shape,
                    'actual_ndim': actual_ndim,
                    'save_coord': save_coord,
                    'description': f"{description} ({op})",
                    'stride_input': tensor.shape[1] if tensor is not None and self.num_trials > 1 else 0,
                    'k': effective_k,
                    'is_time_index': is_arg_op,  # argmax/argmin store integer indices
                }
                self._metadata[out_name] = meta
                
                # Classify as outer if it is a compound op (e.g. max_mean)
                self._output_is_outer[out_name] = len(op_parts) > 1


        # Generate kernels and prepare states for all requested variables/ops
        self._generate_aggregator_function()
        self._prepare_kernel_states()

    

    def update_statistics(self, weight: float, total_weight: float = 0.0, 
                          is_inner_first: bool = False, is_inner_last: bool = False, 
                          is_outer_first: bool = False, is_outer_last: bool = False,
                          BLOCK_SIZE: int = 128, custom_step_index: Optional[int] = None,
                          # Legacy kwargs support
                          is_first: bool = False, is_last: bool = False, 
                          is_middle: bool = False, is_macro_step_end: bool = False) -> None:
        if not self._aggregator_generated:
            raise RuntimeError("Statistics aggregation not initialized. Call initialize_streaming_aggregation() first.")
        
        # Handle legacy or new parameters
        _is_inner_first = is_inner_first or is_first
        _is_inner_last = is_inner_last or is_last
        
        if _is_inner_first:
            self._step_count = 0
        
        # Reset macro_step_index at the start of each outer statistics period
        # This ensures argmax/argmin indices are always relative to the start of the period
        if is_outer_first:
            self._macro_step_index = 0
            self._current_macro_step_count = 0.0
            
        if _is_inner_last:
             for out_name, is_outer in self._output_is_outer.items():
                 # We only trigger dirty for non-outer (Standard) ops when inner loop ends
                 if not is_outer:
                     self._dirty_outputs.add(out_name)
        
        if is_outer_last:
             for out_name, is_outer in self._output_is_outer.items():
                 # We trigger dirty for outer ops when outer loop ends
                 if is_outer:
                     self._dirty_outputs.add(out_name)

        if is_macro_step_end: # Legacy support or counter
            self._current_macro_step_count += 1.0

        macro_count_val = self._current_macro_step_count
            
        # Ensure kernel states is actually populated correctly for new keys
            
        self._aggregator_function(self._kernel_states, weight, total_weight, macro_count_val, 
                                  _is_inner_first, _is_inner_last, is_middle, 
                                  is_outer_first, is_outer_last,
                                  self._macro_step_index, self._step_count, BLOCK_SIZE)
        
        self._step_count += 1

    
