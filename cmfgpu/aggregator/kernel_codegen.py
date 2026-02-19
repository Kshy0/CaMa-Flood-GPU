# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import tempfile
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Set

if TYPE_CHECKING:
    from cmfgpu.aggregator.aggregator import StatisticsAggregator


class KernelCodegenMixin:
    """Mixin providing Triton kernel code generation and compilation."""

    def _generate_kernel_header(self: StatisticsAggregator) -> List[str]:
        """Generate the header for the kernel file with documentation."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        var_list = sorted(list(self._variables))
        
        header = [
            '"""',
            f'Auto-generated Triton kernels for CaMa-Flood-GPU statistics aggregation (mean/max/min/last)',
            f'Generated at: {timestamp}',
            f'Rank: {self.rank}',
            f'Variables: {", ".join(var_list)}',
            f'Device: {self.device}',
            '',
            'Kernel Logic:',
            '- Load save_idx values to get original grid indices',
            '- Use idx to access original data: data[idx]',
            '- Store outputs using sequential indexing: out[offs]',
            '- max/min ops automatically update corresponding argmax/argmin (step index)',
            '- argmax/argmin indices are converted to datetime on NC file write',
            '- For mid: stores val when is_middle is True',
            '',
            'Optimizations Applied:',
            '- tl.static_range for compile-time loop unrolling (num_trials, bubble sort)',
            '- Merged max3+argmax3 and min3+argmin3 when coexisting to share comparisons',
            '- Base offset precomputation (shared across max/min/argmax/argmin for same var+K)',
            '- Merged maxK+minK bubble insert in single loop with shared offset',
            '- Precise mask for tl.store: mask & swap_mask to reduce write pressure',
            '"""',
            "",
            "import triton",
            "import triton.language as tl",
            "from triton.language.extra import libdevice",
            "",
            '# ============================================================================',
            f"# Generated Triton kernels for statistics aggregation - Rank {self.rank}",
            "# ============================================================================",
            "",
        ]
        return header
    

    def _save_kernel_file(self: StatisticsAggregator, kernel_code: str) -> None:
        """
        Save the generated kernel code to a permanent file for inspection.
        
        Args:
            kernel_code: Generated kernel code as string
        """
        # Use unique name generation
        unique_name = self._generate_unique_name()
        filename = f"kern_{unique_name}.py"
        
        self._saved_kernel_file = self.kernels_dir / filename
        
        with open(self._saved_kernel_file, 'w', encoding='utf-8') as f:
            f.write(kernel_code)

    

    def _write_and_import_kernels(self: StatisticsAggregator, kernel_code: str) -> None:
        """
        Write kernel code to a temporary file and import the module.
        
        Args:
            kernel_code: Generated kernel code as string
        """
        # Create temporary file with unique name
        unique_name = self._generate_unique_name()
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{unique_name}.py', delete=False) as f:
            f.write(kernel_code)
            self._temp_kernel_file = f.name
        
        # Import the module from the temporary file
        module_name = f"aggr_kernels_r{self.rank}_{unique_name}"
        spec = importlib.util.spec_from_file_location(module_name, self._temp_kernel_file)
        module = importlib.util.module_from_spec(spec)
        # Add to sys.modules to ensure proper import
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # Bind to instance
        self._kernel_module = module
        self._aggregator_function = getattr(module, 'internal_update_statistics')
        self._aggregator_generated = True


    def _transform_pow_expr(self: StatisticsAggregator, expr: str) -> str:
        """
        Transform power operations in an expression string to Triton-compatible tl.exp(log()).
        Power operator ** or ^ is transformed.
        """
        if '**' not in expr and '^' not in expr:
            return expr
            
        safe_expr = expr.replace('^', '**')
        
        def _visit_and_transform_pow(node):
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    new_list = []
                    for item in value:
                        if isinstance(item, ast.AST):
                            new_list.append(_visit_and_transform_pow(item))
                        else:
                            new_list.append(item)
                    setattr(node, field, new_list)
                elif isinstance(value, ast.AST):
                    setattr(node, field, _visit_and_transform_pow(value))
            
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                return ast.Call(
                    func=ast.Attribute(value=ast.Name(id='libdevice', ctx=ast.Load()), attr='pow', ctx=ast.Load()),
                    args=[node.left, node.right],
                    keywords=[]
                )
            return node

        try:
            expr_tree = ast.parse(safe_expr, mode='eval')
            expr_tree = _visit_and_transform_pow(expr_tree)
            return ast.unparse(expr_tree)
        except Exception as e:
            print(f"Warning: Failed to transform power expression '{safe_expr}': {e}")
            return safe_expr


    def _emit_variable_load(self: StatisticsAggregator, var_name: str, code_lines: List[str], emitted: Set[str], is_2d: bool = False):
        """Helper to emit load instructions or expression evaluation recursively."""
        if var_name in emitted:
            return
        
        # Get safe name for this variable
        safe_var_name = self._get_safe_name(var_name)
        
        info = self._field_registry.get(var_name)
        json_extra = getattr(info, 'json_schema_extra', {})
        cat = json_extra.get('category', 'param')
        
        if cat == 'virtual':
             expr = json_extra.get('expr')
             # Find dependencies
             tokens = set(re.findall(r'\b[a-zA-Z_]\w*\b', expr))
             safe_expr = expr
             for token in tokens:
                  # Only recurse if it's a known variable (field or registered tensor)
                  if token in self._field_registry or token in self._tensor_registry:
                       self._emit_variable_load(token, code_lines, emitted, is_2d)
                       # Replace token in expression with safe token
                       safe_token = self._get_safe_name(token)
                       safe_expr = re.sub(r'\b' + token + r'\b', safe_token, safe_expr)
             
             safe_expr = self._transform_pow_expr(safe_expr)
             
             indent = "        " if is_2d else "    "
             code_lines.append(f"{indent}{safe_var_name} = {safe_expr}")
        else:
             # Real tensor load
             indent = "        " if is_2d else "    "
             if is_2d:
                  code_lines.append(f"{indent}{safe_var_name} = tl.load({safe_var_name}_ptr + idx * n_levels + level, mask=mask, other=0.0)")
             else:
                  code_lines.append(f"{indent}{safe_var_name} = tl.load({safe_var_name}_ptr + idx, mask=mask, other=0.0)")
        
        emitted.add(var_name)
    

    def _generate_1d_vars_grouped(self: StatisticsAggregator, kernel_code_lines: List[str], dims_1d: List[str],
                                    indent: str, indent2: str, indent3: str, indent4: str, indent5: str) -> None:
        """
        Generate 1D variable processing code with conditions grouped for efficiency.
        All operations under the same condition are emitted in a single if block.
        Supports all ops including maxK/minK bubble insert.
        
        Optimization: arg operations (argmax, argmin, argmax3, etc.) can only be outer ops.
        When max+argmax or max3+argmax3 coexist, they are merged to share comparisons.
        """
        from collections import defaultdict
        
        if not dims_1d:
            return
            
        kernel_code_lines.append(f"{indent}# 1D variables")
        
        # Phase 0: Analyze max/argmax pairs for merging optimization
        # For each variable, detect which ops coexist to enable merging
        var_op_analysis = {}  # var -> {'max': bool, 'argmax': bool, 'max3': int, 'argmax3': int, ...}
        for var in dims_1d:
            ops = self._variable_ops[var]
            analysis = {
                'max': 'max' in ops,
                'argmax': 'argmax' in ops,
                'min': 'min' in ops,
                'argmin': 'argmin' in ops,
                'maxK': {},   # k -> True
                'argmaxK': {},  # k -> True
                'minK': {},
                'argminK': {},
            }
            for op in ops:
                match_maxk = re.match(r'^max(\d+)$', op)
                match_argmaxk = re.match(r'^argmax(\d+)$', op)
                match_mink = re.match(r'^min(\d+)$', op)
                match_argmink = re.match(r'^argmin(\d+)$', op)
                if match_maxk:
                    analysis['maxK'][int(match_maxk.group(1))] = True
                if match_argmaxk:
                    analysis['argmaxK'][int(match_argmaxk.group(1))] = True
                if match_mink:
                    analysis['minK'][int(match_mink.group(1))] = True
                if match_argmink:
                    analysis['argminK'][int(match_argmink.group(1))] = True
            var_op_analysis[var] = analysis
        
        # Phase 1: Classify variables by when their value is needed
        vars_need_val = set()  # vars that need val loaded unconditionally (every sub-step)
        vars_conditional_only = set()  # vars only used conditionally (first/last/mid or compound with last/first/mid inner)
        
        # Simple ops that are conditional (only need val at specific points)
        simple_conditional_ops = {'first', 'last', 'mid'}
        # Inner op types that only need the value conditionally (at is_inner_last)
        conditional_inner_ops = {'last', 'first', 'mid'}
        
        for var in dims_1d:
            ops = self._variable_ops[var]
            needs_unconditional = False
            for op in ops:
                if op in simple_conditional_ops:
                    continue  # Conditional simple op
                op_parts = op.split('_')
                if len(op_parts) > 1:
                    inner = op_parts[1]
                    if inner in conditional_inner_ops:
                        continue  # Compound op with conditional inner type
                # Any other op needs unconditional val (mean, sum, max, min, etc.)
                needs_unconditional = True
                break
            if needs_unconditional:
                vars_need_val.add(var)
            else:
                vars_conditional_only.add(var)
        
        # Helper to emit variable value load
        emitted_vars = set()
        def emit_val(v_name, to_lines):
            safe_v_name = self._get_safe_name(v_name)
            if safe_v_name in emitted_vars:
                return f"{safe_v_name}_val"
            
            info = self._field_registry.get(v_name)
            cat = getattr(info, 'json_schema_extra', {}).get('category', 'param')
            
            if cat == 'virtual':
                expr = getattr(info, 'json_schema_extra', {}).get('expr', '')
                safe_expr = expr
                toks = set(re.findall(r'\b[a-zA-Z_]\w*\b', expr))
                for t in toks:
                    if t in self._field_registry or t in self._tensor_registry:
                        emit_val(t, to_lines)
                        safe_t = self._get_safe_name(t)
                        safe_expr = re.sub(r'\b' + t + r'\b', f"{safe_t}_val", safe_expr)
                safe_expr = self._transform_pow_expr(safe_expr)
                to_lines.append(f"{indent}{safe_v_name}_val = {safe_expr}")
            else:
                in_ptr_loc = f"{safe_v_name}_ptr + t * stride_input + idx"
                to_lines.append(f"{indent}{safe_v_name}_val = tl.load({in_ptr_loc}, mask=mask, other=0.0)")
            
            emitted_vars.add(safe_v_name)
            return f"{safe_v_name}_val"
        
        # Phase 2: Collect all operations grouped by condition
        ops_unconditional = []
        ops_is_inner_first = []
        ops_not_is_inner_first = []
        ops_is_inner_last = []
        ops_is_inner_last_is_outer_first = []
        
        # Special storage for maxK/minK operations (need for loop)
        self._maxk_ops = []
        self._argmaxk_ops = []
        ops_is_inner_last_not_is_outer_first = []
        ops_is_inner_last_is_outer_last = []
        ops_is_inner_last_not_is_outer_last = []
        ops_is_middle = []
        
        # Track which inner aggregations are needed
        inner_aggregations_needed = defaultdict(set)  # inner_type -> set of vars
        
        # Track which merged operations we've already processed
        processed_merged_ops = set()  # (var, op_type, k) tuples
        
        for var in dims_1d:
            safe_var = self._get_safe_name(var)
            ops = self._variable_ops[var]
            out_offset = "t * n_saved_points + offs"
            analysis = var_op_analysis[var]
            
            # Check for compound ops that need inner aggregation
            for op in ops:
                if '_' in op:
                    inner = op.split('_')[1]
                    inner_aggregations_needed[inner].add(var)
                
            # Process each operation
            for op in ops:
                out_ptr = f"{safe_var}_{op}_ptr + {out_offset}"
                op_parts = op.split('_')
                
                # ===== Compound operations (e.g., max_mean, min_mean) =====
                if len(op_parts) > 1:
                    outer = op_parts[0]
                    inner = op_parts[1]
                    
                    # Parse maxK/minK/argmaxK/argminK pattern
                    k_val = 1
                    match_k = re.match(r'(arg)?(max|min)(\d+)$', outer)
                    is_arg_compound = outer.startswith('arg')
                    if match_k:
                        is_arg_compound = match_k.group(1) is not None
                        outer_base = match_k.group(2)  # 'max' or 'min'
                        k_val = int(match_k.group(3))
                    else:
                        outer_base = outer.lstrip('arg')  # Remove 'arg' prefix if present
                    
                    # Use variable-specific inner aggregation result
                    # For 'last' inner type, directly use the variable value (no intermediate state)
                    if inner == 'last':
                        val_var = f"{safe_var}_val"
                    else:
                        val_var = f"val_for_{safe_var}_{inner}"
                    
                    if is_arg_compound:
                        # Compound argmax/argmin (e.g., argmax_mean, argmax3_mean)
                        arg_type = outer_base  # 'max' or 'min'
                        aux_ptr_base = f"{safe_var}_{arg_type}{k_val if k_val > 1 else ''}_aux_ptr"
                        
                        if k_val == 1:
                            aux_ptr = f"{aux_ptr_base} + {out_offset}"
                            if arg_type == 'max':
                                ops_is_inner_last_is_outer_first.extend([
                                    f"tl.store({out_ptr}, macro_step_index, mask=mask)",
                                    f"tl.store({aux_ptr}, {val_var}, mask=mask)",
                                ])
                                ops_is_inner_last_not_is_outer_first.extend([
                                    f"{safe_var}_{op}_aux_old = tl.load({aux_ptr}, mask=mask, other={val_var})",
                                    f"{safe_var}_{op}_cond = {val_var} > {safe_var}_{op}_aux_old",
                                    f"tl.store({aux_ptr}, tl.where({safe_var}_{op}_cond, {val_var}, {safe_var}_{op}_aux_old), mask=mask)",
                                    f"tl.store({out_ptr}, macro_step_index, mask=mask & {safe_var}_{op}_cond)",
                                ])
                            else:  # min
                                ops_is_inner_last_is_outer_first.extend([
                                    f"tl.store({out_ptr}, macro_step_index, mask=mask)",
                                    f"tl.store({aux_ptr}, {val_var}, mask=mask)",
                                ])
                                ops_is_inner_last_not_is_outer_first.extend([
                                    f"{safe_var}_{op}_aux_old = tl.load({aux_ptr}, mask=mask, other={val_var})",
                                    f"{safe_var}_{op}_cond = {val_var} < {safe_var}_{op}_aux_old",
                                    f"tl.store({aux_ptr}, tl.where({safe_var}_{op}_cond, {val_var}, {safe_var}_{op}_aux_old), mask=mask)",
                                    f"tl.store({out_ptr}, macro_step_index, mask=mask & {safe_var}_{op}_cond)",
                                ])
                        else:
                            # ArgmaxK/ArgminK compound bubble insert
                            self._argmaxk_ops.append({
                                'var': safe_var, 'op': op, 'k': k_val, 'val_var': val_var,
                                'out_offset': out_offset, 'type': f'arg{arg_type}',
                                'has_val_output': False  # compound arg doesn't need val output
                            })
                    elif outer_base == 'max':
                        # Compound max without automatic arg (e.g., max_mean, max3_mean)
                        if k_val == 1:
                            ops_is_inner_last_is_outer_first.append(
                                f"tl.store({out_ptr}, {val_var}, mask=mask)")
                            ops_is_inner_last_not_is_outer_first.extend([
                                f"{safe_var}_{op}_old = tl.load({out_ptr}, mask=mask, other={val_var})",
                                f"tl.store({out_ptr}, tl.maximum({safe_var}_{op}_old, {val_var}), mask=mask)",
                            ])
                        else:
                            # maxK bubble insert without arg tracking
                            self._maxk_ops.append({
                                'var': safe_var, 'op': op, 'k': k_val, 'val_var': val_var,
                                'out_offset': out_offset, 'type': 'max'
                            })
                            
                    elif outer_base == 'min':
                        # Compound min without automatic arg (e.g., min_mean, min3_mean)
                        if k_val == 1:
                            ops_is_inner_last_is_outer_first.append(
                                f"tl.store({out_ptr}, {val_var}, mask=mask)")
                            ops_is_inner_last_not_is_outer_first.extend([
                                f"{safe_var}_{op}_old = tl.load({out_ptr}, mask=mask, other={val_var})",
                                f"tl.store({out_ptr}, tl.minimum({safe_var}_{op}_old, {val_var}), mask=mask)",
                            ])
                        else:
                            # minK bubble insert without arg tracking
                            self._maxk_ops.append({
                                'var': safe_var, 'op': op, 'k': k_val, 'val_var': val_var,
                                'out_offset': out_offset, 'type': 'min'
                            })
                            
                    elif outer == 'mean':
                        ops_is_inner_last_is_outer_first.append(
                            f"{safe_var}_{op}_accum = {val_var}")
                        ops_is_inner_last_not_is_outer_first.append(
                            f"{safe_var}_{op}_accum = tl.load({out_ptr}, mask=mask, other=0.0) + {val_var}")
                        ops_is_inner_last_is_outer_last.append(
                            f"tl.store({out_ptr}, {safe_var}_{op}_accum / num_macro_steps, mask=mask)")
                        ops_is_inner_last_not_is_outer_last.append(
                            f"tl.store({out_ptr}, {safe_var}_{op}_accum, mask=mask)")
                            
                    elif outer == 'sum':
                        ops_is_inner_last_is_outer_first.append(
                            f"tl.store({out_ptr}, {val_var}, mask=mask)")
                        ops_is_inner_last_not_is_outer_first.extend([
                            f"{safe_var}_{op}_old = tl.load({out_ptr}, mask=mask, other=0.0)",
                            f"tl.store({out_ptr}, {safe_var}_{op}_old + {val_var}, mask=mask)",
                        ])
                    elif outer == 'last':
                        # Compound last (e.g., last_mean) — store the last inner value
                        # Simply overwrite on every is_inner_last step
                        ops_is_inner_last.append(f"tl.store({out_ptr}, {val_var}, mask=mask)")
                    elif outer == 'first':
                        # Compound first (e.g., first_mean) — store only at is_outer_first
                        ops_is_inner_last_is_outer_first.append(
                            f"tl.store({out_ptr}, {val_var}, mask=mask)")
                    continue
                
                # ===== Simple operations (non-compound) =====
                if op == 'mean':
                    inner_ops = set(o.split('_')[1] for o in ops if '_' in o)
                    if 'mean' in inner_ops:
                        # Reuse val_for_{safe_var}_mean from inner aggregation
                        ops_is_inner_last.append(f"tl.store({out_ptr}, val_for_{safe_var}_mean, mask=mask)")
                    else:
                        # Standalone mean - needs state (use variable-specific val)
                        ops_unconditional.extend([
                            f"# Standalone mean for {safe_var}",
                            f"{safe_var}_mean_old = tl.where(is_inner_first, tl.zeros_like({safe_var}_val), tl.load({out_ptr}, mask=mask, other=0.0))",
                            f"{safe_var}_mean_new = {safe_var}_mean_old + {safe_var}_val * weight",
                            f"{safe_var}_mean_out = tl.where(is_inner_last, {safe_var}_mean_new / total_weight, {safe_var}_mean_new)",
                        ])
                        ops_unconditional.append(f"tl.store({out_ptr}, {safe_var}_mean_out, mask=mask)")
                        
                elif op == 'sum':
                    ops_unconditional.extend([
                        f"{safe_var}_sum_old = tl.where(is_inner_first, tl.zeros_like({safe_var}_val), tl.load({out_ptr}, mask=mask, other=0.0))",
                        f"tl.store({out_ptr}, {safe_var}_sum_old + {safe_var}_val * weight, mask=mask)",
                    ])
                
                # ===== max/argmax MERGED handling =====
                elif op == 'max':
                    has_argmax = analysis['argmax']
                    merge_key = (var, 'max', 1)
                    if merge_key in processed_merged_ops:
                        continue  # Already processed as merged
                    processed_merged_ops.add(merge_key)
                    
                    if has_argmax:
                        # MERGED: max + argmax share the same comparison
                        aux_ptr = f"{safe_var}_max_aux_ptr + {out_offset}"
                        argmax_ptr = f"{safe_var}_argmax_ptr + {out_offset}"
                        ops_is_inner_first.extend([
                            f"# Merged max + argmax for {safe_var}",
                            f"tl.store({aux_ptr}, {safe_var}_val, mask=mask)",  # aux stores the max value
                            f"tl.store({out_ptr}, {safe_var}_val, mask=mask)",  # max output
                            f"tl.store({argmax_ptr}, macro_step_index, mask=mask)",  # argmax output
                        ])
                        ops_not_is_inner_first.extend([
                            f"{safe_var}_max_old = tl.load({aux_ptr}, mask=mask, other={safe_var}_val)",
                            f"{safe_var}_max_cond = {safe_var}_val > {safe_var}_max_old",
                            f"{safe_var}_max_new = tl.where({safe_var}_max_cond, {safe_var}_val, {safe_var}_max_old)",
                            f"tl.store({aux_ptr}, {safe_var}_max_new, mask=mask)",
                            f"tl.store({out_ptr}, {safe_var}_max_new, mask=mask)",
                            f"tl.store({argmax_ptr}, macro_step_index, mask=mask & {safe_var}_max_cond)",
                        ])
                    else:
                        # Simple max only (no argmax)
                        ops_is_inner_first.extend([
                            f"tl.store({out_ptr}, {safe_var}_val, mask=mask)",
                        ])
                        ops_not_is_inner_first.extend([
                            f"{safe_var}_max_old = tl.load({out_ptr}, mask=mask, other={safe_var}_val)",
                            f"tl.store({out_ptr}, tl.where({safe_var}_val > {safe_var}_max_old, {safe_var}_val, {safe_var}_max_old), mask=mask)",
                        ])
                
                elif op == 'argmax':
                    has_max = analysis['max']
                    merge_key = (var, 'max', 1)
                    if has_max:
                        # Will be handled by 'max' branch
                        continue
                    
                    # argmax only (no max)
                    aux_ptr = f"{safe_var}_max_aux_ptr + {out_offset}"
                    ops_is_inner_first.extend([
                        f"tl.store({out_ptr}, macro_step_index, mask=mask)",
                        f"tl.store({aux_ptr}, {safe_var}_val, mask=mask)",
                    ])
                    ops_not_is_inner_first.extend([
                        f"{safe_var}_argmax_aux_old = tl.load({aux_ptr}, mask=mask, other={safe_var}_val)",
                        f"{safe_var}_argmax_cond = {safe_var}_val > {safe_var}_argmax_aux_old",
                        f"tl.store({aux_ptr}, tl.where({safe_var}_argmax_cond, {safe_var}_val, {safe_var}_argmax_aux_old), mask=mask)",
                        f"tl.store({out_ptr}, macro_step_index, mask=mask & {safe_var}_argmax_cond)",
                    ])
                    
                # ===== min/argmin MERGED handling =====
                elif op == 'min':
                    has_argmin = analysis['argmin']
                    merge_key = (var, 'min', 1)
                    if merge_key in processed_merged_ops:
                        continue
                    processed_merged_ops.add(merge_key)
                    
                    if has_argmin:
                        # MERGED: min + argmin share the same comparison
                        aux_ptr = f"{safe_var}_min_aux_ptr + {out_offset}"
                        argmin_ptr = f"{safe_var}_argmin_ptr + {out_offset}"
                        ops_is_inner_first.extend([
                            f"# Merged min + argmin for {safe_var}",
                            f"tl.store({aux_ptr}, {safe_var}_val, mask=mask)",
                            f"tl.store({out_ptr}, {safe_var}_val, mask=mask)",
                            f"tl.store({argmin_ptr}, macro_step_index, mask=mask)",
                        ])
                        ops_not_is_inner_first.extend([
                            f"{safe_var}_min_old = tl.load({aux_ptr}, mask=mask, other={safe_var}_val)",
                            f"{safe_var}_min_cond = {safe_var}_val < {safe_var}_min_old",
                            f"{safe_var}_min_new = tl.where({safe_var}_min_cond, {safe_var}_val, {safe_var}_min_old)",
                            f"tl.store({aux_ptr}, {safe_var}_min_new, mask=mask)",
                            f"tl.store({out_ptr}, {safe_var}_min_new, mask=mask)",
                            f"tl.store({argmin_ptr}, macro_step_index, mask=mask & {safe_var}_min_cond)",
                        ])
                    else:
                        # Simple min only (no argmin)
                        ops_is_inner_first.extend([
                            f"tl.store({out_ptr}, {safe_var}_val, mask=mask)",
                        ])
                        ops_not_is_inner_first.extend([
                            f"{safe_var}_min_old = tl.load({out_ptr}, mask=mask, other={safe_var}_val)",
                            f"tl.store({out_ptr}, tl.where({safe_var}_val < {safe_var}_min_old, {safe_var}_val, {safe_var}_min_old), mask=mask)",
                        ])
                
                elif op == 'argmin':
                    has_min = analysis['min']
                    merge_key = (var, 'min', 1)
                    if has_min:
                        # Will be handled by 'min' branch
                        continue
                    
                    # argmin only (no min)
                    aux_ptr = f"{safe_var}_min_aux_ptr + {out_offset}"
                    ops_is_inner_first.extend([
                        f"tl.store({out_ptr}, macro_step_index, mask=mask)",
                        f"tl.store({aux_ptr}, {safe_var}_val, mask=mask)",
                    ])
                    ops_not_is_inner_first.extend([
                        f"{safe_var}_argmin_aux_old = tl.load({aux_ptr}, mask=mask, other={safe_var}_val)",
                        f"{safe_var}_argmin_cond = {safe_var}_val < {safe_var}_argmin_aux_old",
                        f"tl.store({aux_ptr}, tl.where({safe_var}_argmin_cond, {safe_var}_val, {safe_var}_argmin_aux_old), mask=mask)",
                        f"tl.store({out_ptr}, macro_step_index, mask=mask & {safe_var}_argmin_cond)",
                    ])
                
                # ===== maxK / argmaxK handling =====
                elif op.startswith('max') and re.match(r'^max(\d+)$', op):
                    match = re.match(r'^max(\d+)$', op)
                    k_val = int(match.group(1))
                    has_argmaxk = k_val in analysis['argmaxK']
                    merge_key = (var, 'max', k_val)
                    if merge_key in processed_merged_ops:
                        continue
                    processed_merged_ops.add(merge_key)
                    
                    if has_argmaxk:
                        # MERGED: maxK + argmaxK - store both val and idx in bubble insert
                        self._argmaxk_ops.append({
                            'var': safe_var, 'op': f'argmax{k_val}', 'k': k_val, 'val_var': f'{safe_var}_val',
                            'out_offset': out_offset, 'type': 'argmax',
                            'has_val_output': True,  # Also output max values
                            'val_output_ptr': f"{safe_var}_max{k_val}_ptr"
                        })
                    else:
                        # maxK only - simple bubble insert storing values
                        self._maxk_ops.append({
                            'var': safe_var, 'op': op, 'k': k_val, 'val_var': f'{safe_var}_val',
                            'out_offset': out_offset, 'type': 'max'
                        })
                
                elif op.startswith('argmax') and re.match(r'^argmax(\d+)$', op):
                    match = re.match(r'^argmax(\d+)$', op)
                    k_val = int(match.group(1))
                    has_maxk = k_val in analysis['maxK']
                    merge_key = (var, 'max', k_val)
                    if has_maxk:
                        # Will be handled by maxK branch
                        continue
                    
                    # argmaxK only - bubble insert with aux for values
                    self._argmaxk_ops.append({
                        'var': safe_var, 'op': op, 'k': k_val, 'val_var': f'{safe_var}_val',
                        'out_offset': out_offset, 'type': 'argmax',
                        'has_val_output': False
                    })
                
                # ===== minK / argminK handling =====
                elif op.startswith('min') and re.match(r'^min(\d+)$', op):
                    match = re.match(r'^min(\d+)$', op)
                    k_val = int(match.group(1))
                    has_argmink = k_val in analysis['argminK']
                    merge_key = (var, 'min', k_val)
                    if merge_key in processed_merged_ops:
                        continue
                    processed_merged_ops.add(merge_key)
                    
                    if has_argmink:
                        # MERGED: minK + argminK
                        self._argmaxk_ops.append({
                            'var': safe_var, 'op': f'argmin{k_val}', 'k': k_val, 'val_var': f'{safe_var}_val',
                            'out_offset': out_offset, 'type': 'argmin',
                            'has_val_output': True,
                            'val_output_ptr': f"{safe_var}_min{k_val}_ptr"
                        })
                    else:
                        # minK only
                        self._maxk_ops.append({
                            'var': safe_var, 'op': op, 'k': k_val, 'val_var': f'{safe_var}_val',
                            'out_offset': out_offset, 'type': 'min'
                        })
                
                elif op.startswith('argmin') and re.match(r'^argmin(\d+)$', op):
                    match = re.match(r'^argmin(\d+)$', op)
                    k_val = int(match.group(1))
                    has_mink = k_val in analysis['minK']
                    merge_key = (var, 'min', k_val)
                    if has_mink:
                        # Will be handled by minK branch
                        continue
                    
                    # argminK only
                    self._argmaxk_ops.append({
                        'var': safe_var, 'op': op, 'k': k_val, 'val_var': f'{safe_var}_val',
                        'out_offset': out_offset, 'type': 'argmin',
                        'has_val_output': False
                    })
                    
                elif op == 'last':
                    if var in vars_conditional_only:
                        # Check if this var also has compound ops with 'last' inner type
                        # If so, the deferred load inside is_inner_last block will already load the val
                        has_compound_last = any(
                            '_last' in other_op and other_op != 'last' 
                            for other_op in self._variable_ops[var]
                        )
                        if has_compound_last:
                            # Reuse the val loaded by deferred load (no duplicate load needed)
                            ops_is_inner_last.append(f"tl.store({out_ptr}, {safe_var}_val, mask=mask)")
                        else:
                            # Load inline
                            in_ptr_loc = f"{safe_var}_ptr + t * stride_input + idx"
                            ops_is_inner_last.extend([
                                f"{safe_var}_val = tl.load({in_ptr_loc}, mask=mask, other=0.0)",
                                f"tl.store({out_ptr}, {safe_var}_val, mask=mask)",
                            ])
                    else:
                        ops_is_inner_last.append(f"tl.store({out_ptr}, {safe_var}_val, mask=mask)")
                        
                elif op == 'first':
                    if var in vars_conditional_only:
                        has_compound_first = any(
                            '_first' in other_op and other_op != 'first'
                            for other_op in self._variable_ops[var]
                        )
                        if has_compound_first:
                            # Val will be loaded elsewhere (from unconditional or conditional load)
                            ops_is_inner_first.append(f"tl.store({out_ptr}, {safe_var}_val, mask=mask)")
                        else:
                            in_ptr_loc = f"{safe_var}_ptr + t * stride_input + idx"
                            ops_is_inner_first.extend([
                                f"{safe_var}_val = tl.load({in_ptr_loc}, mask=mask, other=0.0)",
                                f"tl.store({out_ptr}, {safe_var}_val, mask=mask)",
                            ])
                    else:
                        ops_is_inner_first.append(f"tl.store({out_ptr}, {safe_var}_val, mask=mask)")
                        
                elif op == 'mid':
                    if var in vars_conditional_only:
                        in_ptr_loc = f"{safe_var}_ptr + t * stride_input + idx"
                        ops_is_middle.extend([
                            f"{safe_var}_val = tl.load({in_ptr_loc}, mask=mask, other=0.0)",
                            f"tl.store({out_ptr}, {safe_var}_val, mask=mask)",
                        ])
                    else:
                        ops_is_middle.append(f"tl.store({out_ptr}, {safe_var}_val, mask=mask)")
                
        # Phase 3: Emit loads for vars that need unconditional val
        for var in vars_need_val:
            emit_val(var, kernel_code_lines)
        
        # For conditional-only vars used in compound ops with 'last' inner type,
        # ensure the variable val is emitted (will be loaded inside is_inner_last block later)
        # We need to track them but NOT emit unconditional loads here.
        # The load will be emitted inside the is_inner_last block in Phase 6.
        
        # Phase 4: Emit inner aggregation state updates (per-variable)
        # Each variable gets its own inner aggregation state (val_for_{safe_var}_{inner_type})
        # For 'last' inner type, no state is needed - the value is simply the current variable value
        # used directly inside the `if is_inner_last:` block.
        for inner_type, inner_vars in inner_aggregations_needed.items():
            for var in inner_vars:
                safe_var = self._get_safe_name(var)
                out_offset = "t * n_saved_points + offs"
                val_for_var_inner = f"val_for_{safe_var}_{inner_type}"
                var_val = f"{safe_var}_val"
                
                if inner_type == 'last':
                    # 'last' is the simplest: val_for_X_last == X_val at is_inner_last.
                    # No state storage, no load/store needed.
                    pass
                elif inner_type == 'mean':
                    inner_ptr = f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    weight_ptr = f"{safe_var}_{inner_type}_weight_state_ptr + {out_offset}"
                    kernel_code_lines.append(f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})")
                    kernel_code_lines.extend([
                        f"{indent}{safe_var}_inner_{inner_type}_old = tl.load({inner_ptr}, mask=mask, other=0.0)",
                        f"{indent}{safe_var}_weight_{inner_type}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                        f"{indent}{safe_var}_inner_{inner_type}_new = {safe_var}_inner_{inner_type}_old + {var_val} * weight",
                        f"{indent}{safe_var}_weight_{inner_type}_new = {safe_var}_weight_{inner_type}_old + weight",
                    ])
                    # Store based on condition - use tl.where for efficiency
                    kernel_code_lines.extend([
                        f"{indent}tl.store({inner_ptr}, tl.where(is_inner_last, 0.0, {safe_var}_inner_{inner_type}_new), mask=mask)",
                        f"{indent}tl.store({weight_ptr}, tl.where(is_inner_last, 0.0, {safe_var}_weight_{inner_type}_new), mask=mask)",
                        f"{indent}{val_for_var_inner} = tl.where(is_inner_last, {safe_var}_inner_{inner_type}_new / {safe_var}_weight_{inner_type}_new, {val_for_var_inner})",
                    ])
                elif inner_type == 'sum':
                    inner_ptr = f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    kernel_code_lines.append(f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})")
                    kernel_code_lines.extend([
                        f"{indent}{safe_var}_inner_{inner_type}_old = tl.load({inner_ptr}, mask=mask, other=0.0)",
                        f"{indent}{safe_var}_inner_{inner_type}_new = {safe_var}_inner_{inner_type}_old + {var_val} * weight",
                        f"{indent}tl.store({inner_ptr}, tl.where(is_inner_last, 0.0, {safe_var}_inner_{inner_type}_new), mask=mask)",
                        f"{indent}{val_for_var_inner} = tl.where(is_inner_last, {safe_var}_inner_{inner_type}_new, {val_for_var_inner})",
                    ])
                elif inner_type == 'max':
                    inner_ptr = f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    kernel_code_lines.append(f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})")
                    kernel_code_lines.extend([
                        f"{indent}{safe_var}_inner_{inner_type}_old = tl.load({inner_ptr}, mask=mask, other={var_val})",
                        f"{indent}{safe_var}_inner_{inner_type}_new = tl.where(is_inner_first & (macro_step_index==0), {var_val}, tl.maximum({safe_var}_inner_{inner_type}_old, {var_val}))",
                        f"{indent}tl.store({inner_ptr}, tl.where(is_inner_last, -float('inf'), {safe_var}_inner_{inner_type}_new), mask=mask)",
                        f"{indent}{val_for_var_inner} = tl.where(is_inner_last, {safe_var}_inner_{inner_type}_new, {val_for_var_inner})",
                    ])
                elif inner_type == 'min':
                    inner_ptr = f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    kernel_code_lines.append(f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})")
                    kernel_code_lines.extend([
                        f"{indent}{safe_var}_inner_{inner_type}_old = tl.load({inner_ptr}, mask=mask, other={var_val})",
                        f"{indent}{safe_var}_inner_{inner_type}_new = tl.where(is_inner_first & (macro_step_index==0), {var_val}, tl.minimum({safe_var}_inner_{inner_type}_old, {var_val}))",
                        f"{indent}tl.store({inner_ptr}, tl.where(is_inner_last, float('inf'), {safe_var}_inner_{inner_type}_new), mask=mask)",
                        f"{indent}{val_for_var_inner} = tl.where(is_inner_last, {safe_var}_inner_{inner_type}_new, {val_for_var_inner})",
                    ])
                elif inner_type == 'first':
                    # 'first' inner: store the value at is_inner_first, read it back at is_inner_last
                    inner_ptr = f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    kernel_code_lines.append(f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})")
                    kernel_code_lines.extend([
                        f"{indent}tl.store({inner_ptr}, {var_val}, mask=mask & is_inner_first)",
                        f"{indent}{val_for_var_inner} = tl.where(is_inner_last, tl.load({inner_ptr}, mask=mask, other=0.0), {val_for_var_inner})",
                    ])
                elif inner_type == 'mid':
                    inner_ptr = f"{safe_var}_{inner_type}_inner_state_ptr + {out_offset}"
                    kernel_code_lines.append(f"{indent}{val_for_var_inner} = tl.zeros_like({var_val})")
                    kernel_code_lines.extend([
                        f"{indent}tl.store({inner_ptr}, {var_val}, mask=mask & is_middle)",
                        f"{indent}{val_for_var_inner} = tl.where(is_inner_last, tl.load({inner_ptr}, mask=mask, other=0.0), {val_for_var_inner})",
                    ])
                # Note: removed 'break' - now generate for each variable in inner_vars
        
        # Phase 5: Emit unconditional ops
        for line in ops_unconditional:
            kernel_code_lines.append(f"{indent}{line}")
        
        # Phase 6: Emit grouped conditional blocks
        if ops_is_inner_first:
            kernel_code_lines.append(f"{indent}if is_inner_first:")
            for line in ops_is_inner_first:
                kernel_code_lines.append(f"{indent2}{line}")
                
        if ops_not_is_inner_first:
            if ops_is_inner_first:
                kernel_code_lines.append(f"{indent}else:")
            else:
                kernel_code_lines.append(f"{indent}if not is_inner_first:")
            for line in ops_not_is_inner_first:
                kernel_code_lines.append(f"{indent2}{line}")
        
        if ops_is_middle:
            kernel_code_lines.append(f"{indent}if is_middle:")
            for line in ops_is_middle:
                kernel_code_lines.append(f"{indent2}{line}")
        
        # Nested conditions for is_inner_last with outer conditions
        has_argmaxk_ops = hasattr(self, '_argmaxk_ops') and self._argmaxk_ops
        has_inner_last_ops = (ops_is_inner_last or ops_is_inner_last_is_outer_first or 
                             ops_is_inner_last_not_is_outer_first or ops_is_inner_last_is_outer_last or
                             ops_is_inner_last_not_is_outer_last or self._maxk_ops or has_argmaxk_ops)
        
        if has_inner_last_ops:
            kernel_code_lines.append(f"{indent}if is_inner_last:")
            
            # Emit deferred loads for conditional-only vars used in compound ops
            # These vars are only needed inside is_inner_last, so we load them here
            for var in dims_1d:
                if var in vars_conditional_only and var in inner_aggregations_needed.get('last', set()):
                    emit_val(var, kernel_code_lines)
                    # Patch the emitted line to use is_inner_last indentation
                    # The emit_val appends to kernel_code_lines with 'indent' (8 spaces)
                    # We need it at indent2 (12 spaces) since we're inside 'if is_inner_last:'
                    if kernel_code_lines and kernel_code_lines[-1].startswith(indent) and not kernel_code_lines[-1].startswith(indent2):
                        last_line = kernel_code_lines.pop()
                        kernel_code_lines.append(f"{indent2}{last_line.lstrip()}")
            
            # is_outer_first / not is_outer_first
            if ops_is_inner_last_is_outer_first or ops_is_inner_last_not_is_outer_first:
                kernel_code_lines.append(f"{indent2}if is_outer_first:")
                for line in ops_is_inner_last_is_outer_first:
                    kernel_code_lines.append(f"{indent3}{line}")
                if ops_is_inner_last_not_is_outer_first:
                    kernel_code_lines.append(f"{indent2}else:")
                    for line in ops_is_inner_last_not_is_outer_first:
                        kernel_code_lines.append(f"{indent3}{line}")
            
            # is_outer_last / not is_outer_last (for mean finalization)
            if ops_is_inner_last_is_outer_last or ops_is_inner_last_not_is_outer_last:
                kernel_code_lines.append(f"{indent2}if is_outer_last:")
                for line in ops_is_inner_last_is_outer_last:
                    kernel_code_lines.append(f"{indent3}{line}")
                if ops_is_inner_last_not_is_outer_last:
                    kernel_code_lines.append(f"{indent2}else:")
                    for line in ops_is_inner_last_not_is_outer_last:
                        kernel_code_lines.append(f"{indent3}{line}")
            
            # Simple is_inner_last ops
            for line in ops_is_inner_last:
                kernel_code_lines.append(f"{indent2}{line}")
            
            # ================================================================
            # Optimized MaxK/MinK + ArgmaxK/ArgminK bubble insert operations
            
            # Group by (var, k, out_offset) to share base offset across all operations
            from collections import defaultdict
            grouped_by_var_k = defaultdict(lambda: {'max': None, 'min': None, 'argmax': None, 'argmin': None})
            
            for maxk_op in self._maxk_ops:
                key = (maxk_op['var'], maxk_op['k'], maxk_op['out_offset'])
                grouped_by_var_k[key][maxk_op['type']] = maxk_op
            
            if hasattr(self, '_argmaxk_ops') and self._argmaxk_ops:
                for argk_op in self._argmaxk_ops:
                    key = (argk_op['var'], argk_op['k'], argk_op['out_offset'])
                    op_type = 'argmax' if 'max' in argk_op['type'] else 'argmin'
                    grouped_by_var_k[key][op_type] = argk_op
            
            # Process grouped operations with shared offset
            for (safe_var, k_val, out_offset), ops_dict in grouped_by_var_k.items():
                has_max = ops_dict['max'] is not None
                has_min = ops_dict['min'] is not None
                has_argmax = ops_dict['argmax'] is not None
                has_argmin = ops_dict['argmin'] is not None
                
                # Get val_var from each operation (may differ: val for max/min, val_for_mean for argmax/argmin)
                max_val_var = ops_dict['max']['val_var'] if has_max else None
                min_val_var = ops_dict['min']['val_var'] if has_min else None
                argmax_val_var = ops_dict['argmax']['val_var'] if has_argmax else None
                argmin_val_var = ops_dict['argmin']['val_var'] if has_argmin else None
                
                # Compute shared base offset once
                out_offset_k = f"({out_offset}) * {k_val}"
                
                # Generate header comment
                op_names = []
                if has_max: op_names.append(f"max{k_val}")
                if has_min: op_names.append(f"min{k_val}")
                if has_argmax: op_names.append(f"argmax{k_val}")
                if has_argmin: op_names.append(f"argmin{k_val}")
                kernel_code_lines.append(f"{indent2}# Merged Bubble Insert [{'+'.join(op_names)}] for {safe_var} (shared offset, precise mask)")
                
                # Shared base offset computation
                kernel_code_lines.append(f"{indent2}{safe_var}_k{k_val}_base_offs = {out_offset_k}")
                
                # Initialize new values for bubble insert (using correct val_var for each op type)
                if has_max:
                    kernel_code_lines.append(f"{indent2}new_val_max_{safe_var} = {max_val_var}")
                if has_min:
                    kernel_code_lines.append(f"{indent2}new_val_min_{safe_var} = {min_val_var}")
                if has_argmax:
                    kernel_code_lines.append(f"{indent2}new_val_argmax_{safe_var} = {argmax_val_var}")
                if has_argmin:
                    kernel_code_lines.append(f"{indent2}new_val_argmin_{safe_var} = {argmin_val_var}")
                if has_argmax or has_argmin:
                    kernel_code_lines.append(f"{indent2}new_idx_{safe_var} = tl.full([BLOCK_SIZE], macro_step_index, dtype=tl.int32)")
                
                # is_outer_first branch: initialize all arrays
                kernel_code_lines.append(f"{indent2}if is_outer_first:")
                
                # First position stores the initial value
                if has_max:
                    max_ptr = f"{safe_var}_{ops_dict['max']['op']}_ptr"
                    kernel_code_lines.append(f"{indent3}tl.store({max_ptr} + {safe_var}_k{k_val}_base_offs, new_val_max_{safe_var}, mask=mask)")
                if has_min:
                    min_ptr = f"{safe_var}_{ops_dict['min']['op']}_ptr"
                    kernel_code_lines.append(f"{indent3}tl.store({min_ptr} + {safe_var}_k{k_val}_base_offs, new_val_min_{safe_var}, mask=mask)")
                if has_argmax:
                    argmax_op = ops_dict['argmax']
                    argmax_aux_ptr = f"{safe_var}_max{k_val}_aux_ptr"
                    argmax_idx_ptr = f"{safe_var}_{argmax_op['op']}_ptr"
                    kernel_code_lines.append(f"{indent3}tl.store({argmax_idx_ptr} + {safe_var}_k{k_val}_base_offs, new_idx_{safe_var}, mask=mask)")
                    kernel_code_lines.append(f"{indent3}tl.store({argmax_aux_ptr} + {safe_var}_k{k_val}_base_offs, new_val_argmax_{safe_var}, mask=mask)")
                    if argmax_op.get('has_val_output') and argmax_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent3}tl.store({argmax_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs, new_val_argmax_{safe_var}, mask=mask)")
                if has_argmin:
                    argmin_op = ops_dict['argmin']
                    argmin_aux_ptr = f"{safe_var}_min{k_val}_aux_ptr"
                    argmin_idx_ptr = f"{safe_var}_{argmin_op['op']}_ptr"
                    kernel_code_lines.append(f"{indent3}tl.store({argmin_idx_ptr} + {safe_var}_k{k_val}_base_offs, new_idx_{safe_var}, mask=mask)")
                    kernel_code_lines.append(f"{indent3}tl.store({argmin_aux_ptr} + {safe_var}_k{k_val}_base_offs, new_val_argmin_{safe_var}, mask=mask)")
                    if argmin_op.get('has_val_output') and argmin_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent3}tl.store({argmin_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs, new_val_argmin_{safe_var}, mask=mask)")
                
                # Initialize remaining positions with inf/-inf
                kernel_code_lines.append(f"{indent3}for k in tl.static_range(1, {k_val}):")
                if has_max:
                    kernel_code_lines.append(f"{indent4}tl.store({max_ptr} + {safe_var}_k{k_val}_base_offs + k, -float('inf'), mask=mask)")
                if has_min:
                    kernel_code_lines.append(f"{indent4}tl.store({min_ptr} + {safe_var}_k{k_val}_base_offs + k, float('inf'), mask=mask)")
                if has_argmax:
                    kernel_code_lines.append(f"{indent4}tl.store({argmax_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, 0, mask=mask)")
                    kernel_code_lines.append(f"{indent4}tl.store({argmax_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, -float('inf'), mask=mask)")
                    if argmax_op.get('has_val_output') and argmax_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent4}tl.store({argmax_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs + k, -float('inf'), mask=mask)")
                if has_argmin:
                    kernel_code_lines.append(f"{indent4}tl.store({argmin_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, 0, mask=mask)")
                    kernel_code_lines.append(f"{indent4}tl.store({argmin_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, float('inf'), mask=mask)")
                    if argmin_op.get('has_val_output') and argmin_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent4}tl.store({argmin_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs + k, float('inf'), mask=mask)")
                
                # else branch: bubble insert
                kernel_code_lines.append(f"{indent2}else:")
                kernel_code_lines.append(f"{indent3}for k in tl.static_range({k_val}):")
                
                # Load old values and compute swap masks
                if has_max:
                    kernel_code_lines.extend([
                        f"{indent4}old_max_k = tl.load({max_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=-float('inf'))",
                        f"{indent4}swap_max = new_val_max_{safe_var} > old_max_k",
                        f"{indent4}max_to_store = tl.where(swap_max, new_val_max_{safe_var}, old_max_k)",
                        f"{indent4}new_val_max_{safe_var} = tl.where(swap_max, old_max_k, new_val_max_{safe_var})",
                        f"{indent4}tl.store({max_ptr} + {safe_var}_k{k_val}_base_offs + k, max_to_store, mask=mask & swap_max)",
                    ])
                if has_min:
                    kernel_code_lines.extend([
                        f"{indent4}old_min_k = tl.load({min_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=float('inf'))",
                        f"{indent4}swap_min = new_val_min_{safe_var} < old_min_k",
                        f"{indent4}min_to_store = tl.where(swap_min, new_val_min_{safe_var}, old_min_k)",
                        f"{indent4}new_val_min_{safe_var} = tl.where(swap_min, old_min_k, new_val_min_{safe_var})",
                        f"{indent4}tl.store({min_ptr} + {safe_var}_k{k_val}_base_offs + k, min_to_store, mask=mask & swap_min)",
                    ])
                if has_argmax:
                    kernel_code_lines.extend([
                        f"{indent4}old_argmax_aux_k = tl.load({argmax_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=-float('inf'))",
                        f"{indent4}old_argmax_idx_k = tl.load({argmax_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=0)",
                        f"{indent4}swap_argmax = new_val_argmax_{safe_var} > old_argmax_aux_k",
                        f"{indent4}argmax_aux_store = tl.where(swap_argmax, new_val_argmax_{safe_var}, old_argmax_aux_k)",
                        f"{indent4}argmax_idx_store = tl.where(swap_argmax, new_idx_{safe_var}, old_argmax_idx_k)",
                        f"{indent4}new_val_argmax_{safe_var} = tl.where(swap_argmax, old_argmax_aux_k, new_val_argmax_{safe_var})",
                        f"{indent4}new_idx_{safe_var} = tl.where(swap_argmax, old_argmax_idx_k, new_idx_{safe_var})",
                        f"{indent4}tl.store({argmax_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, argmax_aux_store, mask=mask & swap_argmax)",
                        f"{indent4}tl.store({argmax_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, argmax_idx_store, mask=mask & swap_argmax)",
                    ])
                    if argmax_op.get('has_val_output') and argmax_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent4}tl.store({argmax_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs + k, argmax_aux_store, mask=mask & swap_argmax)")
                if has_argmin:
                    kernel_code_lines.extend([
                        f"{indent4}old_argmin_aux_k = tl.load({argmin_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=float('inf'))",
                        f"{indent4}old_argmin_idx_k = tl.load({argmin_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, mask=mask, other=0)",
                        f"{indent4}swap_argmin = new_val_argmin_{safe_var} < old_argmin_aux_k",
                        f"{indent4}argmin_aux_store = tl.where(swap_argmin, new_val_argmin_{safe_var}, old_argmin_aux_k)",
                        f"{indent4}argmin_idx_store = tl.where(swap_argmin, new_idx_{safe_var}, old_argmin_idx_k)",
                        f"{indent4}new_val_argmin_{safe_var} = tl.where(swap_argmin, old_argmin_aux_k, new_val_argmin_{safe_var})",
                        f"{indent4}new_idx_{safe_var} = tl.where(swap_argmin, old_argmin_idx_k, new_idx_{safe_var})",
                        f"{indent4}tl.store({argmin_aux_ptr} + {safe_var}_k{k_val}_base_offs + k, argmin_aux_store, mask=mask & swap_argmin)",
                        f"{indent4}tl.store({argmin_idx_ptr} + {safe_var}_k{k_val}_base_offs + k, argmin_idx_store, mask=mask & swap_argmin)",
                    ])
                    if argmin_op.get('has_val_output') and argmin_op.get('val_output_ptr'):
                        kernel_code_lines.append(f"{indent4}tl.store({argmin_op['val_output_ptr']} + {safe_var}_k{k_val}_base_offs + k, argmin_aux_store, mask=mask & swap_argmin)")
            
            # Reset operation lists
            self._maxk_ops = []
            if hasattr(self, '_argmaxk_ops'):
                self._argmaxk_ops = []
        
        kernel_code_lines.append("")


    def _generate_kernel_for_group(self: StatisticsAggregator, kernel_code_lines: List[str], kernel_name: str,
                                   save_idx: str, var_list: List[str],
                                   tensor_info: Dict[str, Dict[str, Any]]) -> None:
        """Generate kernel code for a specific save_idx group supporting ops."""
        if self.num_trials > 1:
            dims_1d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 2]
            dims_2d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 3]
        else:
            dims_1d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 1]
            dims_2d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 2]

        # Header
        kernel_code_lines.extend([
            f"# Kernel for save_idx: {save_idx}",
            f"# Variables: {', '.join(var_list)}",
            f"# 1D: {', '.join(dims_1d) if dims_1d else 'None'}",
            f"# 2D: {', '.join(dims_2d) if dims_2d else 'None'}",
            "",
            "@triton.jit",
            f"def {kernel_name}(",
            f"    {save_idx}_ptr,",
        ])

        # Gather input pointers (resolving virtuals)
        input_ptrs = set()
        def _gather_inputs(name):
             info = self._field_registry.get(name)
             if getattr(info, 'json_schema_extra', {}).get('category') == 'virtual':
                  expr = getattr(info, 'json_schema_extra', {}).get('expr', '')
                  toks = re.findall(r'\b[a-zA-Z_]\w*\b', expr)
                  for t in toks:
                       if t in self._field_registry or t in self._tensor_registry:
                            _gather_inputs(t)
             else:
                  input_ptrs.add(name)
        
        for var in var_list:
             _gather_inputs(var)

        # Pointers
        # Inputs
        sorted_inputs = sorted(list(input_ptrs))
        for var in sorted_inputs:
            safe_var = self._get_safe_name(var)
            # Avoid duplicate argument if save_idx matches input var
            if safe_var == save_idx:
                continue
            kernel_code_lines.append(f"    {safe_var}_ptr,")

        for var in var_list:
            safe_var = self._get_safe_name(var)
            # Track which extra state pointers have been added to avoid duplicates
            added_aux_ptrs = set()  # Track aux pointers already added (for explicit argmax/argmin)
            
            for op in self._variable_ops[var]:
                kernel_code_lines.append(f"    {safe_var}_{op}_ptr,")
                
                # For EXPLICIT argmax/argmin operators, add aux pointer for tracking values
                # NO automatic argmax/argmin generation for max/min operations
                op_parts = op.split('_')
                outer_op = op_parts[0]
                
                # Check for explicit argmax/argmin (e.g., argmax, argmax3, argmin, argmin3)
                arg_match = re.match(r'arg(max|min)(\d*)$', outer_op)
                if arg_match:
                    arg_type = arg_match.group(1)  # 'max' or 'min'
                    arg_k_str = arg_match.group(2)  # '' or '3' etc
                    # aux pointer name: {safe_var}_{arg_type}{k}_aux_ptr (e.g., var_max_aux_ptr, var_max3_aux_ptr)
                    aux_name = f"{arg_type}{arg_k_str}_aux"  # e.g., 'max_aux', 'max3_aux'
                    if aux_name not in added_aux_ptrs:
                        kernel_code_lines.append(f"    {safe_var}_{aux_name}_ptr,")
                        added_aux_ptrs.add(aux_name)
            
            # Inner state pointers (only for ops that need cross-step state)
            added_inner = set()
            for op in self._variable_ops[var]:
                if '_' in op:
                    inner = op.split('_')[1]
                    if inner not in added_inner:
                        # 'last' inner op directly uses current value, no state needed
                        if inner != 'last':
                            kernel_code_lines.append(f"    {safe_var}_{inner}_inner_state_ptr,")
                        if inner == 'mean':
                            kernel_code_lines.append(f"    {safe_var}_{inner}_weight_state_ptr,")
                        added_inner.add(inner)

        kernel_code_lines.extend([
            "    weight,",
            "    total_weight,",
            "    num_macro_steps,",
            "    is_inner_first,",
            "    is_inner_last,",
            "    is_middle,",
            "    is_outer_first,",
            "    is_outer_last,",
            "    macro_step_index,",
            "    step_count_val,",
            "    n_saved_points: tl.constexpr,",
        ])
        if dims_2d:
            kernel_code_lines.append("    n_levels: tl.constexpr,")
        kernel_code_lines.extend([
            "    BLOCK_SIZE: tl.constexpr,",
            "    num_trials: tl.constexpr,",
            "    stride_input: tl.constexpr,",
            "):",
            "    pid = tl.program_id(0)",
            "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
            "    mask = offs < n_saved_points",
            "",
            f"    idx = tl.load({save_idx}_ptr + offs, mask=mask)",
            "",
        ])

        # Loop over trials - use tl.static_range for compile-time unrolling
        kernel_code_lines.append("    for t in tl.static_range(num_trials):")
        indent = "        "
        indent2 = indent + "    "
        indent3 = indent2 + "    "
        indent4 = indent3 + "    "
        indent5 = indent4 + "    "

        # 1D processing - use grouped generation for all vars
        if dims_1d:
            self._generate_1d_vars_grouped(kernel_code_lines, dims_1d, 
                                           indent, indent2, indent3, indent4, indent5)

        # 2D processing
        if dims_2d:
            non_last_only = [v for v in dims_2d if not (len(self._variable_ops[v]) == 1 and self._variable_ops[v][0] == 'last')]
            last_only_vars = [v for v in dims_2d if (len(self._variable_ops[v]) == 1 and self._variable_ops[v][0] == 'last')]

            if non_last_only:
                kernel_code_lines.extend([
                    f"{indent}# 2D variables (mean/min/max and mixed)",
                    f"{indent}for level in tl.static_range(n_levels):",
                ])
                emitted_vars_2d = set()
                def emit_val_2d(v_name):
                    safe_v_name = self._get_safe_name(v_name)
                    if safe_v_name in emitted_vars_2d: return f"{safe_v_name}_val"
                    
                    info = self._field_registry.get(v_name)
                    cat = getattr(info, 'json_schema_extra', {}).get('category', 'param') if info else 'param'
                    
                    if cat == 'virtual' and info:
                         expr = getattr(info, 'json_schema_extra', {}).get('expr')
                         import re
                         safe_expr = expr
                         toks = set(re.findall(r'\b[a-zA-Z_]\w*\b', expr))
                         for t in toks:
                              if t in self._field_registry or t in self._tensor_registry:
                                   emit_val_2d(t)
                                   safe_t = self._get_safe_name(t)
                                   safe_expr = re.sub(r'\b' + t + r'\b', f"{safe_t}_val", safe_expr)
                         safe_expr = self._transform_pow_expr(safe_expr)
                         kernel_code_lines.append(f"{indent2}{safe_v_name}_val = {safe_expr}")
                    else:
                         in_ptr_loc = f"{safe_v_name}_ptr + (t * stride_input + idx) * n_levels + level"
                         kernel_code_lines.append(f"{indent2}{safe_v_name}_val = tl.load({in_ptr_loc}, mask=mask, other=0.0)")
                    
                    emitted_vars_2d.add(safe_v_name)
                    return f"{safe_v_name}_val"

                for var in non_last_only:
                    safe_var = self._get_safe_name(var)
                    out_offset = f"(t * n_saved_points + offs) * n_levels + level"
                    
                    val_name = emit_val_2d(var)
                    kernel_code_lines.append(f"{indent2}val = {val_name}")

                    # Inner states update
                    ops = self._variable_ops[var]
                    inner_ops = set(op.split('_')[1] for op in ops if '_' in op)
                    for inner in inner_ops:
                        # Initialize val_for_inner to avoid UnboundLocalError/NameError in generated code
                        # This value is used if is_update_outer is True, where it gets overwritten.
                        kernel_code_lines.append(f"{indent2}val_for_{inner} = tl.zeros_like(val)")

                        inner_ptr = f"{safe_var}_{inner}_inner_state_ptr + {out_offset}"
                        if inner == 'mean':
                             weight_ptr = f"{safe_var}_{inner}_weight_state_ptr + {out_offset}"
                             kernel_code_lines.extend([
                                 f"{indent2}inner_{inner}_old = tl.load({inner_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}weight_{inner}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}inner_{inner}_new = inner_{inner}_old + val * weight",
                                 f"{indent2}weight_{inner}_new = weight_{inner}_old + weight",
                                 f"{indent2}if is_macro_step_end:",
                                 f"{indent3}tl.store({inner_ptr}, 0.0, mask=mask)",
                                 f"{indent3}tl.store({weight_ptr}, 0.0, mask=mask)",
                                 f"{indent3}val_for_{inner} = inner_{inner}_new / (weight_{inner}_new)",
                                 f"{indent2}else:",
                                 f"{indent3}tl.store({inner_ptr}, inner_{inner}_new, mask=mask)",
                                 f"{indent3}tl.store({weight_ptr}, weight_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'sum':
                             kernel_code_lines.extend([
                                 f"{indent2}inner_{inner}_old = tl.load({inner_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}inner_{inner}_new = inner_{inner}_old + val * weight",
                                 f"{indent2}if is_macro_step_end:",
                                 f"{indent3}tl.store({inner_ptr}, 0.0, mask=mask)",
                                 f"{indent3}val_for_{inner} = inner_{inner}_new",
                                 f"{indent2}else:",
                                 f"{indent3}tl.store({inner_ptr}, inner_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'max':
                             weight_ptr = f"{safe_var}_{inner}_weight_state_ptr + {out_offset}"
                             kernel_code_lines.extend([
                                 f"{indent2}weight_{inner}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}weight_{inner}_new = weight_{inner}_old + weight",
                                 f"{indent2}if is_first and macro_step_index==0:", 
                                 f"{indent3}inner_{inner}_new = val",
                                 f"{indent2}else:",
                                 f"{indent3}inner_{inner}_old = tl.load({inner_ptr}, mask=mask, other=val)",
                                 f"{indent3}inner_{inner}_new = tl.maximum(inner_{inner}_old, val)",
                                 f"{indent2}if is_macro_step_end:",
                                 f"{indent3}tl.store({inner_ptr}, -float('inf'), mask=mask)",
                                 f"{indent3}val_for_{inner} = inner_{inner}_new",
                                 f"{indent3}val_weight_for_{inner} = weight_{inner}_new",
                                 f"{indent3}tl.store({weight_ptr}, 0.0, mask=mask)",
                                 f"{indent2}else:",
                                 f"{indent3}tl.store({inner_ptr}, inner_{inner}_new, mask=mask)",
                                 f"{indent3}tl.store({weight_ptr}, weight_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'min':
                             weight_ptr = f"{safe_var}_{inner}_weight_state_ptr + {out_offset}"
                             kernel_code_lines.extend([
                                 f"{indent2}weight_{inner}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}weight_{inner}_new = weight_{inner}_old + weight",
                                 f"{indent2}if is_first and macro_step_index==0:",
                                 f"{indent3}inner_{inner}_new = val",
                                 f"{indent2}else:",
                                 f"{indent3}inner_{inner}_old = tl.load({inner_ptr}, mask=mask, other=val)",
                                 f"{indent3}inner_{inner}_new = tl.minimum(inner_{inner}_old, val)",
                                 f"{indent2}if is_macro_step_end:",
                                 f"{indent3}tl.store({inner_ptr}, float('inf'), mask=mask)",
                                 f"{indent3}val_for_{inner} = inner_{inner}_new",
                                 f"{indent3}val_weight_for_{inner} = weight_{inner}_new",
                                 f"{indent3}tl.store({weight_ptr}, 0.0, mask=mask)",
                                 f"{indent2}else:",
                                 f"{indent3}tl.store({inner_ptr}, inner_{inner}_new, mask=mask)",
                                 f"{indent3}tl.store({weight_ptr}, weight_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'first':
                             weight_ptr = f"{safe_var}_{inner}_weight_state_ptr + {out_offset}"
                             kernel_code_lines.extend([
                                 f"{indent2}weight_{inner}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}weight_{inner}_new = weight_{inner}_old + weight",
                                 f"{indent2}val_stored = tl.load({inner_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}if weight_{inner}_old == 0.0:",
                                 f"{indent3}inner_{inner}_new = val",
                                 f"{indent2}else:",
                                 f"{indent3}inner_{inner}_new = val_stored",
                                 f"{indent2}if is_macro_step_end:",
                                 f"{indent3}val_for_{inner} = inner_{inner}_new",
                                 f"{indent3}val_weight_for_{inner} = weight_{inner}_new",
                                 f"{indent3}tl.store({weight_ptr}, 0.0, mask=mask)",
                                 f"{indent3}tl.store({inner_ptr}, 0.0, mask=mask)",
                                 f"{indent2}else:",
                                 f"{indent3}tl.store({inner_ptr}, inner_{inner}_new, mask=mask)",
                                 f"{indent3}tl.store({weight_ptr}, weight_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'last':
                             weight_ptr = f"{safe_var}_{inner}_weight_state_ptr + {out_offset}"
                             kernel_code_lines.extend([
                                 f"{indent2}weight_{inner}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}weight_{inner}_new = weight_{inner}_old + weight",
                                 f"{indent2}if is_macro_step_end:",
                                 f"{indent3}val_for_{inner} = val",
                                 f"{indent3}val_weight_for_{inner} = weight_{inner}_new",
                                 f"{indent3}tl.store({weight_ptr}, 0.0, mask=mask)",
                                 f"{indent2}else:",
                                 f"{indent3}tl.store({weight_ptr}, weight_{inner}_new, mask=mask)",
                             ])
                        elif inner == 'mid':
                             weight_ptr = f"{safe_var}_{inner}_weight_state_ptr + {out_offset}"
                             kernel_code_lines.extend([
                                 f"{indent2}weight_{inner}_old = tl.load({weight_ptr}, mask=mask, other=0.0)",
                                 f"{indent2}weight_{inner}_new = weight_{inner}_old + weight",
                                 f"{indent2}if is_middle:",
                                 f"{indent3}tl.store({inner_ptr}, val, mask=mask)",
                                 f"{indent2}if is_inner_last:",
                                 f"{indent3}val_for_{inner} = tl.load({inner_ptr}, mask=mask, other=0.0)",
                                 f"{indent3}val_weight_for_{inner} = weight_{inner}_new",
                                 f"{indent3}tl.store({weight_ptr}, 0.0, mask=mask)",
                                 f"{indent2}else:",
                                 f"{indent3}tl.store({weight_ptr}, weight_{inner}_new, mask=mask)",
                             ])

                    for op in self._variable_ops[var]:
                        out_ptr = f"{safe_var}_{op}_ptr + {out_offset}"
                        op_parts = op.split('_')
                        if len(op_parts) > 1:
                            outer = op_parts[0]
                            inner = op_parts[1]
                            
                            # Parse K
                            k_val = 1
                            match_k = re.match(r'(max|min)(\d+)$', outer)
                            if match_k:
                                outer = match_k.group(1) # normalize
                                k_val = int(match_k.group(2))

                            val_var = f"val_for_{inner}"
                            kernel_code_lines.append(f"{indent2}if is_macro_step_end:")
                            
                            if outer == 'max':
                                # argmax pointer (automatically created alongside max)
                                argmax_ptr = f"{safe_var}_arg{op}_ptr + {out_offset}"
                                if k_val == 1:
                                    kernel_code_lines.extend([
                                        f"{indent3}if is_outer_first and macro_step_index==0:",
                                        f"{indent4}tl.store({out_ptr}, {val_var}, mask=mask)",
                                        f"{indent4}tl.store({argmax_ptr}, macro_step_index, mask=mask)",
                                        f"{indent3}else:",
                                        f"{indent4}old = tl.load({out_ptr}, mask=mask, other={val_var})",
                                        f"{indent4}cond_mask = {val_var} > old",
                                        f"{indent4}new = tl.maximum(old, {val_var})",
                                        f"{indent4}tl.store({out_ptr}, new, mask=mask)",
                                        f"{indent4}tl.store({argmax_ptr}, macro_step_index, mask=mask & cond_mask)",
                                    ])
                                else:
                                    # Bubble Insert Max K with ArgMax
                                    argmax_op = f"arg{op}"
                                    kernel_code_lines.extend([
                                        f"{indent3}# Bubble Insert Max K={k_val} with ArgMax (static_range optimized)",
                                        f"{indent3}new_val = {val_var}",
                                        f"{indent3}new_idx = tl.full([BLOCK_SIZE], macro_step_index, tl.int32)",
                                        f"{indent3}k_offset = ({out_offset}) * {k_val}",
                                        f"{indent3}base_ptr = {safe_var}_{op}_ptr + k_offset",
                                        f"{indent3}idx_base_ptr = {safe_var}_{argmax_op}_ptr + k_offset",
                                        
                                        f"{indent3}if is_outer_first and macro_step_index==0:",
                                        f"{indent4}tl.store(base_ptr, new_val, mask=mask)",
                                        f"{indent4}tl.store(idx_base_ptr, new_idx, mask=mask)",
                                        f"{indent4}for k in tl.static_range(1, {k_val}):",
                                        f"{indent5}tl.store(base_ptr + k, -float('inf'), mask=mask)",
                                        f"{indent5}tl.store(idx_base_ptr + k, 0, mask=mask)",
                                        f"{indent3}else:",
                                        f"{indent4}for k in tl.static_range({k_val}):",
                                        f"{indent5}old_k = tl.load(base_ptr + k, mask=mask, other=-float('inf'))",
                                        f"{indent5}old_idx_k = tl.load(idx_base_ptr + k, mask=mask, other=0)",
                                        f"{indent5}swap_mask = new_val > old_k",
                                        f"{indent5}val_to_store = tl.where(swap_mask, new_val, old_k)",
                                        f"{indent5}idx_to_store = tl.where(swap_mask, new_idx, old_idx_k)",
                                        f"{indent5}new_val = tl.where(swap_mask, old_k, new_val)",
                                        f"{indent5}new_idx = tl.where(swap_mask, old_idx_k, new_idx)",
                                        f"{indent5}tl.store(base_ptr + k, val_to_store, mask=mask)",
                                        f"{indent5}tl.store(idx_base_ptr + k, idx_to_store, mask=mask)",
                                    ])

                            elif outer == 'min':
                                # argmin pointer (automatically created alongside min)
                                argmin_ptr = f"{safe_var}_arg{op}_ptr + {out_offset}"
                                if k_val == 1:
                                    kernel_code_lines.extend([
                                        f"{indent3}if is_outer_first and macro_step_index==0:",
                                        f"{indent4}tl.store({out_ptr}, {val_var}, mask=mask)",
                                        f"{indent4}tl.store({argmin_ptr}, macro_step_index, mask=mask)",
                                        f"{indent3}else:",
                                        f"{indent4}old = tl.load({out_ptr}, mask=mask, other={val_var})",
                                        f"{indent4}cond_mask = {val_var} < old",
                                        f"{indent4}new = tl.minimum(old, {val_var})",
                                        f"{indent4}tl.store({out_ptr}, new, mask=mask)",
                                        f"{indent4}tl.store({argmin_ptr}, macro_step_index, mask=mask & cond_mask)",
                                    ])
                                else:
                                    # Min K with ArgMin
                                    argmin_op = f"arg{op}"
                                    kernel_code_lines.extend([
                                        f"{indent3}# Bubble Insert Min K={k_val} with ArgMin (static_range optimized)",
                                        f"{indent3}new_val = {val_var}",
                                        f"{indent3}new_idx = tl.full([BLOCK_SIZE], macro_step_index, tl.int32)",
                                        f"{indent3}k_offset = ({out_offset}) * {k_val}",
                                        f"{indent3}base_ptr = {safe_var}_{op}_ptr + k_offset",
                                        f"{indent3}idx_base_ptr = {safe_var}_{argmin_op}_ptr + k_offset",
                                        
                                        f"{indent3}if is_outer_first and macro_step_index==0:",
                                        f"{indent4}tl.store(base_ptr, new_val, mask=mask)",
                                        f"{indent4}tl.store(idx_base_ptr, new_idx, mask=mask)",
                                        f"{indent4}for k in tl.static_range(1, {k_val}):",
                                        f"{indent5}tl.store(base_ptr + k, float('inf'), mask=mask)",
                                        f"{indent5}tl.store(idx_base_ptr + k, 0, mask=mask)",
                                        f"{indent3}else:",
                                        f"{indent4}for k in tl.static_range({k_val}):",
                                        f"{indent5}old_k = tl.load(base_ptr + k, mask=mask, other=float('inf'))",
                                        f"{indent5}old_idx_k = tl.load(idx_base_ptr + k, mask=mask, other=0)",
                                        f"{indent5}swap_mask = new_val < old_k",
                                        f"{indent5}val_to_store = tl.where(swap_mask, new_val, old_k)",
                                        f"{indent5}idx_to_store = tl.where(swap_mask, new_idx, old_idx_k)",
                                        f"{indent5}new_val = tl.where(swap_mask, old_k, new_val)",
                                        f"{indent5}new_idx = tl.where(swap_mask, old_idx_k, new_idx)",
                                        f"{indent5}tl.store(base_ptr + k, val_to_store, mask=mask)",
                                        f"{indent5}tl.store(idx_base_ptr + k, idx_to_store, mask=mask)",
                                    ])
                            elif outer == 'sum':
                                kernel_code_lines.extend([
                                    f"{indent3}if is_outer_first and macro_step_index==0:",
                                    f"{indent4}tl.store({out_ptr}, {val_var}, mask=mask)",
                                    f"{indent3}else:",
                                    f"{indent4}old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                    f"{indent4}tl.store({out_ptr}, old + {val_var}, mask=mask)",
                                ])
                            elif outer == 'mean':
                                term = f"{val_var}"
                                kernel_code_lines.extend([
                                    f"{indent3}if is_outer_first and macro_step_index==0:",
                                    f"{indent4}tl.store({out_ptr}, {term}, mask=mask)",
                                    f"{indent3}else:",
                                    f"{indent4}old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                    f"{indent4}tl.store({out_ptr}, old + {term}, mask=mask)",
                                    f"{indent3}if is_outer_last:",
                                    f"{indent4}chk = tl.load({out_ptr}, mask=mask)",
                                    f"{indent4}tl.store({out_ptr}, chk / num_macro_steps, mask=mask)",
                                ])
                            continue

                        if op == 'mean':
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}old = tl.zeros_like(val)",
                                f"{indent2}else:",
                                f"{indent3}old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                f"{indent2}new = old + val * weight",
                                f"{indent2}if is_inner_last:",
                                f"{indent3}new = new / total_weight",
                                f"{indent2}tl.store({out_ptr}, new, mask=mask)",
                            ])
                        elif op == 'sum':
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}old = tl.zeros_like(val)",
                                f"{indent2}else:",
                                f"{indent3}old = tl.load({out_ptr}, mask=mask, other=0.0)",
                                f"{indent2}new = old + val * weight",
                                f"{indent2}tl.store({out_ptr}, new, mask=mask)",
                            ])
                        elif op == 'max':
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                                f"{indent2}else:",
                                f"{indent3}old = tl.load({out_ptr}, mask=mask, other=val)",
                                f"{indent3}new = tl.maximum(old, val)",
                                f"{indent3}tl.store({out_ptr}, new, mask=mask)",
                            ])
                        elif op == 'min':
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                                f"{indent2}else:",
                                f"{indent3}old = tl.load({out_ptr}, mask=mask, other=val)",
                                f"{indent3}new = tl.minimum(old, val)",
                                f"{indent3}tl.store({out_ptr}, new, mask=mask)",
                            ])
                        elif op == 'argmax':
                            # 2D argmax logic
                            max_ptr = f"{safe_var}_max_ptr + {out_offset}"
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}tl.store({out_ptr}, macro_step_index, mask=mask)",
                                f"{indent2}else:",
                                f"{indent3}curr_max = tl.load({max_ptr}, mask=mask, other=val)",
                                f"{indent3}cond_mask = val > curr_max",
                                f"{indent3}tl.store({out_ptr}, macro_step_index, mask=mask & cond_mask)",
                            ])
                        elif op == 'argmin':
                            min_ptr = f"{safe_var}_min_ptr + {out_offset}"
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}tl.store({out_ptr}, macro_step_index, mask=mask)",
                                f"{indent2}else:",
                                f"{indent3}curr_min = tl.load({min_ptr}, mask=mask, other=val)",
                                f"{indent3}cond_mask = val < curr_min",
                                f"{indent3}tl.store({out_ptr}, macro_step_index, mask=mask & cond_mask)",
                            ])
                        elif op == 'last':
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_last:",
                                f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                            ])
                        elif op == 'first':
                            kernel_code_lines.extend([
                                f"{indent2}if is_inner_first:",
                                f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                            ])
                        elif op == 'mid':
                            kernel_code_lines.extend([
                                f"{indent2}if is_middle:",
                                f"{indent3}tl.store({out_ptr}, val, mask=mask)",
                            ])
                kernel_code_lines.append("")

            if last_only_vars:
                kernel_code_lines.extend([
                    f"{indent}# 2D variables (last-only)",
                    f"{indent}if is_inner_last:",
                    f"{indent2}for level in tl.static_range(n_levels):",
                ])
                for var in last_only_vars:
                    safe_var = self._get_safe_name(var)
                    out_offset = f"(t * n_saved_points + offs) * n_levels + level"
                    
                    val_name = emit_val_2d(var)
                    kernel_code_lines.extend([
                        f"{indent3}val = {val_name}",
                        f"{indent3}tl.store({safe_var}_last_ptr + {out_offset}, val, mask=mask)",
                    ])
        kernel_code_lines.append("")


    def _generate_main_function(self: StatisticsAggregator, kernel_code_lines: List[str],
                                grouped_by_save_idx: Dict[str, List[str]],
                                tensor_info: Dict[str, Dict[str, Any]]) -> None:
        """Generate the main python function that calls kernels."""
        kernel_code_lines.extend([
            "# Main update function",
            "def internal_update_statistics(states, weight, total_weight, num_macro_steps, is_inner_first, is_inner_last, is_middle, is_outer_first, is_outer_last, macro_step_index, step_count_val, BLOCK_SIZE):",
        ])
        
        if self.num_trials > 1:
             kernel_code_lines.append(f"    num_trials = {self.num_trials}")
        else:
             kernel_code_lines.append(f"    num_trials = 1")

        for save_idx, var_list in grouped_by_save_idx.items():
            kernel_name = f"kernel_{save_idx}"
            
            # Get stride_input from metadata of first variable
            first_var = var_list[0]
            stride_input = 0
            for out_name, meta in self._metadata.items():
                if meta['original_variable'] == first_var:
                    stride_input = meta.get('stride_input', 0)
                    break
            
            kernel_code_lines.extend([
                f"    # Launch kernel for {save_idx}",
                f"    save_idx_len = len(states['{save_idx}'])",
                f"    stride_input = {stride_input}",
                f"    grid_{save_idx} = lambda meta: (triton.cdiv(save_idx_len, meta['BLOCK_SIZE']),)",
                f"    {kernel_name}[grid_{save_idx}](",
                f"        {save_idx}_ptr=states['{save_idx}'],",
            ])
            
            # Gather input pointers (resolving virtuals)
            input_args = set()
            def _gather_inputs(name):
                 info = self._field_registry.get(name)
                 if getattr(info, 'json_schema_extra', {}).get('category') == 'virtual':
                      expr = getattr(info, 'json_schema_extra', {}).get('expr', '')
                      import re
                      toks = set(re.findall(r'\b[a-zA-Z_]\w*\b', expr))
                      for t in toks:
                           if t in self._field_registry or t in self._tensor_registry:
                                _gather_inputs(t)
                 else:
                      input_args.add(name)
            
            for var in var_list:
                 _gather_inputs(var)

            # Add Input pointers
            sorted_inputs = sorted(list(input_args))
            for var in sorted_inputs:
                 safe_var = self._get_safe_name(var)
                 # Avoid duplicate argument if save_idx matches input var
                 if safe_var == save_idx:
                     continue
                 kernel_code_lines.append(f"        {safe_var}_ptr=states['{var}'],")
            
            # Add variable output pointers
            for var in var_list:
                safe_var = self._get_safe_name(var)
                added_aux_ptrs = set()  # Track aux pointers for explicit argmax/argmin
                for op in self._variable_ops[var]:
                    kernel_code_lines.append(f"        {safe_var}_{op}_ptr=states['{var}_{op}'],")
                    
                    # For EXPLICIT argmax/argmin operations, add aux pointer
                    # NO automatic argmax/argmin generation for max/min operations
                    op_parts = op.split('_')
                    outer_op = op_parts[0]
                    
                    # Check for explicit argmax/argmin (e.g., argmax, argmax3, argmin, argmin3)
                    arg_match = re.match(r'arg(max|min)(\d*)$', outer_op)
                    if arg_match:
                        arg_type = arg_match.group(1)  # 'max' or 'min'
                        arg_k_str = arg_match.group(2)  # '' or '3' etc
                        aux_name = f"{arg_type}{arg_k_str or ''}_aux"  # e.g., 'max_aux', 'max3_aux'
                        if aux_name not in added_aux_ptrs:
                            aux_storage_key = f"{var}_{arg_type}{arg_k_str if arg_k_str else ''}_aux"
                            kernel_code_lines.append(f"        {safe_var}_{aux_name}_ptr=states['{aux_storage_key}'],")
                            added_aux_ptrs.add(aux_name)
                
                # Inner state pointers (only for ops that need cross-step state)
                added_inner = set()
                for op in self._variable_ops[var]:
                    if '_' in op:
                        inner = op.split('_')[1]
                        if inner not in added_inner:
                             # 'last' inner op directly uses current value, no state needed
                             if inner != 'last':
                                 kernel_code_lines.append(f"        {safe_var}_{inner}_inner_state_ptr=states['{var}_{inner}_inner_state'],")
                             if inner == 'mean':
                                 kernel_code_lines.append(f"        {safe_var}_{inner}_weight_state_ptr=states['{var}_{inner}_weight_state'],")
                             added_inner.add(inner)
            
            kernel_code_lines.extend([
                "        weight=weight,",
                "        total_weight=total_weight,",
                "        num_macro_steps=num_macro_steps,",
                "        is_inner_first=is_inner_first,",
                "        is_inner_last=is_inner_last,",
                "        is_middle=is_middle,",
                "        is_outer_first=is_outer_first,",
                "        is_outer_last=is_outer_last,",
                "        macro_step_index=macro_step_index,",
                "        step_count_val=step_count_val,",
                "        n_saved_points=save_idx_len,",
            ])
            
            # Add second dimension if needed (use actual shape)
            if self.num_trials > 1:
                dims_2d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 3]
            else:
                dims_2d = [v for v in var_list if tensor_info[v]['actual_ndim'] == 2]

            if dims_2d:
                var_2d = dims_2d[0]
                actual_shape = tensor_info[var_2d]['actual_shape']
                n_levels = actual_shape[-1]
                kernel_code_lines.append(f"        n_levels={n_levels},")
            
            kernel_code_lines.extend([
                "        BLOCK_SIZE=BLOCK_SIZE,",
                "        num_trials=num_trials,",
                "        stride_input=stride_input,",
                "    )",
                "",
            ])


    def _generate_aggregator_function(self: StatisticsAggregator) -> None:
        """
        Generate and compile the aggregation kernel function.
        """
        if not self._variables:
            raise ValueError("No variables initialized for statistics aggregation")

        # Analyze tensor information and group by save_idx
        tensor_info = {}
        grouped_by_save_idx = {}
        
        for var_name in self._variables:
            field_info = self._field_registry[var_name]
            tensor = self._tensor_registry.get(var_name)
            
            # Virtual fallback for meta info construction
            if tensor is None:
                 # Check if meta info was constructed during init
                 first_op = self._variable_ops[var_name][0]
                 out_name = f"{var_name}_{first_op}"
                 meta = self._metadata[out_name]
                 
                 tensor_info[var_name] = {
                    'tensor': None,
                    'tensor_shape': meta['tensor_shape'],
                    'actual_shape': meta['actual_shape'],
                    'actual_ndim': meta['actual_ndim']
                }
            else:
                json_schema_extra = getattr(field_info, 'json_schema_extra', {})
                save_idx = json_schema_extra.get('save_idx')
                tensor_shape = json_schema_extra.get('tensor_shape', ())
                
                tensor_info[var_name] = {
                    'tensor': tensor,
                    'tensor_shape': tensor_shape,  # Logical grid shape
                    'actual_shape': tensor.shape,  # Sampled data shape
                    'actual_ndim': tensor.ndim     # Based on actual data
                }
                
            # Need save_idx to group
            json_schema_extra = getattr(field_info, 'json_schema_extra', {})
            save_idx = json_schema_extra.get('save_idx')
            
            if save_idx not in grouped_by_save_idx:
                grouped_by_save_idx[save_idx] = []
            grouped_by_save_idx[save_idx].append(var_name)
        
        # Generate kernel code
        kernel_code_lines = self._generate_kernel_header()
        
        # Generate kernels for each save_idx group
        for save_idx, var_list in grouped_by_save_idx.items():
            kernel_name = f"kernel_{save_idx}"
            self._generate_kernel_for_group(kernel_code_lines, kernel_name, save_idx, var_list, tensor_info)
        
        # Generate main function
        self._generate_main_function(kernel_code_lines, grouped_by_save_idx, tensor_info)
        
        # Write kernel code to temporary file and import
        kernel_code = "\n".join(kernel_code_lines)
        self._write_and_import_kernels(kernel_code)
        
        # Save kernel file for external inspection if enabled
        if self.save_kernels:
            self._save_kernel_file(kernel_code)
