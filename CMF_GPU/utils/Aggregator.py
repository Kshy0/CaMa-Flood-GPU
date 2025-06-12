from pathlib import Path

LEGAL_STATS = [
    "min",
    "max",
    "mean",
]

LEGAL_AGG_ARRAYS = {
    ("catchment_save_idx", ("num_catchments_to_save",)): [
        "river_storage",
        "flood_storage",
        "river_depth",
        "river_outflow",
        "flood_depth",
        "flood_outflow",
        "river_cross_section_depth",
        "flood_cross_section_depth",
        "flood_cross_section_area",
        "total_storage",
        "water_surface_elevation",
        "limit_rate",
        "flood_area",
        "flood_fraction",
    ],
    ("bifurcation_path_save_idx", ("num_bifurcation_paths_to_save", "num_bifurcation_levels")): [
        "bifurcation_outflow",
        "bifurcation_cross_section_depth",
    ],
    # ("num_bifurcation_paths",): [
    #     "bifurcation_outflow_sum",
    # ],
}

# dim_length: coord_name
DIM_INFO = {
    "num_catchments_to_save": "catchment_save_idx",
    "num_bifurcation_paths_to_save": "bifurcation_path_save_idx",
    "num_bifurcation_levels": "bifurcation_level_idx",  
}

def generate_triton_aggregator_script(script_path: Path, agg_keys: dict):
    """
    Generate a Triton kernel script for aggregation based on agg_keys.

    Parameters
    ----------
    script_path : Path
        The path to write the generated script
    agg_keys : dict
        {'min': [...], 'max': [...], 'mean': [...]} style input
    """

    # ---------- 1. Map variable to shape key ----------
    default_lines = [
        "def update_statistics(params, states, current_step, num_sub_steps, BLOCK_SIZE):",
        "    pass"
    ]
    var_to_shape = {}
    for (save_idx_name, shape_key_tuple), vars_ in LEGAL_AGG_ARRAYS.items():
        for v in vars_:
            var_to_shape[v] = (save_idx_name, shape_key_tuple)

    # ---------- 2. Group by dim0 and record agg types for each variable ----------
    grouped = {}  # key = (dim0, save_idx_name) → var → set(aggs)

    for agg, vars_ in agg_keys.items():
        if vars_ is not None:
            for v in vars_:
                if v not in var_to_shape:
                    raise ValueError(f"Variable '{v}' not in LEGAL_AGG_ARRAYS")
                save_idx_name, shape_keys = var_to_shape[v]
                dim0 = shape_keys[0]
                grouped.setdefault((dim0, save_idx_name), {}).setdefault(v, set()).add(agg)
    if not grouped:
        script_path.write_text("\n".join(default_lines), encoding="utf-8")
        return
    # ---------- 3. Code buffer ----------
    lines: list[str] = [
        "import triton",
        "import triton.language as tl",
        "",
    ]

    # ---------- 4. Generate kernel for each dim0 ----------
    for (dim0, save_idx_name), var_to_aggs in grouped.items():
        # 4.1 Classify 1-D / 2-D variables
        vars_1d = [v for v in var_to_aggs if len(var_to_shape[v][1]) == 1]
        vars_2d = [v for v in var_to_aggs if len(var_to_shape[v][1]) == 2]
        dim1 = None
        if vars_2d:
            second_dims = {var_to_shape[v][1][1] for v in vars_2d}
            if len(second_dims) != 1:
                raise ValueError(
                    f"Vars under dim0 '{dim0}' have multiple dim1 names: {second_dims}")
            dim1 = second_dims.pop()  # e.g. 'num_bifurcation_levels'

        kernel_name = f"update_stats_kernel__{dim0}"
        # 4.2 Parameter list (deduplicated)
        ptrs = set()
        ptrs.add(f"{save_idx_name}_ptr")
        for v, aggs in var_to_aggs.items():
            ptrs.add(f"{v}_ptr")
            for agg in ("min", "max", "mean"):
                if agg in aggs:
                    ptrs.add(f"{v}_{agg}_ptr")

        arg_lines = [f"    {p}," for p in sorted(ptrs)]
        arg_lines += [
            "    current_step,",
            "    num_sub_steps,",
            f"    {dim0}: tl.constexpr,",
        ]
        if dim1:
            arg_lines.append(f"    {dim1}: tl.constexpr,")
        arg_lines.append("    BLOCK_SIZE: tl.constexpr,")
        # 4.3 Kernel header
        lines += [
            "@triton.jit",
            f"def {kernel_name}(",
            *arg_lines,
            "):",
        ]
        lines.append("    pid  = tl.program_id(0)")
        lines.append("    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
        lines.append(f"    mask = offs < {dim0}")
        lines.append(f"    idx  = tl.load({save_idx_name}_ptr + offs, mask=mask)")
        # ------------------ Handle 1-D variables ------------------ #
        if vars_1d:
            lines.append("    # -------- 1-D variables --------")
            for v in vars_1d:
                lines.append(f"    {v} = tl.load({v}_ptr + idx, mask=mask, other=0.0)")
            lines.append("    # init / load previous")
            lines.append("    if current_step == 0:")
            for v in vars_1d:
                aggs = var_to_aggs[v]
                if "min" in aggs:
                    lines.append(
                        f"        {v}_old_min  = tl.full((BLOCK_SIZE,), float('inf'), {v}.dtype)")
                if "max" in aggs:
                    lines.append(
                        f"        {v}_old_max  = tl.full((BLOCK_SIZE,), float('-inf'), {v}.dtype)")
                if "mean" in aggs:
                    lines.append(
                        f"        {v}_old_mean = tl.zeros_like({v})")
            lines.append("    else:")
            for v in vars_1d:
                aggs = var_to_aggs[v]
                if "min" in aggs:
                    lines.append(
                        f"        {v}_old_min  = tl.load({v}_min_ptr  + offs, mask=mask, other=float('inf'))")
                if "max" in aggs:
                    lines.append(
                        f"        {v}_old_max  = tl.load({v}_max_ptr  + offs, mask=mask, other=float('-inf'))")
                if "mean" in aggs:
                    lines.append(
                        f"        {v}_old_mean = tl.load({v}_mean_ptr + offs, mask=mask, other=0.0)")
            # compute
            for v in vars_1d:
                aggs = var_to_aggs[v]
                if "min" in aggs:
                    lines.append(f"    {v}_new_min  = tl.minimum({v}, {v}_old_min)")
                if "max" in aggs:
                    lines.append(f"    {v}_new_max  = tl.maximum({v}, {v}_old_max)")
                if "mean" in aggs:
                    lines.append(f"    {v}_new_mean = {v}_old_mean + {v} / num_sub_steps")
            # store
            for v in vars_1d:
                aggs = var_to_aggs[v]
                if "min" in aggs:
                    lines.append(
                        f"    tl.store({v}_min_ptr  + offs, {v}_new_min,  mask=mask)")
                if "max" in aggs:
                    lines.append(
                        f"    tl.store({v}_max_ptr  + offs, {v}_new_max,  mask=mask)")
                if "mean" in aggs:
                    lines.append(
                        f"    tl.store({v}_mean_ptr + offs, {v}_new_mean, mask=mask)")
            lines.append("")

        # ------------------ Handle 2-D variables ------------------ #
        if vars_2d:
            lines.append("    # -------- 2-D variables --------")
            lines.append(f"    for level in tl.static_range({dim1}):")
            for v in vars_2d:
                lines.append(
                    f"        {v} = tl.load({v}_ptr + idx * {dim1} + level, mask=mask, other=0.0)")
            lines.append("        if current_step == 0:")
            for v in vars_2d:
                aggs = var_to_aggs[v]
                if "min" in aggs:
                    lines.append(
                        f"            {v}_old_min  = tl.full((BLOCK_SIZE,), float('inf'), {v}.dtype)")
                if "max" in aggs:
                    lines.append(
                        f"            {v}_old_max  = tl.full((BLOCK_SIZE,), float('-inf'), {v}.dtype)")
                if "mean" in aggs:
                    lines.append(
                        f"            {v}_old_mean = tl.zeros_like({v})")
            lines.append("        else:")
            for v in vars_2d:
                aggs = var_to_aggs[v]
                if "min" in aggs:
                    lines.append(
                        f"            {v}_old_min  = tl.load({v}_min_ptr  + offs * {dim1} + level, mask=mask, other=float('inf'))")
                if "max" in aggs:
                    lines.append(
                        f"            {v}_old_max  = tl.load({v}_max_ptr  + offs * {dim1} + level, mask=mask, other=float('-inf'))")
                if "mean" in aggs:
                    lines.append(
                        f"            {v}_old_mean = tl.load({v}_mean_ptr + offs * {dim1} + level, mask=mask, other=0.0)")
            # compute
            for v in vars_2d:
                aggs = var_to_aggs[v]
                if "min" in aggs:
                    lines.append(
                        f"        {v}_new_min  = tl.minimum({v}, {v}_old_min)")
                if "max" in aggs:
                    lines.append(
                        f"        {v}_new_max  = tl.maximum({v}, {v}_old_max)")
                if "mean" in aggs:
                    lines.append(
                        f"        {v}_new_mean = {v}_old_mean + {v} / num_sub_steps")
            # store
            for v in vars_2d:
                aggs = var_to_aggs[v]
                if "min" in aggs:
                    lines.append(
                        f"        tl.store({v}_min_ptr  + offs * {dim1} + level, {v}_new_min,  mask=mask)")
                if "max" in aggs:
                    lines.append(
                        f"        tl.store({v}_max_ptr  + offs * {dim1} + level, {v}_new_max,  mask=mask)")
                if "mean" in aggs:
                    lines.append(
                        f"        tl.store({v}_mean_ptr + offs * {dim1} + level, {v}_new_mean, mask=mask)")
            lines.append("")

    # ---------- 5. Unified entry point function update_stats ----------
    lines += [
        "",
        "def update_statistics(params, states, current_step, num_sub_steps, BLOCK_SIZE):",
    ]
    for (dim0, save_idx_name), var_to_aggs in grouped.items():
        kernel_name = f"update_stats_kernel__{dim0}"
        grid_name   = f'grid_{dim0}'
        lines.append(f"    {grid_name} = lambda meta: (triton.cdiv(params['{dim0}'], meta['BLOCK_SIZE']),)")
        lines.append(f"    {kernel_name}[{grid_name}](")
        # 5.1 Pass pointers
        lines.append(f"        {save_idx_name}_ptr=params['{save_idx_name}'],")
        for v, aggs in sorted(var_to_aggs.items()):
            lines.append(f"        {v}_ptr=states['{v}'],")
            if 'min' in aggs:
                lines.append(f"        {v}_min_ptr=states['{v}_min'],")
            if 'max' in aggs:
                lines.append(f"        {v}_max_ptr=states['{v}_max'],")
            if 'mean' in aggs:
                lines.append(f"        {v}_mean_ptr=states['{v}_mean'],")
        # 5.2 Other parameters
        lines += [
            "        current_step=current_step,",
            "        num_sub_steps=num_sub_steps,",
            f"        {dim0}=params['{dim0}'],",
        ]
        # If there are 2-D variables, add dim1
        vars_2d = [v for v in var_to_aggs if len(var_to_shape[v][1]) == 2]
        if vars_2d:
            dim1 = var_to_shape[vars_2d[0]][1][1]
            lines.append(f"        {dim1}=params['{dim1}'],")
        lines.append("        BLOCK_SIZE=BLOCK_SIZE")
        lines.append("    )\n")

    # ---------- 6. Write to file ----------
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\n".join(lines), encoding="utf-8")

# --------------------------- Example run --------------------------- #

if __name__ == "__main__":
    agg_keys = {
        'min':  None,
        'max':  ['river_storage'],
        'mean': ['river_storage', 'bifurcation_outflow'],
    }
    out = Path("triton_aggregator.py")
    generate_triton_aggregator_script(out, agg_keys)
    print(f"Generated Triton aggregator script at: {out.resolve()}")
