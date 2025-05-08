import triton
import triton.language as tl

@triton.jit
def update_stats_aggregator(
    river_outflow_ptr,         # *f32 [N]
    river_outflow_min_ptr,     # *f32 [N]
    river_outflow_max_ptr,     # *f32 [N]
    river_outflow_mean_ptr,    # *f32 [N]
    current_step,
    num_sub_steps,
    num_catchments: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_catchments
    curr     = tl.load(river_outflow_ptr + offs, mask=mask, other=0.0)

    if current_step == 0:
        old_min = tl.zeros_like(curr)
        old_max = tl.zeros_like(curr)
        old_mean = tl.zeros_like(curr)
    else:
        old_min  = tl.load(river_outflow_min_ptr  + offs, mask=mask, other=0.0)
        old_max  = tl.load(river_outflow_max_ptr  + offs, mask=mask, other=0.0)
        old_mean = tl.load(river_outflow_mean_ptr + offs, mask=mask, other=0.0)

    new_min  = tl.minimum(curr, old_min)
    new_max  = tl.maximum(curr, old_max)
    new_mean = old_mean + curr / num_sub_steps

    tl.store(river_outflow_min_ptr  + offs, new_min,  mask=mask)
    tl.store(river_outflow_max_ptr  + offs, new_max,  mask=mask)
    tl.store(river_outflow_mean_ptr + offs, new_mean, mask=mask)