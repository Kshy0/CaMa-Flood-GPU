import triton
import triton.language as tl

default_statistics = {"river_outflow": ["min", "max", "mean"]}

@triton.jit
def update_stats_aggregator(
    river_outflow,         # *f32 [N]
    river_outflow_min,     # *f32 [N]
    river_outflow_max,     # *f32 [N]
    river_outflow_mean,    # *f32 [N]
    num_sub_steps,         # *i32 [1] or scalar
    num_catchments: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_catchments

    curr     = tl.load(river_outflow      + offs, mask=mask, other=0.0)
    old_min  = tl.load(river_outflow_min  + offs, mask=mask, other=0.0)
    old_max  = tl.load(river_outflow_max  + offs, mask=mask, other=0.0)
    old_mean = tl.load(river_outflow_mean + offs, mask=mask, other=0.0)
    nss      = tl.load(num_sub_steps)  # 假定广播

    new_min  = tl.minimum(old_min, curr)
    new_max  = tl.maximum(old_max, curr)
    new_mean = old_mean + curr / nss

    tl.store(river_outflow_min  + offs, new_min,  mask=mask)
    tl.store(river_outflow_max  + offs, new_max,  mask=mask)
    tl.store(river_outflow_mean + offs, new_mean, mask=mask)