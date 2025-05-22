import torch
import torch.distributed as dist
import triton
import triton.language as tl
import os

@triton.jit
def compute_adaptive_time_step_kernel(
    is_reservoir_ptr,                       # *bool mask: 1 means reservoir
    downstream_idx_ptr,                     # *i32 downstream index
    river_depth_ptr,                        # *f32 river depth
    downstream_distance_ptr,                # *f32 distance to downstream unit
    min_time_step_ptr,
    adaptation_factor: tl.constexpr ,
    gravity: tl.constexpr ,                                # f32 scalar gravity acceleration
    default_time_step: tl.constexpr,                             # f32 scalar time step
    num_catchments: tl.constexpr,           # total number of elements
    BLOCK_SIZE: tl.constexpr                # block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_catchments

    #----------------------------------------------------------------------
    # (1) Load input variables
    #----------------------------------------------------------------------
    downstream_idx = tl.load(downstream_idx_ptr + offs, mask=mask, other=0)
    is_reservoir = tl.load(is_reservoir_ptr + offs, mask=mask, other=0)
    is_reservoir_downstream = tl.load(is_reservoir_ptr + downstream_idx, mask=mask, other=0)
    # omit reservoirs grids (reservoir or upstream of reservoir) 
    mask = ~(is_reservoir_downstream | is_reservoir) & mask 
    downstream_distance = tl.load(downstream_distance_ptr + offs, mask=mask, other=float('inf'))
    # Clamp river depth to minimum 0.01 for stability
    river_depth = tl.load(river_depth_ptr + offs, mask=mask, other=0)
    depth = tl.maximum(river_depth, 0.01)
    dt = adaptation_factor * downstream_distance / tl.sqrt(gravity * depth)
    dt_clamped = tl.minimum(dt, default_time_step)
    
    min_time_step = tl.min(dt_clamped)
    tl.atomic_min(min_time_step_ptr, min_time_step)


def compute_adaptive_time_step(
    is_reservoir: torch.Tensor,
    downstream_idx: torch.Tensor,
    river_depth: torch.Tensor,
    downstream_distance: torch.Tensor,
    min_time_step: torch.Tensor,
    default_time_step: float,
    adaptation_factor: float,
    gravity: float,
    num_catchments:int,
    block_size: int
):

    grid = lambda meta: (triton.cdiv(num_catchments, meta['BLOCK_SIZE']),)
    min_time_step.fill_(float('inf'))
    compute_adaptive_time_step_kernel[grid](
        is_reservoir_ptr=is_reservoir,
        downstream_idx_ptr=downstream_idx,
        river_depth_ptr=river_depth,
        downstream_distance_ptr=downstream_distance,
        min_time_step_ptr=min_time_step,
        adaptation_factor=adaptation_factor,
        gravity=gravity,
        default_time_step=default_time_step,
        num_catchments=num_catchments,
        BLOCK_SIZE=block_size
    )
    if int(os.getenv('WORLD_SIZE', 1)) > 1:
        dist.all_reduce(min_time_step, op=dist.ReduceOp.MIN)
    num_sub_steps = int(round(default_time_step / min_time_step.item() - 0.01) + 1)
    
    adaptive_time_step = default_time_step / num_sub_steps

    return adaptive_time_step, num_sub_steps
