import triton
import triton.language as tl


@triton.jit
def compute_adaptive_time_step_kernel(
    is_reservoir_ptr,                       # *bool mask: 1 means reservoir
    downstream_idx_ptr,                     # *i32 downstream index
    river_depth_ptr,                        # *f32 river depth
    downstream_distance_ptr,                # *f32 distance to downstream unit
    min_time_sub_step_ptr,
    time_step,
    adaptive_time_factor: tl.constexpr ,
    gravity: tl.constexpr ,                                # f32 scalar gravity acceleration
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
    dt = adaptive_time_factor * downstream_distance / tl.sqrt(gravity * depth)
    dt_clamped = tl.minimum(dt, time_step)
    
    min_time_sub_step = tl.min(dt_clamped)
    tl.atomic_min(min_time_sub_step_ptr, min_time_sub_step)
