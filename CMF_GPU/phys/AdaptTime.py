import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_compute_ts(
    river_depth_ptr,           # *f32: River depth
    downstream_dist_ptr,       # *f32: Distance to downstream unit
    local_ts_ptr,              # *f32: Local time step (output)
    n_elements,                # int: Total number of elements
    gravity,                   # f32: Gravity acceleration
    adaptation_factor,         # f32: Adaptation factor for time step
    default_time_step,         # f32: Maximum allowed time step
    BLOCK_SIZE: tl.constexpr   # Block size
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load and clamp depth
    depth = tl.load(river_depth_ptr + offsets, mask=mask, other=0.0)
    depth = tl.maximum(depth, 0.01)

    # Load distance to downstream
    dist = tl.load(downstream_dist_ptr + offsets, mask=mask, other=0.0)

    # Compute local time step and clip to default
    ts = adaptation_factor * dist / tl.sqrt(gravity * depth)
    ts_clipped = tl.minimum(default_time_step, ts)

    # Store result
    tl.store(local_ts_ptr + offsets, ts_clipped, mask=mask)

def compute_adaptive_time_step(
    river_depth: torch.Tensor,
    downstream_distance: torch.Tensor,
    default_time_step: float,
    adaptation_factor: float,
    gravity: float,
    BLOCK_SIZE: int = 1024
):
    """
    Computes an adaptive time step for flood simulation using a Triton kernel.

    Args:
        river_depth (torch.Tensor): 1D CUDA tensor of river depths.
        downstream_distance (torch.Tensor): 1D CUDA tensor of downstream distances.
        default_time_step (float): Default simulation time step.
        adaptation_factor (float): Factor controlling time step adaptation.
        gravity (float): Gravitational acceleration.

    Returns:
        tuple: (adaptive_time_step, num_sub_steps)
            - adaptive_time_step (float): Adjusted time step.
            - num_sub_steps (int): Number of sub-steps for the simulation.
    """

    # Total number of elements
    n = river_depth.numel()
    # Temporary buffer for per‐element time‐steps
    local_ts = torch.empty_like(river_depth)

    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)

    # Compute per‐element time‐steps
    _kernel_compute_ts[grid](
        river_depth, downstream_distance, local_ts,
        n, gravity, adaptation_factor, default_time_step,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Reduce to find the minimum time‐step
    min_ts = local_ts.min().item()

    # Compute number of sub‐steps and final time‐step
    num_sub_steps = int(round(default_time_step / min_ts - 0.01) + 1)
    adaptive_time_step = default_time_step / num_sub_steps

    return adaptive_time_step, num_sub_steps

if __name__ == "__main__":
    # A simple test case
    device = 'cuda'
    # Some example depths (one value is below 0.01 to test the clamp)
    river_depth = torch.tensor([0.005, 0.1, 0.2, 0.05], dtype=torch.float32, device=device)
    downstream_distance = torch.tensor([10.0, 20.0, 5.0, 15.0], dtype=torch.float32, device=device)

    default_dt = 1.0
    adapt_factor = 0.5
    gravity = 9.81

    adaptive_dt, steps = compute_adaptive_time_step(
        river_depth, downstream_distance, default_dt, adapt_factor, gravity
    )
    print(f"Adaptive time step: {adaptive_dt}")
    print(f"Number of sub-steps: {steps}")