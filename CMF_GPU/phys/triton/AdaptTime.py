import torch

def compute_adaptive_time_step(
    river_depth: torch.Tensor,
    downstream_distance: torch.Tensor,
    default_time_step: float,
    adaptation_factor: float,
    gravity: float,
):
    """
    Computes an adaptive time step for flood simulation (no Triton kernel).

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

    # Clamp river depth to minimum 0.01 for stability
    depth = torch.clamp(river_depth, min=0.01)
    # Compute per-element time step and clip to default
    dt_min = min((adaptation_factor * downstream_distance / torch.sqrt(gravity * depth)).min().item(), default_time_step)

    # Compute number of sub-steps and final time step``
    num_sub_steps = int(round(default_time_step / dt_min - 0.01) + 1)
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