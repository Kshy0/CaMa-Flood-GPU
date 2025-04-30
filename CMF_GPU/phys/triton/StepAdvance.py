import torch
from triton import cdiv
from CMF_GPU.phys.triton.AdaptTime import compute_adaptive_time_step
from CMF_GPU.phys.triton.Outflow import compute_outflow_kernel, compute_inflow_kernel
from CMF_GPU.phys.triton.Storage import compute_flood_stage_kernel
from CMF_GPU.utils.Aggregator import update_stats_aggregator

def do_one_substep(
    runtime_flags: dict,
    params: dict,
    states: dict,
    aggregator: dict,
    runoff: torch.Tensor,
    dT: float,
    num_sub_steps: int,
    BLOCK_SIZE: int = 1024
):
    """
    Perform one sub-step of the flood simulation using Triton kernels.
    """
    # Determine tensor precision
    num_catchments = len(states["river_outflow"])
    states["river_inflow"].zero_()
    states["flood_inflow"].zero_()
    states["outgoing_storage"].zero_()
    grid = lambda meta: (cdiv(num_catchments, meta['BLOCK_SIZE']),)
    # ---------------------------
    # 1) Compute river and flood outflows
    # ---------------------------
    # atomic
    compute_outflow_kernel[grid](
        is_river_mouth_ptr=params["is_river_mouth"],
        downstream_idx_ptr=params["downstream_idx"],
        river_outflow_ptr=states["river_outflow"],
        flood_outflow_ptr=states["flood_outflow"],
        river_manning_ptr=params["river_manning"],
        flood_manning_ptr=params["flood_manning"],
        river_depth_ptr=states["river_depth"],
        river_width_ptr=params["river_width"],
        river_length_ptr=params["river_length"],
        river_elevation_ptr=params["river_elevation"],
        river_storage_ptr=states["river_storage"],
        catchment_elevation_ptr=params["catchment_elevation"],
        downstream_distance_ptr=params["downstream_distance"],
        flood_depth_ptr=states["flood_depth"],
        flood_storage_ptr=states["flood_storage"],
        river_cross_section_depth_prev_ptr=states["river_cross_section_depth"],
        flood_cross_section_depth_prev_ptr=states["flood_cross_section_depth"],
        flood_cross_section_area_prev_ptr=states["flood_cross_section_area"],
        outgoing_storage_ptr=states["outgoing_storage"],  
        water_surface_elevation_ptr=states["water_surface_elevation"],
        gravity=params["gravity"],
        time_step=dT,
        num_catchments=num_catchments,
        BLOCK_SIZE=BLOCK_SIZE
    )
    # ---------------------------
    # 2) Accumulate inflows
    # ---------------------------
    # atomic
    compute_inflow_kernel[grid](
        is_river_mouth_ptr=params["is_river_mouth"],
        downstream_idx_ptr=params["downstream_idx"],
        river_outflow_ptr=states["river_outflow"],
        flood_outflow_ptr=states["flood_outflow"],
        total_storage_ptr=states["total_storage"],
        outgoing_storage_ptr=states["outgoing_storage"],
        river_inflow_ptr=states["river_inflow"],
        flood_inflow_ptr=states["flood_inflow"],
        limit_rate_ptr=states["limit_rate"],
        num_catchments=num_catchments,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # ---------------------------
    # 3) Compute flood stage and update water levels
    # ---------------------------
    
    compute_flood_stage_kernel[grid](
        river_inflow_ptr=states["river_inflow"],
        flood_inflow_ptr=states["flood_inflow"],
        river_outflow_ptr=states["river_outflow"],
        flood_outflow_ptr=states["flood_outflow"],
        runoff_ptr=runoff,
        time_step=dT,
        total_storage_ptr=states["total_storage"],
        river_storage_ptr=states["river_storage"],
        flood_storage_ptr=states["flood_storage"],
        river_depth_ptr=states["river_depth"],
        flood_depth_ptr=states["flood_depth"],
        flood_fraction_ptr=states["flood_fraction"],
        flood_area_ptr=states["flood_area"],
        river_max_storage_ptr=params["river_max_storage"],
        river_area_ptr=params["river_area"],
        max_flood_area_ptr=params["max_flood_area"],
        total_storage_table_ptr=params["total_storage_table"],
        flood_depth_table_ptr=params["flood_depth_table"],
        total_width_table_ptr=params["total_width_table"],
        flood_gradient_table_ptr=params["flood_gradient_table"],
        catchment_area_ptr=params["catchment_area"],
        river_width_ptr=params["river_width"],
        river_length_ptr=params["river_length"],
        num_catchments=num_catchments,
        num_flood_levels=10,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # ---------------------------
    # 4) Update statistics aggregator
    # ---------------------------
    update_stats_aggregator[grid](
        river_outflow=states["river_outflow"],
        river_outflow_mean=aggregator["river_outflow_mean"],
        river_outflow_max=aggregator["river_outflow_max"],
        river_outflow_min=aggregator["river_outflow_min"],
        num_sub_steps=num_sub_steps,
        num_catchments=num_catchments,
        BLOCK_SIZE=BLOCK_SIZE
    )
    pass


def advance_step(runtime_flags, params, states, runoff, dT_def, BLOCK_SIZE=1024):
    """
    Advance the step using either adaptive or fixed time step.
    """
    use_adapt = runtime_flags.get("enable_adaptive_time_step", False)
    if use_adapt:
        dT, num_sub_steps = compute_adaptive_time_step(
            states["river_depth"],
            params["downstream_distance"],
            runtime_flags["time_step"],
            params["adaptation_factor"],
            params["gravity"],
        )
        print(f"Adaptive time step: {dT:.4f}, Number of sub-steps: {num_sub_steps}")
        num_sub_steps_gpu = torch.tensor(num_sub_steps, device=params["is_river_mouth"].device)
    else:
        dT = dT_def
        num_sub_steps = runtime_flags["default_sub_iters"]
        num_sub_steps_gpu = torch.tensor(runtime_flags["default_sub_iters"], device=params["is_river_mouth"].device)
    
    num_catchments = len(states["river_outflow"])
    dtype = torch.float32 if runtime_flags["precision"] == "float32" else torch.float64
    aggregator = {"river_outflow_mean": torch.zeros(num_catchments, dtype=dtype, device=params["is_river_mouth"].device),
                  "river_outflow_max": torch.zeros(num_catchments, dtype=dtype, device=params["is_river_mouth"].device),
                  "river_outflow_min": torch.zeros(num_catchments, dtype=dtype, device=params["is_river_mouth"].device)}
    for _ in range(num_sub_steps):
        do_one_substep(
            runtime_flags,
            params,
            states,
            aggregator,
            runoff,
            dT,
            num_sub_steps_gpu,
            BLOCK_SIZE
        )
    return aggregator



if __name__ == "__main__":
    # Example usage
    config = {
        "runoff_timestep": 3600,
        "iters_per_runoff_step": 1,
        "default_sub_iters": 10,
        "enable_adaptive_dt": True,
        "precision": "float32"  # Can also be "float64"
    }

    param = {
        "is_river_mouth": torch.tensor([False, False, True], dtype=torch.int32, device="cuda:0"),
        "downstream_idx": torch.tensor([1, 2, 2], dtype=torch.int32, device="cuda:0"),
        "river_width": torch.tensor([20.0, 30.0, 40.0], dtype=torch.float32, device="cuda:0"),
        "river_length": torch.tensor([1000.0, 1000.0, 1000.0], dtype=torch.float32, device="cuda:0"),
        "river_elevation": torch.tensor([100.0, 90.0, 80.0], dtype=torch.float32, device="cuda:0"),
        "catchment_elevation": torch.tensor([95.0, 85.0, 75.0], dtype=torch.float32, device="cuda:0"),
        "downstream_distance": torch.tensor([1000.0, 1000.0, 1000.0], dtype=torch.float32, device="cuda:0"),
        "river_manning": torch.tensor([0.03, 0.03, 0.03], dtype=torch.float32, device="cuda:0"),
        "flood_manning": torch.tensor([0.05, 0.05, 0.05], dtype=torch.float32, device="cuda:0"),
        "gravity": 9.81,
        "river_max_storage": torch.tensor([1000.0, 1000.0, 1000.0], dtype=torch.float32, device="cuda:0"),
        "river_area": torch.tensor([2000.0, 3000.0, 4000.0], dtype=torch.float32, device="cuda:0"),
        "max_flood_area": torch.tensor([5000.0, 5000.0, 5000.0], dtype=torch.float32, device="cuda:0"),
        "total_storage_table": torch.ones((3, 10), dtype=torch.float32, device="cuda:0"),
        "flood_depth_table": torch.ones((3, 10), dtype=torch.float32, device="cuda:0"),
        "total_width_table": torch.ones((3, 10), dtype=torch.float32, device="cuda:0"),
        "flood_gradient_table": torch.ones((3, 10), dtype=torch.float32, device="cuda:0"),
        "catchment_area": torch.tensor([1e6, 1e6, 1e6], dtype=torch.float32, device="cuda:0"),
        "adaptation_factor": 0.9,
    }

    # Initial state
    state = {
        "river_outflow": torch.zeros(3, dtype=torch.float32, device="cuda:0"),
        "flood_outflow": torch.zeros(3, dtype=torch.float32, device="cuda:0"),
        "river_depth": torch.ones(3, dtype=torch.float32, device="cuda:0"),
        "flood_depth": torch.zeros(3, dtype=torch.float32, device="cuda:0"),
        "river_storage": torch.ones(3, dtype=torch.float32, device="cuda:0") * 100.0,
        "flood_storage": torch.zeros(3, dtype=torch.float32, device="cuda:0"),
        "total_storage": torch.ones(3, dtype=torch.float32, device="cuda:0") * 100.0,
        "river_cross_section_depth": torch.ones(3, dtype=torch.float32, device="cuda:0"),
        "flood_cross_section_depth": torch.zeros(3, dtype=torch.float32, device="cuda:0"),
        "flood_cross_section_area": torch.zeros(3, dtype=torch.float32, device="cuda:0"),
    }

    # External runoff
    runoff = torch.tensor([10.0, 5.0, 0.0], dtype=torch.float32, device="cuda:0")

    print("=== Step Test (Adaptive ON) ===")
    advance_step(config, param, state, runoff, config["runoff_timestep"])
