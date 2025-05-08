import torch
from triton import cdiv
from CMF_GPU.phys.AdaptTime import compute_adaptive_time_step
from CMF_GPU.phys.triton.Outflow import compute_outflow_kernel, compute_inflow_kernel
from CMF_GPU.phys.triton.Storage import compute_flood_stage_kernel, compute_flood_stage_log_kernel
from CMF_GPU.utils.Aggregator import update_stats_aggregator

def do_one_substep(
    runtime_flags: dict,
    params: dict,
    states: dict,
    runoff: torch.Tensor,
    dT: float,
    current_step: int,
    num_sub_steps: int,
    BLOCK_SIZE: int = 1024
):
    """
    Perform one sub-step of the flood simulation using Triton kernels.
    """
    grid = lambda meta: (cdiv(params["num_catchments"], meta['BLOCK_SIZE']),)
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
        total_storage_ptr=states["total_storage"],   
        outgoing_storage_ptr=states["outgoing_storage"],  
        water_surface_elevation_ptr=states["water_surface_elevation"],
        gravity=params["gravity"],
        time_step=dT,
        num_catchments=params["num_catchments"],
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
        num_catchments=params["num_catchments"],
        BLOCK_SIZE=BLOCK_SIZE
    )

    # ---------------------------
    # 3) Compute flood stage and update water levels
    # ---------------------------
    if "log" in runtime_flags["modules"]:
        compute_flood_stage_log_kernel[grid](
            river_inflow_ptr=states["river_inflow"],
            flood_inflow_ptr=states["flood_inflow"],
            river_outflow_ptr=states["river_outflow"],
            flood_outflow_ptr=states["flood_outflow"],
            runoff_ptr=runoff,
            time_step=dT,
            river_storage_ptr=states["river_storage"],
            flood_storage_ptr=states["flood_storage"],
            outgoing_storage_ptr=states["outgoing_storage"],
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
            total_storage_pre_sum_ptr=states["total_storage_pre_sum"],
            total_storage_next_sum_ptr=states["total_storage_next_sum"],
            total_storage_new_ptr=states["total_storage_new"],
            total_inflow_ptr=states["total_inflow"],
            total_outflow_ptr=states["total_outflow"],
            total_storage_stage_new_ptr=states["total_storage_stage_new"],
            total_river_storage_ptr=states["total_river_storage"],
            total_flood_storage_ptr=states["total_flood_storage"],
            total_flood_area_ptr=states["total_flood_area"],
            total_inflow_error_ptr=states["total_inflow_error"],
            total_stage_error_ptr=states["total_stage_error"],
            num_catchments=params["num_catchments"],
            current_step=current_step,
            num_flood_levels=params["num_flood_levels"],
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        compute_flood_stage_kernel[grid](
            river_inflow_ptr=states["river_inflow"],
            flood_inflow_ptr=states["flood_inflow"],
            river_outflow_ptr=states["river_outflow"],
            flood_outflow_ptr=states["flood_outflow"],
            runoff_ptr=runoff,
            time_step=dT,
            river_storage_ptr=states["river_storage"],
            flood_storage_ptr=states["flood_storage"],
            outgoing_storage_ptr=states["outgoing_storage"],
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
            num_catchments=params["num_catchments"],
            num_flood_levels=params["num_flood_levels"],
            BLOCK_SIZE=BLOCK_SIZE
        )

    # ---------------------------
    # 4) Update statistics aggregator
    # ---------------------------
    update_stats_aggregator[grid](
        river_outflow_ptr=states["river_outflow"],
        river_outflow_mean_ptr=states["river_outflow_mean"],
        river_outflow_max_ptr=states["river_outflow_max"],
        river_outflow_min_ptr=states["river_outflow_min"],
        current_step=current_step,
        num_sub_steps=num_sub_steps,
        num_catchments=params["num_catchments"],
        BLOCK_SIZE=BLOCK_SIZE
    )
    pass


def advance_step(runtime_flags, params, states, runoff, dT_def, logger, BLOCK_SIZE=1024):
    """
    Advance the step using either adaptive or fixed time step.
    """

    if "adaptive_time_step" in runtime_flags["modules"]:
        dT, num_sub_steps = compute_adaptive_time_step(
            states["river_depth"],
            params["downstream_distance"],
            runtime_flags["time_step"],
            params["adaptation_factor"],
            params["gravity"],
        )
        print(f"Adaptive time step: {dT:.4f}, Number of sub-steps: {num_sub_steps}")
    else:
        dT = dT_def
        num_sub_steps = runtime_flags["default_sub_iters"]
    logger.set_time_step(dT, num_sub_steps, states)
    for current_step in range(num_sub_steps):
        do_one_substep(
            runtime_flags,
            params,
            states,
            runoff,
            dT,
            current_step,
            num_sub_steps,
            BLOCK_SIZE
        )
    logger.write_step(states)