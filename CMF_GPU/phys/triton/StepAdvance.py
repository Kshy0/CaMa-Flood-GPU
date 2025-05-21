import torch
from triton import cdiv
from CMF_GPU.phys.AdaptTime import compute_adaptive_time_step
from CMF_GPU.phys.triton.Outflow import compute_outflow_kernel, compute_inflow_kernel
from CMF_GPU.phys.triton.Bifurcation import compute_bifurcation_outflow_kernel
from CMF_GPU.phys.triton.Storage import compute_flood_stage_kernel, compute_flood_stage_log_kernel
from CMF_GPU.phys.triton.Reservoir import compute_reservoir_outflow

def do_one_substep(
    runtime_flags: dict,
    params: dict,
    states: dict,
    runoff: torch.Tensor,
    dT: float,
    update_statistics: callable,
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
        is_reservoir_ptr=params["is_reservoir"],
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
        river_cross_section_depth_ptr=states["river_cross_section_depth"],
        flood_cross_section_depth_ptr=states["flood_cross_section_depth"],
        flood_cross_section_area_ptr=states["flood_cross_section_area"],
        total_storage_ptr=states["total_storage"],   
        outgoing_storage_ptr=states["outgoing_storage"],  
        water_surface_elevation_ptr=states["water_surface_elevation"],
        gravity=params["gravity"],
        time_step=dT,
        num_catchments=params["num_catchments"],
        BLOCK_SIZE=BLOCK_SIZE
    )

    if "bifurcation" in runtime_flags["modules"]:
        bifurcation_grid = lambda meta: (cdiv(params["num_bifurcation_paths"], meta['BLOCK_SIZE']),)
        compute_bifurcation_outflow_kernel[bifurcation_grid](
            catchment_idx_ptr=params["bifurcation_catchment_idx"],
            downstream_idx_ptr=params["bifurcation_downstream_idx"],
            bifurcation_manning_ptr=params["bifurcation_manning"],
            bifurcation_outflow_ptr=states["bifurcation_outflow"],
            bifurcation_width_ptr=params["bifurcation_width"],
            bifurcation_length_ptr=params["bifurcation_length"],
            bifurcation_elevation_ptr=params["bifurcation_elevation"],
            bifurcation_cross_section_depth_ptr=states["bifurcation_cross_section_depth"],
            water_surface_elevation_ptr=states["water_surface_elevation"],
            total_storage_ptr=states["total_storage"],
            outgoing_storage_ptr=states["outgoing_storage"],
            gravity=params["gravity"],
            time_step=dT,
            num_paths=params["num_bifurcation_paths"],
            num_bifurcation_levels=params["num_bifurcation_levels"],
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    if "reservoir" in runtime_flags["modules"]:
        compute_reservoir_outflow(
            reservoir_catchment_idx_ptr=params["reservoir_catchment_idx"],
            reservoir_upstream_idx_ptr=params["reservoir_upstream_idx"],
            river_inflow_ptr=states["river_inflow"],
            river_outflow_ptr=states["river_outflow"],
            river_storage_ptr=states["river_storage"],
            flood_inflow_ptr=states["flood_inflow"],
            flood_storage_ptr=states["flood_storage"],
            conservation_volume_ptr=params["conservation_volume"],
            emergency_volume_ptr=params["emergency_volume"],
            adjustment_volume_ptr=params["adjustment_volume"],
            normal_outflow_ptr=params["normal_outflow"],
            adjustment_outflow_ptr=params["adjustment_outflow"],
            flood_control_outflow_ptr=params["flood_control_outflow"],
            river_manning_ptr=params["river_manning"],
            river_depth_ptr=params["river_depth"],
            river_width_ptr=params["river_width"],
            river_length_ptr=params["river_length"],
            flood_manning_ptr=params["flood_manning"],
            flood_depth_ptr=params["flood_depth"],
            catchment_elevation_ptr=params["catchment_elevation"],
            downstream_distance_ptr=params["downstream_distance"],
            runoff_ptr=runoff,
            total_storage_ptr=params["total_storage_ptr"],
            outgoing_storage_ptr=params["outgoing_storage_ptr"],
            time_step=dT,
            num_reservoirs=params["num_reservoirs"],
            num_upstreams=params["num_upstreams"],
            min_slope=params["min_slope"],
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
    update_statistics(
        params,
        states,
        current_step=current_step,
        num_sub_steps=num_sub_steps,
        BLOCK_SIZE=BLOCK_SIZE
    )


def advance_step(runtime_flags, params, states, runoff, dT_def, logger, update_statistics, BLOCK_SIZE=1024):
    """
    Advance the step using either adaptive or fixed time step.
    """

    if "adaptive_time_step" in runtime_flags["modules"]:
        dT, num_sub_steps = compute_adaptive_time_step(
            params["is_reservoir"],
            params["downstream_idx"],
            states["river_depth"],
            params["downstream_distance"],
            states["min_time_step"],
            runtime_flags["time_step"],
            params["adaptation_factor"],
            params["gravity"],
            params["num_catchments"],
            BLOCK_SIZE
        )
        print(f"Adaptive time step: {dT:.4f}, Number of sub-steps: {num_sub_steps}")
    else:
        dT = dT_def
        num_sub_steps = runtime_flags["default_num_sub_steps"]
    logger.set_time_step(dT, num_sub_steps, states)
    for current_step in range(num_sub_steps):
        do_one_substep(
            runtime_flags,
            params,
            states,
            runoff,
            dT,
            update_statistics,
            current_step,
            num_sub_steps,
            BLOCK_SIZE
        )
    logger.write_step(states)