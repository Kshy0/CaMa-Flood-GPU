import torch

def compute_bifurcation_outflow(
    catchment_idx,
    downstream_idx,
    bifurcation_manning,
    bifurcation_outflow,
    bifurcation_width, #[num_paths, 5]
    bifurcation_length,
    bifurcation_elevation,
    water_surface_elevation,
    bifurcation_cross_section_depth_prev,
    total_storage,
    outgoing_storage,
    gravity,
    time_step,
):
    bifurcation_water_surface_elevation = water_surface_elevation[catchment_idx]
    bifurcation_water_surface_elevation_downstream = water_surface_elevation[downstream_idx]
    # Calculate water surface elevation for bifurcation paths
    max_bifurcation_water_surface_elevation = torch.maximum(bifurcation_water_surface_elevation, bifurcation_water_surface_elevation_downstream)
    
    # Calculate bifurcation cross-section depth
    bifurcation_cross_section_depth = (max_bifurcation_water_surface_elevation[:, None] - bifurcation_elevation).clamp(min=0.0)
    
    # Calculate semi-implicit flow depth for bifurcation
    bifurcation_semi_implicit_flow_depth = torch.maximum(torch.sqrt(
        bifurcation_cross_section_depth * bifurcation_cross_section_depth_prev
    ), torch.sqrt(bifurcation_cross_section_depth * 0.01))
    
    
    # Bifurcation slope (clamped similarly to flood slope)
    bifurcation_slope = (bifurcation_water_surface_elevation - bifurcation_water_surface_elevation_downstream) / bifurcation_length
    bifurcation_slope = torch.clamp(bifurcation_slope, min=-0.005, max=0.005)
    
    # Calculate updated bifurcation outflow
    bifurcation_condition = (bifurcation_semi_implicit_flow_depth > 1e-5)
    
    # Original bifurcation outflow (per unit width)
    unit_bifurcation_outflow = bifurcation_outflow / bifurcation_width
    
    numerator = bifurcation_width * (
        unit_bifurcation_outflow + gravity * time_step 
        * bifurcation_semi_implicit_flow_depth * bifurcation_slope[:, None]
    )
    
    denominator = 1.0 + gravity * time_step * (bifurcation_manning * bifurcation_manning) * torch.abs(unit_bifurcation_outflow) \
                    * torch.pow(bifurcation_semi_implicit_flow_depth, -7.0/3.0)
                    
    updated_bifurcation_outflow = numerator / denominator
    updated_bifurcation_outflow = torch.where(bifurcation_condition, updated_bifurcation_outflow, torch.zeros_like(updated_bifurcation_outflow))
    bifurcation_outflow_sum = updated_bifurcation_outflow.sum(dim=1)

    # Storage change limiter (to prevent sudden increase of upstream water level)
    limit_rate = torch.where(
        bifurcation_outflow_sum != 0,
        torch.clamp(0.05 * torch.minimum(total_storage[catchment_idx], total_storage[downstream_idx]) / (torch.abs(bifurcation_outflow_sum) * time_step), max=1.0),
        torch.ones_like(bifurcation_outflow_sum)
    )

    updated_bifurcation_outflow *= limit_rate[:, None]
    bifurcation_outflow_sum *= limit_rate
    

    pos_flow = torch.clamp(bifurcation_outflow_sum, min=0.0)
    neg_flow = torch.clamp(bifurcation_outflow_sum, max=0.0)
    outgoing_storage.scatter_add_(0, catchment_idx, pos_flow * time_step)
    outgoing_storage.scatter_add_(0, downstream_idx, -neg_flow * time_step)

    return updated_bifurcation_outflow, bifurcation_cross_section_depth, outgoing_storage
