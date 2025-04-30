import torch
import numpy as np

def compute_outflow_torch(
    is_river_mouth,                     # bool mask: 1 means river mouth 
    downstream_idx,                     # int downstream index

    # river variables
    river_outflow,                      # in/out river outflow
    river_manning,                      # river Manning coefficient
    river_depth,                        # river depth
    river_width,                        # river width
    river_length,                       # river length
    river_elevation,                    # river bed elevation
    river_storage,                      # river storage

    # flood variables
    flood_outflow,                      # in/out flood outflow
    flood_manning,                      # flood Manning coefficient
    flood_depth,                        # flood depth
    catchment_elevation,                # catchment ground elevation
    downstream_distance,                # distance to downstream unit
    flood_storage,                      # flood storage

    # previous time step variables
    river_cross_section_depth_prev,     # previous river cross-section depth
    flood_cross_section_depth_prev,     # previous flood cross-section depth
    flood_cross_section_area_prev,      # previous flood cross-section area

    # other 
    gravity,                            # scalar gravity acceleration
    time_step                           # scalar time step
):
    # Create output tensor for outgoing storage
    outgoing_storage = torch.zeros_like(river_outflow)

    #----------------------------------------------------------------------
    # (2) Compute current river water surface elevation & downstream water surface elevation
    #----------------------------------------------------------------------
    water_surface_elevation = river_depth + river_elevation
    
    water_surface_elevation_downstream = water_surface_elevation[downstream_idx]
    
    # For river mouth, treat downstream water surface as sea level
    water_surface_elevation_downstream = torch.where(
        is_river_mouth, 
        catchment_elevation, 
        water_surface_elevation_downstream
    )
    
    # (3) Compute maximum water surface elevation
    max_water_surface_elevation = torch.maximum(water_surface_elevation, water_surface_elevation_downstream)
    
    #----------------------------------------------------------------------
    # (4) Longitudinal water surface slope & truncated flood slope
    #----------------------------------------------------------------------
    river_slope = (water_surface_elevation - water_surface_elevation_downstream) / downstream_distance
    flood_slope = torch.clamp(river_slope, min=-0.005, max=0.005)

    #----------------------------------------------------------------------
    # (5) Current river/flood cross-section depth + semi-implicit flow depth
    #----------------------------------------------------------------------
    river_cross_section_depth = max_water_surface_elevation - river_elevation
    river_semi_implicit_flow_depth = torch.sqrt(
        river_cross_section_depth * river_cross_section_depth_prev
    ).clamp(min=1e-6)

    flood_cross_section_depth = torch.clamp(
        max_water_surface_elevation - catchment_elevation,
        min=0.0
    )
    flood_semi_implicit_flow_depth = torch.sqrt(
        flood_cross_section_depth * flood_cross_section_depth_prev
    ).clamp(min=1e-6)

    #----------------------------------------------------------------------
    # (6) Current flood area (approximate) & semi-implicit effective area
    #----------------------------------------------------------------------
    flood_cross_section_area = torch.clamp(
        flood_storage / river_length - flood_depth * river_width,
        min=0.0
    )
    flood_implicit_area = torch.sqrt(
        flood_cross_section_area * flood_cross_section_area_prev
    ).clamp(min=1e-6)

    #----------------------------------------------------------------------
    # (7) Update river outflow
    #----------------------------------------------------------------------
    river_cross_section_area = river_cross_section_depth * river_width
    river_condition = (river_semi_implicit_flow_depth > 1e-5) & (river_cross_section_area > 1e-5)

    # Original river outflow (per unit width)
    unit_river_outflow = river_outflow / river_width

    numerator_river = river_width * (
        unit_river_outflow + gravity * time_step 
        * river_semi_implicit_flow_depth * river_slope
    )
    
    # Calculate power using torch.pow
    denominator_river = 1.0 + gravity * time_step * (river_manning * river_manning) * torch.abs(unit_river_outflow) \
                      * torch.pow(river_semi_implicit_flow_depth, -7.0/3.0)
                      
    updated_river_outflow = numerator_river / denominator_river
    updated_river_outflow = torch.where(river_condition, updated_river_outflow, torch.zeros_like(updated_river_outflow))

    #----------------------------------------------------------------------
    # (8) Update flood outflow
    #----------------------------------------------------------------------
    flood_condition = (flood_semi_implicit_flow_depth > 1e-5) & (flood_implicit_area > 1e-5)

    numerator_flood = flood_outflow + gravity * time_step * flood_implicit_area * flood_slope
    
    # Calculate power using torch.pow
    denominator_flood = 1.0 + gravity * time_step * (flood_manning * flood_manning) * torch.abs(flood_outflow) \
                      * torch.pow(flood_semi_implicit_flow_depth, -4.0/3.0) / flood_implicit_area
                      
    updated_flood_outflow = numerator_flood / denominator_flood
    updated_flood_outflow = torch.where(flood_condition, updated_flood_outflow, torch.zeros_like(updated_flood_outflow))

    #----------------------------------------------------------------------
    # (9) Prevent flood and river from flowing in opposite directions
    #----------------------------------------------------------------------
    opposite_direction = (updated_river_outflow * updated_flood_outflow) < 0.0
    updated_flood_outflow = torch.where(opposite_direction, torch.zeros_like(updated_flood_outflow), updated_flood_outflow)
    non_river_mouth = ~is_river_mouth
    is_negative_flow = (updated_river_outflow < 0.0) & non_river_mouth
    total_negative_flow = torch.where(is_negative_flow, 
                                     (-updated_river_outflow - updated_flood_outflow) * time_step,
                                     torch.ones_like(updated_river_outflow))
    
    limit_rate = torch.clamp(
        torch.where(
            is_negative_flow,
            0.05 * (river_storage + flood_storage) / (total_negative_flow),
            torch.ones_like(updated_river_outflow)
        ),
        max=1.0
    )
    
    updated_river_outflow = torch.where(is_negative_flow, 
                                       updated_river_outflow * limit_rate, 
                                       updated_river_outflow)
    updated_flood_outflow = torch.where(is_negative_flow,
                                       updated_flood_outflow * limit_rate,
                                       updated_flood_outflow)

    #----------------------------------------------------------------------
    # (11) Compute outgoing storage
    #----------------------------------------------------------------------
    # Split positive/negative flow
    pos_flow = torch.clamp(updated_river_outflow, min=0.0) + torch.clamp(updated_flood_outflow, min=0.0)
    neg_flow = torch.clamp(updated_river_outflow, max=0.0) + torch.clamp(updated_flood_outflow, max=0.0)

    outgoing_storage.add_(pos_flow * time_step)
    
    # Add negative flow to downstream catchments (only for non-river mouths)

    ds_idx = downstream_idx[non_river_mouth]
    neg_values = -neg_flow[non_river_mouth] * time_step
    outgoing_storage.scatter_add_(0, ds_idx, neg_values)

    return updated_river_outflow, updated_flood_outflow, outgoing_storage, water_surface_elevation, river_cross_section_depth, flood_cross_section_depth, flood_cross_section_area


def compute_inflow_torch(
    is_river_mouth,                # River mouth mask
    downstream_idx,                # Downstream indices
    river_outflow,                 # River outflow (in/out)
    flood_outflow,                 # Flood outflow (in/out)
    total_storage,                 # Total storage
    outgoing_storage,               # Outgoing storage
):
    river_inflow = torch.zeros_like(river_outflow)
    flood_inflow = torch.zeros_like(flood_outflow)

    # Apply limits based on available storage
    limit_rate = torch.where(
        outgoing_storage > 1e-8,
        torch.clamp(total_storage / outgoing_storage, max=1.0),
        torch.ones_like(outgoing_storage)
    )
    limit_rate_downstream = limit_rate[downstream_idx]
    non_river_mouth = ~is_river_mouth

    # Apply limits
    river_outflow_limited = torch.where(river_outflow >= 0.0, river_outflow * limit_rate, river_outflow * limit_rate_downstream)
    flood_outflow_limited = torch.where(flood_outflow >= 0.0, flood_outflow * limit_rate, flood_outflow * limit_rate_downstream)
    
    # Accumulate inflows - only for non-river mouth segments
    ds_idx = downstream_idx[non_river_mouth]
    river_out_update = river_outflow_limited[non_river_mouth]
    flood_out_update = flood_outflow_limited[non_river_mouth]
    

    river_inflow.scatter_add_(0, ds_idx, river_out_update)
    flood_inflow.scatter_add_(0, ds_idx, flood_out_update)

    return river_outflow_limited, flood_outflow_limited, river_inflow, flood_inflow, limit_rate