# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

import triton
import triton.language as tl


@triton.jit
def compute_outflow_kernel(
    is_river_mouth_ptr,                     # *bool mask: 1 means river mouth 
    downstream_idx_ptr,                     # *i32 downstream index

    # river variables
    river_inflow_ptr,                       # *f32 river inflow (turn to zero)
    river_outflow_ptr,                      # *f32 in/out river outflow
    river_manning_ptr,                      # *f32 river Manning coefficient
    river_depth_ptr,                        # *f32 river depth
    river_width_ptr,                        # *f32 river width
    river_length_ptr,                       # *f32 river length
    river_height_ptr,                       # *f32 river bank height
    river_storage_ptr,                      # *f32 river storage

    # flood variables
    flood_inflow_ptr,                       # *f32 flood inflow (turn to zero)
    flood_outflow_ptr,                      # *f32 in/out flood outflow
    flood_manning_ptr,                      # *f32 flood Manning coefficient
    flood_depth_ptr,                        # *f32 flood depth
    protected_depth_ptr,                    # *f32 protected depth
    catchment_elevation_ptr,                # *f32 catchment ground elevation
    downstream_distance_ptr,                # *f32 distance to downstream unit
    flood_storage_ptr,                      # *f32 flood storage
    protected_storage_ptr,                  # *f32 protected storage

    # previous time step variables
    river_cross_section_depth_ptr,     # *f32 previous river cross-section depth
    flood_cross_section_depth_ptr,     # *f32 previous flood cross-section depth
    flood_cross_section_area_ptr,      # *f32 previous flood cross-section area

    # other 
    global_bifurcation_outflow_ptr,          # *f32 global bifurcation outflow (turn to zero)
    total_storage_ptr,
    outgoing_storage_ptr,                   # *f32 output for storage (fused part)
    water_surface_elevation_ptr,            # *f32 water surface elevation
    protected_water_surface_elevation_ptr,  # *f32 protected water surface elevation
    gravity,                                # f32 scalar gravity acceleration
    time_step,                              # f32 scalar time step
    num_catchments: tl.constexpr,           # total number of elements
    BLOCK_SIZE: tl.constexpr                # block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_catchments

    #----------------------------------------------------------------------
    # (1) Load previous time step input variables
    #----------------------------------------------------------------------
    is_river_mouth = tl.load(is_river_mouth_ptr + offs, mask=mask, other=0)
    downstream_idx = tl.load(downstream_idx_ptr + offs, mask=mask, other=0)

    # river variables
    river_outflow = tl.load(river_outflow_ptr + offs, mask=mask, other=0.0)
    river_manning = tl.load(river_manning_ptr + offs, mask=mask, other=1.0)
    river_depth = tl.load(river_depth_ptr + offs, mask=mask, other=0.0)
    river_width = tl.load(river_width_ptr + offs, mask=mask, other=1.0)
    river_length = tl.load(river_length_ptr + offs, mask=mask, other=1.0)
    river_height = tl.load(river_height_ptr + offs, mask=mask, other=0.0)
    river_storage = tl.load(river_storage_ptr + offs, mask=mask, other=0.0)

    # flood variables
    flood_outflow = tl.load(flood_outflow_ptr + offs, mask=mask, other=0.0)
    flood_manning = tl.load(flood_manning_ptr + offs, mask=mask, other=1.0)
    flood_depth = tl.load(flood_depth_ptr + offs, mask=mask, other=0.0)
    protected_depth = tl.load(protected_depth_ptr + offs, mask=mask, other=0.0)
    catchment_elevation = tl.load(catchment_elevation_ptr + offs, mask=mask, other=0.0)
    downstream_distance = tl.load(downstream_distance_ptr + offs, mask=mask, other=1.0)
    flood_storage = tl.load(flood_storage_ptr + offs, mask=mask, other=0.0)
    protected_storage = tl.load(protected_storage_ptr + offs, mask=mask, other=0.0)

    # cross section variables
    river_cross_section_depth = tl.load(river_cross_section_depth_ptr + offs, mask=mask, other=0.0)
    flood_cross_section_depth = tl.load(flood_cross_section_depth_ptr + offs, mask=mask, other=0.0)
    flood_cross_section_area = tl.load(flood_cross_section_area_ptr + offs, mask=mask, other=0.0)
    #----------------------------------------------------------------------
    # (2) Compute current river water surface elevation & downstream water surface elevation
    #----------------------------------------------------------------------
    river_elevation = catchment_elevation - river_height
    water_surface_elevation = river_depth + river_elevation
    protected_water_surface_elevation = tl.minimum(catchment_elevation + protected_depth, water_surface_elevation)
    total_storage = river_storage + flood_storage + protected_storage
    # Downstream water surface elevation
    river_depth_downstream = tl.load(river_depth_ptr + downstream_idx, mask=mask, other=0.0)
    river_height_downstream = tl.load(river_height_ptr + downstream_idx, mask=mask, other=0.0)
    catchment_elevation_downstream = tl.load(catchment_elevation_ptr + downstream_idx, mask=mask, other=0.0)
    river_elevation_downstream = catchment_elevation_downstream - river_height_downstream
    water_surface_elevation_downstream = river_depth_downstream + river_elevation_downstream
    
    # (3) Compute maximum water surface elevation
    max_water_surface_elevation = tl.maximum(water_surface_elevation, water_surface_elevation_downstream)
    
    # For river mouth, treat downstream water surface as sea level or fixed boundary
    water_surface_elevation_downstream = tl.where(is_river_mouth, catchment_elevation, water_surface_elevation_downstream)
    
    #----------------------------------------------------------------------
    # (4) Longitudinal water surface slope & truncated flood slope
    #----------------------------------------------------------------------
    river_slope = (water_surface_elevation - water_surface_elevation_downstream) / downstream_distance
    flood_slope = tl.clamp(river_slope, -0.005, 0.005)

    #----------------------------------------------------------------------
    # (5) Current river/flood cross-section depth + semi-implicit flow depth
    #----------------------------------------------------------------------
    updated_river_cross_section_depth = max_water_surface_elevation - river_elevation
    river_semi_implicit_flow_depth = tl.maximum(tl.sqrt(
        updated_river_cross_section_depth * river_cross_section_depth
    ), 1e-6)

    updated_flood_cross_section_depth = tl.maximum(
        max_water_surface_elevation - catchment_elevation,
        0.0
    )
    flood_semi_implicit_flow_depth = tl.maximum(
        tl.sqrt(updated_flood_cross_section_depth * flood_cross_section_depth), 
        1e-6
    )

    #----------------------------------------------------------------------
    # (6) Current flood area (approximate) & semi-implicit effective area
    #----------------------------------------------------------------------
    updated_flood_cross_section_area = tl.maximum(
        flood_storage / river_length - flood_depth * river_width,
        0.0
    )
    flood_implicit_area = tl.maximum(tl.sqrt(
        updated_flood_cross_section_area * tl.maximum(flood_cross_section_area, 1e-6)
    ), 1e-6)

    #----------------------------------------------------------------------
    # (7) Update river outflow
    #----------------------------------------------------------------------
    river_cross_section_area = updated_river_cross_section_depth * river_width
    river_condition = (river_semi_implicit_flow_depth > 1e-5) & (river_cross_section_area > 1e-5)

    # Original river outflow (per unit width)
    unit_river_outflow = river_outflow / river_width

    numerator_river = river_width * (
        unit_river_outflow + gravity * time_step 
        * river_semi_implicit_flow_depth * river_slope
    )
    
    # Use exp(log()) for power calculation
    denominator_river = 1.0 + gravity * time_step * (river_manning * river_manning) * tl.abs(unit_river_outflow) \
                      * tl.exp((-7.0/3.0) * tl.log(river_semi_implicit_flow_depth))

    updated_river_outflow = numerator_river / denominator_river
    updated_river_outflow = tl.where(river_condition, updated_river_outflow, 0.0)

    #----------------------------------------------------------------------
    # (8) Update flood outflow
    #----------------------------------------------------------------------
    flood_condition = (flood_semi_implicit_flow_depth > 1e-5) & (updated_flood_cross_section_area > 1e-5)

    numerator_flood = flood_outflow + gravity * time_step * flood_implicit_area * flood_slope
    
    # Use exp(log()) for power calculation
    denominator_flood = 1.0 + gravity * time_step * (flood_manning * flood_manning) * tl.abs(flood_outflow) \
                      * tl.exp((-4.0/3.0) * tl.log(flood_semi_implicit_flow_depth)) / flood_implicit_area
                      
    updated_flood_outflow = numerator_flood / denominator_flood
    updated_flood_outflow = tl.where(flood_condition, updated_flood_outflow, 0.0)

    #----------------------------------------------------------------------
    # (9) Prevent flood and river from flowing in opposite directions
    #----------------------------------------------------------------------
    opposite_direction = (updated_river_outflow * updated_flood_outflow) < 0.0
    updated_flood_outflow = tl.where(opposite_direction, 0.0, updated_flood_outflow)
    is_negative_flow = (updated_river_outflow < 0.0) & ~is_river_mouth
    total_negative_flow = tl.where(is_negative_flow, 
                                   (-updated_river_outflow - updated_flood_outflow) * time_step, 
                                   1.0)
    limit_rate = tl.minimum(tl.where(is_negative_flow,
                   0.05 * total_storage / total_negative_flow,
                   1.0), 1.0)
    updated_river_outflow = tl.where(is_negative_flow, updated_river_outflow * limit_rate, updated_river_outflow)
    updated_flood_outflow = tl.where(is_negative_flow, updated_flood_outflow * limit_rate, updated_flood_outflow)

    #----------------------------------------------------------------------
    # (10) Store results - in-place update
    #----------------------------------------------------------------------
    tl.store(river_outflow_ptr + offs, updated_river_outflow, mask=mask)
    tl.store(flood_outflow_ptr + offs, updated_flood_outflow, mask=mask)
    tl.store(water_surface_elevation_ptr + offs, water_surface_elevation, mask=mask)
    tl.store(protected_water_surface_elevation_ptr + offs, protected_water_surface_elevation, mask=mask)
    tl.store(river_cross_section_depth_ptr + offs, updated_river_cross_section_depth, mask=mask)
    tl.store(flood_cross_section_depth_ptr + offs, updated_flood_cross_section_depth, mask=mask)
    tl.store(flood_cross_section_area_ptr + offs, updated_flood_cross_section_area, mask=mask)
    tl.store(total_storage_ptr + offs, total_storage, mask=mask)
    
    tl.store(river_inflow_ptr + offs, 0.0, mask=mask)
    tl.store(flood_inflow_ptr + offs, 0.0, mask=mask)
    tl.store(global_bifurcation_outflow_ptr + offs, 0.0, mask=mask)

    #----------------------------------------------------------------------
    # (11) Fused outgoing storage computation (was compute_outgoing_storage_kernel)
    #----------------------------------------------------------------------
    # Split positive/negative flow
    pos = tl.maximum(updated_river_outflow, 0.0) + tl.maximum(updated_flood_outflow, 0.0)
    neg = tl.minimum(updated_river_outflow, 0.0) + tl.minimum(updated_flood_outflow, 0.0)

    tl.atomic_add(outgoing_storage_ptr + offs, pos * time_step, mask=mask)

    # Scatter-add negative flow to downstream
    # Only non-river-mouth applies negative flow
    to_add    = tl.where(~is_river_mouth, -neg * time_step, 0.0)
    tl.atomic_add(outgoing_storage_ptr + downstream_idx, to_add, mask=mask)
    


@triton.jit
def compute_inflow_kernel(
    is_river_mouth_ptr,            # *i32: River mouth mask
    downstream_idx_ptr,            # *i32: Downstream indices
    river_outflow_ptr,             # *f32: River outflow (in/out)
    flood_outflow_ptr,             # *f32: Flood outflow (in/out)
    total_storage_ptr,             # *f32: Total storage
    outgoing_storage_ptr,          # *f32: Outgoing storage
    river_inflow_ptr,              # *f32: River inflow (output, atomic add)
    flood_inflow_ptr,              # *f32: Flood inflow (output, atomic add)
    limit_rate_ptr,                # *f32: Limit rate (reference)
    num_catchments: tl.constexpr,  # Total number of units
    BLOCK_SIZE: tl.constexpr       # Block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_catchments

    # -------- Load for limiting --------
    river_outflow   = tl.load(river_outflow_ptr      + offs, mask=mask, other=0.0)
    flood_outflow   = tl.load(flood_outflow_ptr      + offs, mask=mask, other=0.0)
    total_storage = tl.load(total_storage_ptr      + offs, mask=mask, other=0.0)
    outgoing_storage = tl.load(outgoing_storage_ptr   + offs, mask=mask, other=0.0)

    # Local limit
    limit_rate = tl.where(outgoing_storage > 1e-8, tl.minimum(total_storage / outgoing_storage, 1.0), 1.0)

    # Downstream limiting
    downstream_idx   = tl.load(downstream_idx_ptr        + offs, mask=mask, other=0)
    outgoing_storage_downstream   = tl.load(outgoing_storage_ptr + downstream_idx,  mask=mask, other=0.0)
    total_storage_downstream   = tl.load(total_storage_ptr     + downstream_idx,  mask=mask, other=0.0)
    limit_rate_downstream = tl.where(outgoing_storage_downstream > 1e-8, tl.minimum(total_storage_downstream / outgoing_storage_downstream, 1.0), 1.0)

    # Apply limits
    updated_river_outflow = tl.where(river_outflow >= 0.0, river_outflow * limit_rate,   river_outflow * limit_rate_downstream)
    updated_flood_outflow = tl.where(flood_outflow >= 0.0, flood_outflow * limit_rate,   flood_outflow * limit_rate_downstream)

    # Write back limited values
    tl.store(river_outflow_ptr + offs, updated_river_outflow, mask=mask)
    tl.store(flood_outflow_ptr + offs, updated_flood_outflow, mask=mask)
    tl.store(limit_rate_ptr + offs, limit_rate, mask=mask)

    # -------- Accumulate inflows --------
    is_river_mouth = tl.load(is_river_mouth_ptr + offs, mask=mask, other=0)
    tl.atomic_add(river_inflow_ptr + downstream_idx, updated_river_outflow, mask=mask&(~is_river_mouth))
    tl.atomic_add(flood_inflow_ptr + downstream_idx, updated_flood_outflow, mask=mask&(~is_river_mouth))
