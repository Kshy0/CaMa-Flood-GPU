import triton
import triton.language as tl

@triton.jit
def compute_outflow_kernel(
    is_river_mouth_ptr,                     # *i32 mask: 1 means river mouth
    downstream_idx_ptr,                     # *i32 downstream index
    river_outflow_ptr,                      # *f32 in/out river outflow
    flood_outflow_ptr,                      # *f32 in/out flood outflow
    river_manning_ptr,                      # *f32 river Manning coefficient
    flood_manning_ptr,                      # *f32 flood Manning coefficient
    river_depth_ptr,                        # *f32 river depth
    river_width_ptr,                        # *f32 river width
    river_length_ptr,                       # *f32 river length
    river_elevation_ptr,                    # *f32 river bed elevation
    catchment_elevation_ptr,                # *f32 catchment ground elevation
    downstream_distance_ptr,                # *f32 distance to downstream unit
    flood_depth_ptr,                        # *f32 flood depth
    flood_storage_ptr,                      # *f32 flood storage
    river_cross_section_depth_prev_ptr,     # *f32 previous river cross-section depth
    flood_cross_section_depth_prev_ptr,     # *f32 previous flood cross-section depth
    flood_cross_section_area_prev_ptr,      # *f32 previous flood cross-section area
    gravity,                                # f32 scalar gravity acceleration
    time_step,                              # f32 scalar time step
    num_catchments: tl.constexpr,           # total number of elements
    BLOCK_SIZE: tl.constexpr                # block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_catchments

    #----------------------------------------------------------------------
    # (1) Load input variables
    #----------------------------------------------------------------------
    is_river_mouth = tl.load(is_river_mouth_ptr + offs, mask=mask, other=0)
    downstream_idx = tl.load(downstream_idx_ptr + offs, mask=mask, other=0)
    river_outflow = tl.load(river_outflow_ptr + offs, mask=mask, other=0.0)
    flood_outflow = tl.load(flood_outflow_ptr + offs, mask=mask, other=0.0)
    river_manning = tl.load(river_manning_ptr + offs, mask=mask, other=1.0)
    flood_manning = tl.load(flood_manning_ptr + offs, mask=mask, other=1.0)
    river_depth = tl.load(river_depth_ptr + offs, mask=mask, other=0.0)
    river_width = tl.load(river_width_ptr + offs, mask=mask, other=1.0)
    river_length = tl.load(river_length_ptr + offs, mask=mask, other=1.0)
    river_elevation = tl.load(river_elevation_ptr + offs, mask=mask, other=0.0)
    catchment_elevation = tl.load(catchment_elevation_ptr + offs, mask=mask, other=0.0)
    downstream_distance = tl.load(downstream_distance_ptr + offs, mask=mask, other=1.0)
    flood_depth = tl.load(flood_depth_ptr + offs, mask=mask, other=0.0)
    flood_storage = tl.load(flood_storage_ptr + offs, mask=mask, other=0.0)
    
    # Load previous time step variables
    river_cross_section_depth_prev = tl.load(river_cross_section_depth_prev_ptr + offs, mask=mask, other=0.0)
    flood_cross_section_depth_prev = tl.load(flood_cross_section_depth_prev_ptr + offs, mask=mask, other=0.0)
    flood_cross_section_area_prev = tl.load(flood_cross_section_area_prev_ptr + offs, mask=mask, other=0.0)

    #----------------------------------------------------------------------
    # (2) Compute current river water surface elevation & downstream water surface elevation
    #----------------------------------------------------------------------
    water_surface_elevation = river_depth + river_elevation
    
    # Downstream water surface elevation
    downstream_river_depth = tl.load(river_depth_ptr + downstream_idx, mask=mask, other=0.0)
    downstream_river_elevation = tl.load(river_elevation_ptr + downstream_idx, mask=mask, other=0.0)
    water_surface_elevation_downstream = downstream_river_depth + downstream_river_elevation
    
    # (3) Compute maximum water surface elevation
    max_water_surface_elevation = tl.maximum(water_surface_elevation, water_surface_elevation_downstream)
    
    # For river mouth, treat downstream water surface as sea level or fixed boundary
    water_surface_elevation_downstream = tl.where(is_river_mouth == 1, catchment_elevation, water_surface_elevation_downstream)
    
    #----------------------------------------------------------------------
    # (4) Longitudinal water surface slope & truncated flood slope
    #----------------------------------------------------------------------
    river_slope = (water_surface_elevation - water_surface_elevation_downstream) / downstream_distance
    flood_slope = tl.maximum(tl.minimum(river_slope, 0.005), -0.005)

    #----------------------------------------------------------------------
    # (5) Current river/flood cross-section depth + semi-implicit flow depth
    #----------------------------------------------------------------------
    river_cross_section_depth = max_water_surface_elevation - river_elevation
    river_semi_implicit_flow_depth = tl.maximum(tl.sqrt(
        river_cross_section_depth * river_cross_section_depth_prev
    ), 1e-6)

    flood_cross_section_depth = tl.maximum(
        max_water_surface_elevation - catchment_elevation,
        0.0
    )
    flood_semi_implicit_flow_depth = tl.maximum(
        tl.sqrt(flood_cross_section_depth * flood_cross_section_depth_prev), 
        1e-6
    )

    #----------------------------------------------------------------------
    # (6) Current flood area (approximate) & semi-implicit effective area
    #----------------------------------------------------------------------
    flood_cross_section_area = tl.maximum(
        flood_storage / river_length - flood_depth * river_width,
        0.0
    )
    flood_implicit_area = tl.maximum(tl.sqrt(
        flood_cross_section_area * flood_cross_section_area_prev
    ), 1e-6)

    #----------------------------------------------------------------------
    # (7) Update river outflow
    #----------------------------------------------------------------------
    # Condition: river flow depth > 1e-5 and cross-section area > 1e-5
    river_cross_section_area = river_cross_section_depth * river_width
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
    flood_condition = (flood_semi_implicit_flow_depth > 1e-5) & (flood_implicit_area > 1e-5)

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

    #----------------------------------------------------------------------
    # (10) Store results - in-place update
    #----------------------------------------------------------------------
    tl.store(river_outflow_ptr + offs, updated_river_outflow, mask=mask)
    tl.store(flood_outflow_ptr + offs, updated_flood_outflow, mask=mask)
    tl.store(river_cross_section_depth_prev_ptr + offs, river_cross_section_depth, mask=mask)
    tl.store(flood_cross_section_depth_prev_ptr + offs, flood_cross_section_depth, mask=mask)
    tl.store(flood_cross_section_area_prev_ptr + offs, flood_cross_section_area, mask=mask)