import triton
import triton.language as tl

@triton.jit
def compute_bifurcation_outflow_kernel(
    # Indices and configuration
    catchment_idx_ptr,                          # *i32: Catchment indices
    downstream_idx_ptr,                         # *i32: Downstream indices
    bifurcation_manning_ptr,                    # *f32: Bifurcation Manning coefficient
    bifurcation_outflow_ptr,                    # *f32: Bifurcation outflow (in/out)
    bifurcation_width_ptr,                      # *f32: Bifurcation width
    bifurcation_length_ptr,                     # *f32: Bifurcation length
    bifurcation_elevation_ptr,                     # *f32: Bifurcation length
    bifurcation_cross_section_depth_prev_ptr,   # *f32: Previous bifurcation cross-section depth
    water_surface_elevation_ptr,                            # *f32: River depth
    total_storage_ptr,                          # *f32: Total storage (in/out)
    outgoing_storage_ptr,                       # *f32: Outgoing storage (in/out)
    gravity: tl.constexpr,                      # f32: Gravity constant
    time_step,                                  # f32: Time step
    num_paths: tl.constexpr,                    # Total number of bifurcation paths
    num_bifurcation_levels: tl.constexpr,       # int: Number of bifurcation levels    
    BLOCK_SIZE: tl.constexpr                    # Block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_paths
    
    # Load indices
    catchment_idx = tl.load(catchment_idx_ptr + offs, mask=mask, other=0)
    downstream_idx = tl.load(downstream_idx_ptr + offs, mask=mask, other=0)
    
    # Load bifurcation properties
    
    bifurcation_length = tl.load(bifurcation_length_ptr + offs, mask=mask, other=0.0)
    
    # Load river properties for catchment and downstream
    bifurcation_water_surface_elevation = tl.load(water_surface_elevation_ptr + catchment_idx, mask=mask, other=0.0)
    bifurcation_water_surface_elevation_downstream = tl.load(water_surface_elevation_ptr + downstream_idx, mask=mask, other=0.0)
    max_bifurcation_water_surface_elevation = tl.maximum(bifurcation_water_surface_elevation, bifurcation_water_surface_elevation_downstream)

    # Bifurcation slope (clamped similarly to flood slope)
    bifurcation_slope = (bifurcation_water_surface_elevation - bifurcation_water_surface_elevation_downstream) / bifurcation_length
    bifurcation_slope = tl.clamp(bifurcation_slope, -0.005, 0.005)

    # Storage change limiter calculation
    bifurcation_total_storage = tl.load(total_storage_ptr + catchment_idx, mask=mask, other=0.0)
    bifurcation_total_storage_downstream = tl.load(total_storage_ptr + downstream_idx, mask=mask, other=0.0)
    bifurcation_outflow_sum = tl.zeros_like(bifurcation_length)

    for level in tl.static_range(num_bifurcation_levels):
        
        level_idx = offs * num_bifurcation_levels + level
        bifurcation_manning = tl.load(bifurcation_manning_ptr + level_idx, mask=mask, other=0.0)
        bifurcation_cross_section_depth_prev = tl.load(bifurcation_cross_section_depth_prev_ptr + level_idx, mask=mask, other=0.0)
        bifurcation_elevation = tl.load(bifurcation_elevation_ptr + level_idx, mask=mask, other=0.0)
        # Calculate bifurcation cross-section depth
        bifurcation_cross_section_depth = tl.maximum(max_bifurcation_water_surface_elevation - bifurcation_elevation, 0.0)
        # Calculate semi-implicit flow depth for bifurcation
        bifurcation_semi_implicit_flow_depth = tl.maximum(
            tl.sqrt(bifurcation_cross_section_depth * bifurcation_cross_section_depth_prev),
            tl.sqrt(bifurcation_cross_section_depth * 0.01)
        )
        bifurcation_width = tl.load(bifurcation_width_ptr + level_idx, mask=mask, other=0.0)
        bifurcation_outflow = tl.load(bifurcation_outflow_ptr + level_idx, mask=mask, other=0.0)

        unit_bifurcation_outflow = bifurcation_outflow / bifurcation_width

        numerator = bifurcation_width * (
            unit_bifurcation_outflow + gravity * time_step 
            * bifurcation_semi_implicit_flow_depth * bifurcation_slope
        )
        denominator = 1.0 + gravity * time_step * (bifurcation_manning * bifurcation_manning) * tl.abs(unit_bifurcation_outflow) \
                    * tl.exp((-7.0/3.0) * tl.log(bifurcation_semi_implicit_flow_depth))
        
        updated_bifurcation_outflow = numerator / denominator
        bifurcation_condition = (bifurcation_semi_implicit_flow_depth > 1e-5)
        updated_bifurcation_outflow = tl.where(bifurcation_condition, updated_bifurcation_outflow, 0.0)
        bifurcation_outflow_sum += updated_bifurcation_outflow
        tl.store(bifurcation_cross_section_depth_prev_ptr + level_idx, bifurcation_cross_section_depth, mask=mask)
        tl.store(bifurcation_outflow_ptr + level_idx, updated_bifurcation_outflow, mask=mask)
    limit_rate = tl.minimum(0.05 * tl.minimum(bifurcation_total_storage, bifurcation_total_storage_downstream) / (tl.abs(bifurcation_outflow_sum) * time_step), 1.0)
    bifurcation_outflow_sum *= limit_rate
    for level in tl.static_range(num_bifurcation_levels):
        level_idx = offs * num_bifurcation_levels + level
        updated_bifurcation_outflow = tl.load(bifurcation_outflow_ptr + level_idx, mask=mask)
        updated_bifurcation_outflow *= limit_rate
        tl.store(bifurcation_outflow_ptr + level_idx, updated_bifurcation_outflow, mask=mask)

    pos_flow = tl.maximum(bifurcation_outflow_sum, 0.0)
    neg_flow = tl.minimum(bifurcation_outflow_sum, 0.0)
    tl.atomic_add(outgoing_storage_ptr + catchment_idx, pos_flow * time_step, mask=mask)
    tl.atomic_add(outgoing_storage_ptr + downstream_idx, -neg_flow * time_step, mask=mask)