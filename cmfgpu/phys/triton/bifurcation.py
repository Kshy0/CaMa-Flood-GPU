# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

import triton
import triton.language as tl

from cmfgpu.phys.triton.utils import to_compute_dtype, typed_pow, typed_sqrt


@triton.jit
def compute_bifurcation_outflow_kernel(
    # Indices and configuration
    bifurcation_catchment_idx_ptr,                          # *i32: Catchment indices
    bifurcation_downstream_idx_ptr,                         # *i32: Downstream indices
    bifurcation_manning_ptr,                    # *f32: Bifurcation Manning coefficient
    bifurcation_outflow_ptr,                    # *f32: Bifurcation outflow (in/out)
    bifurcation_width_ptr,                      # *f32: Bifurcation width
    bifurcation_length_ptr,                     # *f32: Bifurcation length
    bifurcation_elevation_ptr,                  # *f32: Bifurcation length
    bifurcation_cross_section_depth_ptr,   # *f32: Bifurcation cross-section depth
    water_surface_elevation_ptr,                # *f32: River depth
    total_storage_ptr,                          # *f64: Total storage (in/out)
    outgoing_storage_ptr,                       # *f64: Outgoing storage (in/out)
    gravity: tl.constexpr,                      # f32: Gravity constant
    time_step,                                  # f32: Time step
    num_bifurcation_paths: tl.constexpr,        # Total number of bifurcation paths
    num_bifurcation_levels: tl.constexpr,       # int: Number of bifurcation levels    
    BLOCK_SIZE: tl.constexpr                    # Block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_bifurcation_paths
    
    # Load indices
    bifurcation_catchment_idx = tl.load(bifurcation_catchment_idx_ptr + offs, mask=mask, other=0)
    bifurcation_downstream_idx = tl.load(bifurcation_downstream_idx_ptr + offs, mask=mask, other=0)
    
    # Load bifurcation properties
    
    bifurcation_length = tl.load(bifurcation_length_ptr + offs, mask=mask, other=0.0)
    
    # Load river properties for catchment and downstream
    bifurcation_water_surface_elevation = tl.load(water_surface_elevation_ptr + bifurcation_catchment_idx, mask=mask, other=0.0)
    bifurcation_water_surface_elevation_downstream = tl.load(water_surface_elevation_ptr + bifurcation_downstream_idx, mask=mask, other=0.0)
    max_bifurcation_water_surface_elevation = tl.maximum(bifurcation_water_surface_elevation, bifurcation_water_surface_elevation_downstream)

    # Bifurcation slope (clamped similarly to flood slope)
    bifurcation_slope = (bifurcation_water_surface_elevation - bifurcation_water_surface_elevation_downstream) / bifurcation_length
    bifurcation_slope = tl.clamp(bifurcation_slope, -0.005, 0.005)

    # Storage change limiter calculation
    bifurcation_total_storage = to_compute_dtype(tl.load(total_storage_ptr + bifurcation_catchment_idx, mask=mask, other=0.0), bifurcation_length)
    bifurcation_total_storage_downstream = to_compute_dtype(tl.load(total_storage_ptr + bifurcation_downstream_idx, mask=mask, other=0.0), bifurcation_length)
    sum_bifurcation_outflow = tl.zeros_like(bifurcation_length)

    for level in tl.static_range(num_bifurcation_levels):
        
        level_idx = offs * num_bifurcation_levels + level
        bifurcation_manning = tl.load(bifurcation_manning_ptr + level_idx, mask=mask, other=0.0)
        bifurcation_cross_section_depth = tl.load(bifurcation_cross_section_depth_ptr + level_idx, mask=mask, other=0.0)
        bifurcation_elevation = tl.load(bifurcation_elevation_ptr + level_idx, mask=mask, other=0.0)
        # Calculate bifurcation cross-section depth
        updated_bifurcation_cross_section_depth = tl.maximum(max_bifurcation_water_surface_elevation - bifurcation_elevation, 0.0)
        # Calculate semi-implicit flow depth for bifurcation
        bifurcation_semi_implicit_flow_depth = tl.maximum(
            typed_sqrt(updated_bifurcation_cross_section_depth * bifurcation_cross_section_depth),
            typed_sqrt(updated_bifurcation_cross_section_depth * 0.01)
        )
        bifurcation_width = tl.load(bifurcation_width_ptr + level_idx, mask=mask, other=0.0)
        bifurcation_outflow = tl.load(bifurcation_outflow_ptr + level_idx, mask=mask, other=0.0)

        unit_bifurcation_outflow = bifurcation_outflow / bifurcation_width

        numerator = bifurcation_width * (
            unit_bifurcation_outflow + gravity * time_step 
            * bifurcation_semi_implicit_flow_depth * bifurcation_slope
        )
        denominator = 1.0 + gravity * time_step * (bifurcation_manning * bifurcation_manning) * tl.abs(unit_bifurcation_outflow) \
                    * typed_pow(bifurcation_semi_implicit_flow_depth, -7.0/3.0)
        
        updated_bifurcation_outflow = numerator / denominator
        bifurcation_condition = (bifurcation_semi_implicit_flow_depth > 1e-5)
        updated_bifurcation_outflow = tl.where(bifurcation_condition, updated_bifurcation_outflow, 0.0)
        sum_bifurcation_outflow += updated_bifurcation_outflow
        tl.store(bifurcation_cross_section_depth_ptr + level_idx, updated_bifurcation_cross_section_depth, mask=mask)
        tl.store(bifurcation_outflow_ptr + level_idx, updated_bifurcation_outflow, mask=mask)
    limit_rate = tl.minimum(0.05 * tl.minimum(bifurcation_total_storage, bifurcation_total_storage_downstream) / (tl.abs(sum_bifurcation_outflow) * time_step), 1.0)
    sum_bifurcation_outflow *= limit_rate
    for level in tl.static_range(num_bifurcation_levels):
        level_idx = offs * num_bifurcation_levels + level
        updated_bifurcation_outflow = tl.load(bifurcation_outflow_ptr + level_idx, mask=mask)
        updated_bifurcation_outflow *= limit_rate
        tl.store(bifurcation_outflow_ptr + level_idx, updated_bifurcation_outflow, mask=mask)

    pos_flow = tl.maximum(sum_bifurcation_outflow, 0.0)
    neg_flow = tl.minimum(sum_bifurcation_outflow, 0.0)
    tl.atomic_add(outgoing_storage_ptr + bifurcation_catchment_idx, pos_flow * time_step, mask=mask)
    tl.atomic_add(outgoing_storage_ptr + bifurcation_downstream_idx, -neg_flow * time_step, mask=mask)

@triton.jit
def compute_bifurcation_inflow_kernel(
    # Indices and configuration
    bifurcation_catchment_idx_ptr,                          # *i32: Catchment indices
    bifurcation_downstream_idx_ptr,                         # *i32: Downstream indices
    limit_rate_ptr,                             # *f32: Limit rate
    bifurcation_outflow_ptr,                    # *f32: Bifurcation inflow (in/out)
    global_bifurcation_outflow_ptr,            # *f64: Global bifurcation outflow
    num_bifurcation_paths: tl.constexpr,                    # Total number of bifurcation paths
    num_bifurcation_levels: tl.constexpr,       # int: Number of bifurcation levels
    BLOCK_SIZE: tl.constexpr                     # Block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_bifurcation_paths
    
    # Load indices
    bifurcation_catchment_idx = tl.load(bifurcation_catchment_idx_ptr + offs, mask=mask, other=0)
    bifurcation_downstream_idx = tl.load(bifurcation_downstream_idx_ptr + offs, mask=mask, other=0)
    # Load limit rate
    limit_rate = tl.load(limit_rate_ptr + bifurcation_catchment_idx, mask=mask, other=1.0)
    limit_rate_downstream = tl.load(limit_rate_ptr + bifurcation_downstream_idx, mask=mask, other=1.0)
    sum_bifurcation_outflow = tl.zeros_like(limit_rate)

    for level in tl.static_range(num_bifurcation_levels):
        level_idx = offs * num_bifurcation_levels + level
        updated_bifurcation_outflow = tl.load(bifurcation_outflow_ptr + level_idx, mask=mask)
        updated_bifurcation_outflow = tl.where(updated_bifurcation_outflow >= 0.0, updated_bifurcation_outflow * limit_rate, updated_bifurcation_outflow * limit_rate_downstream)
        sum_bifurcation_outflow += updated_bifurcation_outflow
        tl.store(bifurcation_outflow_ptr + level_idx, updated_bifurcation_outflow, mask=mask)
        
    tl.atomic_add(global_bifurcation_outflow_ptr + bifurcation_catchment_idx, sum_bifurcation_outflow, mask=mask)
    tl.atomic_add(global_bifurcation_outflow_ptr + bifurcation_downstream_idx, -sum_bifurcation_outflow, mask=mask)


@triton.jit
def compute_bifurcation_outflow_batched_kernel(
    # Indices and configuration
    bifurcation_catchment_idx_ptr,                          # *i32: Catchment indices
    bifurcation_downstream_idx_ptr,                         # *i32: Downstream indices
    bifurcation_manning_ptr,                    # *f32: Bifurcation Manning coefficient
    bifurcation_outflow_ptr,                    # *f32: Bifurcation outflow (in/out)
    bifurcation_width_ptr,                      # *f32: Bifurcation width
    bifurcation_length_ptr,                     # *f32: Bifurcation length
    bifurcation_elevation_ptr,                  # *f32: Bifurcation length
    bifurcation_cross_section_depth_ptr,   # *f32: Bifurcation cross-section depth
    water_surface_elevation_ptr,                # *f32: River depth
    total_storage_ptr,                          # *f64: Total storage (in/out)
    outgoing_storage_ptr,                       # *f64: Outgoing storage (in/out)
    gravity: tl.constexpr,                      # f32: Gravity constant
    time_step,                                  # f32: Time step
    num_bifurcation_paths: tl.constexpr,        # Total number of bifurcation paths
    num_bifurcation_levels: tl.constexpr,       # int: Number of bifurcation levels    
    num_trials: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,                    # Block size
    num_catchments: tl.constexpr,               # Need this for stride of catchment arrays
    # Batch flags
    batched_bifurcation_manning: tl.constexpr,
    batched_bifurcation_width: tl.constexpr,
    batched_bifurcation_length: tl.constexpr,
    batched_bifurcation_elevation: tl.constexpr
):
    pid_x = tl.program_id(0)
    idx = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate trial and path indices
    trial_idx = idx // num_bifurcation_paths
    offs = idx % num_bifurcation_paths
    
    mask = idx < (num_bifurcation_paths * num_trials)
    
    trial_offset_paths = trial_idx * num_bifurcation_paths
    trial_offset_catchments = trial_idx * num_catchments
    trial_offset_levels = trial_idx * num_bifurcation_paths * num_bifurcation_levels
    
    # Load indices
    # Topology is never batched
    bifurcation_catchment_idx = tl.load(bifurcation_catchment_idx_ptr + offs, mask=mask, other=0)
    bifurcation_downstream_idx = tl.load(bifurcation_downstream_idx_ptr + offs, mask=mask, other=0)
    
    # Load bifurcation properties
    bifurcation_length = tl.load(bifurcation_length_ptr + (trial_offset_paths if batched_bifurcation_length else 0) + offs, mask=mask, other=0.0)
    
    # Load river properties for catchment and downstream
    bifurcation_water_surface_elevation = tl.load(water_surface_elevation_ptr + trial_offset_catchments + bifurcation_catchment_idx, mask=mask, other=0.0)
    bifurcation_water_surface_elevation_downstream = tl.load(water_surface_elevation_ptr + trial_offset_catchments + bifurcation_downstream_idx, mask=mask, other=0.0)
    max_bifurcation_water_surface_elevation = tl.maximum(bifurcation_water_surface_elevation, bifurcation_water_surface_elevation_downstream)

    # Bifurcation slope (clamped similarly to flood slope)
    bifurcation_slope = (bifurcation_water_surface_elevation - bifurcation_water_surface_elevation_downstream) / bifurcation_length
    bifurcation_slope = tl.clamp(bifurcation_slope, -0.005, 0.005)

    # Storage change limiter calculation
    bifurcation_total_storage = to_compute_dtype(tl.load(total_storage_ptr + trial_offset_catchments + bifurcation_catchment_idx, mask=mask, other=0.0), bifurcation_length)
    bifurcation_total_storage_downstream = to_compute_dtype(tl.load(total_storage_ptr + trial_offset_catchments + bifurcation_downstream_idx, mask=mask, other=0.0), bifurcation_length)
    sum_bifurcation_outflow = tl.zeros_like(bifurcation_length)
    
    # Base offsets for level-dependent arrays
    manning_base = (trial_offset_levels if batched_bifurcation_manning else 0)
    width_base = (trial_offset_levels if batched_bifurcation_width else 0)
    elevation_base = (trial_offset_levels if batched_bifurcation_elevation else 0)
    
    for level in tl.static_range(num_bifurcation_levels):
        
        level_idx = offs * num_bifurcation_levels + level
        
        bifurcation_manning = tl.load(bifurcation_manning_ptr + manning_base + level_idx, mask=mask, other=0.0)
        bifurcation_cross_section_depth = tl.load(bifurcation_cross_section_depth_ptr + trial_offset_levels + level_idx, mask=mask, other=0.0)
        bifurcation_elevation = tl.load(bifurcation_elevation_ptr + elevation_base + level_idx, mask=mask, other=0.0)
        # Calculate bifurcation cross-section depth
        updated_bifurcation_cross_section_depth = tl.maximum(max_bifurcation_water_surface_elevation - bifurcation_elevation, 0.0)
        # Calculate semi-implicit flow depth for bifurcation
        bifurcation_semi_implicit_flow_depth = tl.maximum(
            typed_sqrt(updated_bifurcation_cross_section_depth * bifurcation_cross_section_depth),
            typed_sqrt(updated_bifurcation_cross_section_depth * 0.01)
        )
        bifurcation_width = tl.load(bifurcation_width_ptr + width_base + level_idx, mask=mask, other=0.0)
        bifurcation_outflow = tl.load(bifurcation_outflow_ptr + trial_offset_levels + level_idx, mask=mask, other=0.0)

        unit_bifurcation_outflow = bifurcation_outflow / bifurcation_width

        numerator = bifurcation_width * (
            unit_bifurcation_outflow + gravity * time_step 
            * bifurcation_semi_implicit_flow_depth * bifurcation_slope
        )
        denominator = 1.0 + gravity * time_step * (bifurcation_manning * bifurcation_manning) * tl.abs(unit_bifurcation_outflow) \
                    * typed_pow(bifurcation_semi_implicit_flow_depth, -7.0/3.0)
        
        updated_bifurcation_outflow = numerator / denominator
        bifurcation_condition = (bifurcation_semi_implicit_flow_depth > 1e-5)
        updated_bifurcation_outflow = tl.where(bifurcation_condition, updated_bifurcation_outflow, 0.0)
        sum_bifurcation_outflow += updated_bifurcation_outflow
        tl.store(bifurcation_cross_section_depth_ptr + trial_offset_levels + level_idx, updated_bifurcation_cross_section_depth, mask=mask)
        tl.store(bifurcation_outflow_ptr + trial_offset_levels + level_idx, updated_bifurcation_outflow, mask=mask)
    limit_rate = tl.minimum(0.05 * tl.minimum(bifurcation_total_storage, bifurcation_total_storage_downstream) / (tl.abs(sum_bifurcation_outflow) * time_step), 1.0)
    sum_bifurcation_outflow *= limit_rate
    for level in tl.static_range(num_bifurcation_levels):
        level_idx = offs * num_bifurcation_levels + level
        updated_bifurcation_outflow = tl.load(bifurcation_outflow_ptr + trial_offset_levels + level_idx, mask=mask)
        updated_bifurcation_outflow *= limit_rate
        tl.store(bifurcation_outflow_ptr + trial_offset_levels + level_idx, updated_bifurcation_outflow, mask=mask)

    pos_flow = tl.maximum(sum_bifurcation_outflow, 0.0)
    neg_flow = tl.minimum(sum_bifurcation_outflow, 0.0)
    tl.atomic_add(outgoing_storage_ptr + trial_offset_catchments + bifurcation_catchment_idx, pos_flow * time_step, mask=mask)
    tl.atomic_add(outgoing_storage_ptr + trial_offset_catchments + bifurcation_downstream_idx, -neg_flow * time_step, mask=mask)


@triton.jit
def compute_bifurcation_inflow_batched_kernel(
    # Indices and configuration
    bifurcation_catchment_idx_ptr,                          # *i32: Catchment indices
    bifurcation_downstream_idx_ptr,                         # *i32: Downstream indices
    limit_rate_ptr,                             # *f32: Limit rate
    bifurcation_outflow_ptr,                    # *f32: Bifurcation inflow (in/out)
    global_bifurcation_outflow_ptr,            # *f64: Global bifurcation outflow
    num_bifurcation_paths: tl.constexpr,                    # Total number of bifurcation paths
    num_bifurcation_levels: tl.constexpr,       # int: Number of bifurcation levels
    num_trials: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,                     # Block size
    num_catchments: tl.constexpr,
):
    pid_x = tl.program_id(0)
    idx = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate trial and path indices
    trial_idx = idx // num_bifurcation_paths
    offs = idx % num_bifurcation_paths
    
    mask = idx < (num_bifurcation_paths * num_trials)
    
    trial_offset_catchments = trial_idx * num_catchments
    trial_offset_levels = trial_idx * num_bifurcation_paths * num_bifurcation_levels
    
    # Load indices
    # Topology is never batched
    bifurcation_catchment_idx = tl.load(bifurcation_catchment_idx_ptr + offs, mask=mask, other=0)
    bifurcation_downstream_idx = tl.load(bifurcation_downstream_idx_ptr + offs, mask=mask, other=0)
    
    # Load limit rate
    limit_rate = tl.load(limit_rate_ptr + trial_offset_catchments + bifurcation_catchment_idx, mask=mask, other=1.0)
    limit_rate_downstream = tl.load(limit_rate_ptr + trial_offset_catchments + bifurcation_downstream_idx, mask=mask, other=1.0)
    sum_bifurcation_outflow = tl.zeros_like(limit_rate)

    for level in tl.static_range(num_bifurcation_levels):
        level_idx = offs * num_bifurcation_levels + level
        updated_bifurcation_outflow = tl.load(bifurcation_outflow_ptr + trial_offset_levels + level_idx, mask=mask)
        updated_bifurcation_outflow = tl.where(updated_bifurcation_outflow >= 0.0, updated_bifurcation_outflow * limit_rate, updated_bifurcation_outflow * limit_rate_downstream)
        sum_bifurcation_outflow += updated_bifurcation_outflow
        tl.store(bifurcation_outflow_ptr + trial_offset_levels + level_idx, updated_bifurcation_outflow, mask=mask)
        
    tl.atomic_add(global_bifurcation_outflow_ptr + trial_offset_catchments + bifurcation_catchment_idx, sum_bifurcation_outflow, mask=mask)
    tl.atomic_add(global_bifurcation_outflow_ptr + trial_offset_catchments + bifurcation_downstream_idx, -sum_bifurcation_outflow, mask=mask)
