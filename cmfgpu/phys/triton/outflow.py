# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

import numpy as np
import triton
import triton.language as tl
from hydroforge.kernels import triton_math as hm
from cmfgpu import config as _constants

BACKFLOW_STORAGE_FRACTION = tl.constexpr(_constants.BACKFLOW_STORAGE_FRACTION)
OUTGOING_VOLUME_FLOOR = tl.constexpr(_constants.OUTGOING_VOLUME_FLOOR)
BACKFLOW_VOLUME_FLOOR = tl.constexpr(float(np.float32(_constants.OUTGOING_VOLUME_FLOOR)))
ROUTING_SLOPE_LIMIT = tl.constexpr(_constants.ROUTING_SLOPE_LIMIT)


@triton.jit
def outgoing_flows_inline(river_outflow, flood_outflow, storage_dtype):
    """P2STOOUT flows of CALC_INFLOW_LSPAMAT: the cell's own positive
    flows, and the flow reversed into its downstream cell."""
    own_flow = (
        tl.maximum(river_outflow, 0.0) + tl.maximum(flood_outflow, 0.0)
    ).to(storage_dtype)
    reversed_flow = (
        tl.maximum(-river_outflow, 0.0).to(storage_dtype)
        + tl.maximum(-flood_outflow, 0.0).to(storage_dtype)
    )
    return own_flow, reversed_flow


@triton.jit
def supply_rate_inline(outgoing_flow, storage, step, like):
    """Supply-side rate of CALC_INFLOW_LSPAMAT."""
    volume = hm.at_least(
        hm.to_compute(outgoing_flow * step, like), OUTGOING_VOLUME_FLOOR,
    )
    return hm.at_most(
        hm.divide(hm.to_compute(storage, like), volume), 1.0,
    )


@triton.jit
def compute_outflow_kernel(
    downstream_idx_ptr,                     # *i32 downstream index

    # river variables
    river_inflow_ptr,                       # *f64 river inflow (turn to zero)
    river_outflow_ptr,                      # *f32 in/out river outflow
    river_manning_ptr,                      # *f32 river Manning coefficient
    river_depth_ptr,                        # *f32 river depth
    river_width_ptr,                        # *f32 river width
    river_length_ptr,                       # *f32 river length
    river_height_ptr,                       # *f32 river bank height
    river_storage_ptr,                      # *f64 river storage

    # flood variables
    flood_inflow_ptr,                       # *f64 flood inflow (turn to zero)
    flood_outflow_ptr,                      # *f32 in/out flood outflow
    flood_manning_ptr,                      # *f32 flood Manning coefficient
    flood_depth_ptr,                        # *f32 flood depth
    catchment_elevation_ptr,
    downstream_distance_ptr,                # *f32 distance to downstream unit
    flood_storage_ptr,                      # *f64 flood storage
    protected_storage_ptr,                  # *f64 protected storage

    # previous time step variables
    river_cross_section_depth_ptr,     # *f32 previous river cross-section depth
    flood_cross_section_depth_ptr,     # *f32 previous flood cross-section depth
    flood_cross_section_area_ptr,      # *f32 previous flood cross-section area

    # other 
    global_bifurcation_outflow_ptr,          # *f64 global bifurcation outflow (turn to zero)
    outgoing_storage_ptr,                   # *f64 output for storage (fused part)
    gravity: tl.constexpr,                  # f32 scalar gravity acceleration
    time_step_ptr,                              # f32 scalar time step
    num_catchments: tl.constexpr,           # total number of elements
    BLOCK_SIZE: tl.constexpr,               # block size
    HAS_BIFURCATION: tl.constexpr = True,   # whether bifurcation module is active
    HAS_LEVEE: tl.constexpr = False,
    is_dam_upstream_ptr=None,               # *bool  upstream-of-dam mask (catchment-indexed)
    HAS_RESERVOIR: tl.constexpr = False,    # whether reservoir module is active
    min_kinematic_slope: tl.constexpr = 1.0e-5,  # minimum bed slope for kinematic wave
    sea_surface_elevation_ptr=None,
    catchment_sea_level_idx_ptr=None,
    num_sea_level_boundaries: tl.constexpr = 0,
    HAS_SEA_LEVEL: tl.constexpr = False,
):
    _ = num_sea_level_boundaries
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_catchments
    time_step = tl.load(time_step_ptr)

    #----------------------------------------------------------------------
    # (1) Load previous time step input variables
    #----------------------------------------------------------------------
    downstream_idx = tl.load(downstream_idx_ptr + offs, mask=mask, other=0)
    is_river_mouth = downstream_idx == offs

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
    catchment_elevation = tl.load(catchment_elevation_ptr + offs, mask=mask, other=0.0)
    downstream_distance = tl.load(downstream_distance_ptr + offs, mask=mask, other=1.0)
    flood_storage = tl.load(flood_storage_ptr + offs, mask=mask, other=0.0)

    # cross section variables
    river_cross_section_depth = tl.load(river_cross_section_depth_ptr + offs, mask=mask, other=0.0)
    flood_cross_section_depth = tl.load(flood_cross_section_depth_ptr + offs, mask=mask, other=0.0)
    flood_cross_section_area = tl.load(flood_cross_section_area_ptr + offs, mask=mask, other=0.0)

    storage_sum = river_storage + flood_storage
    if HAS_LEVEE:
        storage_sum += tl.load(protected_storage_ptr + offs, mask=mask, other=0.0)
    total_storage = hm.to_compute(storage_sum, river_outflow)
    river_storage = hm.to_compute(river_storage, river_outflow)
    flood_storage = hm.to_compute(flood_storage, river_outflow)

    #----------------------------------------------------------------------
    # (2) Compute current river water surface elevation & downstream water surface elevation
    #----------------------------------------------------------------------
    river_elevation = catchment_elevation - river_height
    water_surface_elevation = river_depth + river_elevation
    # Downstream water surface elevation
    river_depth_downstream = tl.load(river_depth_ptr + downstream_idx, mask=mask, other=0.0)
    river_height_downstream = tl.load(river_height_ptr + downstream_idx, mask=mask, other=0.0)
    catchment_elevation_downstream = tl.load(catchment_elevation_ptr + downstream_idx, mask=mask, other=0.0)
    river_elevation_downstream = catchment_elevation_downstream - river_height_downstream
    water_surface_elevation_downstream = river_depth_downstream + river_elevation_downstream
    
    water_surface_elevation_downstream = tl.where(is_river_mouth, catchment_elevation, water_surface_elevation_downstream)
    if HAS_SEA_LEVEL:
        sea_level_idx = tl.load(
            catchment_sea_level_idx_ptr + offs, mask=mask, other=-1,
        )
        prescribed_level = tl.load(
            sea_surface_elevation_ptr + sea_level_idx,
            mask=mask & (sea_level_idx >= 0), other=0.0,
        )
        water_surface_elevation_downstream = tl.where(
            sea_level_idx >= 0, prescribed_level,
            water_surface_elevation_downstream,
        )
    max_water_surface_elevation = tl.maximum(
        water_surface_elevation, water_surface_elevation_downstream,
    )
    
    #----------------------------------------------------------------------
    # (4) Longitudinal water surface slope & truncated flood slope
    #----------------------------------------------------------------------
    river_slope = hm.divide(water_surface_elevation - water_surface_elevation_downstream, downstream_distance)
    flood_slope = hm.clamp(river_slope, -ROUTING_SLOPE_LIMIT, ROUTING_SLOPE_LIMIT)

    #----------------------------------------------------------------------
    # (5) Current river/flood cross-section depth + semi-implicit flow depth
    #----------------------------------------------------------------------
    # CaMa-Flood applies a separate river-mouth boundary: the prescribed
    # downstream level controls slope, but the local channel flow depth remains
    # the actual river depth (rather than max(river_depth, bankfull depth)).
    updated_river_cross_section_depth = tl.where(
        is_river_mouth,
        river_depth,
        max_water_surface_elevation - river_elevation,
    )
    river_semi_implicit_flow_depth = hm.at_least(hm.sqrt(
        updated_river_cross_section_depth * river_cross_section_depth
    ), 1e-6)

    flood_cross_section_surface = tl.where(
        is_river_mouth,
        water_surface_elevation,
        max_water_surface_elevation,
    )
    updated_flood_cross_section_depth = tl.maximum(
        flood_cross_section_surface - catchment_elevation,
        0.0
    )
    flood_semi_implicit_flow_depth = hm.at_least(
        hm.sqrt(updated_flood_cross_section_depth * flood_cross_section_depth),
        1e-6,
    )

    #----------------------------------------------------------------------
    # (6) Current flood area (approximate) & semi-implicit effective area
    #----------------------------------------------------------------------
    updated_flood_cross_section_area = tl.maximum(
        hm.divide(flood_storage, river_length) - flood_depth * river_width,
        0.0
    )
    flood_implicit_area = hm.at_least(hm.sqrt(
        updated_flood_cross_section_area * hm.at_least(flood_cross_section_area, 1e-6)
    ), 1e-6)

    #----------------------------------------------------------------------
    # (7) Update river outflow
    #----------------------------------------------------------------------
    river_cross_section_area = updated_river_cross_section_depth * river_width
    flow_threshold = hm.constant(1e-5, river_slope)
    river_condition = (river_semi_implicit_flow_depth > flow_threshold) & (river_cross_section_area > flow_threshold)

    # Original river outflow (per unit width)
    unit_river_outflow = hm.divide(river_outflow, river_width)

    numerator_river = river_width * (
        unit_river_outflow + gravity * time_step 
        * river_semi_implicit_flow_depth * river_slope
    )
    
    # Use libdevice.pow() for power calculation
    denominator_river = 1.0 + gravity * time_step * (river_manning * river_manning) * tl.abs(unit_river_outflow) \
                      * hm.divide(1.0, river_semi_implicit_flow_depth * river_semi_implicit_flow_depth * hm.cbrt(river_semi_implicit_flow_depth))

    updated_river_outflow = hm.divide(numerator_river, denominator_river)
    updated_river_outflow = tl.where(river_condition, updated_river_outflow, 0.0)

    #----------------------------------------------------------------------
    # (8) Update flood outflow
    #----------------------------------------------------------------------
    flood_condition = (flood_semi_implicit_flow_depth > flow_threshold) & (updated_flood_cross_section_area > flow_threshold)

    numerator_flood = flood_outflow + gravity * time_step * flood_implicit_area * flood_slope
    
    # Use libdevice.pow() for power calculation
    denominator_flood = 1.0 + hm.divide(
        gravity * time_step * (flood_manning * flood_manning) * tl.abs(flood_outflow)
        * hm.divide(1.0, flood_semi_implicit_flow_depth * hm.cbrt(flood_semi_implicit_flow_depth)),
        flood_implicit_area,
    )
                      
    updated_flood_outflow = hm.divide(numerator_flood, denominator_flood)
    updated_flood_outflow = tl.where(flood_condition, updated_flood_outflow, 0.0)

    #----------------------------------------------------------------------
    # (9) Flood-direction mask and storage-change limiter
    #----------------------------------------------------------------------
    # Floodplain flow only moves with the river flow (rivout*fldout > 0).
    same_direction = (updated_river_outflow * updated_flood_outflow) > 0.0
    updated_flood_outflow = tl.where(same_direction, updated_flood_outflow, 0.0)
    # v4.23 storage-change limiter on every non-mouth cell: flow towards the
    # upstream cell removes at most 5% of the storage per step.
    backflow = hm.at_least(
        (-updated_river_outflow - updated_flood_outflow) * time_step,
        BACKFLOW_VOLUME_FLOOR,
    )
    limit_rate = hm.at_most(
        hm.divide(BACKFLOW_STORAGE_FRACTION * total_storage, backflow), 1.0,
    )
    updated_river_outflow = tl.where(is_river_mouth, updated_river_outflow, updated_river_outflow * limit_rate)
    updated_flood_outflow = tl.where(is_river_mouth, updated_flood_outflow, updated_flood_outflow * limit_rate)

    #----------------------------------------------------------------------
    # (9b) Kinematic wave override for upstream-of-dam cells.
    #      Uses bed slope (catchment elevation gradient) instead of water-surface slope.
    #----------------------------------------------------------------------
    if HAS_RESERVOIR:
        is_dam_up = tl.load(
            is_dam_upstream_ptr + offs, mask=mask, other=0,
        ) != 0
        # Bed slope
        bed_slope = hm.divide(catchment_elevation - tl.load(catchment_elevation_ptr + downstream_idx, mask=mask, other=0.0), downstream_distance)
        bed_slope = hm.at_least(bed_slope, min_kinematic_slope)
        # River kinematic: Q = W * n^{-1} * S^{0.5} * d^{5/3}
        kin_riv_vel = hm.divide(1.0, river_manning) * hm.sqrt(bed_slope) * hm.cbrt(river_depth * river_depth)
        # CaMa bounds the kinematic flows by the storage only (MIN), so a
        # negative storage (negative runoff) gives a negative flow.
        kin_riv = tl.minimum(river_width * river_depth * kin_riv_vel, hm.divide(river_storage, time_step))
        # Flood kinematic: slope clamped to 0.005
        bed_slope_f = hm.at_most(bed_slope, ROUTING_SLOPE_LIMIT)
        kin_fld_vel = hm.divide(1.0, flood_manning) * hm.sqrt(bed_slope_f) * hm.cbrt(flood_depth * flood_depth)
        kin_fld_area = tl.maximum(hm.divide(flood_storage, river_length) - flood_depth * river_width, 0.0)
        kin_fld = tl.minimum(kin_fld_area * kin_fld_vel, hm.divide(flood_storage, time_step))
        # Override
        updated_river_outflow = tl.where(is_dam_up, kin_riv, updated_river_outflow)
        updated_flood_outflow = tl.where(is_dam_up, kin_fld, updated_flood_outflow)

    #----------------------------------------------------------------------
    # (10) Store results - in-place update
    #----------------------------------------------------------------------
    tl.store(river_outflow_ptr + offs, updated_river_outflow, mask=mask)
    tl.store(flood_outflow_ptr + offs, updated_flood_outflow, mask=mask)
    tl.store(river_cross_section_depth_ptr + offs, updated_river_cross_section_depth, mask=mask)
    tl.store(flood_cross_section_depth_ptr + offs, updated_flood_cross_section_depth, mask=mask)
    # Next step's DARE_pr uses D2FLDDPH_PRE = max(D2RIVDPH_PRE - D2RIVHGT, 0).
    previous_flood_area = tl.maximum(
        hm.divide(flood_storage, river_length)
        - tl.maximum(river_depth - river_height, 0.0) * river_width,
        0.0,
    )
    tl.store(flood_cross_section_area_ptr + offs, previous_flood_area, mask=mask)
    
    tl.store(river_inflow_ptr + offs, 0.0, mask=mask)
    tl.store(flood_inflow_ptr + offs, 0.0, mask=mask)
    if HAS_BIFURCATION:
        tl.store(global_bifurcation_outflow_ptr + offs, 0.0, mask=mask)

    #----------------------------------------------------------------------
    # (11) Fused outgoing storage computation (was compute_outgoing_storage_kernel)
    #----------------------------------------------------------------------
    own_flow, reversed_flow = outgoing_flows_inline(
        updated_river_outflow, updated_flood_outflow,
        outgoing_storage_ptr.dtype.element_ty,
    )
    tl.atomic_add(outgoing_storage_ptr + offs, own_flow, mask=mask, sem="relaxed")
    tl.atomic_add(outgoing_storage_ptr + downstream_idx, reversed_flow, mask=mask & ~is_river_mouth, sem="relaxed")
    


@triton.jit
def compute_inflow_kernel(
    downstream_idx_ptr,            # *i32: Downstream indices
    river_outflow_ptr,             # *f32: River outflow (in/out)
    flood_outflow_ptr,             # *f32: Flood outflow (in/out)
    river_storage_ptr,             # *f64: River storage (rivsto)
    flood_storage_ptr,             # *f64: Flood storage (fldsto)
    outgoing_storage_ptr,          # *f64: Outgoing flow sum (P2STOOUT / step)
    time_step_ptr,                 # *f32: Time step
    river_inflow_ptr,              # *f64: River inflow (output, atomic add)
    flood_inflow_ptr,              # *f64: Flood inflow (output, atomic add)
    limit_rate_ptr,                # *f32: Limit rate diagnostic
    reservoir_total_inflow_ptr,    # *f64: Reservoir total inflow (catchment-sized, atomic add)
    is_reservoir_ptr,              # *i1:  Boolean mask for reservoir catchments
    num_catchments: tl.constexpr,  # Total number of units
    HAS_BIFURCATION: tl.constexpr, # Whether bifurcation module is active
    HAS_RESERVOIR: tl.constexpr,   # Whether reservoir module is active
    BLOCK_SIZE: tl.constexpr       # Block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_catchments

    # -------- Load for limiting --------
    river_outflow   = tl.load(river_outflow_ptr      + offs, mask=mask, other=0.0)
    flood_outflow   = tl.load(flood_outflow_ptr      + offs, mask=mask, other=0.0)

    # CaMa-Flood v4.23 supply-side limiter (CALC_INFLOW_LSPAMAT): a cell
    # releases at most its storage.  The rate of the cell a flow leaves sets
    # both of its flows, chosen by the river flow's direction.
    step = tl.load(time_step_ptr).to(outgoing_storage_ptr.dtype.element_ty)
    limit_rate = supply_rate_inline(
        tl.load(outgoing_storage_ptr + offs, mask=mask, other=0.0),
        tl.load(river_storage_ptr + offs, mask=mask, other=0.0)
        + tl.load(flood_storage_ptr + offs, mask=mask, other=0.0),
        step, river_outflow,
    )
    downstream_idx   = tl.load(downstream_idx_ptr        + offs, mask=mask, other=0)
    limit_rate_downstream = supply_rate_inline(
        tl.load(outgoing_storage_ptr + downstream_idx, mask=mask, other=0.0),
        tl.load(river_storage_ptr + downstream_idx, mask=mask, other=0.0)
        + tl.load(flood_storage_ptr + downstream_idx, mask=mask, other=0.0),
        step, river_outflow,
    )
    rate = tl.where(river_outflow > 0.0, limit_rate, limit_rate_downstream)
    updated_river_outflow = river_outflow * rate
    updated_flood_outflow = flood_outflow * rate

    # Write back limited values
    tl.store(river_outflow_ptr + offs, updated_river_outflow, mask=mask)
    tl.store(flood_outflow_ptr + offs, updated_flood_outflow, mask=mask)
    if HAS_BIFURCATION:
        tl.store(limit_rate_ptr + offs, limit_rate, mask=mask)

    # -------- Accumulate inflows --------
    is_river_mouth = downstream_idx == offs
    not_mouth = mask & (~is_river_mouth)
    tl.atomic_add(river_inflow_ptr + downstream_idx, updated_river_outflow, mask=not_mouth, sem="relaxed")
    tl.atomic_add(flood_inflow_ptr + downstream_idx, updated_flood_outflow, mask=not_mouth, sem="relaxed")

    # -------- Accumulate reservoir inflow --------
    if HAS_RESERVOIR:
        is_downstream_res = tl.load(is_reservoir_ptr + downstream_idx, mask=not_mouth, other=0) != 0
        reservoir_dtype = reservoir_total_inflow_ptr.dtype.element_ty
        total_outflow = updated_river_outflow.to(reservoir_dtype) + updated_flood_outflow.to(reservoir_dtype)
        tl.atomic_add(reservoir_total_inflow_ptr + downstream_idx, total_outflow, mask=not_mouth & is_downstream_res, sem="relaxed")


@triton.jit
def compute_outflow_batched_kernel(
    downstream_idx_ptr,                     # *i32 downstream index

    # river variables
    river_inflow_ptr,                       # *f64 river inflow (turn to zero)
    river_outflow_ptr,                      # *f32 in/out river outflow
    river_manning_ptr,                      # *f32 river Manning coefficient
    river_depth_ptr,                        # *f32 river depth
    river_width_ptr,                        # *f32 river width
    river_length_ptr,                       # *f32 river length
    river_height_ptr,                       # *f32 river bank height
    river_storage_ptr,                      # *f64 river storage

    # flood variables
    flood_inflow_ptr,                       # *f64 flood inflow (turn to zero)
    flood_outflow_ptr,                      # *f32 in/out flood outflow
    flood_manning_ptr,                      # *f32 flood Manning coefficient
    flood_depth_ptr,                        # *f32 flood depth
    catchment_elevation_ptr,
    downstream_distance_ptr,                # *f32 distance to downstream unit
    flood_storage_ptr,                      # *f64 flood storage
    protected_storage_ptr,                  # *f64 protected storage

    # previous time step variables
    river_cross_section_depth_ptr,     # *f32 previous river cross-section depth
    flood_cross_section_depth_ptr,     # *f32 previous flood cross-section depth
    flood_cross_section_area_ptr,      # *f32 previous flood cross-section area

    # other 
    global_bifurcation_outflow_ptr,          # *f64 global bifurcation outflow (turn to zero)
    outgoing_storage_ptr,                   # *f64 output for storage (fused part)
    gravity: tl.constexpr,                  # f32 scalar gravity acceleration
    time_step_ptr,                              # f32 scalar time step
    num_catchments: tl.constexpr,           # total number of elements
    ensemble_size: tl.constexpr,                             # number of members
    BLOCK_SIZE: tl.constexpr,                # block size
    
    # Batch flags
    batched_river_manning: tl.constexpr,
    batched_flood_manning: tl.constexpr,
    batched_river_width: tl.constexpr,
    batched_river_length: tl.constexpr,
    batched_river_height: tl.constexpr,
    batched_catchment_elevation: tl.constexpr,
    batched_downstream_distance: tl.constexpr,
    batched_sea_surface_elevation: tl.constexpr,
    HAS_BIFURCATION: tl.constexpr = True,   # whether bifurcation module is active
    HAS_LEVEE: tl.constexpr = False,
    is_dam_upstream_ptr=None,
    HAS_RESERVOIR: tl.constexpr = False,
    min_kinematic_slope: tl.constexpr = 1.0e-5,
    sea_surface_elevation_ptr=None,
    catchment_sea_level_idx_ptr=None,
    num_sea_level_boundaries: tl.constexpr = 0,
    HAS_SEA_LEVEL: tl.constexpr = False,
):
    pid_x = tl.program_id(0)
    idx = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < num_catchments * ensemble_size
    time_step = tl.load(time_step_ptr)
    
    catchment_idx = idx % num_catchments
    
    #----------------------------------------------------------------------
    # (1) Load previous time step input variables
    #----------------------------------------------------------------------
    # Topology is never batched
    downstream_idx = tl.load(downstream_idx_ptr + catchment_idx, mask=mask, other=0)
    is_river_mouth = downstream_idx == catchment_idx

    # river variables
    river_outflow = tl.load(river_outflow_ptr + idx, mask=mask, other=0.0)
    river_manning = tl.load(river_manning_ptr + (idx if batched_river_manning else catchment_idx), mask=mask, other=1.0)
    river_depth = tl.load(river_depth_ptr + idx, mask=mask, other=0.0)
    river_width = tl.load(river_width_ptr + (idx if batched_river_width else catchment_idx), mask=mask, other=1.0)
    river_length = tl.load(river_length_ptr + (idx if batched_river_length else catchment_idx), mask=mask, other=1.0)
    river_height = tl.load(river_height_ptr + (idx if batched_river_height else catchment_idx), mask=mask, other=0.0)
    river_storage = tl.load(river_storage_ptr + idx, mask=mask, other=0.0)

    # flood variables
    flood_outflow = tl.load(flood_outflow_ptr + idx, mask=mask, other=0.0)
    flood_manning = tl.load(flood_manning_ptr + (idx if batched_flood_manning else catchment_idx), mask=mask, other=1.0)
    flood_depth = tl.load(flood_depth_ptr + idx, mask=mask, other=0.0)
    catchment_elevation = tl.load(catchment_elevation_ptr + (idx if batched_catchment_elevation else catchment_idx), mask=mask, other=0.0)
    downstream_distance = tl.load(
        downstream_distance_ptr
        + (idx if batched_downstream_distance else catchment_idx),
        mask=mask,
        other=1.0,
    )
    flood_storage = tl.load(flood_storage_ptr + idx, mask=mask, other=0.0)

    # cross section variables
    river_cross_section_depth = tl.load(river_cross_section_depth_ptr + idx, mask=mask, other=0.0)
    flood_cross_section_depth = tl.load(flood_cross_section_depth_ptr + idx, mask=mask, other=0.0)
    flood_cross_section_area = tl.load(flood_cross_section_area_ptr + idx, mask=mask, other=0.0)

    storage_sum = river_storage + flood_storage
    if HAS_LEVEE:
        storage_sum += tl.load(protected_storage_ptr + idx, mask=mask, other=0.0)
    total_storage = hm.to_compute(storage_sum, river_outflow)
    river_storage = hm.to_compute(river_storage, river_outflow)
    flood_storage = hm.to_compute(flood_storage, river_outflow)

    #----------------------------------------------------------------------
    # (2) Compute current river water surface elevation & downstream water surface elevation
    #----------------------------------------------------------------------
    river_elevation = catchment_elevation - river_height
    water_surface_elevation = river_depth + river_elevation
    
    # Downstream water surface elevation
    member_offset = (idx // num_catchments) * num_catchments
    downstream_idx_global = member_offset + downstream_idx
    
    river_depth_downstream = tl.load(river_depth_ptr + downstream_idx_global, mask=mask, other=0.0)
    river_height_downstream = tl.load(river_height_ptr + (downstream_idx_global if batched_river_height else downstream_idx), mask=mask, other=0.0)
    catchment_elevation_downstream = tl.load(catchment_elevation_ptr + (downstream_idx_global if batched_catchment_elevation else downstream_idx), mask=mask, other=0.0)
    river_elevation_downstream = catchment_elevation_downstream - river_height_downstream
    water_surface_elevation_downstream = river_depth_downstream + river_elevation_downstream
    
    water_surface_elevation_downstream = tl.where(is_river_mouth, catchment_elevation, water_surface_elevation_downstream)
    if HAS_SEA_LEVEL:
        sea_level_idx = tl.load(
            catchment_sea_level_idx_ptr + catchment_idx, mask=mask, other=-1,
        )
        sea_member_offset = (
            (idx // num_catchments) * num_sea_level_boundaries
            if batched_sea_surface_elevation else 0
        )
        prescribed_level = tl.load(
            sea_surface_elevation_ptr + sea_member_offset + sea_level_idx,
            mask=mask & (sea_level_idx >= 0), other=0.0,
        )
        water_surface_elevation_downstream = tl.where(
            sea_level_idx >= 0, prescribed_level,
            water_surface_elevation_downstream,
        )
    max_water_surface_elevation = tl.maximum(
        water_surface_elevation, water_surface_elevation_downstream,
    )
    
    #----------------------------------------------------------------------
    # (4) Longitudinal water surface slope & truncated flood slope
    #----------------------------------------------------------------------
    river_slope = hm.divide(water_surface_elevation - water_surface_elevation_downstream, downstream_distance)
    flood_slope = hm.clamp(river_slope, -ROUTING_SLOPE_LIMIT, ROUTING_SLOPE_LIMIT)

    #----------------------------------------------------------------------
    # (5) Current river/flood cross-section depth + semi-implicit flow depth
    #----------------------------------------------------------------------
    updated_river_cross_section_depth = tl.where(
        is_river_mouth,
        river_depth,
        max_water_surface_elevation - river_elevation,
    )
    river_semi_implicit_flow_depth = hm.at_least(hm.sqrt(
        updated_river_cross_section_depth * river_cross_section_depth
    ), 1e-6)

    flood_cross_section_surface = tl.where(
        is_river_mouth,
        water_surface_elevation,
        max_water_surface_elevation,
    )
    updated_flood_cross_section_depth = tl.maximum(
        flood_cross_section_surface - catchment_elevation,
        0.0
    )
    flood_semi_implicit_flow_depth = hm.at_least(
        hm.sqrt(updated_flood_cross_section_depth * flood_cross_section_depth),
        1e-6,
    )

    #----------------------------------------------------------------------
    # (6) Current flood area (approximate) & semi-implicit effective area
    #----------------------------------------------------------------------
    updated_flood_cross_section_area = tl.maximum(
        hm.divide(flood_storage, river_length) - flood_depth * river_width,
        0.0
    )
    flood_implicit_area = hm.at_least(hm.sqrt(
        updated_flood_cross_section_area * hm.at_least(flood_cross_section_area, 1e-6)
    ), 1e-6)

    #----------------------------------------------------------------------
    # (7) Update river outflow
    #----------------------------------------------------------------------
    river_cross_section_area = updated_river_cross_section_depth * river_width
    flow_threshold = hm.constant(1e-5, river_slope)
    river_condition = (river_semi_implicit_flow_depth > flow_threshold) & (river_cross_section_area > flow_threshold)

    # Original river outflow (per unit width)
    unit_river_outflow = hm.divide(river_outflow, river_width)

    numerator_river = river_width * (
        unit_river_outflow + gravity * time_step 
        * river_semi_implicit_flow_depth * river_slope
    )
    
    # Use libdevice.pow() for power calculation
    denominator_river = 1.0 + gravity * time_step * (river_manning * river_manning) * tl.abs(unit_river_outflow) \
                      * hm.divide(1.0, river_semi_implicit_flow_depth * river_semi_implicit_flow_depth * hm.cbrt(river_semi_implicit_flow_depth))

    updated_river_outflow = hm.divide(numerator_river, denominator_river)
    updated_river_outflow = tl.where(river_condition, updated_river_outflow, 0.0)

    #----------------------------------------------------------------------
    # (8) Update flood outflow
    #----------------------------------------------------------------------
    flood_condition = (flood_semi_implicit_flow_depth > flow_threshold) & (updated_flood_cross_section_area > flow_threshold)

    numerator_flood = flood_outflow + gravity * time_step * flood_implicit_area * flood_slope
    
    # Use libdevice.pow() for power calculation
    denominator_flood = 1.0 + hm.divide(
        gravity * time_step * (flood_manning * flood_manning) * tl.abs(flood_outflow)
        * hm.divide(1.0, flood_semi_implicit_flow_depth * hm.cbrt(flood_semi_implicit_flow_depth)),
        flood_implicit_area,
    )
                      
    updated_flood_outflow = hm.divide(numerator_flood, denominator_flood)
    updated_flood_outflow = tl.where(flood_condition, updated_flood_outflow, 0.0)

    #----------------------------------------------------------------------
    # (9) Flood-direction mask and storage-change limiter
    #----------------------------------------------------------------------
    # Floodplain flow only moves with the river flow (rivout*fldout > 0).
    same_direction = (updated_river_outflow * updated_flood_outflow) > 0.0
    updated_flood_outflow = tl.where(same_direction, updated_flood_outflow, 0.0)
    # v4.23 storage-change limiter on every non-mouth cell: flow towards the
    # upstream cell removes at most 5% of the storage per step.
    backflow = hm.at_least(
        (-updated_river_outflow - updated_flood_outflow) * time_step,
        BACKFLOW_VOLUME_FLOOR,
    )
    limit_rate = hm.at_most(
        hm.divide(BACKFLOW_STORAGE_FRACTION * total_storage, backflow), 1.0,
    )
    updated_river_outflow = tl.where(is_river_mouth, updated_river_outflow, updated_river_outflow * limit_rate)
    updated_flood_outflow = tl.where(is_river_mouth, updated_flood_outflow, updated_flood_outflow * limit_rate)

    # Match the shared kernel's kinematic-wave override at dam-upstream cells.
    if HAS_RESERVOIR:
        is_dam_up = tl.load(
            is_dam_upstream_ptr + catchment_idx, mask=mask, other=0,
        ) != 0
        downstream_elevation = tl.load(
            catchment_elevation_ptr
            + (downstream_idx_global if batched_catchment_elevation else downstream_idx),
            mask=mask, other=0.0,
        )
        bed_slope = hm.divide(catchment_elevation - downstream_elevation, downstream_distance)
        bed_slope = hm.at_least(bed_slope, min_kinematic_slope)
        kin_riv_vel = hm.divide(1.0, river_manning) * hm.sqrt(bed_slope) * hm.cbrt(river_depth * river_depth)
        kin_riv = tl.minimum(river_width * river_depth * kin_riv_vel, hm.divide(river_storage, time_step))
        bed_slope_f = hm.at_most(bed_slope, ROUTING_SLOPE_LIMIT)
        kin_fld_vel = hm.divide(1.0, flood_manning) * hm.sqrt(bed_slope_f) * hm.cbrt(flood_depth * flood_depth)
        kin_fld_area = tl.maximum(
            hm.divide(flood_storage, river_length) - flood_depth * river_width, 0.0,
        )
        kin_fld = tl.minimum(kin_fld_area * kin_fld_vel, hm.divide(flood_storage, time_step))
        updated_river_outflow = tl.where(is_dam_up, kin_riv, updated_river_outflow)
        updated_flood_outflow = tl.where(is_dam_up, kin_fld, updated_flood_outflow)

    #----------------------------------------------------------------------
    # (10) Store results - in-place update
    #----------------------------------------------------------------------
    tl.store(river_outflow_ptr + idx, updated_river_outflow, mask=mask)
    tl.store(flood_outflow_ptr + idx, updated_flood_outflow, mask=mask)
    tl.store(river_cross_section_depth_ptr + idx, updated_river_cross_section_depth, mask=mask)
    tl.store(flood_cross_section_depth_ptr + idx, updated_flood_cross_section_depth, mask=mask)
    # Next step's DARE_pr uses D2FLDDPH_PRE = max(D2RIVDPH_PRE - D2RIVHGT, 0).
    previous_flood_area = tl.maximum(
        hm.divide(flood_storage, river_length)
        - tl.maximum(river_depth - river_height, 0.0) * river_width,
        0.0,
    )
    tl.store(flood_cross_section_area_ptr + idx, previous_flood_area, mask=mask)
    
    tl.store(river_inflow_ptr + idx, 0.0, mask=mask)
    tl.store(flood_inflow_ptr + idx, 0.0, mask=mask)
    if HAS_BIFURCATION:
        tl.store(global_bifurcation_outflow_ptr + idx, 0.0, mask=mask)

    #----------------------------------------------------------------------
    # (11) Fused outgoing storage computation (was compute_outgoing_storage_kernel)
    #----------------------------------------------------------------------
    own_flow, reversed_flow = outgoing_flows_inline(
        updated_river_outflow, updated_flood_outflow,
        outgoing_storage_ptr.dtype.element_ty,
    )
    tl.atomic_add(outgoing_storage_ptr + idx, own_flow, mask=mask, sem="relaxed")
    tl.atomic_add(outgoing_storage_ptr + downstream_idx_global, reversed_flow, mask=mask & ~is_river_mouth, sem="relaxed")


@triton.jit
def compute_inflow_batched_kernel(
    downstream_idx_ptr,            # *i32: Downstream indices
    river_outflow_ptr,             # *f32: River outflow (in/out)
    flood_outflow_ptr,             # *f32: Flood outflow (in/out)
    river_storage_ptr,             # *f64: River storage (rivsto)
    flood_storage_ptr,             # *f64: Flood storage (fldsto)
    outgoing_storage_ptr,          # *f64: Outgoing flow sum (P2STOOUT / step)
    time_step_ptr,                 # *f32: Time step
    river_inflow_ptr,              # *f64: River inflow (output, atomic add)
    flood_inflow_ptr,              # *f64: Flood inflow (output, atomic add)
    limit_rate_ptr,                # *f32: Limit rate diagnostic
    reservoir_total_inflow_ptr,    # *f64: Reservoir total inflow (catchment-sized, atomic add)
    is_reservoir_ptr,              # *i1:  Boolean mask for reservoir catchments
    num_catchments: tl.constexpr,  # Total number of units
    ensemble_size: tl.constexpr,
    HAS_BIFURCATION: tl.constexpr, # Whether bifurcation module is active
    HAS_RESERVOIR: tl.constexpr,   # Whether reservoir module is active
    BLOCK_SIZE: tl.constexpr       # Block size
):
    pid_x = tl.program_id(0)
    idx = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < num_catchments * ensemble_size
    
    catchment_idx = idx % num_catchments
    
    # -------- Load for limiting --------
    river_outflow   = tl.load(river_outflow_ptr      + idx, mask=mask, other=0.0)
    flood_outflow   = tl.load(flood_outflow_ptr      + idx, mask=mask, other=0.0)

    # CaMa-Flood v4.23 supply-side limiter, as in compute_inflow_kernel.
    step = tl.load(time_step_ptr).to(outgoing_storage_ptr.dtype.element_ty)
    limit_rate = supply_rate_inline(
        tl.load(outgoing_storage_ptr + idx, mask=mask, other=0.0),
        tl.load(river_storage_ptr + idx, mask=mask, other=0.0)
        + tl.load(flood_storage_ptr + idx, mask=mask, other=0.0),
        step, river_outflow,
    )
    # Topology is never batched
    downstream_idx   = tl.load(downstream_idx_ptr        + catchment_idx, mask=mask, other=0)
    member_offset = (idx // num_catchments) * num_catchments
    downstream_idx_global = member_offset + downstream_idx
    limit_rate_downstream = supply_rate_inline(
        tl.load(outgoing_storage_ptr + downstream_idx_global, mask=mask, other=0.0),
        tl.load(river_storage_ptr + downstream_idx_global, mask=mask, other=0.0)
        + tl.load(flood_storage_ptr + downstream_idx_global, mask=mask, other=0.0),
        step, river_outflow,
    )
    rate = tl.where(river_outflow > 0.0, limit_rate, limit_rate_downstream)
    updated_river_outflow = river_outflow * rate
    updated_flood_outflow = flood_outflow * rate

    # Write back limited values
    tl.store(river_outflow_ptr + idx, updated_river_outflow, mask=mask)
    tl.store(flood_outflow_ptr + idx, updated_flood_outflow, mask=mask)
    if HAS_BIFURCATION:
        tl.store(limit_rate_ptr + idx, limit_rate, mask=mask)

    # -------- Accumulate inflows --------
    is_river_mouth = downstream_idx == catchment_idx
    not_mouth = mask & (~is_river_mouth)
    tl.atomic_add(river_inflow_ptr + downstream_idx_global, updated_river_outflow, mask=not_mouth, sem="relaxed")
    tl.atomic_add(flood_inflow_ptr + downstream_idx_global, updated_flood_outflow, mask=not_mouth, sem="relaxed")

    # -------- Accumulate reservoir inflow --------
    if HAS_RESERVOIR:
        is_downstream_res = tl.load(is_reservoir_ptr + downstream_idx, mask=not_mouth, other=0) != 0
        reservoir_dtype = reservoir_total_inflow_ptr.dtype.element_ty
        total_outflow = updated_river_outflow.to(reservoir_dtype) + updated_flood_outflow.to(reservoir_dtype)
        tl.atomic_add(reservoir_total_inflow_ptr + downstream_idx_global, total_outflow, mask=not_mouth & is_downstream_res, sem="relaxed")
