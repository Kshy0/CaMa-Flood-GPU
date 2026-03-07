# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

import triton
import triton.language as tl

from cmfgpu.phys.triton.utils import to_compute_dtype


# -----------------------------------------------------------------------------
# Kernel: Compute flood stage, river/flood storage, depths, and flood fraction
# -----------------------------------------------------------------------------
@triton.jit
def compute_flood_stage_kernel(
    # Storage update pointers
    river_inflow_ptr,            # *f64: River inflow (in/out, return to zero)
    flood_inflow_ptr,            # *f64: Flood inflow  (in/out, return to zero)
    river_outflow_ptr,           # *f32: River outflow
    flood_outflow_ptr,           # *f32: Flood outflow
    global_bifurcation_outflow_ptr, # *f64: Global bifurcation outflow
    runoff_ptr,                  # *f32: External runoff
    time_step_ptr,                   # f32: Time step
    # Storage and output pointers
    outgoing_storage_ptr,           # *f64: outgoing storage (out, return to zero)
    river_storage_ptr,           # *f64: River storage (in/out)
    flood_storage_ptr,           # *f64: Flood storage (in/out)
    protected_storage_ptr,       # *f64: Protected storage (in/out)
    river_depth_ptr,             # *f32: River depth (out)
    flood_depth_ptr,             # *f32: Flood depth (out)
    protected_depth_ptr,         # *f32: Protected depth (out)
    flood_fraction_ptr,          # *f32: Flood fraction (out)
    # Reference/lookup table pointers
    river_height_ptr,            # *f32: River height
    flood_depth_table_ptr,       # *f32: Lookup table - flood depth
    catchment_area_ptr,         # *f32: Catchment area
    river_width_ptr,             # *f32: River width
    river_length_ptr,            # *f32: River length
    # Constants
    num_catchments: tl.constexpr,
    num_flood_levels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_BIFURCATION: tl.constexpr = True,   # whether bifurcation module is active
):
    # --- Block and lane indexing ---
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_catchments
    time_step = tl.load(time_step_ptr)

    # ---- 1. Storage update (from update_storage_kernel) ----
    river_storage = tl.load(river_storage_ptr + offs, mask=mask, other=0.0)
    flood_storage = tl.load(flood_storage_ptr + offs, mask=mask, other=0.0)
    protected_storage = tl.load(protected_storage_ptr + offs, mask=mask, other=0.0)
    river_inflow = tl.load(river_inflow_ptr + offs, mask=mask, other=0.0)
    flood_inflow = tl.load(flood_inflow_ptr + offs, mask=mask, other=0.0)
    river_outflow = tl.load(river_outflow_ptr + offs, mask=mask, other=0.0)
    flood_outflow = tl.load(flood_outflow_ptr + offs, mask=mask, other=0.0)
    if HAS_BIFURCATION:
        global_bifurcation_outflow = tl.load(global_bifurcation_outflow_ptr + offs, mask=mask, other=0.0)
    runoff = tl.load(runoff_ptr + offs, mask=mask, other=0.0)

    # Downcast hpfloat to computation dtype (Fortran: D2RIVINF=REAL(P2RIVINF, KIND=JPRB))
    river_inflow = to_compute_dtype(river_inflow, river_outflow)
    flood_inflow = to_compute_dtype(flood_inflow, river_outflow)
    if HAS_BIFURCATION:
        global_bifurcation_outflow = to_compute_dtype(global_bifurcation_outflow, river_outflow)

    river_storage_updated = river_storage + (river_inflow - river_outflow) * time_step
    flood_storage_updated = flood_storage + tl.where(river_storage_updated < 0.0, river_storage_updated, 0.0) + (flood_inflow - flood_outflow - (global_bifurcation_outflow if HAS_BIFURCATION else 0.0)) * time_step
    river_storage_updated = tl.maximum(river_storage_updated, 0.0)
    river_storage_updated = tl.where(
        flood_storage_updated < 0.0,
        tl.maximum(river_storage_updated + flood_storage_updated, 0.0),
        river_storage_updated
    )
    flood_storage_updated = tl.maximum(flood_storage_updated, 0.0)
    total_storage = tl.maximum(river_storage_updated + flood_storage_updated + protected_storage + runoff * time_step, 0.0)

    # Downcast total_storage to computation dtype for flood stage (Fortran: D2STORGE is JPRB)
    total_storage = to_compute_dtype(total_storage, river_outflow)

    # ---- 2. Flood stage computation (from original compute_flood_stage_kernel) ----
    river_height        = tl.load(river_height_ptr        + offs, mask=mask)
    catchment_area      = tl.load(catchment_area_ptr      + offs, mask=mask)
    river_width         = tl.load(river_width_ptr         + offs, mask=mask)
    river_length        = tl.load(river_length_ptr        + offs, mask=mask)
    
    river_max_storage = river_length * river_width * river_height
    catchment_width = catchment_area / river_length
    width_increment = catchment_width / num_flood_levels

    # Determine flood level by scanning the storage table
    level = tl.where(total_storage > river_max_storage, 0, -1).to(tl.int32)
    
    S_accum = river_max_storage
    prev_H = 0.0
    prev_W = river_width
    
    prev_total_storage = river_max_storage
    prev_flood_depth = 0.0
    next_flood_depth = 0.0
    
    for i in tl.static_range(num_flood_levels):
        H_curr = tl.load(flood_depth_table_ptr + offs * num_flood_levels + i, mask=mask)
        W_curr = river_width + (i + 1) * width_increment
        dS = river_length * 0.5 * (prev_W + W_curr) * (H_curr - prev_H)
        S_curr = S_accum + dS
        
        next_flood_depth = tl.where(level == i, H_curr, next_flood_depth)
        
        is_above = total_storage > S_curr
        level += tl.where(is_above, 1, 0)
        prev_total_storage = tl.where(is_above, S_curr, prev_total_storage)
        prev_flood_depth = tl.where(is_above, H_curr, prev_flood_depth)

        S_accum = S_curr
        prev_H = H_curr
        prev_W = W_curr

    no_flood_cond = level < 0
    level = tl.maximum(level, 0)
    
    prev_total_width = river_width + level * width_increment

    flood_grad = tl.where(
        level == num_flood_levels,
        0.0,
        (next_flood_depth - prev_flood_depth) / width_increment
    )

    diff_width = tl.sqrt(
        prev_total_width * prev_total_width +
        2.0 * (total_storage - prev_total_storage) / (flood_grad * river_length)
    ) - prev_total_width
    flood_depth_if_mid = prev_flood_depth + diff_width * flood_grad
    flood_depth_if_top = prev_flood_depth + (total_storage - prev_total_storage) / (prev_total_width * river_length)

    flood_depth = tl.where(
        no_flood_cond, 0.0,
        tl.where(level == num_flood_levels, flood_depth_if_top, flood_depth_if_mid)
    )

    river_storage_final = tl.where(
        no_flood_cond,
        total_storage,
        tl.minimum(river_max_storage + river_length * river_width * flood_depth, total_storage)
    )
    river_depth = river_storage_final / (river_length * river_width)


    flood_fraction_mid = tl.clamp((prev_total_width + diff_width - river_width) * river_length / catchment_area, 0.0, 1.0)
    flood_fraction = tl.where(
        no_flood_cond, 0.0,
        tl.where(level == num_flood_levels, 1.0, flood_fraction_mid)
    )

    flood_storage_final = tl.maximum(total_storage - river_storage_final, 0.0)

    # Return to zero
    tl.store(outgoing_storage_ptr + offs, 0.0, mask=mask)

    # Store outputs (in-place update)
    tl.store(river_storage_ptr    + offs, river_storage_final, mask=mask)
    tl.store(flood_storage_ptr    + offs, flood_storage_final, mask=mask)
    tl.store(protected_storage_ptr + offs, 0.0, mask=mask)
    tl.store(river_depth_ptr      + offs, river_depth, mask=mask)
    tl.store(flood_depth_ptr      + offs, flood_depth, mask=mask)
    tl.store(protected_depth_ptr  + offs, flood_depth, mask=mask)
    tl.store(flood_fraction_ptr   + offs, flood_fraction, mask=mask)
    

@triton.jit
def compute_flood_stage_log_kernel(
    # Storage update pointers
    river_inflow_ptr,            # *f64: River inflow (in/out, return to zero)
    flood_inflow_ptr,            # *f64: Flood inflow  (in/out, return to zero)
    river_outflow_ptr,           # *f32: River outflow
    flood_outflow_ptr,           # *f32: Flood outflow
    global_bifurcation_outflow_ptr,  # *f64: Global bifurcation outflow
    runoff_ptr,                  # *f32: External runoff
    time_step_ptr,                   # f32: Time step
    # Storage and output pointers
    outgoing_storage_ptr,           # *f64: outgoing storage (out, return to zero)
    river_storage_ptr,           # *f64: River storage (in/out)
    flood_storage_ptr,           # *f64: Flood storage (in/out)
    protected_storage_ptr,       # *f64: Protected storage (in/out)
    river_depth_ptr,             # *f32: River depth (in/out)
    flood_depth_ptr,             # *f32: Flood depth (in/out)
    protected_depth_ptr,         # *f32: Protected depth (in/out)
    flood_fraction_ptr,          # *f32: Flood fraction (in/out)
    # Reference/lookup table pointers
    river_height_ptr,            # *f32: River height
    flood_depth_table_ptr,       # *f32: Lookup table - flood depth
    catchment_area_ptr,         # *f32: Catchment area
    river_width_ptr,             # *f32: River width
    river_length_ptr,            # *f32: River length
    is_levee_ptr,                # *bool: Boolean mask for catchments governed by levee physics
    # Packed log sums: (NUM_LOG_VARS, log_buffer_size) contiguous tensor
    log_sums_ptr,                # *f32: packed log accumulator
    # Constants
    current_step_ptr,
    num_catchments: tl.constexpr,
    num_flood_levels: tl.constexpr,
    log_buffer_size: tl.constexpr = 1000,
    BLOCK_SIZE: tl.constexpr = 128,
    HAS_BIFURCATION: tl.constexpr = True,   # whether bifurcation module is active
):
    # --- Block and lane indexing ---
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_catchments
    time_step = tl.load(time_step_ptr)
    current_step = tl.load(current_step_ptr)

    is_levee = tl.load(is_levee_ptr + offs, mask=mask, other=True)
    non_levee = ~is_levee

    # ---- 1. Storage update (from update_storage_kernel) ----
    river_storage = tl.load(river_storage_ptr + offs, mask=mask, other=0.0)
    flood_storage = tl.load(flood_storage_ptr + offs, mask=mask, other=0.0)
    protected_storage = tl.load(protected_storage_ptr + offs, mask=mask, other=0.0)
    river_inflow = tl.load(river_inflow_ptr + offs, mask=mask, other=0.0)
    flood_inflow = tl.load(flood_inflow_ptr + offs, mask=mask, other=0.0)
    river_outflow = tl.load(river_outflow_ptr + offs, mask=mask, other=0.0)
    flood_outflow = tl.load(flood_outflow_ptr + offs, mask=mask, other=0.0)
    runoff = tl.load(runoff_ptr + offs, mask=mask, other=0.0)
    if HAS_BIFURCATION:
        global_bifurcation_outflow = tl.load(global_bifurcation_outflow_ptr + offs, mask=mask, other=0.0)
    total_stage_pre = river_storage + flood_storage + protected_storage
    # log_sums layout: row 0=pre, 1=next, 2=new, 3=inflow_err, 4=inflow, 5=outflow,
    #                  6=stage, 7=stage_err, 8=riv_sto, 9=fld_sto, 10=fld_area
    tl.atomic_add(log_sums_ptr + 0 * log_buffer_size + current_step, tl.sum(total_stage_pre) * 1e-9)

    # Downcast hpfloat to computation dtype (Fortran: D2RIVINF=REAL(P2RIVINF, KIND=JPRB))
    river_inflow = to_compute_dtype(river_inflow, river_outflow)
    flood_inflow = to_compute_dtype(flood_inflow, river_outflow)
    if HAS_BIFURCATION:
        global_bifurcation_outflow = to_compute_dtype(global_bifurcation_outflow, river_outflow)

    river_storage_updated = river_storage + (river_inflow - river_outflow) * time_step
    flood_storage_updated = flood_storage + tl.where(river_storage_updated < 0.0, river_storage_updated, 0.0) + (flood_inflow - flood_outflow - (global_bifurcation_outflow if HAS_BIFURCATION else 0.0)) * time_step
    river_storage_updated = tl.maximum(river_storage_updated, 0.0)
    river_storage_updated = tl.where(
        flood_storage_updated < 0.0,
        tl.maximum(river_storage_updated + flood_storage_updated, 0.0),
        river_storage_updated
    )
    flood_storage_updated = tl.maximum(flood_storage_updated, 0.0)
    total_storage_next = river_storage_updated + flood_storage_updated + protected_storage + runoff * time_step
    tl.atomic_add(log_sums_ptr + 1 * log_buffer_size + current_step, tl.sum(tl.where(non_levee, total_storage_next, 0)) * 1e-9)
    total_storage = tl.maximum(river_storage_updated + flood_storage_updated + protected_storage + runoff * time_step, 0.0)
    tl.atomic_add(log_sums_ptr + 2 * log_buffer_size + current_step, tl.sum(tl.where(non_levee, total_storage, 0)) * 1e-9)
    tl.atomic_add(log_sums_ptr + 4 * log_buffer_size + current_step, tl.sum(tl.where(non_levee, (river_inflow + flood_inflow) * time_step, 0)) * 1e-9)
    tl.atomic_add(log_sums_ptr + 5 * log_buffer_size + current_step, tl.sum(tl.where(non_levee, (river_outflow + flood_outflow) * time_step, 0)) * 1e-9)
    tl.atomic_add(log_sums_ptr + 3 * log_buffer_size + current_step, tl.sum(tl.where(non_levee, total_stage_pre - total_storage_next + (river_inflow + flood_inflow + runoff - river_outflow - flood_outflow - (global_bifurcation_outflow if HAS_BIFURCATION else 0.0)) * time_step, 0)) * 1e-9)

    # Downcast total_storage to computation dtype for flood stage (Fortran: D2STORGE is JPRB)
    total_storage = to_compute_dtype(total_storage, river_outflow)

    # ---- 2. Flood stage computation (from original compute_flood_stage_kernel) ----
    river_height        = tl.load(river_height_ptr        + offs, mask=mask)
    catchment_area     = tl.load(catchment_area_ptr     + offs, mask=mask)
    river_width         = tl.load(river_width_ptr         + offs, mask=mask)
    river_length        = tl.load(river_length_ptr        + offs, mask=mask)
    
    river_max_storage = river_length * river_width * river_height
    catchment_width = catchment_area / river_length
    width_increment = catchment_width / num_flood_levels

    # Determine flood level by scanning the storage table
    level = tl.where(total_storage > river_max_storage, 0, -1).to(tl.int32)
    
    S_accum = river_max_storage
    prev_H = 0.0
    prev_W = river_width
    
    prev_total_storage = river_max_storage
    prev_flood_depth = 0.0
    next_flood_depth = 0.0
    
    for i in tl.static_range(num_flood_levels):
        H_curr = tl.load(flood_depth_table_ptr + offs * num_flood_levels + i, mask=mask)
        W_curr = river_width + (i + 1) * width_increment
        dS = river_length * 0.5 * (prev_W + W_curr) * (H_curr - prev_H)
        S_curr = S_accum + dS
        
        next_flood_depth = tl.where(level == i, H_curr, next_flood_depth)
        
        is_above = total_storage > S_curr
        level += tl.where(is_above, 1, 0)
        prev_total_storage = tl.where(is_above, S_curr, prev_total_storage)
        prev_flood_depth = tl.where(is_above, H_curr, prev_flood_depth)

        S_accum = S_curr
        prev_H = H_curr
        prev_W = W_curr

    no_flood_cond = level < 0
    level = tl.maximum(level, 0)
    
    prev_total_width = river_width + level * width_increment

    flood_grad = tl.where(
        level == num_flood_levels,
        0.0,
        (next_flood_depth - prev_flood_depth) / width_increment
    )
    
    diff_width = tl.sqrt(
        prev_total_width * prev_total_width +
        2.0 * (total_storage - prev_total_storage) / (flood_grad * river_length)
    ) - prev_total_width
    flood_depth_if_mid = prev_flood_depth + diff_width * flood_grad
    flood_depth_if_top = prev_flood_depth + (total_storage - prev_total_storage) / (prev_total_width * river_length)

    flood_depth = tl.where(
        no_flood_cond, 0.0,
        tl.where(level == num_flood_levels, flood_depth_if_top, flood_depth_if_mid)
    )

    river_storage_final = tl.where(
        no_flood_cond,
        total_storage,
        tl.minimum(river_max_storage + river_length * river_width * flood_depth, total_storage)
    )
    river_depth = river_storage_final / (river_length * river_width)

    flood_fraction_mid = tl.clamp((prev_total_width + diff_width - river_width) * river_length / catchment_area, 0.0, 1.0)
    flood_fraction = tl.where(
        no_flood_cond, 0.0,
        tl.where(level == num_flood_levels, 1.0, flood_fraction_mid)
    )

    flood_area    = flood_fraction * catchment_area
    flood_storage_final = tl.maximum(total_storage - river_storage_final, 0.0)

    # log
    total_storage_stage_new = river_storage_final + flood_storage_final
    tl.atomic_add(log_sums_ptr + 6 * log_buffer_size + current_step, tl.sum(total_storage_stage_new) * 1e-9)
    tl.atomic_add(log_sums_ptr + 8 * log_buffer_size + current_step, tl.sum(tl.where(non_levee, river_storage_final, 0)) * 1e-9)
    tl.atomic_add(log_sums_ptr + 9 * log_buffer_size + current_step, tl.sum(tl.where(non_levee, flood_storage_final, 0)) * 1e-9)
    tl.atomic_add(log_sums_ptr + 10 * log_buffer_size + current_step, tl.sum(tl.where(non_levee, flood_area, 0)) * 1e-9)
    tl.atomic_add(log_sums_ptr + 7 * log_buffer_size + current_step, tl.sum(tl.where(non_levee, (total_storage_stage_new - total_storage) * 1e-9, 0)))
    # Return to zero
    tl.store(outgoing_storage_ptr + offs, 0.0, mask=mask)

    # Store outputs (in-place update)
    tl.store(river_storage_ptr    + offs, river_storage_final, mask=mask)
    tl.store(flood_storage_ptr    + offs, flood_storage_final, mask=mask)
    tl.store(protected_storage_ptr + offs, 0.0, mask=mask)
    tl.store(river_depth_ptr      + offs, river_depth, mask=mask)
    tl.store(flood_depth_ptr      + offs, flood_depth, mask=mask)
    tl.store(protected_depth_ptr  + offs, flood_depth, mask=mask)
    tl.store(flood_fraction_ptr   + offs, flood_fraction, mask=mask)


@triton.jit
def compute_flood_stage_batched_kernel(
    # Storage update pointers
    river_inflow_ptr,            # *f64: River inflow (in/out, return to zero)
    flood_inflow_ptr,            # *f64: Flood inflow  (in/out, return to zero)
    river_outflow_ptr,           # *f32: River outflow
    flood_outflow_ptr,           # *f32: Flood outflow
    global_bifurcation_outflow_ptr, # *f64: Global bifurcation outflow
    runoff_ptr,                  # *f32: External runoff
    time_step_ptr,                   # f32: Time step
    # Storage and output pointers
    outgoing_storage_ptr,           # *f64: outgoing storage (out, return to zero)
    river_storage_ptr,           # *f64: River storage (in/out)
    flood_storage_ptr,           # *f64: Flood storage (in/out)
    protected_storage_ptr,       # *f64: Protected storage (in/out)
    river_depth_ptr,             # *f32: River depth (in/out)
    flood_depth_ptr,             # *f32: Flood depth (in/out)
    protected_depth_ptr,         # *f32: Protected depth (in/out)
    flood_fraction_ptr,          # *f32: Flood fraction (in/out)
    # Reference/lookup table pointers
    river_height_ptr,            # *f32: River height
    flood_depth_table_ptr,       # *f32: Lookup table - flood depth
    catchment_area_ptr,         # *f32: Catchment area
    river_width_ptr,             # *f32: River width
    river_length_ptr,            # *f32: River length
    # Constants
    num_catchments: tl.constexpr,
    num_flood_levels: tl.constexpr,
    num_trials: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    # Batch flags
    batched_runoff: tl.constexpr,
    batched_river_height: tl.constexpr,
    batched_flood_depth_table: tl.constexpr,
    batched_catchment_area: tl.constexpr,
    batched_river_width: tl.constexpr,
    batched_river_length: tl.constexpr,
    HAS_BIFURCATION: tl.constexpr = True,   # whether bifurcation module is active
):
    # --- Loop-based batched kernel ---
    # Grid = cdiv(num_catchments, BLOCK_SIZE), each block loops over trials.
    # Shared (non-trial) parameters are loaded once and reused across trials,
    # reducing memory traffic and register pressure.
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_catchments
    time_step = tl.load(time_step_ptr)

    # ---- Load shared (non-trial) parameters once ----
    if not batched_river_height:
        river_height_shared = tl.load(river_height_ptr + offs, mask=mask)
    if not batched_catchment_area:
        catchment_area_shared = tl.load(catchment_area_ptr + offs, mask=mask)
    if not batched_river_width:
        river_width_shared = tl.load(river_width_ptr + offs, mask=mask)
    if not batched_river_length:
        river_length_shared = tl.load(river_length_ptr + offs, mask=mask)
    if not batched_runoff:
        runoff_shared = tl.load(runoff_ptr + offs, mask=mask, other=0.0)

    # Pre-compute derived constants that don't change across trials
    if not batched_river_height and not batched_river_width and not batched_river_length:
        river_max_storage_shared = river_length_shared * river_width_shared * river_height_shared
    if not batched_catchment_area and not batched_river_length:
        catchment_width_shared = catchment_area_shared / river_length_shared
        width_increment_shared = catchment_width_shared / num_flood_levels

    # ---- Loop over trials ----
    for t in tl.static_range(num_trials):
        trial_offset = t * num_catchments
        idx = trial_offset + offs

        # ---- 1. Storage update ----
        river_storage = tl.load(river_storage_ptr + idx, mask=mask, other=0.0)
        flood_storage = tl.load(flood_storage_ptr + idx, mask=mask, other=0.0)
        protected_storage = tl.load(protected_storage_ptr + idx, mask=mask, other=0.0)
        river_inflow = tl.load(river_inflow_ptr + idx, mask=mask, other=0.0)
        flood_inflow = tl.load(flood_inflow_ptr + idx, mask=mask, other=0.0)
        river_outflow = tl.load(river_outflow_ptr + idx, mask=mask, other=0.0)
        flood_outflow = tl.load(flood_outflow_ptr + idx, mask=mask, other=0.0)
        if HAS_BIFURCATION:
            global_bifurcation_outflow = tl.load(global_bifurcation_outflow_ptr + idx, mask=mask, other=0.0)

        runoff = tl.load(runoff_ptr + idx, mask=mask, other=0.0) if batched_runoff else runoff_shared

        # Downcast hpfloat
        river_inflow = to_compute_dtype(river_inflow, river_outflow)
        flood_inflow = to_compute_dtype(flood_inflow, river_outflow)
        if HAS_BIFURCATION:
            global_bifurcation_outflow = to_compute_dtype(global_bifurcation_outflow, river_outflow)

        river_storage_updated = river_storage + (river_inflow - river_outflow) * time_step
        flood_storage_updated = flood_storage + tl.where(river_storage_updated < 0.0, river_storage_updated, 0.0) + (flood_inflow - flood_outflow - (global_bifurcation_outflow if HAS_BIFURCATION else 0.0)) * time_step
        river_storage_updated = tl.maximum(river_storage_updated, 0.0)
        river_storage_updated = tl.where(
            flood_storage_updated < 0.0,
            tl.maximum(river_storage_updated + flood_storage_updated, 0.0),
            river_storage_updated
        )
        flood_storage_updated = tl.maximum(flood_storage_updated, 0.0)
        total_storage = tl.maximum(river_storage_updated + flood_storage_updated + protected_storage + runoff * time_step, 0.0)
        total_storage = to_compute_dtype(total_storage, river_outflow)

        # ---- 2. Flood stage computation ----
        # Use pre-loaded shared values or load per-trial batched values
        river_height = tl.load(river_height_ptr + idx, mask=mask) if batched_river_height else river_height_shared
        catchment_area = tl.load(catchment_area_ptr + idx, mask=mask) if batched_catchment_area else catchment_area_shared
        river_width = tl.load(river_width_ptr + idx, mask=mask) if batched_river_width else river_width_shared
        river_length = tl.load(river_length_ptr + idx, mask=mask) if batched_river_length else river_length_shared

        # Use pre-computed derived constants when possible
        if batched_river_height or batched_river_width or batched_river_length:
            river_max_storage = river_length * river_width * river_height
        else:
            river_max_storage = river_max_storage_shared
        if batched_catchment_area or batched_river_length:
            catchment_width = catchment_area / river_length
            width_increment = catchment_width / num_flood_levels
        else:
            width_increment = width_increment_shared

        level = tl.where(total_storage > river_max_storage, 0, -1).to(tl.int32)

        S_accum = river_max_storage
        prev_H = 0.0
        prev_W = river_width

        prev_total_storage = river_max_storage
        prev_flood_depth = 0.0
        next_flood_depth = 0.0

        # Table offset: when not batched, table_base_offset is 0 and addresses
        # don't depend on trial → Triton compiler can hoist these loads.
        if batched_flood_depth_table:
            table_base_offset = trial_offset * num_flood_levels
        else:
            table_base_offset = 0

        for i in tl.static_range(num_flood_levels):
            H_curr = tl.load(flood_depth_table_ptr + table_base_offset + offs * num_flood_levels + i, mask=mask)
            W_curr = river_width + (i + 1) * width_increment
            dS = river_length * 0.5 * (prev_W + W_curr) * (H_curr - prev_H)
            S_curr = S_accum + dS

            next_flood_depth = tl.where(level == i, H_curr, next_flood_depth)

            is_above = total_storage > S_curr
            level += tl.where(is_above, 1, 0)
            prev_total_storage = tl.where(is_above, S_curr, prev_total_storage)
            prev_flood_depth = tl.where(is_above, H_curr, prev_flood_depth)

            S_accum = S_curr
            prev_H = H_curr
            prev_W = W_curr

        no_flood_cond = level < 0
        level = tl.maximum(level, 0)

        prev_total_width = river_width + level * width_increment

        flood_grad = tl.where(
            level == num_flood_levels,
            0.0,
            (next_flood_depth - prev_flood_depth) / width_increment
        )

        diff_width = tl.sqrt(
            prev_total_width * prev_total_width +
            2.0 * (total_storage - prev_total_storage) / (flood_grad * river_length)
        ) - prev_total_width
        flood_depth_if_mid = prev_flood_depth + diff_width * flood_grad
        flood_depth_if_top = prev_flood_depth + (total_storage - prev_total_storage) / (prev_total_width * river_length)

        flood_depth = tl.where(
            no_flood_cond, 0.0,
            tl.where(level == num_flood_levels, flood_depth_if_top, flood_depth_if_mid)
        )

        river_storage_final = tl.where(
            no_flood_cond,
            total_storage,
            tl.minimum(river_max_storage + river_length * river_width * flood_depth, total_storage)
        )
        river_depth = river_storage_final / (river_length * river_width)

        flood_fraction_mid = tl.clamp((prev_total_width + diff_width - river_width) * river_length / catchment_area, 0.0, 1.0)
        flood_fraction = tl.where(
            no_flood_cond, 0.0,
            tl.where(level == num_flood_levels, 1.0, flood_fraction_mid)
        )

        flood_storage_final = tl.maximum(total_storage - river_storage_final, 0.0)

        # Return to zero
        tl.store(outgoing_storage_ptr + idx, 0.0, mask=mask)

        # Store outputs
        tl.store(river_storage_ptr    + idx, river_storage_final, mask=mask)
        tl.store(flood_storage_ptr    + idx, flood_storage_final, mask=mask)
        tl.store(protected_storage_ptr + idx, 0.0, mask=mask)
        tl.store(river_depth_ptr      + idx, river_depth, mask=mask)
        tl.store(flood_depth_ptr      + idx, flood_depth, mask=mask)
        tl.store(protected_depth_ptr  + idx, flood_depth, mask=mask)
        tl.store(flood_fraction_ptr   + idx, flood_fraction, mask=mask)
