import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Kernel: Compute flood stage, river/flood storage, depths, and flood fraction
# -----------------------------------------------------------------------------
@triton.jit
def compute_flood_stage_kernel(
    # Storage update pointers
    river_inflow_ptr,            # *f32: River inflow (in/out, return to zero)
    flood_inflow_ptr,            # *f32: Flood inflow  (in/out, return to zero)
    river_outflow_ptr,           # *f32: River outflow
    flood_outflow_ptr,           # *f32: Flood outflow
    global_bifurcation_outflow_ptr, # *f32: Global bifurcation outflow
    runoff_ptr,                  # *f32: External runoff
    time_step,                   # f32: Time step
    # Storage and output pointers
    outgoing_storage_ptr,           # *f32: outgoing storage (out, return to zero)
    river_storage_ptr,           # *f32: River storage (in/out)
    flood_storage_ptr,           # *f32: Flood storage (in/out)
    river_depth_ptr,             # *f32: River depth (in/out)
    flood_depth_ptr,             # *f32: Flood depth (in/out)
    flood_fraction_ptr,          # *f32: Flood fraction (in/out)
    flood_area_ptr,              # *f32: Flood area (in/out)
    # Reference/lookup table pointers
    river_max_storage_ptr,       # *f32: River max storage
    total_storage_table_ptr,     # *f32: Lookup table - total storage
    flood_depth_table_ptr,       # *f32: Lookup table - flood depth
    total_width_table_ptr,       # *f32: Lookup table - total width
    flood_gradient_table_ptr,    # *f32: Lookup table - flood gradient
    catchment_area_ptr,          # *f32: Catchment area
    river_width_ptr,             # *f32: River width
    river_length_ptr,            # *f32: River length
    # Constants
    num_catchments: tl.constexpr,
    num_flood_levels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # --- Block and lane indexing ---
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_catchments

    # ---- 1. Storage update (from update_storage_kernel) ----
    river_storage = tl.load(river_storage_ptr + offs, mask=mask, other=0.0)
    flood_storage = tl.load(flood_storage_ptr + offs, mask=mask, other=0.0)
    river_inflow = tl.load(river_inflow_ptr + offs, mask=mask, other=0.0)
    flood_inflow = tl.load(flood_inflow_ptr + offs, mask=mask, other=0.0)
    river_outflow = tl.load(river_outflow_ptr + offs, mask=mask, other=0.0)
    flood_outflow = tl.load(flood_outflow_ptr + offs, mask=mask, other=0.0)
    global_bifurcation_outflow = tl.load(global_bifurcation_outflow_ptr + offs, mask=mask, other=0.0)
    runoff = tl.load(runoff_ptr + offs, mask=mask, other=0.0)

    river_storage_updated = river_storage + (river_inflow - river_outflow) * time_step
    flood_storage_updated = flood_storage + tl.where(river_storage_updated < 0.0, river_storage_updated, 0.0) + (flood_inflow - flood_outflow - global_bifurcation_outflow) * time_step
    river_storage_updated = tl.maximum(river_storage_updated, 0.0)
    river_storage_updated = tl.where(
        flood_storage_updated < 0.0,
        tl.maximum(river_storage_updated + flood_storage_updated, 0.0),
        river_storage_updated
    )
    flood_storage_updated = tl.where(
        flood_storage_updated < 0.0,
        0.0,
        flood_storage_updated
    )
    total_storage = tl.maximum(river_storage_updated + flood_storage_updated + runoff * time_step, 0.0)

    # ---- 2. Flood stage computation (from original compute_flood_stage_kernel) ----
    river_max_storage   = tl.load(river_max_storage_ptr   + offs, mask=mask)
    catchment_area      = tl.load(catchment_area_ptr      + offs, mask=mask)
    river_width         = tl.load(river_width_ptr         + offs, mask=mask)
    river_length        = tl.load(river_length_ptr        + offs, mask=mask)

    # Determine flood level by scanning the storage table
    level = tl.zeros([BLOCK_SIZE], dtype=tl.int32) - 1
    for i in tl.static_range(num_flood_levels+1):
        prev_total_storage = tl.load(total_storage_table_ptr + offs * (num_flood_levels+1) + i, mask=mask)
        level += tl.where(total_storage > prev_total_storage, 1, 0)
    no_flood_cond = level < 0
    prev_flood_depth   = tl.load(flood_depth_table_ptr   + offs * (num_flood_levels+1) + level, mask=mask)
    prev_total_width   = tl.load(total_width_table_ptr   + offs * (num_flood_levels+1) + level, mask=mask)
    prev_total_storage = tl.load(total_storage_table_ptr + offs * (num_flood_levels+1) + level, mask=mask)
    flood_grad         = tl.load(flood_gradient_table_ptr + offs * (num_flood_levels+1) + level, mask=mask)

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

    # Return to zero
    tl.store(river_inflow_ptr + offs, 0.0, mask=mask)
    tl.store(flood_inflow_ptr + offs, 0.0, mask=mask)
    tl.store(outgoing_storage_ptr + offs, 0.0, mask=mask)
    tl.store(global_bifurcation_outflow_ptr + offs, 0.0, mask=mask)

    # Store outputs (in-place update)
    tl.store(river_storage_ptr    + offs, river_storage_final, mask=mask)
    tl.store(flood_storage_ptr    + offs, flood_storage_final, mask=mask)
    tl.store(river_depth_ptr      + offs, river_depth, mask=mask)
    tl.store(flood_depth_ptr      + offs, flood_depth, mask=mask)
    tl.store(flood_fraction_ptr   + offs, flood_fraction, mask=mask)
    tl.store(flood_area_ptr       + offs, flood_area, mask=mask)
    

@triton.jit
def compute_flood_stage_log_kernel(
    # Storage update pointers
    river_inflow_ptr,            # *f32: River inflow (in/out, return to zero)
    flood_inflow_ptr,            # *f32: Flood inflow  (in/out, return to zero)
    river_outflow_ptr,           # *f32: River outflow
    flood_outflow_ptr,           # *f32: Flood outflow
    global_bifurcation_outflow_ptr,  # *f32: Global bifurcation outflow
    runoff_ptr,                  # *f32: External runoff
    time_step,                   # f32: Time step
    # Storage and output pointers
    outgoing_storage_ptr,           # *f32: outgoing storage (out, return to zero)
    river_storage_ptr,           # *f32: River storage (in/out)
    flood_storage_ptr,           # *f32: Flood storage (in/out)
    river_depth_ptr,             # *f32: River depth (in/out)
    flood_depth_ptr,             # *f32: Flood depth (in/out)
    flood_fraction_ptr,          # *f32: Flood fraction (in/out)
    flood_area_ptr,              # *f32: Flood area (in/out)
    # Reference/lookup table pointers
    river_max_storage_ptr,       # *f32: River max storage
    total_storage_table_ptr,     # *f32: Lookup table - total storage
    flood_depth_table_ptr,       # *f32: Lookup table - flood depth
    total_width_table_ptr,       # *f32: Lookup table - total width
    flood_gradient_table_ptr,    # *f32: Lookup table - flood gradient
    catchment_area_ptr,          # *f32: Catchment area
    river_width_ptr,             # *f32: River width
    river_length_ptr,            # *f32: River length
    # scalar for log storage
    total_storage_pre_sum_ptr, # *f32
    total_storage_next_sum_ptr, # *f32
    total_storage_new_ptr, # *f32
    total_inflow_ptr,
    total_outflow_ptr,
    total_storage_stage_new_ptr, # *f32
    total_river_storage_ptr, # *f32
    total_flood_storage_ptr, # *f32
    total_flood_area_ptr, # *f32
    total_inflow_error_ptr, # *f32
    total_stage_error_ptr, # *f32
    # Constants
    current_step,
    num_catchments: tl.constexpr,
    num_flood_levels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # --- Block and lane indexing ---
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_catchments

    # ---- 1. Storage update (from update_storage_kernel) ----
    river_storage = tl.load(river_storage_ptr + offs, mask=mask, other=0.0)
    flood_storage = tl.load(flood_storage_ptr + offs, mask=mask, other=0.0)
    river_inflow = tl.load(river_inflow_ptr + offs, mask=mask, other=0.0)
    flood_inflow = tl.load(flood_inflow_ptr + offs, mask=mask, other=0.0)
    river_outflow = tl.load(river_outflow_ptr + offs, mask=mask, other=0.0)
    flood_outflow = tl.load(flood_outflow_ptr + offs, mask=mask, other=0.0)
    runoff = tl.load(runoff_ptr + offs, mask=mask, other=0.0)
    global_bifurcation_outflow = tl.load(global_bifurcation_outflow_ptr + offs, mask=mask, other=0.0)
    total_stage_pre = river_storage + flood_storage
    tl.atomic_add(total_storage_pre_sum_ptr + current_step, tl.sum(total_stage_pre) * 1e-9)
    river_storage_updated = river_storage + (river_inflow - river_outflow) * time_step
    flood_storage_updated = flood_storage + tl.where(river_storage_updated < 0.0, river_storage_updated, 0.0) + (flood_inflow - flood_outflow - global_bifurcation_outflow) * time_step
    river_storage_updated = tl.maximum(river_storage_updated, 0.0)
    river_storage_updated = tl.where(
        flood_storage_updated < 0.0,
        tl.maximum(river_storage_updated + flood_storage_updated, 0.0),
        river_storage_updated
    )
    flood_storage_updated = tl.where(
        flood_storage_updated < 0.0,
        0.0,
        flood_storage_updated
    )
    total_storage_next = river_storage_updated + flood_storage_updated
    tl.atomic_add(total_storage_next_sum_ptr + current_step, tl.sum(total_storage_next) * 1e-9)
    total_storage = tl.maximum(river_storage_updated + flood_storage_updated + runoff * time_step, 0.0)
    tl.atomic_add(total_storage_new_ptr + current_step, tl.sum(total_storage) * 1e-9)
    tl.atomic_add(total_inflow_ptr + current_step, tl.sum((river_inflow + flood_inflow) * time_step) * 1e-9)
    tl.atomic_add(total_outflow_ptr + current_step, tl.sum((river_outflow + flood_outflow) * time_step) * 1e-9)
    tl.atomic_add(total_inflow_error_ptr + current_step, tl.sum(total_stage_pre - total_storage_next + (river_inflow + flood_inflow - river_outflow - flood_outflow) * time_step) * 1e-9)


    # ---- 2. Flood stage computation (from original compute_flood_stage_kernel) ----
    river_max_storage   = tl.load(river_max_storage_ptr   + offs, mask=mask)
    catchment_area      = tl.load(catchment_area_ptr      + offs, mask=mask)
    river_width         = tl.load(river_width_ptr         + offs, mask=mask)
    river_length        = tl.load(river_length_ptr        + offs, mask=mask)

    # Determine flood level by scanning the storage table
    level = tl.zeros([BLOCK_SIZE], dtype=tl.int32) - 1
    for i in tl.static_range(num_flood_levels+1):
        prev_total_storage = tl.load(total_storage_table_ptr + offs * (num_flood_levels+1) + i, mask=mask)
        level += tl.where(total_storage > prev_total_storage, 1, 0)
    no_flood_cond = level < 0
    prev_flood_depth   = tl.load(flood_depth_table_ptr   + offs * (num_flood_levels+1) + level, mask=mask)
    prev_total_width   = tl.load(total_width_table_ptr   + offs * (num_flood_levels+1) + level, mask=mask)
    prev_total_storage = tl.load(total_storage_table_ptr + offs * (num_flood_levels+1) + level, mask=mask)
    flood_grad         = tl.load(flood_gradient_table_ptr + offs * (num_flood_levels+1) + level, mask=mask)

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
    tl.atomic_add(total_storage_stage_new_ptr + current_step, tl.sum(total_storage_stage_new) * 1e-9)
    tl.atomic_add(total_river_storage_ptr + current_step, tl.sum(river_storage_final) * 1e-9)
    tl.atomic_add(total_flood_storage_ptr + current_step, tl.sum(flood_storage_final) * 1e-9)
    tl.atomic_add(total_flood_area_ptr + current_step, tl.sum(flood_area) * 1e-9)
    tl.atomic_add(total_stage_error_ptr + current_step, tl.sum((total_storage_stage_new - total_storage) * 1e-9))

    # Return to zero
    tl.store(river_inflow_ptr + offs, 0.0, mask=mask)
    tl.store(flood_inflow_ptr + offs, 0.0, mask=mask)
    tl.store(outgoing_storage_ptr + offs, 0.0, mask=mask)
    tl.store(global_bifurcation_outflow_ptr + offs, 0.0, mask=mask)

    # Store outputs (in-place update)
    tl.store(river_storage_ptr    + offs, river_storage_final, mask=mask)
    tl.store(flood_storage_ptr    + offs, flood_storage_final, mask=mask)
    tl.store(river_depth_ptr      + offs, river_depth, mask=mask)
    tl.store(flood_depth_ptr      + offs, flood_depth, mask=mask)
    tl.store(flood_fraction_ptr   + offs, flood_fraction, mask=mask)
    tl.store(flood_area_ptr       + offs, flood_area, mask=mask)