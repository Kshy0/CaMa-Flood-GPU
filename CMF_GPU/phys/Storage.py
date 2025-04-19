import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Kernel: Compute flood stage, river/flood storage, depths, and flood fraction
# -----------------------------------------------------------------------------
@triton.jit
def compute_flood_stage_kernel(
    total_storage_ptr,           # *f32: Total storage (in/out)
    river_storage_ptr,           # *f32: River storage (in/out)
    flood_storage_ptr,           # *f32: Flood storage (in/out)
    river_depth_ptr,             # *f32: River depth (in/out)
    flood_depth_ptr,             # *f32: Flood depth (in/out)
    flood_fraction_ptr,          # *f32: Flood fraction (in/out)
    flood_area_ptr,              # *f32: Flood area (in/out)
    river_max_storage_ptr,       # *f32: River max storage (reference)
    river_area_ptr,              # *f32: River area (reference)
    max_flood_area_ptr,          # *f32: Max flood area (reference)
    total_storage_table_ptr,     # *f32: Lookup table - total storage (reference)
    flood_depth_table_ptr,       # *f32: Lookup table - flood depth (reference)
    total_width_table_ptr,       # *f32: Lookup table - total width (reference)
    flood_gradient_table_ptr,    # *f32: Lookup table - flood gradient (reference)
    catchment_area_ptr,          # *f32: Catchment area (reference)
    river_width_ptr,             # *f32: River width (reference)
    river_length_ptr,            # *f32: River length (reference)
    num_catchments: tl.constexpr,
    num_flood_levels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Block and lane indexing
    bid = tl.program_id(0)
    lane = tl.arange(0, BLOCK_SIZE)
    pid = bid * BLOCK_SIZE + lane
    mask = pid < num_catchments

    # Load scalar inputs
    total_storage       = tl.load(total_storage_ptr       + pid, mask=mask)
    river_max_storage   = tl.load(river_max_storage_ptr   + pid, mask=mask)
    river_area          = tl.load(river_area_ptr          + pid, mask=mask)
    max_flood_area      = tl.load(max_flood_area_ptr      + pid, mask=mask)
    catchment_area      = tl.load(catchment_area_ptr      + pid, mask=mask)
    river_width         = tl.load(river_width_ptr         + pid, mask=mask)
    river_length        = tl.load(river_length_ptr        + pid, mask=mask)

    # Compute table dimensions
    level_max = num_flood_levels + 2

    # Determine flood level by scanning the storage table
    level = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for i in range(level_max):
        total_storage_i = tl.load(total_storage_table_ptr + pid * (level_max + 1) + i, mask=mask)
        level += tl.where(total_storage_i < total_storage, 1, 0)
    level = tl.maximum(level - 1, 0)

    # Gather table values for the determined level
    prev_flood_depth   = tl.load(flood_depth_table_ptr   + pid * (level_max + 1) + level, mask=mask)
    prev_total_width   = tl.load(total_width_table_ptr   + pid * (level_max + 1) + level, mask=mask)
    prev_total_storage = tl.load(total_storage_table_ptr + pid * (level_max + 1) + level, mask=mask)
    flood_grad         = tl.load(flood_gradient_table_ptr+ pid * level_max + level, mask=mask)

    # Compute two candidate flood depths
    diff_width = tl.sqrt(
        prev_total_width * prev_total_width +
        2.0 * (total_storage - prev_total_storage) / (flood_grad * river_length)
    ) - prev_total_width
    flood_depth_if_mid = prev_flood_depth + diff_width * flood_grad
    flood_depth_if_top = prev_flood_depth + (total_storage - prev_total_storage) / max_flood_area

    # Select flood depth by case
    flood_depth = tl.where(
        level == 0, 0.0,
        tl.where(level == num_flood_levels + 1, flood_depth_if_top, flood_depth_if_mid)
    )

    # River storage and depth
    river_storage = tl.where(
        level == 0,
        total_storage,
        tl.minimum(river_max_storage + river_area * flood_depth, total_storage)
    )
    river_depth = river_storage / river_area

    # Flood fraction
    flood_fraction_mid = (prev_total_width + diff_width - river_width) * river_length / catchment_area
    flood_fraction_mid = tl.maximum(0.0, tl.minimum(1.0, flood_fraction_mid))
    flood_fraction = tl.where(
        level == 0, 0.0,
        tl.where(level == level_max - 1, 1.0, flood_fraction_mid)
    )

    # Flood area and storage
    flood_area    = flood_fraction * catchment_area
    flood_storage = tl.maximum(total_storage - river_storage, 0.0)

    # Store outputs (in-place update)
    tl.store(river_storage_ptr    + pid, river_storage, mask=mask)
    tl.store(flood_storage_ptr    + pid, flood_storage, mask=mask)
    tl.store(river_depth_ptr      + pid, river_depth, mask=mask)
    tl.store(flood_depth_ptr      + pid, flood_depth, mask=mask)
    tl.store(flood_fraction_ptr   + pid, flood_fraction, mask=mask)
    tl.store(flood_area_ptr       + pid, flood_area, mask=mask)


# -----------------------------------------------------------------------------
# Kernel 1: Accumulate inflows (scatter add with river_out_update and flood_out_update inside the kernel)
# -----------------------------------------------------------------------------
@triton.jit
def accumulate_inflows_kernel(
    is_river_mouth_ptr,            # *i32: River mouth mask
    river_outflow_ptr,             # *f32: River outflow
    flood_outflow_ptr,             # *f32: Flood outflow
    downstream_idx_ptr,            # *i32: Downstream unit indices
    river_inflow_ptr,              # *f32: River inflow (output)
    flood_inflow_ptr,              # *f32: Flood inflow (output)
    num_catchments: tl.constexpr,  # Total number of units
    BLOCK_SIZE: tl.constexpr       # Block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_catchments

    # Load inputs
    is_river_mouth = tl.load(is_river_mouth_ptr + offs, mask=mask, other=0)
    downstream_idx = tl.load(downstream_idx_ptr + offs, mask=mask, other=0)
    river_outflow = tl.load(river_outflow_ptr + offs, mask=mask, other=0.0)
    flood_outflow = tl.load(flood_outflow_ptr + offs, mask=mask, other=0.0)

    # Compute river_out_update and flood_out_update inside the kernel
    river_out_update = tl.where(~is_river_mouth, river_outflow, 0.0)
    flood_out_update = tl.where(~is_river_mouth, flood_outflow, 0.0)

    # Atomic add to accumulate outflow to the downstream units
    tl.atomic_add(river_inflow_ptr + downstream_idx, river_out_update, mask=mask)
    tl.atomic_add(flood_inflow_ptr + downstream_idx, flood_out_update, mask=mask)


# -----------------------------------------------------------------------------
# Kernel 2: Update river, flood storage, and calculate total storage
# -----------------------------------------------------------------------------
@triton.jit
def update_storage_kernel(
    river_storage_ptr,             # *f32: River storage (in/out)
    flood_storage_ptr,             # *f32: Flood storage (in/out)
    total_storage_ptr,             # *f32: Total storage (output)
    river_inflow_ptr,              # *f32: River inflow
    flood_inflow_ptr,              # *f32: Flood inflow
    river_outflow_ptr,             # *f32: River outflow
    flood_outflow_ptr,             # *f32: Flood outflow
    runoff_ptr,                    # *f32: External runoff
    time_step,                     # f32: Time step
    num_catchments: tl.constexpr,  # Total number of units
    BLOCK_SIZE: tl.constexpr       # Block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_catchments

    # Load inputs
    river_storage = tl.load(river_storage_ptr + offs, mask=mask, other=0.0)
    flood_storage = tl.load(flood_storage_ptr + offs, mask=mask, other=0.0)
    river_inflow = tl.load(river_inflow_ptr + offs, mask=mask, other=0.0)
    flood_inflow = tl.load(flood_inflow_ptr + offs, mask=mask, other=0.0)
    river_outflow = tl.load(river_outflow_ptr + offs, mask=mask, other=0.0)
    flood_outflow = tl.load(flood_outflow_ptr + offs, mask=mask, other=0.0)
    runoff = tl.load(runoff_ptr + offs, mask=mask, other=0.0)

    # Update river storage
    river_storage_updated = river_storage + (river_inflow - river_outflow) * time_step
    river_storage_clamped = tl.maximum(river_storage_updated, 0.0)

    # Update flood storage
    flood_storage_updated = flood_storage + (flood_inflow - flood_outflow) * time_step
    flood_storage_clamped = tl.maximum(flood_storage_updated, 0.0)

    # Calculate total storage (integrating the logic from compute_next_time_storage)
    total_storage = tl.maximum(river_storage_clamped + flood_storage_clamped + runoff * time_step, 0.0)

    # Store results (in-place update)
    tl.store(river_storage_ptr + offs, river_storage_clamped, mask=mask)
    tl.store(flood_storage_ptr + offs, flood_storage_clamped, mask=mask)
    tl.store(total_storage_ptr + offs, total_storage, mask=mask)