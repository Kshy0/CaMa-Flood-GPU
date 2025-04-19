import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Kernel 1: Compute outgoing_storage = (positive flow) * dt,
# and use atomic_add to scatter negative flow contributions to downstream indices
# -----------------------------------------------------------------------------
@triton.jit
def compute_outgoing_storage_kernel(
    is_river_mouth_ptr,       # *bool: 1 means river mouth, do not scatter negative flow downstream
    downstream_idx_ptr,       # *i32 downstream indices
    river_outflow_ptr,        # *f32 input
    flood_outflow_ptr,        # *f32 input
    outgoing_storage_ptr,     # *f32 output
    time_step,                # *f32 constexpr
    num_catchments: tl.constexpr,          # total number of elements
    BLOCK_SIZE: tl.constexpr  # block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_catchments

    # Load river/flood flow
    r = tl.load(river_outflow_ptr + offs, mask=mask, other=0.0)
    f = tl.load(flood_outflow_ptr + offs, mask=mask, other=0.0)

    # Split positive/negative flow
    pos = tl.maximum(r, 0.0) + tl.maximum(f, 0.0)
    neg = tl.minimum(r, 0.0) + tl.minimum(f, 0.0)

    # Local out = pos * dt
    out_local = pos * time_step
    tl.store(outgoing_storage_ptr + offs, out_local, mask=mask)

    # Scatter-add negative flow to downstream
    idx       = tl.load(downstream_idx_ptr   + offs, mask=mask, other=0)
    is_mouth  = tl.load(is_river_mouth_ptr   + offs, mask=mask, other=1)
    # Only non-river-mouth applies negative flow
    to_add    = tl.where(mask & (is_mouth == 0), -neg * time_step, 0.0)
    tl.atomic_add(outgoing_storage_ptr + idx, to_add, mask=mask)


# -----------------------------------------------------------------------------
# Kernel 2: Apply storage limits to river and flood outflow
# -----------------------------------------------------------------------------
@triton.jit
def apply_storage_limits_kernel(
    downstream_idx_ptr,             # *i32 downstream indices (same as above)
    river_outflow_ptr,        # *f32 in/out
    flood_outflow_ptr,        # *f32 in/out
    total_storage_ptr,        # *f32 total storage
    outgoing_storage_ptr,     # *f32 outgoing flow (already includes positive and negative multiplied by dt)
    num_catchments: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_catchments

    # In-place load
    r   = tl.load(river_outflow_ptr      + offs, mask=mask, other=0.0)
    f   = tl.load(flood_outflow_ptr      + offs, mask=mask, other=0.0)
    tot = tl.load(total_storage_ptr      + offs, mask=mask, other=0.0)
    out = tl.load(outgoing_storage_ptr   + offs, mask=mask, other=0.0)

    # Local limiting
    limit   = tl.where(out > 1e-8, tl.minimum(tot / out, 1.0), 1.0)

    # Downstream limiting
    n_idx   = tl.load(downstream_idx_ptr        + offs, mask=mask, other=0)
    out_d   = tl.load(outgoing_storage_ptr + n_idx,  mask=mask, other=0.0)
    tot_d   = tl.load(total_storage_ptr     + n_idx,  mask=mask, other=0.0)
    limit_d = tl.where(out_d > 1e-8, tl.minimum(tot_d / out_d, 1.0), 1.0)

    # Apply limits
    r_new = tl.where(r >= 0.0, r * limit,   r * limit_d)
    f_new = tl.where(f >= 0.0, f * limit,   f * limit_d)

    # Write back
    tl.store(river_outflow_ptr + offs, r_new, mask=mask)
    tl.store(flood_outflow_ptr + offs, f_new, mask=mask)