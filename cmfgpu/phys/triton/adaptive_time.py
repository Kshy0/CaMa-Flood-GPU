# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

import triton
import triton.language as tl

from cmfgpu.phys.triton.utils import nonnegative_to_index_inline


@triton.jit
def compute_adaptive_time_step_kernel(
    river_depth_ptr,                        # *f32 river depth
    downstream_distance_ptr,                # *f32 distance to downstream unit
    is_dam_related_ptr,                     # *bool: True for dam + upstream-of-dam cells (I2MASK > 0)
    max_sub_steps_ptr,                      # *i32 max sub steps
    outer_time_step,
    adaptive_time_factor: tl.constexpr ,
    gravity: tl.constexpr ,                                # f32 scalar gravity acceleration
    num_catchments: tl.constexpr,           # total number of elements
    BLOCK_SIZE: tl.constexpr,               # block size
    HAS_RESERVOIR: tl.constexpr = False,    # whether reservoir module is active
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_catchments

    # Skip dam-related cells from the CFL calculation.
    if HAS_RESERVOIR:
        is_dam = tl.load(is_dam_related_ptr + offs, mask=mask, other=False)
        mask = mask & (~is_dam)

    # Compute a per-cell step count instead of reducing a padded minimum dt.
    # This leaves max_sub_steps untouched when every cell is dam-related and
    # lets the host apply the intentional one-step fallback for that case.
    downstream_distance = tl.load(
        downstream_distance_ptr + offs, mask=mask, other=1.0,
    )
    river_depth = tl.load(river_depth_ptr + offs, mask=mask, other=0)
    depth = tl.maximum(river_depth, 0.01)
    dt = adaptive_time_factor * downstream_distance / tl.sqrt(gravity * depth)
    dt_clamped = tl.minimum(dt, outer_time_step)

    # ``x - x == 0`` is true only for finite x, avoiding backend-specific
    # isfinite helpers while rejecting both NaN and infinities.
    valid_dt = (
        (river_depth - river_depth == 0.0)
        & (downstream_distance - downstream_distance == 0.0)
        & (dt_clamped > 0.0)
        & (dt_clamped == dt_clamped)
    )
    n_steps_float = tl.floor(outer_time_step / dt_clamped - 0.01) + 1.0
    # max_sub_steps has an int32 ABI. Reserve INT_MAX as the failure marker;
    # the next smaller representable fp32 value is safe to cast to int32.
    valid_count = valid_dt & (n_steps_float < 2147483647.0)
    bounded = tl.where(valid_count, n_steps_float, 0.0)
    n_steps = nonnegative_to_index_inline(bounded)
    n_steps = tl.where(valid_count, n_steps, 2147483647)
    n_steps = tl.where(mask, n_steps, 0)

    tl.atomic_max(max_sub_steps_ptr, tl.max(n_steps))


@triton.jit
def compute_adaptive_time_step_batched_kernel(
    river_depth_ptr,                        # *f32 river depth
    downstream_distance_ptr,                # *f32 distance to downstream unit
    is_dam_related_ptr,                     # *bool: True for dam + upstream-of-dam cells (I2MASK > 0)
    max_sub_steps_ptr,                      # *i32 (size 1, shared_state)
    outer_time_step,
    adaptive_time_factor: tl.constexpr ,
    gravity: tl.constexpr ,                                # f32 scalar gravity acceleration
    num_catchments: tl.constexpr,           # total number of elements
    num_trials: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,                # block size
    # Batch flags
    batched_downstream_distance: tl.constexpr,
    HAS_RESERVOIR: tl.constexpr = False,    # whether reservoir module is active
):
    pid_x = tl.program_id(0)
    idx = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate trial and catchment indices
    trial_idx = idx // num_catchments
    offs = idx % num_catchments
    
    mask = idx < (num_catchments * num_trials)
    
    # Skip dam-related cells from the CFL calculation.
    if HAS_RESERVOIR:
        is_dam = tl.load(is_dam_related_ptr + offs, mask=mask, other=False)
        mask = mask & (~is_dam)

    trial_offset = trial_idx * num_catchments

    downstream_distance = tl.load(
        downstream_distance_ptr
        + (trial_offset if batched_downstream_distance else 0) + offs,
        mask=mask, other=1.0,
    )
    river_depth = tl.load(river_depth_ptr + trial_offset + offs, mask=mask, other=0)

    # Upcast to fp32 for intermediate computation:
    # outer_time_step (86400) exceeds fp16 max (65504), so fp16 would overflow.
    downstream_distance = downstream_distance
    river_depth = river_depth

    depth = tl.maximum(river_depth, 0.01)
    dt = adaptive_time_factor * downstream_distance / tl.sqrt(gravity * depth)
    dt_clamped = tl.minimum(dt, outer_time_step)

    valid_dt = (
        (river_depth - river_depth == 0.0)
        & (downstream_distance - downstream_distance == 0.0)
        & (dt_clamped > 0.0)
        & (dt_clamped == dt_clamped)
    )
    n_steps_float = tl.floor(outer_time_step / dt_clamped - 0.01) + 1.0
    valid_count = valid_dt & (n_steps_float < 2147483647.0)
    bounded = tl.where(valid_count, n_steps_float, 0.0)
    n_steps = nonnegative_to_index_inline(bounded)
    n_steps = tl.where(valid_count, n_steps, 2147483647)
    n_steps = tl.where(mask, n_steps, 0)

    tl.atomic_max(max_sub_steps_ptr, tl.max(n_steps))
