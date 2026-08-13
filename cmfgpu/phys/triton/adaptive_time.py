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
    outer_time_step_ptr,
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

    downstream_distance = tl.load(
        downstream_distance_ptr + offs, mask=mask, other=1.0,
    )
    river_depth = tl.load(river_depth_ptr + offs, mask=mask, other=0)
    depth = tl.maximum(river_depth, 0.01)
    factor = tl.full(depth.shape, adaptive_time_factor, depth.dtype)
    gravity_value = tl.full(depth.shape, gravity, depth.dtype)
    dt = factor * downstream_distance / tl.sqrt(gravity_value * depth)
    outer_time_step = tl.load(outer_time_step_ptr)
    dt_clamped = tl.minimum(dt, outer_time_step)

    n_steps_float = tl.floor(outer_time_step / dt_clamped - 0.01) + 1.0
    n_steps = nonnegative_to_index_inline(n_steps_float)
    n_steps = tl.where(mask, n_steps, 1)

    tl.atomic_max(max_sub_steps_ptr, tl.max(n_steps))


@triton.jit
def compute_adaptive_time_step_batched_kernel(
    river_depth_ptr,                        # *f32 river depth
    downstream_distance_ptr,                # *f32 distance to downstream unit
    is_dam_related_ptr,                     # *bool: True for dam + upstream-of-dam cells (I2MASK > 0)
    max_sub_steps_ptr,                      # *i32 (size 1, shared_state)
    outer_time_step_ptr,
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

    depth = tl.maximum(river_depth, 0.01)
    factor = tl.full(depth.shape, adaptive_time_factor, depth.dtype)
    gravity_value = tl.full(depth.shape, gravity, depth.dtype)
    dt = factor * downstream_distance / tl.sqrt(gravity_value * depth)
    outer_time_step = tl.load(outer_time_step_ptr)
    dt_clamped = tl.minimum(dt, outer_time_step)

    n_steps_float = tl.floor(outer_time_step / dt_clamped - 0.01) + 1.0
    n_steps = nonnegative_to_index_inline(n_steps_float)
    n_steps = tl.where(mask, n_steps, 1)

    tl.atomic_max(max_sub_steps_ptr, tl.max(n_steps))
