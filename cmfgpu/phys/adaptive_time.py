# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

import triton
import triton.language as tl

from cmfgpu.phys.utils import typed_sqrt


@triton.jit
def compute_adaptive_time_step_kernel(
    river_depth_ptr,                        # *f32 river depth
    downstream_distance_ptr,                # *f32 distance to downstream unit
    max_sub_steps_ptr,                      # *i64 max sub steps
    time_step,
    adaptive_time_factor: tl.constexpr ,
    gravity: tl.constexpr ,                                # f32 scalar gravity acceleration
    num_catchments: tl.constexpr,           # total number of elements
    BLOCK_SIZE: tl.constexpr                # block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_catchments

    #----------------------------------------------------------------------
    # (1) Load input variables
    #----------------------------------------------------------------------
    downstream_distance = tl.load(downstream_distance_ptr + offs, mask=mask, other=float('inf'))
    # Clamp river depth to minimum 0.01 for stability
    river_depth = tl.load(river_depth_ptr + offs, mask=mask, other=0)

    # Upcast to fp32 for intermediate computation:
    # time_step (86400) exceeds fp16 max (65504), so fp16 would overflow.
    downstream_distance = downstream_distance.to(tl.float32)
    river_depth = river_depth.to(tl.float32)

    depth = tl.maximum(river_depth, 0.01)
    dt = adaptive_time_factor * downstream_distance / typed_sqrt(gravity * depth)
    dt_clamped = tl.minimum(dt, time_step)
    
    min_dt = tl.min(dt_clamped)
    
    # Calculate num_sub_steps
    # Align with int(round(time_step / min_dt - 0.01) + 1)
    n_steps_float = tl.floor(time_step / min_dt + 0.49) + 1.0
    n_steps = n_steps_float.to(tl.int32)
    
    tl.atomic_max(max_sub_steps_ptr, n_steps)


@triton.jit
def compute_adaptive_time_step_batched_kernel(
    river_depth_ptr,                        # *f32 river depth
    downstream_distance_ptr,                # *f32 distance to downstream unit
    max_sub_steps_ptr,                      # *i64 (size num_trials)
    time_step,
    adaptive_time_factor: tl.constexpr ,
    gravity: tl.constexpr ,                                # f32 scalar gravity acceleration
    num_catchments: tl.constexpr,           # total number of elements
    num_trials: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,                # block size
    # Batch flags
    batched_downstream_distance: tl.constexpr
):
    pid_x = tl.program_id(0)
    idx = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate trial and catchment indices
    trial_idx = idx // num_catchments
    offs = idx % num_catchments
    
    mask = idx < (num_catchments * num_trials)
    
    trial_offset = trial_idx * num_catchments

    #----------------------------------------------------------------------
    # (1) Load input variables
    #----------------------------------------------------------------------
    downstream_distance = tl.load(downstream_distance_ptr + (trial_offset if batched_downstream_distance else 0) + offs, mask=mask, other=float('inf'))
    # Clamp river depth to minimum 0.01 for stability
    river_depth = tl.load(river_depth_ptr + trial_offset + offs, mask=mask, other=0)

    # Upcast to fp32 for intermediate computation:
    # time_step (86400) exceeds fp16 max (65504), so fp16 would overflow.
    downstream_distance = downstream_distance.to(tl.float32)
    river_depth = river_depth.to(tl.float32)

    depth = tl.maximum(river_depth, 0.01)
    dt = adaptive_time_factor * downstream_distance / typed_sqrt(gravity * depth)
    dt_clamped = tl.minimum(dt, time_step)
    
    min_dt = tl.min(dt_clamped)
    
    # Calculate num_sub_steps
    # Align with int(round(time_step / min_dt - 0.01) + 1)
    n_steps_float = tl.floor(time_step / min_dt + 0.49) + 1.0
    n_steps = n_steps_float.to(tl.int32)
    
    tl.atomic_max(max_sub_steps_ptr + trial_idx, n_steps)
