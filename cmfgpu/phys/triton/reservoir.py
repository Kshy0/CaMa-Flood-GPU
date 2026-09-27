# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Reservoir outflow Triton kernels.
"""

import numpy as np
import triton
import triton.language as tl
from hydroforge.kernels import triton_math as hm

from cmfgpu.phys.triton.outflow import outgoing_flows_inline


from cmfgpu import config as _constants

RESERVOIR_RELEASE_EXPONENT = tl.constexpr(
    float(np.float32(_constants.RESERVOIR_RELEASE_EXPONENT))
)


@triton.jit
def reservoir_release_inline(
    dam_volume, river_flood_storage, reservoir_inflow, time_step,
    conservation_volume, emergency_volume, adjustment_volume,
    normal_outflow, adjustment_outflow, flood_control_outflow,
):
    """Yamazaki & Funato release of CMF_DAMOUT_CALC for the dam volume
    DamVol, then its flow limiter against DamVol and REAL(P2RIVSTO+P2FLDSTO).
    The nested selections follow the CUDA if/else-if regime chain."""
    # Case 1: water use, up to the conservation volume.
    water_use = normal_outflow * hm.sqrt(hm.divide(dam_volume, conservation_volume))
    # Case 2: water excess, up to the adjustment volume.
    frac2 = hm.divide(dam_volume - conservation_volume, adjustment_volume - conservation_volume)
    water_excess = normal_outflow + hm.exp(3.0 * hm.log(frac2)) * (
        adjustment_outflow - normal_outflow
    )
    # Case 3: flood control, up to the emergency volume.
    frac3 = hm.divide(dam_volume - adjustment_volume, emergency_volume - adjustment_volume)
    controlled = adjustment_outflow + hm.exp(RESERVOIR_RELEASE_EXPONENT * hm.log(frac3)) * (
        flood_control_outflow - adjustment_outflow
    )
    flood_period = reservoir_inflow >= flood_control_outflow
    flood = normal_outflow + hm.divide(
        dam_volume - conservation_volume, emergency_volume - conservation_volume,
    ) * (reservoir_inflow - normal_outflow)
    flood_control = tl.where(flood_period, tl.maximum(flood, controlled), controlled)
    # Case 4: emergency operation.
    emergency = tl.where(flood_period, reservoir_inflow, flood_control_outflow)

    reservoir_outflow = tl.where(
        dam_volume <= conservation_volume, water_use,
        tl.where(
            dam_volume <= adjustment_volume, water_excess,
            tl.where(dam_volume <= emergency_volume, flood_control, emergency),
        ),
    )

    # Flow limiter: the minimum first, so a negative storage releases nothing.
    reservoir_outflow = tl.minimum(
        tl.minimum(reservoir_outflow, hm.divide(dam_volume, time_step)),
        hm.divide(river_flood_storage, time_step),
    )
    return hm.at_least(reservoir_outflow, 0.0)


@triton.jit
def compute_reservoir_outflow_kernel(
    reservoir_catchment_idx_ptr,            # *i32  reservoir → catchment index
    downstream_idx_ptr,                     # *i32  catchment-level downstream index

    # Accumulated total inflow from upstream (catchment-indexed, from previous sub-step's inflow kernel)
    reservoir_total_inflow_ptr,             # *f64  accumulated upstream inflow (read & zero, catchment-sized)

    # Catchment-level arrays (indexed via reservoir_catchment_idx)
    river_outflow_ptr,                      # *f32  in/out: overwritten with reservoir outflow
    flood_outflow_ptr,                      # *f32  in/out: zeroed for reservoir catchments
    river_storage_ptr,                      # *f64  river storage
    flood_storage_ptr,                      # *f64  flood storage
    protected_storage_ptr,                  # *f64  levee-protected storage (HAS_LEVEE)

    # Reservoir parameters (reservoir-indexed)
    conservation_volume_ptr,                # *f32  conservation storage
    emergency_volume_ptr,                   # *f32  emergency storage
    adjustment_volume_ptr,                  # *f32  adjustment storage
    effective_normal_outflow_ptr,                     # *f32  normal outflow
    adjustment_outflow_ptr,                 # *f32  adjustment outflow
    flood_control_outflow_ptr,              # *f32  flood control outflow

    # Other catchment-level arrays
    runoff_ptr,                             # *f32  runoff (catchment-indexed)
    outgoing_storage_ptr,                   # *f64  outgoing storage (catchment-indexed, in/out)

    time_step_ptr,                              # f32   scalar time step
    num_reservoirs,                         # i32   total number of reservoirs
    HAS_LEVEE: tl.constexpr,                # levee-protected storage present
    BLOCK_SIZE: tl.constexpr,               # block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_reservoirs
    time_step = tl.load(time_step_ptr)

    # ---------- Index mapping ----------
    catchment_idx = tl.load(reservoir_catchment_idx_ptr + offs, mask=mask, other=0)
    downstream_idx = tl.load(downstream_idx_ptr + catchment_idx, mask=mask, other=0)
    is_river_mouth = downstream_idx == catchment_idx

    # ================================================================== #
    # 1. Undo the main outflow kernel's outgoing_storage contribution
    # ================================================================== #
    old_river_outflow = tl.load(river_outflow_ptr + catchment_idx, mask=mask, other=0.0)
    old_flood_outflow = tl.load(flood_outflow_ptr + catchment_idx, mask=mask, other=0.0)

    old_own, old_reversed = outgoing_flows_inline(
        old_river_outflow, old_flood_outflow, outgoing_storage_ptr.dtype.element_ty,
    )
    tl.atomic_add(outgoing_storage_ptr + catchment_idx, -old_own, mask=mask)
    tl.atomic_add(
        outgoing_storage_ptr + downstream_idx, -old_reversed,
        mask=mask & ~is_river_mouth,
    )

    # ================================================================== #
    # 2. Compute reservoir outflow
    # ================================================================== #
    storage = tl.load(river_storage_ptr + catchment_idx, mask=mask, other=0.0) + tl.load(
        flood_storage_ptr + catchment_idx, mask=mask, other=0.0,
    )
    river_flood_storage = hm.to_compute(storage, old_river_outflow)
    if HAS_LEVEE:
        storage += tl.load(protected_storage_ptr + catchment_idx, mask=mask, other=0.0)
    dam_volume = hm.to_compute(storage, old_river_outflow)

    total_inflow = tl.load(reservoir_total_inflow_ptr + catchment_idx, mask=mask, other=0.0)
    runoff = tl.load(runoff_ptr + catchment_idx, mask=mask, other=0.0)
    reservoir_inflow = hm.to_compute(
        total_inflow + runoff.to(total_inflow.dtype), old_river_outflow,
    )
    # Zero the accumulator for next sub-step
    tl.store(
        reservoir_total_inflow_ptr + catchment_idx,
        tl.zeros_like(total_inflow), mask=mask,
    )

    # Reservoir parameters (reservoir-indexed)
    conservation_volume = tl.load(conservation_volume_ptr + offs, mask=mask, other=0.0)
    emergency_volume = tl.load(emergency_volume_ptr + offs, mask=mask, other=0.0)
    adjustment_volume = tl.load(adjustment_volume_ptr + offs, mask=mask, other=0.0)
    normal_outflow = tl.load(effective_normal_outflow_ptr + offs, mask=mask, other=0.0)
    adjustment_outflow = tl.load(adjustment_outflow_ptr + offs, mask=mask, other=0.0)
    flood_control_outflow = tl.load(flood_control_outflow_ptr + offs, mask=mask, other=0.0)

    reservoir_outflow = reservoir_release_inline(
        dam_volume, river_flood_storage, reservoir_inflow, time_step,
        conservation_volume, emergency_volume, adjustment_volume,
        normal_outflow, adjustment_outflow, flood_control_outflow,
    )

    # ================================================================== #
    # 3. Store results
    # ================================================================== #
    tl.store(river_outflow_ptr + catchment_idx, reservoir_outflow, mask=mask)
    tl.store(flood_outflow_ptr + catchment_idx, 0.0, mask=mask)

    # The release is clamped non-negative, so it only leaves this cell.
    tl.atomic_add(
        outgoing_storage_ptr + catchment_idx,
        reservoir_outflow.to(outgoing_storage_ptr.dtype.element_ty), mask=mask,
    )


@triton.jit
def compute_reservoir_outflow_batched_kernel(
    reservoir_catchment_idx_ptr,
    downstream_idx_ptr,
    reservoir_total_inflow_ptr,
    river_outflow_ptr,
    flood_outflow_ptr,
    river_storage_ptr,
    flood_storage_ptr,
    protected_storage_ptr,
    conservation_volume_ptr,
    emergency_volume_ptr,
    adjustment_volume_ptr,
    effective_normal_outflow_ptr,
    adjustment_outflow_ptr,
    flood_control_outflow_ptr,
    runoff_ptr,
    outgoing_storage_ptr,
    time_step_ptr,
    num_reservoirs: tl.constexpr,
    num_catchments: tl.constexpr,
    HAS_LEVEE: tl.constexpr,
    ensemble_size: tl.constexpr,
    batched_runoff: tl.constexpr,
    batched_conservation_volume: tl.constexpr,
    batched_emergency_volume: tl.constexpr,
    batched_adjustment_volume: tl.constexpr,
    batched_effective_normal_outflow: tl.constexpr,
    batched_adjustment_outflow: tl.constexpr,
    batched_flood_control_outflow: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = num_reservoirs * ensemble_size
    mask = idx < total
    reservoir_idx = idx % num_reservoirs
    member_index = idx // num_reservoirs
    member_offset = member_index * num_catchments
    time_step = tl.load(time_step_ptr)

    local_catchment = tl.load(
        reservoir_catchment_idx_ptr + reservoir_idx, mask=mask, other=0,
    )
    local_downstream = tl.load(
        downstream_idx_ptr + local_catchment, mask=mask, other=0,
    )
    is_river_mouth = local_downstream == local_catchment
    catchment_idx = member_offset + local_catchment
    downstream_idx = member_offset + local_downstream

    old_river_outflow = tl.load(
        river_outflow_ptr + catchment_idx, mask=mask, other=0.0,
    )
    old_flood_outflow = tl.load(
        flood_outflow_ptr + catchment_idx, mask=mask, other=0.0,
    )
    old_own, old_reversed = outgoing_flows_inline(
        old_river_outflow, old_flood_outflow, outgoing_storage_ptr.dtype.element_ty,
    )
    tl.atomic_add(outgoing_storage_ptr + catchment_idx, -old_own, mask=mask)
    tl.atomic_add(
        outgoing_storage_ptr + downstream_idx, -old_reversed,
        mask=mask & ~is_river_mouth,
    )

    storage = tl.load(
        river_storage_ptr + catchment_idx, mask=mask, other=0.0,
    ) + tl.load(flood_storage_ptr + catchment_idx, mask=mask, other=0.0)
    river_flood_storage = hm.to_compute(storage, old_river_outflow)
    if HAS_LEVEE:
        storage += tl.load(protected_storage_ptr + catchment_idx, mask=mask, other=0.0)
    dam_volume = hm.to_compute(storage, old_river_outflow)
    total_inflow = tl.load(
        reservoir_total_inflow_ptr + catchment_idx, mask=mask, other=0.0,
    )
    runoff_idx = catchment_idx if batched_runoff else local_catchment
    runoff = tl.load(runoff_ptr + runoff_idx, mask=mask, other=0.0)
    reservoir_inflow = hm.to_compute(
        total_inflow + runoff.to(total_inflow.dtype), old_river_outflow,
    )
    tl.store(
        reservoir_total_inflow_ptr + catchment_idx,
        tl.zeros_like(total_inflow), mask=mask,
    )

    # ``idx`` is this member's reservoir in a parameter with a member axis.
    conservation_volume = tl.load(
        conservation_volume_ptr
        + (idx if batched_conservation_volume else reservoir_idx),
        mask=mask, other=0.0,
    )
    emergency_volume = tl.load(
        emergency_volume_ptr + (idx if batched_emergency_volume else reservoir_idx),
        mask=mask, other=0.0,
    )
    adjustment_volume = tl.load(
        adjustment_volume_ptr + (idx if batched_adjustment_volume else reservoir_idx),
        mask=mask, other=0.0,
    )
    normal_outflow = tl.load(
        effective_normal_outflow_ptr
        + (idx if batched_effective_normal_outflow else reservoir_idx),
        mask=mask, other=0.0,
    )
    adjustment_outflow = tl.load(
        adjustment_outflow_ptr
        + (idx if batched_adjustment_outflow else reservoir_idx),
        mask=mask, other=0.0,
    )
    flood_control_outflow = tl.load(
        flood_control_outflow_ptr
        + (idx if batched_flood_control_outflow else reservoir_idx),
        mask=mask, other=0.0,
    )

    reservoir_outflow = reservoir_release_inline(
        dam_volume, river_flood_storage, reservoir_inflow, time_step,
        conservation_volume, emergency_volume, adjustment_volume,
        normal_outflow, adjustment_outflow, flood_control_outflow,
    )

    tl.store(
        river_outflow_ptr + catchment_idx, reservoir_outflow, mask=mask,
    )
    tl.store(flood_outflow_ptr + catchment_idx, 0.0, mask=mask)
    tl.atomic_add(
        outgoing_storage_ptr + catchment_idx,
        reservoir_outflow.to(outgoing_storage_ptr.dtype.element_ty), mask=mask,
    )
