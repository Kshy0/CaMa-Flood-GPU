# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Reservoir outflow Triton kernels.
"""

import triton
import triton.language as tl

from cmfgpu.phys.triton.utils import to_compute_dtype, typed_sqrt


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

    # Reservoir parameters (reservoir-indexed)
    conservation_volume_ptr,                # *f32  conservation storage
    emergency_volume_ptr,                   # *f32  emergency storage
    adjustment_volume_ptr,                  # *f32  adjustment storage
    normal_outflow_ptr,                     # *f32  normal outflow
    adjustment_outflow_ptr,                 # *f32  adjustment outflow
    flood_control_outflow_ptr,              # *f32  flood control outflow

    # Other catchment-level arrays
    runoff_ptr,                             # *f32  runoff (catchment-indexed)
    total_storage_ptr,                      # *f64  total storage (catchment-indexed, in/out)
    outgoing_storage_ptr,                   # *f64  outgoing storage (catchment-indexed, in/out)

    time_step,                              # f32   scalar time step
    num_reservoirs,                         # i32   total number of reservoirs
    BLOCK_SIZE: tl.constexpr,               # block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_reservoirs

    # ---------- Index mapping ----------
    catchment_idx = tl.load(reservoir_catchment_idx_ptr + offs, mask=mask, other=0)
    downstream_idx = tl.load(downstream_idx_ptr + catchment_idx, mask=mask, other=0)
    is_river_mouth = downstream_idx == catchment_idx

    # ================================================================== #
    # 1. Undo the main outflow kernel's outgoing_storage contribution
    # ================================================================== #
    old_river_outflow = tl.load(river_outflow_ptr + catchment_idx, mask=mask, other=0.0)
    old_flood_outflow = tl.load(flood_outflow_ptr + catchment_idx, mask=mask, other=0.0)

    old_pos = tl.maximum(old_river_outflow, 0.0) + tl.maximum(old_flood_outflow, 0.0)
    old_neg = tl.minimum(old_river_outflow, 0.0) + tl.minimum(old_flood_outflow, 0.0)

    # Subtract the local positive contribution
    tl.atomic_add(outgoing_storage_ptr + catchment_idx, -(old_pos * time_step).to(tl.float64), mask=mask)

    # Undo the downstream scatter of negative flow
    undo_downstream = tl.where(~is_river_mouth, (old_neg * time_step).to(tl.float64), 0.0)
    tl.atomic_add(outgoing_storage_ptr + downstream_idx, undo_downstream, mask=mask)

    # ================================================================== #
    # 2. Compute reservoir outflow
    # ================================================================== #
    river_storage = tl.load(river_storage_ptr + catchment_idx, mask=mask, other=0.0)
    flood_storage = tl.load(flood_storage_ptr + catchment_idx, mask=mask, other=0.0)

    # Downcast hpfloat storage to computation dtype (Fortran: REAL(P2VAR, KIND=JPRB))
    river_storage = to_compute_dtype(river_storage, old_river_outflow)
    flood_storage = to_compute_dtype(flood_storage, old_river_outflow)

    total_storage = river_storage + flood_storage

    # Accumulated total inflow from upstream (from previous sub-step's inflow kernel)
    total_inflow = to_compute_dtype(
        tl.load(reservoir_total_inflow_ptr + catchment_idx, mask=mask, other=0.0), old_river_outflow
    )
    # Zero the accumulator for next sub-step
    tl.store(reservoir_total_inflow_ptr + catchment_idx, tl.zeros_like(total_inflow).to(tl.float64), mask=mask)

    runoff = tl.load(runoff_ptr + catchment_idx, mask=mask, other=0.0)
    reservoir_inflow = total_inflow + runoff

    # Reservoir parameters (reservoir-indexed)
    conservation_volume = tl.load(conservation_volume_ptr + offs, mask=mask, other=0.0)
    emergency_volume = tl.load(emergency_volume_ptr + offs, mask=mask, other=0.0)
    adjustment_volume = tl.load(adjustment_volume_ptr + offs, mask=mask, other=0.0)
    normal_outflow = tl.load(normal_outflow_ptr + offs, mask=mask, other=0.0)
    adjustment_outflow = tl.load(adjustment_outflow_ptr + offs, mask=mask, other=0.0)
    flood_control_outflow = tl.load(flood_control_outflow_ptr + offs, mask=mask, other=0.0)

    reservoir_outflow = tl.zeros_like(total_storage)

    # ---- Case 1: below conservation volume ----
    cond1 = total_storage <= conservation_volume
    reservoir_outflow = tl.where(
        cond1,
        normal_outflow * typed_sqrt(total_storage / conservation_volume),
        reservoir_outflow,
    )

    # ---- Case 2: above conservation, below adjustment volume ----
    cond2 = (total_storage > conservation_volume) & (total_storage <= adjustment_volume)
    frac2 = (total_storage - conservation_volume) / (adjustment_volume - conservation_volume)
    reservoir_outflow = tl.where(
        cond2,
        normal_outflow + tl.exp(3.0 * tl.log(frac2)) * (adjustment_outflow - normal_outflow),
        reservoir_outflow,
    )

    # ---- Case 3: above adjustment, below emergency volume ----
    cond3 = (total_storage > adjustment_volume) & (total_storage <= emergency_volume)
    flood_period = reservoir_inflow >= flood_control_outflow

    # Flood period
    outflow_flood = normal_outflow + (
        (total_storage - conservation_volume) / (emergency_volume - conservation_volume)
    ) * (reservoir_inflow - normal_outflow)
    frac3 = (total_storage - adjustment_volume) / (emergency_volume - adjustment_volume)
    outflow_tmp = adjustment_outflow + tl.exp(0.1 * tl.log(frac3)) * (
        flood_control_outflow - adjustment_outflow
    )
    outflow_combined = tl.maximum(outflow_flood, outflow_tmp)

    # Non-flood period
    outflow_nonflood = adjustment_outflow + tl.exp(0.1 * tl.log(frac3)) * (
        flood_control_outflow - adjustment_outflow
    )

    reservoir_outflow = tl.where(cond3 & flood_period, outflow_combined, reservoir_outflow)
    reservoir_outflow = tl.where(cond3 & ~flood_period, outflow_nonflood, reservoir_outflow)

    # ---- Case 4: above emergency volume ----
    cond4 = total_storage > emergency_volume
    outflow_emergency = tl.where(
        reservoir_inflow >= flood_control_outflow,
        reservoir_inflow,
        flood_control_outflow,
    )
    reservoir_outflow = tl.where(cond4, outflow_emergency, reservoir_outflow)

    # Clamp to [0, total_storage / time_step]
    reservoir_outflow = tl.clamp(reservoir_outflow, 0.0, total_storage / time_step)

    # ================================================================== #
    # 3. Store results
    # ================================================================== #
    tl.store(river_outflow_ptr + catchment_idx, reservoir_outflow, mask=mask)
    tl.store(flood_outflow_ptr + catchment_idx, 0.0, mask=mask)
    tl.store(total_storage_ptr + catchment_idx, total_storage, mask=mask)

    # Re-add corrected contribution to outgoing_storage
    # Reservoir outflow is always >= 0 (clamped above), so only positive branch
    new_pos = tl.maximum(reservoir_outflow, 0.0)
    tl.atomic_add(outgoing_storage_ptr + catchment_idx, (new_pos * time_step).to(tl.float64), mask=mask)

    new_neg = tl.minimum(reservoir_outflow, 0.0)
    to_add = tl.where(~is_river_mouth, -(new_neg * time_step).to(tl.float64), 0.0)
    tl.atomic_add(outgoing_storage_ptr + downstream_idx, to_add, mask=mask)
