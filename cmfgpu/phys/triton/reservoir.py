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
from hydroforge.kernels import triton_math as hm

from cmfgpu.phys.triton.outflow import outgoing_flows_inline


@triton.jit
def yamazaki_funato_release_inline(
    dam_volume, reservoir_inflow,
    conservation_volume, emergency_volume, adjustment_volume,
    normal_outflow, adjustment_outflow, flood_control_outflow, release_exponent,
):
    """Yamazaki & Funato reservoir release for the dam volume DamVol.
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
    controlled = adjustment_outflow + hm.exp(hm.constant(release_exponent, frac3) * hm.log(frac3)) * (
        flood_control_outflow - adjustment_outflow
    )
    flood_period = reservoir_inflow >= flood_control_outflow
    flood = normal_outflow + hm.divide(
        dam_volume - conservation_volume, emergency_volume - conservation_volume,
    ) * (reservoir_inflow - normal_outflow)
    flood_control = tl.where(flood_period, tl.maximum(flood, controlled), controlled)
    # Case 4: emergency operation.
    emergency = tl.where(flood_period, reservoir_inflow, flood_control_outflow)

    return tl.where(
        dam_volume <= conservation_volume, water_use,
        tl.where(
            dam_volume <= adjustment_volume, water_excess,
            tl.where(dam_volume <= emergency_volume, flood_control, emergency),
        ),
    )


@triton.jit
def h22_release_inline(
    dam_volume, reservoir_inflow,
    normal_volume, conservation_volume, emergency_volume,
    normal_outflow, flood_control_outflow, flood_inflow_share,
):
    """Hanazaki et al. (2022) release (CaMa-Flood v4 LDAMH22) for DamVol,
    with the regime order of the CUDA if/else-if chain."""
    half_normal = normal_outflow * 0.5
    # Case 1: water use, up to the normal volume.
    water_use = normal_outflow * hm.divide(dam_volume, conservation_volume)
    # Outside floods: storage-dependent supply up to the emergency volume.
    frac = hm.divide(dam_volume - normal_volume, emergency_volume - normal_volume)
    supply = half_normal + frac * frac * (flood_control_outflow - normal_outflow)
    flood_period = flood_control_outflow <= reservoir_inflow
    # Case 2: water supply, up to the conservation volume.
    water_supply = tl.where(
        flood_period,
        half_normal + hm.divide(
            dam_volume - normal_volume, conservation_volume - normal_volume,
        ) * (flood_control_outflow - normal_outflow),
        supply,
    )
    # Case 3: flood control below the emergency volume.
    flood_control = tl.where(
        flood_period,
        flood_control_outflow + hm.divide(
            flood_inflow_share * (dam_volume - conservation_volume),
            emergency_volume - conservation_volume,
        ) * (reservoir_inflow - flood_control_outflow),
        supply,
    )
    # Case 4: emergency operation.
    emergency = tl.maximum(reservoir_inflow, flood_control_outflow)

    return tl.where(
        dam_volume <= normal_volume, water_use,
        tl.where(
            dam_volume <= conservation_volume, water_supply,
            tl.where(dam_volume < emergency_volume, flood_control, emergency),
        ),
    )


@triton.jit
def limit_release_inline(release, dam_volume, river_flood_storage, time_step):
    """Flow limiter against DamVol and REAL(P2RIVSTO+P2FLDSTO): the minimum
    first, so a negative storage releases nothing."""
    release = tl.minimum(
        tl.minimum(release, hm.divide(dam_volume, time_step)),
        hm.divide(river_flood_storage, time_step),
    )
    return hm.at_least(release, 0.0)


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
    normal_volume_ptr,                      # *f32  H22 normal volume (HAS_H22)
    flood_inflow_share_ptr,                 # *f32  H22 share of flood inflow (HAS_H22)

    # Other catchment-level arrays
    runoff_ptr,                             # *f32  runoff (catchment-indexed)
    outgoing_storage_ptr,                   # *f64  outgoing storage (catchment-indexed, in/out)

    time_step_ptr,                              # f32   scalar time step
    num_reservoirs,                         # i32   total number of reservoirs
    HAS_LEVEE: tl.constexpr,                # levee-protected storage present
    release_exponent: tl.constexpr,
    HAS_YF: tl.constexpr,
    HAS_H22: tl.constexpr,                  # uniform Hanazaki (2022) release choice
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
    tl.atomic_add(outgoing_storage_ptr + catchment_idx, -old_own, mask=mask, sem="relaxed")
    tl.atomic_add(outgoing_storage_ptr + downstream_idx, -old_reversed, mask=mask & ~is_river_mouth, sem="relaxed")

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
    normal_outflow = tl.load(effective_normal_outflow_ptr + offs, mask=mask, other=0.0)
    flood_control_outflow = tl.load(flood_control_outflow_ptr + offs, mask=mask, other=0.0)

    if HAS_H22:
        reservoir_outflow = h22_release_inline(
            dam_volume, reservoir_inflow,
            tl.load(normal_volume_ptr + offs, mask=mask, other=0.0),
            conservation_volume, emergency_volume,
            normal_outflow,
            flood_control_outflow,
            tl.load(flood_inflow_share_ptr + offs, mask=mask, other=0.0),
        )
    else:
        adjustment_volume = tl.load(adjustment_volume_ptr + offs, mask=mask, other=0.0)
        adjustment_outflow = tl.load(adjustment_outflow_ptr + offs, mask=mask, other=0.0)
        reservoir_outflow = yamazaki_funato_release_inline(
            dam_volume, reservoir_inflow,
            conservation_volume, emergency_volume, adjustment_volume,
            normal_outflow, adjustment_outflow, flood_control_outflow, release_exponent,
        )
    reservoir_outflow = limit_release_inline(
        reservoir_outflow, dam_volume, river_flood_storage, time_step,
    )

    # ================================================================== #
    # 3. Store results
    # ================================================================== #
    tl.store(river_outflow_ptr + catchment_idx, reservoir_outflow, mask=mask)
    tl.store(flood_outflow_ptr + catchment_idx, 0.0, mask=mask)

    # The release is clamped non-negative, so it only leaves this cell.
    tl.atomic_add(outgoing_storage_ptr + catchment_idx, reservoir_outflow.to(outgoing_storage_ptr.dtype.element_ty), mask=mask, sem="relaxed")


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
    normal_volume_ptr,
    flood_inflow_share_ptr,
    runoff_ptr,
    outgoing_storage_ptr,
    time_step_ptr,
    num_reservoirs: tl.constexpr,
    num_catchments: tl.constexpr,
    HAS_LEVEE: tl.constexpr,
    release_exponent: tl.constexpr,
    HAS_YF: tl.constexpr,
    HAS_H22: tl.constexpr,
    ensemble_size: tl.constexpr,
    batched_runoff: tl.constexpr,
    batched_conservation_volume: tl.constexpr,
    batched_emergency_volume: tl.constexpr,
    batched_adjustment_volume: tl.constexpr,
    batched_effective_normal_outflow: tl.constexpr,
    batched_adjustment_outflow: tl.constexpr,
    batched_flood_control_outflow: tl.constexpr,
    batched_normal_volume: tl.constexpr,
    batched_flood_inflow_share: tl.constexpr,
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
    tl.atomic_add(outgoing_storage_ptr + catchment_idx, -old_own, mask=mask, sem="relaxed")
    tl.atomic_add(outgoing_storage_ptr + downstream_idx, -old_reversed, mask=mask & ~is_river_mouth, sem="relaxed")

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
    normal_outflow = tl.load(
        effective_normal_outflow_ptr
        + (idx if batched_effective_normal_outflow else reservoir_idx),
        mask=mask, other=0.0,
    )
    flood_control_outflow = tl.load(
        flood_control_outflow_ptr
        + (idx if batched_flood_control_outflow else reservoir_idx),
        mask=mask, other=0.0,
    )

    if HAS_H22:
        reservoir_outflow = h22_release_inline(
            dam_volume, reservoir_inflow,
            tl.load(
                normal_volume_ptr + (idx if batched_normal_volume else reservoir_idx),
                mask=mask, other=0.0,
            ),
            conservation_volume, emergency_volume,
            normal_outflow,
            flood_control_outflow,
            tl.load(
                flood_inflow_share_ptr
                + (idx if batched_flood_inflow_share else reservoir_idx),
                mask=mask, other=0.0,
            ),
        )
    else:
        adjustment_volume = tl.load(
            adjustment_volume_ptr + (idx if batched_adjustment_volume else reservoir_idx),
            mask=mask, other=0.0,
        )
        adjustment_outflow = tl.load(
            adjustment_outflow_ptr
            + (idx if batched_adjustment_outflow else reservoir_idx),
            mask=mask, other=0.0,
        )
        reservoir_outflow = yamazaki_funato_release_inline(
            dam_volume, reservoir_inflow,
            conservation_volume, emergency_volume, adjustment_volume,
            normal_outflow, adjustment_outflow, flood_control_outflow, release_exponent,
        )
    reservoir_outflow = limit_release_inline(
        reservoir_outflow, dam_volume, river_flood_storage, time_step,
    )

    tl.store(
        river_outflow_ptr + catchment_idx, reservoir_outflow, mask=mask,
    )
    tl.store(flood_outflow_ptr + catchment_idx, 0.0, mask=mask)
    tl.atomic_add(outgoing_storage_ptr + catchment_idx, reservoir_outflow.to(outgoing_storage_ptr.dtype.element_ty), mask=mask, sem="relaxed")



@triton.jit
def fill_dam_cells_inline(
    cell, volume, mask, conservation_volume_ptr, river_storage_ptr, flood_storage_ptr,
):
    """CMF_DAMOUT_INIT for dam storage ``cell`` and its conservation volume:
    below that volume, the dam starts full to it with an empty floodplain."""
    river = tl.load(river_storage_ptr + cell, mask=mask, other=0.0)
    flood = tl.load(flood_storage_ptr + cell, mask=mask, other=0.0)
    conservation = tl.load(
        conservation_volume_ptr + volume, mask=mask, other=0.0
    ).to(river.dtype)
    fill = mask & (river + flood < conservation)
    tl.store(river_storage_ptr + cell, conservation, mask=fill)
    tl.store(flood_storage_ptr + cell, tl.zeros_like(flood), mask=fill)


@triton.jit
def fill_dam_storage_kernel(
    reservoir_catchment_idx_ptr, conservation_volume_ptr,
    river_storage_ptr, flood_storage_ptr,
    num_reservoirs,
    BLOCK_SIZE: tl.constexpr,
):
    reservoir = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = reservoir < num_reservoirs
    cell = tl.load(reservoir_catchment_idx_ptr + reservoir, mask=mask, other=0)
    fill_dam_cells_inline(
        cell, reservoir, mask, conservation_volume_ptr, river_storage_ptr, flood_storage_ptr,
    )


@triton.jit
def fill_dam_storage_batched_kernel(
    reservoir_catchment_idx_ptr, conservation_volume_ptr,
    river_storage_ptr, flood_storage_ptr,
    num_reservoirs: tl.constexpr,
    num_catchments: tl.constexpr,
    ensemble_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    batched_conservation_volume: tl.constexpr,
):
    # One lane per dam and member.
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < num_reservoirs * ensemble_size
    reservoir = idx % num_reservoirs
    member_offset = (idx // num_reservoirs) * num_catchments
    cell = member_offset + tl.load(reservoir_catchment_idx_ptr + reservoir, mask=mask, other=0)
    fill_dam_cells_inline(
        cell, idx if batched_conservation_volume else reservoir, mask,
        conservation_volume_ptr, river_storage_ptr, flood_storage_ptr,
    )
