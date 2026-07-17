# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Registered main-channel outflow and inflow implementations."""

from hydroforge.runtime.backend import BackendRegistry, make_batched_dispatcher


def _metal_outflow():
    from cmfgpu.phys.metal import (
        compute_outflow_batched_kernel, compute_outflow_kernel,
    )
    return make_batched_dispatcher(
        compute_outflow_kernel, compute_outflow_batched_kernel,
    )


def _metal_inflow():
    from cmfgpu.phys.metal import (
        compute_inflow_batched_kernel, compute_inflow_kernel,
    )
    return make_batched_dispatcher(
        compute_inflow_kernel, compute_inflow_batched_kernel,
    )


def _cuda_outflow():
    from cmfgpu.phys.cuda import compute_outflow
    return compute_outflow


def _cuda_inflow():
    from cmfgpu.phys.cuda import compute_inflow
    return compute_inflow


def _triton_outflow():
    from hydroforge.runtime.backend import make_triton_dispatcher
    from cmfgpu.phys.triton.outflow import (
        compute_outflow_batched_kernel, compute_outflow_kernel,
    )
    return make_triton_dispatcher(
        compute_outflow_kernel, batched_kernel=compute_outflow_batched_kernel,
        batched_drop=("is_dam_upstream_ptr", "HAS_RESERVOIR", "MIN_KINEMATIC_SLOPE"),
        non_batched_drop=("num_sea_level_boundaries",),
        optional_buffers={
            "global_bifurcation_outflow_ptr": "HAS_BIFURCATION",
            "is_dam_upstream_ptr": "HAS_RESERVOIR",
            "sea_surface_elevation_ptr": "HAS_SEA_LEVEL",
            "catchment_sea_level_idx_ptr": "HAS_SEA_LEVEL",
        },
    )


def _triton_inflow():
    from hydroforge.runtime.backend import make_triton_dispatcher
    from cmfgpu.phys.triton.outflow import (
        compute_inflow_batched_kernel, compute_inflow_kernel,
    )
    return make_triton_dispatcher(
        compute_inflow_kernel, batched_kernel=compute_inflow_batched_kernel,
        optional_buffers={
            "reservoir_total_inflow_ptr": "HAS_RESERVOIR",
            "is_reservoir_ptr": "HAS_RESERVOIR",
        },
    )


compute_outflow = BackendRegistry(
    {"metal": _metal_outflow, "cuda": _cuda_outflow, "triton": _triton_outflow},
    name="compute_outflow",
).selected
compute_inflow = BackendRegistry(
    {"metal": _metal_inflow, "cuda": _cuda_inflow, "triton": _triton_inflow},
    name="compute_inflow",
).selected
