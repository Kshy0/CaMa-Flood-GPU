# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Registered flood-stage implementations."""

from hydroforge.runtime.backend import BackendRegistry, make_batched_dispatcher


def _metal_stage():
    from cmfgpu.phys.metal import (
        compute_flood_stage_batched_kernel, compute_flood_stage_kernel,
    )
    return make_batched_dispatcher(
        compute_flood_stage_kernel, compute_flood_stage_batched_kernel,
    )


def _metal_log():
    from cmfgpu.phys.metal import compute_flood_stage_log_kernel
    return compute_flood_stage_log_kernel


def _cuda_stage():
    from cmfgpu.phys.cuda import compute_flood_stage
    return compute_flood_stage


def _cuda_log():
    from cmfgpu.phys.cuda import compute_flood_stage_log
    return compute_flood_stage_log


def _triton_stage():
    from hydroforge.runtime.backend import make_triton_dispatcher
    from cmfgpu.phys.triton.storage import (
        compute_flood_stage_batched_kernel, compute_flood_stage_kernel,
    )
    return make_triton_dispatcher(
        compute_flood_stage_kernel, batched_kernel=compute_flood_stage_batched_kernel,
        batched_grid="loop", non_batched_drop=("num_inflow_gauges",),
        optional_buffers={
            "global_bifurcation_outflow_ptr": "HAS_BIFURCATION",
            "inflow_ptr": "HAS_INFLOW", "catchment_inflow_idx_ptr": "HAS_INFLOW",
        },
    )


def _triton_log():
    from hydroforge.runtime.backend import make_triton_dispatcher
    from cmfgpu.phys.triton.storage import compute_flood_stage_log_kernel
    return make_triton_dispatcher(
        compute_flood_stage_log_kernel, non_batched_drop=("num_inflow_gauges",),
        optional_buffers={
            "global_bifurcation_outflow_ptr": "HAS_BIFURCATION",
            "inflow_ptr": "HAS_INFLOW", "catchment_inflow_idx_ptr": "HAS_INFLOW",
            "is_levee_ptr": "HAS_LEVEE",
        },
    )


compute_flood_stage = BackendRegistry(
    {"metal": _metal_stage, "cuda": _cuda_stage, "triton": _triton_stage},
    name="compute_flood_stage",
).selected
compute_flood_stage_log = BackendRegistry(
    {"metal": _metal_log, "cuda": _cuda_log, "triton": _triton_log},
    name="compute_flood_stage_log",
).selected
