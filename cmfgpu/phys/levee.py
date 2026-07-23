# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Registered levee implementations."""

from hydroforge.kernels.registry import (
    BackendRegistry, make_triton_dispatcher,
)
from cmfgpu.phys.specs import (
    LEVEE_BIFURCATION_OUTFLOW, LEVEE_STAGE, LEVEE_STAGE_LOG,
)


def _metal_stage():
    from cmfgpu.phys import metal
    return metal.levee_stage()


def _metal_log():
    from cmfgpu.phys import metal
    return metal.levee_stage_log()


def _metal_bif():
    from cmfgpu.phys import metal
    return metal.levee_bifurcation_outflow()


def _cuda_stage():
    from cmfgpu.phys import cuda
    return cuda.levee_stage()


def _cuda_log():
    from cmfgpu.phys import cuda
    return cuda.levee_stage_log()


def _cuda_bif():
    from cmfgpu.phys import cuda
    return cuda.levee_bifurcation_outflow()


def _triton_stage():
    from cmfgpu.phys.triton.levee import (
        compute_levee_stage_batched_kernel, compute_levee_stage_kernel,
    )
    return make_triton_dispatcher(
        compute_levee_stage_kernel, batched_kernel=compute_levee_stage_batched_kernel,
        batched_grid="loop",
    )


def _triton_log():
    from cmfgpu.phys.triton.levee import compute_levee_stage_log_kernel
    return make_triton_dispatcher(compute_levee_stage_log_kernel)


def _triton_bif():
    from cmfgpu.phys.triton.levee import (
        compute_levee_bifurcation_outflow_batched_kernel,
        compute_levee_bifurcation_outflow_kernel,
    )
    return make_triton_dispatcher(
        compute_levee_bifurcation_outflow_kernel,
        batched_kernel=compute_levee_bifurcation_outflow_batched_kernel,
    )


compute_levee_stage = BackendRegistry(
    {
        "metal": _metal_stage,
        "cuda": _cuda_stage,
        "triton": _triton_stage,
    },
    name="compute_levee_stage",
    spec=LEVEE_STAGE,
).selected
compute_levee_stage_log = BackendRegistry(
    {
        "metal": _metal_log,
        "cuda": _cuda_log,
        "triton": _triton_log,
    },
    name="compute_levee_stage_log",
    spec=LEVEE_STAGE_LOG,
).selected
compute_levee_bifurcation_outflow = BackendRegistry(
    {"metal": _metal_bif, "cuda": _cuda_bif,
     "triton": _triton_bif},
    name="compute_levee_bifurcation_outflow",
    spec=LEVEE_BIFURCATION_OUTFLOW,
).selected
