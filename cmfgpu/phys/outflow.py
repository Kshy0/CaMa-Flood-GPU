# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Registered main-channel outflow and inflow implementations."""

from hydroforge.kernels.registry import BackendRegistry
from cmfgpu.phys.specs import INFLOW, OUTFLOW


def _metal_outflow():
    from cmfgpu.phys import metal
    return metal.outflow()


def _metal_inflow():
    from cmfgpu.phys import metal
    return metal.inflow()


def _cuda_outflow():
    from cmfgpu.phys import cuda
    return cuda.outflow()


def _cuda_inflow():
    from cmfgpu.phys import cuda
    return cuda.inflow()


def _triton_outflow():
    from hydroforge.kernels.registry import make_triton_dispatcher
    from cmfgpu.phys.triton.outflow import (
        compute_outflow_batched_kernel, compute_outflow_kernel,
    )
    return make_triton_dispatcher(
        compute_outflow_kernel, batched_kernel=compute_outflow_batched_kernel,
    )


def _triton_inflow():
    from hydroforge.kernels.registry import make_triton_dispatcher
    from cmfgpu.phys.triton.outflow import (
        compute_inflow_batched_kernel, compute_inflow_kernel,
    )
    return make_triton_dispatcher(
        compute_inflow_kernel, batched_kernel=compute_inflow_batched_kernel,
    )


compute_outflow = BackendRegistry(
    {"metal": _metal_outflow, "cuda": _cuda_outflow, "triton": _triton_outflow},
    name="compute_outflow",
    spec=OUTFLOW,
).selected
compute_inflow = BackendRegistry(
    {"metal": _metal_inflow, "cuda": _cuda_inflow, "triton": _triton_inflow},
    name="compute_inflow",
    spec=INFLOW,
).selected
