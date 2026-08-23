# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Registered reservoir-outflow implementations."""

from hydroforge.kernels import BackendRegistry
from cmfgpu.phys.specs import RESERVOIR_OUTFLOW


def _metal():
    from cmfgpu.phys import metal
    return metal.reservoir_outflow()


def _cuda():
    from cmfgpu.phys import cuda
    return cuda.reservoir_outflow()


def _triton():
    from hydroforge.kernels import make_triton_dispatcher
    from cmfgpu.phys.triton.reservoir import (
        compute_reservoir_outflow_batched_kernel,
        compute_reservoir_outflow_kernel,
    )
    return make_triton_dispatcher(
        compute_reservoir_outflow_kernel,
        batched_kernel=compute_reservoir_outflow_batched_kernel,
    )


compute_reservoir_outflow_registry = BackendRegistry(
    implementations={"metal": _metal, "cuda": _cuda, "triton": _triton},
    name="compute_reservoir_outflow",
    spec=RESERVOIR_OUTFLOW,
)
compute_reservoir_outflow = compute_reservoir_outflow_registry.selected
