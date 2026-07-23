# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Registered reservoir-outflow implementations."""

from hydroforge.kernels.registry import BackendRegistry
from cmfgpu.phys.specs import RESERVOIR_OUTFLOW


def _metal():
    from cmfgpu.phys import metal
    return metal.reservoir_outflow()


def _cuda():
    from cmfgpu.phys import cuda
    return cuda.reservoir_outflow()


def _triton():
    from hydroforge.kernels.registry import make_triton_dispatcher
    from cmfgpu.phys.triton.reservoir import (
        compute_reservoir_outflow_batched_kernel,
        compute_reservoir_outflow_kernel,
    )
    return make_triton_dispatcher(
        compute_reservoir_outflow_kernel,
        batched_kernel=compute_reservoir_outflow_batched_kernel,
    )


compute_reservoir_outflow = BackendRegistry(
    {"metal": _metal, "cuda": _cuda, "triton": _triton},
    name="compute_reservoir_outflow",
    spec=RESERVOIR_OUTFLOW,
).selected
