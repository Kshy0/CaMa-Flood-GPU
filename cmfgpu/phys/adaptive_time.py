# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Registered adaptive-time implementations."""

from hydroforge.kernels.registry import BackendRegistry
from cmfgpu.phys.specs import ADAPTIVE_TIME


def _metal():
    from cmfgpu.phys import metal

    return metal.adaptive_time()


def _cuda():
    from cmfgpu.phys import cuda
    return cuda.adaptive_time()


def _triton():
    from hydroforge.kernels.registry import make_triton_dispatcher
    from cmfgpu.phys.triton.adaptive_time import (
        compute_adaptive_time_step_batched_kernel,
        compute_adaptive_time_step_kernel,
    )
    return make_triton_dispatcher(
        compute_adaptive_time_step_kernel,
        batched_kernel=compute_adaptive_time_step_batched_kernel,
    )


compute_adaptive_time_step = BackendRegistry(
    {"metal": _metal, "cuda": _cuda, "triton": _triton},
    name="compute_adaptive_time_step",
    spec=ADAPTIVE_TIME,
).selected
