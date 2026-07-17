# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Registered adaptive-time implementations."""

from hydroforge.runtime.backend import BackendRegistry, make_batched_dispatcher


def _metal():
    from cmfgpu.phys.metal import (
        compute_adaptive_time_step_batched_kernel as batched,
        compute_adaptive_time_step_kernel as shared,
    )

    return make_batched_dispatcher(shared, batched)


def _cuda():
    from cmfgpu.phys.cuda import compute_adaptive_time_step
    return compute_adaptive_time_step


def _triton():
    from hydroforge.runtime.backend import make_triton_dispatcher
    from cmfgpu.phys.triton.adaptive_time import (
        compute_adaptive_time_step_batched_kernel,
        compute_adaptive_time_step_kernel,
    )
    return make_triton_dispatcher(
        compute_adaptive_time_step_kernel,
        batched_kernel=compute_adaptive_time_step_batched_kernel,
        optional_buffers={"is_dam_related_ptr": "HAS_RESERVOIR"},
    )


compute_adaptive_time_step = BackendRegistry(
    {"metal": _metal, "cuda": _cuda, "triton": _triton},
    name="compute_adaptive_time_step",
).selected
