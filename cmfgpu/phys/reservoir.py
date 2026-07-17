# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Registered reservoir-outflow implementations."""

from hydroforge.runtime.backend import BackendRegistry, make_batched_dispatcher


def _metal():
    from cmfgpu.phys.metal import (
        compute_reservoir_outflow_batched_kernel as batched,
        compute_reservoir_outflow_kernel as shared,
    )

    return make_batched_dispatcher(shared, batched)


def _cuda():
    from cmfgpu.phys.cuda import compute_reservoir_outflow
    return compute_reservoir_outflow


def _triton():
    from hydroforge.runtime.backend import make_triton_dispatcher
    from cmfgpu.phys.triton.reservoir import compute_reservoir_outflow_kernel
    return make_triton_dispatcher(
        compute_reservoir_outflow_kernel, size_key="num_reservoirs",
    )


compute_reservoir_outflow = BackendRegistry(
    {"metal": _metal, "cuda": _cuda, "triton": _triton},
    name="compute_reservoir_outflow",
).selected
