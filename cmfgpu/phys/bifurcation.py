# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Registered bifurcation implementations."""

from hydroforge.kernels.registry import (
    BackendRegistry, make_triton_dispatcher,
)
from cmfgpu.phys.specs import BIFURCATION_INFLOW, BIFURCATION_OUTFLOW


def _metal_outflow():
    from cmfgpu.phys import metal
    return metal.bifurcation_outflow()


def _metal_inflow():
    from cmfgpu.phys import metal
    return metal.bifurcation_inflow()


def _cuda_outflow():
    from cmfgpu.phys import cuda
    return cuda.bifurcation_outflow()


def _cuda_inflow():
    from cmfgpu.phys import cuda
    return cuda.bifurcation_inflow()


def _triton(which):
    def factory():
        from cmfgpu.phys.triton import bifurcation
        return make_triton_dispatcher(
            getattr(bifurcation, f"compute_bifurcation_{which}_kernel"),
            batched_kernel=getattr(
                bifurcation, f"compute_bifurcation_{which}_batched_kernel",
            ),
        )
    return factory


compute_bifurcation_outflow = BackendRegistry(
    {
        "metal": _metal_outflow,
        "cuda": _cuda_outflow,
        "triton": _triton("outflow"),
    },
    name="compute_bifurcation_outflow",
    spec=BIFURCATION_OUTFLOW,
).selected
compute_bifurcation_inflow = BackendRegistry(
    {
        "metal": _metal_inflow,
        "cuda": _cuda_inflow,
        "triton": _triton("inflow"),
    },
    name="compute_bifurcation_inflow",
    spec=BIFURCATION_INFLOW,
).selected
