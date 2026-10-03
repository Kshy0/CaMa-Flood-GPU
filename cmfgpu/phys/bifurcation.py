# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Registered bifurcation implementations."""

from hydroforge.kernels import BackendRegistry, TritonKernel
from cmfgpu.phys import cuda, metal
from cmfgpu.phys.specs import (
    BIFURCATION_INFLOW,
    BIFURCATION_OUTFLOW,
    METAL_BIFURCATION_INFLOW,
    METAL_BIFURCATION_OUTFLOW,
)


def _triton_outflow():
    from cmfgpu.phys.triton import bifurcation

    return TritonKernel(
        bifurcation.compute_bifurcation_outflow_kernel,
        batched=bifurcation.compute_bifurcation_outflow_batched_kernel,
        batch_axis="ensemble_size",
    )


def _triton_inflow():
    from cmfgpu.phys.triton import bifurcation

    return TritonKernel(
        bifurcation.compute_bifurcation_inflow_kernel,
        batched=bifurcation.compute_bifurcation_inflow_batched_kernel,
        batch_axis="ensemble_size",
    )


compute_bifurcation_outflow = BackendRegistry(
    BIFURCATION_OUTFLOW,
    {
        "metal": metal.BIFURCATION_OUTFLOW,
        "cuda": cuda.BIFURCATION_OUTFLOW,
        "triton": _triton_outflow,
    },
    backend_specs={"metal": METAL_BIFURCATION_OUTFLOW},
)
compute_bifurcation_inflow = BackendRegistry(
    BIFURCATION_INFLOW,
    {
        "metal": metal.BIFURCATION_INFLOW,
        "cuda": cuda.BIFURCATION_INFLOW,
        "triton": _triton_inflow,
    },
    backend_specs={"metal": METAL_BIFURCATION_INFLOW},
)
