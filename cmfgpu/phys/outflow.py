# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Registered main-channel outflow and inflow implementations."""

from hydroforge.kernels import BackendRegistry, TritonKernel
from cmfgpu.phys import cuda, metal
from cmfgpu.phys.specs import (
    INFLOW,
    METAL_INFLOW,
    METAL_OUTFLOW,
    OUTFLOW,
)


def _triton_outflow():
    from cmfgpu.phys.triton import outflow

    return TritonKernel(
        outflow.compute_outflow_kernel,
        batched=outflow.compute_outflow_batched_kernel,
        batch_axis="ensemble_size",
    )


def _triton_inflow():
    from cmfgpu.phys.triton import outflow

    return TritonKernel(
        outflow.compute_inflow_kernel,
        batched=outflow.compute_inflow_batched_kernel,
        batch_axis="ensemble_size",
    )


compute_outflow = BackendRegistry(
    OUTFLOW,
    {"metal": metal.OUTFLOW, "cuda": cuda.OUTFLOW, "triton": _triton_outflow},
    backend_specs={"metal": METAL_OUTFLOW},
)
compute_inflow = BackendRegistry(
    INFLOW,
    {"metal": metal.INFLOW, "cuda": cuda.INFLOW, "triton": _triton_inflow},
    backend_specs={"metal": METAL_INFLOW},
)
