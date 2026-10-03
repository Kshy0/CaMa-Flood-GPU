# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Registered reservoir-outflow implementations."""

from hydroforge.kernels import BackendRegistry, TritonKernel
from cmfgpu.phys import cuda, metal
from cmfgpu.phys.specs import (
    METAL_RESERVOIR_OUTFLOW,
    RESERVOIR_OUTFLOW,
)


def _triton():
    from cmfgpu.phys.triton import reservoir

    return TritonKernel(
        reservoir.compute_reservoir_outflow_kernel,
        batched=reservoir.compute_reservoir_outflow_batched_kernel,
        batch_axis="ensemble_size",
    )


compute_reservoir_outflow = BackendRegistry(
    RESERVOIR_OUTFLOW,
    {
        "metal": metal.RESERVOIR_OUTFLOW,
        "cuda": cuda.RESERVOIR_OUTFLOW,
        "triton": _triton,
    },
    backend_specs={"metal": METAL_RESERVOIR_OUTFLOW},
)
