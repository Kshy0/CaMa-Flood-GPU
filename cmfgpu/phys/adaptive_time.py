# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Registered adaptive-time implementations."""

from hydroforge.kernels import BackendRegistry, TritonKernel
from cmfgpu.phys import cuda, metal
from cmfgpu.phys.specs import ADAPTIVE_TIME


def _triton():
    from cmfgpu.phys.triton import adaptive_time

    return TritonKernel(
        adaptive_time.compute_adaptive_time_step_kernel,
        batched=adaptive_time.compute_adaptive_time_step_batched_kernel,
        batch_axis="ensemble_size",
    )


compute_adaptive_time_step = BackendRegistry(
    ADAPTIVE_TIME,
    {"metal": metal.ADAPTIVE_TIME, "cuda": cuda.ADAPTIVE_TIME, "triton": _triton},
)
