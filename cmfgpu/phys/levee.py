# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Registered levee implementations."""

from hydroforge.kernels import BackendRegistry, TritonKernel
from cmfgpu.phys import cuda, metal
from cmfgpu.phys.specs import (
    LEVEE_BIFURCATION_OUTFLOW,
    LEVEE_STAGE,
    LEVEE_STAGE_LOG,
    METAL_LEVEE_BIFURCATION_OUTFLOW,
)


def _triton_stage():
    from cmfgpu.phys.triton import levee

    return TritonKernel(
        levee.compute_levee_stage_kernel,
        batched=levee.compute_levee_stage_batched_kernel,
        batch_axis="ensemble_size",
        batch_layout="loop",
    )


def _triton_log():
    from cmfgpu.phys.triton.levee import compute_levee_stage_log_kernel

    return TritonKernel(compute_levee_stage_log_kernel)


def _triton_bif():
    from cmfgpu.phys.triton import levee

    return TritonKernel(
        levee.compute_levee_bifurcation_outflow_kernel,
        batched=levee.compute_levee_bifurcation_outflow_batched_kernel,
        batch_axis="ensemble_size",
    )


compute_levee_stage = BackendRegistry(
    LEVEE_STAGE,
    {"metal": metal.LEVEE_STAGE, "cuda": cuda.LEVEE_STAGE, "triton": _triton_stage},
)
compute_levee_stage_log = BackendRegistry(
    LEVEE_STAGE_LOG,
    {
        "metal": metal.LEVEE_STAGE_LOG,
        "cuda": cuda.LEVEE_STAGE_LOG,
        "triton": _triton_log,
    },
)
compute_levee_bifurcation_outflow = BackendRegistry(
    LEVEE_BIFURCATION_OUTFLOW,
    {
        "metal": metal.LEVEE_BIFURCATION_OUTFLOW,
        "cuda": cuda.LEVEE_BIFURCATION_OUTFLOW,
        "triton": _triton_bif,
    },
    backend_specs={"metal": METAL_LEVEE_BIFURCATION_OUTFLOW},
)
