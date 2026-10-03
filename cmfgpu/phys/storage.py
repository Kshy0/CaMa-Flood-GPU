# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Registered flood-stage implementations."""

from hydroforge.kernels import BackendRegistry, TritonKernel
from cmfgpu.phys import cuda, metal
from cmfgpu.phys.specs import (
    FLOOD_STAGE,
    FLOOD_STAGE_LOG,
    METAL_FLOOD_STAGE,
    METAL_FLOOD_STAGE_LOG,
)


def _triton_stage():
    from cmfgpu.phys.triton import storage

    return TritonKernel(
        storage.compute_flood_stage_kernel,
        batched=storage.compute_flood_stage_batched_kernel,
        batch_axis="ensemble_size",
        batch_layout="loop",
    )


def _triton_log():
    from cmfgpu.phys.triton.storage import compute_flood_stage_log_kernel

    return TritonKernel(compute_flood_stage_log_kernel)


compute_flood_stage = BackendRegistry(
    FLOOD_STAGE,
    {"metal": metal.FLOOD_STAGE, "cuda": cuda.FLOOD_STAGE, "triton": _triton_stage},
    backend_specs={"metal": METAL_FLOOD_STAGE},
)
compute_flood_stage_log = BackendRegistry(
    FLOOD_STAGE_LOG,
    {
        "metal": metal.FLOOD_STAGE_LOG,
        "cuda": cuda.FLOOD_STAGE_LOG,
        "triton": _triton_log,
    },
    backend_specs={"metal": METAL_FLOOD_STAGE_LOG},
)
