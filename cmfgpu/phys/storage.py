# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Registered flood-stage implementations."""

from hydroforge.kernels.registry import BackendRegistry
from cmfgpu.phys.specs import FLOOD_STAGE, FLOOD_STAGE_LOG


def _metal_stage():
    from cmfgpu.phys import metal
    return metal.flood_stage()


def _metal_log():
    from cmfgpu.phys import metal
    return metal.flood_stage_log()


def _cuda_stage():
    from cmfgpu.phys import cuda
    return cuda.flood_stage()


def _cuda_log():
    from cmfgpu.phys import cuda
    return cuda.flood_stage_log()


def _triton_stage():
    from hydroforge.kernels.registry import make_triton_dispatcher
    from cmfgpu.phys.triton.storage import (
        compute_flood_stage_batched_kernel, compute_flood_stage_kernel,
    )
    return make_triton_dispatcher(
        compute_flood_stage_kernel, batched_kernel=compute_flood_stage_batched_kernel,
        batched_grid="loop",
    )


def _triton_log():
    from hydroforge.kernels.registry import make_triton_dispatcher
    from cmfgpu.phys.triton.storage import compute_flood_stage_log_kernel
    return make_triton_dispatcher(compute_flood_stage_log_kernel)


compute_flood_stage = BackendRegistry(
    {"metal": _metal_stage, "cuda": _cuda_stage, "triton": _triton_stage},
    name="compute_flood_stage",
    spec=FLOOD_STAGE,
).selected
compute_flood_stage_log = BackendRegistry(
    {"metal": _metal_log, "cuda": _cuda_log, "triton": _triton_log},
    name="compute_flood_stage_log",
    spec=FLOOD_STAGE_LOG,
).selected
