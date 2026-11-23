# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Registered flood-stage implementations."""

from hydroforge.kernels import BackendRegistry, TritonKernel
from cmfgpu.phys import cuda, metal
from cmfgpu.phys.specs import (
    DOWNSCALE_BACKWATER_DEPTH,
    DOWNSCALE_FLOOD_DEPTH,
    DOWNSCALE_FLOOD_DURATION,
    DOWNSCALE_PERMANENT_WATER,
    DOWNSCALE_TRIBUTARY_DEPTH,
    FLOOD_STAGE,
    FLOOD_STAGE_LOG,
    INITIAL_FLOOD_STAGE,
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


def _triton_initial():
    from cmfgpu.phys.triton import storage

    return TritonKernel(
        storage.compute_initial_flood_stage_kernel,
        batched=storage.compute_initial_flood_stage_batched_kernel,
        batch_axis="ensemble_size",
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
compute_initial_flood_stage = BackendRegistry(
    INITIAL_FLOOD_STAGE,
    {
        "metal": metal.INITIAL_FLOOD_STAGE,
        "cuda": cuda.INITIAL_FLOOD_STAGE,
        "triton": _triton_initial,
    },
)


def _triton_downscale_depth():
    from cmfgpu.phys.triton import downscale

    return TritonKernel(
        downscale.compute_downscale_flood_depth_kernel,
        batched=downscale.compute_downscale_flood_depth_batched_kernel,
        batch_axis="ensemble_size",
    )


def _triton_downscale_tributary():
    from cmfgpu.phys.triton import downscale

    return TritonKernel(
        downscale.compute_downscale_tributary_depth_kernel,
        batched=downscale.compute_downscale_tributary_depth_batched_kernel,
        batch_axis="ensemble_size",
    )


def _triton_downscale_backwater():
    from cmfgpu.phys.triton import downscale

    return TritonKernel(
        downscale.compute_downscale_backwater_depth_kernel,
        batched=downscale.compute_downscale_backwater_depth_batched_kernel,
        batch_axis="ensemble_size",
    )


def _triton_downscale_perm():
    from cmfgpu.phys.triton import downscale

    return TritonKernel(
        downscale.compute_downscale_permanent_water_kernel,
        batched=downscale.compute_downscale_permanent_water_batched_kernel,
        batch_axis="ensemble_size",
    )


def _triton_downscale_duration():
    from cmfgpu.phys.triton import downscale

    return TritonKernel(
        downscale.compute_downscale_flood_duration_kernel,
        batched=downscale.compute_downscale_flood_duration_batched_kernel,
        batch_axis="ensemble_size",
    )


compute_downscale_flood_depth = BackendRegistry(
    DOWNSCALE_FLOOD_DEPTH,
    {
        "metal": metal.DOWNSCALE_FLOOD_DEPTH,
        "cuda": cuda.DOWNSCALE_FLOOD_DEPTH,
        "triton": _triton_downscale_depth,
    },
)
compute_downscale_tributary_depth = BackendRegistry(
    DOWNSCALE_TRIBUTARY_DEPTH,
    {
        "metal": metal.DOWNSCALE_TRIBUTARY_DEPTH,
        "cuda": cuda.DOWNSCALE_TRIBUTARY_DEPTH,
        "triton": _triton_downscale_tributary,
    },
)
compute_downscale_backwater_depth = BackendRegistry(
    DOWNSCALE_BACKWATER_DEPTH,
    {
        "metal": metal.DOWNSCALE_BACKWATER_DEPTH,
        "cuda": cuda.DOWNSCALE_BACKWATER_DEPTH,
        "triton": _triton_downscale_backwater,
    },
)
compute_downscale_permanent_water = BackendRegistry(
    DOWNSCALE_PERMANENT_WATER,
    {
        "metal": metal.DOWNSCALE_PERMANENT_WATER,
        "cuda": cuda.DOWNSCALE_PERMANENT_WATER,
        "triton": _triton_downscale_perm,
    },
)
compute_downscale_flood_duration = BackendRegistry(
    DOWNSCALE_FLOOD_DURATION,
    {
        "metal": metal.DOWNSCALE_FLOOD_DURATION,
        "cuda": cuda.DOWNSCALE_FLOOD_DURATION,
        "triton": _triton_downscale_duration,
    },
)
