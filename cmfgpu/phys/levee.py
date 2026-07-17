# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Registered levee implementations."""

from hydroforge.runtime.backend import (
    BackendRegistry, make_batched_dispatcher, make_triton_dispatcher,
)


def _metal_stage():
    from cmfgpu.phys.metal import (
        compute_levee_stage_batched_kernel, compute_levee_stage_kernel,
    )
    return make_batched_dispatcher(
        compute_levee_stage_kernel, compute_levee_stage_batched_kernel,
    )


def _metal_log():
    from cmfgpu.phys.metal import compute_levee_stage_log_kernel
    return compute_levee_stage_log_kernel


def _metal_bif():
    from cmfgpu.phys.metal import (
        compute_levee_bifurcation_outflow_batched_kernel,
        compute_levee_bifurcation_outflow_kernel,
    )
    return make_batched_dispatcher(
        compute_levee_bifurcation_outflow_kernel,
        compute_levee_bifurcation_outflow_batched_kernel,
    )


def _cuda(name):
    def factory():
        from cmfgpu.phys import cuda
        return getattr(cuda, name)
    return factory


def _triton_stage():
    from cmfgpu.phys.triton.levee import (
        compute_levee_stage_batched_kernel, compute_levee_stage_kernel,
    )
    return make_triton_dispatcher(
        compute_levee_stage_kernel, batched_kernel=compute_levee_stage_batched_kernel,
        size_key="num_levees", batched_grid="loop",
        non_batched_drop=("num_catchments",),
    )


def _triton_log():
    from cmfgpu.phys.triton.levee import compute_levee_stage_log_kernel
    return make_triton_dispatcher(compute_levee_stage_log_kernel, size_key="num_levees")


def _triton_bif():
    from cmfgpu.phys.triton.levee import (
        compute_levee_bifurcation_outflow_batched_kernel,
        compute_levee_bifurcation_outflow_kernel,
    )
    return make_triton_dispatcher(
        compute_levee_bifurcation_outflow_kernel,
        batched_kernel=compute_levee_bifurcation_outflow_batched_kernel,
        size_key="num_bifurcation_paths", non_batched_drop=("num_catchments",),
    )


compute_levee_stage = BackendRegistry(
    {"metal": _metal_stage, "cuda": _cuda("compute_levee_stage"), "triton": _triton_stage},
    name="compute_levee_stage",
).selected
compute_levee_stage_log = BackendRegistry(
    {"metal": _metal_log, "cuda": _cuda("compute_levee_stage_log"), "triton": _triton_log},
    name="compute_levee_stage_log",
).selected
compute_levee_bifurcation_outflow = BackendRegistry(
    {"metal": _metal_bif, "cuda": _cuda("compute_levee_bifurcation_outflow"),
     "triton": _triton_bif},
    name="compute_levee_bifurcation_outflow",
).selected
