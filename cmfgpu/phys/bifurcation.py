# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Registered bifurcation implementations."""

from hydroforge.runtime.backend import (
    BackendRegistry, make_batched_dispatcher, make_triton_dispatcher,
)


def _metal(which):
    def factory():
        from cmfgpu.phys import metal
        shared = getattr(metal, f"compute_bifurcation_{which}_kernel")
        batched = getattr(metal, f"compute_bifurcation_{which}_batched_kernel")
        return make_batched_dispatcher(shared, batched)
    return factory


def _cuda(which):
    def factory():
        from cmfgpu.phys import cuda
        return getattr(cuda, f"compute_bifurcation_{which}")
    return factory


def _triton(which):
    def factory():
        from cmfgpu.phys.triton import bifurcation
        return make_triton_dispatcher(
            getattr(bifurcation, f"compute_bifurcation_{which}_kernel"),
            batched_kernel=getattr(
                bifurcation, f"compute_bifurcation_{which}_batched_kernel",
            ),
            size_key="num_bifurcation_paths",
            non_batched_drop=("num_catchments",),
        )
    return factory


def _registered(which):
    return BackendRegistry(
        {"metal": _metal(which), "cuda": _cuda(which), "triton": _triton(which)},
        name=f"compute_bifurcation_{which}",
    ).selected


compute_bifurcation_outflow = _registered("outflow")
compute_bifurcation_inflow = _registered("inflow")
