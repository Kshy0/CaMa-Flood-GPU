# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Unified bifurcation outflow / inflow interface — backend-agnostic."""

from hydroforge.runtime.backend import KERNEL_BACKEND

if KERNEL_BACKEND == "metal":
    from cmfgpu.phys.metal import \
        compute_bifurcation_inflow_batched_kernel as _bi_b
    from cmfgpu.phys.metal import compute_bifurcation_inflow_kernel as _bi_nb
    from cmfgpu.phys.metal import \
        compute_bifurcation_outflow_batched_kernel as _bo_b
    from cmfgpu.phys.metal import compute_bifurcation_outflow_kernel as _bo_nb

    def compute_bifurcation_outflow(**kw):
        nt = kw.get("num_trials")
        if nt is not None and nt > 1:
            kw["_grid_size"] = kw["num_bifurcation_paths"] * nt
            _bo_b(**kw)
        else:
            _bo_nb(**kw)

    def compute_bifurcation_inflow(**kw):
        nt = kw.get("num_trials")
        if nt is not None and nt > 1:
            kw["_grid_size"] = kw["num_bifurcation_paths"] * nt
            _bi_b(**kw)
        else:
            _bi_nb(**kw)

elif KERNEL_BACKEND == "cuda":
    from cmfgpu.phys.cuda import (compute_bifurcation_inflow,
                                  compute_bifurcation_outflow)

elif KERNEL_BACKEND == "triton":
    from hydroforge.runtime.backend import make_triton_dispatcher

    from cmfgpu.phys.triton.bifurcation import (
        compute_bifurcation_inflow_batched_kernel,
        compute_bifurcation_inflow_kernel,
        compute_bifurcation_outflow_batched_kernel,
        compute_bifurcation_outflow_kernel)
    compute_bifurcation_outflow = make_triton_dispatcher(
        compute_bifurcation_outflow_kernel,
        batched_kernel=compute_bifurcation_outflow_batched_kernel,
        size_key="num_bifurcation_paths",
        non_batched_drop=("num_catchments",),
    )
    compute_bifurcation_inflow = make_triton_dispatcher(
        compute_bifurcation_inflow_kernel,
        batched_kernel=compute_bifurcation_inflow_batched_kernel,
        size_key="num_bifurcation_paths",
        non_batched_drop=("num_catchments",),
    )

else:
    raise ValueError(f"Unsupported cmfgpu backend: {KERNEL_BACKEND!r}")
