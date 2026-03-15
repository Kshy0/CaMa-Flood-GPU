# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Unified bifurcation outflow / inflow interface — backend-agnostic."""

from hydroforge.runtime.backend import KERNEL_BACKEND

if KERNEL_BACKEND == "cuda":
    from cmfgpu.phys.cuda import \
        compute_bifurcation_inflow_kernel as compute_bifurcation_inflow
    from cmfgpu.phys.cuda import \
        compute_bifurcation_outflow_kernel as \
        compute_bifurcation_outflow  # noqa: F401

elif KERNEL_BACKEND == "hip":
    from cmfgpu.phys.hip import \
        compute_bifurcation_inflow_kernel as compute_bifurcation_inflow
    from cmfgpu.phys.hip import \
        compute_bifurcation_outflow_kernel as \
        compute_bifurcation_outflow  # noqa: F401

elif KERNEL_BACKEND == "metal":
    from cmfgpu.phys.metal import \
        compute_bifurcation_inflow_kernel as compute_bifurcation_inflow
    from cmfgpu.phys.metal import \
        compute_bifurcation_outflow_kernel as \
        compute_bifurcation_outflow  # noqa: F401

elif KERNEL_BACKEND == "torch":
    from hydroforge.runtime.backend import adapt_kernel

    from cmfgpu.phys.torch.bifurcation import \
        compute_bifurcation_inflow_kernel as _raw_bif_in
    from cmfgpu.phys.torch.bifurcation import \
        compute_bifurcation_outflow_kernel as _raw_bif_out
    compute_bifurcation_outflow = adapt_kernel(_raw_bif_out, compile=False)
    compute_bifurcation_inflow = adapt_kernel(_raw_bif_in, compile=False)

else:  # triton
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
