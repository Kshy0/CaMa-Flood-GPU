# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Unified outflow / inflow interface — backend-agnostic."""

from hydroforge.runtime.backend import KERNEL_BACKEND

if KERNEL_BACKEND == "cuda":
    from cmfgpu.phys.cuda import compute_inflow_kernel as compute_inflow
    from cmfgpu.phys.cuda import \
        compute_outflow_kernel as compute_outflow  # noqa: F401

elif KERNEL_BACKEND == "hip":
    from cmfgpu.phys.hip import compute_inflow_kernel as compute_inflow
    from cmfgpu.phys.hip import \
        compute_outflow_kernel as compute_outflow  # noqa: F401

elif KERNEL_BACKEND == "metal":
    from cmfgpu.phys.metal import compute_inflow_kernel as compute_inflow
    from cmfgpu.phys.metal import \
        compute_outflow_kernel as compute_outflow  # noqa: F401

elif KERNEL_BACKEND == "torch":
    from hydroforge.runtime.backend import adapt_kernel

    from cmfgpu.phys.torch.outflow import compute_inflow_kernel as _raw_inflow
    from cmfgpu.phys.torch.outflow import \
        compute_outflow_kernel as _raw_outflow
    compute_outflow = adapt_kernel(_raw_outflow, compile=False)
    compute_inflow = adapt_kernel(_raw_inflow, compile=False)

else:  # triton
    from hydroforge.runtime.backend import make_triton_dispatcher

    from cmfgpu.phys.triton.outflow import (compute_inflow_batched_kernel,
                                            compute_inflow_kernel,
                                            compute_outflow_batched_kernel,
                                            compute_outflow_kernel)
    compute_outflow = make_triton_dispatcher(
        compute_outflow_kernel, batched_kernel=compute_outflow_batched_kernel,
        batched_drop=("is_dam_upstream_ptr", "HAS_RESERVOIR", "MIN_KINEMATIC_SLOPE"),
    )
    compute_inflow = make_triton_dispatcher(
        compute_inflow_kernel, batched_kernel=compute_inflow_batched_kernel,
    )
