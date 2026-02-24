# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Backend dispatcher for cmfgpu.phys.outflow.

Imports kernel functions from either the Triton or Torch backend
based on the CMFGPU_BACKEND environment variable (default: triton).
"""

from cmfgpu.phys._backend import KERNEL_BACKEND

if KERNEL_BACKEND == "torch":
    from cmfgpu.phys._backend import adapt_torch_kernel
    from cmfgpu.phys.torch.outflow import (
        compute_outflow_kernel as _compute_outflow_kernel,
        compute_inflow_kernel as _compute_inflow_kernel,
    )
    # compile=False: these kernels split their compilable body from
    # scatter_add_ and handle torch.compile internally.
    compute_outflow_kernel = adapt_torch_kernel(_compute_outflow_kernel, compile=False)
    compute_inflow_kernel = adapt_torch_kernel(_compute_inflow_kernel, compile=False)
    compute_outflow_batched_kernel = None
    compute_inflow_batched_kernel = None
else:
    from cmfgpu.phys.triton.outflow import (  # noqa: F401
        compute_outflow_kernel,
        compute_inflow_kernel,
        compute_outflow_batched_kernel,
        compute_inflow_batched_kernel
    )
