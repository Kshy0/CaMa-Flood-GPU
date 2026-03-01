# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Backend dispatcher for cmfgpu.phys.reservoir.

Imports reservoir outflow kernel functions from the Triton backend.
"""

from cmfgpu.phys._backend import KERNEL_BACKEND

if KERNEL_BACKEND == "torch":
    from cmfgpu.phys._backend import adapt_torch_kernel
    from cmfgpu.phys.torch.reservoir import (
        compute_reservoir_outflow_kernel as _compute_reservoir_outflow_kernel,
    )
    # compile=False: kernel uses scatter_add_ which breaks torch.compile graph
    compute_reservoir_outflow_kernel = adapt_torch_kernel(
        _compute_reservoir_outflow_kernel, compile=False
    )
else:
    from cmfgpu.phys.triton.reservoir import (  # noqa: F401
        compute_reservoir_outflow_kernel,
    )
