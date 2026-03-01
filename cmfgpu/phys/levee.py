# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Backend dispatcher for cmfgpu.phys.levee.

Imports kernel functions from either the Triton or Torch backend
based on the CMFGPU_BACKEND environment variable (default: triton).
"""

from cmfgpu.phys._backend import KERNEL_BACKEND

if KERNEL_BACKEND == "torch":
    from cmfgpu.phys._backend import adapt_torch_kernel
    from cmfgpu.phys.torch.levee import \
        compute_levee_bifurcation_outflow_kernel as \
        _compute_levee_bifurcation_outflow_kernel
    from cmfgpu.phys.torch.levee import \
        compute_levee_stage_kernel as _compute_levee_stage_kernel
    from cmfgpu.phys.torch.levee import \
        compute_levee_stage_log_kernel as _compute_levee_stage_log_kernel
    compute_levee_stage_kernel = adapt_torch_kernel(_compute_levee_stage_kernel)
    compute_levee_stage_log_kernel = adapt_torch_kernel(_compute_levee_stage_log_kernel)
    # compile=False: this kernel splits its compilable body from
    # scatter_add_ and handles torch.compile internally.
    compute_levee_bifurcation_outflow_kernel = adapt_torch_kernel(_compute_levee_bifurcation_outflow_kernel, compile=False)
    compute_levee_stage_batched_kernel = None
    compute_levee_bifurcation_outflow_batched_kernel = None
else:
    from cmfgpu.phys.triton.levee import (  # noqa: F401
        compute_levee_bifurcation_outflow_batched_kernel,
        compute_levee_bifurcation_outflow_kernel,
        compute_levee_stage_batched_kernel, compute_levee_stage_kernel,
        compute_levee_stage_log_kernel)
