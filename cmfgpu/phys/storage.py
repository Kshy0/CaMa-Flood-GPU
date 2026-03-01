# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Backend dispatcher for cmfgpu.phys.storage.

Imports kernel functions from either the Triton or Torch backend
based on the CMFGPU_BACKEND environment variable (default: triton).
"""

from cmfgpu.phys._backend import KERNEL_BACKEND

if KERNEL_BACKEND == "torch":
    from cmfgpu.phys._backend import adapt_torch_kernel
    from cmfgpu.phys.torch.storage import \
        compute_flood_stage_kernel as _compute_flood_stage_kernel
    from cmfgpu.phys.torch.storage import \
        compute_flood_stage_log_kernel as _compute_flood_stage_log_kernel
    compute_flood_stage_kernel = adapt_torch_kernel(_compute_flood_stage_kernel)
    compute_flood_stage_log_kernel = adapt_torch_kernel(_compute_flood_stage_log_kernel)
    compute_flood_stage_batched_kernel = None
else:
    from cmfgpu.phys.triton.storage import (  # noqa: F401
        compute_flood_stage_batched_kernel, compute_flood_stage_kernel,
        compute_flood_stage_log_kernel)
