# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Unified adaptive-time-step interface — backend-agnostic."""

from hydroforge.runtime.backend import KERNEL_BACKEND

if KERNEL_BACKEND == "cuda":
    from cmfgpu.phys.cuda import \
        compute_adaptive_time_step_kernel as \
        compute_adaptive_time_step  # noqa: F401

elif KERNEL_BACKEND == "metal":
    from cmfgpu.phys.metal import \
        compute_adaptive_time_step_kernel as \
        compute_adaptive_time_step  # noqa: F401

elif KERNEL_BACKEND == "torch":
    from hydroforge.runtime.backend import adapt_kernel

    from cmfgpu.phys.torch.adaptive_time import \
        compute_adaptive_time_step_kernel as _raw_adaptive
    compute_adaptive_time_step = adapt_kernel(_raw_adaptive)

else:  # triton
    from hydroforge.runtime.backend import make_triton_dispatcher

    from cmfgpu.phys.triton.adaptive_time import (
        compute_adaptive_time_step_batched_kernel,
        compute_adaptive_time_step_kernel)
    compute_adaptive_time_step = make_triton_dispatcher(
        compute_adaptive_time_step_kernel,
        batched_kernel=compute_adaptive_time_step_batched_kernel,
    )
