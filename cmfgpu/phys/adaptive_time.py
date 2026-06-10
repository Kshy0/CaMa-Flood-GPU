# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Unified adaptive-time-step interface — backend-agnostic."""

from hydroforge.runtime.backend import KERNEL_BACKEND

if KERNEL_BACKEND == "metal":
    from cmfgpu.phys.metal import \
        compute_adaptive_time_step_batched_kernel as _at_b
    from cmfgpu.phys.metal import compute_adaptive_time_step_kernel as _at_nb

    def compute_adaptive_time_step(**kw):
        nt = kw.get("num_trials")
        if nt is not None and nt > 1:
            kw["_grid_size"] = kw["num_catchments"] * nt
            _at_b(**kw)
        else:
            _at_nb(**kw)

elif KERNEL_BACKEND == "cuda":
    from cmfgpu.phys.cuda import compute_adaptive_time_step

elif KERNEL_BACKEND == "triton":
    from hydroforge.runtime.backend import make_triton_dispatcher

    from cmfgpu.phys.triton.adaptive_time import (
        compute_adaptive_time_step_batched_kernel,
        compute_adaptive_time_step_kernel)
    compute_adaptive_time_step = make_triton_dispatcher(
        compute_adaptive_time_step_kernel,
        batched_kernel=compute_adaptive_time_step_batched_kernel,
    )

else:
    raise ValueError(f"Unsupported cmfgpu backend: {KERNEL_BACKEND!r}")
