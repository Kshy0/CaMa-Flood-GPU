# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Unified flood-stage interface — backend-agnostic."""

from hydroforge.runtime.backend import KERNEL_BACKEND

if KERNEL_BACKEND == "metal":
    from cmfgpu.phys.metal import \
        compute_flood_stage_kernel as compute_flood_stage  # noqa: F401
    from cmfgpu.phys.metal import \
        compute_flood_stage_log_kernel as compute_flood_stage_log

elif KERNEL_BACKEND == "torch":
    from hydroforge.runtime.backend import adapt_kernel

    from cmfgpu.phys.torch.storage import \
        compute_flood_stage_kernel as _raw_stage
    from cmfgpu.phys.torch.storage import \
        compute_flood_stage_log_kernel as _raw_stage_log
    compute_flood_stage = adapt_kernel(_raw_stage)
    compute_flood_stage_log = adapt_kernel(_raw_stage_log)

else:  # triton
    from hydroforge.runtime.backend import make_triton_dispatcher

    from cmfgpu.phys.triton.storage import (compute_flood_stage_batched_kernel,
                                            compute_flood_stage_kernel,
                                            compute_flood_stage_log_kernel)
    compute_flood_stage = make_triton_dispatcher(
        compute_flood_stage_kernel, batched_kernel=compute_flood_stage_batched_kernel,
        batched_grid="loop",
    )
    compute_flood_stage_log = make_triton_dispatcher(compute_flood_stage_log_kernel)
