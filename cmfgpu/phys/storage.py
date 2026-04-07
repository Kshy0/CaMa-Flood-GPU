# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Unified flood-stage interface — backend-agnostic."""

from hydroforge.runtime.backend import KERNEL_BACKEND

if KERNEL_BACKEND == "metal":
    from cmfgpu.phys.metal import compute_flood_stage_batched_kernel as _fs_b
    from cmfgpu.phys.metal import compute_flood_stage_kernel as _fs_nb
    from cmfgpu.phys.metal import \
        compute_flood_stage_log_kernel as compute_flood_stage_log

    def compute_flood_stage(**kw):
        nt = kw.get("num_trials")
        if nt is not None and nt > 1:
            _fs_b(**kw)
        else:
            _fs_nb(**kw)

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
