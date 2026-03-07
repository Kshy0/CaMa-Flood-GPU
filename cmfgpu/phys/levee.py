# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Unified levee interface — backend-agnostic."""

from hydroforge.runtime.backend import KERNEL_BACKEND

if KERNEL_BACKEND == "cuda":
    from cmfgpu.phys.cuda import \
        compute_levee_bifurcation_outflow_kernel as \
        compute_levee_bifurcation_outflow
    from cmfgpu.phys.cuda import \
        compute_levee_stage_kernel as compute_levee_stage  # noqa: F401
    from cmfgpu.phys.cuda import \
        compute_levee_stage_log_kernel as compute_levee_stage_log

elif KERNEL_BACKEND == "metal":
    from cmfgpu.phys.metal import \
        compute_levee_bifurcation_outflow_kernel as \
        compute_levee_bifurcation_outflow
    from cmfgpu.phys.metal import \
        compute_levee_stage_kernel as compute_levee_stage  # noqa: F401
    from cmfgpu.phys.metal import \
        compute_levee_stage_log_kernel as compute_levee_stage_log

elif KERNEL_BACKEND == "torch":
    from hydroforge.runtime.backend import adapt_kernel

    from cmfgpu.phys.torch.levee import \
        compute_levee_bifurcation_outflow_kernel as _raw_levee_bif
    from cmfgpu.phys.torch.levee import \
        compute_levee_stage_kernel as _raw_levee_stage
    from cmfgpu.phys.torch.levee import \
        compute_levee_stage_log_kernel as _raw_levee_stage_log
    compute_levee_stage = adapt_kernel(_raw_levee_stage)
    compute_levee_stage_log = adapt_kernel(_raw_levee_stage_log)
    compute_levee_bifurcation_outflow = adapt_kernel(_raw_levee_bif, compile=False)

else:  # triton
    from hydroforge.runtime.backend import make_triton_dispatcher

    from cmfgpu.phys.triton.levee import (
        compute_levee_bifurcation_outflow_batched_kernel,
        compute_levee_bifurcation_outflow_kernel,
        compute_levee_stage_batched_kernel, compute_levee_stage_kernel,
        compute_levee_stage_log_kernel)
    compute_levee_stage = make_triton_dispatcher(
        compute_levee_stage_kernel, batched_kernel=compute_levee_stage_batched_kernel,
        size_key="num_levees", batched_grid="loop",
        non_batched_drop=("num_catchments",),
    )
    compute_levee_stage_log = make_triton_dispatcher(
        compute_levee_stage_log_kernel, size_key="num_levees")
    compute_levee_bifurcation_outflow = make_triton_dispatcher(
        compute_levee_bifurcation_outflow_kernel,
        batched_kernel=compute_levee_bifurcation_outflow_batched_kernel,
        size_key="num_bifurcation_paths",
        non_batched_drop=("num_catchments",),
    )
