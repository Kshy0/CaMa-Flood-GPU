# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Unified reservoir-outflow interface — backend-agnostic."""

from hydroforge.runtime.backend import KERNEL_BACKEND

if KERNEL_BACKEND == "metal":
    from cmfgpu.phys.metal import \
        compute_reservoir_outflow_batched_kernel as _ro_b
    from cmfgpu.phys.metal import compute_reservoir_outflow_kernel as _ro_nb

    def compute_reservoir_outflow(**kw):
        nt = kw.get("num_trials")
        if nt is not None and nt > 1:
            kw["_grid_size"] = kw["num_reservoirs"] * nt
            _ro_b(**kw)
        else:
            _ro_nb(**kw)

elif KERNEL_BACKEND == "torch":
    from hydroforge.runtime.backend import adapt_kernel

    from cmfgpu.phys.torch.reservoir import \
        compute_reservoir_outflow_kernel as _raw_reservoir
    compute_reservoir_outflow = adapt_kernel(_raw_reservoir, compile=False)

else:  # triton
    from hydroforge.runtime.backend import make_triton_dispatcher

    from cmfgpu.phys.triton.reservoir import compute_reservoir_outflow_kernel
    compute_reservoir_outflow = make_triton_dispatcher(
        compute_reservoir_outflow_kernel, size_key="num_reservoirs")
