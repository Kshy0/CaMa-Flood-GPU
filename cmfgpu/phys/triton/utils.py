# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Utility helpers for Triton physics kernels."""

import torch
import triton
from triton.language.extra import libdevice
import triton.language as tl


IS_HIP_BUILD = torch.version.hip is not None


@triton.jit
def hpfloat_to_compute_inline(hp_value, ref_value):
    """Downcast a high-precision (hpfloat) value to match the base computation dtype.

    In the GPU code, ``hp_value`` is loaded from an hpfloat pointer (e.g. float64
    when mixed_precision is enabled) and ``ref_value`` is any variable already in
    base precision (e.g. float32).  The cast ensures all subsequent arithmetic
    stays in the selected computation dtype.

    When mixed_precision is disabled both dtypes are identical and the ``.to()``
    compiles to a no-op with zero overhead.
    """
    return hp_value.to(ref_value.dtype)


@triton.jit
def nonnegative_to_index_inline(value):
    """Quantize an already integral, non-negative algorithmic count."""
    return value.to(tl.int32)


if IS_HIP_BUILD:
    @triton.jit
    def cbrt_compat_inline(x):
        # HIP libdevice may not expose cbrt; use pow(x, 1/3) with dtype-matched exponent.
        one_third = tl.zeros_like(x) + (1.0 / 3.0)
        return libdevice.pow(x, one_third)
else:
    @triton.jit
    def cbrt_compat_inline(x):
        return libdevice.cbrt(x)
