# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Utility helpers for Triton physics kernels."""

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def typed_pow(base, exp: tl.constexpr):
    """Type-safe pow() wrapping libdevice.pow.

    ``libdevice.pow`` requires both arguments to share the same dtype and
    only supports fp32 / fp64.  This helper upcasts fp16 to fp32,
    calls pow, and casts back — analogous to ``typed_sqrt``.
    """
    compute_dtype = tl.float64 if base.dtype == tl.float64 else tl.float32
    base_up = base.to(compute_dtype)
    exp_tensor = tl.full(base_up.shape, exp, dtype=compute_dtype)
    return libdevice.pow(base_up, exp_tensor).to(base.dtype)


@triton.jit
def typed_sqrt(x):
    """fp16-safe sqrt.

    ``tl.sqrt`` only accepts fp32 and fp64.  This helper selects the
    compute dtype at compile time (fp64 if input is fp64, else fp32),
    upcasts if needed, computes sqrt, and casts back.  For fp32/fp64
    inputs the casts are no-ops with zero overhead.
    """
    compute_dtype = tl.float64 if x.dtype == tl.float64 else tl.float32
    return tl.sqrt(x.to(compute_dtype)).to(x.dtype)


@triton.jit
def to_compute_dtype(hp_value, ref_value):
    """Downcast a high-precision (hpfloat) value to match the base computation dtype.

    Mirrors Fortran CaMa-Flood's pattern of explicitly casting double-precision
    storage variables to single-precision before arithmetic::

        DFSTO = REAL(P2FLDSTO(ISEQ,1), KIND=JPRB)   ! JPRD → JPRB
        DSTO  = REAL((P2RIVSTO+P2FLDSTO), KIND=JPRB) ! JPRD → JPRB

    In the GPU code, ``hp_value`` is loaded from an hpfloat pointer (e.g. float64
    when mixed_precision is enabled) and ``ref_value`` is any variable already in
    base precision (e.g. float32).  The cast ensures all subsequent arithmetic
    stays in base precision, matching the Fortran semantics.

    When mixed_precision is disabled both dtypes are identical and the ``.to()``
    compiles to a no-op with zero overhead.
    """
    return hp_value.to(ref_value.dtype)
