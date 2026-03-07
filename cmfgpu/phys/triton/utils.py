# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Utility helpers for Triton physics kernels."""

import triton


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
