"""Metal kernels of CaMa-Flood.

Each kernel's body is selected from its ``.metal`` source by the spec name;
HydroForge generates the argument buffer and entry from the spec.
"""

from pathlib import Path

from hydroforge.kernels import MetalKernel

from cmfgpu import config as constants

PHYSICAL_CONSTANT_SOURCE = "".join(
    f"constant float CMF_{name} = {value!r}f;\n"
    for name, value in vars(constants).items() if name.isupper()
)

_DIR = Path(__file__).parent


_COMMON_HELPERS = """
#ifdef HF_HP_ENABLED
using cmf_storage = hf_hp;
#else
using cmf_storage = float;
#endif

static inline void cmf_decompose_index(
    uint i, long n, long ensemble_size, thread long& idx, thread long& member
) {
    if (ensemble_size == 1) {
        idx = (long)i;
        member = 0;
    } else {
        uint u_n = (uint)n;
        uint u_m = i / u_n;
        member = (long)u_m;
        idx = (long)(i - u_m * u_n);
    }
}

// Physics float operations.  HydroForge defines HYDROFORGE_FAST_MATH for
// physics sources under HYDROFORGE_FAST_MATH=1, also for float32x2 libraries,
// which keep safe math for their pair arithmetic.  Measured on Apple M5
// against FP64: the default cube root is <= 1 ULP (d^-7/3 <= 3.4 ULP), the
// fast one <= 16 ULP (d^-7/3 <= 9.2 ULP); fast::sqrt is not correctly rounded.
#if HYDROFORGE_FAST_MATH
static inline float cmf_div(float a, float b) { return fast::divide(a, b); }
static inline float cmf_sqrt(float x) { return fast::sqrt(x); }
static inline float cmf_cbrt(float x) {
    if (x == 0.0f) return x;
    return copysign(fast::powr(fabs(x), 1.0f / 3.0f), x);
}
#else
static inline float cmf_div(float a, float b) { return a / b; }
static inline float cmf_sqrt(float x) { return precise::sqrt(x); }
// exp2(log2(x) / 3) and one Newton correction; ~10x faster than safe pow().
static inline float cmf_cbrt(float x) {
    if (x == 0.0f || !isfinite(x)) return x;
    float a = fabs(x);
    float y = precise::exp2((1.0f / 3.0f) * precise::log2(a));
    float r = a / (y * y);
    return copysign(y + (r - y) * (1.0f / 3.0f), x);
}
#endif

// d^(-7/3) and d^(-4/3) in the CUDA/Triton form 1 / (d * d * cbrt(d)).
static inline float cmf_inv_d73(float d) {
    return cmf_div(1.0f, d * d * cmf_cbrt(d));
}

static inline float cmf_inv_d43(float d) {
    return cmf_div(1.0f, d * cmf_cbrt(d));
}
"""


def _kernel(filename: str, *, members: bool = True) -> MetalKernel:
    """``members`` runs one thread per catchment and ensemble member."""

    return MetalKernel(
        _DIR / filename,
        prelude=PHYSICAL_CONSTANT_SOURCE + _COMMON_HELPERS,
        batch_axis="ensemble_size" if members else None,
    )


OUTFLOW = _kernel("outflow.metal")
INFLOW = _kernel("outflow.metal")
FLOOD_STAGE = _kernel("storage.metal")
FLOOD_STAGE_LOG = _kernel("storage.metal", members=False)
INITIAL_FLOOD_STAGE = _kernel("storage.metal")
ADAPTIVE_TIME = _kernel("adaptive_time.metal")
BIFURCATION_OUTFLOW = _kernel("bifurcation.metal")
BIFURCATION_INFLOW = _kernel("bifurcation.metal")
RESERVOIR_OUTFLOW = _kernel("reservoir.metal")
DAM_FILL = _kernel("reservoir.metal")
LEVEE_STAGE = _kernel("levee.metal")
LEVEE_STAGE_LOG = _kernel("levee.metal", members=False)
LEVEE_BIFURCATION_OUTFLOW = _kernel("levee.metal")
DOWNSCALE_FLOOD_DEPTH = _kernel("downscale.metal")
DOWNSCALE_TRIBUTARY_DEPTH = _kernel("downscale.metal")
DOWNSCALE_BACKWATER_DEPTH = _kernel("downscale.metal")
DOWNSCALE_PERMANENT_WATER = _kernel("downscale.metal")
DOWNSCALE_FLOOD_DURATION = _kernel("downscale.metal")
