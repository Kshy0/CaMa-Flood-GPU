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


def _kernel(filename: str, *, members: bool = True) -> MetalKernel:
    """``members`` runs one thread per catchment and ensemble member."""

    return MetalKernel(
        _DIR / filename,
        prelude=PHYSICAL_CONSTANT_SOURCE + "\n#ifdef HF_HP_ENABLED\nusing cmf_storage = hf_hp;\n#else\nusing cmf_storage = float;\n#endif\n",
        batch_axis="ensemble_size" if members else None,
    )


OUTFLOW = _kernel("outflow.metal")
INFLOW = _kernel("outflow.metal")
FLOOD_STAGE = _kernel("storage.metal")
FLOOD_STAGE_LOG = _kernel("storage.metal", members=False)
ADAPTIVE_TIME = _kernel("adaptive_time.metal")
BIFURCATION_OUTFLOW = _kernel("bifurcation.metal")
BIFURCATION_INFLOW = _kernel("bifurcation.metal")
RESERVOIR_OUTFLOW = _kernel("reservoir.metal")
LEVEE_STAGE = _kernel("levee.metal")
LEVEE_STAGE_LOG = _kernel("levee.metal", members=False)
LEVEE_BIFURCATION_OUTFLOW = _kernel("levee.metal")
