"""Lazy native Metal implementation catalog for CaMa-Flood.

Only shader/source strategy lives here.  Public ABI metadata is inherited from
the active BackendRegistry when a factory returned by :func:`route` is called.
"""

from pathlib import Path

from hydroforge.kernels import (
    make_spec_metal_dispatcher,
    registry_factory,
)

from cmfgpu import config as constants

PHYSICAL_CONSTANT_SOURCE = "".join(
    f"constant float CMF_{name} = {value!r}f;\n"
    for name, value in vars(constants).items() if name.isupper()
)

_DIR = Path(__file__).parent


def _template(filename: str, *, parallel_axes: tuple[str, ...] = ()):
    """Generate the complete Metal ABI from the active KernelSpec."""

    @registry_factory
    def factory():
        return make_spec_metal_dispatcher(
            source=PHYSICAL_CONSTANT_SOURCE + (_DIR / filename).read_text(),
            parallel_axes=parallel_axes,
        )

    return factory


outflow = _template(
    "outflow.metal",
    parallel_axes=("ensemble_size",),
)
inflow = _template(
    "outflow.metal",
    parallel_axes=("ensemble_size",),
)
flood_stage = _template(
    "storage.metal",
    parallel_axes=("ensemble_size",),
)
flood_stage_log = _template(
    "storage.metal",
)
adaptive_time = _template(
    "adaptive_time.metal",
    parallel_axes=("ensemble_size",),
)
bifurcation_outflow = _template(
    "bifurcation.metal",
    parallel_axes=("ensemble_size",),
)
bifurcation_inflow = _template(
    "bifurcation.metal",
    parallel_axes=("ensemble_size",),
)
reservoir_outflow = _template(
    "reservoir.metal",
    parallel_axes=("ensemble_size",),
)
levee_stage = _template(
    "levee.metal",
    parallel_axes=("ensemble_size",),
)
levee_stage_log = _template(
    "levee.metal",
)
levee_bifurcation_outflow = _template(
    "levee.metal",
    parallel_axes=("ensemble_size",),
)
