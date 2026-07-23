"""Lazy native Metal implementation catalog for CaMa-Flood.

Only shader/source strategy lives here.  Public ABI metadata is inherited from
the active BackendRegistry when a factory returned by :func:`route` is called.
"""

from pathlib import Path

from hydroforge.kernels.registry import (
    make_spec_metal_dispatcher, registry_factory,
)

_DIR = Path(__file__).parent


def _template(filename: str, *, parallel_axes: tuple[str, ...] = ()):
    """Generate the complete Metal ABI from the active KernelSpec."""

    @registry_factory
    def factory():
        return make_spec_metal_dispatcher(
            source=(_DIR / filename).read_text(),
            parallel_axes=parallel_axes,
        )

    return factory


outflow = _template(
    "outflow.metal", parallel_axes=("num_trials",),
)
inflow = _template(
    "outflow.metal", parallel_axes=("num_trials",),
)
flood_stage = _template(
    "storage.metal",
    parallel_axes=("num_trials",),
)
flood_stage_log = _template(
    "storage.metal",
)
adaptive_time = _template(
    "adaptive_time.metal", parallel_axes=("num_trials",),
)
bifurcation_outflow = _template(
    "bifurcation.metal",
    parallel_axes=("num_trials",),
)
bifurcation_inflow = _template(
    "bifurcation.metal", parallel_axes=("num_trials",),
)
reservoir_outflow = _template(
    "reservoir.metal", parallel_axes=("num_trials",),
)
levee_stage = _template(
    "levee.metal",
    parallel_axes=("num_trials",),
)
levee_stage_log = _template(
    "levee.metal",
)
levee_bifurcation_outflow = _template(
    "levee.metal",
    parallel_axes=("num_trials",),
)


__all__ = []
