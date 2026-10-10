"""Shared routing physical constants and model options."""

from typing import Final

from hydroforge.contracts import OptionsConfig
from pydantic import Field

# Gravitational acceleration [m s-2]; preserve the CaMa-Flood default.
GRAVITY: Final = 9.8

ROUTING_SLOPE_LIMIT: Final = 0.005
BACKFLOW_STORAGE_FRACTION: Final = 0.05
# Minimum outgoing volume [m3] used by the flow limiters.
OUTGOING_VOLUME_FLOOR: Final = 1e-10
RESERVOIR_RELEASE_EXPONENT: Final = 0.1
# Bed elevation [m] of a closed bifurcation level (no water surface reaches it).
DISABLED_BIFURCATION_ELEVATION: Final = 1.0e20


class CaMaOptions(OptionsConfig):
    """Immutable physical and numerical settings shared by routing modules."""

    gravity: float = Field(
        default=GRAVITY,
        description="Gravitational acceleration in m s-2",
        gt=0.0,
        allow_inf_nan=False,
    )
    min_kinematic_slope: float = Field(
        default=1.0e-5,
        description="Minimum bed slope for kinematic wave (dimensionless)",
        gt=0.0,
        allow_inf_nan=False,
    )
    adaptive_time_factor: float = Field(
        default=0.7,
        description="Adaptive time-step CFL factor (dimensionless)",
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
