"""Shared routing physical constants."""

from typing import Final


ROUTING_SLOPE_LIMIT: Final = 0.005
BACKFLOW_STORAGE_FRACTION: Final = 0.05
# Minimum outgoing volume [m3] used by the flow limiters.
OUTGOING_VOLUME_FLOOR: Final = 1e-10
RESERVOIR_RELEASE_EXPONENT: Final = 0.1
# Bed elevation [m] of a closed bifurcation level (no water surface reaches it).
DISABLED_BIFURCATION_ELEVATION: Final = 1.0e20
