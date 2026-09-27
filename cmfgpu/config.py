"""Shared routing physical constants."""

from typing import Final


ROUTING_SLOPE_LIMIT: Final = 0.005
BACKFLOW_STORAGE_FRACTION: Final = 0.05
# Minimum outgoing volume [m3] used by the flow limiters.
OUTGOING_VOLUME_FLOOR: Final = 1e-10
RESERVOIR_RELEASE_EXPONENT: Final = 0.1
