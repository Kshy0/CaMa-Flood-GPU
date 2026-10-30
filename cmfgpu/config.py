"""Shared routing physical constants and model options."""

from typing import Final, Literal, Self

from hydroforge.contracts import OptionField, OptionsConfig, opt, option
from pydantic import Field, model_validator

# Gravitational acceleration [m s-2]; preserve the CaMa-Flood default.
GRAVITY: Final = 9.8

ROUTING_SLOPE_LIMIT: Final = 0.005
BACKFLOW_STORAGE_FRACTION: Final = 0.05
# Minimum outgoing volume [m3] used by the flow limiters.
OUTGOING_VOLUME_FLOOR: Final = 1e-10
# Bed elevation [m] of a closed bifurcation level (no water surface reaches it).
DISABLED_BIFURCATION_ELEVATION: Final = 1.0e20


class ReservoirOptions(OptionsConfig):
    """Uniform release law and empirical coefficients fixed at construction."""

    release_scheme: Literal["yamazaki_funato", "hanazaki_2022"] = OptionField(
        default="yamazaki_funato",
        codes={"yamazaki_funato": 0, "hanazaki_2022": 1},
        description="Reservoir release scheme (CaMa-Flood v4 default or LDAMH22)",
        affects_parameters=True,
    )
    flood_volume_fraction: float = option(
        default=0.95, gt=0.0, le=1.0, allow_inf_nan=False,
        description="Emergency-minus-conservation volume as a fraction of the flood pool",
        affects_parameters=True,
    )
    adjustment_volume_fraction: float = option(
        default=0.1, gt=0.0, lt=1.0, allow_inf_nan=False,
        description="Yamazaki & Funato adjustment volume above conservation, as a flood-pool fraction",
        affects_parameters=True,
        relevant_when=opt("release_scheme") == "yamazaki_funato",
    )
    dry_storage_fraction: float = option(
        default=0.7, ge=0.0, le=1.0, allow_inf_nan=False,
        description="Conservation storage fraction available in the dry season",
        affects_parameters=True,
        relevant_when=opt("release_scheme") == "yamazaki_funato",
    )
    dry_inflow_fraction: float = option(
        default=0.25, ge=0.0, allow_inf_nan=False,
        description="Annual inflow fraction available in the dry season",
        affects_parameters=True,
        relevant_when=opt("release_scheme") == "yamazaki_funato",
    )
    dry_season_days: float = option(
        default=180.0, gt=0.0, allow_inf_nan=False,
        description="Dry-season duration used for the supply rate [days]",
        affects_parameters=True,
        relevant_when=opt("release_scheme") == "yamazaki_funato",
    )
    normal_outflow_factor: float = option(
        default=1.5, gt=0.0, allow_inf_nan=False,
        description="Yamazaki & Funato multiplier of the storage-limited normal outflow",
        affects_parameters=True,
        relevant_when=opt("release_scheme") == "yamazaki_funato",
    )
    release_exponent: float = option(
        default=0.1, gt=0.0, allow_inf_nan=False,
        description="Yamazaki & Funato exponent in the flood-control storage regime",
        relevant_when=opt("release_scheme") == "yamazaki_funato",
    )
    normal_volume_fraction: float = option(
        default=0.5, gt=0.0, lt=1.0, allow_inf_nan=False,
        description="Hanazaki (2022) normal volume as a conservation-volume fraction",
        affects_parameters=True,
        relevant_when=opt("release_scheme") == "hanazaki_2022",
    )
    flood_storage_depth: float = option(
        default=0.2, gt=0.0, allow_inf_nan=False,
        description="Hanazaki (2022) flood-storage depth over upstream drainage area [m]",
        affects_parameters=True,
        relevant_when=opt("release_scheme") == "hanazaki_2022",
    )

    @model_validator(mode="after")
    def validate_adjustment_fraction(self) -> Self:
        if (self.release_scheme == "yamazaki_funato"
                and self.adjustment_volume_fraction >= self.flood_volume_fraction):
            raise ValueError("adjustment_volume_fraction must be below flood_volume_fraction")
        return self


class CaMaOptions(OptionsConfig):
    """Immutable physical and numerical settings shared by routing modules."""

    reservoir: ReservoirOptions = Field(default_factory=ReservoirOptions)

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
