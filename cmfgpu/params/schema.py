"""NetCDF schema derived from CaMa-Flood module field metadata."""

from __future__ import annotations

from hydroforge.contracts.fields import ModuleFieldSchema, parse_module_schema

from cmfgpu.modules.base import BaseModule
from cmfgpu.modules.bifurcation import BifurcationModule
from cmfgpu.modules.levee import LeveeModule
from cmfgpu.modules.reservoir import ReservoirModule


# Runtime tensor shapes use scalar property names. NetCDF uses stable axis names.
MODEL_DIMENSIONS: dict[str, str] = {
    "num_catchments": "catchment",
    "num_output_catchments": "saved_points",
    "num_flood_levels": "flood_level",
    "num_bifurcation_paths": "bifurcation_path",
    "num_bifurcation_levels": "bifurcation_level",
    "num_levees": "levee",
    "num_reservoirs": "reservoir",
}


PARAMETER_MODULES = (
    BaseModule,
    BifurcationModule,
    LeveeModule,
    ReservoirModule,
)


# These fields are required by their runtime module but may be estimated and
# appended after the initial map-construction pass.
DEFERRED_PARAMETER_FIELDS: frozenset[str] = frozenset(
    {
        "river_width",
        "river_height",
    }
)


# These are map-construction metadata, not tensors owned by a runtime module.
MAP_REQUIRED_FIELD_DIMS: dict[str, tuple[str, ...]] = {
    "nx": (),
    "ny": (),
    "num_basins": (),
    "catchment_x": ("catchment",),
    "catchment_y": ("catchment",),
    "catchment_mainstem_basin_id": ("catchment",),
    "basin_sizes": ("basin",),
    "upstream_area": ("catchment",),
    "river_mouth_id": ("catchment",),
}


MAP_OPTIONAL_FIELD_DIMS: dict[str, tuple[str, ...]] = {
    "bifurcation_catchment_x": ("bifurcation_path",),
    "bifurcation_downstream_x": ("bifurcation_path",),
    "bifurcation_catchment_y": ("bifurcation_path",),
    "bifurcation_downstream_y": ("bifurcation_path",),
    "levee_catchment_x": ("levee",),
    "levee_catchment_y": ("levee",),
    "gauge_catchment_id": ("gauge",),
    "gauge_station_id": ("gauge",),
    "gauge_reported_area_km2": ("gauge",),
    "gauge_allocated_area_km2": ("gauge",),
    "gauge_alloc_error": ("gauge",),
    "longitude": ("catchment",),
    "latitude": ("catchment",),
    "satellite_width": ("catchment",),
}


def _is_parameter_field(field: ModuleFieldSchema) -> bool:
    # Optional selections configure output views rather than parameter data.
    # A required selection such as BaseModule.output_catchment_id is input data.
    if field.selects is not None and not field.required:
        return False
    return field.output != "disabled" or field.required


MODEL_SCHEMA = parse_module_schema(PARAMETER_MODULES)
MODULE_FIELD_DIMS = MODEL_SCHEMA.resolve_dimensions(
    MODEL_DIMENSIONS,
    include=_is_parameter_field,
)


PARAMETER_FIELD_DIMS: dict[str, tuple[str, ...]] = {
    **MAP_REQUIRED_FIELD_DIMS,
    **MAP_OPTIONAL_FIELD_DIMS,
}
for module_name, fields in MODULE_FIELD_DIMS.items():
    for name, dims in fields.items():
        previous = PARAMETER_FIELD_DIMS.setdefault(name, dims)
        if previous != dims:
            raise RuntimeError(
                f"Parameter field {name!r} has conflicting dimensions: "
                f"{previous} and {dims} from module {module_name!r}"
            )


_base_required = tuple(
    field.name
    for field in MODEL_SCHEMA.fields(BaseModule.module_name)
    if field.required
    and field.name in MODULE_FIELD_DIMS[BaseModule.module_name]
    and field.name not in DEFERRED_PARAMETER_FIELDS
)
PARAMETER_REQUIRED_FIELDS: tuple[str, ...] = (
    *MAP_REQUIRED_FIELD_DIMS,
    *_base_required,
)


_required = set(PARAMETER_REQUIRED_FIELDS)
PARAMETER_OPTIONAL_FIELDS: tuple[str, ...] = tuple(
    name for name in PARAMETER_FIELD_DIMS if name not in _required
)


_module_axes = {
    BifurcationModule.module_name: "bifurcation_path",
    LeveeModule.module_name: "levee",
    ReservoirModule.module_name: "reservoir",
    "gauge": "gauge",
}
PARAMETER_MODULE_FIELDS: dict[str, frozenset[str]] = {
    module_name: frozenset(
        name for name, dims in PARAMETER_FIELD_DIMS.items() if dims and dims[0] == axis
    )
    for module_name, axis in _module_axes.items()
}


if len(PARAMETER_FIELD_DIMS) != len(
    set(PARAMETER_REQUIRED_FIELDS) | set(PARAMETER_OPTIONAL_FIELDS)
):
    raise RuntimeError("Parameter field classification contains duplicates")
