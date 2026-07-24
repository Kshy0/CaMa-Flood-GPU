# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Lazy compiled-CUDA implementation catalog for CaMa-Flood."""

from pathlib import Path

from hydroforge.kernels.backends.cuda.dispatcher import (
    CudaExtensionGroup,
    CudaNativeProjection,
)
from hydroforge.kernels.backends.cuda.spec import CudaExtensionSpec

_DIR = Path(__file__).resolve().parent
_MODULE_EXTENSIONS = {
    "base": {"storage", "outflow"},
    "inflow": {"outflow"},
    "adaptive_time": {"adaptive"},
    "bifurcation": {"bifurcation"},
    "reservoir": {"reservoir"},
    "levee": {"levee"},
    "log": {"storage"},
    "sea_level": set(),
}
_CUDA = CudaExtensionGroup(
    __name__,
    {
        "storage": CudaExtensionSpec(
            _DIR / "storage.cu",
        ),
        "outflow": CudaExtensionSpec(
            _DIR / "outflow.cu",
        ),
        "adaptive": CudaExtensionSpec(
            _DIR / "adaptive_time.cu",
        ),
        "bifurcation": CudaExtensionSpec(
            _DIR / "bifurcation.cu",
        ),
        "reservoir": CudaExtensionSpec(
            _DIR / "reservoir.cu",
        ),
        "levee": CudaExtensionSpec(
            _DIR / "levee.cu",
        ),
    },
    binary_prefix="cmfgpu_cuda",
    module_extensions=_MODULE_EXTENSIONS,
)


def _shared(
    disabled: tuple[str, ...] = (),
) -> CudaNativeProjection:
    """Declare only the non-inferable single-trial preconditions."""
    return CudaNativeProjection(
        fixed={"num_trials": 1, **dict.fromkeys(disabled, False)},
    )


flood_stage = _CUDA.route(
    "storage",
    "launch_flood_stage",
    projection=_shared(
        disabled=(
            "batched_catchment_area",
            "batched_flood_depth_table",
            "batched_river_height",
            "batched_river_length",
            "batched_river_width",
            "batched_runoff",
        ),
    ),
)
flood_stage_log = _CUDA.route(
    "storage",
    "launch_flood_stage_log",
)
outflow = _CUDA.route(
    "outflow",
    "launch_outflow",
    projection=_shared(
        disabled=(
            "batched_catchment_elevation",
            "batched_flood_manning",
            "batched_river_height",
            "batched_river_length",
            "batched_river_manning",
            "batched_river_width",
        ),
    ),
)
inflow = _CUDA.route(
    "outflow",
    "launch_inflow",
    projection=_shared(),
)
adaptive_time = _CUDA.route(
    "adaptive",
    "launch_adaptive_time",
    projection=_shared(disabled=("batched_downstream_distance",)),
)
bifurcation_outflow = _CUDA.route(
    "bifurcation",
    "launch_bif_outflow",
    projection=_shared(
        disabled=(
            "batched_bifurcation_elevation",
            "batched_bifurcation_length",
            "batched_bifurcation_manning",
            "batched_bifurcation_width",
        ),
    ),
)
bifurcation_inflow = _CUDA.route(
    "bifurcation",
    "launch_bif_inflow",
    projection=_shared(),
)
reservoir_outflow = _CUDA.route(
    "reservoir",
    "launch_reservoir_outflow",
    projection=_shared(),
)
levee_stage = _CUDA.route(
    "levee",
    "launch_levee_stage",
    projection=_shared(
        disabled=(
            "batched_catchment_area",
            "batched_flood_depth_table",
            "batched_levee_base_height",
            "batched_levee_crown_height",
            "batched_levee_fraction",
            "batched_river_height",
            "batched_river_length",
            "batched_river_width",
        ),
    ),
)
levee_stage_log = _CUDA.route(
    "levee",
    "launch_levee_stage_log",
)
levee_bifurcation_outflow = _CUDA.route(
    "levee",
    "launch_levee_bif_outflow",
    projection=_shared(
        disabled=(
            "batched_bifurcation_elevation",
            "batched_bifurcation_length",
            "batched_bifurcation_manning",
            "batched_bifurcation_width",
        ),
    ),
)
__all__ = []
