# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Lazy compiled-CUDA implementation catalog for CaMa-Flood."""

from pathlib import Path

from hydroforge.kernels.backends.cuda import (
    CudaExtensionGroup,
    CudaExtensionSpec,
    CudaNativeProjection,
    CudaRoute,
)
from cmfgpu.phys.specs import (
    ADAPTIVE_TIME,
    BIFURCATION_INFLOW,
    BIFURCATION_OUTFLOW,
    FLOOD_STAGE,
    FLOOD_STAGE_LOG,
    INFLOW,
    LEVEE_BIFURCATION_OUTFLOW,
    LEVEE_STAGE,
    LEVEE_STAGE_LOG,
    OUTFLOW,
    RESERVOIR_OUTFLOW,
)

_DIR = Path(__file__).resolve().parent
# Block-level reductions shared by the kernels that fold per-catchment values
# into a single global scalar.
_BLOCK_REDUCE = _DIR / "block_reduce.cuh"
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


def _shared(
    disabled: tuple[str, ...] = (),
) -> CudaNativeProjection:
    """Declare only the non-inferable single-trial preconditions."""
    return CudaNativeProjection(
        fixed={"num_trials": 1, **dict.fromkeys(disabled, False)},
    )


_CUDA = CudaExtensionGroup(
    owner_module=__name__,
    specs={
        "storage": CudaExtensionSpec(
            source=_DIR / "storage.cu",
            inline_includes=(_BLOCK_REDUCE,),
        ),
        "outflow": CudaExtensionSpec(
            source=_DIR / "outflow.cu",
        ),
        "adaptive": CudaExtensionSpec(
            source=_DIR / "adaptive_time.cu",
            inline_includes=(_BLOCK_REDUCE,),
        ),
        "bifurcation": CudaExtensionSpec(
            source=_DIR / "bifurcation.cu",
        ),
        "reservoir": CudaExtensionSpec(
            source=_DIR / "reservoir.cu",
        ),
        "levee": CudaExtensionSpec(
            source=_DIR / "levee.cu",
            inline_includes=(_BLOCK_REDUCE,),
        ),
    },
    routes=(
        CudaRoute(
            extension="storage",
            launch="launch_flood_stage",
            spec=FLOOD_STAGE,
            projection=_shared(
                disabled=(
                    "batched_catchment_area",
                    "batched_flood_depth_table",
                    "batched_river_height",
                    "batched_river_length",
                    "batched_river_width",
                    "batched_runoff",
                    "batched_inflow",
                ),
            ),
        ),
        CudaRoute(
            extension="storage",
            launch="launch_flood_stage_log",
            spec=FLOOD_STAGE_LOG,
        ),
        CudaRoute(
            extension="outflow",
            launch="launch_outflow",
            spec=OUTFLOW,
            projection=_shared(
                disabled=(
                    "batched_catchment_elevation",
                    "batched_downstream_distance",
                    "batched_flood_manning",
                    "batched_river_height",
                    "batched_river_length",
                    "batched_river_manning",
                    "batched_river_width",
                    "batched_sea_surface_elevation",
                ),
            ),
        ),
        CudaRoute(
            extension="outflow",
            launch="launch_inflow",
            spec=INFLOW,
            projection=_shared(),
        ),
        CudaRoute(
            extension="adaptive",
            launch="launch_adaptive_time",
            spec=ADAPTIVE_TIME,
            projection=_shared(disabled=("batched_downstream_distance",)),
        ),
        CudaRoute(
            extension="bifurcation",
            launch="launch_bif_outflow",
            spec=BIFURCATION_OUTFLOW,
            projection=_shared(
                disabled=(
                    "batched_bifurcation_elevation",
                    "batched_bifurcation_length",
                    "batched_bifurcation_manning",
                    "batched_bifurcation_width",
                ),
            ),
        ),
        CudaRoute(
            extension="bifurcation",
            launch="launch_bif_inflow",
            spec=BIFURCATION_INFLOW,
            projection=_shared(),
        ),
        CudaRoute(
            extension="reservoir",
            launch="launch_reservoir_outflow",
            spec=RESERVOIR_OUTFLOW,
            projection=_shared(disabled=("batched_runoff",)),
        ),
        CudaRoute(
            extension="levee",
            launch="launch_levee_stage",
            spec=LEVEE_STAGE,
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
        ),
        CudaRoute(
            extension="levee",
            launch="launch_levee_stage_log",
            spec=LEVEE_STAGE_LOG,
        ),
        CudaRoute(
            extension="levee",
            launch="launch_levee_bif_outflow",
            spec=LEVEE_BIFURCATION_OUTFLOW,
            projection=_shared(
                disabled=(
                    "batched_bifurcation_elevation",
                    "batched_bifurcation_length",
                    "batched_bifurcation_manning",
                    "batched_bifurcation_width",
                ),
            ),
        ),
    ),
    binary_prefix="cmfgpu_cuda",
    module_extensions=_MODULE_EXTENSIONS,
)

flood_stage = _CUDA.factory("storage", "launch_flood_stage")
flood_stage_log = _CUDA.factory("storage", "launch_flood_stage_log")
outflow = _CUDA.factory("outflow", "launch_outflow")
inflow = _CUDA.factory("outflow", "launch_inflow")
adaptive_time = _CUDA.factory("adaptive", "launch_adaptive_time")
bifurcation_outflow = _CUDA.factory("bifurcation", "launch_bif_outflow")
bifurcation_inflow = _CUDA.factory("bifurcation", "launch_bif_inflow")
reservoir_outflow = _CUDA.factory("reservoir", "launch_reservoir_outflow")
levee_stage = _CUDA.factory("levee", "launch_levee_stage")
levee_stage_log = _CUDA.factory("levee", "launch_levee_stage_log")
levee_bifurcation_outflow = _CUDA.factory(
    "levee", "launch_levee_bif_outflow",
)
__all__ = []
