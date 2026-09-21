# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Lazy compiled-CUDA implementation catalog for CaMa-Flood."""

from pathlib import Path

from hydroforge.kernels.backends.cuda import (
    CudaExtensionGroup,
    CudaExtensionSpec,
    CudaRoute,
)
from cmfgpu import config as constants
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

PHYSICAL_CONSTANT_FLAGS = tuple(
    f"-DCMF_{name}={value!r}"
    for name, value in vars(constants).items() if name.isupper()
)

_DIR = Path(__file__).resolve().parent
# Block-level reductions shared by the kernels that fold per-catchment values
# into a single global scalar.
_BLOCK_REDUCE = _DIR / "block_reduce.cuh"
_ROUTING_CFLAGS = ("-O3", "--use_fast_math", "--ftz=false", *PHYSICAL_CONSTANT_FLAGS)
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
    owner_module=__name__,
    specs={
        "storage": CudaExtensionSpec(
            source=_DIR / "storage.cu",
            cflags=_ROUTING_CFLAGS,
            inline_includes=(_BLOCK_REDUCE,),
        ),
        "outflow": CudaExtensionSpec(
            source=_DIR / "outflow.cu",
            cflags=_ROUTING_CFLAGS,
        ),
        "adaptive": CudaExtensionSpec(
            source=_DIR / "adaptive_time.cu",
            cflags=("-O3", "--use_fast_math", *PHYSICAL_CONSTANT_FLAGS),
            inline_includes=(_BLOCK_REDUCE,),
        ),
        "bifurcation": CudaExtensionSpec(
            source=_DIR / "bifurcation.cu",
            cflags=_ROUTING_CFLAGS,
        ),
        "reservoir": CudaExtensionSpec(
            source=_DIR / "reservoir.cu",
            cflags=("-O3", "--use_fast_math", *PHYSICAL_CONSTANT_FLAGS),
        ),
        "levee": CudaExtensionSpec(
            source=_DIR / "levee.cu",
            cflags=_ROUTING_CFLAGS,
            inline_includes=(_BLOCK_REDUCE,),
        ),
    },
    routes=(
        CudaRoute(
            extension="storage",
            launch="launch_flood_stage",
            spec=FLOOD_STAGE,
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
        ),
        CudaRoute(
            extension="outflow",
            launch="launch_inflow",
            spec=INFLOW,
        ),
        CudaRoute(
            extension="adaptive",
            launch="launch_adaptive_time",
            spec=ADAPTIVE_TIME,
        ),
        CudaRoute(
            extension="bifurcation",
            launch="launch_bif_outflow",
            spec=BIFURCATION_OUTFLOW,
        ),
        CudaRoute(
            extension="bifurcation",
            launch="launch_bif_inflow",
            spec=BIFURCATION_INFLOW,
        ),
        CudaRoute(
            extension="reservoir",
            launch="launch_reservoir_outflow",
            spec=RESERVOIR_OUTFLOW,
        ),
        CudaRoute(
            extension="levee",
            launch="launch_levee_stage",
            spec=LEVEE_STAGE,
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
