# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Runtime-compiled CUDA routes for CaMa-Flood.

Each route runs a ``__device__`` body per item of its spec's size key on the
canonical values packed into one struct ``args``; HydroForge generates the
``__global__`` entry. Ensembles of more than one member run the body's
``*_members`` variant over ``blockIdx.y``. Each ``.cu`` file keeps its
single-member and ensemble bodies together.
"""

from pathlib import Path

from hydroforge.kernels.backends.cuda import (
    CudaExtensionGroup,
    CudaExtensionSpec,
    CudaKernel,
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
# Routing keeps subnormal storages and fluxes, also under HydroForge fast math.
_SUBNORMAL_OPTIONS = ("--ftz=false", *PHYSICAL_CONSTANT_FLAGS)
_REAL_STO = "{river_depth_ptr}, {river_storage_ptr}"
_FLOOD_STAGE = (
    _REAL_STO + ", {HAS_BIFURCATION}, {HAS_INFLOW}, {HAS_LEVEE}, "
    "{HAS_TOTAL_STORAGE_OUTPUT}"
)


def _route(extension, spec, body, templates, *, guard=True):
    """Run ``body<templates>(args, index)``; with an ensemble axis, more than
    one member runs ``body_members<templates>(args, index, member)``.

    ``guard=False`` keeps out-of-range threads for bodies that check the index
    themselves, as block reductions must.
    """
    members = "ensemble_size" in spec.parameters
    return CudaRoute(
        extension=extension,
        spec=spec,
        steps=(
            CudaKernel(
                device=f"{body}<{templates}>",
                batched=f"{body}_members<{templates}>" if members else None,
                batch_axis="ensemble_size" if members else None,
                pack="struct",
                pass_index=True,
                guard=guard,
            ),
        ),
    )


_CUDA = CudaExtensionGroup(
    specs={
        extension: CudaExtensionSpec(
            source=_DIR / source, options=options, include_root=_DIR
        )
        for extension, source, options in (
            ("storage", "storage.cu", _SUBNORMAL_OPTIONS),
            ("outflow", "outflow.cu", _SUBNORMAL_OPTIONS),
            ("adaptive", "adaptive_time.cu", PHYSICAL_CONSTANT_FLAGS),
            ("bifurcation", "bifurcation.cu", _SUBNORMAL_OPTIONS),
            ("reservoir", "reservoir.cu", PHYSICAL_CONSTANT_FLAGS),
            ("levee", "levee.cu", _SUBNORMAL_OPTIONS),
        )
    },
    routes=(
        # The stage bodies check the index themselves for the LOG reduction.
        _route("storage", FLOOD_STAGE, "flood_stage", _FLOOD_STAGE, guard=False),
        _route(
            "storage", FLOOD_STAGE_LOG, "flood_stage", _FLOOD_STAGE + ", true",
            guard=False,
        ),
        _route(
            "outflow", OUTFLOW, "outflow",
            _REAL_STO + ", {HAS_BIFURCATION}, {HAS_LEVEE}, {HAS_RESERVOIR}, "
            "{HAS_SEA_LEVEL}",
        ),
        _route(
            "outflow", INFLOW, "inflow",
            "{river_outflow_ptr}, {river_storage_ptr}, {HAS_BIFURCATION}, "
            "{HAS_RESERVOIR}",
        ),
        _route(
            "adaptive", ADAPTIVE_TIME, "adaptive_time",
            "{river_depth_ptr}, {HAS_RESERVOIR}", guard=False,
        ),
        _route("bifurcation", BIFURCATION_OUTFLOW, "bif_outflow", _REAL_STO),
        _route(
            "bifurcation", BIFURCATION_INFLOW, "bif_inflow",
            "{bifurcation_outflow_ptr}, {global_bifurcation_outflow_ptr}",
        ),
        _route(
            "reservoir", RESERVOIR_OUTFLOW, "reservoir_outflow",
            "{river_outflow_ptr}, {river_storage_ptr}, {HAS_LEVEE}",
        ),
        _route("levee", LEVEE_STAGE, "levee_stage", _REAL_STO, guard=False),
        _route(
            "levee", LEVEE_STAGE_LOG, "levee_stage", _REAL_STO + ", true", guard=False,
        ),
        _route(
            "levee", LEVEE_BIFURCATION_OUTFLOW, "levee_bif_outflow", _REAL_STO,
        ),
    ),
)

flood_stage = _CUDA.factory("storage", FLOOD_STAGE.name)
flood_stage_log = _CUDA.factory("storage", FLOOD_STAGE_LOG.name)
outflow = _CUDA.factory("outflow", OUTFLOW.name)
inflow = _CUDA.factory("outflow", INFLOW.name)
adaptive_time = _CUDA.factory("adaptive", ADAPTIVE_TIME.name)
bifurcation_outflow = _CUDA.factory("bifurcation", BIFURCATION_OUTFLOW.name)
bifurcation_inflow = _CUDA.factory("bifurcation", BIFURCATION_INFLOW.name)
reservoir_outflow = _CUDA.factory("reservoir", RESERVOIR_OUTFLOW.name)
levee_stage = _CUDA.factory("levee", LEVEE_STAGE.name)
levee_stage_log = _CUDA.factory("levee", LEVEE_STAGE_LOG.name)
levee_bifurcation_outflow = _CUDA.factory("levee", LEVEE_BIFURCATION_OUTFLOW.name)
__all__ = []
