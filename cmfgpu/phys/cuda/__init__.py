# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""Runtime-compiled CUDA kernels for CaMa-Flood.

Each kernel runs a ``__device__`` body per item of its spec's size on the
canonical values packed into one struct ``args``; HydroForge generates the
``__global__`` entry. Ensembles of more than one member run the body's
``*_members`` variant over ``blockIdx.y``. Each ``.cu`` file keeps its
single-member and ensemble bodies together.
"""

from pathlib import Path

from hydroforge.kernels import CudaCall, CudaKernel, CudaSource
from cmfgpu import config as constants

PHYSICAL_CONSTANT_FLAGS = tuple(
    f"-DCMF_{name}={value!r}"
    for name, value in vars(constants).items() if name.isupper()
)

_DIR = Path(__file__).resolve().parent
# Routing keeps subnormal storages and fluxes, also under HydroForge fast math.
_SUBNORMAL_OPTIONS = ("--ftz=false", *PHYSICAL_CONSTANT_FLAGS)
_SOURCES = {
    name: CudaSource(path=_DIR / f"{name}.cu", options=options, include_root=_DIR)
    for name, options in (
        ("storage", _SUBNORMAL_OPTIONS),
        ("outflow", _SUBNORMAL_OPTIONS),
        ("adaptive_time", PHYSICAL_CONSTANT_FLAGS),
        ("bifurcation", _SUBNORMAL_OPTIONS),
        ("reservoir", PHYSICAL_CONSTANT_FLAGS),
        ("levee", _SUBNORMAL_OPTIONS),
    )
}
_REAL_STO = "{river_depth_ptr}, {river_storage_ptr}"
_FLOOD_STAGE = (
    _REAL_STO + ", {HAS_BIFURCATION}, {HAS_INFLOW}, {HAS_LEVEE}, "
    "{HAS_TOTAL_STORAGE_OUTPUT}"
)


def _kernel(source, body, templates, *, members=True, guard=True):
    """Run ``body<templates>(args, index)``; with ``members``, more than one
    ensemble member runs ``body_members<templates>(args, index, member)``.

    ``guard=False`` keeps out-of-range threads for bodies that check the index
    themselves, as block reductions must.
    """
    return CudaKernel(
        _SOURCES[source],
        steps=(
            CudaCall(
                device=f"{body}<{templates}>",
                batched=f"{body}_members<{templates}>" if members else None,
                batch_axis="ensemble_size" if members else None,
                pack="struct",
                pass_index=True,
                guard=guard,
            ),
        ),
    )


# The stage bodies check the index themselves for the LOG reduction.
FLOOD_STAGE = _kernel("storage", "flood_stage", _FLOOD_STAGE, guard=False)
FLOOD_STAGE_LOG = _kernel(
    "storage", "flood_stage", _FLOOD_STAGE + ", true", members=False, guard=False
)
OUTFLOW = _kernel(
    "outflow", "outflow",
    _REAL_STO + ", {HAS_BIFURCATION}, {HAS_LEVEE}, {HAS_RESERVOIR}, {HAS_SEA_LEVEL}",
)
INFLOW = _kernel(
    "outflow", "inflow",
    "{river_outflow_ptr}, {river_storage_ptr}, {HAS_BIFURCATION}, {HAS_RESERVOIR}",
)
ADAPTIVE_TIME = _kernel(
    "adaptive_time", "adaptive_time", "{river_depth_ptr}, {HAS_RESERVOIR}", guard=False
)
BIFURCATION_OUTFLOW = _kernel("bifurcation", "bif_outflow", _REAL_STO)
BIFURCATION_INFLOW = _kernel(
    "bifurcation", "bif_inflow",
    "{bifurcation_outflow_ptr}, {global_bifurcation_outflow_ptr}",
)
RESERVOIR_OUTFLOW = _kernel(
    "reservoir", "reservoir_outflow",
    "{river_outflow_ptr}, {river_storage_ptr}, {HAS_LEVEE}",
)
LEVEE_STAGE = _kernel("levee", "levee_stage", _REAL_STO, guard=False)
LEVEE_STAGE_LOG = _kernel(
    "levee", "levee_stage", _REAL_STO + ", true", members=False, guard=False
)
LEVEE_BIFURCATION_OUTFLOW = _kernel("levee", "levee_bif_outflow", _REAL_STO)
__all__ = []
