# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
CUDA backend for CaMa-Flood-GPU physics kernels.

JIT-compiles .cu / .cpp / .cuh sources in the ``cuda/`` subdirectory via
hydroforge's ``load_cu_module()`` on first use, then exposes each launcher
as a ``LazyCudaKernel`` compatible with the Triton calling convention.

Compile with ``CMF_STORAGE_FLOAT=1`` (env var) to use float32 storage
instead of the default float64.
"""

import os
from pathlib import Path

import torch
from hydroforge.runtime.cuda_kernel import (CudaKernel, LazyCudaKernel,
                                            load_cu_module)

# ── Precision configuration ───────────────────────────────────────────────

STORAGE_FLOAT: bool = os.environ.get("CMF_STORAGE_FLOAT", "") == "1"
STORAGE_DTYPE = torch.float32 if STORAGE_FLOAT else torch.float64


def configure_storage_precision(use_float32: bool) -> None:
    """Set CUDA storage precision before kernels are compiled.

    Called automatically by the model when ``mixed_precision=False``
    (all tensors float32).  Must be called before the first kernel
    invocation (i.e. before ``_get_module()``).
    """
    global STORAGE_FLOAT, STORAGE_DTYPE
    if use_float32:
        os.environ["CMF_STORAGE_FLOAT"] = "1"
        STORAGE_FLOAT = True
        STORAGE_DTYPE = torch.float32
    else:
        os.environ.pop("CMF_STORAGE_FLOAT", None)
        STORAGE_FLOAT = False
        STORAGE_DTYPE = torch.float64


# ── Module loading ────────────────────────────────────────────────────────

_CU_DIR = Path(__file__).parent

_CU_FILES = [
    _CU_DIR / "outflow.cu",
    _CU_DIR / "inflow.cu",
    _CU_DIR / "storage.cu",
    _CU_DIR / "adaptive_time_step.cu",
    _CU_DIR / "bifurcation.cu",
    _CU_DIR / "reservoir.cu",
    _CU_DIR / "levee.cu",
    _CU_DIR / "bindings.cpp",
]

_module = None


def _get_module():
    global _module
    if _module is not None:
        return _module
    flags = ["-O3", "--use_fast_math", f"-I{_CU_DIR}"]
    if STORAGE_FLOAT:
        flags.append("-DCMF_STORAGE_FLOAT")
    _module = load_cu_module(
        "cmfgpu_cuda_kernels" if not STORAGE_FLOAT else "cmfgpu_cuda_kernels_f32",
        _CU_FILES,
        extra_cuda_cflags=flags,
    )
    return _module


def get_cuda_kernels():
    """Return the compiled CUDA kernel module."""
    return _get_module()


# ── Outflow / Inflow ─────────────────────────────────────────────────────

def _make_outflow():
    return CudaKernel(_get_module(), "launch_outflow", args=[
        "downstream_idx_ptr",
        "river_inflow_ptr", "river_outflow_ptr",
        "river_manning_ptr", "river_depth_ptr",
        "river_width_ptr", "river_length_ptr",
        "river_height_ptr", "river_storage_ptr",
        "flood_inflow_ptr", "flood_outflow_ptr",
        "flood_manning_ptr", "flood_depth_ptr",
        "protected_depth_ptr", "catchment_elevation_ptr",
        "downstream_distance_ptr", "flood_storage_ptr",
        "protected_storage_ptr",
        "river_cross_section_depth_ptr", "flood_cross_section_depth_ptr",
        "flood_cross_section_area_ptr",
        "global_bifurcation_outflow_ptr", "total_storage_ptr",
        "outgoing_storage_ptr", "water_surface_elevation_ptr",
        "protected_water_surface_elevation_ptr",
        "is_dam_upstream_ptr",
        ("gravity", float), "time_step_ptr",
        ("MIN_KINEMATIC_SLOPE", float, 1e-5),
        ("num_catchments", int),
        ("HAS_BIFURCATION", bool, True),
        ("HAS_RESERVOIR", bool, False),
    ], nullable={"is_dam_upstream_ptr": "torch.bool"})


def _make_inflow():
    sto_dtype = "float32" if STORAGE_FLOAT else "float64"
    return CudaKernel(_get_module(), "launch_inflow", args=[
        "downstream_idx_ptr",
        "river_outflow_ptr", "flood_outflow_ptr",
        "river_storage_ptr", "flood_storage_ptr",
        "outgoing_storage_ptr",
        "river_inflow_ptr", "flood_inflow_ptr",
        "limit_rate_ptr",
        "reservoir_total_inflow_ptr", "is_reservoir_ptr",
        ("num_catchments", int),
        ("HAS_RESERVOIR", bool, False),
    ], nullable={
        "reservoir_total_inflow_ptr": sto_dtype,
        "is_reservoir_ptr": "torch.bool",
    })


compute_outflow_kernel = LazyCudaKernel(_make_outflow)
compute_inflow_kernel = LazyCudaKernel(_make_inflow)
compute_outflow_batched_kernel = None
compute_inflow_batched_kernel = None

# ── Flood stage ───────────────────────────────────────────────────────────

def _make_flood_stage():
    return CudaKernel(_get_module(), "launch_flood_stage", args=[
        "river_inflow_ptr", "flood_inflow_ptr",
        "river_outflow_ptr", "flood_outflow_ptr",
        "global_bifurcation_outflow_ptr",
        "runoff_ptr", "time_step_ptr",
        "outgoing_storage_ptr",
        "river_storage_ptr", "flood_storage_ptr", "protected_storage_ptr",
        "river_depth_ptr", "flood_depth_ptr",
        "protected_depth_ptr", "flood_fraction_ptr",
        "river_height_ptr", "flood_depth_table_ptr",
        "catchment_area_ptr", "river_width_ptr", "river_length_ptr",
        ("num_catchments", int), ("num_flood_levels", int),
        ("HAS_BIFURCATION", bool, True),
    ])


compute_flood_stage_kernel = LazyCudaKernel(_make_flood_stage)
compute_flood_stage_log_kernel = None
compute_flood_stage_batched_kernel = None

# ── Adaptive time step ────────────────────────────────────────────────────

def _make_adaptive_time_step():
    return CudaKernel(_get_module(), "launch_adaptive_time_step", args=[
        "river_depth_ptr", "downstream_distance_ptr",
        "is_dam_related_ptr", "max_sub_steps_ptr",
        ("time_step", float), ("adaptive_time_factor", float),
        ("gravity", float), ("num_catchments", int),
        ("HAS_RESERVOIR", bool, False),
    ], nullable={"is_dam_related_ptr": "torch.bool"})


compute_adaptive_time_step_kernel = LazyCudaKernel(_make_adaptive_time_step)
compute_adaptive_time_step_batched_kernel = None

# ── Bifurcation ───────────────────────────────────────────────────────────

def _make_bifurcation_outflow():
    return CudaKernel(_get_module(), "launch_bifurcation_outflow", args=[
        "bifurcation_catchment_idx_ptr", "bifurcation_downstream_idx_ptr",
        "bifurcation_manning_ptr", "bifurcation_outflow_ptr",
        "bifurcation_width_ptr", "bifurcation_length_ptr",
        "bifurcation_elevation_ptr", "bifurcation_cross_section_depth_ptr",
        "water_surface_elevation_ptr",
        "total_storage_ptr", "outgoing_storage_ptr",
        ("gravity", float), "time_step_ptr",
        ("num_bifurcation_paths", int), ("num_bifurcation_levels", int),
    ])


def _make_bifurcation_inflow():
    return CudaKernel(_get_module(), "launch_bifurcation_inflow", args=[
        "bifurcation_catchment_idx_ptr", "bifurcation_downstream_idx_ptr",
        "limit_rate_ptr", "bifurcation_outflow_ptr",
        "global_bifurcation_outflow_ptr",
        ("num_bifurcation_paths", int), ("num_bifurcation_levels", int),
    ])


compute_bifurcation_outflow_kernel = LazyCudaKernel(_make_bifurcation_outflow)
compute_bifurcation_inflow_kernel = LazyCudaKernel(_make_bifurcation_inflow)
compute_bifurcation_outflow_batched_kernel = None
compute_bifurcation_inflow_batched_kernel = None

# ── Reservoir ─────────────────────────────────────────────────────────────

def _make_reservoir_outflow():
    return CudaKernel(_get_module(), "launch_reservoir_outflow", args=[
        "reservoir_catchment_idx_ptr", "downstream_idx_ptr",
        "reservoir_total_inflow_ptr",
        "river_outflow_ptr", "flood_outflow_ptr",
        "river_storage_ptr", "flood_storage_ptr",
        "conservation_volume_ptr", "emergency_volume_ptr",
        "adjustment_volume_ptr", "normal_outflow_ptr",
        "adjustment_outflow_ptr", "flood_control_outflow_ptr",
        "runoff_ptr",
        "total_storage_ptr", "outgoing_storage_ptr",
        "time_step_ptr", ("num_reservoirs", int),
    ])


compute_reservoir_outflow_kernel = LazyCudaKernel(_make_reservoir_outflow)

# ── Levee ──────────────────────────────────────────────────────────────────

def _make_levee_stage():
    return CudaKernel(_get_module(), "launch_levee_stage", args=[
        "levee_catchment_idx_ptr",
        "river_storage_ptr", "flood_storage_ptr", "protected_storage_ptr",
        "river_depth_ptr", "flood_depth_ptr", "protected_depth_ptr",
        "river_height_ptr", "flood_depth_table_ptr",
        "catchment_area_ptr", "river_width_ptr", "river_length_ptr",
        "levee_base_height_ptr", "levee_crown_height_ptr",
        "levee_fraction_ptr", "flood_fraction_ptr",
        ("num_levees", int), ("num_flood_levels", int),
    ])


def _make_levee_bifurcation_outflow():
    return CudaKernel(_get_module(), "launch_levee_bifurcation_outflow", args=[
        "bifurcation_catchment_idx_ptr", "bifurcation_downstream_idx_ptr",
        "bifurcation_manning_ptr", "bifurcation_outflow_ptr",
        "bifurcation_width_ptr", "bifurcation_length_ptr",
        "bifurcation_elevation_ptr", "bifurcation_cross_section_depth_ptr",
        "water_surface_elevation_ptr", "protected_water_surface_elevation_ptr",
        "total_storage_ptr", "outgoing_storage_ptr",
        ("gravity", float), "time_step_ptr",
        ("num_bifurcation_paths", int), ("num_bifurcation_levels", int),
    ])


compute_levee_stage_kernel = LazyCudaKernel(_make_levee_stage)
compute_levee_stage_log_kernel = None
compute_levee_bifurcation_outflow_kernel = LazyCudaKernel(_make_levee_bifurcation_outflow)
compute_levee_stage_batched_kernel = None
compute_levee_bifurcation_outflow_batched_kernel = None
