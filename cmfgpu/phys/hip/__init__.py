# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
HIP backend for CaMa-Flood-GPU physics kernels (AMD ROCm).

JIT-compiles .hip / .cpp / .cuh sources in the ``hip/`` subdirectory via
hydroforge's ``load_cu_module()`` on first use, then exposes each launcher
as a ``LazyCudaKernel`` compatible with the Triton calling convention.

Compile-time constants (NUM_FLOOD_LEVELS, NUM_BIF_LEVELS, HAS_BIFURCATION_CONST,
HAS_RESERVOIR_CONST, CMF_GRAVITY, CMF_MIN_KINEMATIC_SLOPE) must be set via
:func:`configure` before any kernel is called.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from hydroforge.runtime.cuda_kernel import (
    CudaKernel,
    LazyCudaKernel,
    check_build_manifest,
    load_cu_module,
    update_build_manifest,
)

# ── Precision configuration ───────────────────────────────────────────────

STORAGE_FLOAT: bool = os.environ.get("CMF_STORAGE_FLOAT", "") == "1"
STORAGE_DTYPE = torch.float32 if STORAGE_FLOAT else torch.float64


def configure_storage_precision(use_float32: bool) -> None:
    """Set HIP storage precision before kernels are compiled.

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


# ── Compile-time constants ────────────────────────────────────────────────

_compile_config: Dict[str, Any] = {}
_build_dir: Optional[str] = None


def configure(
    *,
    num_flood_levels: int,
    num_bif_levels: int,
    has_bifurcation: bool,
    has_reservoir: bool,
    gravity: float = 9.80665,
    min_kinematic_slope: float = 1e-5,
    build_directory: Optional[str] = None,
) -> None:
    """Set compile-time constants before HIP kernels are compiled.

    Must be called before the first kernel invocation.  Invalidates any
    previously compiled module so that the next call to ``_get_module()``
    recompiles with the new constants.
    """
    global _compile_config, _build_dir, _module
    _compile_config = {
        "NUM_FLOOD_LEVELS": num_flood_levels,
        "NUM_BIF_LEVELS": num_bif_levels,
        "HAS_BIFURCATION_CONST": int(has_bifurcation),
        "HAS_RESERVOIR_CONST": int(has_reservoir),
        "CMF_GRAVITY": f"{gravity}f",
        "CMF_MIN_KINEMATIC_SLOPE": f"{min_kinematic_slope}f",
    }
    if build_directory is not None:
        _build_dir = str(build_directory)
    _module = None  # invalidate cached module


# ── Module loading ────────────────────────────────────────────────────────

_HIP_DIR = Path(__file__).parent

_HIP_FILES = [
    _HIP_DIR / "outflow.hip",
    _HIP_DIR / "inflow.hip",
    _HIP_DIR / "storage.hip",
    _HIP_DIR / "adaptive_time_step.hip",
    _HIP_DIR / "bifurcation.hip",
    _HIP_DIR / "reservoir.hip",
    _HIP_DIR / "levee.hip",
    _HIP_DIR / "bindings.cpp",
]

_module = None


def _get_module():
    global _module
    if _module is not None:
        return _module
    if not _compile_config:
        raise RuntimeError(
            "HIP compile-time constants not configured. "
            "Call cmfgpu.phys.hip.configure(...) before using HIP kernels."
        )
    # Build module name encoding all compile-time constants
    cc = _compile_config
    name = (
        f"cmfgpu_hip_fl{cc['NUM_FLOOD_LEVELS']}"
        f"_bl{cc['NUM_BIF_LEVELS']}"
        f"_bif{cc['HAS_BIFURCATION_CONST']}"
        f"_res{cc['HAS_RESERVOIR_CONST']}"
    )
    if STORAGE_FLOAT:
        name += "_f32"
    # Validate precompiled cache before (potentially slow) compilation
    if _build_dir is not None:
        check_build_manifest(_build_dir, "physics", name)
    flags = ["-O3", "-ffast-math", f"-I{_HIP_DIR}"]
    for key, val in cc.items():
        flags.append(f"-D{key}={val}")
    if STORAGE_FLOAT:
        flags.append("-DCMF_STORAGE_FLOAT")
    _module = load_cu_module(
        name,
        _HIP_FILES,
        extra_cuda_cflags=flags,
        build_directory=_build_dir,
    )
    # Record in manifest for future validation
    if _build_dir is not None:
        update_build_manifest(_build_dir, "physics", {
            "module_name": name,
            "config": {k: v for k, v in cc.items()},
            "STORAGE_FLOAT": STORAGE_FLOAT,
        })
    return _module


def get_hip_kernels():
    """Return the compiled HIP kernel module."""
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
        "time_step_ptr",
        ("num_catchments", int),
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
        ("num_catchments", int),
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
        ("num_catchments", int),
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
        "time_step_ptr",
        ("num_bifurcation_paths", int),
    ])


def _make_bifurcation_inflow():
    return CudaKernel(_get_module(), "launch_bifurcation_inflow", args=[
        "bifurcation_catchment_idx_ptr", "bifurcation_downstream_idx_ptr",
        "limit_rate_ptr", "bifurcation_outflow_ptr",
        "global_bifurcation_outflow_ptr",
        ("num_bifurcation_paths", int),
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
        ("num_levees", int),
    ])


def _make_levee_bifurcation_outflow():
    return CudaKernel(_get_module(), "launch_levee_bifurcation_outflow", args=[
        "bifurcation_catchment_idx_ptr", "bifurcation_downstream_idx_ptr",
        "bifurcation_manning_ptr", "bifurcation_outflow_ptr",
        "bifurcation_width_ptr", "bifurcation_length_ptr",
        "bifurcation_elevation_ptr", "bifurcation_cross_section_depth_ptr",
        "water_surface_elevation_ptr", "protected_water_surface_elevation_ptr",
        "total_storage_ptr", "outgoing_storage_ptr",
        "time_step_ptr",
        ("num_bifurcation_paths", int),
    ])


compute_levee_stage_kernel = LazyCudaKernel(_make_levee_stage)
compute_levee_stage_log_kernel = None
compute_levee_bifurcation_outflow_kernel = LazyCudaKernel(_make_levee_bifurcation_outflow)
compute_levee_stage_batched_kernel = None
compute_levee_bifurcation_outflow_batched_kernel = None
