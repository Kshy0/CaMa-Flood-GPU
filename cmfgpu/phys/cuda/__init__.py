# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""CUDA backend for cmfgpu physics kernels.

JIT-compiles the hand-written CUDA kernels and exposes dispatch functions
matching the unified hydroforge kwargs convention.  Storage uses the per-lane
early-exit implementation only.
"""

import functools
from pathlib import Path

from hydroforge.runtime.cuda_kernel import (
    load_inline_cu_module as _load_inline_module,
    precompile_extension_builders,
)

_DIR = Path(__file__).resolve().parent
_MODULE_PREFIX = "cmfgpu_cuda"


def _load_or_build(name, cpp, src, cflags, funcs):
    return _load_inline_module(
        name,
        cpp_sources=cpp,
        cuda_sources=src,
        functions=funcs,
        extra_cuda_cflags=cflags,
    )


@functools.lru_cache(maxsize=1)
def _ext():
    cpp = (
        "void launch_flood_stage(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,"
        "at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,"
        "at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,"
        "at::Tensor,long,long,long,long);\n"
    )
    src = (_DIR / "storage.cu").read_text()
    return _load_or_build(f"{_MODULE_PREFIX}_storage", cpp, src, ["-O3", "--use_fast_math"],
                          ["launch_flood_stage"])


@functools.lru_cache(maxsize=1)
def _ext_outflow():
    """Compile the outflow / inflow kernels.

    Built **without** ``--use_fast_math`` so that ``/`` and ``sqrt`` keep IEEE
    round-to-nearest, matching Triton's default precise lowering (these kernels
    are division / sqrt / cbrt heavy and benefit from exact rounding parity).
    """
    src = (_DIR / "outflow.cu").read_text()
    cpp = (
        "void launch_outflow(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,"
        "at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,"
        "at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,"
        "at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,"
        "double,at::Tensor,long,long,at::Tensor,long,double,long);\n"
        "void launch_inflow(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,"
        "at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,long,long,long);\n"
    )
    return _load_or_build(f"{_MODULE_PREFIX}_outflow", cpp, src, ["-O3", "--use_fast_math"],
                          ["launch_outflow", "launch_inflow"])


@functools.lru_cache(maxsize=1)
def _ext_adaptive():
    src = (_DIR / "adaptive_time.cu").read_text()
    cpp = (
        "void launch_adaptive_time(at::Tensor,at::Tensor,at::Tensor,at::Tensor,"
        "double,double,double,long,long,long);\n"
    )
    return _load_or_build(f"{_MODULE_PREFIX}_adaptive", cpp, src, ["-O3", "--use_fast_math"],
                          ["launch_adaptive_time"])


@functools.lru_cache(maxsize=1)
def _ext_bifurcation():
    """Bifurcation outflow/inflow — precise math (slope div may yield inf)."""
    src = (_DIR / "bifurcation.cu").read_text()
    cpp = (
        "void launch_bif_outflow(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,"
        "at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double,at::Tensor,"
        "long,long,long);\n"
        "void launch_bif_inflow(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,"
        "long,long,long);\n"
    )
    return _load_or_build(f"{_MODULE_PREFIX}_bifurcation", cpp, src, ["-O3", "--use_fast_math"],
                          ["launch_bif_outflow", "launch_bif_inflow"])


@functools.lru_cache(maxsize=1)
def _ext_reservoir():
    """Reservoir outflow — precise math (sqrt/exp/log regime selection)."""
    src = (_DIR / "reservoir.cu").read_text()
    cpp = (
        "void launch_reservoir_outflow(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,"
        "at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,"
        "at::Tensor,at::Tensor,at::Tensor,at::Tensor,long,long);\n"
    )
    return _load_or_build(f"{_MODULE_PREFIX}_reservoir", cpp, src, ["-O3", "--use_fast_math"],
                          ["launch_reservoir_outflow"])


@functools.lru_cache(maxsize=1)
def _ext_levee():
    """Levee stage + levee bifurcation outflow CUDA kernels."""
    src = (_DIR / "levee.cu").read_text()
    cpp = (
        "void launch_levee_stage(at::Tensor,at::Tensor,at::Tensor,at::Tensor,"
        "at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,"
        "at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,"
        "long,long,long);\n"
        "void launch_levee_bif_outflow(at::Tensor,at::Tensor,at::Tensor,at::Tensor,"
        "at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,"
        "at::Tensor,at::Tensor,double,at::Tensor,long,long,long);\n"
    )
    return _load_or_build(f"{_MODULE_PREFIX}_levee", cpp, src, ["-O3", "--use_fast_math"],
                          ["launch_levee_stage", "launch_levee_bif_outflow"])


_EXTENSION_BUILDERS = (
    ("storage", "_ext"),
    ("outflow", "_ext_outflow"),
    ("adaptive", "_ext_adaptive"),
    ("bifurcation", "_ext_bifurcation"),
    ("reservoir", "_ext_reservoir"),
    ("levee", "_ext_levee"),
)


_precompiled = False


def _ensure_precompiled() -> None:
    """Build every extension in parallel on first dispatch (idempotent).

    Without this, a script that never called :func:`precompile_cuda_extensions`
    would compile the six extensions one at a time as each kernel is first
    touched.  Hooked at the dispatch layer rather than inside the per-extension
    ``_ext_*`` builders, because the worker subprocesses spawned by
    ``precompile_extension_builders`` invoke those builders directly and must
    not re-enter here.
    """
    global _precompiled
    if _precompiled:
        return
    _precompiled = True
    precompile_extension_builders(__name__, _EXTENSION_BUILDERS)


def precompile_cuda_extensions():
    """Precompile all cmfgpu CUDA extensions via hydroforge's shared loader."""
    global _precompiled
    _precompiled = True
    return precompile_extension_builders(__name__, _EXTENSION_BUILDERS)


# Cached zero-element placeholders for absent optional pointers.  These are
# device-constant, so allocating them once (instead of every dispatch call)
# removes ~16us/sub-step of aten::empty overhead from the hot loop.
_PLACEHOLDER_CACHE: dict = {}


def _ph_double(ref):
    import torch
    key = ("d", ref.device)
    t = _PLACEHOLDER_CACHE.get(key)
    if t is None:
        t = torch.empty(0, dtype=torch.float64, device=ref.device)
        _PLACEHOLDER_CACHE[key] = t
    return t


def _ph_float0(ref):
    import torch
    key = ("f0", ref.device)
    t = _PLACEHOLDER_CACHE.get(key)
    if t is None:
        t = torch.empty(0, dtype=torch.float32, device=ref.device)
        _PLACEHOLDER_CACHE[key] = t
    return t


def _ph_bool(ref):
    import torch
    key = ("b", ref.device)
    t = _PLACEHOLDER_CACHE.get(key)
    if t is None:
        t = torch.empty(0, dtype=torch.bool, device=ref.device)
        _PLACEHOLDER_CACHE[key] = t
    return t


def _ph_runoff(ref, n):
    """Cached zero runoff buffer (read-only in the kernels, so sharing is safe)."""
    import torch
    key = ("runoff", ref.device, n)
    t = _PLACEHOLDER_CACHE.get(key)
    if t is None:
        t = torch.zeros(n, dtype=torch.float32, device=ref.device)
        _PLACEHOLDER_CACHE[key] = t
    return t


def compute_flood_stage(**kw):
    """Dispatch the fused storage-update + flood-stage CUDA kernel.

    Accepts the same kwargs as the Triton ``compute_flood_stage`` dispatcher.
    Optional pointers (``global_bifurcation_outflow_ptr`` when bifurcation is
    off) may be ``None``; a zero-element placeholder is substituted.
    """
    _ensure_precompiled()
    ext = _ext()
    block = int(kw.get("BLOCK_SIZE", 256))
    num_catchments = int(kw["num_catchments"])
    num_flood_levels = int(kw["num_flood_levels"])
    has_bif = 1 if kw.get("HAS_BIFURCATION", True) else 0

    river_outflow = kw["river_outflow_ptr"]
    gbif = kw.get("global_bifurcation_outflow_ptr")
    if gbif is None:
        gbif = _ph_double(river_outflow)
        has_bif = 0
    runoff = kw.get("runoff_ptr")
    if runoff is None:
        runoff = _ph_runoff(river_outflow, num_catchments)

    ext.launch_flood_stage(
        kw["river_inflow_ptr"], kw["flood_inflow_ptr"],
        river_outflow, kw["flood_outflow_ptr"],
        gbif, runoff, kw["time_step_ptr"],
        kw["outgoing_storage_ptr"],
        kw["river_storage_ptr"], kw["flood_storage_ptr"], kw["protected_storage_ptr"],
        kw["river_depth_ptr"], kw["flood_depth_ptr"], kw["protected_depth_ptr"],
        kw["flood_fraction_ptr"],
        kw["river_height_ptr"], kw["flood_depth_table_ptr"],
        kw["catchment_area_ptr"], kw["river_width_ptr"], kw["river_length_ptr"],
        num_catchments, num_flood_levels, has_bif, block,
    )


# The log / batched variants are not yet ported to CUDA; callers that need them
# should keep the Triton backend for those paths.
compute_flood_stage_log = None


def compute_outflow(**kw):
    """Dispatch the fused outflow + outgoing-storage CUDA kernel.

    Accepts the same kwargs as the Triton ``compute_outflow`` dispatcher.
    ``global_bifurcation_outflow_ptr`` may be ``None`` (bifurcation off) and
    ``is_dam_upstream_ptr`` may be ``None`` (reservoir off); zero-element
    placeholders are substituted and the corresponding flag cleared.
    """
    _ensure_precompiled()
    ext = _ext_outflow()
    block = int(kw.get("BLOCK_SIZE", 256))
    num_catchments = int(kw["num_catchments"])
    has_bif = 1 if kw.get("HAS_BIFURCATION", True) else 0
    has_res = 1 if kw.get("HAS_RESERVOIR", False) else 0

    river_outflow = kw["river_outflow_ptr"]
    gbif = kw.get("global_bifurcation_outflow_ptr")
    if gbif is None:
        gbif = _ph_double(river_outflow)
        has_bif = 0
    is_dam_up = kw.get("is_dam_upstream_ptr")
    if is_dam_up is None:
        is_dam_up = _ph_bool(river_outflow)
        has_res = 0

    ext.launch_outflow(
        kw["downstream_idx_ptr"],
        kw["river_inflow_ptr"], river_outflow, kw["river_manning_ptr"],
        kw["river_depth_ptr"], kw["river_width_ptr"], kw["river_length_ptr"],
        kw["river_height_ptr"], kw["river_storage_ptr"],
        kw["flood_inflow_ptr"], kw["flood_outflow_ptr"], kw["flood_manning_ptr"],
        kw["flood_depth_ptr"], kw["protected_depth_ptr"], kw["catchment_elevation_ptr"],
        kw["downstream_distance_ptr"], kw["flood_storage_ptr"], kw["protected_storage_ptr"],
        kw["river_cross_section_depth_ptr"], kw["flood_cross_section_depth_ptr"],
        kw["flood_cross_section_area_ptr"],
        gbif, kw["total_storage_ptr"],
        kw["outgoing_storage_ptr"], kw["water_surface_elevation_ptr"],
        kw["protected_water_surface_elevation_ptr"],
        float(kw["gravity"]), kw["time_step_ptr"],
        num_catchments, has_bif,
        is_dam_up, has_res, float(kw.get("MIN_KINEMATIC_SLOPE", 1.0e-5)),
        block,
    )


def compute_inflow(**kw):
    """Dispatch the inflow-limiting + scatter CUDA kernel.

    Accepts the same kwargs as the Triton ``compute_inflow`` dispatcher.
    ``reservoir_total_inflow_ptr`` / ``is_reservoir_ptr`` may be ``None`` when
    the reservoir module is inactive.
    """
    _ensure_precompiled()
    ext = _ext_outflow()
    block = int(kw.get("BLOCK_SIZE", 256))
    num_catchments = int(kw["num_catchments"])
    has_res = 1 if kw.get("HAS_RESERVOIR", False) else 0

    river_outflow = kw["river_outflow_ptr"]
    res_inflow = kw.get("reservoir_total_inflow_ptr")
    is_res = kw.get("is_reservoir_ptr")
    if res_inflow is None:
        res_inflow = _ph_double(river_outflow)
        has_res = 0
    if is_res is None:
        is_res = _ph_bool(river_outflow)

    ext.launch_inflow(
        kw["downstream_idx_ptr"],
        river_outflow, kw["flood_outflow_ptr"],
        kw["river_storage_ptr"], kw["flood_storage_ptr"], kw["outgoing_storage_ptr"],
        kw["river_inflow_ptr"], kw["flood_inflow_ptr"], kw["limit_rate_ptr"],
        res_inflow, is_res,
        num_catchments, has_res, block,
    )


def compute_adaptive_time_step(**kw):
    """Dispatch the CFL adaptive-time-step CUDA kernel (per-thread atomicMax).

    ``time_step`` is a runtime scalar (matches the Triton kernel signature, not
    a pointer).  ``is_dam_related_ptr`` may be ``None`` when reservoir is off.
    """
    _ensure_precompiled()
    ext = _ext_adaptive()
    block = int(kw.get("BLOCK_SIZE", 256))
    num_catchments = int(kw["num_catchments"])
    has_res = 1 if kw.get("HAS_RESERVOIR", False) else 0
    river_depth = kw["river_depth_ptr"]
    is_dam = kw.get("is_dam_related_ptr")
    if is_dam is None:
        is_dam = _ph_bool(river_depth)
        has_res = 0
    ext.launch_adaptive_time(
        river_depth, kw["downstream_distance_ptr"], is_dam, kw["max_sub_steps_ptr"],
        float(kw["time_step"]), float(kw["adaptive_time_factor"]), float(kw["gravity"]),
        num_catchments, has_res, block,
    )


def compute_bifurcation_outflow(**kw):
    """Dispatch the bifurcation outflow CUDA kernel."""
    _ensure_precompiled()
    ext = _ext_bifurcation()
    block = int(kw.get("BLOCK_SIZE", 256))
    num_paths = int(kw["num_bifurcation_paths"])
    num_levels = int(kw["num_bifurcation_levels"])
    ext.launch_bif_outflow(
        kw["bifurcation_catchment_idx_ptr"], kw["bifurcation_downstream_idx_ptr"],
        kw["bifurcation_manning_ptr"], kw["bifurcation_outflow_ptr"],
        kw["bifurcation_width_ptr"], kw["bifurcation_length_ptr"],
        kw["bifurcation_elevation_ptr"], kw["bifurcation_cross_section_depth_ptr"],
        kw["water_surface_elevation_ptr"], kw["total_storage_ptr"],
        kw["outgoing_storage_ptr"],
        float(kw["gravity"]), kw["time_step_ptr"],
        num_paths, num_levels, block,
    )


def compute_bifurcation_inflow(**kw):
    """Dispatch the bifurcation inflow (limiter + global scatter) CUDA kernel."""
    _ensure_precompiled()
    ext = _ext_bifurcation()
    block = int(kw.get("BLOCK_SIZE", 256))
    num_paths = int(kw["num_bifurcation_paths"])
    num_levels = int(kw["num_bifurcation_levels"])
    ext.launch_bif_inflow(
        kw["bifurcation_catchment_idx_ptr"], kw["bifurcation_downstream_idx_ptr"],
        kw["limit_rate_ptr"], kw["bifurcation_outflow_ptr"],
        kw["global_bifurcation_outflow_ptr"],
        num_paths, num_levels, block,
    )


def compute_reservoir_outflow(**kw):
    """Dispatch the reservoir outflow CUDA kernel.

    ``runoff_ptr`` is supplied at call time (like flood-stage); the reservoir
    parameter arrays are reservoir-indexed.
    """
    import torch
    _ensure_precompiled()
    ext = _ext_reservoir()
    block = int(kw.get("BLOCK_SIZE", 256))
    num_reservoirs = int(kw["num_reservoirs"])
    runoff = kw.get("runoff_ptr")
    if runoff is None:
        runoff = _ph_runoff(kw["river_outflow_ptr"], 1)
    ext.launch_reservoir_outflow(
        kw["reservoir_catchment_idx_ptr"], kw["downstream_idx_ptr"],
        kw["reservoir_total_inflow_ptr"], kw["river_outflow_ptr"], kw["flood_outflow_ptr"],
        kw["river_storage_ptr"], kw["flood_storage_ptr"],
        kw["conservation_volume_ptr"], kw["emergency_volume_ptr"], kw["adjustment_volume_ptr"],
        kw["normal_outflow_ptr"], kw["adjustment_outflow_ptr"], kw["flood_control_outflow_ptr"],
        runoff, kw["total_storage_ptr"], kw["outgoing_storage_ptr"],
        kw["time_step_ptr"], num_reservoirs, block,
    )


def compute_levee_stage(**kw):
    """Dispatch the levee-aware flood-stage CUDA kernel."""
    _ensure_precompiled()
    ext = _ext_levee()
    block = int(kw.get("BLOCK_SIZE", 256))
    ext.launch_levee_stage(
        kw["levee_catchment_idx_ptr"],
        kw["river_storage_ptr"], kw["flood_storage_ptr"], kw["protected_storage_ptr"],
        kw["river_depth_ptr"], kw["flood_depth_ptr"], kw["protected_depth_ptr"],
        kw["river_height_ptr"], kw["flood_depth_table_ptr"],
        kw["catchment_area_ptr"], kw["river_width_ptr"], kw["river_length_ptr"],
        kw["levee_base_height_ptr"], kw["levee_crown_height_ptr"],
        kw["levee_fraction_ptr"], kw["flood_fraction_ptr"],
        int(kw["num_levees"]), int(kw["num_flood_levels"]), block,
    )


# Log / batched levee variants are not yet ported to CUDA.  Matching storage,
# callers fall back to the non-log CUDA stage when this is None.
compute_levee_stage_log = None


def compute_levee_bifurcation_outflow(**kw):
    """Dispatch the levee-aware bifurcation outflow CUDA kernel."""
    _ensure_precompiled()
    ext = _ext_levee()
    block = int(kw.get("BLOCK_SIZE", 256))
    ext.launch_levee_bif_outflow(
        kw["bifurcation_catchment_idx_ptr"], kw["bifurcation_downstream_idx_ptr"],
        kw["bifurcation_manning_ptr"], kw["bifurcation_outflow_ptr"],
        kw["bifurcation_width_ptr"], kw["bifurcation_length_ptr"],
        kw["bifurcation_elevation_ptr"], kw["bifurcation_cross_section_depth_ptr"],
        kw["water_surface_elevation_ptr"], kw["protected_water_surface_elevation_ptr"],
        kw["total_storage_ptr"], kw["outgoing_storage_ptr"],
        float(kw["gravity"]), kw["time_step_ptr"],
        int(kw["num_bifurcation_paths"]), int(kw["num_bifurcation_levels"]), block,
    )
