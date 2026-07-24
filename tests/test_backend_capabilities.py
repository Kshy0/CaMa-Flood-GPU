from pathlib import Path

import pytest
import torch

from cmfgpu.models.cama_flood_model import CaMaFlood
from cmfgpu.phys.adaptive_time import compute_adaptive_time_step
from cmfgpu.phys.outflow import compute_outflow
from cmfgpu.phys.reservoir import compute_reservoir_outflow
from cmfgpu.phys.specs import ADAPTIVE_TIME, _spec


def test_cama_specs_require_one_explicit_access_for_every_pointer():
    with pytest.raises(ValueError, match="every pointer requires one explicit"):
        _spec(
            "incomplete_access",
            ("state_ptr", "num_catchments"),
            "num_catchments",
        )

    with pytest.raises(ValueError, match="conflicting access"):
        _spec(
            "conflicting_access",
            ("state_ptr", "num_catchments"),
            "num_catchments",
            read=("state_ptr",),
            write=("state_ptr",),
        )


def test_cuda_declares_trials_unsupported_in_backend_contract():
    assert CaMaFlood.backend_requirements["cuda"].trials is False
    assert CaMaFlood.backend_requirements["cuda"].precision is None


def test_cuda_sources_dispatch_real_and_storage_precision_independently():
    root = Path(__file__).parents[1] / "cmfgpu" / "phys" / "cuda"
    dual_precision = {
        "bifurcation.cu",
        "outflow.cu",
        "storage.cu",
        "reservoir.cu",
        "levee.cu",
    }
    for filename in dual_precision:
        source = (root / filename).read_text()
        assert "double, double" in source
        assert "float, double" in source
        assert "float, float" in source

    adaptive = (root / "adaptive_time.cu").read_text()
    assert "k_adaptive_time<double>" in adaptive
    assert "k_adaptive_time<float>" in adaptive


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_cuda_adaptive_time_executes_in_declared_compute_precision(dtype):
    implementation = compute_adaptive_time_step.registry.resolve(
        "cuda",
    ).implementation
    river_depth = torch.tensor([1.0, 4.0], device="cuda", dtype=dtype)
    downstream_distance = torch.full(
        (2,),
        1000.0,
        device="cuda",
        dtype=dtype,
    )
    max_sub_steps = torch.zeros(1, device="cuda", dtype=torch.int32)
    arguments = {
        "river_depth_ptr": river_depth,
        "downstream_distance_ptr": downstream_distance,
        "is_dam_related_ptr": None,
        "max_sub_steps_ptr": max_sub_steps,
        "outer_time_step": 3600.0,
        "adaptive_time_factor": 0.7,
        "gravity": 9.80665,
        "num_catchments": 2,
        "HAS_RESERVOIR": False,
        "num_trials": 1,
        "batched_downstream_distance": False,
        "BLOCK_SIZE": 256,
    }
    buffer_dtypes = {
        "river_depth_ptr": dtype,
        "downstream_distance_ptr": dtype,
        "is_dam_related_ptr": torch.bool,
        "max_sub_steps_ptr": torch.int32,
    }

    implementation.specialize(
        arguments,
        frozenset(),
        buffer_dtypes=buffer_dtypes,
    )()
    torch.cuda.synchronize()

    assert max_sub_steps.item() == 33


def test_outflow_reservoir_contract_is_identical_across_backends():
    metadata = compute_outflow.metadata_by_backend()
    expected = set(metadata["cuda"].parameters)

    for backend in ("cuda", "triton", "metal"):
        actual = metadata[backend]
        assert set(actual.parameters) == expected
        assert actual.optional_buffers["is_dam_upstream_ptr"] == "HAS_RESERVOIR"
        assert actual.compile_time["HAS_RESERVOIR"] == "bool"


def test_metal_outflow_source_has_one_trial_aware_dam_upstream_override():
    source = (
        Path(__file__).parents[1] / "cmfgpu" / "phys" / "metal" / "outflow.metal"
    ).read_text()

    assert source.count("if (HAS_RESERVOIR && args.is_dam_upstream_ptr[") == 1
    assert source.count("MIN_KINEMATIC_SLOPE") >= 1
    assert source.count("flood_storage / time_step") >= 1


def test_adaptive_metal_abi_is_generated_from_spec_only():
    root = Path(__file__).parents[1] / "cmfgpu" / "phys" / "metal"
    authored_source = (root / "adaptive_time.metal").read_text()
    implementation = compute_adaptive_time_step.registry.resolve(
        "metal",
    ).implementation
    source = implementation.source_for_types(
        {
            "river_depth_ptr": torch.float32,
            "downstream_distance_ptr": torch.float32,
            "is_dam_related_ptr": torch.bool,
            "max_sub_steps_ptr": torch.int32,
        }
    )

    assert authored_source.count("// HYDROFORGE METAL KERNEL BODY") == 1
    assert "[[id(" not in authored_source
    assert "kernel void" not in authored_source
    assert implementation.parallel_axes == ("num_trials",)
    assert "device atomic_int* max_sub_steps_ptr" in source
    assert "constant constexpr uint HF_BLOCK_SIZE = 256;" in source
    assert (
        compute_adaptive_time_step.metadata.buffers["max_sub_steps_ptr"] == "atomic_max"
    )


def test_adaptive_outer_interval_is_not_bound_to_substep_state():
    assert "outer_time_step" in ADAPTIVE_TIME.parameters
    assert "time_step" not in ADAPTIVE_TIME.parameters


def test_metal_sources_group_related_physics_bodies():
    root = Path(__file__).parents[1] / "cmfgpu" / "phys" / "metal"
    expected = {
        "adaptive_time.metal": ("compute_adaptive_time_step",),
        "outflow.metal": ("compute_outflow", "compute_inflow"),
        "storage.metal": ("compute_flood_stage", "compute_flood_stage_log"),
        "bifurcation.metal": (
            "compute_bifurcation_outflow",
            "compute_bifurcation_inflow",
        ),
        "reservoir.metal": ("compute_reservoir_outflow",),
        "levee.metal": (
            "compute_levee_stage",
            "compute_levee_stage_log",
            "compute_levee_bifurcation_outflow",
        ),
    }
    assert {path.name for path in root.glob("*.metal")} == set(expected)
    marker = "// HYDROFORGE METAL KERNEL BODY"
    for filename, bodies in expected.items():
        source = (root / filename).read_text()
        if len(bodies) == 1:
            assert source.count(marker) == 1
        else:
            assert (
                tuple(
                    line.split(":", 1)[1].strip()
                    for line in source.splitlines()
                    if line.startswith(f"{marker}:")
                )
                == bodies
            )


def test_reservoir_metal_abi_and_trial_axis_are_generated_from_spec_only():
    root = Path(__file__).parents[1] / "cmfgpu" / "phys" / "metal"
    authored_source = (root / "reservoir.metal").read_text()
    implementation = compute_reservoir_outflow.registry.resolve(
        "metal",
    ).implementation
    source = implementation.source_for_types(
        {
            name: (
                torch.int32
                if name
                in {
                    "reservoir_catchment_idx_ptr",
                    "downstream_idx_ptr",
                }
                else torch.float32
            )
            for name in compute_reservoir_outflow.metadata.buffers
        }
    )

    assert authored_source.count("// HYDROFORGE METAL KERNEL BODY") == 1
    assert "[[id(" not in authored_source
    assert "kernel void" not in authored_source
    assert implementation.parallel_axes == ("num_trials",)
    assert "device float* reservoir_total_inflow_ptr" in source
    assert "device atomic_float* outgoing_storage_ptr" in source
    assert (
        compute_reservoir_outflow.metadata.buffers["reservoir_total_inflow_ptr"]
        == "read_write"
    )
    assert (
        compute_reservoir_outflow.metadata.buffers["outgoing_storage_ptr"]
        == "atomic_add"
    )
    assert compute_reservoir_outflow.metadata.buffers["total_storage_ptr"] == "write"
