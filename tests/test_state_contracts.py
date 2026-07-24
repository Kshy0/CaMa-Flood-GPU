"""State-lifetime contracts for streamed CaMa-Flood inputs."""

from cmfgpu.modules.base import BaseModule
from cmfgpu.modules.inflow import InflowModule
from cmfgpu.modules.sea_level import SeaLevelModule


def _category(module, field: str) -> str:
    return module.get_tensor_schema(field).tensor.category


def test_required_step_inputs_are_not_checkpoint_state() -> None:
    assert _category(BaseModule, "runoff") == "state"
    assert _category(InflowModule, "inflow") == "state"
    assert _category(SeaLevelModule, "sea_surface_elevation") == "state"
