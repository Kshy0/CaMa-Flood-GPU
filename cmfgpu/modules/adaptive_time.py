from typing import ClassVar
import torch
from functools import cached_property
from pydantic import Field
from cmfgpu.modules.abstract_module import AbstractModule, computed_tensor_field

class AdaptiveTimeModule(AbstractModule):
    """
    Bifurcation hydraulic module for CaMa-Flood-GPU.
    Handles bifurcation flow calculations in river networks.
    """
    
    # Module metadata
    module_name: ClassVar[str] = "adaptive_time"
    description: ClassVar[str] = "Adaptive time step calculation module for river networks"
    dependencies: ClassVar[list] = ["base"]
    adaptive_time_factor: float = Field(0.7, description="Factor to adjust adaptive time step calculation", gt=0.0)
    
    @computed_tensor_field(
        description="Minimum time step across all processes",
        shape=("one",),
    )
    @cached_property
    def min_time_sub_step(self) -> torch.Tensor:
        return torch.zeros((1,), dtype=self.precision, device=self.device)
    
    @cached_property
    def one(self) -> int:
        return 1