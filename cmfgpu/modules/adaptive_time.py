# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from functools import cached_property
from typing import ClassVar

import torch
from pydantic import Field

from cmfgpu.modules.abstract_module import (AbstractModule,
                                            computed_tensor_field)


class AdaptiveTimeModule(AbstractModule):
    """
    Adaptive time step calculation module for river networks.
    """
    
    # Module metadata
    module_name: ClassVar[str] = "adaptive_time"
    description: ClassVar[str] = "Adaptive time step calculation module for river networks"
    dependencies: ClassVar[list] = ["base"]
    adaptive_time_factor: float = Field(0.7, description="Factor to adjust adaptive time step calculation", gt=0.0)
    
    @computed_tensor_field(
        description="Minimum time step across all processes",
        shape=("one",),
        category="shared_state",
    )
    @cached_property
    def min_time_sub_step(self) -> torch.Tensor:
        return torch.zeros((1,), dtype=self.precision, device=self.device)

    @computed_tensor_field(
        description="Maximum number of sub-steps across all processes",
        shape=("one",),
        category="shared_state",
        dtype="int",
    )
    @cached_property
    def max_sub_steps(self) -> torch.Tensor:
        return torch.zeros((1,), dtype=torch.int32, device=self.device)
    
    @cached_property
    def one(self) -> int:
        return 1
