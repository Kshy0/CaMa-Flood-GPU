# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Log module for CaMa-Flood-GPU using TensorField / computed_tensor_field helpers.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import ClassVar, List, Literal, Tuple

import numpy as np
import torch
import torch.distributed as dist
from pydantic import Field, PrivateAttr

from cmfgpu.modules.abstract_module import (AbstractModule,
                                            computed_tensor_field)


def computed_log_field(
    description: str,
    shape: Tuple[str, ...] = ("log_buffer_size",),
    dtype: Literal["float", "int", "bool"] = "float",
    **kwargs
):
    return computed_tensor_field(
        description=description,
        shape=shape,
        dtype=dtype,
        **kwargs
    )


class LogModule(AbstractModule):
    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    module_name: ClassVar[str] = "log"
    description: ClassVar[str] = "Log module for storing and managing simulation data"
    dependencies: ClassVar[list] = ["base"]

    log_buffer_size: int = Field(
        default=1000,
        description="Size of the log buffer",
        ge=0,
    )

    _time_step: float = PrivateAttr()
    _num_steps: int = PrivateAttr()
    _current_time: datetime = PrivateAttr()
    _times: List[datetime] = PrivateAttr(default_factory=list)
    _log_initialized: bool = PrivateAttr(default=False)

    # ------------------------------------------------------------------ #
    # Methods
    # ------------------------------------------------------------------ #
    @cached_property
    def log_vars(self) -> List[str]:
        return list(LogModule.model_computed_fields.keys())

    def write_header(self, log_path: Path) -> None:
        headers = [
            "StepStartTime", "StoragePre", "StorageNext",
            "StorageNew", "InflowError", "Inflow",
            "Outflow", "StorageStage", "StageError",
            "RiverStorage", "FloodStorage", "FloodArea",
        ]
        widths = [18] + [16] * (len(headers) - 1)
        with log_path.open("w") as f:
            f.write(
                "".join(
                    f"{h:<{w}}" if i == 0 else f"{h:>{w}}"
                    for i, (h, w) in enumerate(zip(headers, widths))
                )
                + "\n"
            )

    def set_time(self, time_step: float, num_steps: int, current_time: datetime) -> None:
        if not isinstance(current_time, datetime):
            raise ValueError(
                f"`current_time` must be a `datetime.datetime` instance. "
                f"Got {type(current_time).__name__} instead. "
                f"This error occurred because the log module is activated "
            )
        self._time_step = time_step
        self._num_steps = num_steps
        self._current_time = current_time
        self._times = [
            self._current_time + timedelta(seconds=time_step * i) for i in range(num_steps)
        ]
        if num_steps > self.log_buffer_size:
            self.log_buffer_size = num_steps + 20
            for field in self.log_vars:
                getattr(self, field).resize_(self.log_buffer_size)
    
    def gather_results(self) -> None:
        """
        Gathers results from the model and prepares them for logging.
        This method should be called after each time step to collect data.
        """
        for field in self.log_vars:
            dist.reduce(getattr(self, field), dst=0, op=dist.ReduceOp.SUM)

    def write_step(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._log_initialized:
            self.write_header(log_path)
            self._log_initialized = True
        with log_path.open("a") as f:
            f.write(
                f"Time Step: {self._time_step:.4f} seconds    Number of Steps: {self._num_steps}\n"
            )
        print(f"Processed step at {self._current_time.strftime('%Y-%m-%d %H:%M:%S')}, adaptive_time_step={self._num_steps}")

        num_steps = self._num_steps
        time_strs = np.array(
            [t.strftime("%Y-%m-%d %H:%M") for t in self._times[:num_steps]], dtype=str
        )
        data_arrays = {
            field: getattr(self, field).cpu().numpy()[:num_steps] for field in self.log_vars
        }
        fmt = ["%-18s"] + ["%16.6g"] * 3 + ["%16.3e"] + ["%16.6g"] * 3 + ["%16.3e"] + [
            "%16.6g"
        ] * 3
        with log_path.open("a") as f:
            for i in range(num_steps):
                row = [time_strs[i]] + [data_arrays[field][i] for field in self.log_vars]
                f.write("".join(f_ % v for f_, v in zip(fmt, row)) + "\n")
        for field in self.log_vars:
            getattr(self, field).zero_()

    # ------------------------------------------------------------------ #
    # Computed tensor fields (log buffers)
    # ------------------------------------------------------------------ #
    @computed_log_field(
        description="Running sum of storage before routing step",
    )
    @cached_property
    def total_storage_pre_sum(self) -> torch.Tensor:
        return torch.zeros((self.log_buffer_size,), dtype=self.precision, device=self.device)

    @computed_log_field(
        description="Running sum of storage after routing step",
    )
    @cached_property
    def total_storage_next_sum(self) -> torch.Tensor:
        return torch.zeros((self.log_buffer_size,), dtype=self.precision, device=self.device)

    @computed_log_field(
        description="Running sum of new storage",
    )
    @cached_property
    def total_storage_new_sum(self) -> torch.Tensor:
        return torch.zeros((self.log_buffer_size,), dtype=self.precision, device=self.device)

    @computed_log_field(
        description="Running sum of inflow errors",
    )
    @cached_property
    def total_inflow_error_sum(self) -> torch.Tensor:
        return torch.zeros((self.log_buffer_size,), dtype=self.precision, device=self.device)

    @computed_log_field(
        description="Running sum of inflow",
    )
    @cached_property
    def total_inflow_sum(self) -> torch.Tensor:
        return torch.zeros((self.log_buffer_size,), dtype=self.precision, device=self.device)

    @computed_log_field(
        description="Running sum of outflow",
    )
    @cached_property
    def total_outflow_sum(self) -> torch.Tensor:
        return torch.zeros((self.log_buffer_size,), dtype=self.precision, device=self.device)

    @computed_log_field(
        description="Running sum of storage stage",
    )
    @cached_property
    def total_storage_stage_sum(self) -> torch.Tensor:
        return torch.zeros((self.log_buffer_size,), dtype=self.precision, device=self.device)

    @computed_log_field(
        description="Running sum of stage error",
    )
    @cached_property
    def total_stage_error_sum(self) -> torch.Tensor:
        return torch.zeros((self.log_buffer_size,), dtype=self.precision, device=self.device)

    @computed_log_field(
        description="Running sum of river storage",
    )
    @cached_property
    def river_storage_sum(self) -> torch.Tensor:
        return torch.zeros((self.log_buffer_size,), dtype=self.precision, device=self.device)

    @computed_log_field(
        description="Running sum of flood storage",
    )
    @cached_property
    def flood_storage_sum(self) -> torch.Tensor:
        return torch.zeros((self.log_buffer_size,), dtype=self.precision, device=self.device)

    @computed_log_field(
        description="Running sum of flood area",
    )
    @cached_property
    def flood_area_sum(self) -> torch.Tensor:
        return torch.zeros((self.log_buffer_size,), dtype=self.precision, device=self.device)
