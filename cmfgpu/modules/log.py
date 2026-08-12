# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Log module for CaMa-Flood-GPU using independent computed log buffers.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import ClassVar, List, Literal, Optional, Tuple, Union

import cftime
import numpy as np
import torch
from hydroforge.execution import reduce_many_
from hydroforge.model.module import (
    AbstractModule,
    computed_tensor_field,
    module_ref,
)
from pydantic import Field, PrivateAttr

from cmfgpu.modules.base import BaseModule


def computed_log_field(
    description: str,
    shape: Tuple[str, ...] = ("log_buffer_size",),
    dtype: Literal["float", "int", "bool"] = "float",
    category: Literal["topology", "derived_param", "state", "virtual"] = "state",
    expr: Optional[str] = None,
    **kwargs,
):
    return computed_tensor_field(
        description=description,
        shape=shape,
        dtype=dtype,
        category=category,
        expr=expr,
        **kwargs,
    )


class LogModule(AbstractModule):
    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    module_name: ClassVar[str] = "log"
    description: ClassVar[str] = "Log module for storing and managing simulation data"
    base = module_ref(BaseModule)

    log_buffer_size: int = Field(
        default=20000,
        description="Size of the log buffer (fixed constant for CUDA Graph compatibility)",
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
        return list(type(self).model_computed_fields.keys())

    def extra_log_columns(self) -> Tuple[Tuple[str, str], ...]:
        """Return subclass-owned ``(header, printf-format)`` log columns."""

        return ()

    def write_header(self, log_path: Path) -> None:
        headers = [
            "StepStartTime",
            "StoragePre",
            "StorageNext",
            "StorageNew",
            "InflowError",
            "Inflow",
            "Outflow",
            "StorageStage",
            "StageError",
            "RiverStorage",
            "FloodStorage",
            "FloodArea",
        ]
        headers.extend(header for header, _fmt in self.extra_log_columns())
        widths = [18] + [16] * (len(headers) - 1)
        with log_path.open("w") as f:
            f.write(
                "".join(
                    f"{h:<{w}}" if i == 0 else f"{h:>{w}}"
                    for i, (h, w) in enumerate(zip(headers, widths, strict=True))
                )
                + "\n"
            )

    def set_time(
        self,
        time_step: float,
        num_steps: int,
        current_time: Union[datetime, cftime.datetime],
    ) -> None:
        if type(time_step) not in {int, float}:
            raise TypeError("time_step must be an exact int or float")
        time_step = float(time_step)
        if not math.isfinite(time_step) or time_step <= 0:
            raise ValueError("time_step must be finite and greater than zero")
        if type(num_steps) is not int:
            raise TypeError("num_steps must be an exact int")
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if num_steps > self.log_buffer_size:
            raise ValueError(
                f"num_steps ({num_steps}) exceeds log_buffer_size "
                f"({self.log_buffer_size}). Increase log_buffer_size in the "
                "log module configuration."
            )
        if not isinstance(current_time, (datetime, cftime.datetime)):
            raise TypeError(
                "current_time must be a datetime or cftime.datetime value"
            )

        # Build the complete replacement before mutating private state.  A
        # datetime overflow (or any other construction failure) therefore
        # leaves the previous log interval intact.
        times = [
            current_time + timedelta(seconds=time_step * i)
            for i in range(num_steps)
        ]
        self._time_step = time_step
        self._num_steps = num_steps
        self._current_time = current_time
        self._times = times

    def gather_results(self) -> None:
        """
        Gathers results from the model and prepares them for logging.
        This method should be called after each time step to collect data.

        The eleven buffers reduce as one batch: a per-buffer reduce would pay
        eleven managed-step handshakes per outer step, which dominates the
        cost of this call in multi-rank runs.
        """
        reduce_many_(
            [getattr(self, field) for field in self.log_vars],
            destination=0, reduction="sum",
        )

    def clear_buffers(self) -> None:
        """Clear every per-substep accumulator before the next outer step."""
        for field in self.log_vars:
            getattr(self, field).zero_()

    def write_step(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._log_initialized:
            self.write_header(log_path)
            self._log_initialized = True
        with log_path.open("a") as f:
            f.write(
                f"Time Step: {self._time_step:.4f} seconds    Number of Steps: {self._num_steps}\n"
            )

        num_steps = self._num_steps
        time_strs = np.array(
            [t.strftime("%Y-%m-%d %H:%M") for t in self._times[:num_steps]], dtype=str
        )
        data_arrays = {
            field: getattr(self, field).cpu().numpy()[:num_steps]
            for field in self.log_vars
        }
        fmt = (
            ["%-18s"]
            + ["%16.6g"] * 3
            + ["%16.3e"]
            + ["%16.6g"] * 3
            + ["%16.3e"]
            + ["%16.6g"] * 3
        )
        fmt.extend(
            column_format
            for _header, column_format in self.extra_log_columns()
        )
        with log_path.open("a") as f:
            for i in range(num_steps):
                row = [time_strs[i]] + [
                    data_arrays[field][i] for field in self.log_vars
                ]
                f.write("".join(f_ % v for f_, v in zip(fmt, row, strict=True)) + "\n")
        self.clear_buffers()

    # ------------------------------------------------------------------ #
    # Computed tensor fields (independent log buffers)
    # ------------------------------------------------------------------ #
    @computed_log_field(
        description="Running sum of storage before routing step",
    )
    @cached_property
    def total_storage_pre_sum(self) -> torch.Tensor:
        return torch.zeros(
            self.log_buffer_size,
            dtype=self.precision,
            device=self.device,
        )

    @computed_log_field(
        description="Running sum of storage after routing step",
    )
    @cached_property
    def total_storage_next_sum(self) -> torch.Tensor:
        return torch.zeros(
            self.log_buffer_size,
            dtype=self.precision,
            device=self.device,
        )

    @computed_log_field(
        description="Running sum of new storage",
    )
    @cached_property
    def total_storage_new_sum(self) -> torch.Tensor:
        return torch.zeros(
            self.log_buffer_size,
            dtype=self.precision,
            device=self.device,
        )

    @computed_log_field(
        description="Running sum of inflow errors",
    )
    @cached_property
    def total_inflow_error_sum(self) -> torch.Tensor:
        return torch.zeros(
            self.log_buffer_size,
            dtype=self.precision,
            device=self.device,
        )

    @computed_log_field(
        description="Running sum of inflow",
    )
    @cached_property
    def total_inflow_sum(self) -> torch.Tensor:
        return torch.zeros(
            self.log_buffer_size,
            dtype=self.precision,
            device=self.device,
        )

    @computed_log_field(
        description="Running sum of outflow",
    )
    @cached_property
    def total_outflow_sum(self) -> torch.Tensor:
        return torch.zeros(
            self.log_buffer_size,
            dtype=self.precision,
            device=self.device,
        )

    @computed_log_field(
        description="Running sum of storage stage",
    )
    @cached_property
    def total_storage_stage_sum(self) -> torch.Tensor:
        return torch.zeros(
            self.log_buffer_size,
            dtype=self.precision,
            device=self.device,
        )

    @computed_log_field(
        description="Running sum of stage error",
    )
    @cached_property
    def total_stage_error_sum(self) -> torch.Tensor:
        return torch.zeros(
            self.log_buffer_size,
            dtype=self.precision,
            device=self.device,
        )

    @computed_log_field(
        description="Running sum of river storage",
    )
    @cached_property
    def river_storage_sum(self) -> torch.Tensor:
        return torch.zeros(
            self.log_buffer_size,
            dtype=self.precision,
            device=self.device,
        )

    @computed_log_field(
        description="Running sum of flood storage",
    )
    @cached_property
    def flood_storage_sum(self) -> torch.Tensor:
        return torch.zeros(
            self.log_buffer_size,
            dtype=self.precision,
            device=self.device,
        )

    @computed_log_field(
        description="Running sum of flood area",
    )
    @cached_property
    def flood_area_sum(self) -> torch.Tensor:
        return torch.zeros(
            self.log_buffer_size,
            dtype=self.precision,
            device=self.device,
        )
