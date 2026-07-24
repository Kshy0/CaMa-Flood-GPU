# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0

"""External gauge-inflow topology, timing metadata, and forcing buffers."""

from __future__ import annotations

from functools import cached_property
from typing import ClassVar, List, Literal, Optional, Self

import torch
from hydroforge.model.module import (AbstractModule, CoordinateField,
                                        ReferenceIndexField, TensorField)
from pydantic import Field, computed_field, model_validator

from cmfgpu.modules.base import BaseModule


class InflowModule(AbstractModule):
    """Keep prescribed inflow separate from runoff until the physics boundary."""

    module_name: ClassVar[str] = "inflow"
    description: ClassVar[str] = "External gauge inflow forcing"
    dependencies: ClassVar[List[str]] = ["base"]

    base: BaseModule = Field(exclude=True)

    inflow_catchment_id: torch.Tensor = CoordinateField(
        description="Catchment ID for each inflow injection gauge",
        dtype="int",
        shape=("num_inflow_gauges",),
        references="catchment_id",
    )

    basin_shift_days: Optional[torch.Tensor] = TensorField(
        description=(
            "Per-gauge day offset applied when reading prescribed inflow."
        ),
        dtype="int",
        shape=("num_inflow_gauges",),
        dim_coords="inflow_catchment_id",
        category="param",
        mode="cpu",
        default=None,
    )

    basin_valid_length_days: Optional[torch.Tensor] = TensorField(
        description="Per-gauge length of the contiguous valid observation span",
        dtype="int",
        shape=("num_inflow_gauges",),
        dim_coords="inflow_catchment_id",
        category="param",
        mode="cpu",
        default=None,
    )

    @computed_field(description="Number of inflow injection gauges")
    @cached_property
    def num_inflow_gauges(self) -> int:
        return self.inflow_catchment_id.shape[0]

    inflow_catchment_idx = ReferenceIndexField("inflow_catchment_id")
    catchment_inflow_idx = ReferenceIndexField(
        "inflow_catchment_id", inverse=True,
    )

    def check_topology(
        self,
        gauge_catchment_id: Optional[torch.Tensor] = None,
        inflow_catchment_id: Optional[torch.Tensor] = None,
        *,
        placement: Literal["same", "downstream"] = "downstream",
        require_headwater: bool = True,
    ) -> None:
        """Explicitly validate injection placement and retained mainstem inputs.

        ``downstream`` is the outlet-gauge convention: discharge observed at a
        catchment outlet enters its downstream catchment. ``same`` reproduces
        CaMa-Flood CPU's upstream-inflow convention. Bifurcations are ignored.
        """
        if inflow_catchment_id is None:
            inflow_catchment_id = self.inflow_catchment_id
        injection_id = inflow_catchment_id.to(self.base.catchment_id.device)
        if gauge_catchment_id is not None:
            gauge_id = gauge_catchment_id.to(
                device=self.base.catchment_id.device, dtype=self.base.catchment_id.dtype,
            )
            if gauge_id.shape != injection_id.shape:
                raise ValueError(
                    "gauge_catchment_id and inflow_catchment_id must have equal shape; "
                    "pass the unaggregated source-to-injection arrays when multiple "
                    "gauges feed one runtime inflow catchment"
                )
            if placement == "same":
                expected = gauge_id
            else:
                from hydroforge.data.distributed import find_indices_in_torch
                gauge_idx = find_indices_in_torch(gauge_id, self.base.catchment_id)
                if torch.any(gauge_idx < 0):
                    missing = gauge_id[gauge_idx < 0][:5].detach().cpu().tolist()
                    raise ValueError(f"Gauge catchments are not local: {missing}")
                expected = self.base.downstream_id[gauge_idx.to(torch.int64)]
            mismatch = injection_id != expected
            if torch.any(mismatch):
                rows = torch.nonzero(mismatch, as_tuple=False).flatten()[:10]
                detail = [
                    (int(gauge_id[i]), int(injection_id[i]), int(expected[i]))
                    for i in rows.detach().cpu().tolist()
                ]
                raise ValueError(
                    "Inflow placement mismatch as (gauge, actual, expected): "
                    f"{detail}"
                )

        if require_headwater:
            target = torch.unique(injection_id)
            source_id = self.base.catchment_id
            downstream_id = self.base.downstream_id
            violations = []
            for cid in target.detach().cpu().tolist():
                incoming = source_id[
                    (downstream_id == cid) & (source_id != cid)
                ]
                if incoming.numel():
                    violations.append((cid, incoming[:10].detach().cpu().tolist()))
            if violations:
                raise ValueError(
                    "Retained mainstem inflow exists above injection catchments: "
                    f"{violations[:10]}"
                )

    inflow: torch.Tensor = TensorField(
        description="Current compact prescribed inflow forcing",
        shape=("num_inflow_gauges",),
        dtype="float",
        dim_coords="inflow_catchment_id",
        category="state",
        output="disabled",
        default=0,
    )

    @model_validator(mode="after")
    def validate_valid_lengths(self) -> Self:
        if self.basin_valid_length_days is None:
            return self
        if (
            self.basin_shift_days is not None
            and self.basin_shift_days.numel()
            != self.basin_valid_length_days.numel()
        ):
            raise ValueError(
                "basin_shift_days and basin_valid_length_days must have equal size"
            )
        if torch.any(self.basin_valid_length_days <= 0):
            bad = self.inflow_catchment_id[
                self.basin_valid_length_days <= 0
            ].tolist()
            raise ValueError(
                "basin_valid_length_days must be > 0 for inflow catchments; "
                f"invalid catchment IDs: {bad}"
            )
        return self
