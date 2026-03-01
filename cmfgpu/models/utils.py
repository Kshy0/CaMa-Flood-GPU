# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import cftime
import numpy as np
import torch
from numba import njit

if TYPE_CHECKING:
    from cmfgpu.models.abstract_model import AbstractModel
    from cmfgpu.modules.abstract_module import AbstractModule


@njit
def compute_group_to_rank(world_size: int, group_assignments: np.ndarray):
    """
    Compute a mapping from each original group ID to a rank, using a greedy load balance.

    Returns:
      full_map: array of length (max_original_id+1), where
                full_map[original_group_id] = assigned_rank (or -1 if absent)
    """
    # Handle edge cases early
    if world_size <= 0 or group_assignments.size == 0:
        max_gid = int(group_assignments.max()) if group_assignments.size > 0 else -1
        return np.full(max_gid + 1, -1, np.int64)

    # 1) Compress original IDs → 0..n_groups-1
    # unique_ids: sorted unique original IDs
    # inv: for each entry in group_assignments, its compressed ID
    unique_ids, inv = np.unique(group_assignments), None
    # compute inv via a dense id_map:
    max_gid = int(unique_ids[-1]) if unique_ids.size > 0 else -1
    id_map = np.full(max_gid + 1, -1, np.int64)
    id_map[unique_ids] = np.arange(unique_ids.size, dtype=np.int64)

    inv = id_map[group_assignments]
    n_groups = unique_ids.size

    # 2) Count sizes of each compressed group
    group_sizes = np.bincount(inv, minlength=n_groups).astype(np.int64)

    # 3) Greedy assignment: largest groups first → argmin(rank_loads)
    order = np.argsort(group_sizes)          # ascending
    rank_loads = np.zeros(world_size, np.int64)
    comp_to_rank = np.empty(n_groups, np.int64)

    for i in range(order.size - 1, -1, -1):  # iterate from largest to smallest
        g = order[i]
        r = int(np.argmin(rank_loads))       # lightest rank
        comp_to_rank[g] = r
        rank_loads[r] += group_sizes[g]

    # 4) Expand back to the full original ID space (fill -1 for IDs not present)
    full_map = np.full(max_gid + 1, -1, np.int64)
    # unique_ids are the only valid original IDs; assign directly
    full_map[unique_ids] = comp_to_rank

    return full_map


# ---------------------------------------------------------------------------
# ProgressTracker
# ---------------------------------------------------------------------------

@dataclass
class ProgressTracker:
    """Lightweight progress tracker with sliding-window speed estimation."""

    total_steps: int = 0
    current_step: int = 0
    _wall_start: float = 0.0
    _last_tick: float = 0.0
    _recent_dts: list = field(default_factory=list)
    _window_size: int = 50

    def start(self, total: int) -> None:
        """Reset and start tracking with *total* steps."""
        self.total_steps = total
        self.current_step = 0
        self._wall_start = _time.perf_counter()
        self._last_tick = self._wall_start
        self._recent_dts.clear()

    def tick(self) -> None:
        """Record one completed step."""
        now = _time.perf_counter()
        if self.current_step > 0:
            self._recent_dts.append(now - self._last_tick)
            if len(self._recent_dts) > self._window_size:
                self._recent_dts.pop(0)
        self._last_tick = now
        self.current_step += 1

    @property
    def elapsed(self) -> float:
        return _time.perf_counter() - self._wall_start

    @property
    def speed(self) -> float:
        """Steps per second (overall average)."""
        e = self.elapsed
        return self.current_step / e if e > 0 else 0.0

    @property
    def recent_speed(self) -> float:
        """Steps per second (sliding window)."""
        if not self._recent_dts:
            return self.speed
        return len(self._recent_dts) / sum(self._recent_dts)

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}min"
        else:
            return f"{seconds / 3600:.1f}h"

    def format_progress(self) -> str:
        """Return a short progress string like ``[25.3% 92/364] 2.8 steps/s ETA 1.6min``."""
        spd = self.recent_speed
        remaining = self.total_steps - self.current_step
        eta = remaining / spd if spd > 0 else float('inf')
        pct = self.current_step / max(self.total_steps, 1) * 100
        return (
            f"[{pct:5.1f}% {self.current_step}/{self.total_steps}] "
            f"{spd:.2f} steps/s ETA {self._fmt_duration(eta)}"
        )


# ---------------------------------------------------------------------------
# PlanItem / ActivePlan
# ---------------------------------------------------------------------------

@dataclass
class PlanItem:
    """A single parameter-change instruction."""

    # User inputs
    variable_name: str
    start_time: Union[datetime, cftime.datetime]
    active_steps: int = 1
    delta: Union[float, torch.Tensor] = 0.0
    target_value: Optional[Union[float, torch.Tensor]] = None
    target_ids: Optional[Union[List[int], torch.Tensor]] = None

    # Cached execution context (resolved once)
    _module: Optional[Any] = None
    _attr_name: str = ""
    _indices: Optional[torch.Tensor] = None
    _is_ready: bool = False

    @property
    def is_set_value(self) -> bool:
        return self.target_value is not None

    @property
    def is_incremental(self) -> bool:
        return not self.is_set_value


@dataclass
class ActivePlan:
    """Bookkeeping wrapper around a PlanItem that is currently being executed."""

    item: PlanItem
    steps_executed: int = 0
    executed_once: bool = False


# ---------------------------------------------------------------------------
# ParameterPlanMixin
# ---------------------------------------------------------------------------

class ParameterPlanMixin:
    """Mixin providing runtime parameter-change plan functionality.

    Expects the host class to expose:
      - ``self.variable_map``  (cached_property)
      - ``self.device``        (torch.device field)
      - ``self.rank``          (int field)
    """

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_grouped_changes(self: AbstractModel, module: AbstractModule, attr: str, plans: List[ActivePlan]):
        try:
            current_val = getattr(module, attr)
            is_tensor = isinstance(current_val, torch.Tensor)

            global_delta = 0.0
            global_set_value = None
            sparse_updates: List[Tuple] = []

            # Sort: Set values first, then Incremental
            plans.sort(key=lambda x: x.item.is_incremental)

            for active in plans:
                item = active.item

                if item.is_set_value:
                    val = item.target_value
                    is_set = True
                else:
                    val = item.delta
                    is_set = False

                active.steps_executed += 1
                active.executed_once = True

                if item._indices is None:
                    if is_set:
                        global_set_value = val
                        global_delta = 0.0
                    else:
                        global_delta += val
                else:
                    sparse_updates.append((item._indices, val, is_set))

            if is_tensor:
                if global_set_value is not None:
                    current_val.fill_(global_set_value)
                if global_delta != 0.0:
                    current_val.add_(global_delta)
                for indices, val, is_set in sparse_updates:
                    if is_set:
                        current_val[indices] = val
                    else:
                        current_val[indices] += val
            else:
                new_val = current_val
                if global_set_value is not None:
                    new_val = global_set_value
                new_val += global_delta
                if sparse_updates:
                    print(f"ParameterChangePlan Warning: Sparse updates ignored for scalar variable {attr}.")
                setattr(module, attr, new_val)

        except Exception as e:
            print(f"ParameterChangePlan Error: Failed to update {attr}. {e}")

    def _resolve_id_tensor(self: AbstractModel, module: AbstractModule, id_attr: Optional[str]) -> Optional[torch.Tensor]:
        """Resolve the ID tensor from a module given an attribute path."""
        if not id_attr:
            return None

        if "." in id_attr:
            parts = id_attr.split(".")
            curr = module
            found = True
            for part in parts:
                if hasattr(curr, part):
                    curr = getattr(curr, part)
                else:
                    found = False
                    break
            if found and isinstance(curr, torch.Tensor):
                return curr
        elif hasattr(module, id_attr):
            val = getattr(module, id_attr)
            if isinstance(val, torch.Tensor):
                return val

        if id_attr in self.variable_map:
            mod_inst, field_name, _ = self.variable_map[id_attr]
            if hasattr(mod_inst, field_name):
                val = getattr(mod_inst, field_name)
                if isinstance(val, torch.Tensor):
                    return val

        return None

    def _resolve_plan_item(self: AbstractModel, item: PlanItem) -> None:
        from cmfgpu.utils import find_indices_in_torch

        variable_map = self.variable_map

        if item.variable_name in variable_map:
            module, attr, id_attr = variable_map[item.variable_name]
            item._module = module
            item._attr_name = attr

            if item.target_ids is not None:
                id_tensor = self._resolve_id_tensor(module, id_attr)
                if id_tensor is not None:
                    if item.target_ids.device != id_tensor.device:
                        item.target_ids = item.target_ids.to(id_tensor.device)
                    indices = find_indices_in_torch(item.target_ids, id_tensor)
                    if torch.any(indices < 0):
                        raise ValueError(
                            f"ParameterChangePlan Error: Some target_ids for "
                            f"{item.variable_name} were not found in {id_attr}."
                        )
                    item._indices = indices
                else:
                    print(
                        f"ParameterChangePlan Warning: Cannot find ID tensor "
                        f"'{id_attr}' for {item.variable_name}. Applying to ALL."
                    )

            if isinstance(item.delta, torch.Tensor):
                item.delta = item.delta.to(module.device)
            if isinstance(item.target_value, torch.Tensor):
                item.target_value = item.target_value.to(module.device)

            item._is_ready = True
        else:
            print(f"ParameterChangePlan Warning: Variable {item.variable_name} not found in model.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_parameter_change_plan(self: AbstractModel, current_time: Union[datetime, cftime.datetime]) -> None:
        """Execute the plans for the current time step."""
        if current_time is None:
            return

        plans_changed = False

        while self._next_plan_idx < len(self._plans):
            plan = self._plans[self._next_plan_idx]
            if current_time >= plan.start_time:
                self._active_plans.append(ActivePlan(item=plan))
                self._next_plan_idx += 1
                plans_changed = True
            else:
                break

        if not self._active_plans:
            self._cached_grouped_plans = None
            return

        initial_count = len(self._active_plans)
        self._active_plans = [
            active for active in self._active_plans
            if active.steps_executed < active.item.active_steps
        ]
        if len(self._active_plans) != initial_count:
            plans_changed = True

        if not self._active_plans:
            self._cached_grouped_plans = None
            return

        if plans_changed or self._cached_grouped_plans is None:
            grouped_plans: Dict[Tuple[int, str], List[ActivePlan]] = {}
            for active in self._active_plans:
                if active.item._is_ready:
                    key = (id(active.item._module), active.item._attr_name)
                    grouped_plans.setdefault(key, []).append(active)
            self._cached_grouped_plans = grouped_plans

        for (_, attr), plans in self._cached_grouped_plans.items():
            if not plans:
                continue
            module = plans[0].item._module
            self._apply_grouped_changes(module, attr, plans)

    def add_parameter_change_plan(
        self: AbstractModel,
        variable_name: str,
        start_time: Union[datetime, cftime.datetime],
        active_steps: int = 1,
        delta: Union[float, torch.Tensor] = 0.0,
        target_value: Optional[Union[float, torch.Tensor]] = None,
        target_ids: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> None:
        """Add a parameter change plan."""
        if active_steps < 1:
            raise ValueError("active_steps must be >= 1")

        if target_ids is not None and not isinstance(target_ids, torch.Tensor):
            target_ids = torch.tensor(target_ids, dtype=torch.int64)

        item = PlanItem(
            variable_name=variable_name,
            start_time=start_time,
            active_steps=active_steps,
            delta=delta,
            target_value=target_value,
            target_ids=target_ids,
        )

        self._resolve_plan_item(item)
        self._plans.append(item)
        self._plans.sort(key=lambda x: x.start_time)
        self._next_plan_idx = 0
        self._active_plans.clear()

    def get_variable(self: AbstractModel, variable_name: str) -> Any:
        if variable_name not in self.variable_map:
            raise ValueError(f"Variable '{variable_name}' not found in model.")
        module, attr, _ = self.variable_map[variable_name]
        return getattr(module, attr)

    def set_variable_value(
        self: AbstractModel,
        variable_name: str,
        value: Union[float, torch.Tensor],
        target_ids: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> None:
        """Directly set the value of a variable for specific IDs immediately."""
        from cmfgpu.utils import find_indices_in_torch

        if variable_name not in self.variable_map:
            raise ValueError(f"Variable '{variable_name}' not found in model.")

        module, attr, id_attr = self.variable_map[variable_name]
        current_val = getattr(module, attr)

        if isinstance(value, torch.Tensor):
            value = value.to(self.device)

        if target_ids is None:
            if isinstance(current_val, torch.Tensor):
                current_val[:] = value
            else:
                setattr(module, attr, value)
            return

        if not isinstance(current_val, torch.Tensor):
            print(f"Warning: Ignoring target_ids for scalar variable '{variable_name}'. Updating globally.")
            setattr(module, attr, value)
            return

        id_tensor = self._resolve_id_tensor(module, id_attr)
        if id_tensor is None:
            raise ValueError(
                f"Cannot resolve ID tensor '{id_attr}' for variable "
                f"'{variable_name}', so target_ids cannot be used."
            )

        if not isinstance(target_ids, torch.Tensor):
            target_ids = torch.tensor(target_ids, dtype=torch.int64, device=self.device)
        else:
            target_ids = target_ids.to(self.device)

        if id_tensor.device != target_ids.device:
            target_ids = target_ids.to(id_tensor.device)

        indices = find_indices_in_torch(target_ids, id_tensor)
        if torch.any(indices < 0):
            raise ValueError(f"Some target_ids for '{variable_name}' were not found in '{id_attr}'.")
        current_val[indices] = value

    def summarize_plan(self: AbstractModel) -> None:
        """Print a summary of the parameter change plan and check for conflicts."""
        print(f"\n[rank {self.rank}] === Parameter Change Plan Summary ===")

        if not self._plans:
            print("No parameter change plans defined.")
            return

        sorted_plans = sorted(self._plans, key=lambda x: x.start_time)

        set_plans_map: Dict[Tuple[str, Any], List[PlanItem]] = {}
        for plan in sorted_plans:
            if plan.is_set_value:
                key = (plan.variable_name, plan.start_time)
                set_plans_map.setdefault(key, []).append(plan)

        conflicts: List[str] = []
        for (var_name, time), plans in set_plans_map.items():
            if len(plans) > 1:
                for i in range(len(plans)):
                    for j in range(i + 1, len(plans)):
                        p1 = plans[i]
                        p2 = plans[j]
                        if p1.target_ids is None or p2.target_ids is None:
                            conflicts.append(
                                f"Conflict: Variable '{var_name}' set multiple times at {time}. "
                                "One or both plans target ALL."
                            )
                            continue
                        ids1 = p1.target_ids
                        if isinstance(ids1, torch.Tensor):
                            ids1 = ids1.detach().cpu().numpy()
                        else:
                            ids1 = np.array(ids1)
                        ids2 = p2.target_ids
                        if isinstance(ids2, torch.Tensor):
                            ids2 = ids2.detach().cpu().numpy()
                        else:
                            ids2 = np.array(ids2)
                        intersection = np.intersect1d(ids1, ids2)
                        if intersection.size > 0:
                            sample_conflict = intersection[:5].tolist()
                            conflicts.append(
                                f"Conflict: Variable '{var_name}' set multiple times at {time} "
                                f"for IDs {sample_conflict}..."
                            )

        print(
            f"{'Time':<25} | {'Variable':<20} | {'Type':<8} | "
            f"{'Value':<10} | {'Steps':<10} | {'Target'}"
        )
        print("-" * 100)

        for plan in sorted_plans:
            type_str = "SET" if plan.is_set_value else "ADD"
            val = plan.target_value if plan.is_set_value else plan.delta
            if isinstance(val, torch.Tensor):
                val_str = f"{val.item():.4g}" if val.numel() == 1 else "Tensor"
            else:
                val_str = f"{val:.4g}"
            dur_str = f"{plan.active_steps}" if plan.is_incremental else "-"
            if plan.target_ids is None:
                target_str = "ALL"
            else:
                count = len(plan.target_ids)
                id_attr_name = "IDs"
                if plan.variable_name in self.variable_map:
                    _, _, id_attr = self.variable_map[plan.variable_name]
                    if id_attr:
                        id_attr_name = id_attr
                if count <= 5:
                    ids_list = (
                        plan.target_ids.tolist()
                        if isinstance(plan.target_ids, torch.Tensor)
                        else plan.target_ids
                    )
                    target_str = f"{str(ids_list)} ({id_attr_name})"
                else:
                    target_str = f"{count} {id_attr_name}"
            print(
                f"{str(plan.start_time):<25} | {plan.variable_name:<20} | "
                f"{type_str:<8} | {val_str:<10} | {dur_str:<10} | {target_str}"
            )

        print("-" * 100)
        if conflicts:
            error_msg = "\n".join(conflicts)
            raise ValueError(f"Parameter Plan Conflicts Detected:\n{error_msg}")


# ---------------------------------------------------------------------------
# ProgressMixin
# ---------------------------------------------------------------------------

class ProgressMixin:
    """Mixin providing lightweight progress-tracking with ETA."""

    def set_total_steps(self: AbstractModel, total: int) -> None:
        """Enable progress tracking with ETA by specifying the total number of steps."""
        self._progress = ProgressTracker()
        self._progress.start(total)

    def progress_tick(self: AbstractModel) -> None:
        """Record one completed step (called automatically by step_advance)."""
        if self._progress is not None:
            self._progress.tick()

    def format_progress(self: AbstractModel) -> str:
        """Return progress string, or empty string if tracking is not enabled."""
        if self._progress is not None:
            return self._progress.format_progress()
        return ""
