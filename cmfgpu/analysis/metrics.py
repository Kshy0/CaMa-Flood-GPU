# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Streamflow comparison metrics (NSE, KGE, PBias) operating on 1-D series."""
from __future__ import annotations

from typing import Tuple

import numpy as np


_MIN_VALID = 10


def _aligned_pair(pred: np.ndarray, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred)
    obs = np.asarray(obs)
    mask = np.isfinite(pred) & np.isfinite(obs)
    return pred[mask], obs[mask]


def nse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency (``pred`` first to match HydroNet convention)."""
    p, o = _aligned_pair(pred, obs)
    if p.size < _MIN_VALID:
        return float("nan")
    ss_res = float(((p - o) ** 2).sum())
    ss_tot = float(((o - o.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-10)


def kge(pred: np.ndarray, obs: np.ndarray) -> float:
    """Kling-Gupta Efficiency (2009)."""
    p, o = _aligned_pair(pred, obs)
    if p.size < _MIN_VALID:
        return float("nan")
    if p.std() == 0 or o.std() == 0:
        return float("nan")
    r = float(np.corrcoef(p, o)[0, 1])
    a = float(p.std() / o.std())
    b = float(p.mean() / o.mean()) if o.mean() > 0 else 0.0
    return 1.0 - float(np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2))


def pbias(pred: np.ndarray, obs: np.ndarray) -> float:
    """Percent bias (``100 * sum(pred - obs) / sum(obs)``)."""
    p, o = _aligned_pair(pred, obs)
    if p.size < _MIN_VALID:
        return float("nan")
    s = float(o.sum())
    if abs(s) < 1e-10:
        return float("nan")
    return 100.0 * float((p - o).sum()) / s


def compute_per_gauge_metrics(
    pred: np.ndarray,
    obs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(NSE, KGE, PBias)`` arrays computed column-wise.

    Parameters
    ----------
    pred, obs : ndarray of shape ``(T, N)``
        Aligned simulation and observation columns.  NaN entries are
        ignored per column.
    """
    pred = np.asarray(pred)
    obs = np.asarray(obs)
    if pred.shape != obs.shape:
        raise ValueError(
            f"pred shape {pred.shape} != obs shape {obs.shape}"
        )
    n_gauges = pred.shape[1]
    out = np.full((3, n_gauges), np.nan, dtype=np.float64)
    for i in range(n_gauges):
        out[0, i] = nse(pred[:, i], obs[:, i])
        out[1, i] = kge(pred[:, i], obs[:, i])
        out[2, i] = pbias(pred[:, i], obs[:, i])
    return out[0], out[1], out[2]
