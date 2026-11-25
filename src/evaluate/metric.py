"""Thin wrapper around the official competition metric (`topometrics`).

The call and its arguments are copied from the official metric notebook ("Vesuvius 2025 Metric Demo" by Sohier,
https://www.kaggle.com/code/sohier/vesuvius-2025-metric-demo), so local scores match the leaderboard's per-volume
score:  Score = 0.30 * TopoScore + 0.35 * SurfaceDice@2 + 0.35 * VOI_score, GT label 2 ignored.

`topometrics` is not on PyPI. It ships in the Kaggle dataset "vesuvius-metric-resources"
(folder `topological-metrics-kaggle/`, which builds a C++ Betti-matching extension with `make build-betti`).
Install it into the env, or put its `src/` folder on PYTHONPATH / pass it as `topometrics_src`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _import_leaderboard(topometrics_src: str | Path | None = None):
    if topometrics_src is not None and str(topometrics_src) not in sys.path:
        sys.path.append(str(topometrics_src))
    try:
        import topometrics.leaderboard as lb
    except ImportError as err:
        raise ImportError(
            "topometrics is not installed. Get it from the Kaggle dataset 'sohier/vesuvius-metric-resources' "
            "(see src/evaluate/metric.py)."
        ) from err
    return lb


def score_volume(pred: np.ndarray, label: np.ndarray, surface_tolerance: float = 2.0,
                 topometrics_src: str | Path | None = None) -> dict[str, float]:
    """Official per-volume score of a binary prediction against a label volume (0 / 1 / 2 = ignore).

    Returns {"score": ...} plus any numeric fields of the metric's report (e.g. the three sub-scores).
    """
    lb = _import_leaderboard(topometrics_src)
    report = lb.compute_leaderboard_score(
        predictions=pred,
        labels=label,
        dims=(0, 1, 2),
        spacing=(1.0, 1.0, 1.0),
        surface_tolerance=surface_tolerance,
        voi_connectivity=26,
        voi_transform="one_over_one_plus",
        voi_alpha=0.3,
        combine_weights=(0.3, 0.35, 0.35),  # (Topo, SurfaceDice, VOI)
        fg_threshold=None,                  # foreground = pred != 0
        ignore_label=2,
        ignore_mask=None,
    )
    out = {"score": float(np.clip(report.score, 0.0, 1.0))}
    for key, val in getattr(report, "__dict__", {}).items():
        if key != "score" and isinstance(val, (int, float, np.integer, np.floating)):
            out[key] = float(val)
    return out
