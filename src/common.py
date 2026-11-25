"""Paths and small helpers shared by the inference, post-processing and evaluation scripts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "cache"          # probability maps and masks we produce (not committed)
PROBS = CACHE / "probs"         # PROBS/<model_name>/<id>.npy, float16 foreground probability


def image_path(vol_id: str, split: str = "train") -> Path:
    return DATA / f"{split}_images" / f"{vol_id}.tif"


def label_path(vol_id: str) -> Path:
    return DATA / "train_labels" / f"{vol_id}.tif"


def read_volume(path: Path) -> np.ndarray:
    return tifffile.imread(path)


def train_table() -> pd.DataFrame:
    return pd.read_csv(DATA / "train.csv", dtype={"id": str, "scroll_id": str})


def pick_ids(per_scroll: int = 2, seed: int = 0) -> list[str]:
    """A small evaluation set: `per_scroll` random volumes from every scroll, so each scroll is represented."""
    df = train_table()
    picked = (
        df.groupby("scroll_id", group_keys=False)
        .apply(lambda g: g.sample(min(per_scroll, len(g)), random_state=seed))
    )
    return sorted(picked["id"].tolist())


def prob_path(model_name: str, vol_id: str) -> Path:
    return PROBS / model_name / f"{vol_id}.npy"


def load_ensemble(vol_id: str, models: dict[str, float]) -> np.ndarray:
    """Weighted average of cached foreground probabilities, weights normalised to sum to 1 (as in rank 2)."""
    total = sum(models.values())
    prob = None
    for name, w in models.items():
        p = np.load(prob_path(name, vol_id)).astype(np.float32) * (w / total)
        prob = p if prob is None else prob + p
    return prob
