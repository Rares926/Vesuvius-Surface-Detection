"""How many voxels of the true label each saved mask hits (a quick voxel check, not the official metric).

Reads data/cache/masks/<model>/<variant>/<id>.tif for every variant folder, and the label from
data/cache/masks/<model>/true_label/<id>.tif (0 = background, 1 = sheet, 2 = not labelled, ignored).

  python -m src.workbench.label_overlap --model mine_198_fold_0

Columns, per variant and volume:
  label_vox   sheet voxels in the label (label == 1)
  hit         mask voxels on a sheet voxel (true positives)
  recall      hit / label_vox: share of the label the mask covers
  false_pos   mask voxels on labelled background (label == 0)
  precision   hit / (hit + false_pos), on the labelled region only
  dice        2 * hit / (mask voxels in the labelled region + label_vox)
  unlabelled  mask voxels where label == 2 (not scored)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import tifffile

from src.common import CACHE

LABEL_DIR = "true_label"


def overlap(mask: np.ndarray, label: np.ndarray) -> dict:
    m = mask > 0
    sheet, bg = label == 1, label == 0
    hit = int((m & sheet).sum())
    fp = int((m & bg).sum())
    n_label = int(sheet.sum())
    return {
        "label_vox": n_label,
        "hit": hit,
        "recall": hit / n_label if n_label else np.nan,
        "false_pos": fp,
        "precision": hit / (hit + fp) if hit + fp else np.nan,
        "dice": 2 * hit / (hit + fp + n_label) if hit + fp + n_label else np.nan,
        "unlabelled": int((m & (label == 2)).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mine_198_fold_0", help="folder under data/cache/masks/")
    ap.add_argument("--out", help="optional CSV path")
    args = ap.parse_args()

    root = CACHE / "masks" / args.model
    labels = {p.stem: p for p in (root / LABEL_DIR).glob("*.tif")}
    if not labels:
        raise SystemExit(f"No labels in {root / LABEL_DIR}")

    rows = []
    for vol_id, label_path in sorted(labels.items()):
        label = tifffile.imread(label_path)
        for variant_dir in sorted(d for d in root.iterdir() if d.is_dir() and d.name != LABEL_DIR):
            mask_path = variant_dir / f"{vol_id}.tif"
            if mask_path.exists():
                rows.append({"id": vol_id, "variant": variant_dir.name, **overlap(tifffile.imread(mask_path), label)})

    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.round(4).to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
