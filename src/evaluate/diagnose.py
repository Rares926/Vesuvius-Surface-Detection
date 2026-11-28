"""Measure, per volume, the errors of a predicted mask that the competition metric punishes.

The metric sets label-2 voxels (unlabelled) to background in BOTH the prediction and the label, then scores
  Surface Dice (tolerance 2 voxels), VOI (merges / splits of 26-connected pieces) and TopoScore (Betti numbers:
  k=0 pieces, k=1 holes / tunnels, k=2 cavities).
This script applies the same ignore rule and counts each kind of error directly, so we can see which ones a
post-processing step has to fix. It does not compute the official score (that needs `topometrics`, see metric.py).

Betti numbers: b0 = pieces (26-connectivity), b2 = cavities (enclosed background, 6-connectivity), and
b1 = b0 + b2 - Euler number, since in 3D the Euler number is b0 - b1 + b2.

Usage (from the project root):
  python -m src.evaluate.diagnose --pred data/cache/masks/mine_198_fold_0/threshold_0.5 \
                                  --label data/cache/masks/mine_198_fold_0/true_label
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.measure import euler_number

S26 = np.ones((3, 3, 3), bool)
SPECK = 1000          # pieces smaller than this (voxels) count as specks
MIN_OVERLAP = 500     # voxels of overlap for a piece to "touch" a label sheet in the merge / split checks
TOL = 2.0             # Surface Dice tolerance, in voxels
BAND = 3              # border band width, in voxels


def betti(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(b0 pieces, b1 holes/tunnels, b2 cavities, euler) of a binary volume."""
    _, b0 = ndi.label(mask, S26)
    _, b2 = ndi.label(ndi.binary_fill_holes(mask) & ~mask)       # default structure = 6-connectivity
    chi = int(euler_number(mask, connectivity=3))
    return b0, b0 + b2 - chi, b2, chi


def overlaps(a_lab: np.ndarray, b_lab: np.ndarray) -> int:
    """Number of pieces of `a_lab` that overlap >= 2 pieces of `b_lab` by >= MIN_OVERLAP voxels each."""
    n = 0
    for i, sl in enumerate(ndi.find_objects(a_lab), 1):
        if sl is None:
            continue
        ids, cnt = np.unique(b_lab[sl][a_lab[sl] == i], return_counts=True)
        n += int(((ids > 0) & (cnt >= MIN_OVERLAP)).sum() >= 2)
    return n


def diagnose(pred_raw: np.ndarray, label: np.ndarray) -> dict:
    ignore = label == 2
    gt = label == 1
    pred = (pred_raw > 0) & ~ignore                                 # what the metric sees

    gb0, gb1, gb2, _ = betti(gt)
    pb0, pb1, pb2, _ = betti(pred)
    p_lab, _ = ndi.label(pred, S26)
    g_lab, _ = ndi.label(gt, S26)
    sizes = np.bincount(p_lab.ravel())[1:]

    # surface-Dice-style errors: voxels farther than TOL from the other mask
    d_to_gt = ndi.distance_transform_edt(~gt) if gt.any() else np.full(gt.shape, np.inf)
    d_to_pred = ndi.distance_transform_edt(~pred) if pred.any() else np.full(gt.shape, np.inf)
    far_fp = int((pred & (d_to_gt > TOL)).sum())
    missed = int((gt & (d_to_pred > TOL)).sum())

    band = np.ones_like(gt)
    band[BAND:-BAND, BAND:-BAND, BAND:-BAND] = False
    return {
        "labelled_%": round(100 * (1 - ignore.mean()), 1),
        "gt_pieces": gb0, "gt_holes": gb1, "gt_cavities": gb2,
        "pred_pieces": pb0, "pred_specks": int((sizes < SPECK).sum()),
        "pred_holes": pb1, "pred_cavities": pb2,
        "merges": overlaps(p_lab, g_lab),      # predicted pieces spanning >= 2 label sheets
        "splits": overlaps(g_lab, p_lab),      # label sheets broken into >= 2 predicted pieces
        "gt_voxels": int(gt.sum()), "pred_voxels": int(pred.sum()),
        "far_fp_%": round(100 * far_fp / max(pred.sum(), 1), 1),
        "missed_%": round(100 * missed / max(gt.sum(), 1), 1),
        "band_pred_voxels": int((pred & band).sum()), "band_gt_voxels": int((gt & band).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pred", required=True, help="folder of predicted <id>.tif masks")
    ap.add_argument("--label", required=True, help="folder of label <id>.tif volumes (0 / 1 / 2)")
    ap.add_argument("--out", help="CSV path; default <pred folder>/../diagnose_<pred folder name>.csv")
    args = ap.parse_args()

    pred_dir, label_dir = Path(args.pred), Path(args.label)
    rows = []
    for path in sorted(pred_dir.glob("*.tif")):
        lab_path = label_dir / path.name
        if not lab_path.exists():
            print(f"skip {path.stem}: no label")
            continue
        row = {"id": path.stem, **diagnose(tifffile.imread(path), tifffile.imread(lab_path))}
        print(row)
        rows.append(row)
    df = pd.DataFrame(rows)
    out = Path(args.out) if args.out else pred_dir.parent / f"diagnose_{pred_dir.name}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
