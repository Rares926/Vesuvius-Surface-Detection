"""Run post-processing variants on cached probability maps and score them with the official metric.

Reads PROBS/<model>/<id>.npy (written by src.infer.predict_probs), so it needs no GPU.
Writes one row per (volume, variant) to a CSV and prints the mean per variant.

Example (from the project root):
  python -m src.workbench.run_variants --models m160=0.6 m128=0.4 --variants rank2_no_interp rank2_full
  python -m src.workbench.run_variants --models m160=1 --variants rank2_full --no-metric --save-masks

Note: the rank-2 weights were trained on all training volumes (fold "all"), so scores on training volumes are
optimistic. Use them to compare variants with each other, and confirm real gains with a Kaggle submission.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from src.common import CACHE, PROBS, label_path, load_ensemble, read_volume
from src.workbench.variants import VARIANTS


def cached_ids(models: dict[str, float]) -> list[str]:
    """Volumes that have a cached probability map for every model."""
    sets = [{p.stem for p in (PROBS / m).glob("*.npy")} for m in models]
    return sorted(set.intersection(*sets)) if sets else []


def model_tag(models: dict[str, float]) -> str:
    """Folder name for the masks: the model name, or e.g. m160x0.6+m128x0.4 for an ensemble."""
    if len(models) == 1:
        return next(iter(models))
    return "+".join(f"{m}x{w:g}" for m, w in models.items())


def process_volume(vol_id: str, models: dict[str, float], variants: list[str], no_metric: bool,
                   topometrics_src: str | None, save_masks: bool) -> list[dict]:
    """All variants of one volume: one row per variant. Top-level so it can run in a worker process."""
    if not no_metric:
        from src.evaluate.metric import score_volume

    prob = load_ensemble(vol_id, models)
    label = None if no_metric else read_volume(label_path(vol_id))
    rows = []
    for name in variants:
        t0 = time.time()
        mask = VARIANTS[name](prob)
        row = {"id": vol_id, "variant": name, "seconds": round(time.time() - t0, 1),
               "fg_fraction": float(mask.mean())}
        if label is not None:
            row.update(score_volume(mask, label, topometrics_src=topometrics_src))
        if save_masks:
            path = CACHE / "masks" / model_tag(models) / name / f"{vol_id}.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(path, mask.astype(np.uint8))
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, metavar="NAME=WEIGHT",
                    help="cached models and their ensemble weights, e.g. m160=0.6 m128=0.4")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--ids", nargs="*", help="default: every volume cached for all models")
    ap.add_argument("--no-metric", action="store_true", help="only run the variants (timing / saving masks)")
    ap.add_argument("--topometrics-src", help="path to topological-metrics-kaggle/src if not installed")
    ap.add_argument("--save-masks", action="store_true", help="write masks to data/cache/masks/<model>/<variant>/<id>.tif")
    ap.add_argument("--out", help="CSV path; default data/cache/results/<timestamp>.csv")
    ap.add_argument("--workers", type=int, default=1,
                    help="volumes processed in parallel (the metric takes ~48 s per volume on one CPU core)")
    args = ap.parse_args()

    models = {k: float(v) for k, v in (s.split("=", 1) for s in args.models)}
    ids = args.ids or cached_ids(models)
    if not ids:
        raise SystemExit(f"No cached probability maps for {list(models)} in {PROBS}. Run src.infer.predict_probs.")

    out_csv = Path(args.out) if args.out else CACHE / "results" / f"{datetime.now():%Y%m%d_%H%M%S}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    job = dict(models=models, variants=args.variants, no_metric=args.no_metric,
               topometrics_src=args.topometrics_src, save_masks=args.save_masks)
    rows = []

    def collect(vol_rows: list[dict]) -> None:
        rows.extend(vol_rows)
        for row in vol_rows:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)   # rewrite after every volume: a crash keeps partial results

    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            for fut in as_completed([pool.submit(process_volume, v, **job) for v in ids]):
                collect(fut.result())
    else:
        for vol_id in ids:
            collect(process_volume(vol_id, **job))

    df = pd.DataFrame(rows)
    print(f"\nSaved {out_csv}\n")
    print(df.drop(columns=["id"]).groupby("variant").mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
