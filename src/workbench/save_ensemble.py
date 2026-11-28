"""Average cached probability maps of several models into one ensemble map, saved like a single model's.

Reads PROBS/<model>/<id>.npy (written by src.infer.predict_probs) and writes PROBS/<name>/<id>.npy (float16), so the
ensemble can then be used as one model: `run_variants --models <name>=1 ...`. No prediction, no GPU.

  python -m src.workbench.save_ensemble --models ft192_all=0.6 m160=0.4 --name ens_192x0.6_160x0.4
  python -m src.workbench.save_ensemble --models ft192_all=0.6 m160=0.4 --name ens_192x0.6_160x0.4 --ids 1004283650

Weights are normalised to sum to 1 (load_ensemble, as in rank 2).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from src.common import PROBS, load_ensemble
from src.workbench.run_variants import cached_ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, metavar="NAME=WEIGHT",
                    help="cached models and their weights, e.g. ft192_all=0.6 m160=0.4")
    ap.add_argument("--name", required=True, help="output folder under data/cache/probs/")
    ap.add_argument("--ids", nargs="*", help="default: every volume cached for all models")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    models = {k: float(v) for k, v in (s.split("=", 1) for s in args.models)}
    if args.name in models:
        raise SystemExit(f"--name {args.name} is one of the input models; pick a new name")
    ids = args.ids or cached_ids(models)
    if not ids:
        raise SystemExit(f"No volume is cached for all of {list(models)} in {PROBS}. Run src.infer.predict_probs.")
    missing = [(m, i) for i in ids for m in models if not (PROBS / m / f"{i}.npy").exists()]
    if missing:
        raise SystemExit(f"Missing probability maps (model, id): {missing}")

    out_dir = PROBS / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    total = sum(models.values())
    print(f"{len(ids)} volumes -> {out_dir}  weights: "
          + ", ".join(f"{m}={w / total:g}" for m, w in models.items()), flush=True)
    for vol_id in ids:
        out = out_dir / f"{vol_id}.npy"
        if out.exists() and not args.overwrite:
            print(f"{vol_id}: exists (use --overwrite)")
            continue
        t0 = time.time()
        np.save(out, load_ensemble(vol_id, models).astype(np.float16))
        print(f"{vol_id}: saved in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
