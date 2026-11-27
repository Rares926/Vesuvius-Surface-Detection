"""Steps 2-4: build nnunet/preprocessed/ from nnunet/raw/.

  python -m src.train.preprocess                         # plan, then preprocess, then split
  python -m src.train.preprocess --only plan             # one step on its own (plan | preprocess | split)

- plan:       rank 2's plans, fingerprint and dataset.json, plus our 3d_fullres_<patch> configurations (default 192)
- preprocess: nnU-Net's preprocessing of 3d_fullres (CT normalisation with rank 2's statistics), shared by all
              patch sizes
- split:      nnU-Net's default 5-fold split, written to splits_final.json; we use fold 0 (80/20)
"""
from __future__ import annotations

import argparse
import json
import shutil
import time

import numpy as np

from src.common import DATA
from src.train.nnunet_env import (DATASET_ID, DATASET_NAME, PLANS, PREPROCESSED, RANK2_PLANS_DIR, RAW, SPLITS_DIR,
                                  set_nnunet_env, train_ids)


# ---------------------------------------------------------------------------------------------------------------
# plan: rank 2's plans + our configurations
# ---------------------------------------------------------------------------------------------------------------
def cmd_plan(args) -> None:
    out = PREPROCESSED / DATASET_NAME
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(RAW / DATASET_NAME / "dataset.json", out / "dataset.json")
    shutil.copy(RANK2_PLANS_DIR / "dataset_fingerprint.json", out / "dataset_fingerprint.json")

    plans = json.loads((RANK2_PLANS_DIR / "plans.json").read_text())
    for p in args.patch:
        # inherits everything from 3d_fullres (network, normalisation, preprocessed data); only the patch changes
        plans["configurations"][f"3d_fullres_{p}"] = {
            "inherits_from": "3d_fullres",
            "patch_size": [p, p, p],
            "batch_size": 2,
        }
    (out / f"{PLANS}.json").write_text(json.dumps(plans, indent=4))

    base = plans["configurations"]["3d_fullres"]
    arch = base["architecture"]["arch_kwargs"]
    print(f"plans: {out / (PLANS + '.json')}")
    print(f"network: {base['architecture']['network_class_name'].split('.')[-1]}, "
          f"blocks/stage {arch['n_blocks_per_stage']}, features {arch['features_per_stage']}")
    for name, cfg in plans["configurations"].items():
        if name.startswith("3d_fullres"):
            print(f"  {name}: patch {cfg.get('patch_size', base['patch_size'])}, "
                  f"batch {cfg.get('batch_size', base['batch_size'])}")


# ---------------------------------------------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------------------------------------------
def cmd_preprocess(args) -> None:
    from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess

    t0 = time.time()
    preprocess([DATASET_ID], plans_identifier=PLANS, configurations=("3d_fullres",),
               num_processes=(args.workers,))
    plans = json.loads((PREPROCESSED / DATASET_NAME / f"{PLANS}.json").read_text())
    folder = PREPROCESSED / DATASET_NAME / plans["configurations"]["3d_fullres"]["data_identifier"]
    n_cases = len(list(folder.glob("*.pkl")))        # one .pkl (+ image .b2nd + _seg.b2nd) per case
    print(f"preprocessed cases: {n_cases} in {folder}  ({time.time() - t0:.0f} s)")


# ---------------------------------------------------------------------------------------------------------------
# split: nnU-Net's default 5-fold split, written out explicitly
# ---------------------------------------------------------------------------------------------------------------
def default_splits(ids: list[str]) -> list[dict]:
    """Exactly what nnUNetTrainer.do_split creates when no splits_final.json exists: sorted case ids,
    sklearn KFold(5, shuffle=True, random_state=12345)."""
    from nnunetv2.utilities.crossval_split import generate_crossval_split
    return generate_crossval_split(list(np.sort(ids)), seed=12345, n_splits=5)


def cmd_split(args) -> None:
    import pandas as pd

    ids = train_ids()[: args.limit] if args.limit else train_ids()
    splits = default_splits(ids)
    splits_file = PREPROCESSED / DATASET_NAME / "splits_final.json"
    splits_file.parent.mkdir(parents=True, exist_ok=True)
    splits_file.write_text(json.dumps(splits, indent=2))
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val"):
        (SPLITS_DIR / f"fold0_{name}.txt").write_text("\n".join(splits[0][name]) + "\n")

    df = pd.read_csv(DATA / "train.csv", dtype={"id": str, "scroll_id": str}).set_index("id")
    tr, va = splits[0]["train"], splits[0]["val"]
    print(f"{splits_file}\nfold 0: {len(tr)} train, {len(va)} val  (ids also in {SPLITS_DIR})")
    counts = pd.DataFrame({"train": df.loc[tr, "scroll_id"].value_counts(),
                           "val": df.loc[va, "scroll_id"].value_counts()}).fillna(0).astype(int)
    print("volumes per scroll:\n" + counts.to_string())


# ---------------------------------------------------------------------------------------------------------------
STEPS = {"plan": cmd_plan, "preprocess": cmd_preprocess, "split": cmd_split}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(STEPS), help="run one step instead of all three")
    ap.add_argument("--patch", type=int, nargs="+", default=[192], help="plan: patch sizes of our configurations")
    ap.add_argument("--workers", type=int, default=16, help="preprocess: CPU processes")
    ap.add_argument("--limit", type=int, help="split: only the first N volumes (for testing)")
    args = ap.parse_args()
    set_nnunet_env()
    for name in ([args.only] if args.only else STEPS):
        print(f"--- {name}")
        STEPS[name](args)


if __name__ == "__main__":
    main()
