"""Train nnU-Net models for the Vesuvius surface task, starting from rank 2's setup.

A thin wrapper around nnU-Net v2: it prepares the data the way rank 2 did, adds our own configurations and trainers,
and calls nnU-Net's own planning, preprocessing and training code. Run on the VM from the project root, in the
`vesuvius` env, one step at a time:

  python -m src.train.train prepare                 # tif -> nnU-Net raw format (.nii.gz), once
  python -m src.train.train plan                    # rank 2's M plans + our 3d_fullres_192 configuration
  python -m src.train.train preprocess              # nnU-Net preprocessing, once
  python -m src.train.train split                   # nnU-Net's default 5-fold split; we use fold 0 (80/20)
  python -m src.train.train check-split             # does rank 2's fold_0 model fit our fold 0 split?
  python -m src.train.train train --trainer nnUNetTrainer_FT_5epochs     # timing test
  python -m src.train.train train --trainer nnUNetTrainer_FT             # the real run (500 epochs)

Credits: rank 2 ("risk of overfitting", Duong Nguyen & Marius Heuser) for the plans, dataset definition and the
pretrained weights (Kaggle model nguyncdngs/nnunet-resenc, MIT). nnU-Net v2 by Isensee et al. (Apache 2.0).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.common import DATA, ROOT, image_path, label_path, read_volume

# ---------------------------------------------------------------------------------------------------------------
# Paths and names
# ---------------------------------------------------------------------------------------------------------------
NNUNET = ROOT / "nnunet"                         # nnU-Net's three folders live here (ignored by git)
RAW, PREPROCESSED, RESULTS = NNUNET / "raw", NNUNET / "preprocessed", NNUNET / "results"
TRAINERS = Path(__file__).resolve().parent / "trainers"

DATASET_ID = 501
DATASET_NAME = "Dataset501_Vesuvius320"          # same id and name as rank 2, so their plans fit unchanged
PLANS = "nnUNetResEncUNetMPlans"                 # rank 2's plans identifier (ResEnc M)
RANK2 = ROOT / "models"                          # rank 2's weights: v2 = 128^3, v3 = 160^3
RANK2_PLANS_DIR = RANK2 / "v2" / "nnUNet_ResEnc"
RANK2_FOLD0 = RANK2_PLANS_DIR / "fold_0" / "checkpoint_final.pth"
SPLITS_DIR = DATA / "cache" / "splits"


def set_nnunet_env(da_workers: int | None = None) -> None:
    """nnU-Net reads its folders from environment variables; set them here so nobody has to export them."""
    os.environ["nnUNet_raw"] = str(RAW)
    os.environ["nnUNet_preprocessed"] = str(PREPROCESSED)
    os.environ["nnUNet_results"] = str(RESULTS)
    os.environ["nnUNet_extTrainer"] = str(TRAINERS)
    if da_workers:
        os.environ["nnUNet_n_proc_DA"] = str(da_workers)   # CPU processes for data augmentation


def train_ids() -> list[str]:
    return sorted(p.stem for p in (DATA / "train_images").glob("*.tif"))


# ---------------------------------------------------------------------------------------------------------------
# prepare: tif -> nnU-Net raw format
# ---------------------------------------------------------------------------------------------------------------
def _write_nifti(arr: np.ndarray, path: Path) -> None:
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(arr)            # array (z, y, x) -> image; nnU-Net reads it back as (z, y, x)
    img.SetSpacing((1.0, 1.0, 1.0))
    tmp = path.with_name(path.name + ".tmp.nii.gz")
    sitk.WriteImage(img, str(tmp), useCompression=True)
    tmp.replace(path)                            # only complete files get the final name, so reruns can resume


def _convert_one(vol_id: str) -> str:
    img_out = RAW / DATASET_NAME / "imagesTr" / f"{vol_id}_0000.nii.gz"
    lab_out = RAW / DATASET_NAME / "labelsTr" / f"{vol_id}.nii.gz"
    if not img_out.exists():
        _write_nifti(read_volume(image_path(vol_id)), img_out)
    if not lab_out.exists():
        _write_nifti(read_volume(label_path(vol_id)).astype(np.uint8), lab_out)
    return vol_id


def cmd_prepare(args) -> None:
    ids = train_ids()[: args.limit] if args.limit else train_ids()
    ds = RAW / DATASET_NAME
    (ds / "imagesTr").mkdir(parents=True, exist_ok=True)
    (ds / "labelsTr").mkdir(parents=True, exist_ok=True)

    dataset_json = json.loads((RANK2_PLANS_DIR / "dataset.json").read_text())   # rank 2's: CT, labels 0/1/ignore 2
    dataset_json["numTraining"] = len(ids)
    (ds / "dataset.json").write_text(json.dumps(dataset_json, indent=4))

    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(_convert_one, ids), 1):
            if i % 50 == 0 or i == len(ids):
                print(f"  {i}/{len(ids)} converted ({time.time() - t0:.0f} s)")

    n_img = len(list((ds / "imagesTr").glob("*_0000.nii.gz")))
    n_lab = len(list((ds / "labelsTr").glob("*.nii.gz")))
    print(f"imagesTr: {n_img}   labelsTr: {n_lab}   (expected {len(ids)} each)")

    # check one volume: the converted file must equal the tif
    import SimpleITK as sitk
    vid = ids[0]
    back = sitk.GetArrayFromImage(sitk.ReadImage(str(ds / "labelsTr" / f"{vid}.nii.gz")))
    orig = read_volume(label_path(vid))
    print(f"check {vid}: shape {back.shape} == {orig.shape}, identical: {np.array_equal(back, orig)}, "
          f"labels {np.unique(back).tolist()}")


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
# check-split: does rank 2's fold_0 model fit our split?
# ---------------------------------------------------------------------------------------------------------------
def _dice(prob: np.ndarray, label: np.ndarray) -> float:
    keep = label != 2
    pred = (prob >= 0.5) & keep
    gt = (label == 1) & keep
    denom = pred.sum() + gt.sum()
    return float(2 * (pred & gt).sum() / denom) if denom else 1.0


def cmd_check_split(args) -> None:
    """Paired test. Rank 2's fold_all model saw every volume; their fold_0 model saw only its own 80 %.
    On the same volume, fold_all - fold_0 is therefore ~0 where fold_0 trained, and positive where it didn't.
    If our fold 0 is their fold 0: ~0 on our train volumes, clearly positive on our val volumes.
    (Comparing fold_0's raw Dice on train vs val is too noisy: Dice varies a lot between volumes.)"""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from src.infer.predict_probs import predict_foreground

    val = (SPLITS_DIR / "fold0_val.txt").read_text().split()
    tr = (SPLITS_DIR / "fold0_train.txt").read_text().split()
    rng = np.random.default_rng(args.seed)
    picks = [("val", v) for v in rng.choice(val, args.n, replace=False)] + \
            [("train", v) for v in rng.choice(tr, args.n, replace=False)]

    predictors = {}
    for fold in (0, "all"):
        p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                            perform_everything_on_device=True, device=torch.device("cuda"),
                            verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        p.initialize_from_trained_model_folder(str(RANK2_PLANS_DIR), use_folds=(fold,),
                                               checkpoint_name="checkpoint_final.pth")
        predictors[fold] = p

    diffs = {"val": [], "train": []}
    for group, vid in picks:
        label = read_volume(label_path(vid))
        d = {f: _dice(predict_foreground(p, image_path(vid)).astype(np.float32), label) for f, p in predictors.items()}
        diffs[group].append(d["all"] - d[0])
        print(f"  {group:5s} {vid}: fold_0 {d[0]:.4f}  fold_all {d['all']:.4f}  diff {d['all'] - d[0]:+.4f}")

    print(f"\nDice at 0.5 (label 2 ignored), {args.n} volumes per group, fold_all - fold_0:")
    for group, vals in diffs.items():
        v = np.array(vals)
        print(f"  our fold-0 {group:5s}: mean {v.mean():+.4f}  median {np.median(v):+.4f}  "
              f"fold_all better in {(v > 0).sum()}/{len(v)}")
    print("Split matches rank 2's if: val clearly positive (fold_all better on nearly all), train around 0.")


# ---------------------------------------------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------------------------------------------
def cmd_train(args) -> None:
    from nnunetv2.run.run_training import run_training

    pretrained = None if args.no_pretrained else str(args.pretrained)
    print(f"dataset {DATASET_NAME}, plans {PLANS}, config {args.config}, fold {args.fold}, trainer {args.trainer}")
    print(f"pretrained: {pretrained}   continue: {args.resume}")
    run_training(DATASET_NAME, args.config, args.fold, trainer_class_name=args.trainer, plans_identifier=PLANS,
                 pretrained_weights=None if args.resume else pretrained, continue_training=args.resume)


# ---------------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--da-workers", type=int, default=16, help="CPU processes for data augmentation in training")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="convert the training tifs to nnU-Net's raw format")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, help="only the first N volumes (for testing)")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("plan", help="write rank 2's plans plus our 3d_fullres_<patch> configurations")
    p.add_argument("--patch", type=int, nargs="+", default=[192])
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("preprocess", help="nnU-Net preprocessing of 3d_fullres (shared by all patch sizes)")
    p.add_argument("--workers", type=int, default=16)
    p.set_defaults(func=cmd_preprocess)

    p = sub.add_parser("split", help="write nnU-Net's default 5-fold split to splits_final.json")
    p.add_argument("--limit", type=int, help="only the first N volumes (for testing)")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("check-split", help="paired test: rank 2's fold_all vs fold_0 on our fold-0 val and train")
    p.add_argument("--n", type=int, default=25, help="volumes per group")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_check_split)

    p = sub.add_parser("train", help="run nnU-Net training")
    p.add_argument("--config", default="3d_fullres_192")
    p.add_argument("--fold", default="0", help="0-4, or 'all'")
    p.add_argument("--trainer", default="nnUNetTrainer_FT")
    p.add_argument("--pretrained", type=Path, default=RANK2_FOLD0, help="checkpoint to start from")
    p.add_argument("--no-pretrained", action="store_true", help="train from scratch")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run from its last checkpoint")
    p.set_defaults(func=cmd_train)

    args = ap.parse_args()
    set_nnunet_env(args.da_workers)
    args.func(args)


if __name__ == "__main__":
    main()
