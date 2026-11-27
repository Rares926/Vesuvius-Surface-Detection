"""Shared by all training steps: nnU-Net's folders, the dataset and plans names, and rank 2's files.

The training code is split by the folder each step builds:

  python -m src.train.prepare                      # Kaggle tif -> nnunet/raw/ (.nii.gz), once
  python -m src.train.preprocess                   # nnunet/preprocessed/: rank 2's plans + ours, nnU-Net
                                                   #   preprocessing, the 5-fold split (or --only plan|preprocess|split)
  python -m src.train.check_split                  # does rank 2's fold_0 model fit our fold 0 split?
  python -m src.train.train --trainer nnUNetTrainer_FT_5epochs    # timing test       -> nnunet/results/
  python -m src.train.train --trainer nnUNetTrainer_FT            # the real run (500 epochs)

Credits: rank 2 ("risk of overfitting", Duong Nguyen & Marius Heuser) for the plans, dataset definition and the
pretrained weights (Kaggle model nguyncdngs/nnunet-resenc, MIT). nnU-Net v2 by Isensee et al. (Apache 2.0).
"""
from __future__ import annotations

import os
from pathlib import Path

from src.common import DATA, ROOT

NNUNET = ROOT / "nnunet"                         # nnU-Net's three folders live here (ignored by git)
RAW, PREPROCESSED, RESULTS = NNUNET / "raw", NNUNET / "preprocessed", NNUNET / "results"
TRAINERS = Path(__file__).resolve().parent / "trainers"

DATASET_ID = 501
DATASET_NAME = "Dataset501_Vesuvius320"          # same id and name as rank 2, so their plans fit unchanged
PLANS = "nnUNetResEncUNetMPlans"                 # rank 2's plans identifier (ResEnc M)
RANK2 = ROOT / "models" / "rank_2"               # rank 2's weights: v2 = 128^3, v3 = 160^3
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
