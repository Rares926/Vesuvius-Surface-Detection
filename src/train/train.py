"""Step 5: train an nnU-Net model on the prepared data (nnunet/preprocessed/ -> nnunet/results/).

  python -m src.train.train --trainer nnUNetTrainer_FT_5epochs     # timing test
  python -m src.train.train --trainer nnUNetTrainer_FT             # the real run: rank 2's fold_0 at 192^3, 500 epochs
  python -m src.train.train --trainer nnUNetTrainer_FT --resume    # continue an interrupted run

Run `python -m src.train.prepare` and `python -m src.train.preprocess` first. Trainers are in src/train/trainers/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.train.nnunet_env import DATASET_NAME, PLANS, RANK2_FOLD0, set_nnunet_env


def cmd_train(args) -> None:
    from nnunetv2.run.run_training import run_training

    pretrained = None if args.no_pretrained else str(args.pretrained)
    print(f"dataset {DATASET_NAME}, plans {PLANS}, config {args.config}, fold {args.fold}, trainer {args.trainer}")
    print(f"pretrained: {pretrained}   continue: {args.resume}")
    run_training(DATASET_NAME, args.config, args.fold, trainer_class_name=args.trainer, plans_identifier=PLANS,
                 pretrained_weights=None if args.resume else pretrained, continue_training=args.resume)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="3d_fullres_192")
    ap.add_argument("--fold", default="0", help="0-4, or 'all'")
    ap.add_argument("--trainer", default="nnUNetTrainer_FT")
    ap.add_argument("--pretrained", type=Path, default=RANK2_FOLD0, help="checkpoint to start from")
    ap.add_argument("--no-pretrained", action="store_true", help="train from scratch")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run from its last checkpoint")
    ap.add_argument("--da-workers", type=int, default=16, help="CPU processes for data augmentation")
    args = ap.parse_args()
    set_nnunet_env(args.da_workers)
    cmd_train(args)


if __name__ == "__main__":
    main()
