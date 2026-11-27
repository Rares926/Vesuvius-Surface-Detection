"""Check that our fold 0 is the split rank 2's fold_0 model was trained on (needs a GPU).

  python -m src.train.check_split               # 25 validation + 25 training volumes
  python -m src.train.check_split --n 10

Paired test. Rank 2's fold_all model saw every volume; their fold_0 model saw only its own 80 %. On the same volume,
fold_all - fold_0 is therefore ~0 where fold_0 trained, and positive where it didn't. If our fold 0 is their fold 0:
~0 on our train volumes, clearly positive on our val volumes. (Comparing fold_0's raw Dice on train vs val is too
noisy: Dice varies a lot between volumes.) Reads the fold-0 ids written by `python -m src.train.preprocess`.
"""
from __future__ import annotations

import argparse

import numpy as np

from src.common import image_path, label_path, read_volume
from src.train.nnunet_env import RANK2_PLANS_DIR, SPLITS_DIR, set_nnunet_env


def _dice(prob: np.ndarray, label: np.ndarray) -> float:
    keep = label != 2
    pred = (prob >= 0.5) & keep
    gt = (label == 1) & keep
    denom = pred.sum() + gt.sum()
    return float(2 * (pred & gt).sum() / denom) if denom else 1.0


def cmd_check_split(args) -> None:
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=25, help="volumes per group")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    set_nnunet_env()
    cmd_check_split(args)


if __name__ == "__main__":
    main()
