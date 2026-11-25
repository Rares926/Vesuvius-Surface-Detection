"""Run nnU-Net models on volumes and cache the foreground probability map of each (model, volume).

Inference is the only GPU step, so it runs once; every post-processing experiment then reads the cache on CPU.
Each model is cached separately, so the ensemble weights can be changed later without re-running inference.

Predictor settings follow the rank-2 final notebook: tile step 0.5, Gaussian weighting, mirroring TTA,
spacing (1, 1, 1), fold "all", `checkpoint_final.pth`.

Rank-2 weights (Kaggle model nguyncdngs/nnunet-resenc, MIT licence, by Duong Nguyen):
  version 3 -> the model the notebook weights 0.6 (the 160^3 model in the write-up)
  version 2 -> the model the notebook weights 0.4 (the 128^3 model)
Each version unpacks to a folder `nnUNet_ResEnc/` with `dataset.json`, `plans.json` and `fold_all/`.

Example (from the project root):
  python -m src.infer.predict_probs --model m160=weights/v3/nnUNet_ResEnc --model m128=weights/v2/nnUNet_ResEnc \
      --per-scroll 2
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from src.common import image_path, pick_ids, prob_path


def build_predictor(model_dir: str, device: torch.device, step: float, mirroring: bool, on_device: bool,
                    checkpoint: str):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=step,
        use_gaussian=True,
        use_mirroring=mirroring,
        perform_everything_on_device=on_device,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(model_dir, use_folds=("all",), checkpoint_name=checkpoint)
    return predictor


def predict_foreground(predictor, path) -> np.ndarray:
    """Return the foreground (class 1) probability map, shape (D, H, W), float16."""
    from nnunetv2.imageio.tif_reader_writer import Tiff3DIO

    image, _ = Tiff3DIO().read_images([str(path)])
    _, probs = predictor.predict_single_npy_array(
        image, {"spacing": [1.0, 1.0, 1.0]}, None, None, True
    )
    return probs[1].astype(np.float16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=DIR",
                    help="model name and nnU-Net trained-model folder; repeat for several models")
    ap.add_argument("--checkpoint", default="checkpoint_final.pth")
    ap.add_argument("--ids", nargs="*", help="volume ids; default: --per-scroll volumes from each scroll")
    ap.add_argument("--per-scroll", type=int, default=2)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--step", type=float, default=0.5, help="sliding-window tile step (rank 2: 0.5)")
    ap.add_argument("--no-mirroring", action="store_true", help="disable mirroring TTA (8x faster, less accurate)")
    ap.add_argument("--on-device", action="store_true",
                    help="keep the whole volume on the GPU (faster; needs more VRAM than a 4 GB card has)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ids = args.ids or pick_ids(args.per_scroll)
    device = torch.device(args.device)
    print(f"{len(ids)} volumes on {device}: {ids}")

    for spec in args.model:
        name, model_dir = spec.split("=", 1)
        predictor = build_predictor(model_dir, device, args.step, not args.no_mirroring, args.on_device,
                                    args.checkpoint)
        for vol_id in ids:
            out = prob_path(name, vol_id)
            if out.exists() and not args.overwrite:
                print(f"[{name}] {vol_id}: cached")
                continue
            t0 = time.time()
            prob = predict_foreground(predictor, image_path(vol_id, args.split))
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, prob)
            print(f"[{name}] {vol_id}: {prob.shape} in {time.time() - t0:.0f}s")
        del predictor
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
