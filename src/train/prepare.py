"""Step 1: convert the Kaggle training tifs to nnU-Net's raw format (nnunet/raw/), the way rank 2 did.

  python -m src.train.prepare                # all 786 volumes, 16 processes
  python -m src.train.prepare --limit 10     # only the first 10 (for testing)

Images become imagesTr/<id>_0000.nii.gz (_0000 = channel 0), labels labelsTr/<id>.nii.gz (uint8), spacing 1, 1, 1;
voxel values are unchanged. Writes rank 2's dataset.json. Files that already exist are skipped, so it can be rerun.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.common import image_path, label_path, read_volume
from src.train.nnunet_env import DATASET_NAME, RANK2_PLANS_DIR, RAW, set_nnunet_env, train_ids


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=16, help="CPU processes for the conversion")
    ap.add_argument("--limit", type=int, help="only the first N volumes (for testing)")
    args = ap.parse_args()
    set_nnunet_env()
    cmd_prepare(args)


if __name__ == "__main__":
    main()
