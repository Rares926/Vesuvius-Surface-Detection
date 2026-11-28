"""Statistics of the training labels themselves (no predictions): one row per volume, plus a per-scroll summary.

  python -m src.workbench.label_stats --workers 3
  python -m src.workbench.label_stats --limit 2          # quick test
  python -m src.workbench.label_stats --json-only        # rebuild the per-scroll CSV and the JSON from the CSV

Writes data/cache/results/label_stats.csv (rewritten after every volume, so a crash keeps what is done) and, at the
end, data/cache/results/label_stats_per_scroll.csv. Volumes already in the CSV are skipped, so a stopped run resumes.

Per volume (label: 0 = background, 1 = sheet, 2 = not labelled; a "sheet" = a 26-connected component of label 1):
  frac_bg / frac_sheet / frac_unlabelled   share of the volume with label 0 / 1 / 2
  sheet_vox                                voxels with label 1
  n_sheets, n_small_sheets                 sheets, and sheets under 1000 voxels (rank 2's min_size)
  min_sheet_vox, max_sheet_vox             smallest / largest sheet
  smallest_5                               the five smallest sheet sizes
  largest_share                            largest sheet / sheet_vox
  thickness                                mean sheet thickness ~ 2 * volume / surface (surface = voxel faces)
  euler                                    Euler number of all sheets (26-connectivity): components - tunnels + cavities
  near2_vox, near4_vox                     voxels within 2 / 4 voxels (chessboard) of two different sheets: merge risk
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.measure import euler_number

from src.common import CACHE, DATA

OUT = CACHE / "results" / "label_stats.csv"
OUT_SCROLL = CACHE / "results" / "label_stats_per_scroll.csv"
OUT_JSON = CACHE / "results" / "label_stats.json"           # both, grouped by scroll
SPLITS = DATA.parent / "nnunet" / "preprocessed" / "Dataset501_Vesuvius320" / "splits_final.json"
STRUCT26 = np.ones((3, 3, 3), bool)


def _faces(m: np.ndarray) -> int:
    """Voxel faces between sheet and non-sheet (including the volume border)."""
    n = 0
    for ax in range(3):
        pad = np.pad(m, [(1, 1) if a == ax else (0, 0) for a in range(3)])
        n += int(np.count_nonzero(np.diff(pad.view(np.int8), axis=ax)))
    return n


def _near_two(lab: np.ndarray, r: int) -> int:
    """Voxels whose (2r+1)^3 neighbourhood holds two different sheets."""
    hi = ndi.maximum_filter(lab, size=2 * r + 1)
    big = np.iinfo(lab.dtype).max
    lo = ndi.minimum_filter(np.where(lab > 0, lab, big), size=2 * r + 1)
    return int(np.count_nonzero((hi > 0) & (lo < big) & (hi != lo)))


def volume_stats(vol_id: str) -> dict:
    label = tifffile.imread(DATA / "train_labels" / f"{vol_id}.tif")
    sheet = label == 1
    n = label.size
    row = {
        "id": vol_id,
        "frac_bg": float((label == 0).sum() / n),
        "frac_sheet": float(sheet.sum() / n),
        "frac_unlabelled": float((label == 2).sum() / n),
        "sheet_vox": int(sheet.sum()),
    }
    lab, k = ndi.label(sheet, STRUCT26)
    sizes = np.sort(np.bincount(lab.ravel())[1:]) if k else np.array([], int)
    row.update(
        n_sheets=int(k),
        n_small_sheets=int((sizes < 1000).sum()),
        min_sheet_vox=int(sizes[0]) if k else 0,
        max_sheet_vox=int(sizes[-1]) if k else 0,
        smallest_5=" ".join(map(str, sizes[:5].tolist())),
        largest_share=float(sizes[-1] / sizes.sum()) if k else np.nan,
        thickness=float(2 * sheet.sum() / _faces(sheet)) if k else np.nan,
        euler=int(euler_number(sheet, connectivity=3)) if k else 0,
    )
    lab = lab.astype(np.int16) if k < 32000 else lab
    row.update(near2_vox=_near_two(lab, 2), near4_vox=_near_two(lab, 4))
    return row


def per_scroll(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("scroll_id")
    return pd.DataFrame({
        "volumes": g.size(),
        "min_sheet_vox": g.min_sheet_vox.min(),
        "min_sheet_vox_median": g.min_sheet_vox.median(),
        "n_sheets_mean": g.n_sheets.mean().round(2),
        "n_small_sheets_total": g.n_small_sheets.sum(),
        "frac_sheet_mean": g.frac_sheet.mean().round(4),
        "frac_unlabelled_mean": g.frac_unlabelled.mean().round(4),
        "thickness_mean": g.thickness.mean().round(2),
        "near4_vox_mean": g.near4_vox.mean().round(0),
        "volumes_with_near4": g.near4_vox.apply(lambda s: int((s > 0).sum())),
    }).reset_index()


def _clean(v):
    """numpy / pandas scalars -> plain Python for json (NaN -> None)."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 4)
    return v


def write_json(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Same information as the two CSVs, grouped by scroll: meta, overall, scrolls -> summary + volumes."""
    df = df.copy()
    df["scroll_id"] = df["scroll_id"].astype(str)
    summary = summary.copy()
    summary["scroll_id"] = summary["scroll_id"].astype(str)
    data = {
        "meta": {
            "source": "data/train_labels/*.tif",
            "volumes": int(len(df)),
            "label_values": {"0": "background", "1": "sheet", "2": "not labelled"},
            "sheet": "26-connected component of label 1",
            "fields": {
                "frac": "share of the volume with label 0 / 1 / 2",
                "sheet_vox": "voxels with label 1",
                "sheets.n / n_small": "sheets, and sheets under 1000 voxels (rank 2's min_size)",
                "sheets.min_vox / max_vox / smallest_5": "smallest / largest / five smallest sheets, in voxels",
                "sheets.largest_share": "largest sheet / sheet_vox",
                "thickness": "2 * volume / surface (voxel faces); relative, reads low on rough sheets",
                "euler": "Euler number of all sheets (26-connectivity)",
                "near_vox.r2 / r4": "voxels within 2 / 4 voxels of two different sheets (merge risk)",
                "fold0": "train or val in our fold-0 split",
            },
        },
        "overall": {
            "min_sheet_vox": int(df.min_sheet_vox[df.n_sheets > 0].min()),
            "n_small_sheets_total": int(df.n_small_sheets.sum()),
            "volumes_with_small_sheets": int((df.n_small_sheets > 0).sum()),
            "volumes_without_sheets": int((df.n_sheets == 0).sum()),
        },
        "scrolls": {},
    }
    for _, s in summary.sort_values("scroll_id").iterrows():
        sid = s["scroll_id"]
        vols = {}
        for _, r in df[df.scroll_id == sid].sort_values("id").iterrows():
            smallest = str(r["smallest_5"]) if pd.notna(r["smallest_5"]) else ""
            vols[str(r["id"])] = {
                "fold0": r["fold0"],
                "frac": {"bg": _clean(r.frac_bg), "sheet": _clean(r.frac_sheet),
                         "unlabelled": _clean(r.frac_unlabelled)},
                "sheet_vox": int(r.sheet_vox),
                "sheets": {"n": int(r.n_sheets), "n_small": int(r.n_small_sheets), "min_vox": int(r.min_sheet_vox),
                           "max_vox": int(r.max_sheet_vox), "smallest_5": [int(x) for x in smallest.split()],
                           "largest_share": _clean(r.largest_share)},
                "thickness": _clean(r.thickness),
                "euler": int(r.euler),
                "near_vox": {"r2": int(r.near2_vox), "r4": int(r.near4_vox)},
            }
        data["scrolls"][sid] = {
            "summary": {k: _clean(v) for k, v in s.items() if k != "scroll_id"},
            "volumes": vols,
        }
    OUT_JSON.write_text(json.dumps(data, indent=2))
    print(f"Saved {OUT_JSON}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3, help="parallel processes (~1 GB RAM each)")
    ap.add_argument("--limit", type=int, help="only the first N volumes (test)")
    ap.add_argument("--json-only", action="store_true", help="only rebuild the summary CSV and JSON from the CSV")
    args = ap.parse_args()

    if args.json_only:
        df = pd.read_csv(OUT, dtype={"id": str, "scroll_id": str})
        summary = per_scroll(df)
        summary.to_csv(OUT_SCROLL, index=False)
        write_json(df, summary)
        return

    train = pd.read_csv(DATA / "train.csv", dtype=str).set_index("id")["scroll_id"]
    val0 = set(json.loads(SPLITS.read_text())[0]["val"]) if SPLITS.exists() else set()
    ids = sorted(p.stem for p in (DATA / "train_labels").glob("*.tif"))[: args.limit]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(OUT, dtype={"id": str, "scroll_id": str}).to_dict("records") if OUT.exists() else []
    done = {r["id"] for r in rows}
    todo = [i for i in ids if i not in done]
    print(f"{len(ids)} volumes, {len(done)} already done, {len(todo)} to do", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(volume_stats, i): i for i in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            row = {"id": row["id"], "scroll_id": train.get(row["id"], ""),
                   "fold0": "val" if row["id"] in val0 else "train", **{k: v for k, v in row.items() if k != "id"}}
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"[{n}/{len(todo)}] {row['id']}  sheets={row['n_sheets']}  min={row['min_sheet_vox']}  "
                  f"near4={row['near4_vox']}  ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    summary = per_scroll(df)
    summary.to_csv(OUT_SCROLL, index=False)
    write_json(df, summary)
    print(f"\nSaved {OUT} and {OUT_SCROLL}\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
