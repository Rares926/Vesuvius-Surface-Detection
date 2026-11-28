"""Our own post-processing, built one problem at a time (planner task I-5).

Each step fixes one problem measured in .claude/context/I-5_own_postprocessing/context/postprocessing_problems.md
(section 2) and is recorded with its reason in section 3 of that file.

Structure (the per-sheet frame follows rank 1's post-processing; the repairs inside it are our own choices):

  whole mask
    0. Threshold          p >= 0.3 -> the raw mask the problems are measured on (variant threshold_0.3)
    1. Remove small pieces 26-connected pieces under `min_piece` voxels are deleted
                           -> problem 2, extra pieces (TopoScore k=0, VOI split); also removes the holes inside them
                           (20 000 is also rank 1's `small_obj_thresh`; our own size check backs it, see section 3)
    4. Split merged pieces for each piece, raise the threshold inside it (`split_thresholds`); if it falls into >= 2
                           parts of >= `split_min_part` voxels, give every voxel to the nearest part (watershed on p)
                           and remove the voxels where two parts touch, so they become separate sheets
                           -> merged sheets (VOI merge) and the holes formed by bridges between them (TopoScore k=1).
                           Runs before the per-sheet loop, so the parts are repaired as separate sheets that never
                           touch. Shared intuition with rank 2 (stricter thresholds separate merged sheets), credited.
  per sheet (`repair_per_sheet`): split the mask into sheets (26-connected pieces), largest first; on each sheet run
  the repairs in order, and after EACH repair
    - drop the voxels it added next to another sheet (no two sheets may touch -> no new merges),
    - keep the repair only if the sheet's holes + cavities did not increase, else undo just that repair;
  then write the sheet back.
    2. Dilate             grow the sheet by `dilate` voxels
                           -> problem 1, holes (TopoScore k=1). Sheets are ~2.5 voxels thin, so a closing cannot
                           close their holes, while Surface Dice tolerates the extra thickness (2 voxels).
    3. Guide the dilation keep only the added voxels with p >= `dilate_min_p`
                           -> fixes what step 2 broke: the outer skin it adds (extra false surface) and the crumbs it
                           leaves at the label-2 boundary. Inside a hole the model still half-sees the sheet (low but
                           non-zero p); on the outer skin it is sure there is no sheet (p ~ 0).

Every step is a field of `OwnParams`; setting it to 0 / False turns the step off. `OwnParams.stage(n)` keeps steps
1..n and turns the later ones off, so each stage of the method stays reproducible as steps are added. Each stage is
registered as variant `own_s<n>` and its masks are saved to data/cache/masks/<model>/own_s<n>/.

To add a per-sheet repair: write a function (one sheet -> repaired sheet), add its parameter to `OwnParams` and
`STEP_OFF`, and append it in `sheet_repairs`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import euler_number
from skimage.segmentation import watershed

S26 = np.ones((3, 3, 3), bool)      # 26-connectivity, as the metric uses for pieces


@dataclass(frozen=True)
class OwnParams:
    threshold: float = 0.3      # step 0 (0.5 before 2026-11-28; section 2.0 of the I-5 md says why 0.3)
    min_piece: int = 20_000     # step 1: smallest piece kept, in voxels; 0 = off
    dilate: int = 1             # step 2 (per sheet): dilation, in voxels; 0 = off
    dilate_min_p: float = 0.1   # step 3 (per sheet): added voxels need p >= this; 0 = off (plain dilation)
    split_thresholds: tuple = (0.4, 0.5, 0.6, 0.7, 0.8)   # step 4 (whole mask): thresholds tried; () = off
    split_min_part: int = 20_000                           # step 4: smallest part that counts as a sheet

    def with_(self, **changes) -> "OwnParams":
        return replace(self, **changes)

    # value that turns each step off, in step order; add one entry per new step
    STEP_OFF = {1: {"min_piece": 0}, 2: {"dilate": 0}, 3: {"dilate_min_p": 0.0}, 4: {"split_thresholds": ()}}

    @classmethod
    def stage(cls, n: int) -> "OwnParams":
        """Default params with steps 1..n on and every later step off."""
        off = {}
        for step, values in cls.STEP_OFF.items():
            if step > n:
                off.update(values)
        return cls(**off)


# ---------------------------------------------------------------- whole-mask steps
def remove_small_pieces(mask: np.ndarray, min_piece: int) -> np.ndarray:
    """Step 1: delete 26-connected pieces with fewer than `min_piece` voxels.

    Sizes are measured on the whole volume, since at test time there is no label to say which part counts."""
    if min_piece <= 0:
        return mask
    labels, _ = ndi.label(mask, structure=S26)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_piece
    keep[0] = False                   # label 0 is the background
    return keep[labels]


def split_merged_pieces(mask: np.ndarray, prob: np.ndarray, thresholds: tuple, min_part: int,
                        report: list | None = None) -> np.ndarray:
    """Step 4: split pieces that hold several sheets joined by low-probability bridges.

    For each 26-connected piece, inside its bounding box:
      1. at each threshold t, take the piece's voxels with p >= t and count the parts of >= `min_part` voxels; keep
         the t that gives the most such parts (the lowest t on a tie). Fewer than 2 parts at every t: leave it alone.
      2. grow those parts back over the whole piece along the probability (watershed on -p), so every voxel of the
         piece goes to one part;
      3. remove the voxels where two parts touch, so the parts are no longer connected.
    """
    if not thresholds:
        return mask
    labels, n = ndi.label(mask, structure=S26)
    out = mask.copy()
    for i, sl in enumerate(ndi.find_objects(labels), 1):
        piece = labels[sl] == i
        if piece.sum() < 2 * min_part:
            continue
        p = prob[sl]
        best_t, best_seeds, best_n = None, None, 1
        for t in thresholds:
            parts, _ = ndi.label(piece & (p >= t), structure=S26)
            sizes = np.bincount(parts.ravel())
            sizes[0] = 0
            big = np.flatnonzero(sizes >= min_part)
            if len(big) > best_n:
                lut = np.zeros(len(sizes), np.int32)
                lut[big] = np.arange(1, len(big) + 1)
                best_t, best_seeds, best_n = t, lut[parts], len(big)
        if best_seeds is None:
            continue
        regions = watershed(-p, markers=best_seeds, mask=piece, connectivity=3)
        cut = (ndi.maximum_filter(regions, size=3) > regions) & (regions > 0)   # touches a higher-numbered part
        out[sl] &= ~cut
        if report is not None:
            report.append({"step": "split", "piece": i, "voxels": int(piece.sum()), "threshold": best_t,
                           "parts": best_n, "cut_voxels": int(cut.sum())})
    return out


# ---------------------------------------------------------------- per-sheet repairs
@dataclass(frozen=True)
class Repair:
    """One repair applied to a single sheet."""
    name: str
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray]   # (sheet crop, probability crop) -> repaired sheet
    reach: int                                # how many voxels beyond the sheet it can add (sizes the crop)


def dilate(sheet: np.ndarray, prob: np.ndarray, r: int, min_p: float = 0.0) -> np.ndarray:
    """Step 2: grow the sheet by r voxels (26-connectivity).
    Step 3 (min_p > 0): keep only the added voxels where the model's probability is >= min_p."""
    grown = ndi.binary_dilation(sheet, structure=S26, iterations=r)
    if min_p > 0:
        grown = sheet | (grown & (prob >= min_p))
    return grown


def sheet_repairs(p: OwnParams) -> list[Repair]:
    """The per-sheet repairs that are switched on, in the order they run."""
    repairs = []
    if p.dilate > 0:
        name = "dilate" if p.dilate_min_p <= 0 else f"dilate_p>={p.dilate_min_p:g}"
        repairs.append(Repair(name, lambda s, pr: dilate(s, pr, p.dilate, p.dilate_min_p), p.dilate))
    return repairs


# ---------------------------------------------------------------- the per-sheet loop
def holes_and_cavities(mask: np.ndarray) -> int:
    """b1 + b2 of a binary volume: holes / tunnels plus enclosed cavities.

    In 3D the Euler number is b0 - b1 + b2, so b1 = b0 + b2 - Euler; then b1 + b2 = b0 + 2 b2 - Euler."""
    _, b0 = ndi.label(mask, structure=S26)
    _, b2 = ndi.label(ndi.binary_fill_holes(mask) & ~mask)       # cavities: enclosed background, 6-connected
    return b0 + 2 * b2 - int(euler_number(mask, connectivity=3))


def repair_per_sheet(mask: np.ndarray, prob: np.ndarray, repairs: list[Repair],
                     report: list | None = None) -> np.ndarray:
    """Split the mask into sheets and repair each one on its own, then write it back.

    Sheets are 26-connected pieces, handled largest first, each inside its bounding box grown by the repairs'
    total reach + 1. For each sheet, every repair runs in order on the result of the previous one; after each:
      - voxels it added next to another sheet (original, or already written) are dropped, so no two sheets merge;
      - it is kept only if the sheet's holes + cavities did not increase; otherwise just that repair is undone.
    Pass a list as `report` to get one dict per sheet and repair (sheet, voxels, repair, holes before/after, kept).
    """
    if not repairs:
        return mask
    labels, n = ndi.label(mask, structure=S26)
    if n == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    pad = sum(r.reach for r in repairs) + 1
    out = np.zeros_like(mask, dtype=bool)
    boxes = ndi.find_objects(labels)
    for i in np.argsort(-sizes[1:]) + 1:                       # largest sheet first
        box = boxes[i - 1]
        sl = tuple(slice(max(s.start - pad, 0), min(s.stop + pad, dim)) for s, dim in zip(box, mask.shape))
        sheet = labels[sl] == i
        prob_crop = prob[sl]
        others = (labels[sl] > 0) & ~sheet | out[sl]            # every other sheet, original or already written
        forbidden = ndi.binary_dilation(others, structure=S26)   # voxels that would touch another sheet

        current, current_holes = sheet, holes_and_cavities(sheet)
        for rep in repairs:
            candidate = rep.fn(current, prob_crop)
            candidate &= ~(forbidden & ~current)                 # drop added voxels that touch another sheet
            cand_holes = holes_and_cavities(candidate)
            kept = cand_holes <= current_holes
            if report is not None:
                report.append({"sheet": int(i), "voxels": int(sizes[i]), "repair": rep.name,
                               "holes_before": current_holes, "holes_after": cand_holes, "kept": kept})
            if kept:
                current, current_holes = candidate, cand_holes
        out[sl] |= current
    return out


# ---------------------------------------------------------------- the method
def own_postprocess(prob: np.ndarray, p: OwnParams = OwnParams(), report: list | None = None) -> np.ndarray:
    """Foreground probability map (D, H, W) in [0, 1] -> binary mask (D, H, W) uint8."""
    prob = np.asarray(prob)
    mask = prob >= p.threshold                                         # 0  whole mask
    mask = remove_small_pieces(mask, p.min_piece)                      # 1  whole mask
    mask = split_merged_pieces(mask, prob, p.split_thresholds, p.split_min_part, report)   # 4  whole mask
    mask = repair_per_sheet(mask, prob, sheet_repairs(p), report)      # 2, 3 per sheet
    return mask.astype(np.uint8)
