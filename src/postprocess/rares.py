"""My post-processing methods. Each one maps a foreground probability map to a binary mask.

rares_mth1 = rank 2's full pipeline, then split components that hold two merged sheets:

  1. Detect: cut each large component into overlapping cubes (32^3, every 16 voxels). In each cube, PCA of the
     voxel coordinates gives the local normal (the direction of least spread). A single sheet projected on its
     normal is one block a few voxels thick; two parallel sheets give two blocks with an empty gap between them.
     Such cubes are "flagged", and the voxels on each side of the gap are the two layers.
  2. Split, method A (probability valley): raise the threshold inside the component (0.6, 0.7, ... 0.95) until it
     falls into >= 2 large pieces. Bridges between sheets are usually less confident than the sheets themselves.
  3. Split, method B (geometric fallback, if A finds no valid split): the flagged cubes' voxels, with the gap
     between the two sides, fall into separate pieces; those pieces are the seeds.
  4. Cut: grow the seeds over the component along the probability (watershed on -prob), then remove the voxels where
     two regions touch, so they are no longer 26-connected.

Safety rule: two regions are only cut apart if some flagged cube has one region below the gap and the other above it
(they are parallel layers). Regions that touch without that evidence are joined again, so a wrong split falls back to
"no change" rather than breaking one sheet in two.

Register each method in src/workbench/variants.py, then run it with
  python -m src.workbench.run_variants --models mine_198_fold_0=1 --ids 1004283650 --variants rares_mth1 --no-metric --save-masks
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from src.postprocess.pipeline import Rank2Params, rank2_postprocess

STRUCT26 = np.ones((3, 3, 3), bool)                  # 26-connectivity, as for the components


@dataclass(frozen=True)
class MyParams:
    """Settings of my methods. `with_` returns a copy with some fields changed, like Rank2Params."""
    rank2: Rank2Params = field(default_factory=Rank2Params)   # rank 2's steps, run first
    split: bool = True                    # run the merged-sheet split after rank 2
    check_min_size: int = 20_000          # only components at least this big are checked
    cube: int = 32                        # cube edge for the local PCA (cubes overlap by half)
    min_cube_voxels: int = 300            # cubes with fewer component voxels are skipped
    min_gap: int = 2                      # empty gap along the normal, in voxels, to call two layers
    min_side_frac: float = 0.2            # each layer must hold at least this share of the cube's voxels
    min_flagged: int = 3                  # flagged cubes needed before a component is split
    valley_thresholds: tuple = (0.6, 0.7, 0.8, 0.9, 0.95)   # method A
    min_seed: int = 2_000                 # smallest piece used as a seed (both methods)

    def with_(self, **changes) -> "MyParams":
        return replace(self, **changes)


MYPRAMSINSTANCE = MyParams()


def my_postprocess(prob: np.ndarray, p: MyParams = MYPRAMSINSTANCE, report: list | None = None) -> np.ndarray:
    """prob: (D, H, W) float in 0..1 -> mask: (D, H, W) uint8 0/1. Pass a list as `report` to get one dict per
    checked component (flagged cubes, method used, voxels removed)."""
    mask = rank2_postprocess(prob, p.rank2).astype(bool)
    if p.split:
        mask = split_merged_sheets(mask, prob, p, report)
    return mask.astype(np.uint8)


# ---------------------------------------------------------------- detection
def _cubes(coords: np.ndarray, step: int) -> list[np.ndarray]:
    """Indices of the voxels in each cube of edge 2*step placed every `step` voxels (each voxel is in 8 cubes)."""
    base = coords // step
    n = len(coords)
    keys = np.concatenate([base - np.array(s) for s in itertools.product((0, 1), repeat=3)])
    idx = np.tile(np.arange(n), 8)
    keys -= keys.min(0)
    lin = np.ravel_multi_index(keys.T, keys.max(0) + 1)
    order = np.argsort(lin, kind="stable")
    lin, idx = lin[order], idx[order]
    return np.split(idx, np.flatnonzero(np.diff(lin)) + 1)


def _two_layers(pts: np.ndarray, p: MyParams) -> np.ndarray | None:
    """If the cube's voxels form two parallel layers, return a bool per voxel (True = above the gap), else None."""
    c = pts - pts.mean(0)
    _, vecs = np.linalg.eigh(c.T @ c)
    t = c @ vecs[:, 0]                                # position along the normal (least-spread direction)
    t -= t.min()
    empty = np.bincount(t.astype(int)) == 0           # 1-voxel bins; the first and last bins are never empty
    edges = np.diff(np.concatenate(([0], empty.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if len(starts) == 0:
        return None
    k = np.argmax(ends - starts)
    if ends[k] - starts[k] < p.min_gap:
        return None
    side = t >= ends[k]
    if min(side.mean(), 1 - side.mean()) < p.min_side_frac:
        return None
    return side


def _find_layers(comp: np.ndarray, p: MyParams) -> list[tuple[np.ndarray, np.ndarray]]:
    """Flagged cubes of one component: (voxel coordinates, side of the gap) for each."""
    coords = np.argwhere(comp)
    flagged = []
    for idx in _cubes(coords, p.cube // 2):
        if len(idx) < p.min_cube_voxels:
            continue
        side = _two_layers(coords[idx].astype(np.float64), p)
        if side is not None:
            flagged.append((coords[idx], side))
    return flagged


# ---------------------------------------------------------------- splitting
def _dominant(labels: np.ndarray) -> int:
    labels = labels[labels > 0]
    if labels.size == 0:
        return 0
    counts = np.bincount(labels)
    best = int(counts.argmax())
    return best if counts[best] >= 0.5 * labels.size else 0


def _contacts(lab: np.ndarray) -> set[tuple[int, int]]:
    """Pairs of labels that touch (26-connectivity)."""
    pairs: set[tuple[int, int]] = set()
    offsets = [o for o in itertools.product((-1, 0, 1), repeat=3) if o > (0, 0, 0)]
    for o in offsets:
        a = lab[tuple(slice(max(0, -d), lab.shape[i] - max(0, d)) for i, d in enumerate(o))]
        b = lab[tuple(slice(max(0, d), lab.shape[i] - max(0, -d)) for i, d in enumerate(o))]
        m = (a > 0) & (b > 0) & (a != b)
        if m.any():
            lo, hi = np.minimum(a[m], b[m]), np.maximum(a[m], b[m])
            pairs.update(map(tuple, np.unique(np.stack([lo, hi], 1), axis=0).tolist()))
    return pairs


def _cut(comp: np.ndarray, prob_c: np.ndarray, seeds: np.ndarray, flagged) -> np.ndarray | None:
    """Grow the seeds over the component along the probability and return the voxels to remove so that regions
    that are parallel layers no longer touch. None if no pair of regions is a pair of parallel layers."""
    lab = watershed(-prob_c, markers=seeds, mask=comp, connectivity=3)
    layered = set()
    for pts, side in flagged:
        values = lab[tuple(pts.T)]
        a, b = _dominant(values[~side]), _dominant(values[side])
        if a and b and a != b:
            layered.add((min(a, b), max(a, b)))
    if not layered:
        return None

    parent = {int(k): int(k) for k in np.unique(lab) if k > 0}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in _contacts(lab):                       # regions that touch without being parallel layers: join
        if (a, b) not in layered:
            parent[find(a)] = find(b)
    if all(find(a) == find(b) for a, b in layered):
        return None
    lut = np.zeros(int(lab.max()) + 1, np.int32)
    for k in parent:
        lut[k] = find(k)
    final = lut[lab]
    return (ndi.maximum_filter(final, size=3) > final) & (final > 0)


def _big_pieces(mask: np.ndarray, min_size: int) -> np.ndarray | None:
    pieces, n = ndi.label(mask, STRUCT26)
    if n < 2:
        return None
    sizes = np.bincount(pieces.ravel())
    sizes[0] = 0
    big = np.flatnonzero(sizes >= min_size)
    if len(big) < 2:
        return None
    return np.where(np.isin(pieces, big), pieces, 0)


def _split(comp, prob_c, flagged, p: MyParams) -> tuple[np.ndarray | None, str]:
    # method A: probability valley
    for t in p.valley_thresholds:
        seeds = _big_pieces(comp & (prob_c >= t), p.min_seed)
        if seeds is not None:
            cut = _cut(comp, prob_c, seeds, flagged)
            if cut is not None:
                return cut, f"valley t={t}"
    # method B: the two sides of the gap in the flagged cubes
    seed_mask = np.zeros_like(comp)
    for pts, _ in flagged:
        seed_mask[tuple(pts.T)] = True
    seeds = _big_pieces(seed_mask, p.min_seed)
    if seeds is not None:
        cut = _cut(comp, prob_c, seeds, flagged)
        if cut is not None:
            return cut, "gap"
    return None, "none"


def split_merged_sheets(mask: np.ndarray, prob: np.ndarray, p: MyParams = MYPRAMSINSTANCE,
                        report: list | None = None) -> np.ndarray:
    """Split components that hold two (or more) parallel sheets joined by bridges."""
    out = mask.copy()
    comps, _ = ndi.label(mask, STRUCT26)
    for i, sl in enumerate(ndi.find_objects(comps), 1):
        comp = comps[sl] == i
        size = int(comp.sum())
        if size < p.check_min_size:
            continue
        flagged = _find_layers(comp, p)
        info = {"component": i, "voxels": size, "flagged_cubes": len(flagged), "method": "none", "removed": 0}
        if len(flagged) >= p.min_flagged:
            cut, method = _split(comp, prob[sl], flagged, p)
            if cut is not None:
                out[sl][cut] = False
                info.update(method=method, removed=int(cut.sum()))
        if report is not None:
            report.append(info)
    return out
