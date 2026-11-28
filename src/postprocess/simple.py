"""A simple post-processing, written from scratch: probability map -> binary mask in five steps.

The model gives, for every voxel, the probability p that it lies on a papyrus surface. The metric then scores the
binary mask on three terms, and each step below targets one kind of error:

  Surface Dice  overlap of the predicted surface with the true one (within a small tolerance)
  TopoScore     Betti-number matching: k=0 extra/missing pieces, k=1 holes/tunnels, k=2 enclosed cavities
  VOI           whether sheets are wrongly merged together or split apart

  1. Hysteresis threshold   keep voxels with p >= t_low only if their piece also holds a voxel with p >= t_high.
                            Faint noise far from any confident voxel is dropped; faint parts of real sheets are kept.
                            -> fewer false pieces (TopoScore k=0, VOI)
  2. Remove small pieces    pieces under `min_size` voxels are noise.  -> TopoScore k=0
  3. Guided closing         a morphological closing (ball of radius `close_radius`) fills small holes in a sheet,
                            but a voxel is only added where p > `guide_p`. A plain closing would also glue two
                            close sheets together; the model's probability says where a bridge is plausible.
                            -> fewer holes (TopoScore k=1) without new merges (VOI)
  4. Fill cavities          enclosed air pockets are filled.  -> TopoScore k=2
  5. Ignore the border      the outer `border` voxels of each face are mostly unlabelled; predicting there only
                            adds errors. Small pieces left over by steps 3-5 are removed again.

Every step is a field of `SimpleParams`, so the ablation is `SimpleParams().with_(step_off...)`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import ball

STRUCT26 = np.ones((3, 3, 3), bool)      # 26-connectivity: voxels touching by a face, an edge or a corner


@dataclass(frozen=True)
class SimpleParams:
    t_high: float = 0.8       # step 1: seed threshold
    t_low: float = 0.3        # step 1: growth threshold
    min_size: int = 1000      # step 2 (and 5): smallest piece kept, in voxels
    close_radius: int = 2     # step 3: ball radius of the closing; 0 = step off
    guide_p: float = 0.05     # step 3: closing may only add voxels with p above this; 0 = plain closing
    fill_cavities: bool = True  # step 4
    border: int = 3           # step 5: voxels zeroed on each face; 0 = step off

    def with_(self, **changes) -> "SimpleParams":
        return replace(self, **changes)


def hysteresis(prob: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    """Step 1: pieces of (p >= t_low) that contain at least one voxel with p >= t_high."""
    labels, _ = ndi.label(prob >= t_low, structure=STRUCT26)
    seeded = np.unique(labels[prob >= t_high])
    seeded = seeded[seeded > 0]
    return np.isin(labels, seeded)


def remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Step 2: drop 26-connected pieces with fewer than `min_size` voxels."""
    labels, _ = ndi.label(mask, structure=STRUCT26)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False                           # label 0 is the background
    return keep[labels]


def guided_closing(mask: np.ndarray, prob: np.ndarray, radius: int, guide_p: float) -> np.ndarray:
    """Step 3: closing (dilate, then erode) adds the voxels of small holes; keep only those with p > guide_p.
    The mask is padded first so the closing does not treat the volume edge as background."""
    if radius <= 0:
        return mask
    pad = radius + 1
    closed = ndi.binary_closing(np.pad(mask, pad), structure=ball(radius))[pad:-pad, pad:-pad, pad:-pad]
    added = closed & ~mask & (prob > guide_p)
    return mask | added


def zero_border(mask: np.ndarray, b: int) -> np.ndarray:
    """Step 5: set the outer `b` voxels of every face to background."""
    if b <= 0:
        return mask
    out = np.zeros_like(mask)
    out[b:-b, b:-b, b:-b] = mask[b:-b, b:-b, b:-b]
    return out


def simple_postprocess(prob: np.ndarray, p: SimpleParams = SimpleParams()) -> np.ndarray:
    """Foreground probability map (D, H, W) in [0, 1] -> binary mask (D, H, W) uint8."""
    prob = np.asarray(prob, dtype=np.float32)
    mask = hysteresis(prob, p.t_low, p.t_high)                   # 1
    mask = remove_small(mask, p.min_size)                        # 2
    mask = guided_closing(mask, prob, p.close_radius, p.guide_p)  # 3
    if p.fill_cavities:
        mask = ndi.binary_fill_holes(mask)                       # 4
    mask = zero_border(mask, p.border)                           # 5
    mask = remove_small(mask, p.min_size)
    return mask.astype(np.uint8)
