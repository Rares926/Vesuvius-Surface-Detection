"""Rank-2 post-processing as one parameterised function.

`rank2_postprocess(prob)` with default `Rank2Params()` reproduces the per-volume steps of the rank-2 final notebook
(see `rank2_reference.py` for the source and licence):

1. Threshold the foreground probability four times with 3D hysteresis (T_high 0.83, T_low 0.2 / 0.5 / 0.6 / 0.7),
   each followed by a z-only closing, dust removal, zeroing a 3-voxel face band and removing components < 1000 voxels.
2. Fill enclosed cavities in the base (T_low 0.2) mask.
3. Crop 3 voxels per face from all masks.
4. Per connected sheet of the base mask: find cubes with holes (Euler number), fit a smooth surface to each region,
   keep it if Dice / coverage pass, else retry on the stricter masks (un-merges sheets).
5. Remove components < 1000 voxels and pad the crop back.

Every step is a field of `Rank2Params`, so a variant is just a different params object.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage import morphology

from . import rank2_reference as ref


@dataclass(frozen=True)
class Rank2Params:
    # step 1: thresholding (hysteresis + closing + cleanup)
    t_high: float = 0.83
    t_low_base: float = 0.2
    t_low_alts: tuple[float, ...] = (0.5, 0.6, 0.7)
    z_radius: int = 1
    xy_radius: int = 0
    dust_min_size: int = 100
    face_zero: int = 3
    min_size: int = 1000
    # steps 2-3
    fill_holes_base: bool = True
    crop: int = 3
    # step 4: local interpolation (off = hysteresis-only baseline)
    interpolate: bool = True
    cube_size: int = 80
    overlap: int = 40
    border_thickness: int = 8
    grid_resolution: int = 80
    thickness: int = 3
    smoothing: float = 1.0
    samples_per_edge: int = 8
    min_dice: float = 0.75
    min_coverage: float = 0.75
    alt_min_dice: float = 0.45
    alt_min_coverage: float = 0.80
    min_component_size: int = 100
    # the reference code prints a lot; silence it by default
    quiet: bool = True

    def with_(self, **changes) -> "Rank2Params":
        """Copy with some fields changed, e.g. `Rank2Params().with_(t_high=0.8)`."""
        return replace(self, **changes)


def _remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove 26-connected components smaller than `min_size` voxels.

    scikit-image 0.26 replaced `min_size` (removes < n) with `max_size` (removes <= n); `max_size=n-1` keeps the
    reference notebook's behaviour.
    """
    return morphology.remove_small_objects(mask.astype(bool), max_size=min_size - 1, connectivity=3)


def threshold_mask(prob: np.ndarray, t_low: float, p: Rank2Params) -> np.ndarray:
    """Step 1 for one T_low: hysteresis, z-closing, dust removal, face band, small-component removal."""
    mask = ref.topo_postprocess(
        prob,
        T_low=t_low,
        T_high=p.t_high,
        z_radius=p.z_radius,
        xy_radius=p.xy_radius,
        dust_min_size=p.dust_min_size,
    ).astype(bool)
    if p.face_zero > 0:
        mask = ref.zero_volume_faces(mask, thickness=p.face_zero)
    return _remove_small(mask, p.min_size)


def _crop(mask: np.ndarray, c: int) -> np.ndarray:
    return mask[c:-c, c:-c, c:-c] if c > 0 else mask


def _pad(mask: np.ndarray, c: int) -> np.ndarray:
    return np.pad(mask, c, mode="constant", constant_values=0) if c > 0 else mask


def rank2_postprocess(prob: np.ndarray, p: Rank2Params = Rank2Params()) -> np.ndarray:
    """Foreground probability map (D, H, W) in [0, 1] -> binary mask (D, H, W) uint8."""
    prob = np.asarray(prob, dtype=np.float32)

    base = threshold_mask(prob, p.t_low_base, p)
    alts = [threshold_mask(prob, t, p) for t in p.t_low_alts] if p.interpolate else []

    if p.fill_holes_base:
        base = binary_fill_holes(base)

    base = _crop(base, p.crop)
    alts = [_crop(a, p.crop) for a in alts]

    if p.interpolate:
        out = io.StringIO() if p.quiet else None
        with contextlib.redirect_stdout(out) if out is not None else contextlib.nullcontext():
            base, _, _ = ref.process_multiple_components_patchwise(
                volume=base,
                alternative_volumes=alts,
                cube_size=p.cube_size,
                overlap=p.overlap,
                border_thickness=p.border_thickness,
                grid_resolution=p.grid_resolution,
                thickness=p.thickness,
                smoothing=p.smoothing,
                samples_per_edge=p.samples_per_edge,
                min_dice=p.min_dice,
                min_coverage=p.min_coverage,
                alt_min_dice=p.alt_min_dice,
                alt_min_coverage=p.alt_min_coverage,
                min_component_size=p.min_component_size,
            )

    base = _remove_small(base, p.min_size)
    return _pad(base, p.crop).astype(np.uint8)


def simple_threshold(prob: np.ndarray, t: float = 0.5) -> np.ndarray:
    """Plain threshold, no post-processing: the floor every variant should beat."""
    return (np.asarray(prob) >= t).astype(np.uint8)
