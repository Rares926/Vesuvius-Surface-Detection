"""A small synthetic P(sheet) volume in which every post-processing step has something visible to do.

Used by draw_inference_postprocess.py to illustrate the steps. The volume is (z, y, x) = (120, 96, 120); sheets are
surfaces x = f(z) that span all of y, like papyrus sheets crossing a CT block. The picture slice is y = SLICE_Y.

Planted defects, and the step meant to fix each one:
  weak stretch inside sheet 1 (P 0.35)        threshold 0.5 cuts it; hysteresis keeps it (connected to seeds)
  1-voxel z gap in sheet 1                     closing along z
  round hole in sheet 1 (radius 7, P 0.05)     local interpolation (variant 3)
  sheets 2 and 3 joined by two weak bridges    stricter masks split them (variant 3)
  isolated weak blob (P 0.4)                   hysteresis (no seed reaches it)
  tiny strong blob (< 100 voxels)              dust removal
  medium strong blob (100-999 voxels)          size filter (< 1000)
  hollow strong shell                          fill cavities
  sheets touching the top and bottom faces     3-voxel face band
"""
from __future__ import annotations

import numpy as np

Z, Y, X = 120, 96, 120
SLICE_Y = 40
P_STRONG = 0.95


def _ball(zz, yy, xx, c, r):
    return (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= r * r


def make_volume() -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Return the probability volume and, for labels, the (z, x) position of each planted defect on SLICE_Y."""
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X].astype(np.float32)
    p = np.zeros((Z, Y, X), np.float32)

    # sheet 1: gently curved, with a weak stretch, a 1-voxel z gap and a round hole
    x1 = 20 + 4 * np.sin(zz / 15)
    s1 = np.abs(xx - x1) <= 1.5
    p[s1] = P_STRONG
    p[s1 & (zz >= 86) & (zz < 100)] = 0.35
    p[s1 & (zz == 30)] = 0.0
    hole = ((zz - 60) ** 2 + (yy - SLICE_Y) ** 2) <= 7 ** 2
    p[s1 & hole] = 0.05

    # sheets 2 and 3: parallel, 12 voxels apart, joined by two weak bridges (one crosses the picture slice)
    s2 = np.abs(xx - 62) <= 1.5
    s3 = np.abs(xx - 74) <= 1.5
    p[s2 | s3] = P_STRONG
    bridge_zone = (zz >= 50) & (zz < 62) & (xx > 62) & (xx < 74)
    for y0 in (SLICE_Y - 4, SLICE_Y + 22):
        p[bridge_zone & (yy >= y0) & (yy < y0 + 8)] = 0.55

    # blobs and a shell, all centred on the picture slice
    p[_ball(zz, yy, xx, (104, SLICE_Y, 42), 4)] = 0.40          # weak blob, no seed
    p[_ball(zz, yy, xx, (16, SLICE_Y, 42), 2)] = 0.90           # tiny strong blob
    p[_ball(zz, yy, xx, (104, SLICE_Y, 100), 5)] = 0.90         # medium strong blob
    shell = _ball(zz, yy, xx, (34, SLICE_Y, 100), 10) & ~_ball(zz, yy, xx, (34, SLICE_Y, 100), 6)
    p[shell] = 0.90

    labels = {
        "weak stretch": (93, 22), "z gap": (30, 22), "hole": (60, 22), "bridges": (56, 68),
        "weak blob": (104, 42), "tiny blob": (16, 42), "medium blob": (104, 100), "shell": (34, 100),
    }
    return p, labels
