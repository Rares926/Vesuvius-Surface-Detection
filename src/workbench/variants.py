"""Post-processing variants to compare. Each one maps a foreground probability map to a binary mask.

To try an idea, add an entry here and run `python -m src.workbench.run_variants --variants <name> ...`.
Keep `rank2_full` unchanged: it is the reference every other variant is measured against.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.postprocess.pipeline import Rank2Params, rank2_postprocess, simple_threshold
from src.postprocess.rares import MYPRAMSINSTANCE, my_postprocess
from src.postprocess.simple import SimpleParams, simple_postprocess
from src.postprocess.own import OwnParams, own_postprocess

R2 = Rank2Params()

VARIANTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    # floor: no post-processing at all
    "threshold_0.5": lambda p: simple_threshold(p, 0.5),
    "threshold_0.3": lambda p: simple_threshold(p, 0.3),
    # rank 2 up to and including hole filling, without the local interpolation (step 4)
    "rank2_no_interp": lambda p: rank2_postprocess(p, R2.with_(interpolate=False)),
    # rank 2's full post-processing, as in the 0.622 notebook
    "rank2_full": lambda p: rank2_postprocess(p, R2),
    # rank 2 full, but each hole region's height map comes from an optimal-surface min-cut over the probability map
    # instead of griddata over the binary voxels (src/postprocess/optimal_surface.py)
    "rank_2_full_modified_1": lambda p: rank2_postprocess(p, R2.with_(fit="optimal_surface")),
    # own post-processing, built problem by problem (src/postprocess/own.py, planner task I-5).
    # own_s<n> = steps 1..n; add one entry per new step so every stage keeps its own masks.
    "own_s1": lambda p: own_postprocess(p, OwnParams.stage(1)),
    "own_s2": lambda p: own_postprocess(p, OwnParams.stage(2)),
    "own_s3": lambda p: own_postprocess(p, OwnParams.stage(3)),
    "own_s4": lambda p: own_postprocess(p, OwnParams.stage(4)),
    # simple post-processing written from scratch (src/postprocess/simple.py), and its ablations
    "simple": lambda p: simple_postprocess(p, SimpleParams()),
    "simple_plain_closing": lambda p: simple_postprocess(p, SimpleParams().with_(guide_p=0.0)),
    "simple_no_closing": lambda p: simple_postprocess(p, SimpleParams().with_(close_radius=0)),
    # my methods (src/postprocess/rares.py)
    "rares_mth1": lambda p: my_postprocess(p, MYPRAMSINSTANCE),
}
