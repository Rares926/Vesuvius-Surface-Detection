"""Post-processing variants to compare. Each one maps a foreground probability map to a binary mask.

To try an idea, add an entry here and run `python -m src.workbench.run_variants --variants <name> ...`.
Keep `rank2_full` unchanged: it is the reference every other variant is measured against.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.postprocess.pipeline import Rank2Params, rank2_postprocess, simple_threshold

R2 = Rank2Params()

VARIANTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    # floor: no post-processing at all
    "threshold_0.5": lambda p: simple_threshold(p, 0.5),
    # rank 2 up to and including hole filling, without the local interpolation (step 4)
    "rank2_no_interp": lambda p: rank2_postprocess(p, R2.with_(interpolate=False)),
    # rank 2's full post-processing, as in the 0.622 notebook
    "rank2_full": lambda p: rank2_postprocess(p, R2),
}
