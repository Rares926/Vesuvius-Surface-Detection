"""Draw the full T-1 pipeline: nnU-Net inference on the real volume, then the three post-processing variants.

Step 1 (inference) uses the real T-1 volume. Step 2 (post-processing) shows each step on a small synthetic P(sheet)
volume (src/diagrams/synthetic_sheets.py) in which every step has a planted defect to fix, so its effect is visible;
the scores in Step 2 are still those of the real volume. All post-processing pictures are computed with the project's
own code, and the synthetic end results are checked against `rank2_postprocess`.

Colours of a step picture: teal = mask after the step, purple = added by the step, amber = removed by the step.

Usage (from the project root; call the env's python directly, `conda run` hangs on the reference code's output):
  Z:/anaconda3/envs/vesuvius/python.exe src/diagrams/draw_inference_postprocess.py <output.svg> [volume_id] [model]
Needs: data/cache/probs/<model>/<id>.npy and data/cache/masks/{rank2_no_interp,rank2_full}/<id>.tif
"""
from __future__ import annotations

import contextlib
import io
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import scipy.ndimage as ndi
import tifffile
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage import morphology
from skimage.measure import label

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common import image_path, label_path, prob_path  # noqa: E402
from src.diagrams.synthetic_sheets import SLICE_Y, make_volume  # noqa: E402
from src.postprocess import rank2_reference as ref  # noqa: E402
from src.postprocess.pipeline import Rank2Params, rank2_postprocess, threshold_mask  # noqa: E402

OUT = Path(sys.argv[1])
VOL = sys.argv[2] if len(sys.argv) > 2 else "105796630"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "m128"
P = Rank2Params()
PLANS = {"lo": 0.0, "hi": 212.0, "mean": 87.54424285888672, "std": 47.74376678466797}   # models/v2/plans.json
PATCH, STEP = 128, 0.5

# ============================================================== real volume: inference pictures and scores
ct = tifffile.imread(image_path(VOL))
gt = tifffile.imread(label_path(VOL))
prob = np.load(prob_path(MODEL, VOL)).astype(np.float32)
saved_noi = tifffile.imread(ROOT / f"data/cache/masks/rank2_no_interp/{VOL}.tif").astype(bool)
saved_full = tifffile.imread(ROOT / f"data/cache/masks/rank2_full/{VOL}.tif").astype(bool)
IMAGE = ct.shape[2]
change = (saved_full != saved_noi).sum(axis=(0, 2))
MARGIN = 40
YS = MARGIN + int(np.argmax(change[MARGIN:-MARGIN]))      # real slice shown in Step 1

known = gt != 2
g1 = gt == 1
dist_to_gt = distance_transform_edt(~g1)


def metrics(m):
    m = m & known
    near = float((dist_to_gt[m] <= 2).mean()) if m.any() else 0.0
    cover = float((distance_transform_edt(~m)[g1] <= 2).mean())
    return m.mean() / known.mean(), near, cover


MET = {"threshold_0.5": metrics(prob >= 0.5), "rank2_no_interp": metrics(saved_noi),
       "rank2_full": metrics(saved_full)}
above_all = float((ct > PLANS["hi"]).mean())

num_steps = math.ceil((IMAGE - PATCH) / (PATCH * STEP)) + 1
starts = [int(round((IMAGE - PATCH) / (num_steps - 1) * i)) for i in range(num_steps)]

# ============================================================== synthetic volume: every post-processing step
toy, DEFECTS = make_volume()
c = P.crop
t_v1 = toy >= 0.5
t_strong, t_weak = toy >= P.t_high, toy >= P.t_low_base
t_hyst = ndi.binary_propagation(t_strong, mask=t_weak, structure=ndi.generate_binary_structure(3, 3))
t_closed = ndi.binary_closing(t_hyst, structure=ref.build_anisotropic_struct(P.z_radius, P.xy_radius))
t_dust = morphology.remove_small_objects(t_closed, max_size=P.dust_min_size - 1)      # reference: min_size=100
t_faced = ref.zero_volume_faces(t_dust, thickness=P.face_zero)
t_big = morphology.remove_small_objects(t_faced, max_size=P.min_size - 1, connectivity=3)
t_filled = binary_fill_holes(t_big)
t_v2 = np.pad(morphology.remove_small_objects(t_filled[c:-c, c:-c, c:-c], max_size=P.min_size - 1, connectivity=3),
              c)
assert np.array_equal(t_v2, rank2_postprocess(toy, P.with_(interpolate=False)).astype(bool)), "variant 2 mismatch"

t_alts = [threshold_mask(toy, t, P)[c:-c, c:-c, c:-c] for t in P.t_low_alts]
t_base = t_filled[c:-c, c:-c, c:-c]
comps = label(t_base)
n_comp = int(comps.max())
regions, n_cubes, n_hole_cubes = [], 0, 0
for i in range(1, n_comp + 1):
    comp = comps == i
    cubes = ref.divide_component_into_cubes(comp, cube_size=P.cube_size, overlap=P.overlap)
    n_cubes += len(cubes)
    holes = [cb for cb in cubes if ref.check_cube_topology(cb["mask"], min_voxels=20)[0]]
    n_hole_cubes += len(holes)
    for g in ref.find_overlapping_cube_groups(holes):
        regions.append(ref.merge_cube_regions(holes, g, comp.shape)[0])
np.random.seed(0)                                         # the surface fit samples points at random
with contextlib.redirect_stdout(io.StringIO()):
    t_interp, _, _ = ref.process_multiple_components_patchwise(
        volume=t_base, alternative_volumes=t_alts, cube_size=P.cube_size, overlap=P.overlap,
        border_thickness=P.border_thickness, grid_resolution=P.grid_resolution, thickness=P.thickness,
        smoothing=P.smoothing, samples_per_edge=P.samples_per_edge, min_dice=P.min_dice,
        min_coverage=P.min_coverage, alt_min_dice=P.alt_min_dice, alt_min_coverage=P.alt_min_coverage,
        min_component_size=P.min_component_size)
t_v3 = np.pad(morphology.remove_small_objects(t_interp, max_size=P.min_size - 1, connectivity=3), c)
t_interp = np.pad(t_interp, c)
np.random.seed(0)
with contextlib.redirect_stdout(io.StringIO()):
    check_v3 = rank2_postprocess(toy, P).astype(bool)
assert np.array_equal(t_v3, check_v3), "variant 3 mismatch"
regions = [(z0 + c, z1 + c, y0 + c, y1 + c, x0 + c, x1 + c) for z0, z1, y0, y1, x0, x1 in regions]
n_final = int(label(t_v3).max())
n_base = int(label(t_v2).max())

# ============================================================== SVG helpers
W = 1400
S = 2                                  # real slices: voxels per pixel
T = 160                                # thumbnail size in pixels
TPX = T / toy.shape[2]                 # synthetic slices: pixels per voxel
svg: list[str] = []
WARN: list[str] = []
CHAR_W = {"th": 7.9, "t": 7.4, "ts": 6.3, "h3": 8.6}
TEAL, PURPLE, AMBER, CORAL, LGRAY, DARK = "#1D9E75", "#534AB7", "#BA7517", "#993C1D", "#D3D1C7", "#2C2C2A"
GRAYS = ["#FFFFFF", "#E4E2DA", "#C9C7BE", "#A3A198", "#7A7870", "#4E4D48", "#2C2C2A"]


def add(s):
    svg.append(s)


def text(x, y, s, cls="t", anchor="start", width=None):
    if width is not None and len(s) * CHAR_W.get(cls, 7) > width:
        WARN.append(f'"{s}" ~{len(s) * CHAR_W.get(cls, 7):.0f}px > {width}px')
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    add(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}"{a}>{escape(s)}</text>')


def raster(codes, palette, x0, y0, px=1.0):
    """Draw a 2D array of small ints as one path per colour (row runs), framed."""
    for code, colour in palette.items():
        d = []
        for r in range(codes.shape[0]):
            row = codes[r] == code
            if not row.any():
                continue
            e = np.flatnonzero(np.diff(np.r_[0, row.astype(np.int8), 0]))
            for a, b in zip(e[::2], e[1::2]):
                d.append(f"M{x0 + a * px:.1f} {y0 + r * px:.1f}h{(b - a) * px:.1f}v{px:.1f}h{-(b - a) * px:.1f}z")
        if d:
            add(f'<path d="{"".join(d)}" fill="{colour}"/>')
    add(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{codes.shape[1] * px:.1f}" height="{codes.shape[0] * px:.1f}" '
        f'fill="none" stroke="#888780" stroke-width="1"/>')


def pool(a, how):
    z, x = (a.shape[0] // S) * S, (a.shape[1] // S) * S
    a = a[:z, :x].reshape(z // S, S, x // S, S)
    return a.max(axis=(1, 3)) if how == "max" else a.mean(axis=(1, 3))


def arrow(x1, y1, x2, y2=None):
    y2 = y1 if y2 is None else y2
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2 - 2:.1f}" y2="{y2:.1f}" class="arr" marker-end="url(#arrow)"/>')


def box(cls, x, y, w, h):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" class="{cls}"/>')


def real_ct(x, y):
    v = pool(ct[:, YS, :].astype(np.float32), "mean")
    raster(np.clip((v / 256 * len(GRAYS)).astype(int), 0, len(GRAYS) - 1), {i: g for i, g in enumerate(GRAYS) if i},
           x, y)


def prob_codes(p):
    return (p >= 0.2).astype(int) + (p >= 0.5) + (p >= 0.83)


PROB_LV = {1: LGRAY, 2: "#888780", 3: DARK}


def real_prob(x, y):
    raster(prob_codes(pool(prob[:, YS, :], "max")), PROB_LV, x, y)


def real_gt(x, y):
    g = gt[:, YS, :]
    raster(np.where(pool(g == 1, "max"), 2, np.where(pool(g == 2, "max"), 1, 0)), {1: LGRAY, 2: CORAL}, x, y)


def toy_prob(x, y, px=TPX):
    raster(prob_codes(toy[:, SLICE_Y, :]), PROB_LV, x, y, px)


def toy_step(prev, cur, x, y):
    p, q = prev[:, SLICE_Y, :], cur[:, SLICE_Y, :]
    codes = np.zeros(p.shape, np.int8)
    codes[q] = 1
    codes[p & ~q] = 2
    codes[q & ~p] = 3
    raster(codes, {1: TEAL, 2: AMBER, 3: PURPLE}, x, y, TPX)
    return int((cur & ~prev).sum()), int((prev & ~cur).sum())


def delta(added, removed):
    return f"+{added:,} / −{removed:,} voxels".replace(",", " ")


# ---- small pictures for the inference boxes (options 1A, 2A, 3B, 4A)
def mini_hist(x, y, w=170, h=96):
    for vals, colour in ((np.bincount(ct.ravel(), minlength=256)[:256], "#888780"),
                         (np.bincount(ct[gt == 1].ravel(), minlength=256)[:256], CORAL)):
        v = vals / vals.max()
        pts = " ".join(f"{x + i * w / 255:.1f},{y + h - v[i] * h:.1f}" for i in range(256))
        add(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="1.3"/>')
    add(f'<rect x="{x + 212 * w / 255:.1f}" y="{y}" width="{43 * w / 255:.1f}" height="{h}" fill="{AMBER}" '
        f'fill-opacity="0.2"/>')
    for v in (0, 212):
        add(f'<line x1="{x + v * w / 255:.1f}" y1="{y - 3}" x2="{x + v * w / 255:.1f}" y2="{y + h}" '
            f'stroke="#854F0B" stroke-width="1.2" stroke-dasharray="3 2"/>')
    add(f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" stroke="#5F5E5A"/>')
    for v in (0, 212, 255):
        text(x + v * w / 255, y + h + 13, str(v), "ts", "middle")
    text(x, y + h + 30, "coral: sheet voxels; gray: all voxels", "ts")


def mini_strip(x, y, w=170):
    k = w / IMAGE
    add(f'<rect x="{x}" y="{y}" width="{w}" height="10" fill="#F1EFE8" stroke="#5F5E5A"/>')
    for i, s in enumerate(starts):
        yy = y + 22 + i * 22
        add(f'<rect x="{x + s * k:.1f}" y="{yy}" width="{PATCH * k:.1f}" height="11" rx="2" fill="#E1F5EE" '
            f'stroke="#0F6E56"/>')
        text(x + s * k + 3, yy + 9.5, f"{s}–{s + PATCH}", "ts")
        if i:
            arrow(x + starts[i - 1] * k, yy - 4, x + s * k + 2, yy - 4)
    text(x + w, y + 22 + num_steps * 22 + 6, f"→ = stride {int(PATCH * STEP)}", "ts", "end")


def mini_flips(x, y):
    flips = ["", "x", "y", "z", "xy", "xz", "yz", "xyz"]
    for i, f in enumerate(flips):
        fx, fy = x + (i % 4) * 42, y + (i // 4) * 58
        add(f'<rect x="{fx}" y="{fy}" width="32" height="32" rx="3" fill="#FFFFFF" stroke="#534AB7"/>')
        sx = -1 if "x" in f else 1
        sy = -1 if "y" in f else 1
        add(f'<text x="0" y="0" class="h3" text-anchor="middle" transform="translate({fx + 16} {fy + 16}) '
            f'scale({sx} {sy}) translate(0 6)">F</text>')
        text(fx + 16, fy + 46, f or "none", "ts", "middle")


def mini_curves(x, y, w=170, h=80):
    k = w / IMAGE
    base = y + h
    add(f'<line x1="{x}" y1="{base}" x2="{x + w}" y2="{base}" stroke="#888780"/>')
    sig = PATCH / 8
    for i, s in enumerate(starts):
        pts = " ".join(f"{x + v * k:.1f},{base - math.exp(-((v - s - PATCH // 2) ** 2) / (2 * sig ** 2)) * (h - 6):.1f}"
                       for v in range(s, s + PATCH + 1, 2))
        dash = "" if i == 0 else ' stroke-dasharray="4 2"'
        add(f'<polyline points="{pts}" fill="none" stroke="#0F6E56" stroke-width="{1.8 if i == 0 else 1.1}"{dash}/>')
    for v in (0, 64, 128, 192, 256, 320):
        text(x + v * k, base + 13, str(v), "ts", "middle")


# ============================================================== title, input, legend
y = 38
text(30, y, f"T-1 pipeline: inference on volume {VOL} (rank-2 128³ model), then the three post-processing variants",
     "title")
y += 22
text(30, y, f"Step 1 uses the real volume (slice y = {YS}, x across, z down). Step 2 shows each step on a small "
     "synthetic P(sheet) built so that every step visibly does something.", "ts")

box("img", 30, 78, 820, 92)
text(46, 100, "Input", "th")
text(46, 120, f"CT volume data/train_images/{VOL}.tif: {ct.shape[0]}×{ct.shape[1]}×{ct.shape[2]} voxels (z, y, x), "
     "uint8", "ts")
text(46, 138, "Model models/v2/nnUNet_ResEnc (128³ patches, fold all), tile step 0.5, Gaussian on, mirroring off", "ts")
text(46, 156, f"Synthetic volume for Step 2: {toy.shape[0]}×{toy.shape[1]}×{toy.shape[2]} voxels, picture slice "
     f"y = {SLICE_Y} (src/diagrams/synthetic_sheets.py)", "ts")

lx, ly = 880, 78
box("panel", lx, ly, 490, 92)
text(lx + 14, ly + 20, "Colours in the pictures", "th")
legend = [(TEAL, "mask after this step"), (PURPLE, "added by this step"), (AMBER, "removed by this step"),
          (CORAL, "ground truth sheet (label 1)"), (LGRAY, "unlabelled (label 2)"),
          (DARK, "P ≥ 0.83 / 0.5 / 0.2 (dark → light)")]
for i, (colr, lab) in enumerate(legend):
    cx = lx + 14 + (i % 2) * 220
    cy = ly + 36 + (i // 2) * 18
    add(f'<rect x="{cx}" y="{cy - 10}" width="12" height="12" fill="{colr}" stroke="#5F5E5A" stroke-width="0.5"/>')
    text(cx + 18, cy, lab, "ts", width=255 if i % 2 else 195)

# ============================================================== STEP 1: inference
add("<!-- ===================== STEP 1 INFERENCE ===================== -->")
py = 190
INF = [
    ("img", "CT slice (real)", ["uint8, 0–255"], real_ct),
    ("sel", "1. Normalise", ["clip to 0 … 212: the 0.5 / 99.5 %", "percentiles of training CT on",
                             f"labelled voxels; {above_all:.2%} of this", "volume is above 212 → set to 212;",
                             f"then (v − {PLANS['mean']:.1f}) / {PLANS['std']:.1f}"],
     lambda x, y: mini_hist(x + 5, y + 6)),
    ("sel", "2. Sliding window", [f"stride = patch × step = {PATCH} × {STEP}", f"= {int(PATCH * STEP)} voxels; "
                                  f"{num_steps} starts per", f"axis → {num_steps}³ = {num_steps ** 3} patches "
                                  f"of {PATCH}³"],
     lambda x, y: mini_strip(x + 5, y + 20)),
    ("sel", "3. Mirroring (off here)", ["each patch predicted 8 times", "(flips of z, y, x), flipped back,",
                                        "logits averaged; 8× slower"],
     lambda x, y: mini_flips(x + 4, y + 22)),
    ("sel", "4. Gaussian blend", [f"weight peaks at the patch centre", f"(sigma = {PATCH} / 8 = {PATCH // 8});",
                                  "logit = Σ w·logit_i / Σ w"],
     lambda x, y: mini_curves(x + 5, y + 50)),
    ("out", "5. Softmax → P(sheet)", ["2 classes; keep the sheet", "channel, cached as float16",
                                      f"{MODEL}/{VOL}.npy"], real_prob),
]
bw, gap = 200, 22
bx = 36
by = py + 66
n_lines = max(len(b[2]) for b in INF)
bh = 34 + T + 18 + n_lines * 16 + 6
ph1 = 66 + bh + 20
box("panel", 20, py, 1360, ph1)
text(36, py + 28, "Step 1 - Inference (GPU, once per volume): src/infer/predict_probs.py", "h2")
text(36, py + 48, "All of 1–5 happens inside nnU-Net's predict_single_npy_array. More detail: "
     "sliding_window_gaussian.svg.", "ts")
for i, (cls, title, lines, draw) in enumerate(INF):
    box(cls, bx, by, bw, bh)
    text(bx + bw / 2, by + 22, title, "th", "middle", width=bw - 10)
    draw(bx + (bw - T) / 2 if draw in (real_ct, real_prob) else bx + 10, by + 34)
    ty = by + 34 + T + 18
    for ln in lines:
        text(bx + bw / 2, ty, ln, "ts", "middle", width=bw - 8)
        ty += 16
    if i < len(INF) - 1:
        arrow(bx + bw, by + bh / 2, bx + bw + gap)
    bx += bw + gap

# ============================================================== STEP 2: post-processing
add("<!-- ===================== STEP 2 POST-PROCESSING ===================== -->")
arrow(700, py + ph1, 700, py + ph1 + 30)
py2 = py + ph1 + 30
col, cgap = 165, 26
start_idx = len(svg)


def step_col(x, y0, title, lines, draw):
    text(x + col / 2, y0 + 16, title, "th", "middle", width=col + 4)
    draw(x + (col - T) / 2, y0 + 26)
    ty = y0 + 26 + T + 18
    for ln in lines:
        text(x + col / 2, ty, ln, "ts", "middle", width=col + 8)
        ty += 16


text(36, py2 + 28, "Step 2 - Post-processing (CPU, re-run freely): src/workbench/run_variants.py", "h2")
text(36, py2 + 48, "Three separate variants, each applied to the same P(sheet). They are alternatives to compare, "
     "not steps of one chain.", "ts")

# real reference row
ry = py2 + 62
x = 36
step_col(x, ry, "Real P(sheet)", ["from Step 1"], real_prob)
x += col + cgap
step_col(x, ry, "Real ground truth", ["for comparison only"], real_gt)
x += col + cgap + 20
mt = ry + 30
text(x, mt, f"Each variant on the real volume {VOL} (labelled voxels only):", "th")
for i, (name, (fg, near, cover)) in enumerate(MET.items()):
    text(x, mt + 26 + i * 22, f"{name}: foreground {fg:.1%}, predicted voxels within 2 voxels of the label "
         f"{near:.2f}, label voxels within 2 voxels of a prediction {cover:.2f}", "ts")
text(x, mt + 26 + 3 * 22 + 4, "Quick checks only, not the competition metric (topometrics is not installed yet).",
     "ts")
y = ry + 26 + T + 18 + 16 + 30

# synthetic input with numbered defects
add("<!-- synthetic input -->")
text(36, y + 4, "The synthetic P(sheet) used for every picture below (computed, same code as for real data)", "h3")
y += 16
SPX = 2.0
toy_prob(36, y, SPX)
fixes = {
    "weak stretch": "weak part of a sheet (P 0.35): threshold 0.5 cuts it, hysteresis keeps it",
    "z gap": "1-voxel gap along z: closing along z (step b)",
    "hole": "round hole in a sheet: local interpolation (variant 3)",
    "bridges": "two sheets joined by weak bridges (P 0.55): stricter masks split them (variant 3)",
    "weak blob": "weak blob (P 0.4) that no seed reaches: hysteresis (step a)",
    "tiny blob": "tiny strong blob (33 voxels): dust removal (step c)",
    "medium blob": "medium strong blob (515 voxels): size filter < 1000 (step d)",
    "shell": "hollow ball: fill cavities (step e)",
}
for n, (name, (zz, xx)) in enumerate(DEFECTS.items(), start=1):
    px_, py_ = 36 + xx * SPX, y + zz * SPX                 # the defect
    cx, cy = px_ + 20, py_                                  # its marker, beside it so the defect stays visible
    add(f'<line x1="{px_ + 5:.1f}" y1="{py_:.1f}" x2="{cx - 8:.1f}" y2="{cy:.1f}" stroke="#534AB7" '
        f'stroke-width="1"/>')
    add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="#FFFFFF" stroke="#534AB7" stroke-width="1.3"/>')
    text(cx, cy + 4, str(n), "ts", "middle")
    text(36 + toy.shape[2] * SPX + 30, y + 20 + (n - 1) * 22, f"{n}. {fixes[name]}", "ts")
text(36 + toy.shape[2] * SPX + 30, y + 20 + len(DEFECTS) * 22 + 6,
     "Also: the sheets touch the top and bottom faces, so the 3-voxel face band (step d) removes their ends.", "ts")
text(36 + toy.shape[2] * SPX + 30, y + 20 + len(DEFECTS) * 22 + 24,
     f"Sheets are surfaces spanning all {toy.shape[1]} y-slices; the bridges sit at y = {SLICE_Y - 4}–{SLICE_Y + 3} "
     f"and {SLICE_Y + 22}–{SLICE_Y + 29}, the hole is centred on y = {SLICE_Y}.", "ts")
y += toy.shape[0] * SPX + 34

# ---- variant 1
add("<!-- variant 1 -->")
text(36, y + 4, "Variant 1 - threshold_0.5 (simple_threshold): the floor every variant should beat", "h3")
y += 14
x = 36
step_col(x, y, "Synthetic P(sheet)", ["input"], toy_prob)
arrow(x + col, y + 26 + T / 2, x + col + cgap)
x += col + cgap
v1_lost = (toy >= 0.2) & ~t_v1
step_col(x, y, "P ≥ 0.5", ["amber: 0.2 ≤ P < 0.5, dropped", f"({int(v1_lost.sum())} voxels); "
                           "the weak", "stretch and weak blob vanish"],
         lambda tx, ty: toy_step(toy >= 0.2, t_v1, tx, ty))
y += 26 + T + 18 + 16 * 3 + 30

# ---- variant 2
add("<!-- variant 2 -->")
text(36, y + 4, "Variant 2 - rank2_no_interp (rank2_postprocess with interpolate=False): thresholding and clean-up",
     "h3")
y += 14
x = 36


def seeds_pic(tx, ty):
    raster(np.where(t_strong[:, SLICE_Y], 2, np.where(t_weak[:, SLICE_Y], 1, 0)), {1: LGRAY, 2: DARK}, tx, ty, TPX)


V2 = [
    ("Seeds, growth area", [f"seeds: P ≥ {P.t_high} (dark)", f"may grow into: P ≥ {P.t_low_base}", "(light)"],
     None, None),
    ("a. Hysteresis", ["seeds grow through", "26-connected P ≥ 0.2;", "weak blob (5) is unreached"], t_weak, t_hyst),
    ("b. Closing along z", ["3×1×1: fills the z gap (2);", "also clears z = 0 and last z", "(SciPy border effect)"],
     t_hyst, t_closed),
    ("c. Dust removal", [f"drop pieces < {P.dust_min_size} voxels", "(6-connected):", "tiny blob (6)"],
     t_closed, t_dust),
    ("d. Face band + size", [f"zero {P.face_zero} voxels per face,", f"drop pieces < {P.min_size}:",
                             "medium blob (7)"], t_dust, t_big),
    ("e. Fill cavities", ["binary_fill_holes:", "the hollow ball (8)", "becomes solid"], t_big, t_filled),
    ("f. Crop, size, pad", [f"crop {c}, drop pieces", f"< {P.min_size}, pad back", "= rank2_no_interp"],
     t_filled, t_v2),
]
for i, (title, lines, prev, cur) in enumerate(V2):
    if prev is None:
        step_col(x, y, title, lines, seeds_pic)
    else:
        res = {}
        step_col(x, y, title, lines, lambda tx, ty, p=prev, q=cur: res.setdefault("d", toy_step(p, q, tx, ty)))
        text(x + col / 2, y + 26 + T + 18 + 16 * len(lines), delta(*res["d"]), "ts", "middle", width=col + 8)
    if i < len(V2) - 1:
        arrow(x + col, y + 26 + T / 2, x + col + cgap)
    x += col + cgap
y += 26 + T + 18 + 16 * 4 + 30

# ---- variant 3
add("<!-- variant 3 -->")
text(36, y + 4, "Variant 3 - rank2_full (rank2_postprocess, defaults): variant 2 steps a–e, then local surface "
     "interpolation", "h3")
y += 30
lx0, lx1 = 36 + col + cgap, 36 + 4 * col + 3 * cgap
text(lx0, y, f"Steps g–i repeat for each connected sheet ({n_comp} here), then for each group of cubes with a hole",
     "th")
add(f'<path d="M{lx0} {y + 8} v6 H{lx1} v-6" fill="none" stroke="#3d3d3a" stroke-width="1.2"/>')
y += 18
x = 36


def draw_base(tx, ty):
    raster(t_filled[:, SLICE_Y].astype(np.int8), {1: TEAL}, tx, ty, TPX)


def draw_cubes(tx, ty):
    draw_base(tx, ty)
    for z0, z1, y0, y1, x0, x1 in regions:
        if y0 <= SLICE_Y < y1:
            add(f'<rect x="{tx + x0 * TPX:.1f}" y="{ty + z0 * TPX:.1f}" width="{(x1 - x0) * TPX:.1f}" '
                f'height="{(z1 - z0) * TPX:.1f}" fill="#BA7517" fill-opacity="0.15" stroke="#854F0B" '
                f'stroke-width="1.2" stroke-dasharray="4 3"/>')


def draw_alts(tx, ty):
    lv = np.zeros(t_filled[:, SLICE_Y].shape, np.int8)
    lv[t_filled[:, SLICE_Y]] = 1
    for k, a in enumerate(t_alts, start=2):
        lv[np.pad(a, c)[:, SLICE_Y]] = k
    raster(lv, {1: "#D3D1C7", 2: "#888780", 3: "#5F5E5A", 4: DARK}, tx, ty, TPX)


V3 = [
    ("Start: steps a–e", ["the variant 2 mask", "before step f", f"({n_base} connected pieces)"], draw_base, None),
    ("g. Find holes", [f"{P.cube_size}³ cubes, overlap {P.overlap}:", f"{n_hole_cubes} of {n_cubes} cubes have",
                       "a hole (Euler number)", f"→ {len(regions)} regions (amber)"], draw_cubes, None),
    ("h. Stricter masks", ["same steps with T_low", "0.5 / 0.6 / 0.7 (darker);", "at 0.6 the bridges (4)",
                           "are gone"], draw_alts, None),
    ("i. Fit and replace", ["plane + height map, 3 voxels", "thick; keep if Dice and", "coverage ≥ 0.75, else fit",
                            "the stricter mask's pieces"], None, (t_filled, t_interp)),
    ("j. Size, pad", [f"drop pieces < {P.min_size},", "pad back = rank2_full",
                      f"{n_base} → {n_final} connected pieces"], None, (t_interp, t_v3)),
]
for i, (title, lines, draw, pair) in enumerate(V3):
    if pair is None:
        step_col(x, y, title, lines, draw)
    else:
        res = {}
        step_col(x, y, title, lines, lambda tx, ty, p=pair[0], q=pair[1]: res.setdefault("d", toy_step(p, q, tx, ty)))
        text(x + col / 2, y + 26 + T + 18 + 16 * len(lines), delta(*res["d"]), "ts", "middle", width=col + 8)
    if i < len(V3) - 1:
        arrow(x + col, y + 26 + T / 2, x + col + cgap)
    x += col + cgap
note_x = x + 10
text(note_x, y + 40, "What variant 3 did here:", "th")
for k, ln in enumerate([f"the hole (3) is filled by the fitted sheet;",
                        f"the bridged sheets (4) are split into two;",
                        f"{n_base} pieces before, {n_final} after.",
                        "",
                        "Why it helps the metric: no holes",
                        "(TopoScore) and no merged sheets (VOI).",
                        "",
                        "The fit samples points at random; this",
                        "picture fixes np.random.seed(0)."]):
    text(note_x, y + 62 + k * 17, ln, "ts", width=1380 - note_x - 16)
y += 26 + T + 18 + 16 * 5 + 24

panel_h = y - py2
svg.insert(start_idx, f'<rect x="20" y="{py2}" width="1360" height="{panel_h}" rx="8" class="panel"/>')

# ============================================================== output
add("<!-- ===================== OUTPUT ===================== -->")
arrow(700, py2 + panel_h, 700, py2 + panel_h + 30)
oy = py2 + panel_h + 30
box("out", 30, oy, 1340, 70)
text(46, oy + 24, "Output", "th")
text(46, oy + 46, f"data/cache/masks/<variant>/{VOL}.tif (uint8 0/1, same shape as the volume) and a CSV row per "
     "variant in data/cache/results/; with topometrics installed, the CSV also gets the official score.", "ts")
H = int(oy + 90)

STYLE = """
text{font-family:'DejaVu Sans Condensed','DejaVu Sans',sans-serif;fill:#2c2c2a}
.title{font-size:22px;font-weight:bold}
.h2{font-size:17px;font-weight:bold}
.h3{font-size:15px;font-weight:bold}
.th{font-size:14px;font-weight:bold}
.t{font-size:14px}
.ts{font-size:12px;fill:#5f5e5a}
.panel{fill:#FFFFFF;stroke:#D3D1C7;stroke-width:1}
.img{fill:#F1EFE8;stroke:#5F5E5A;stroke-width:1}
.sel{fill:#EEEDFE;stroke:#534AB7;stroke-width:1.2}
.out{fill:#E1F5EE;stroke:#0F6E56;stroke-width:1.2}
.arr{stroke:#3d3d3a;stroke-width:1.6;fill:none}
"""
head = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    "<title>T-1 inference and post-processing pipeline</title>",
    f"<desc>nnU-Net inference on volume {VOL}, then the three post-processing variants shown step by step on a "
    "synthetic probability volume.</desc>",
    f"<style>{STYLE}</style>",
    '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" '
    'orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" '
    'stroke-linecap="round" stroke-linejoin="round"/></marker></defs>',
    f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(head + svg + ["</svg>"]), encoding="utf-8")
print(f"wrote {OUT} ({W}x{H}, {OUT.stat().st_size / 1e6:.2f} MB); real slice y={YS}; synthetic: "
      f"hole cubes {n_hole_cubes}/{n_cubes}, regions {len(regions)}, pieces {n_base} -> {n_final}")
for w in WARN:
    print("FIT WARNING:", w)
