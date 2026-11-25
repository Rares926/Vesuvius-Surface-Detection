"""Options sheet for redrawing the T-1 pipeline diagram: 2-4 ways to show each step, labelled A, B, C, D.

Computed pictures use the T-1 volume (CT, label, cached probability map) or a small synthetic volume run through the
project's own post-processing code; sketches are marked as sketches.

Usage (from the project root): python src/diagrams/draw_pipeline_options.py <output.svg> [volume_id] [model_name]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import scipy.ndimage as ndi
import tifffile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common import image_path, label_path, prob_path  # noqa: E402

OUT = Path(sys.argv[1])
VOL = sys.argv[2] if len(sys.argv) > 2 else "105796630"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "m128"

ct = tifffile.imread(image_path(VOL))
gt = tifffile.imread(label_path(VOL))
prob = np.load(prob_path(MODEL, VOL)).astype(np.float32)
PLANS = {"lo": 0.0, "hi": 212.0, "mean": 87.54424285888672, "std": 47.74376678466797}   # models/v2 plans.json
PATCH, IMAGE, STEP = 128, 320, 0.5

W = 1400
svg: list[str] = []
TEAL, PURPLE, AMBER, CORAL, LGRAY, DARK = "#1D9E75", "#534AB7", "#BA7517", "#993C1D", "#D3D1C7", "#2C2C2A"


def add(s):
    svg.append(s)


def text(x, y, s, cls="ts", anchor="start"):
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    add(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}"{a}>{escape(s)}</text>')


def lines(x, y, rows, cls="ts", step=16, anchor="start"):
    for i, r in enumerate(rows):
        text(x, y + i * step, r, cls, anchor)


def raster(codes, palette, x0, y0, px=1.0):
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
    add(f'<rect x="{x0}" y="{y0}" width="{codes.shape[1] * px:.1f}" height="{codes.shape[0] * px:.1f}" '
        f'fill="none" stroke="#888780" stroke-width="1"/>')


def arrow(x1, y1, x2, y2=None):
    y2 = y1 if y2 is None else y2
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="arr" marker-end="url(#arrow)"/>')


def option(x, y, w, h, letter, title):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#FFFFFF" stroke="#D3D1C7"/>')
    add(f'<circle cx="{x + 18}" cy="{y + 18}" r="11" fill="#EEEDFE" stroke="#534AB7"/>')
    text(x + 18, y + 23, letter, "th", "middle")
    text(x + 36, y + 23, title, "th")


def section(y, title, sub):
    text(30, y, title, "h2")
    text(30, y + 20, sub, "ts")
    return y + 34


# ================================================================ header
y = 38
text(30, y, "Options for the redrawn pipeline diagram (T-1): pick one letter per row", "title")
y += 22
text(30, y, f"Pictures marked 'computed' are drawn from real data (volume {VOL}) or from a small synthetic volume run "
     "through the project's code; 'sketch' means drawn by hand, like the figures in rank 2's write-up.", "ts")
y += 30

# ================================================================ 1. normalise
y = section(y, "1. Normalise: why clip to 0 … 212",
            "Clip values = 0.5 % and 99.5 % percentiles of training CT values on voxels with label > 0 "
            "(sheet, and unlabelled), from the model's plans.json.")
oh = 230
ow = 660
# A: histogram, computed
option(30, y, ow, oh, "A", "Histogram with the clip lines (computed, this volume)")
hx, hy, hw, hh = 60, y + 50, 380, 108
vals_all = np.bincount(ct.ravel(), minlength=256)[:256].astype(float)
vals_fg = np.bincount(ct[gt == 1].ravel(), minlength=256)[:256].astype(float)
for vals, colour, op in ((vals_all, "#888780", 0.5), (vals_fg, CORAL, 0.9)):
    v = vals / vals.max()
    pts = " ".join(f"{hx + i * hw / 255:.1f},{hy + hh - v[i] * hh:.1f}" for i in range(256))
    add(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="1.5" opacity="{op}"/>')
add(f'<rect x="{hx + 212 * hw / 255:.1f}" y="{hy}" width="{43 * hw / 255:.1f}" height="{hh}" fill="{AMBER}" '
    f'fill-opacity="0.18"/>')
for v in (0, 212):
    xv = hx + v * hw / 255
    add(f'<line x1="{xv:.1f}" y1="{hy - 4}" x2="{xv:.1f}" y2="{hy + hh}" stroke="#854F0B" stroke-width="1.4" '
        f'stroke-dasharray="4 3"/>')
add(f'<line x1="{hx}" y1="{hy + hh}" x2="{hx + hw}" y2="{hy + hh}" stroke="#5F5E5A"/>')
for v in (0, 64, 128, 212, 255):
    text(hx + v * hw / 255, hy + hh + 15, str(v), "ts", "middle")
above_all = float((ct > 212).mean())
above_fg = float((ct[gt == 1] > 212).mean())
lines(hx + hw + 20, hy + 6, [
    "gray: all voxels of the volume",
    "coral: sheet voxels (label 1)",
    "amber: clipped to 212",
    "",
    f"above 212: {above_all:.2%} of all voxels,",
    f"{above_fg:.2%} of sheet voxels",
])
lines(48, y + oh - 28, ["Why: the brightest values are rare and far from typical papyrus. Clipping them stops a few",
                        "outliers from setting the scale, then (v − 87.5) / 47.7 centres the rest around 0."])
# B: value ramp
bx = 30 + ow + 20
option(bx, y, ow, oh, "B", "Value ramp: what each step does to a voxel value (sketch with real numbers)")
zlo = (PLANS["lo"] - PLANS["mean"]) / PLANS["std"]
zhi = (PLANS["hi"] - PLANS["mean"]) / PLANS["std"]
rx, rw = bx + 40, 560
rows = [("raw uint8", "0", "255", 1.0), ("clip", "0", "212 (213–255 → 212)", 212 / 255),
        ("(v − 87.5) / 47.7", f"{zlo:.2f}", f"{zhi:.2f}", 212 / 255)]
for i, (lab, lo, hi, frac) in enumerate(rows):
    ry = y + 62 + i * 52
    text(rx, ry - 6, lab, "th")
    add(f'<rect x="{rx}" y="{ry}" width="{rw * frac:.1f}" height="12" fill="#C9C7BE" stroke="#5F5E5A"/>')
    if frac < 1:
        add(f'<rect x="{rx + rw * frac:.1f}" y="{ry}" width="{rw * (1 - frac):.1f}" height="12" fill="{AMBER}" '
            f'fill-opacity="0.25" stroke="#854F0B" stroke-dasharray="3 2"/>')
    text(rx, ry + 27, lo, "ts")
    text(rx + rw * frac, ry + 27, hi, "ts", "end")
y += oh + 30

# ================================================================ 2. sliding window
y = section(y, "2. Sliding window: stride = patch × tile_step_size = 128 × 0.5 = 64",
            "nnU-Net then spreads the starts evenly: num_steps = ceil((320 − 128) / 64) + 1 = 4, "
            "starts 0, 64, 128, 192.")
oh = 190
num_steps = math.ceil((IMAGE - PATCH) / (PATCH * STEP)) + 1
starts = [int(round((IMAGE - PATCH) / (num_steps - 1) * i)) for i in range(num_steps)]
# A: 1D strip
option(30, y, ow, oh, "A", "One axis as a strip (computed starts)")
sx, sw = 60, 560
k = sw / IMAGE
add(f'<rect x="{sx}" y="{y + 50}" width="{sw}" height="14" fill="#F1EFE8" stroke="#5F5E5A"/>')
text(sx + sw + 8, y + 62, "320", "ts")
for i, s in enumerate(starts):
    yy = y + 76 + i * 18
    add(f'<rect x="{sx + s * k:.1f}" y="{yy}" width="{PATCH * k:.1f}" height="12" rx="2" fill="#E1F5EE" '
        f'stroke="#0F6E56"/>')
    text(sx + (s + PATCH / 2) * k, yy + 10, f"patch {i + 1}: {s}–{s + PATCH}", "ts", "middle")
    if i:
        add(f'<line x1="{sx + starts[i - 1] * k:.1f}" y1="{yy - 2}" x2="{sx + s * k - 2:.1f}" y2="{yy - 2}" '
            f'class="arr" marker-end="url(#arrow)"/>')
text(48, y + oh - 14, "Each arrow is one stride (64 = half a patch); the same on z, y and x → 4 × 4 × 4 = 64 patches.")
# B: reuse the coverage grid
option(bx, y, ow, oh, "B", "Top view with coverage counts (the grid from sliding_window_gaussian.svg)")
gx, gy, gs = bx + 40, y + 40, 135
kk = gs / IMAGE
edges = sorted({0, IMAGE, *starts, *(s + PATCH for s in starts)})
cov = lambda v: sum(s <= v < s + PATCH for s in starts)  # noqa: E731
fills = {1: "#F1EFE8", 2: "#FAEEDA", 4: "#E8D3A8"}
for a0, a1 in zip(edges[:-1], edges[1:]):
    for b0, b1 in zip(edges[:-1], edges[1:]):
        n = cov((a0 + a1) / 2) * cov((b0 + b1) / 2)
        add(f'<rect x="{gx + a0 * kk:.1f}" y="{gy + b0 * kk:.1f}" width="{(a1 - a0) * kk:.1f}" '
            f'height="{(b1 - b0) * kk:.1f}" fill="{fills[n]}" stroke="#B4B2A9" stroke-width="0.5"/>')
add(f'<rect x="{gx}" y="{gy}" width="{PATCH * kk:.1f}" height="{PATCH * kk:.1f}" fill="#1D9E75" fill-opacity="0.12" '
    f'stroke="#0F6E56" stroke-width="2"/>')
add(f'<rect x="{gx + 64 * kk:.1f}" y="{gy + 2}" width="{PATCH * kk:.1f}" height="{PATCH * kk - 4:.1f}" class="win"/>')
lines(gx + gs + 25, gy + 16, ["solid: patch 1; dashed: patch 2,", "one stride (64) to the right",
                              "", "cell shade: 1, 2 or 4 patches", "cover that area (8 in 3D)"])
y += oh + 30

# ================================================================ 3. mirroring
y = section(y, "3. Mirroring (off in this run)", "Each patch is predicted 8 times (flips over every combination of z, y, x), "
            "each prediction is flipped back, and the 8 logits are averaged.")
oh = 150
option(30, y, ow, oh, "A", "Text only (as now)")
lines(48, y + 54, ["8 flips per patch, flipped back and averaged;", "off here (8× faster)."], "t", 20)
option(bx, y, ow, oh, "B", "Small flip icons (sketch)")
flips = ["", "x", "y", "z", "xy", "xz", "yz", "xyz"]
for i, f in enumerate(flips):
    fx, fy = bx + 30 + i * 52, y + 50
    add(f'<rect x="{fx}" y="{fy}" width="40" height="40" rx="3" fill="#EEEDFE" stroke="#534AB7"/>')
    sxs = -1 if "x" in f else 1
    sys_ = -1 if "y" in f else 1
    cx, cy = fx + 20, fy + 27
    add(f'<text x="0" y="0" class="h2" text-anchor="middle" transform="translate({cx} {cy - 7}) scale({sxs} {sys_}) '
        f'translate(0 7)">F</text>')
    text(fx + 20, fy + 56, f or "none", "ts", "middle")
arrow(bx + 30 + 8 * 52, y + 70, bx + 30 + 8 * 52 + 30)
lines(bx + 30 + 8 * 52 + 38, y + 66, ["flip back,", "average"])
text(bx + 18, y + oh - 14, "(z flips cannot be drawn on a flat icon; the label says which axes are flipped)")
y += oh + 30

# ================================================================ 4. gaussian
y = section(y, "4. Gaussian blend", "Weight peaks at the patch centre (sigma = 128 / 8 = 16); each voxel gets "
            "Σ w·logit / Σ w over the patches covering it.")
oh = 200
option(30, y, ow, oh, "A", "Curves along one axis (part B of sliding_window_gaussian.svg)")
cx0, cw, cb, ch = 60, 580, y + 160, 100
kx = cw / IMAGE
add(f'<line x1="{cx0}" y1="{cb}" x2="{cx0 + cw}" y2="{cb}" stroke="#888780"/>')
for i, s in enumerate(starts):
    pts = " ".join(f"{cx0 + v * kx:.1f},{cb - math.exp(-((v - s - 64) ** 2) / (2 * 16 ** 2)) * ch:.1f}"
                   for v in range(s, s + PATCH + 1))
    dash = "" if i == 0 else ' stroke-dasharray="5 3"'
    add(f'<polyline points="{pts}" fill="none" stroke="#0F6E56" stroke-width="{2.2 if i == 0 else 1.3}"{dash}/>')
for v in edges:
    text(cx0 + v * kx, cb + 15, str(v), "ts", "middle")
option(bx, y, ow, oh, "B", "Weight map of one patch, then the blended weight (computed)")
g1 = np.exp(-((np.arange(PATCH) - 64) ** 2) / (2 * 16 ** 2))
patch_w = np.outer(g1, g1)
lv = np.clip((patch_w * 6).astype(int), 0, 5)
shades = {1: "#E1F5EE", 2: "#B6E3D2", 3: "#7CC9AE", 4: "#3FAE8A", 5: "#0F6E56"}
raster(lv[::1, ::1], shades, bx + 30, y + 44, px=1.1)
text(bx + 30 + 70, y + 44 + 128 * 1.1 + 16, "one 128² patch", "ts", "middle")
arrow(bx + 190, y + 115, bx + 230)
total = np.zeros(IMAGE)
for s in starts:
    total[s:s + PATCH] += g1
tw = np.outer(total, total)
lv2 = np.clip((tw / tw.max() * 6).astype(int), 0, 5)
raster(lv2[::2, ::2], shades, bx + 245, y + 44, px=0.88)
text(bx + 245 + 70, y + 44 + 160 * 0.88 + 16, "Σ w over the 320² slice", "ts", "middle")
lines(bx + 420, y + 64, ["darker = more weight;", "Σ w is low near the faces", "and between patch centres,",
                         "but the division by Σ w", "turns it into a proper", "weighted average"])
y += oh + 30

# ================================================================ 5. post-processing pictures
y = section(y, "5. Post-processing pictures: what each step's picture shows",
            "Shown here for one step, hysteresis (seeds ≥ 0.83 grow through P ≥ 0.2); the chosen style is used for "
            "every step.")
oh = 270
qw = 325

# A: full real slice
option(30, y, qw, oh, "A", "Full real slice (as now)")
YS = 252
p = prob[:, YS, :]
lvp = (p >= 0.2).astype(int) + (p >= 0.83)
raster(lvp[::2, ::2], {1: LGRAY, 2: DARK}, 30 + 85, y + 44)
lines(48, y + 226, ["computed; whole 320² slice at 1 px per 2×2", "voxels: context, but small effects vanish"])

# B: zoomed real crop, where hysteresis removes the most
strong, weak = prob >= 0.83, prob >= 0.2
hyst = ndi.binary_propagation(strong, mask=weak, structure=ndi.generate_binary_structure(3, 3))
gone = weak & ~hyst
H = 48
# the 48x48 window (in an x-z plane, away from the faces) with the most removed voxels
dens = ndi.uniform_filter(gone.astype(np.float32), size=(H, 1, H), mode="constant")
dens_sheet = ndi.uniform_filter(hyst.astype(np.float32), size=(H, 1, H), mode="constant")
m = H // 2 + 8
for d in (dens, dens_sheet):
    d[:m], d[-m:], d[:, :m], d[:, -m:], d[:, :, :m], d[:, :, -m:] = 0, 0, 0, 0, 0, 0
score = dens * (dens_sheet > 0.02)                              # the window must also contain sheet
dens = score if score.max() > 0 else dens
lab_gone, n_gone = ndi.label(gone, structure=np.ones((3, 3, 3)))
biggest = 1 + int(np.argmax(ndi.sum(gone, lab_gone, range(1, n_gone + 1))))
zc, yc, xc = (int(round(v)) for v in ndi.center_of_mass(lab_gone == biggest))   # centre of the largest removed blob
z0, x0 = int(np.clip(zc - H // 2, 0, IMAGE - H)), int(np.clip(xc - H // 2, 0, IMAGE - H))
crop = np.zeros((H, H), np.int8)
w_ = weak[z0:z0 + H, yc, x0:x0 + H]
h_ = hyst[z0:z0 + H, yc, x0:x0 + H]
crop[h_] = 1
crop[w_ & ~h_] = 2
option(30 + qw + 13, y, qw, oh, "B", "Zoomed real crop where the step acts")
raster(crop, {1: TEAL, 2: AMBER}, 30 + qw + 13 + 90, y + 50, px=3)
lines(30 + qw + 31, y + 222, [f"computed: 48² around the largest removed blob;",
                              f"hysteresis removes only {int(gone.sum())} voxels here, so",
                              "real crops can show very little"])

# C: synthetic toy run through the real code
Z, Y, X = 48, 6, 48
toy = np.zeros((Z, Y, X), np.float32)
zz, xx = np.mgrid[0:Z, 0:X]
sheet = np.abs(xx - (14 + 3 * np.sin(zz / 8))) < 1.5
toy[:, :, :][..., ] = 0
pslice = np.where(sheet, 0.95, 0.0)
pslice[(zz > 18) & (zz < 28) & sheet] = 0.35                    # weak stretch inside a sheet: kept
sheet2 = np.abs(xx - 33) < 1.5
pslice = np.maximum(pslice, np.where(sheet2, 0.9, 0.0))
blob = (zz - 38) ** 2 + (xx - 24) ** 2 < 9
pslice[blob] = 0.4                                              # weak blob, no seed: removed
toy[:] = pslice[:, None, :]
t_strong, t_weak = toy >= 0.83, toy >= 0.2
t_h = ndi.binary_propagation(t_strong, mask=t_weak, structure=ndi.generate_binary_structure(3, 3))
mid = Y // 2
qx = 30 + 2 * (qw + 13)
option(qx, y, qw, oh, "C", "Synthetic example (computed)")
lv_in = (toy[:, mid] >= 0.2).astype(int) + (toy[:, mid] >= 0.83)
raster(lv_in, {1: LGRAY, 2: DARK}, qx + 20, y + 60, px=2.7)
arrow(qx + 20 + 48 * 2.7 + 4, y + 125, qx + 20 + 48 * 2.7 + 30)
out = np.zeros((Z, X), np.int8)
out[t_h[:, mid]] = 1
out[t_weak[:, mid] & ~t_h[:, mid]] = 2
raster(out, {1: TEAL, 2: AMBER}, qx + 20 + 48 * 2.7 + 34, y + 60, px=2.7)
lines(qx + 18, y + 226, ["a designed P(sheet): weak stretch in a sheet (kept),",
                         "weak blob with no seed (removed)"])

# D: sketch in rank-2 style
dx = 30 + 3 * (qw + 13)
option(dx, y, qw, oh, "D", "Sketch, rank-2 write-up style")
for ox in (dx + 30, dx + 190):
    add(f'<rect x="{ox}" y="{y + 60}" width="120" height="120" fill="#FFFFFF" stroke="#2c2c2a" stroke-width="2"/>')
add(f'<path d="M{dx + 55} {y + 65} q 15 55 -5 110" fill="none" stroke="#2c2c2a" stroke-width="5"/>')
add(f'<path d="M{dx + 105} {y + 65} q -10 50 5 110" fill="none" stroke="#2c2c2a" stroke-width="5" '
    f'stroke-dasharray="14 10"/>')
add(f'<circle cx="{dx + 82}" cy="{y + 150}" r="7" fill="#888780"/>')
add(f'<path d="M{dx + 215} {y + 65} q 15 55 -5 110" fill="none" stroke="#2c2c2a" stroke-width="5"/>')
add(f'<path d="M{dx + 265} {y + 65} q -10 50 5 110" fill="none" stroke="#2c2c2a" stroke-width="5"/>')
arrow(dx + 155, y + 120, dx + 185)
lines(dx + 18, y + 226, ["hand-drawn: dashed = weak part, gray dot =", "weak blob; no real data, no numbers"])
y += oh + 30

# ================================================================ 6. diff colours
y = section(y, "6. Colours for a step's effect", "Applies to every post-processing picture.")
oh = 120
option(30, y, ow, oh, "A", "Mask + added (purple) + removed (amber) (as now)")
lines(48, y + 56, ["shows exactly what the step changed, and the result;", "needs the legend"])
option(bx, y, ow, oh, "B", "Before and after side by side, no change colours")
lines(bx + 18, y + 56, ["two pictures per step, plain teal masks; simpler to read,", "but small changes are hard to spot"])
y += oh + 30

Hh = int(y)
STYLE = """
text{font-family:'DejaVu Sans Condensed','DejaVu Sans',sans-serif;fill:#2c2c2a}
.title{font-size:22px;font-weight:bold}
.h2{font-size:17px;font-weight:bold}
.h3{font-size:15px;font-weight:bold}
.th{font-size:14px;font-weight:bold}
.t{font-size:14px}
.ts{font-size:12px;fill:#5f5e5a}
.arr{stroke:#3d3d3a;stroke-width:1.5;fill:none}
.win{fill:#1D9E75;fill-opacity:0.10;stroke:#1D9E75;stroke-width:1.5;stroke-dasharray:6 4}
"""
head = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" viewBox="0 0 {W} {Hh}">',
    "<title>Pipeline diagram options</title>",
    "<desc>Alternative ways to draw each step of the T-1 inference and post-processing diagram.</desc>",
    f"<style>{STYLE}</style>",
    '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" '
    'orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" '
    'stroke-linecap="round" stroke-linejoin="round"/></marker></defs>',
    f'<rect width="{W}" height="{Hh}" fill="#FFFFFF"/>',
]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(head + svg + ["</svg>"]), encoding="utf-8")
print(f"wrote {OUT} ({W}x{Hh}, {OUT.stat().st_size / 1e6:.2f} MB); above 212: all {above_all:.3%}, "
      f"sheet {above_fg:.3%}; crop at z={z0}, y={yc}, x={x0}")
