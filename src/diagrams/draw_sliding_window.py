"""Draw nnU-Net's sliding-window inference with Gaussian blending, for one volume of the T-1 run.

Every number is computed with nnU-Net v2's own formulas
(nnunetv2/inference/sliding_window_prediction.py: compute_steps_for_sliding_window, compute_gaussian):
  num_steps = ceil((image - patch) / (patch * step)) + 1, actual step = (image - patch) / (num_steps - 1)
  Gaussian: sigma = patch / 8, centred at patch // 2
  blend:    logits = sum(w_i * logits_i) / sum(w_i)

Usage: python draw_sliding_window.py <output.svg>
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

IMAGE, PATCH, STEP = 320, 128, 0.5   # T-1: volume 105796630, rank-2 128^3 model, tile_step_size 0.5
W = 680

# ---------------------------------------------------------------- nnU-Net formulas
num_steps = math.ceil((IMAGE - PATCH) / (PATCH * STEP)) + 1
actual_step = (IMAGE - PATCH) / (num_steps - 1)
starts = [int(round(actual_step * i)) for i in range(num_steps)]
sigma = PATCH / 8
centre = PATCH // 2
edge_weight = math.exp(-(centre ** 2) / (2 * sigma ** 2))


def coverage_1d(v: float) -> int:
    return sum(s <= v < s + PATCH for s in starts)


edges = sorted({0, IMAGE, *starts, *(s + PATCH for s in starts)})
cells = list(zip(edges[:-1], edges[1:]))
max_cover = max(coverage_1d((a + b) / 2) for a, b in cells)

# ---------------------------------------------------------------- SVG helpers
out: list[str] = []


def add(s: str) -> None:
    out.append(s)


def text(x, y, s, cls="t", anchor="start"):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    add(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}"{a}>{s}</text>')


COVER_FILL = {1: "#F1EFE8", 2: "#FAEEDA", 4: "#E8D3A8"}

# ---------------------------------------------------------------- header
y = 30
text(20, y, "Sliding window and Gaussian blending (nnU-Net inference)", "h2")
y += 20
text(20, y, f"T-1 run: volume {IMAGE}³, patch {PATCH}³, tile_step_size {STEP}, rank-2 128³ model", "ts")

# ---------------------------------------------------------------- A. patch positions
y += 35
text(20, y, "A. Where the patches go (one z-slice seen from above; z works the same way)", "h3")
y += 15
sq_x, sq_y, sq = 40, y, 230
k = sq / IMAGE

for (x0, x1) in cells:
    for (y0, y1) in cells:
        n = coverage_1d((x0 + x1) / 2) * coverage_1d((y0 + y1) / 2)
        add(f'<rect x="{sq_x + x0 * k:.1f}" y="{sq_y + y0 * k:.1f}" width="{(x1 - x0) * k:.1f}" '
            f'height="{(y1 - y0) * k:.1f}" fill="{COVER_FILL[n]}" stroke="#B4B2A9" stroke-width="0.5"/>')
        text(sq_x + (x0 + x1) / 2 * k, sq_y + (y0 + y1) / 2 * k + 5, str(n), "ts", "middle")
add(f'<rect x="{sq_x}" y="{sq_y}" width="{sq}" height="{sq}" fill="none" stroke="#5F5E5A" stroke-width="1"/>')

# first patch (solid) and the next one along x (dashed)
s0, s1 = starts[0], starts[1]
add(f'<rect x="{sq_x + s0 * k + 1:.1f}" y="{sq_y + 1:.1f}" width="{PATCH * k - 2:.1f}" height="{PATCH * k - 2:.1f}" '
    f'fill="#1D9E75" fill-opacity="0.12" stroke="#0F6E56" stroke-width="2.2" rx="2"/>')
add(f'<rect x="{sq_x + s1 * k:.1f}" y="{sq_y + 4:.1f}" width="{PATCH * k:.1f}" height="{PATCH * k - 8:.1f}" '
    f'class="win" rx="2"/>')

for v in edges:
    text(sq_x + v * k, sq_y + sq + 16, str(v), "ts", "middle")
text(sq_x + sq / 2, sq_y + sq + 33, "x (voxels)", "ts", "middle")

tx, ty = 300, sq_y + 14
lines = [
    ("th", f"Patch starts per axis: {', '.join(map(str, starts))}"),
    ("ts", f"num_steps = ceil(({IMAGE} − {PATCH}) / {int(PATCH * STEP)}) + 1 = {num_steps}"),
    ("ts", f"actual step = ({IMAGE} − {PATCH}) / ({num_steps} − 1) = {actual_step:g} voxels"),
    ("gap", ""),
    ("th", f"{num_steps} × {num_steps} × {num_steps} = {num_steps ** 3} patches per volume"),
    ("ts", f"Each patch overlaps the next by {PATCH - int(actual_step)} voxels (half a patch)."),
    ("gap", ""),
    ("ts", "Solid teal: first patch. Dashed teal: the next patch,"),
    ("ts", f"one step ({actual_step:g} voxels) further along x."),
    ("gap", ""),
    ("ts", "Cell numbers: how many patches cover that area in"),
    ("ts", f"this top view (1, 2 or 4). In 3D: up to {max_cover}×{max_cover}×{max_cover} = {max_cover ** 3}."),
]
for cls, s in lines:
    if cls == "gap":
        ty += 8
        continue
    text(tx, ty, s, cls)
    ty += 18 if cls == "ts" else 20

y = sq_y + sq + 60

# ---------------------------------------------------------------- B. Gaussian weights
text(20, y, "B. Gaussian weight of each patch, along x", "h3")
y += 15
px0, px1 = 40, 640
kx = (px1 - px0) / IMAGE
plot_h = 110
base = y + plot_h
add(f'<line x1="{px0}" y1="{base}" x2="{px1}" y2="{base}" stroke="#888780" stroke-width="0.8"/>')
for i, s in enumerate(starts):
    pts = []
    for v in range(s, s + PATCH + 1):
        wv = math.exp(-((v - s - centre) ** 2) / (2 * sigma ** 2))
        pts.append(f"{px0 + v * kx:.1f},{base - wv * (plot_h - 10):.1f}")
    width = 2.4 if i == 0 else 1.4
    dash = "" if i == 0 else ' stroke-dasharray="5 3"'
    add(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#0F6E56" stroke-width="{width}"{dash}/>')
    text(px0 + (s + centre) * kx, base - plot_h + 2, f"patch {i + 1}", "ts", "middle")

# patch extents under the axis
bar_y = base + 8
for i, s in enumerate(starts):
    yy = bar_y + i * 9
    add(f'<rect x="{px0 + s * kx:.1f}" y="{yy}" width="{PATCH * kx:.1f}" height="6" rx="2" '
        f'fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.8"/>')
ticks_y = bar_y + num_steps * 9 + 12
for v in edges:
    add(f'<line x1="{px0 + v * kx:.1f}" y1="{base}" x2="{px0 + v * kx:.1f}" y2="{ticks_y - 10}" class="leader"/>')
    text(px0 + v * kx, ticks_y + 2, str(v), "ts", "middle")
y = ticks_y + 24
text(20, y, f"sigma = {PATCH} / 8 = {sigma:g} voxels, centred at {centre} inside each patch.", "ts")
y += 18
text(20, y, f"At a patch's own edge the weight is e^(−{centre}² / (2·{sigma:g}²)) = {edge_weight:.4f} of its centre weight.", "ts")

# ---------------------------------------------------------------- C. blending
y += 35
text(20, y, "C. Blending (done on the logits, before softmax)", "h3")
y += 15
box_h = 132
add(f'<g class="c-teal"><rect x="20" y="{y}" width="{W - 40}" height="{box_h}" rx="6" fill="#E1F5EE" '
    f'stroke="#0F6E56" stroke-width="1"/></g>')
yy = y + 24
text(36, yy, "For each voxel v, over the patches i that cover it:", "t")
yy += 26
text(W / 2, yy, "logit(v) = Σ w_i(v) · logit_i(v)  /  Σ w_i(v)", "th", "middle")
yy += 26
text(36, yy, "Near a patch's edge its weight is tiny, so each voxel is decided mostly by the patch whose", "ts")
yy += 17
text(36, yy, "centre is nearest. At the volume border only one patch covers a voxel; dividing by Σ w keeps it.", "ts")
yy += 17
text(36, yy, "Then softmax over the 2 classes → P(background), P(sheet); we cache P(sheet) as float16.", "ts")
y += box_h + 20

H = int(y)

STYLE = """
text{font-family:'DejaVu Sans Condensed','DejaVu Sans',sans-serif;fill:#2c2c2a}
.h2{font-size:17px;font-weight:bold}
.h3{font-size:15px;font-weight:bold}
.th{font-size:14px;font-weight:bold}
.t{font-size:14px}
.ts{font-size:12px;fill:#5f5e5a}
.c-teal .th,.c-teal .t{fill:#085041}
.arr{stroke:#3d3d3a;stroke-width:1.5;fill:none}
.leader{stroke:#888780;stroke-width:0.7;stroke-dasharray:3 3;fill:none}
.win{fill:#1D9E75;fill-opacity:0.10;stroke:#1D9E75;stroke-width:1.5;stroke-dasharray:6 4}
"""

svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    "<title>Sliding window and Gaussian blending</title>",
    f"<desc>How nnU-Net tiles a {IMAGE}³ volume into {num_steps ** 3} overlapping {PATCH}³ patches and blends their "
    "logits with Gaussian weights.</desc>",
    f"<style>{STYLE}</style>",
    '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" '
    'orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" '
    'stroke-linecap="round" stroke-linejoin="round"/></marker></defs>',
    f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
    *out,
    "</svg>",
]

dest = Path(sys.argv[1])
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text("\n".join(svg), encoding="utf-8")
print(f"wrote {dest} ({W}x{H}, {dest.stat().st_size} bytes)")
