"""Draw the pipeline diagrams of the top Kaggle solutions (planner task R-1).

Usage: python src/diagrams/draw_solution_pipelines.py <output.svg> <mode>
  mode: rank1 | rank2 | rank3 | rank2_options

All values come from each team's write-up and final notebook (read 2026-11-21); see the md files in
.claude/context/R-1_top3_solutions/context/. Layout is computed from the content: box widths split the row,
box heights follow their number of lines, panel heights follow their rows.
"""
import sys
from pathlib import Path
from xml.sax.saxutils import escape

W = 1400
X0, X1 = 20, 1380            # panel span
PAD = 20                     # inner padding of a panel

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
.net{fill:#FFFFFF;stroke:#3d3d3a;stroke-width:1}
.sel{fill:#EEEDFE;stroke:#534AB7;stroke-width:1.2}
.out{fill:#E1F5EE;stroke:#0F6E56;stroke-width:1.2}
.pp{fill:#FAEEDA;stroke:#854F0B;stroke-width:1.2}
.gt{fill:#FAECE7;stroke:#993C1D;stroke-width:1.2}
.arr{stroke:#3d3d3a;stroke-width:1.6;fill:none}
.c-img{fill:#444441}.c-net{fill:#2c2c2a}.c-sel{fill:#3C3489}.c-out{fill:#085041}.c-pp{fill:#633806}.c-gt{fill:#712B13}
"""

LEGEND = [('img', 'Data / input'), ('net', 'Model / training'), ('sel', 'Inference, tiling and TTA'),
          ('out', 'Probabilities, masks, results'), ('pp', 'Post-processing step')]

CHAR_W = {'th': 7.9, 't': 7.4, 'ts': 6.3, 'h3': 8.6}   # rough width per character, for the fit check
WARNINGS = []


def fits(text, cls, width, where):
    need = len(text) * CHAR_W[cls]
    if need > width:
        WARNINGS.append(f'{where}: "{text}" ~{need:.0f}px > {width:.0f}px')


# ----------------------------------------------------------------------------------------------
# Specs. A box is (class, title, [lines]). A row is dict(boxes=[...], arrows=[bool]*(n-1) or bool).
# ----------------------------------------------------------------------------------------------
SPECS = {}

SPECS['rank1'] = dict(
    title='Rank 1 - Vesuvius Team (private LB 0.627): nnU-Net ensemble + per-sheet post-processing',
    source=['Sources: 1st place write-up and final notebook "nnUNet 4-MODEL 7+5+2+1 - final submit" (tonylica)',
            'Every number below is read from the final notebook unless marked "write-up".'],
    panels=[
        dict(heading='Step 1 - Training data and setup', rows=[
            dict(boxes=[('img', 'Training data', ['all train volumes, fold "all"', 'labels 0 / 1 / 2',
                                                  'label 2 = nnU-Net ignore label']),
                        ('net', 'nnU-Net v2, 3d_fullres', ['3D patches at spacing 1.0',
                                                           'default loss: Dice + CE,', 'deep supervision, batch 2']),
                        ('net', 'Start point', ['Jirka Borovec public nnU-Net', 'notebook (see U-2 pipeline_3d)'])],
                 arrows=True)]),
        dict(heading='Step 2 - Four residual-encoder nnU-Nets (the final ensemble)', rows=[
            dict(boxes=[('net', 'M1 - baseline', ['ResEnc M plans, patch 128', 'from scratch, 4000 epochs',
                                                 'checkpoint: final', 'weight 0.12']),
                        ('net', 'M2 - fine-tuned', ['ResEnc XL plans, patch 192', 'fine-tuned from M1, 250 ep',
                                                    'checkpoint: best', 'weight 0.28']),
                        ('net', 'M5 - fine-tuned', ['ResEnc XL plans, patch 256', 'fine-tuned from M1, 250 ep',
                                                    'checkpoint: final', 'weight 0.18']),
                        ('net', 'M7 - from scratch', ['ResEnc L plans, patch 192', 'from scratch, 4000 epochs',
                                                      'checkpoint: final', 'weight 0.42'])],
                 arrows=False)],
             foot='Long training with a small batch beat fewer epochs with a large batch (write-up).'),
        dict(heading='Step 3 - Inference, for each model and each test volume', rows=[
            dict(boxes=[('img', 'Test volume', ['e.g. (320, 314, 314)', 'uint8']),
                        ('sel', 'Sliding window', ['tile = the model\'s patch size', 'tile_step_size 0.5',
                                                   'windows: M1 64, M2/M7 27, M5 8']),
                        ('sel', 'Mirroring TTA', ['flips on axes (0, 1, 2)', 'averaged']),
                        ('out', 'Softmax probability', ['surface channel', 'saved per model as .npz'])],
                 arrows=True)]),
        dict(heading='Step 4 - Fuse the 4 models and threshold', rows=[
            dict(boxes=[('out', 'Weighted probability sum', ['p = 0.42 M7 + 0.28 M2', '  + 0.18 M5 + 0.12 M1']),
                        ('out', 'Threshold', ['mask = p >= 0.26']),
                        ('out', 'Binary mask', ['same shape as the volume'])],
                 arrows=True)],
             foot='Write-up: fusing logits and a higher threshold (0.35-0.40) would have scored better on the private LB.'),
        dict(heading='Step 5 - Post-processing on the whole mask', rows=[
            dict(boxes=[('pp', 'Zero the border', ['3 voxels on every face']),
                        ('pp', 'Remove non-sheets', ['remove_small_objects', 'min size 20 000 voxels']),
                        ('pp', 'Label sheets', ['26-connected components', 'up to 40 sheets'])],
                 arrows=True)]),
        dict(heading='Step 6 - Post-processing per sheet', loop='Repeat for each sheet (connected component):',
             rows=[
                 dict(boxes=[('pp', 'Projection axis', ['axis with the largest', 'projected sheet area']),
                             ('pp', 'Oriented closing', ['ellipsoid 7 x 7 x 7 along', 'that axis; keep original']),
                             ('pp', 'Height-map patching', ['fill gaps by interpolating', 'along x and y, averaged',
                                                            'only patches < 64 x 64', 'then closing again'])],
                      arrows=True),
                 dict(boxes=[('pp', 'Plug 1-voxel holes', ['2 x 2 x 2 lookup table', '(256 cases): make it',
                                                           '6-connected watertight']),
                             ('pp', 'Check holes', ['holes from Euler number', 'worse than unpatched?',
                                                    '-> keep the unpatched sheet']),
                             ('pp', 'Write the sheet back', ['skip voxels touching sheets', 'already written',
                                                             '(keeps sheets apart)'])],
                      arrows=True)]),
        dict(heading='Step 7 - Final steps and submission', rows=[
            dict(boxes=[('pp', 'Fill cavities', ['binary_fill_holes', 'on the whole mask']),
                        ('pp', 'Zero the border again', ['3 voxels on every face']),
                        ('out', 'Submission', ['uint8 0 / 1 .tif per volume', 'zipped -> submission.zip'])],
                 arrows=True)]),
        dict(heading='Effect of post-processing, cumulative (write-up, best ensemble, private LB)', rows=[
            dict(boxes=[('out', 'No post-proc', ['0.596']), ('out', '+ remove small', ['0.614']),
                        ('out', '+ plug holes', ['0.622']), ('out', '+ patch holes', ['0.625']),
                        ('out', '+ closing', ['0.627']), ('out', '+ fill holes', ['0.627'])],
                 arrows=True)]),
    ])

SPECS['rank2'] = dict(
    title='Rank 2 - risk of overfitting (private LB 0.622): 2 nnU-Nets + local surface interpolation',
    source=['Sources: 2nd place write-up "A postprocessing win" and final notebook "local interpolation interference".',
            'Drawn as the final notebook code (option B). Where the write-up differs, see section 9 of the md.'],
    panels=[
        dict(heading='Step 1 - Training data and setup', rows=[
            dict(boxes=[('img', 'Training data', ['all train volumes, fold "all"', 'label 2 = ignore',
                                                  'first 128 model on 80 % -> tune']),
                        ('net', 'nnU-Net v2, 3d_fullres', ['vanilla ResEnc trainer', 'spacing 1.0, batch 2',
                                                           'default loss: Dice + CE']),
                        ('net', 'Losses tried, dropped', ['Skeaw-BorT: no fit here', 'medial surface: Topo up,',
                                                          'Surface Dice down'])],
                 arrows=[True, False])],
             foot='Their idea: train the model for Surface Dice, then fix the topology in post-processing.'),
        dict(heading='Step 2 - Two models (the final ensemble)', rows=[
            dict(boxes=[('net', 'Model A', ['patch 128 x 128 x 128', '1000 epochs, fold all', 'weight 0.4']),
                        ('net', 'Model B', ['patch 160 x 160 x 160', '1000 epochs, fold all', 'weight 0.6'])],
                 arrows=False)],
             foot='Weights tuned "visually". Thresholds were tuned on model A only; B finished in the last days.'),
        dict(heading='Step 3 - Inference (one model per T4 GPU, in parallel)', rows=[
            dict(boxes=[('img', 'Test volume', ['e.g. (320, 314, 314)', 'uint8']),
                        ('sel', 'Sliding window', ['tile = the model\'s patch size', 'tile_step_size 0.5',
                                                   'Gaussian weighting']),
                        ('sel', 'Mirroring TTA', ['flips, averaged', '(no rotation in the code)']),
                        ('out', 'Fused probability', ['p = 0.6 pB + 0.4 pA', 'surface channel'])],
                 arrows=True)]),
        dict(heading='Step 4 - Four masks from one probability map (topo_postprocess)',
             loop='Repeat for T_low = 0.2 (base), 0.5, 0.6, 0.7 (alternatives), with T_high = 0.83:',
             rows=[
                 dict(boxes=[('pp', '3D hysteresis', ['seeds p >= T_high', 'grow through p >= T_low',
                                                      '26-connected']),
                             ('pp', 'Closing along z', ['anisotropic: z-radius 1,', 'xy-radius 0']),
                             ('pp', 'Remove dust', ['components < 100 voxels']),
                             ('pp', 'Zero the border', ['3 voxels per face', '(the known label-2 band)']),
                             ('pp', 'Remove small', ['components < 1000 voxels', 'base also: fill_holes'])],
                      arrows=True)],
             foot='Higher T_low keeps less, so sheets merged in the base mask separate in the alternatives '
                  '(each one is a subset of the base).'),
        dict(heading='Step 5 - Local interpolation of sheets (the core idea), masks cropped by 3 voxels',
             loop='Repeat for each connected component (sheet) of the base mask:',
             rows=[
                 dict(boxes=[('pp', 'Find holes', ['cubes 80, overlap 40', 'Euler number per cube',
                                                   'negative -> hole / tunnel']),
                             ('pp', 'Group cubes', ['overlapping / adjacent', 'cubes with holes', 'form regions']),
                             ('pp', 'Plane fit (PCA)', ['SVD of the voxels', '-> in-plane axes (u, v)',
                                                        'and height w'])],
                      arrows=True),
                 dict(boxes=[('pp', 'Height grid', ['griddata of w over (u, v)', 'grid resolution 80',
                                                    'flood-remove from edges']),
                             ('pp', 'Rasterise', ['back into 3D', 'sheet 3 voxels thick']),
                             ('pp', 'Quality check', ['Dice and coverage vs the', 'original, both >= 0.75?'])],
                      arrows=True, link=('branch', [0, 1])),
                 dict(boxes=[('out', 'Pass: replace interior', ['region interior <- fit', 'keep 8-voxel border',
                                                                '(stays connected)']),
                             ('pp', 'Fail: probably merged', ['retry the region with the', 'alt masks 0.5 -> 0.6 -> 0.7,',
                                                              'fit each component']),
                             ('pp', 'Alt pass threshold', ['Dice >= 0.45 and', 'coverage >= 0.80'])],
                      arrows=[False, True])]),
        dict(heading='Step 6 - Final steps and submission', rows=[
            dict(boxes=[('pp', 'Remove small', ['components < 1000 voxels']),
                        ('pp', 'Pad the border back', ['3 voxels per face', '-> original shape']),
                        ('out', 'Submission', ['uint8 0 / 1 .tif per volume', 'zipped -> submission.zip'])],
                 arrows=True)]),
        dict(heading='Scores (private LB)', rows=[
            dict(boxes=[('out', 'Local interpolation', ['final submission', '0.622']),
                        ('out', 'Global interpolation', ['whole component, no cubes', '0.631, not selected:',
                                                         'reached into label-2 areas'])],
                 arrows=False)]),
    ])

SPECS['rank3'] = dict(
    title='Rank 3 - W & A (private LB 0.621): two-stage nnU-Net cascade + ridge-detection post-processing',
    source=['Sources: 3rd place write-up, final notebook "nnUNet 1 rot TTA ensemble cascade no fillin", GitHub README.',
            '"not stated" = the value is not in the write-up and the code cell is hidden.'],
    panels=[
        dict(heading='Step 1 - Training setup (custom trainer nnUNetTrainer_RotFlip_ClDice_prob0.8)', rows=[
            dict(boxes=[('img', 'Training data', ['all train volumes', 'no hold-out', 'pseudo-labels tried, dropped']),
                        ('net', 'Augmentation', ['mirror forced on (0, 1, 2)', 'rotation up to +-30 deg',
                                                 'probability 0.8']),
                        ('net', 'Loss', ['Dice + CE (nnU-Net)', '+ clDice (skeleton-based)', 'clDice weight not stated']),
                        ('net', 'Optimizer and patch', ['RAdamScheduleFree,', 'no LR scheduler',
                                                        'train patch 128'])],
                 arrows=True)]),
        dict(heading='Step 2 - Models trained (all ResEnc M plans)', rows=[
            dict(boxes=[('net', 'Stage 1a: 3d_fullres', ['4000 epochs', 'ckpt 4000epochs_iter6 (best)']),
                        ('net', 'Stage 1b: 3d_fullres', ['4000 epochs', 'ckpt org_prob0.8 (best)']),
                        ('net', '3d_lowres', ['8000 epochs', 'only used to train', 'the cascade']),
                        ('net', 'Stage 2: 3d_cascade_fullres', ['8000 epochs', 'input: image + mask', 'of previous stage'])],
                 arrows=[False, False, True])],
             foot='2 x H100: low-res model 14 h, cascade 24 h (8000 epochs each).'),
        dict(heading='Step 3 - Stage 1 inference: full-resolution ensemble', rows=[
            dict(boxes=[('img', 'Test volume', ['e.g. (320, 314, 314)']),
                        ('sel', 'Sliding window', ['tile 192 (trained at 128)', 'step_size 0.85',
                                                   '8 windows per volume']),
                        ('sel', 'TTA', ['mirror (0, 1, 2)', '+ in-plane rotation +-15 deg', '(affine grid sample)']),
                        ('out', 'Probabilities', ['from 1a and from 1b', 'saved as .npz'])],
                 arrows=True),
            dict(boxes=[('out', 'Weighted fusion', ['0.65 x 1a + 0.35 x 1b']),
                        ('out', 'argmax', ['label map 0 / 1', 'written as .tif'])],
                 arrows=True)]),
        dict(heading='Step 4 - Stage 2 inference: cascade refinement', rows=[
            dict(boxes=[('img', 'Test volume', ['the CT image']),
                        ('out', 'Stage-1 label map', ['passed as', '-prev_stage_predictions'])],
                 arrows=False, link='fan'),
            dict(boxes=[('sel', '3d_cascade_fullres', ['input: image + prior mask', 'tile 192, step 0.85',
                                                       'same TTA']),
                        ('out', 'Refined mask', ['cascade output', '(320, 314, 314)'])],
                 arrows=True)],
             foot='Trained as low-res -> cascade, but at inference the full-resolution ensemble is fed as the prior.'),
        dict(heading='Step 5 - Post-processing (post.py, from the host\'s edt_frangi_label script; ~1 h)', rows=[
            dict(boxes=[('pp', 'Inverse-EDT dilation', ['distance-transform based', 'dilation, distance 1']),
                        ('pp', 'Hessian ridge detection', ['Gaussian smoothing,', '3D Hessian eigenvalues',
                                                           'ridge threshold 0.20']),
                        ('pp', 'Z-consistency pruning', ['on in slice z but off in', 'z-1 and z+1 -> remove']),
                        ('pp', 'Closing', ['2D per slice, then', 'anisotropic 3D in XY', 'sizes not stated']),
                        ('pp', 'Remove small objects', ['min size not stated', 'no hole filling'])],
                 arrows=True)],
             foot='Post-processing adds about +0.02 to +0.03 (team estimate).'),
        dict(heading='Step 6 - Output and scores', rows=[
            dict(boxes=[('out', 'Submission', ['uint8 0 / 1 .tif per volume', 'zipped -> submission.zip']),
                        ('out', 'Cascade 4000 epochs', ['public 0.592', 'private 0.620']),
                        ('out', 'Cascade 8000 epochs', ['public 0.586', 'private 0.624 (best)']),
                        ('out', 'Selected final', ['leaderboard private', '0.62090'])],
                 arrows=False)]),
    ])


# ----------------------------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------------------------
class Svg:
    def __init__(self):
        self.parts = []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, cls='t', anchor='start', color=None):
        c = f'{cls} {color}' if color else cls
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" class="{c}" text-anchor="{anchor}">{escape(s)}</text>')

    def rect(self, x, y, w, h, cls, rx=6):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" class="{cls}"/>')

    def line(self, x1, y1, x2, y2, arrow=True):
        m = ' marker-end="url(#arrow)"' if arrow else ''
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="arr"{m}/>')

    def comment(self, s):
        self.add(f'<!-- ===================== {s} ===================== -->')


def box_height(box):
    return 40 + 18 * len(box[2])


def draw_row(svg, row, top, where):
    boxes = row['boxes']
    n = len(boxes)
    arrows = row.get('arrows', True)
    arrows = [arrows] * (n - 1) if isinstance(arrows, bool) else arrows
    gap = 40 if any(arrows) else 18
    avail = (X1 - PAD) - (X0 + PAD)
    bw = (avail - gap * (n - 1)) / n
    h = max(box_height(b) for b in boxes)
    xs = []
    for i, (cls, title, lines) in enumerate(boxes):
        x = X0 + PAD + i * (bw + gap)
        xs.append(x)
        svg.rect(x, top, bw, h, cls, rx=8)
        svg.text(x + bw / 2, top + 24, title, 'th', 'middle', f'c-{cls}')
        fits(title, 'th', bw - 12, where)
        for k, s in enumerate(lines):
            svg.text(x + bw / 2, top + 46 + k * 18, s, 'ts', 'middle')
            fits(s, 'ts', bw - 12, where)
        if i < n - 1 and arrows[i]:
            svg.line(x + bw + 4, top + h / 2, x + bw + gap - 4, top + h / 2)
    return h, xs, bw


def draw_panel(svg, p, top):
    svg.comment(p['heading'].upper())
    y = top + 38
    if p.get('loop'):
        y += 6
        svg.text(X0 + PAD, y + 8, p['loop'], 'h3')
        y += 22
    row_geo = []
    for r_i, row in enumerate(p['rows']):
        if r_i > 0:
            y += 40
        h, xs, bw = draw_row(svg, row, y, p['heading'])
        row_geo.append((y, h, xs, bw))
        y += h
    # links between rows. 'chain' (default): the flow continues from the last box of a row to the first box
    # of the next row (elbow line). 'fan': every box of the row feeds the first box of the next row.
    for r_i, ((ya, ha, xa, bwa), (yb, hb, xb, bwb)) in enumerate(zip(row_geo, row_geo[1:])):
        tx = xb[0] + bwb / 2
        link = p['rows'][r_i].get('link', 'chain')
        if link == 'fan':
            for x in xa:
                svg.line(x + bwa / 2, ya + ha + 2, tx, yb - 3)
        elif isinstance(link, tuple) and link[0] == 'branch':
            # the last box of the row splits to the listed boxes of the next row (e.g. pass / fail)
            sx, ym = xa[-1] + bwa / 2, ya + ha + 20
            for t in link[1]:
                bx = xb[t] + bwb / 2
                svg.add(f'<path d="M{sx:.1f} {ya + ha + 2:.1f} L{sx:.1f} {ym:.1f} L{bx:.1f} {ym:.1f} '
                        f'L{bx:.1f} {yb - 3:.1f}" class="arr" marker-end="url(#arrow)"/>')
        else:
            sx, ym = xa[-1] + bwa / 2, ya + ha + 20
            svg.add(f'<path d="M{sx:.1f} {ya + ha + 2:.1f} L{sx:.1f} {ym:.1f} L{tx:.1f} {ym:.1f} L{tx:.1f} {yb - 3:.1f}" '
                    f'class="arr" marker-end="url(#arrow)"/>')
    if p.get('foot'):
        y += 22
        svg.text(X0 + PAD, y, p['foot'], 'ts')
        fits(p['foot'], 'ts', X1 - X0 - 2 * PAD, p['heading'])
    bottom = y + 18
    svg.text(X0 + 16, top + 26, p['heading'], 'h2')
    return bottom


def draw_header(svg, title, source):
    svg.comment('TITLE AND LEGEND')
    svg.text(30, 42, title, 'title')
    for i, s in enumerate(source):
        svg.text(30, 66 + i * 16, s, 'ts')
    lx, ly = 1080, 92
    svg.rect(lx, ly, 300, 26 + 18 * len(LEGEND), 'panel', rx=8)
    for i, (cls, lab) in enumerate(LEGEND):
        yy = ly + 12 + i * 18
        svg.rect(lx + 12, yy, 22, 12, cls, rx=2)
        svg.text(lx + 42, yy + 11, lab, 'ts')
    return ly + 26 + 18 * len(LEGEND) + 16


def build_pipeline(spec):
    svg = Svg()
    y = draw_header(svg, spec['title'], spec['source'])
    for i, p in enumerate(spec['panels']):
        if i > 0:
            svg.line(W / 2, y + 2, W / 2, y + 26)
            y += 30
        # panel background first so boxes sit on top
        start = len(svg.parts)
        bottom = draw_panel(svg, p, y)
        svg.parts.insert(start, f'<rect x="{X0}" y="{y}" width="{X1 - X0}" height="{bottom - y}" rx="8" class="panel"/>')
        y = bottom
    return svg, y + 24


# ----------------------------------------------------------------------------------------------
# Options diagram (rank 2): three readings of the same post-processing, side by side
# ----------------------------------------------------------------------------------------------
OPTIONS_R2 = [
    ('A', 'Write-up, main text', False, [
        ('sel', 'Inference', ['sum of logits 0.6 / 0.4', 'mirroring + rotation TTA']),
        ('pp', '4 masks', ['hysteresis T_low 0.2 / 0.5 / 0.6 / 0.7']),
        ('pp', 'Find holes', ['small windows, 50 % overlap', 'Euler number per window']),
        ('pp', 'Merge windows', ['overlapping windows with holes', 'merged into patches']),
        ('pp', 'Interpolate patch', ['PCA plane + griddata, edge flood-', 'remove, thicken to 3 voxels']),
        ('pp', 'Quality check', ['Dice + coverage high enough?', 'no -> next higher threshold']),
        ('out', 'One pass', ['remove small, pad border back'])]),
    ('B', 'Final notebook code (0.622) - recommended', True, [
        ('sel', 'Inference', ['probability average 0.6 / 0.4', 'mirroring only (no rotation seen)']),
        ('pp', '4 masks', ['hysteresis T_low 0.2 / 0.5 / 0.6 / 0.7', 'T_high 0.83, z-closing r1']),
        ('pp', 'Find holes', ['cubes 80, overlap 40', 'Euler number per cube']),
        ('pp', 'Merge cubes', ['overlapping / adjacent cubes', 'with holes form groups']),
        ('pp', 'Interpolate group', ['grid 80, thickness 3', 'keep an 8-voxel border']),
        ('pp', 'Quality check', ['base: Dice and cov >= 0.75', 'alt: Dice >= 0.45, cov >= 0.80']),
        ('out', 'One pass', ['remove < 1000, pad 3 back'])]),
    ('C', 'Write-up edit about the final notebook', False, [
        ('sel', 'Inference', ['as in A or B', '(not restated in the edit)']),
        ('pp', '4 masks', ['hysteresis T_low 0.2 / 0.5 / 0.6 / 0.7']),
        ('pp', 'Find holes', ['Euler number of the', 'whole component at once']),
        ('pp', 'Replace component', ['hole found -> replace the', 'whole component']),
        ('pp', 'Interpolate', ['same PCA + griddata fit']),
        ('pp', 'Quality check', ['same Dice / coverage', 'retry with higher thresholds']),
        ('out', 'Two passes', ['the whole loop runs twice'])]),
]


def build_options():
    svg = Svg()
    svg.comment('TITLE')
    svg.text(30, 42, 'Rank 2 - which version of the post-processing should the pipeline diagram show?', 'title')
    svg.text(30, 66, 'The write-up and the final notebook describe the local interpolation differently. '
                     'Pick A, B or C.', 'ts')
    svg.text(30, 82, 'B is recommended: it is the code that produced the 0.622 private-LB submission.', 'ts')
    top, colw, gap = 110, 430, 25
    bottom = top
    for c, (letter, label, rec, steps) in enumerate(OPTIONS_R2):
        x = 20 + c * (colw + gap)
        y = top + 50
        for k, (cls, title, lines) in enumerate(steps):
            h = 40 + 18 * len(lines)
            svg.rect(x + 15, y, colw - 30, h, cls, rx=8)
            svg.text(x + colw / 2, y + 24, title, 'th', 'middle', f'c-{cls}')
            for j, s in enumerate(lines):
                svg.text(x + colw / 2, y + 46 + j * 18, s, 'ts', 'middle')
                fits(s, 'ts', colw - 42, f'option {letter}')
            if k < len(steps) - 1:
                svg.line(x + colw / 2, y + h + 2, x + colw / 2, y + h + 20)
            y += h + 24
        bottom = max(bottom, y)
        heading = f'{letter}) {label}'
        fits(heading, 'h3', colw - 20, 'options heading')
        svg.parts.insert(0, f'<rect x="{x}" y="{top}" width="{colw}" height="{{H_{c}}}" rx="8" class="panel" '
                            f'style="{"stroke:#534AB7;stroke-width:2.5" if rec else ""}"/>')
        svg.text(x + 15, top + 30, heading, 'h3')
    h_col = bottom - top
    svg.parts = [p.replace('{H_0}', str(h_col)).replace('{H_1}', str(h_col)).replace('{H_2}', str(h_col))
                 for p in svg.parts]
    return svg, bottom + 20


def wrap(svg, H, title, desc):
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
            f'<title>{escape(title)}</title>\n<desc>{escape(desc)}</desc>\n<style>{STYLE}</style>\n'
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" '
            'orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" '
            'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>\n'
            f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>\n')
    return head + '\n'.join(svg.parts) + '\n</svg>\n'


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    out, mode = Path(sys.argv[1]), sys.argv[2]
    if mode == 'rank2_options':
        svg, H = build_options()
        doc = wrap(svg, H, 'Rank 2 post-processing options', 'Three readings of the 2nd place post-processing.')
    elif mode in SPECS:
        svg, H = build_pipeline(SPECS[mode])
        doc = wrap(svg, H, SPECS[mode]['title'], f'Full pipeline of the {mode} Kaggle solution.')
    else:
        sys.exit(f'unknown mode {mode}; choose from {sorted(SPECS) + ["rank2_options"]}')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding='utf-8')
    print(f'wrote {out} ({out.stat().st_size} bytes)')
    for w in WARNINGS:
        print('FIT WARNING', w)


if __name__ == '__main__':
    main()
