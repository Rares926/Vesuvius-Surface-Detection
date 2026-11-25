"""Draw one pipeline diagram (input shape -> model architecture -> loss -> inference) per model type.

Usage: python src/diagrams/draw_model_pipelines.py <output.svg> <mode>
  mode: 2d | 25d | 3d

Every number comes from the public Kaggle notebook named in each spec (read 2026-11-21).
Spatial sizes per level are computed from the input size and the strides.
"""
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

W = 1400

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
.img-dark{fill:#D3D1C7;stroke:#5F5E5A;stroke-width:1}
.sel{fill:#EEEDFE;stroke:#534AB7;stroke-width:1.2}
.roi{fill:none;stroke:#534AB7;stroke-width:1.8;stroke-dasharray:6 4}
.net{fill:#FFFFFF;stroke:#3d3d3a;stroke-width:1}
.out{fill:#E1F5EE;stroke:#0F6E56;stroke-width:1.2}
.gt{fill:#FAECE7;stroke:#993C1D;stroke-width:1.2}
.loss{fill:#FAEEDA;stroke:#854F0B;stroke-width:1.2}
.arr{stroke:#3d3d3a;stroke-width:1.6;fill:none}
.skip{stroke:#534AB7;stroke-width:1.2;stroke-dasharray:5 4;fill:none}
.leader{stroke:#888780;stroke-width:0.7;stroke-dasharray:3 3;fill:none}
.tsel{fill:#3C3489}.tout{fill:#085041}.tgt{fill:#712B13}.tloss{fill:#633806}.timg{fill:#444441}
"""

# ----------------------------------------------------------------------------------------------
# Specs: one per model type. Values are as read from each notebook.
# ----------------------------------------------------------------------------------------------
SPECS = {
    '2d': dict(
        title='2D segmentation pipeline - MONAI DynUNet on single slices',
        source=['Source: Jirka Borovec, "Surface: train+inference Slice|2D segmentation" (public score 0.432)',
                'kaggle.com/code/jirkaborovec/surface-train-inference-slice-2d-segmentation'],
        volume=(320, 320, 320),
        pick='slice', n_pick=1,
        pick_lines=['Take one slice along z', '(SLICE_AXIS = "z")', 'resize to 320 x 320'],
        tensor_lines=['Input tensor', '(B, 1, 320, 320)', 'B = 28 slices per batch', '1 channel = 1 slice'],
        net_name='DynUNet (MONAI), spatial_dims = 2',
        in_size=320, in_ch=1,
        enc_strides=[1, 2, 2, 2, 2, 2], enc_ch=[32, 64, 128, 256, 512, 1024],
        enc_note='kernel 3 x 3, dropout 0.2',
        dim='2d', head_lines=['Head: 1 x 1 conv', '-> (B, 2, 320, 320)', 'softmax over 2 classes'],
        dec_ch=True,
        loss_lines=['loss = Tversky + DiceCE', 'Tversky: alpha = 0.3 (FP), beta = 0.7 (FN)',
                    '   -> missing surface costs more than extra surface', 'DiceCE: 0.5 Dice + 0.5 cross-entropy',
                    'label 2 (unlabeled) masked out of the loss'],
        train_lines=['AdamW, lr 1e-3, weight decay 1e-2', '15 epochs, batch 28, 10 % validation split'],
        infer_lines=['For each of the 320 z-slices:', '  predict -> argmax -> (320, 320)', 'Stack -> (320, 320, 320) mask',
                     'Remove 3D components < 25^3 voxels'],
    ),
    '25d': dict(
        title='2.5D segmentation pipeline - ConvNeXt U-Net on 3-slice stacks',
        source=['Source: Vladimir Vilchinsky, "FastAI U-Net 2.5D" (public score 0.437)',
                'kaggle.com/code/vladimirvilchinsky/fastai-u-net-2-5d'],
        volume=(320, 320, 320),
        pick='stack', n_pick=3,
        pick_lines=['Take slices z-1, z, z+1', 'z-score normalise the stack', 'resize to 128 x 128'],
        tensor_lines=['Input tensor', '(B, 3, 128, 128)', 'B = 16 stacks per batch', '3 channels = 3 slices'],
        net_name='fastai DynamicUnet with ConvNeXt-small encoder (ImageNet-22k weights)',
        in_size=128, in_ch=3,
        enc_strides=[4, 2, 2, 2], enc_ch=[96, 192, 384, 768],
        enc_note='ConvNeXt stages; stem downsamples x4',
        dim='2d', head_lines=['Upsample to 128 x 128', '-> (B, 2, 128, 128)', 'softmax over 2 classes'],
        dec_ch=False,
        loss_lines=['loss = cross-entropy + Dice', 'both with ignore_index = 2', '   -> unlabeled voxels give no gradient'],
        train_lines=['1 epoch with the encoder frozen (lr 1e-3)', 'then 7 epochs fine-tuning all layers, batch 16'],
        infer_lines=['For each slice z:', '  stack z-1, z, z+1 -> predict', '  resize back to H x W -> argmax',
                     'Stack -> 3D mask, remove objects < 1000 voxels'],
    ),
    '3d': dict(
        title='3D segmentation pipeline - nnU-Net (ResEnc-M) on 3D patches',
        source=['Source: Jirka Borovec, "Surface: nnUNet training + inference [with 2xT4]" (public score 0.451)',
                'kaggle.com/code/jirkaborovec/surface-nnunet-training-inference-with-2xt4'],
        volume=(320, 314, 314),
        pick='patch', n_pick=1,
        pick_lines=['Config 3d_lowres: resample', 'spacing 1.0 -> 1.558', '(320, 314, 314) -> (205, 202, 202)',
                    'then cut 3D patches P'],
        tensor_lines=['Input tensor', '(B, 1, Pz, Py, Px)', 'patch size P chosen by the', 'nnU-Net planner (not printed)'],
        net_name='nnU-Net ResidualEncoderUNet, ResEnc-M preset (3D)',
        in_size=None, in_ch=1,
        enc_strides=[1, 2, 2, 2, 2, 2], enc_ch=[32, 64, 128, 256, 320, 320],
        enc_blocks=[1, 3, 4, 6, 6, 6],
        enc_note='residual blocks per stage: 1, 3, 4, 6, 6, 6 (ResEnc-M preset)',
        dim='3d', head_lines=['Head: 1 x 1 x 1 conv', '-> (B, 2, Pz, Py, Px)', 'softmax over 2 classes'],
        dec_ch=True,
        loss_lines=['loss = Dice + cross-entropy (nnU-Net default)', 'deep supervision: the loss is also applied',
                    '   to lower-resolution decoder outputs', 'label 2 declared "ignore" -> masked out', ''],
        train_lines=['nnUNetTrainer_100epochs, fold "all"', '2 x T4 GPUs; validation Dice 0.754'],
        infer_lines=['Sliding window of patches over the volume', '(nnU-Net default: overlapping windows,',
                     'Gaussian-weighted, mirroring TTA)', 'Resample back -> argmax -> (320, 314, 314)'],
    ),
}


# ----------------------------------------------------------------------------------------------
# SVG helpers
# ----------------------------------------------------------------------------------------------
class Svg:
    def __init__(self):
        self.parts = []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, cls='t', anchor='start', extra=''):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}"{extra}>{escape(s)}</text>')

    def lines(self, x, y, items, cls='t', step=18, anchor='start', first_cls=None):
        for i, s in enumerate(items):
            if s:
                self.text(x, y + i * step, s, first_cls if (first_cls and i == 0) else cls, anchor)

    def rect(self, x, y, w, h, cls, rx=6):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" class="{cls}"/>')

    def arrow(self, x1, y1, x2, y2, cls='arr'):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" '
                 f'marker-end="url(#arrow)"/>')

    def path(self, d, cls):
        self.add(f'<path d="{d}" class="{cls}"/>')

    def comment(self, s):
        self.add(f'<!-- ===================== {s} ===================== -->')


def cube(svg, x, y, w, h, d, cls='img'):
    """Isometric box: front face at (x, y, w, h), depth offset d up-right."""
    svg.path(f'M{x} {y} L{x + d} {y - d} L{x + w + d} {y - d} L{x + w} {y} Z', cls)
    svg.path(f'M{x + w} {y} L{x + w + d} {y - d} L{x + w + d} {y + h - d} L{x + w} {y + h} Z', cls)
    svg.rect(x, y, w, h, cls, rx=0)


def panel(svg, x, y, w, h, heading):
    svg.rect(x, y, w, h, 'panel', rx=8)
    svg.text(x + 16, y + 26, heading, 'h2')


# ----------------------------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------------------------
def draw_header(svg, s):
    svg.comment('TITLE')
    svg.text(30, 42, s['title'], 'title')
    svg.lines(30, 66, s['source'], 'ts', step=16)
    svg.comment('LEGEND')
    lx, ly = 1040, 18
    svg.rect(lx, ly, 340, 128, 'panel', rx=8)
    svg.text(lx + 14, ly + 22, 'Colours', 'th')
    rows = [('img', 'CT data (input volume)'), ('sel', 'Part fed to the model (slice / stack / patch)'),
            ('net', 'Network block'), ('out', 'Model output / prediction'), ('loss', 'Loss (training only)')]
    for i, (cls, label) in enumerate(rows):
        yy = ly + 36 + i * 18
        svg.rect(lx + 14, yy, 22, 12, cls, rx=2)
        svg.text(lx + 44, yy + 11, label, 'ts')


def draw_input(svg, s, top):
    svg.comment('STEP 1 - INPUT')
    panel(svg, 20, top, 1360, 310, 'Step 1 - Input: from the CT volume to the input tensor')
    z, y, x = s['volume']
    # volume cube
    cx, cy = 70, top + 90
    cube(svg, cx, cy, 120, 120, 40)
    svg.lines(cx + 80, cy + 150, ['CT volume', f'({z}, {y}, {x})', 'uint8, 0-255'], 'ts', 16, 'middle',
              first_cls='th')
    svg.arrow(260, cy + 60, 330, cy + 60)

    # what is picked
    px, py = 360, cy
    if s['pick'] == 'slice':
        cube(svg, px, py, 120, 120, 40, 'img')
        svg.path(f'M{px} {py + 60} L{px + 40} {py + 20} L{px + 160} {py + 20} L{px + 120} {py + 60} Z', 'sel')
        svg.text(px + 186, py + 44, 'slice z', 'ts', extra=' style="fill:#3C3489"')
    elif s['pick'] == 'stack':
        cube(svg, px, py, 120, 120, 40, 'img')
        for k, dy in enumerate((52, 60, 68)):
            svg.path(f'M{px} {py + dy} L{px + 40} {py + dy - 40} L{px + 160} {py + dy - 40} L{px + 120} {py + dy} Z',
                     'sel')
        svg.text(px + 186, py + 40, 'z-1, z, z+1', 'ts', extra=' style="fill:#3C3489"')
    else:  # patch
        cube(svg, px, py, 120, 120, 40, 'img')
        cube(svg, px + 30, py + 30, 50, 50, 18, 'sel')
        svg.text(px + 186, py + 44, 'patch P', 'ts', extra=' style="fill:#3C3489"')
    svg.lines(px + 80, cy + 150, s['pick_lines'], 'ts', 16, 'middle')
    svg.arrow(620, cy + 60, 690, cy + 60)

    # tensor box
    tx, ty = 720, cy - 10
    svg.rect(tx, ty, 260, 130, 'sel', rx=8)
    svg.lines(tx + 130, ty + 30, s['tensor_lines'], 't', 22, 'middle', first_cls='th')
    svg.text(1035, cy + 50, 'paired', 'ts', 'middle')
    svg.text(1035, cy + 66, 'with', 'ts', 'middle')
    svg.rect(1090, ty, 270, 130, 'gt', rx=8)
    svg.add(f'<text x="1225" y="{ty + 30}" class="th tgt" text-anchor="middle">Target mask (same shape)</text>')
    svg.lines(1225, ty + 52, ['values 0 / 1 / 2', '0 background, 1 surface', '2 unlabeled (ignored)'],
              'ts', 22, 'middle')
    return top + 310


def level_sizes(s):
    """Spatial size per encoder level, computed from the input size and strides."""
    if s['in_size'] is None:
        out, div = [], 1
        for st in s['enc_strides']:
            div *= st
            out.append('P' if div == 1 else f'P/{div}')
        return out
    out, size = [], s['in_size']
    for st in s['enc_strides']:
        size //= st
        out.append(str(size))
    return out


def size_label(s, size):
    if s['dim'] == '3d':
        return f'{size}^3' if not size.startswith('P') else size
    return f'{size} x {size}'


def draw_model(svg, s, top):
    svg.comment('STEP 2 - MODEL')
    n = len(s['enc_ch'])
    h = 120 + n * 58
    panel(svg, 20, top, 1360, h, 'Step 2 - Model: ' + s['net_name'])
    svg.text(36, top + 48, 'Encoder (left) halves the size at each level; decoder (right) doubles it back. '
                           'Dashed purple lines = skip connections.', 'ts')
    sizes = level_sizes(s)
    bw, bh, dy = 210, 40, 58
    ex0, dx0 = 120, 1060
    y0 = top + 76
    for i in range(n):
        yy = y0 + i * dy
        ex = ex0 + i * 70
        blocks = f' x{s["enc_blocks"][i]} res' if s.get('enc_blocks') else ''
        lab = f'{s["enc_ch"][i]} ch, {size_label(s, sizes[i])}{blocks}'
        svg.rect(ex, yy, bw, bh, 'net', rx=6)
        svg.text(ex + bw / 2, yy + 25, lab, 't', 'middle')
        if i < n - 1:
            st = s['enc_strides'][i + 1]
            svg.arrow(ex + bw / 2, yy + bh, ex + bw / 2 + 60, yy + dy - 2)
            svg.text(ex + bw / 2 + 70, yy + bh + 14, f'/{st}', 'ts')
        if i < n - 1:  # decoder level mirrored (bottleneck has no decoder twin)
            dx = dx0 - i * 70
            dlab = (f'{s["enc_ch"][i]} ch, {size_label(s, sizes[i])}' if s['dec_ch']
                    else f'{size_label(s, sizes[i])}')
            svg.rect(dx, yy, bw, bh, 'net', rx=6)
            svg.text(dx + bw / 2, yy + 25, dlab, 't', 'middle')
            svg.path(f'M{ex + bw} {yy + bh / 2} L{dx} {yy + bh / 2}', 'skip')
            # up arrow from the level below into this decoder block
            svg.arrow(dx + bw / 2 - 60, yy + dy - 2, dx + bw / 2, yy + bh)
            svg.text(dx + bw / 2 - 70, yy + bh + 14, f'x{s["enc_strides"][i + 1]}', 'ts', 'end')
    # bottleneck label
    yb = y0 + (n - 1) * dy
    svg.text(ex0 + (n - 1) * 70 + bw + 12, yb + 25, 'bottleneck', 'ts')
    svg.text(36, y0 + n * dy + 4, s['enc_note'], 'ts')
    # input and head
    in_size = size_label(s, 'P' if s['in_size'] is None else str(s['in_size']))
    svg.text(ex0 + bw / 2, y0 - 8, f'input: {s["in_ch"]} ch, {in_size}', 'ts', 'middle')
    hx = 1290
    svg.arrow(dx0 + bw, y0 + bh / 2, hx - 10, y0 + bh / 2)
    svg.rect(hx - 8, y0 - 20, 86, 90, 'out', rx=6)
    svg.add(f'<text x="{hx + 35}" y="{y0 + 2}" class="th tout" text-anchor="middle">Output</text>')
    svg.lines(hx + 35, y0 + 20, ['2 class', 'scores'], 'ts', 16, 'middle')
    svg.lines(1060, y0 + n * dy - 50, s['head_lines'], 'ts', 16)
    return top + h


def draw_bottom(svg, s, top):
    svg.comment('STEP 3 - LOSS / STEP 4 - INFERENCE')
    ph = 230
    panel(svg, 20, top, 670, ph, 'Step 3 - Loss (training)')
    svg.rect(40, top + 44, 630, 112, 'loss', rx=8)
    svg.lines(56, top + 66, s['loss_lines'], 't', 20, first_cls='th')
    svg.lines(56, top + 180, ['Training:'] + s['train_lines'], 'ts', 16)
    panel(svg, 710, top, 670, ph, 'Step 4 - Inference: back to a full 3D mask')
    svg.rect(730, top + 44, 630, 112, 'out', rx=8)
    svg.lines(746, top + 66, s['infer_lines'], 't', 20)
    svg.lines(746, top + 180, ['Submission: one uint8 .tif per volume,',
                               'same shape as the input volume, values 0 / 1.'], 'ts', 16)
    return top + ph


def ds_weights(n_outputs):
    """nnU-Net deep-supervision weights: 1/2^k per output, lowest resolution set to 0, normalised to sum 1."""
    w = [1 / 2 ** k for k in range(n_outputs)]
    w[-1] = 0
    total = sum(w)
    return [x / total for x in w]


def draw_bottom_3d(svg, s, top):
    """Detailed loss and inference panels for the nnU-Net (3D) diagram."""
    sizes = level_sizes(s)
    n_out = len(s['enc_ch']) - 1           # one output per decoder level
    weights = ds_weights(n_out)

    # ---------------- Step 3: loss ----------------
    svg.comment('STEP 3 - LOSS (DETAILED)')
    ph3 = 430
    panel(svg, 20, top, 1360, ph3, 'Step 3 - Loss (training): Dice + cross-entropy with deep supervision')
    svg.text(36, top + 48, '(notebook) = read from the notebook.   * = nnU-Net default, not printed in the notebook.',
             'ts')
    svg.text(40, top + 76, 'Each decoder level gives its own prediction; each is scored against the target '
                           'downsampled to its size *', 'th')
    ry, rh, rstep = top + 92, 38, 50
    for k in range(n_out):
        yy = ry + k * rstep
        svg.rect(40, yy, 200, rh, 'out', rx=6)
        svg.add(f'<text x="140" y="{yy + 24}" class="t tout" text-anchor="middle">output at {sizes[k]}</text>')
        svg.arrow(240, yy + rh / 2, 280, yy + rh / 2)
        svg.rect(282, yy, 300, rh, 'loss', rx=6)
        svg.add(f'<text x="432" y="{yy + 24}" class="t tloss" text-anchor="middle">'
                f'Dice + CE vs target at {sizes[k]}</text>')
        svg.arrow(582, yy + rh / 2, 622, yy + rh / 2)
        wtxt = f'x w{k} = {weights[k]:.3f}' if weights[k] else f'x w{k} = 0 (lowest output not used)'
        svg.text(630, yy + 24, wtxt, 'th' if weights[k] else 'ts')
    yl = ry + n_out * rstep + 14
    svg.text(40, yl, 'Total loss  L = w0 (Dice0 + CE0) + w1 (Dice1 + CE1) + ...', 'th')
    raw = ', '.join(f'1/{2 ** k}' for k in range(n_out - 1))
    svg.text(40, yl + 20, f'weights: {raw}, 0  ->  normalised to sum 1 *', 'ts')

    bx = 900
    svg.rect(bx, top + 64, 460, 140, 'loss', rx=8)
    svg.lines(bx + 16, top + 88, ['Each term *',
                                  'CE: cross-entropy, ignore_index = 2',
                                  'Dice: soft Dice on the surface class only',
                                  '(background not scored, smooth = 1e-5)',
                                  'label-2 voxels masked out of both terms',
                                  'target downsampled (nearest) per output'], 'ts', 19, first_cls='th')
    svg.rect(bx, top + 216, 460, 200, 'net', rx=8)
    svg.lines(bx + 16, top + 240, ['Training loop',
                                   '100 epochs, fold "all", 2 x T4 with DDP (notebook)',
                                   'poly LR: lr = lr0 (1 - epoch / 100)^0.9 (notebook)',
                                   'SGD Nesterov, momentum 0.99, lr0 = 0.01 *',
                                   '250 iterations per epoch, weight decay 3e-5 *',
                                   '1/3 of patches forced to contain surface *',
                                   'augment: rotate, scale, noise, blur, brightness,',
                                   'contrast, low-res, gamma, mirror *',
                                   'result: validation Dice 0.754 (notebook)'], 'ts', 19, first_cls='th')
    top += ph3 + 24

    # ---------------- Step 4: inference ----------------
    svg.comment('STEP 4 - INFERENCE (DETAILED)')
    ph4 = 430
    panel(svg, 20, top, 1360, ph4, 'Step 4 - Inference: sliding window over the volume, then back to a 3D mask')
    svg.text(36, top + 48, '(notebook) = read from the notebook.   * = nnU-Net default, not printed in the notebook.',
             'ts')
    # 4a sliding window (one plane of the resampled volume)
    ax, ay, side, wpx = 60, top + 80, 170, 85
    svg.text(40, ay - 10, 'a) Sliding window *', 'h3')
    svg.rect(ax, ay, side, side, 'img', rx=0)
    step = wpx // 2
    for i in range(0, side - wpx + 1, step):
        for j in range(0, side - wpx + 1, step):
            svg.add(f'<rect x="{ax + j}" y="{ay + i}" width="{wpx}" height="{wpx}" '
                    f'style="fill:#1D9E75;fill-opacity:0.06;stroke:#1D9E75;stroke-width:1.2;'
                    f'stroke-dasharray:6 4"/>')
    svg.lines(250, ay + 20, ['one plane of the resampled', 'volume (205, 202, 202)', '',
                             'window = patch P', 'step = P / 2 (50 % overlap)', 'every voxel is predicted',
                             'by several windows'], 'ts', 17)
    # 4b gaussian weighting
    gx, gy = 470, ay
    svg.text(gx, ay - 10, 'b) Gaussian weighting *', 'h3')
    pts = []
    for t in range(0, 181, 6):
        u = (t - 90) / 30
        pts.append(f'{gx + t:.1f} {gy + 120 - 100 * math.exp(-u * u / 2):.1f}')
    svg.path('M' + ' L'.join(pts), 'arr')
    svg.add(f'<line x1="{gx}" y1="{gy + 121}" x2="{gx + 180}" y2="{gy + 121}" class="leader"/>')
    svg.lines(gx, gy + 146, ['inside a window, the centre counts', 'more than the edges, where the',
                             'network sees less context'], 'ts', 17)
    # 4c mirroring
    mx = 800
    svg.text(mx, ay - 10, 'c) Mirroring (test-time augmentation) *', 'h3')
    for k, lab in enumerate(['as is', 'flip z', 'flip y', 'flip x', 'z+y', 'z+x', 'y+x', 'z+y+x']):
        cxk, cyk = mx + (k % 4) * 82, ay + (k // 4) * 52
        svg.rect(cxk, cyk, 72, 40, 'sel', rx=4)
        svg.add(f'<text x="{cxk + 36}" y="{cyk + 25}" class="ts tsel" text-anchor="middle">{lab}</text>')
    svg.lines(mx, ay + 126, ['each window is predicted 8 times (all flip', 'combinations), flipped back and averaged'],
              'ts', 17)

    # 4d flow to the submission
    fy = top + 300
    svg.text(40, fy - 14, 'd) From window predictions to the submission', 'h3')
    boxes = [('out', 'Stitch windows', ['weighted average of logits', '(2, 205, 202, 202) *']),
             ('out', 'Resample back', ['to the original grid', '(2, 320, 314, 314) *']),
             ('out', 'Softmax', ['probabilities saved as .npz', '--save_probabilities (notebook)']),
             ('out', 'argmax', ['mask 0 / 1, uint8', '(320, 314, 314) (notebook)']),
             ('out', 'Write .tif', ['one file per volume id', 'zipped -> submission.zip (notebook)'])]
    bw, gap, x0 = 236, 30, 40
    for i, (cls, head, body) in enumerate(boxes):
        bxx = x0 + i * (bw + gap)
        svg.rect(bxx, fy, bw, 100, cls, rx=8)
        svg.add(f'<text x="{bxx + bw / 2}" y="{fy + 28}" class="th tout" text-anchor="middle">{escape(head)}</text>')
        svg.lines(bxx + bw / 2, fy + 52, body, 'ts', 18, 'middle')
        if i < len(boxes) - 1:
            svg.arrow(bxx + bw + 4, fy + 50, bxx + bw + gap - 4, fy + 50)
    svg.text(40, fy + 120, 'No post-processing step (e.g. removing small components) was found in this notebook.',
             'ts')
    return top + ph4


def build(mode):
    s = SPECS[mode]
    svg = Svg()
    y = 160
    y = draw_input(svg, s, y) + 24
    model_top = y
    y = draw_model(svg, s, model_top) + 24
    y = (draw_bottom_3d if mode == '3d' else draw_bottom)(svg, s, y) + 24
    draw_header(svg, s)
    H = y
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
            f'<title>{escape(s["title"])}</title>\n'
            f'<desc>Input shape, network architecture, loss and inference of the {mode} model type, '
            f'as used in a public Kaggle notebook.</desc>\n'
            f'<style>{STYLE}</style>\n'
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" '
            'orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" '
            'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>\n'
            f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>\n')
    return head + '\n'.join(svg.parts) + '\n</svg>\n'


def main():
    if len(sys.argv) != 3 or sys.argv[2] not in SPECS:
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(sys.argv[2]), encoding='utf-8')
    print(f'wrote {out} ({out.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
