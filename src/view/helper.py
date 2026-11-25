"""Interactive orthogonal-slice viewer for a CT volume and its label mask (for Jupyter).

Source: adapted from Jirka Borovec's Kaggle notebook "Surface📜: EDA🔎 interactive img/mask 3D🪚view"
https://www.kaggle.com/code/jirkaborovec/surface-eda-interactive-img-mask-3d-view
released under the Apache 2.0 license.
Changes from the original:
  - loads from the local `data/` folder (any split) or from any .tif path, instead of the Kaggle input path;
  - the mask is optional (e.g. for test images); without it only the CT is drawn;
  - a "Next +1" button next to each slice slider moves that slice forward by one;
  - removed an unused helper (`_set_axes_labels`) and the Kaggle run-type switch.

What it shows (2x2 figure):
  - top-left:     axial slice      vol[z, :, :]   (red frame)
  - top-right:    sagittal slice   vol[:, :, x].T (green frame)
  - bottom-left:  coronal slice    vol[:, y, :]   (blue frame)
  - bottom-right: 3D box with the three cutting planes, to show where the slices are
  Each 2D slice draws lines where the other two planes cut it. The mask is overlaid with
  0 = light gray, 1 = yellow (surface), 2 = cyan (unlabeled), with adjustable transparency.

Usage (next cell):
    image, mask = load_volume('1004283650')        # by id, looks in DATA_DIR
    interactive_show(image, mask)                    # sliders + Next buttons for Z, Y, X; mask alpha slider

    image, _ = load_volume(r'Z:/path/to/some.tif')   # any .tif file; mask found only for files inside data/

Needs: numpy, matplotlib, ipywidgets, tifffile, imagecodecs (the files are LZW-compressed).
"""
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from IPython.display import display
from ipywidgets import Button, FloatSlider, HBox, IntSlider, Layout, VBox, interactive_output
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath

DATA_DIR = Path(__file__).resolve().parents[2] / "data"   # <project root>/data, wherever the project lives

_SPLITS = (('train_images', 'train_labels'),
           ('deprecated_train_images', 'deprecated_train_labels'),
           ('test_images', None))


def load_volume(id_or_path, data_dir: Path = DATA_DIR):
    """Load a CT volume and, if it exists, its mask.

    `id_or_path` is either a volume id (searched in train, deprecated train, then test under `data_dir`)
    or a path to a .tif file. Returns (image, mask), with mask = None when no label file is found.
    """
    data_dir = Path(data_dir)
    p = Path(str(id_or_path))
    if p.suffix.lower() in ('.tif', '.tiff') and p.exists():
        image_path, vid = p, p.stem
        mask_path = None
        for img_dir, lab_dir in _SPLITS:
            if lab_dir and p.parent.name == img_dir:
                mask_path = p.parent.parent / lab_dir / p.name
    else:
        vid = str(id_or_path)
        image_path = mask_path = None
        for img_dir, lab_dir in _SPLITS:
            cand = data_dir / img_dir / f'{vid}.tif'
            if cand.exists():
                image_path = cand
                mask_path = data_dir / lab_dir / f'{vid}.tif' if lab_dir else None
                break
        if image_path is None:
            raise FileNotFoundError(f'volume {vid} not found under {data_dir}')

    image = tifffile.imread(image_path)
    mask = tifffile.imread(mask_path) if mask_path is not None and mask_path.exists() else None
    print(f"Loaded image '{image_path}' ({image.dtype}) shape {image.shape}, "
          f'values {image.min()} -> {image.max()}')
    if mask is not None:
        print(f"Loaded mask  '{mask_path}' ({mask.dtype}) shape {mask.shape}, labels {np.unique(mask)}")
    else:
        print('No mask found for this volume.')
    return image, mask


def _draw_line(ax, coords, clr='g'):
    line = MplPath(coords, [MplPath.MOVETO, MplPath.LINETO])
    ax.add_patch(PathPatch(line, linewidth=3, edgecolor=clr, facecolor='none'))


_rec_prop = dict(linewidth=5, facecolor='none')

# Mask colours: 0 = background, 1 = foreground (surface), 2 = unlabeled
_MASK_CMAP = ListedColormap(['lightgray', 'yellow', 'cyan'])
_MASK_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], _MASK_CMAP.N)
_LEGEND = [mpatches.Patch(color='lightgray', label='0: Background'),
           mpatches.Patch(color='yellow', label='1: Foreground'),
           mpatches.Patch(color='cyan', label='2: Unlabeled')]


def _overlay(ax, mask2d, mask_alpha):
    if mask2d is not None:
        ax.imshow(mask2d, cmap=_MASK_CMAP, norm=_MASK_NORM, alpha=mask_alpha, interpolation='nearest')


def show_volume(vol, mask_vol, z, y, x, mask_alpha, fig_size=(12, 12)):
    fig = plt.figure(figsize=fig_size)
    ax00 = fig.add_subplot(2, 2, 1)                   # axial
    ax01 = fig.add_subplot(2, 2, 2)                   # sagittal
    ax10 = fig.add_subplot(2, 2, 3)                   # coronal
    ax11 = fig.add_subplot(2, 2, 4, projection='3d')  # where the planes are
    v_z, v_y, v_x = vol.shape
    m = mask_vol

    # Axial slice (XY plane, view along Z)
    ax00.imshow(vol[z, :, :], cmap='gray')
    _overlay(ax00, None if m is None else m[z, :, :], mask_alpha)
    ax00.add_patch(Rectangle((-1, -1), v_x, v_y, edgecolor='r', **_rec_prop))
    _draw_line(ax00, [(x, 0), (x, v_y)], 'g')  # sagittal cut
    _draw_line(ax00, [(0, y), (v_x, y)], 'b')  # coronal cut
    ax00.set_title(f'Axial (XY) Slice - Z: {z}')

    # Sagittal slice (YZ plane, view along X, transposed)
    ax01.imshow(vol[:, :, x].T, cmap='gray')
    _overlay(ax01, None if m is None else m[:, :, x].T, mask_alpha)
    ax01.add_patch(Rectangle((-1, -1), v_z, v_y, edgecolor='g', **_rec_prop))
    _draw_line(ax01, [(z, 0), (z, v_y)], 'r')  # axial cut
    _draw_line(ax01, [(0, y), (v_z, y)], 'b')  # coronal cut
    ax01.set_title(f'Sagittal (YZ) Slice - X: {x}')

    # Coronal slice (XZ plane, view along Y)
    ax10.imshow(vol[:, y, :], cmap='gray')
    _overlay(ax10, None if m is None else m[:, y, :], mask_alpha)
    ax10.add_patch(Rectangle((-1, -1), v_x, v_z, edgecolor='b', **_rec_prop))
    _draw_line(ax10, [(0, z), (v_x, z)], 'r')  # axial cut
    _draw_line(ax10, [(x, 0), (x, v_z)], 'g')  # sagittal cut
    ax10.set_title(f'Coronal (XZ) Slice - Y: {y}')

    # 3D box with the three cutting planes
    Y_grid, X_grid = np.meshgrid(np.arange(v_y), np.arange(v_x))
    Z_grid_cor, X_grid_cor = np.meshgrid(np.arange(v_z), np.arange(v_x))
    Z_grid_sag, Y_grid_sag = np.meshgrid(np.arange(v_z), np.arange(v_y))
    ax11.plot_surface(X_grid, Y_grid, np.full_like(X_grid, z), color='r', alpha=0.2)
    ax11.plot_surface(X_grid_cor, np.full_like(X_grid_cor, y), Z_grid_cor, color='b', alpha=0.2)
    ax11.plot_surface(np.full_like(Y_grid_sag, x), Y_grid_sag, Z_grid_sag, color='g', alpha=0.2)
    ax11.set_xlabel('X (width)')
    ax11.set_ylabel('Y (height)')
    ax11.set_zlabel('Z (depth)')
    ax11.set_xlim([0, v_x])
    ax11.set_ylim([0, v_y])
    ax11.set_zlim([0, v_z])
    ax11.view_init(elev=20, azim=45)
    if m is not None:
        ax11.legend(handles=_LEGEND, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(_LEGEND),
                    fancybox=True, shadow=True)

    fig.tight_layout()
    plt.show()


def _next_button(slider):
    """A button that moves `slider` forward by one, stopping at its maximum."""
    btn = Button(description='Next +1', layout=Layout(width='90px'))
    btn.on_click(lambda _: setattr(slider, 'value', min(slider.value + 1, slider.max)))
    return btn


def interactive_show(volume, mask_vol=None):
    """Sliders and "Next +1" buttons for the Z / Y / X slice positions, plus the mask transparency."""
    vol_z, vol_y, vol_x = volume.shape
    z = IntSlider(min=0, max=vol_z - 1, step=1, value=vol_z // 2, description='Z-slice')
    y = IntSlider(min=0, max=vol_y - 1, step=1, value=vol_y // 2, description='Y-slice')
    x = IntSlider(min=0, max=vol_x - 1, step=1, value=vol_x // 2, description='X-slice')
    mask_alpha = FloatSlider(min=0.0, max=1.0, step=0.05, value=0.3, description='Mask Alpha')

    controls = VBox([HBox([s, _next_button(s)]) for s in (z, y, x)] + [mask_alpha])
    out = interactive_output(
        lambda z, y, x, mask_alpha: show_volume(volume, mask_vol, z, y, x, mask_alpha),
        dict(z=z, y=y, x=x, mask_alpha=mask_alpha),
    )
    display(controls, out)