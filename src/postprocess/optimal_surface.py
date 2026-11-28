"""Optimal-surface sheet fitting: a drop-in for rank 2's griddata height map.

Rank 2 fits each hole region with a PCA plane plus a height map `w(u, v)` from `griddata` over the binary voxels
(`rank2_reference.fit_curved_sheet_to_component_optimized`). That ignores the probability map inside the hole and,
when a region holds two merged sheets, averages their heights so the surface lands between them.

Here the height map is instead the exact minimiser of

    E(w) = sum_(u,v) [ -log p(x(u, v, w)) + lam * |w - w_ref| ]     subject to |w(a) - w(b)| <= delta for grid neighbours

over the probability map, where `w_ref` is rank 2's griddata height (a weak prior for columns without evidence).
This is the optimal-surface / LOGISMOS height-field problem (Ishikawa 2003; Li, Wu, Chen, Sonka, TPAMI 2006),
solved exactly with one min-cut (`scipy.sparse.csgraph.maximum_flow`, Dinic). Each column picks one height, so the
surface follows one probability ridge, and it stays a single-valued height field: no holes by construction.

Probability evidence is kept only on the voxels of the mask being fitted and on voxels outside the base mask (the
holes). Voxels of other base-mask sheets in the cube are set to p = 0 so the cut cannot jump onto them.

Everything after the height map (edge trimming, rasterising, thickening, per-slice fill) follows the reference.

Use `use_optimal_surface(prob, base, params)` as a context manager around `process_multiple_components_patchwise`:
it swaps the fit function inside `rank2_reference` and restores it afterwards. `rank2_reference.py` stays verbatim.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.interpolate import griddata
from scipy.sparse.csgraph import breadth_first_order, maximum_flow
from scipy.spatial import cKDTree

from . import rank2_reference as ref

INF = 1 << 29                       # "infinite" capacity, kept well inside int32
COST_SCALE = 100                    # costs are -log p in [0, ~6.9]; scaled to integers for the max-flow


@dataclass(frozen=True)
class OptSurfaceParams:
    margin: int = 4                 # search window along the normal: the mask's w range +- margin voxels
    max_levels: int = 48            # at most this many height levels per column (coarser steps if the range is big)
    solve_grid: int = 64            # the min-cut runs on at most this many cells per side, then is upsampled
    max_slope: float = 1.0          # |dw| per voxel of in-plane distance between neighbouring cells (1 = 45 deg)
    prior_weight: float = 0.05      # lam: pull toward rank 2's griddata height, per voxel of distance
    p_floor: float = 1e-3           # p is clipped to [p_floor, 1] before -log
    suppress_other_sheets: bool = True


# ---------------------------------------------------------------- the solver
def optimal_height_field(cost: np.ndarray, delta: int) -> np.ndarray:
    """Exact argmin over h of sum cost[i, j, h[i, j]] with |h(a) - h(b)| <= delta for 4-neighbours.

    cost: (G1, G2, K) non-negative integers. Returns h: (G1, G2) int in [0, K).
    Minimum closed set construction of Li et al. 2006: node (i, j, k) has weight cost[k] - cost[k-1]; a closed set
    holds every node below a chosen one in the same column and every node from k - delta down in the neighbours.
    """
    g1, g2, k = cost.shape
    n = g1 * g2 * k
    idx = np.arange(n).reshape(g1, g2, k)
    cost = cost.astype(np.int64)

    w = cost.copy()
    w[..., 1:] = cost[..., 1:] - cost[..., :-1]
    # every non-empty closed set holds all bottom nodes; make it cheaper than the empty set
    w[0, 0, 0] -= int(cost.max(axis=2).sum()) + 1

    src, snk = n, n + 1
    rows, cols, caps = [], [], []

    def add(a, b, c):
        rows.append(a.ravel()); cols.append(b.ravel()); caps.append(np.broadcast_to(c, a.shape).ravel())

    neg, pos = w < 0, w > 0
    add(np.full(neg.sum(), src), idx[neg], -w[neg])
    add(idx[pos], np.full(pos.sum(), snk), w[pos])
    # intra-column: (k) -> (k-1)
    add(idx[..., 1:], idx[..., :-1], INF)
    # inter-column: (a, k) -> (b, max(0, k - delta)), both directions, 4-neighbours
    kk = np.maximum(np.arange(k) - delta, 0)
    for a, b in ((idx[1:], idx[:-1]), (idx[:-1], idx[1:]), (idx[:, 1:], idx[:, :-1]), (idx[:, :-1], idx[:, 1:])):
        add(a, b[..., kk], INF)

    r = np.concatenate(rows); c = np.concatenate(cols); cap = np.concatenate(caps)
    # duplicates (e.g. several k clipped onto the same bottom node) only come from INF arcs; keep one copy
    key = r.astype(np.int64) * (n + 2) + c
    key, first = np.unique(key, return_index=True)
    r, c, cap = r[first], c[first], np.minimum(cap[first], INF)
    graph = sparse.csr_matrix((cap.astype(np.int32), (r, c)), shape=(n + 2, n + 2))

    flow = maximum_flow(graph, src, snk, method="dinic").flow
    fpos = flow.maximum(0) if hasattr(flow, "maximum") else sparse.csr_matrix(np.maximum(flow, 0))
    resid = (graph.astype(np.int64) - fpos + fpos.T).tocsr()
    resid.data = (resid.data > 0).astype(np.int8)
    resid.eliminate_zeros()
    reach = breadth_first_order(resid, src, directed=True, return_predecessors=False)

    in_set = np.zeros(n + 2, bool)
    in_set[reach] = True
    h = in_set[:n].reshape(g1, g2, k).sum(axis=2) - 1
    return np.clip(h, 0, k - 1)


# ---------------------------------------------------------------- the fit
def _griddata_height(uv, wv, grid_u, grid_v):
    if len(uv) > 5000:
        pick = np.random.default_rng(0).choice(len(uv), 5000, replace=False)
        uv, wv = uv[pick], wv[pick]
    try:
        h = griddata(uv, wv, (grid_u, grid_v), method="linear")
    except Exception:
        h = griddata(uv, wv, (grid_u, grid_v), method="nearest")
    if np.isnan(h).any():
        m = np.isnan(h)
        h[m] = griddata(uv, wv, (grid_u, grid_v), method="nearest")[m]
    return ndi.median_filter(h, size=3)


def fit_optimal_surface(component, evidence, grid_resolution=100, thickness=3, smoothing=1.0, samples_per_edge=8,
                        adaptive_resolution=True, p: OptSurfaceParams = OptSurfaceParams()):
    """Same contract as `rank2_reference.fit_curved_sheet_to_component_optimized`, but the height map comes from a
    min-cut over `evidence` (a probability cube aligned with `component`)."""
    coords = np.column_stack(np.nonzero(component))
    if len(coords) < 10:
        return component.copy()
    if adaptive_resolution:
        grid_resolution = ref.adaptive_grid_resolution(component, grid_resolution)
    G = grid_resolution

    mean = coords.mean(axis=0)
    _, _, vt = np.linalg.svd(coords - mean, full_matrices=False)
    t1, t2, nrm = vt[0], vt[1], vt[2]
    uv = (coords - mean) @ np.column_stack([t1, t2])
    wv = (coords - mean) @ nrm

    u_min, u_max = uv[:, 0].min(), uv[:, 0].max()
    v_min, v_max = uv[:, 1].min(), uv[:, 1].max()
    pu, pv = (u_max - u_min) * 0.05, (v_max - v_min) * 0.05
    gu = np.linspace(u_min - pu, u_max + pu, G)
    gv = np.linspace(v_min - pv, v_max + pv, G)
    grid_u, grid_v = np.meshgrid(gu, gv, indexing="ij")
    w_ref = _griddata_height(uv, wv, grid_u, grid_v)

    # --- min-cut on a (possibly coarser) grid
    S = min(G, p.solve_grid)
    su, sv = np.linspace(gu[0], gu[-1], S), np.linspace(gv[0], gv[-1], S)
    s_u, s_v = np.meshgrid(su, sv, indexing="ij")
    w_ref_s = ndi.zoom(w_ref, S / G, order=1) if S != G else w_ref
    w_ref_s = w_ref_s[:S, :S]

    w_lo, w_hi = wv.min() - p.margin, wv.max() + p.margin
    n_levels = int(np.ceil(w_hi - w_lo)) + 1
    step = max(1.0, (w_hi - w_lo) / (p.max_levels - 1)) if n_levels > p.max_levels else 1.0
    levels = np.arange(w_lo, w_hi + 1e-6, step)
    K = len(levels)

    pts = (mean[None, None, None, :]
           + s_u[..., None, None] * t1 + s_v[..., None, None] * t2
           + levels[None, None, :, None] * nrm)                         # (S, S, K, 3)
    pr = ndi.map_coordinates(evidence.astype(np.float32), pts.reshape(-1, 3).T, order=1, mode="constant", cval=0.0)
    pr = pr.reshape(S, S, K)
    cost = -np.log(np.clip(pr, p.p_floor, 1.0))
    cost += p.prior_weight * np.abs(levels[None, None, :] - w_ref_s[..., None])
    cost = np.round(cost * COST_SCALE).astype(np.int64)

    cell = max((su[1] - su[0]) if S > 1 else 1.0, (sv[1] - sv[0]) if S > 1 else 1.0)
    delta = max(1, int(round(p.max_slope * cell / step)))
    h = optimal_height_field(cost, delta)
    w_s = levels[h]
    w_grid = ndi.zoom(w_s, G / S, order=1)[:G, :G] if S != G else w_s
    if w_grid.shape != (G, G):                                          # zoom rounding
        w_grid = np.pad(w_grid, [(0, G - w_grid.shape[0]), (0, G - w_grid.shape[1])], mode="edge")

    if smoothing > 0 and not ref.should_skip_smoothing(component):
        w_grid = ndi.gaussian_filter(w_grid, sigma=smoothing)

    # --- from here on as in the reference: trim the grid outside the data, rasterise, thicken, fill
    tree = cKDTree(uv)
    thr = (u_max - u_min + v_max - v_min) / (2 * G) * 2
    dist, _ = tree.query(np.column_stack([grid_u.ravel(), grid_v.ravel()]), k=1)
    data = dist.reshape(G, G) <= thr
    lab, _ = ndi.label(~data, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    edge_labels = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    w_grid = w_grid.astype(np.float64)
    w_grid[np.isin(lab, edge_labels[edge_labels > 0])] = np.nan

    grid_points = mean + grid_u[..., None] * t1 + grid_v[..., None] * t2 + w_grid[..., None] * nrm
    sheet = np.zeros_like(component, dtype=bool)
    ref.rasterize_surface_numba(grid_points, sheet, samples_per_edge=samples_per_edge)
    sheet = ref.zero_volume_faces(sheet, thickness=5)
    if thickness > 0:
        se = np.array([[[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                       [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                       [[0, 1, 0], [1, 1, 1], [0, 1, 0]]], dtype=bool)
        sheet = ndi.binary_dilation(sheet, structure=se, iterations=max(1, thickness // 2))
    for z in range(sheet.shape[0]):
        if sheet[z].any():
            sheet[z] = ndi.binary_fill_holes(sheet[z])
    return sheet


# ---------------------------------------------------------------- hooking into rank 2
@contextlib.contextmanager
def use_optimal_surface(prob: np.ndarray, base: np.ndarray, p: OptSurfaceParams = OptSurfaceParams()):
    """While active, rank 2's cube-region interpolation fits sheets with `fit_optimal_surface`.

    prob / base: the probability map and base mask, cropped exactly like the volume given to
    `process_multiple_components_patchwise` (cube coordinates index into them).
    """
    orig_region = ref.interpolate_cube_region_with_quality_check
    orig_fit = ref.fit_curved_sheet_to_component_optimized
    current: dict = {}

    def region(component, cube_coords, *args, **kwargs):
        z0, z1, y0, y1, x0, x1 = cube_coords
        current["prob"] = prob[z0:z1, y0:y1, x0:x1]
        current["base"] = base[z0:z1, y0:y1, x0:x1]
        try:
            return orig_region(component, cube_coords, *args, **kwargs)
        finally:
            current.clear()

    def fit(component, grid_resolution=100, thickness=3, smoothing=1.0, use_median_filter=True, max_distance=10,
            use_numba=True, adaptive_resolution=True, samples_per_edge=8):
        cube_prob = current.get("prob")
        if cube_prob is None or cube_prob.shape != component.shape:
            return orig_fit(component, grid_resolution, thickness, smoothing, use_median_filter, max_distance,
                            use_numba, adaptive_resolution, samples_per_edge)
        evidence = cube_prob
        if p.suppress_other_sheets:
            evidence = np.where(current["base"] & ~component, 0.0, cube_prob)
        try:
            return fit_optimal_surface(component, evidence, grid_resolution, thickness, smoothing,
                                       samples_per_edge, adaptive_resolution, p)
        except Exception:
            return orig_fit(component, grid_resolution, thickness, smoothing, use_median_filter, max_distance,
                            use_numba, adaptive_resolution, samples_per_edge)

    ref.interpolate_cube_region_with_quality_check = region
    ref.fit_curved_sheet_to_component_optimized = fit
    try:
        yield
    finally:
        ref.interpolate_cube_region_with_quality_check = orig_region
        ref.fit_curved_sheet_to_component_optimized = orig_fit
