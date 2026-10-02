"""
Robin BC rasterisation for 2D polygon rooms.
"""
import numpy as np
from matplotlib.path import Path as MplPath


def rasterise_room(vertices, nx, ny, dx):
    poly_path = MplPath(np.asarray(vertices))
    xs = np.arange(nx) * dx
    ys = np.arange(ny) * dx
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    return poly_path.contains_points(pts).reshape(nx, ny)


def build_kappa_field(vertices, kappas, nx, ny, dx, mask):
    verts = np.asarray(vertices)
    n_seg = len(verts)
    kappa_grid = np.zeros((nx, ny), dtype=np.float64)

    boundary = np.zeros_like(mask)
    boundary[1:-1, 1:-1] = mask[1:-1, 1:-1] & (
        ~mask[:-2, 1:-1] | ~mask[2:, 1:-1] |
        ~mask[1:-1, :-2] | ~mask[1:-1, 2:])

    bdy_idx = np.argwhere(boundary)
    if len(bdy_idx) == 0:
        return kappa_grid

    bdy_pts = bdy_idx.astype(np.float64) * dx
    for bi in range(len(bdy_pts)):
        pt = bdy_pts[bi]
        best_seg, best_dist = 0, np.inf
        for si in range(n_seg):
            a, b = verts[si], verts[(si + 1) % n_seg]
            ab = b - a
            t = np.clip(np.dot(pt - a, ab) / (np.dot(ab, ab) + 1e-30), 0, 1)
            dist = np.linalg.norm(pt - (a + t * ab))
            if dist < best_dist:
                best_dist, best_seg = dist, si
        kappa_grid[bdy_idx[bi, 0], bdy_idx[bi, 1]] = kappas[best_seg]
    return kappa_grid