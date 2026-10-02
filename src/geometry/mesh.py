"""
FEM mesh generation and matrix assembly for 2D polygonal rooms.
"""

import numpy as np
import triangle
from scipy import sparse


def mesh_room(vertices, segments, max_area, min_angle=30):
    """
    Mesh a polygonal room using Triangle.

    Args:
        vertices: (N, 2) array of polygon vertices
        segments: (N, 2) array of segment index pairs
        max_area: Maximum triangle area in m²
        min_angle: Minimum angle in degrees

    Returns:
        mesh dict with 'vertices', 'triangles', 'segments', 'segment_markers'
    """
    seg_markers = np.arange(1, len(segments) + 1, dtype=int)
    inp = {
        "vertices": np.array(vertices, dtype=float),
        "segments": np.array(segments, dtype=int),
        "segment_markers": seg_markers,
    }
    opts = f"pq{min_angle}a{max_area:.8f}"
    return triangle.triangulate(inp, opts)


def assemble_stiffness_mass(vertices, triangles):
    """
    Assemble stiffness K and consistent mass M matrices for P1 elements.

    K_ij = integral(grad psi_i . grad psi_j, Omega)
    M_ij = integral(psi_i psi_j, Omega)
    """
    n_nodes = len(vertices)
    n_tri = len(triangles)

    rows_K = np.zeros(n_tri * 9, dtype=int)
    cols_K = np.zeros(n_tri * 9, dtype=int)
    vals_K = np.zeros(n_tri * 9)
    rows_M = np.zeros(n_tri * 9, dtype=int)
    cols_M = np.zeros(n_tri * 9, dtype=int)
    vals_M = np.zeros(n_tri * 9)

    for e in range(n_tri):
        i0, i1, i2 = triangles[e]
        x0, y0 = vertices[i0]
        x1, y1 = vertices[i1]
        x2, y2 = vertices[i2]

        area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))

        grad = np.array([
            [y1 - y2, x2 - x1],
            [y2 - y0, x0 - x2],
            [y0 - y1, x1 - x0],
        ]) / (2.0 * area)

        K_elem = area * (grad @ grad.T)
        M_elem = (area / 12.0) * (np.ones((3, 3)) + np.eye(3))

        nodes = [i0, i1, i2]
        for a in range(3):
            for b in range(3):
                idx = e * 9 + a * 3 + b
                rows_K[idx] = nodes[a]
                cols_K[idx] = nodes[b]
                vals_K[idx] = K_elem[a, b]
                rows_M[idx] = nodes[a]
                cols_M[idx] = nodes[b]
                vals_M[idx] = M_elem[a, b]

    K = sparse.coo_matrix(
        (vals_K, (rows_K, cols_K)), shape=(n_nodes, n_nodes)
    ).tocsr()
    M = sparse.coo_matrix(
        (vals_M, (rows_M, cols_M)), shape=(n_nodes, n_nodes)
    ).tocsr()
    return K, M


def assemble_robin_boundary(vertices, boundary_segments, segment_markers,
                            wall_kappas):
    """
    Assemble Robin boundary matrix B.
    B_ij = integral(kappa psi_i psi_j, dOmega)

    For a boundary edge (a, b) with length L and Robin coefficient kappa:
    B_edge = kappa * L/6 * [[2, 1], [1, 2]]
    """
    n_nodes = len(vertices)
    rows, cols, vals = [], [], []

    for k, seg in enumerate(boundary_segments):
        marker = segment_markers[k]
        wall_idx = marker - 1
        if wall_idx < 0 or wall_idx >= len(wall_kappas):
            continue
        kappa = wall_kappas[wall_idx]

        i0, i1 = seg
        L = np.linalg.norm(vertices[i1] - vertices[i0])
        coeff = kappa * L / 6.0
        B_local = coeff * np.array([[2.0, 1.0], [1.0, 2.0]])

        for a in range(2):
            for b in range(2):
                rows.append([i0, i1][a])
                cols.append([i0, i1][b])
                vals.append(B_local[a, b])

    return sparse.coo_matrix(
        (vals, (rows, cols)), shape=(n_nodes, n_nodes)
    ).tocsr()
