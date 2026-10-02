"""
Modal decomposition: FDTD-to-modal projection and measurement matrix construction.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import Delaunay


def interpolate_fdtd_to_fem(snapshot, domain_x, domain_y, fem_vertices):
    """Interpolate a single FDTD snapshot (regular grid) to FEM mesh nodes."""
    nx_out, ny_out = snapshot.shape
    xs = np.linspace(0, domain_x, nx_out)
    ys = np.linspace(0, domain_y, ny_out)
    interp = RegularGridInterpolator(
        (xs, ys), snapshot, method="linear", bounds_error=False, fill_value=0.0
    )
    return interp(fem_vertices)


def project_fdtd_to_modes(snapshots, domain_x, domain_y, mesh_vertices,
                          eigenvectors, M_mat, verbose=False):
    """
    Project ALL FDTD snapshots onto modal basis.

    Args:
        snapshots: (T, nx, ny) FDTD snapshots
        domain_x, domain_y: Physical domain size
        mesh_vertices: (N_nodes, 2) FEM mesh nodes
        eigenvectors: (N_nodes, K) mass-orthonormal eigenvectors
        M_mat: Mass matrix (sparse)

    Returns:
        a: (K, T) modal coefficients
    """
    T_snap = len(snapshots)
    K = eigenvectors.shape[1]

    a = np.zeros((K, T_snap), dtype=np.float64)
    PhiT_M = eigenvectors.T @ M_mat  # (K, N_nodes)

    for t in range(T_snap):
        u_fem = interpolate_fdtd_to_fem(
            snapshots[t], domain_x, domain_y, mesh_vertices
        )
        a[:, t] = PhiT_M @ u_fem

    if verbose:
        rms = float(np.sqrt(np.mean(a**2)))
        print(f"  Modal projection: K={K}, T={T_snap}, RMS(a)={rms:.4e}")

    return a


def build_measurement_matrix(mic_pos, mesh_vertices, mesh_triangles,
                              eigenvectors):
    """
    Build Phi in R^{M x K} where Phi_mk = phi_k(x_m).

    Uses barycentric interpolation on the P1 FEM mesh.
    """
    M = len(mic_pos)
    K = eigenvectors.shape[1]

    tri = Delaunay(mesh_vertices)
    Phi = np.zeros((M, K))
    n_fallback = 0

    for m in range(M):
        pt = mic_pos[m]
        simplex_idx = tri.find_simplex(pt)

        if simplex_idx < 0:
            # Point outside mesh — nearest node fallback
            dists = np.linalg.norm(mesh_vertices - pt, axis=1)
            nearest = np.argmin(dists)
            Phi[m] = eigenvectors[nearest]
            n_fallback += 1
            continue

        tri_nodes = tri.simplices[simplex_idx]
        v0, v1, v2 = mesh_vertices[tri_nodes]

        T = np.array([[v0[0] - v2[0], v1[0] - v2[0]],
                       [v0[1] - v2[1], v1[1] - v2[1]]])
        det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]

        if abs(det) < 1e-30:
            dists = np.linalg.norm(mesh_vertices - pt, axis=1)
            nearest = np.argmin(dists)
            Phi[m] = eigenvectors[nearest]
            n_fallback += 1
            continue

        dp = pt - v2
        lam0 = (T[1, 1] * dp[0] - T[0, 1] * dp[1]) / det
        lam1 = (-T[1, 0] * dp[0] + T[0, 0] * dp[1]) / det
        lam2 = 1.0 - lam0 - lam1

        Phi[m] = (lam0 * eigenvectors[tri_nodes[0]]
                  + lam1 * eigenvectors[tri_nodes[1]]
                  + lam2 * eigenvectors[tri_nodes[2]])

    if n_fallback > 0:
        print(f"  WARNING: {n_fallback} mics outside mesh, used nearest-node fallback")

    return Phi
