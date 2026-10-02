"""
FEM eigenvalue problem solver for the Robin Laplacian.

Solves: (K + B) phi = lambda M phi
  K = stiffness (Laplacian)
  M = mass
  B = Robin boundary
"""

import numpy as np
from scipy.sparse.linalg import eigsh


def solve_eigenproblem(K, B, M, n_modes, sigma=0.0):
    """
    Solve (K + B) phi = lambda M phi using shift-invert.

    Returns eigenvalues and mass-orthonormalized eigenvectors.
    """
    A = K + B
    eigenvalues, eigenvectors = eigsh(A, k=n_modes, M=M, sigma=sigma,
                                       which="LM")
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]


def verify_eigenpairs(eigenvalues, eigenvectors, M_mat, c, verbose=False):
    """
    Verify eigenvalue properties and mass-orthonormality.

    Returns (frequencies, max_orthonormality_error).
    """
    n_modes = len(eigenvalues)
    freqs = c * np.sqrt(np.maximum(eigenvalues, 0)) / (2 * np.pi)

    G = eigenvectors.T @ M_mat @ eigenvectors
    ortho_error = np.max(np.abs(G - np.eye(n_modes)))

    if verbose:
        print(f"  Eigenfrequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
        print(f"  Mass-orthonormality error: {ortho_error:.2e}")
        if eigenvalues.min() < -1e-10:
            print(f"  WARNING: negative eigenvalue: {eigenvalues.min():.6e}")

    return freqs, ortho_error


def estimate_n_modes_weyl(area, fcut, c, margin=1.1):
    """
    Estimate number of modes from Weyl's law: N(f) ~ pi*A*f^2/c^2.

    Args:
        area: Room area in m^2
        fcut: Cutoff frequency in Hz
        c: Speed of sound in m/s
        margin: Safety margin (default 1.1 = 10% extra)

    Returns:
        Number of modes to request
    """
    n_modes_weyl = int(np.ceil(np.pi * area * fcut**2 / c**2))
    return max(int(n_modes_weyl * margin), 10)
