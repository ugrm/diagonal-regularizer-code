"""
Tikhonov / Ridge regression with power-law regularization Gamma_k = lambda_k^p.
"""

import numpy as np

from .metrics import compute_P


def build_gamma_diag(eigenvalues, p, is_wave):
    """
    Build diagonal of Gamma for power-law regularization Gamma = diag(lambda_k^p).

    For wave equation (is_wave=True), state is interleaved [a_1, adot_1, a_2, ...],
    so Gamma has 2K entries with duplicated eigenvalue exponents.
    """
    ev_p = np.power(eigenvalues, p)
    if is_wave:
        K = len(eigenvalues)
        g = np.empty(2 * K)
        g[0::2] = ev_p
        g[1::2] = ev_p
        return g
    return ev_p


def ridge_solve(A, Y, gamma_diag, alpha):
    """
    Solve ridge regression: x = (A^T A + alpha * diag(gamma))^{-1} A^T y.

    Args:
        A: (MT, n_cols) temporal matrix
        Y: (N, MT) observation windows (N samples, MT measurements each)
        gamma_diag: (n_cols,) diagonal of regularization matrix
        alpha: Regularization strength

    Returns:
        X: (n_cols, N) estimated state vectors
    """
    ATA = A.T @ A
    ATY = A.T @ Y.T
    reg = ATA + alpha * np.diag(gamma_diag)
    return np.linalg.solve(reg, ATY)


def ridge_sweep_svd(A, Y, targets, gamma_diag, lambda_grid, is_wave):
    """
    Sweep alpha values for a single (room, T, M, p) using SVD for efficiency.

    Args:
        A: (MT, n_cols) temporal matrix
        Y: (N, MT) observation windows
        targets: (N, K) ground truth modal amplitudes
        gamma_diag: (n_cols,) Gamma diagonal for this p
        lambda_grid: Array of alpha values to try
        is_wave: If True, extract a_n from interleaved state [a_n, adot_n, ...]

    Returns:
        P_values: (len(lambda_grid),) P_modal at each alpha
        best_alpha_idx: Index of best alpha
    """
    n_lambda = len(lambda_grid)
    P_values = np.full(n_lambda, np.nan)

    ATA = A.T @ A
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s2 = s**2
    UTY = U.T @ Y.T
    ATY = (Vt.T * s[None, :]) @ UTY

    p_val = gamma_diag[0] if len(gamma_diag) > 0 else 0.0

    for l_idx, lam in enumerate(lambda_grid):
        if p_val == 0.0 and (gamma_diag == 1.0).all():
            # Gamma=I: efficient SVD shrinkage
            d = s / (s2 + lam)
            X_r = (Vt.T * d[None, :]) @ UTY
        else:
            reg = ATA + lam * np.diag(gamma_diag)
            try:
                X_r = np.linalg.solve(reg, ATY)
            except np.linalg.LinAlgError:
                continue

        preds = X_r[0::2, :].T if is_wave else X_r.T
        P_values[l_idx] = compute_P(preds, targets)

    best_idx = np.nanargmin(P_values)
    return P_values, best_idx
