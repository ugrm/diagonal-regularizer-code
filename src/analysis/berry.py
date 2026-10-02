"""
Berry conjecture verification: C_kn^2 statistics, anisotropy matrix E.
"""

import numpy as np
from scipy.special import j0


def compute_Ckn_squared(Phi, K_use, room_area):
    """
    Compute C_{kn} = (1/M) sum_m phi_k(x_m) phi_n(x_m) for k=1..K, n=K+1..K_total.

    Returns:
        mean_Ckn_sq: (K_use,) mean C_{kn}^2 per retained mode k
        theoretical: scalar 1/(M * |Omega|^2)
    """
    M = Phi.shape[0]
    K_total = Phi.shape[1]

    if K_total <= K_use:
        return np.full(K_use, np.nan), np.nan

    Phi_ret = Phi[:, :K_use]
    Phi_trunc = Phi[:, K_use:]

    C = Phi_ret.T @ Phi_trunc / M
    Ckn_sq = C ** 2
    mean_Ckn_sq = np.mean(Ckn_sq, axis=1)
    theoretical = 1.0 / (M * room_area**2)

    return mean_Ckn_sq, theoretical


def compute_anisotropy_E(Phi, a, eigenvalues, mic_pos, K_use):
    """
    Anisotropy matrix E:
    E_{mm'} = (1/sigma^2_trunc) sum_{n>K} sigma^2_{a,n} J_0(sqrt(lambda_n) * |x_m - x_m'|)

    Returns ||E||_op, ||E||_F, E matrix.
    """
    K_total = a.shape[0]
    M = Phi.shape[0]

    if K_total <= K_use:
        return np.nan, np.nan, np.full((M, M), np.nan)

    sigma_sq_trunc = np.var(a[K_use:], axis=1)
    sigma_sq_total = np.sum(sigma_sq_trunc)

    if sigma_sq_total < 1e-30:
        return np.nan, np.nan, np.full((M, M), np.nan)

    ev_trunc = eigenvalues[K_use:]
    omega_trunc = np.sqrt(np.maximum(ev_trunc, 0))

    E = np.zeros((M, M))
    for m in range(M):
        for mp in range(m + 1, M):
            dist = np.linalg.norm(mic_pos[m] - mic_pos[mp])
            j0_vals = j0(omega_trunc * dist)
            E_val = np.sum(sigma_sq_trunc * j0_vals) / sigma_sq_total
            E[m, mp] = E_val
            E[mp, m] = E_val

    eigenvals_E = np.linalg.eigvalsh(E)
    E_op = np.max(np.abs(eigenvals_E))
    E_fro = np.linalg.norm(E)

    return E_op, E_fro, E
