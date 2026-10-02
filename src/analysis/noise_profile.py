"""
Noise profile analysis: truncation noise (Approach A/B) and prior spectral decay |s|.
"""

import numpy as np

from ..estimation.metrics import fit_power_law


def approach_B(Phi_full, a, K_use):
    """
    Approach B: FEM-only cross-correlation (mismatch-immune).

    C_{kn} = (1/M) sum_m Phi_{mk} Phi_{mn}
    sigma^2_trunc,k(B) = M^2 * sum_{n>K} sigma^2_{a,n} * C_{kn}^2

    Returns sigma^2_trunc,k(B) array of shape (K_use,).
    """
    M = Phi_full.shape[0]
    K_total = Phi_full.shape[1]

    if K_total <= K_use:
        return np.full(K_use, np.nan)

    Phi_retained = Phi_full[:, :K_use]
    Phi_trunc = Phi_full[:, K_use:]
    a_trunc = a[K_use:]

    C = Phi_retained.T @ Phi_trunc / M
    sigma_sq_a_trunc = np.var(a_trunc, axis=1)
    sigma_sq_trunc_B = M**2 * (C**2 @ sigma_sq_a_trunc)

    return sigma_sq_trunc_B


def approach_A(Phi_full, a, y_click, K_use, sps, n_snaps,
               skip_first=5, skip_last=2, max_snaps=200):
    """
    Approach A: Total residual projected onto retained modes.

    Returns sigma^2_trunc,k(A) array of shape (K_use,).
    """
    Phi_K = Phi_full[:, :K_use]
    phi_norms_sq = np.sum(Phi_K**2, axis=0)

    snap_range = range(skip_first, min(n_snaps - skip_last, max_snaps))
    r_k_all = []

    for si in snap_range:
        sim_step = si * sps
        if sim_step >= y_click.shape[1]:
            break

        y_fdtd = y_click[:, sim_step]
        y_K = Phi_K @ a[:K_use, si]
        residual = y_fdtd - y_K
        r_k = (Phi_K.T @ residual) / np.maximum(phi_norms_sq, 1e-30)
        r_k_all.append(r_k)

    r_k_all = np.array(r_k_all)
    return np.var(r_k_all, axis=0)


def measure_prior_s(a, eigenvalues, K_use, skip_first=5, skip_last=2,
                    max_snaps=200):
    """
    Measure prior spectral decay s from modal amplitudes.

    Returns:
        sigma_sq_all: (K_use,) variance across all valid snapshots
        s_full: slope from fit on full range
        r2_s_full: R^2 of the fit
    """
    all_snaps = range(skip_first, min(a.shape[1] - skip_last, max_snaps))
    sigma_sq_all = np.var(a[:K_use, list(all_snaps)], axis=1)

    ev = eigenvalues[:K_use]
    s_full, _, r2_s_full, _ = fit_power_law(ev, sigma_sq_all)

    return sigma_sq_all, s_full, r2_s_full
