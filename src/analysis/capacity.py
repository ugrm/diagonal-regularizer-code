"""
Capacity check analysis: p_target vs p_hat extraction.

Tests whether learned models can recover arbitrary Gamma spectra,
not just |s|-optimal ones.
"""

import numpy as np

from ..estimation.metrics import fit_power_law


def extract_p_hat(gamma_k, eigenvalues):
    """
    Fit log(Gamma_k) vs log(lambda_k) to extract learned exponent p_hat.

    Args:
        gamma_k: (K,) learned regularization spectrum
        eigenvalues: (K,) eigenvalues

    Returns:
        p_hat: Fitted exponent
        r2: R^2 of fit
    """
    slope, _, r2, _ = fit_power_law(eigenvalues, gamma_k)
    return float(slope) if not np.isnan(slope) else np.nan, float(r2)


def capacity_check_analysis(p_hat_per_target, p_targets):
    """
    Analyze capacity check results: deviation from diagonal.

    Args:
        p_hat_per_target: (n_targets, n_seeds) extracted p_hat values
        p_targets: (n_targets,) target p values

    Returns:
        Dict with deviation stats
    """
    median_p_hat = np.nanmedian(p_hat_per_target, axis=1)
    deviation = median_p_hat - np.array(p_targets)
    return {
        "p_targets": p_targets,
        "median_p_hat": median_p_hat.tolist(),
        "deviation": deviation.tolist(),
        "max_deviation": float(np.max(np.abs(deviation))),
    }
