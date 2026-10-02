"""
Performance metrics: P_modal, cost analysis (Method A, Method B).

P = MSE / var(u_gt) — proportion of variance unexplained.
P ~ 1.0 = predicting mean; P = 0 = perfect.
"""

import numpy as np


def compute_P(preds, targets):
    """
    Compute P_modal = sum(MSE_per_mode) / sum(var_per_mode).

    Args:
        preds: (N, K) predicted modal amplitudes
        targets: (N, K) ground truth modal amplitudes

    Returns:
        P_modal scalar (nan if targets have near-zero variance)
    """
    mse_per_mode = np.mean((preds - targets)**2, axis=0)
    var_per_mode = np.var(targets, axis=0)
    total_var = np.sum(var_per_mode)
    if total_var < 1e-30:
        return np.nan
    return float(np.sum(mse_per_mode) / total_var)


def method_A_cost(P_median_curve, p_grid, s_value):
    """
    Method A: Cost of using |s| on the population median P(p) curve.

    cost_A = P_median(|s|) - min_p P_median(p)
    relative_cost_A = cost_A / min_p P_median(p)

    Args:
        P_median_curve: (n_p,) median P across rooms at each p
        p_grid: (n_p,) p values
        s_value: |s| value (the "free" regularization exponent)

    Returns:
        (absolute_cost, relative_cost, p_star)
    """
    p_star_idx = np.nanargmin(P_median_curve)
    p_star = p_grid[p_star_idx]
    P_star = P_median_curve[p_star_idx]

    # Interpolate P at |s|
    P_at_s = np.interp(s_value, p_grid, P_median_curve)

    delta_P = P_at_s - P_star
    relative = delta_P / P_star if P_star > 1e-30 else np.nan

    return float(delta_P), float(relative), float(p_star)


def method_B_cost(P_per_room, p_grid, s_value):
    """
    Method B: Median of per-room costs (correct aggregation).

    For each room: delta_P(r) = P(|s|, r) - min_p P(p, r)
                   delta(r)   = delta_P(r) / min_p P(p, r)
    Then take medians across rooms.

    Args:
        P_per_room: (n_rooms, n_p) P at each p for each room
        p_grid: (n_p,) p values
        s_value: |s| value

    Returns:
        (median_delta_P, median_relative, per_room_delta_P, per_room_relative)
    """
    n_rooms = P_per_room.shape[0]
    delta_P = np.full(n_rooms, np.nan)
    delta_rel = np.full(n_rooms, np.nan)

    for r in range(n_rooms):
        curve = P_per_room[r]
        if np.all(np.isnan(curve)):
            continue
        P_star = np.nanmin(curve)
        P_at_s = np.interp(s_value, p_grid, curve)
        delta_P[r] = P_at_s - P_star
        if P_star > 1e-30:
            delta_rel[r] = delta_P[r] / P_star

    return (
        float(np.nanmedian(delta_P)),
        float(np.nanmedian(delta_rel)),
        delta_P,
        delta_rel,
    )


def fit_power_law(x, y):
    """
    Fit log(y) = slope * log(x) + intercept on valid data.

    Returns (slope, intercept, R2, n_valid).
    """
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if valid.sum() < 5:
        return np.nan, np.nan, np.nan, 0

    log_x = np.log(x[valid])
    log_y = np.log(y[valid])
    slope, intercept = np.polyfit(log_x, log_y, 1)
    pred = slope * log_x + intercept
    ss_res = np.sum((log_y - pred)**2)
    ss_tot = np.sum((log_y - np.mean(log_y))**2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

    return float(slope), float(intercept), float(r2), int(valid.sum())
