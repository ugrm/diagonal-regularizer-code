"""
P(p) landscape sweep and indifference band analysis.
"""

import numpy as np

from ..estimation.ridge import build_gamma_diag, ridge_sweep_svd
from ..estimation.metrics import compute_P


def sweep_p_for_room(room, T_values, p_grid, lambda_grid, M_values,
                     mic_subsets_4=None, is_wave=True, gamma=None, c=343.0,
                     skip_first=5, skip_last=2):
    """
    Sweep P(p) for one room across all (T, M, p) combinations.

    Args:
        room: Dict from load_room_data()
        T_values: Array of T values to evaluate
        p_grid: Array of p values (e.g., 0.0 to 6.0 step 0.1)
        lambda_grid: Array of alpha values
        M_values: List of M values (e.g., [4, 8])
        mic_subsets_4: (20, 4) array of mic indices for M=4 trials
        is_wave: True for acoustic, False for heat
        gamma: Damping (acoustic)
        c: Speed of sound

    Returns:
        P_oracle: (n_T, n_M, n_p) best-alpha P at each (T, M, p)
    """
    from ..physics.temporal import build_wave_temporal_matrix, build_heat_temporal_matrix

    K = room["K"]
    dt = room["dt_sim"]
    sps = room["steps_per_snap"]
    n_snaps = room["n_snaps"]
    Phi = room["Phi"]
    ev = room["eigenvalues"]
    y_click = room["y_click"]
    a = room["a"]

    if gamma is None:
        gamma = room.get("gamma_room", 5.1)

    n_T = len(T_values)
    n_M = len(M_values)
    n_p = len(p_grid)

    gamma_diags = [build_gamma_diag(ev, p, is_wave) for p in p_grid]

    P_oracle = np.full((n_T, n_M, n_p), np.nan)

    for t_idx, T_raw in enumerate(T_values):
        T_raw = int(T_raw)

        # Valid snapshots
        valid = []
        T_sig = y_click.shape[1]
        for si in range(skip_first, n_snaps - skip_last):
            sim_step = si * sps
            end = sim_step + 1
            start = end - T_raw
            if start >= 0 and end <= T_sig:
                valid.append(si)
        if len(valid) < 5:
            continue

        targets = a[:, valid].T  # (N, K)

        # Build temporal matrix for full 8 mics
        M_full = Phi.shape[0]
        if is_wave:
            A_full = build_wave_temporal_matrix(Phi, ev, dt, T_raw, gamma, c)
        else:
            A_full = build_heat_temporal_matrix(Phi, ev, dt, T_raw)

        # Extract observation windows
        N_win = len(valid)
        Y_flat = np.empty((N_win, M_full * T_raw), dtype=np.float64)
        for i, si in enumerate(valid):
            sim_step = si * sps
            end = sim_step + 1
            start = end - T_raw
            Y_flat[i] = y_click[:, start:end].reshape(-1)

        for m_idx, M_val in enumerate(M_values):
            if M_val == M_full:
                mic_sets = [tuple(range(M_full))]
            elif mic_subsets_4 is not None and M_val == 4:
                mic_sets = [tuple(int(x) for x in row) for row in mic_subsets_4]
            else:
                mic_sets = [tuple(range(M_val))]

            trial_results = []
            for mic_idx in mic_sets:
                row_idx = np.concatenate([
                    np.arange(m * T_raw, (m + 1) * T_raw) for m in mic_idx
                ])
                A_sub = A_full[row_idx]
                Y_sub = Y_flat[:, row_idx]

                P_per_p = np.full(n_p, np.nan)
                for p_idx, gamma_diag in enumerate(gamma_diags):
                    P_vals, _ = ridge_sweep_svd(
                        A_sub, Y_sub, targets, gamma_diag, lambda_grid, is_wave
                    )
                    P_per_p[p_idx] = np.nanmin(P_vals)
                trial_results.append(P_per_p)

            P_median = np.nanmedian(np.array(trial_results), axis=0)
            P_oracle[t_idx, m_idx] = P_median

    return P_oracle


def find_p_star(P_median_curve, p_grid):
    """Find p* = argmin of median P(p) curve."""
    idx = np.nanargmin(P_median_curve)
    return float(p_grid[idx])


def compute_indifference_band(P_median_curve, p_grid, threshold_pct=1.0):
    """
    Compute indifference band: range of p where P is within threshold_pct of minimum.

    Returns (p_low, p_high).
    """
    P_min = np.nanmin(P_median_curve)
    threshold = P_min * (1 + threshold_pct / 100)
    in_band = P_median_curve <= threshold

    valid_p = p_grid[in_band]
    if len(valid_p) == 0:
        return (np.nan, np.nan)
    return (float(valid_p[0]), float(valid_p[-1]))
