"""
Cost analysis: Method A and Method B cost of using |s| instead of p*.
"""

import numpy as np

from ..estimation.metrics import method_A_cost, method_B_cost


def compute_cost_table(P_per_room, p_grid, T_values, s_value):
    """
    Compute cost table for all T values.

    Args:
        P_per_room: (n_rooms, n_T, n_p) array
        p_grid: (n_p,) array
        T_values: (n_T,) array
        s_value: |s| value

    Returns:
        Dict with Method A and Method B costs at each T
    """
    n_T = len(T_values)

    results = {
        "T": T_values,
        "method_A_abs": np.full(n_T, np.nan),
        "method_A_rel": np.full(n_T, np.nan),
        "method_B_abs": np.full(n_T, np.nan),
        "method_B_rel": np.full(n_T, np.nan),
        "p_star_A": np.full(n_T, np.nan),
    }

    for t_idx in range(n_T):
        P_median_curve = np.nanmedian(P_per_room[:, t_idx, :], axis=0)

        # Method A
        abs_A, rel_A, p_star = method_A_cost(P_median_curve, p_grid, s_value)
        results["method_A_abs"][t_idx] = abs_A
        results["method_A_rel"][t_idx] = rel_A
        results["p_star_A"][t_idx] = p_star

        # Method B
        abs_B, rel_B, _, _ = method_B_cost(
            P_per_room[:, t_idx, :], p_grid, s_value
        )
        results["method_B_abs"][t_idx] = abs_B
        results["method_B_rel"][t_idx] = rel_B

    return results
