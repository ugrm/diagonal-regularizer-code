"""
Figure 17: Heat Fit Residuals (power-law only) — magnitude and curvature.

Paper reference: Appendix E (heat fit quality)
Data: legacy modal dataset (src.utils.paths.MODAL), room scene_00840
      (eigenvalues; analytic heat amplitudes, no microphone signals needed)

Layout: 1×3 (columns = t = 50/250/495 ms, the T_OBS of fig_heat_diagnostics).
Shows pure power-law residuals; the systematic U-shape and large magnitude
(up to 30 in log-space at late snapshots) carry information that fig11's
R² numbers do not. The exp×power residuals are not drawn: that fit gives
R² > 0.99 (shown by fig11), so its residuals are random scatter.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.visualization.fig_heat_diagnostics import (
    K, ROOM_SID, SEED, T_OBS, heat_amplitudes, load_diag_room,
)
from src.visualization.style import (
    setup_style, save_figure, FULL_WIDTH, WONG,
    enable_all_spines, add_grid, cli_wrapper, legend_bottom,
)


def generate(data_dir=None, output_dir="figures/", config=None):
    """Generate Figure 17: Heat Fit Residuals."""
    setup_style()

    room = load_diag_room()
    ev = room["eigenvalues"]
    room_idx = int(ROOM_SID.split("_")[1])

    n_seeds = 20

    fig, axes = plt.subplots(1, len(T_OBS), figsize=(FULL_WIDTH, 2.7))
    if len(T_OBS) == 1:
        axes = [axes]

    for idx, t_phys in enumerate(T_OBS):
        # Average over seeds
        a2_avg = np.zeros(K)
        for seed_off in range(n_seeds):
            a2_avg += heat_amplitudes(room, SEED + room_idx * 10 + 1 + seed_off, t_phys)**2
        a2_avg /= n_seeds

        valid = (a2_avg > 0) & (ev > 0)
        log_ev = np.log(ev[valid])
        log_a2 = np.log(a2_avg[valid])
        k_valid = np.where(valid)[0]

        # Pure power-law fit (the only fit shown; exp×power has R²>0.99 per fig11)
        coeffs_A = np.polyfit(log_ev, log_a2, 1)
        pred_A = coeffs_A[0] * log_ev + coeffs_A[1]
        residuals_A = pred_A - log_a2

        ax = axes[idx]
        enable_all_spines(ax)
        ax.scatter(k_valid, residuals_A, s=12, alpha=0.7,
                   color=WONG["red"], edgecolors="black", linewidth=0.3,
                   label="residual per mode")
        ax.axhline(0, color="black", linewidth=0.8, label="zero")
        ax.set_title(f"$t = {t_phys*1000:.0f}$ ms")
        ax.set_xlabel("Mode index $k$")
        if idx == 0:
            ax.set_ylabel("Residual (log units)")

        if len(k_valid) > 5:
            window = max(3, len(k_valid) // 10)
            rm_A = np.convolve(residuals_A, np.ones(window) / window, mode="valid")
            k_rm = k_valid[window // 2: window // 2 + len(rm_A)]
            ax.plot(k_rm, rm_A, "-", color=WONG["red"], lw=1.5, alpha=0.7,
                    label="running mean")

    legend_bottom(fig, axes=[axes[0]])
    save_figure(fig, "fig17_heat_fit_residuals", output_dir)
    return fig


if __name__ == "__main__":
    cli_wrapper(generate)
