"""
Figure 13: 2D Sweep Heatmaps — P(p,c) surface for heat vs acoustic.

Paper reference: Supplement
Data: data/supplementary/heat_2d_sweep_results.npz (pre-computed)

Layout: 2×4 heatmaps (rows = {acoustic, heat}, columns = T values).
Acoustic minimum at c~0; heat minimum at c>0.

Pre-computed by: scripts/appendix/E/run_heat_2d_sweep.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.visualization.style import (
    setup_style, save_figure, WONG,
    resolve_data_path, cli_wrapper, enable_all_spines,
    include_width, legend_right,
)


def generate(data_dir=None, output_dir="figures/", config=None):
    """Generate Figure 13: 2D Sweep Heatmaps."""
    setup_style()

    path = resolve_data_path(
        "heat_2d_sweep_results.npz", data_dir)
    d = np.load(path)

    P_heat_2d = d["P_heat_2d"]    # (n_rooms, n_T, n_p, n_c)
    P_wave_2d = d["P_wave_2d"]    # (n_rooms, n_T, n_p, n_c)
    T_values = d["T_values"]
    p_values = d["p_values"]
    c_values = d["c_values"]

    # Drop the T=1 column. At T=1 the heat optimum lands at a corner anomaly
    # (multiple (c, p) regions competitive); the cleaner story is at T>=10.
    T_show_mask = T_values != 1
    T_show = T_values[T_show_mask]
    n_T = len(T_show)

    # Drawn at full text width and included at \linewidth. The message of the
    # figure is WHERE each surface is lowest (c near zero for acoustics, c > 0
    # for heat), so every panel is shown on its own scale from its minimum to
    # its 90th percentile, and one colourbar describes that shared encoding.
    fig, axes = plt.subplots(2, n_T, figsize=(include_width(1.0), 2.25),
                             sharex=True, sharey=True)

    im = None
    for col, T_val in enumerate(T_show):
        T_idx = int(np.where(T_values == T_val)[0][0])

        # Median across rooms
        P_wave_med = np.median(P_wave_2d[:, T_idx, :, :], axis=0)
        P_heat_med = np.median(P_heat_2d[:, T_idx, :, :], axis=0)

        for row, (P_med, pde_label) in enumerate([
            (P_wave_med, "Acoustic"), (P_heat_med, "Heat")
        ]):
            ax = axes[row, col]
            enable_all_spines(ax)

            vmin = P_med.min()
            vmax = np.percentile(P_med, 90)
            P_scaled = np.clip((P_med - vmin) / max(vmax - vmin, 1e-30), 0, 1)
            im = ax.pcolormesh(c_values, p_values, P_scaled, cmap="viridis",
                               vmin=0, vmax=1, shading="auto",
                               rasterized=True, antialiased=False,
                               edgecolors="face", linewidth=0)
            ax.grid(False)

            idx_min = np.unravel_index(np.argmin(P_med), P_med.shape)
            ax.plot(c_values[idx_min[1]], p_values[idx_min[0]], "*",
                    color="white", ms=11, markeredgecolor=WONG["red"],
                    markeredgewidth=1.5, zorder=5,
                    label="minimum of\nmedian $P$")

            ax.set_title(f"{pde_label}, $T={int(T_val)}$")
            if col == 0:
                ax.set_ylabel("Exponent $p$")
            if row == 1:
                ax.set_xlabel("Rate $c$")

    RIGHT = 0.785
    fig.tight_layout(rect=(0.004, 0.006, RIGHT, 0.994), pad=0.6)
    legend_right(fig, axes=[axes[0, 0]], right=RIGHT, pack=False,
                 bbox_to_anchor=(RIGHT + 0.012, 0.86))
    cax = fig.add_axes([RIGHT + 0.04, 0.13, 0.02, 0.52])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("median $P$, rescaled per panel\n(minimum to 90th percentile)")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["low", "high"])
    save_figure(fig, "fig13_heat_2d_sweep", output_dir)
    return fig


if __name__ == "__main__":
    cli_wrapper(generate)
