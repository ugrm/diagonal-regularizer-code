"""
Figure 14: c vs t_target Tracking — Optimal exponential parameter tracks 2t.

Paper reference: Appendix E (cross-PDE consistency check, Table 1)
Data: data/supplementary/heat_c_continuous_fit.npz (pre-computed)

Single-panel: c_fit (continuous spectral fit) vs c_theory (= 2 κ t_obs) across
5 diagnostic rooms. Per-room slope and R² annotated in the legend; pooled
slope + R² annotated in the title. Numbers match §E Table 1.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.visualization.style import (
    setup_style, save_figure, WONG, resolve_data_path, cli_wrapper,
    enable_all_spines, add_grid, include_width, legend_right,
)


def generate(data_dir=None, output_dir="figures/", config=None):
    """Generate Figure 14: c vs t_target Tracking."""
    setup_style()

    path = resolve_data_path("heat_c_continuous_fit.npz", data_dir)
    d = np.load(path, allow_pickle=True)

    c_theory = d["c_theory"]
    c_fit = d["c_fit"]
    c_se = d["c_se"]
    room_ids = np.array([str(r) for r in d["room_ids"]])
    per_room = d["per_room_slopes"].item()
    slope_all5 = float(d["slope_all5"])
    r2_all5 = float(d["slope_all5_r2"])

    room_colors = [WONG["blue"], WONG["orange"], WONG["green"],
                   WONG["red"], WONG["purple"]]

    # Drawn at text width and included at \linewidth.
    fig, ax = plt.subplots(1, 1, figsize=(include_width(1.0), 2.6))
    enable_all_spines(ax)

    cmax = 0.0
    for i, sid in enumerate(sorted(set(room_ids))):
        mask = room_ids == sid
        ct = c_theory[mask]
        cf = c_fit[mask]
        se = c_se[mask]
        short = sid.split("_")[1]
        # Per-room slopes and R^2 live in the appendix table, not here.
        ax.errorbar(ct, cf, yerr=se, fmt="o",
                    color=room_colors[i % len(room_colors)],
                    ms=4, capsize=2, elinewidth=0.6,
                    label=f"room {short}", zorder=3, alpha=0.9)
        cmax = max(cmax, float(ct.max()))

    ax.plot([0, cmax], [0, cmax], "k--", lw=1.0,
            label=r"$\hat c = c_{\mathrm{theory}}$", zorder=2)
    ax.set_xlabel(r"$c_{\mathrm{theory}} = 2\kappa t_{\mathrm{obs}}$")
    ax.set_ylabel(r"$\hat c$ (spectral fit)")
    add_grid(ax)

    legend_right(fig, axes=[ax])
    save_figure(fig, "fig14_c_vs_t_tracking", output_dir)
    return fig


if __name__ == "__main__":
    cli_wrapper(generate)
