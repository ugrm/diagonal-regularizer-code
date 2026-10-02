"""
Figure 23: Dataset Size Ablation — convergence of |s| and cost with sample size.

Paper reference: Section 4, supplementary
Data: size_ablation/val_ablation.npz (scripts/08_size_ablation.py)

Panels:
    (a) Median |s| vs n — horizontal dashed at full-sample value
    (b) Median DP(T=1) and DP(T=50) vs n — vertical line at n=50

Narrative: "The formula's verification converges at N ~ 50 rooms because
the optimality is a property of the PDE, not the training set."
"""

import numpy as np
import matplotlib.pyplot as plt

from src.visualization.style import (
    setup_style, save_figure, FULL_WIDTH, WONG, resolve_data_path, cli_wrapper,
    legend_bottom,
)


def generate(data_dir=None, output_dir="figures/", config=None):
    """Generate Figure 23: Dataset Size Ablation (2-panel)."""
    setup_style()

    data = np.load(
        resolve_data_path("size_ablation/val_ablation.npz", data_dir),
        allow_pickle=True,
    )

    n_rooms = data["n_val_rooms"]      # (6,)
    s_vals = data["median_s"]          # (6,)
    dP_T1 = data["median_dP_T1"]      # (6,)
    dP_T50 = data["median_dP_T50"]    # (6,)

    # Full-sample reference values (n=197, last entry)
    s_full = float(s_vals[-1])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.7))

    # ── Panel (a): |s| vs N ──────────────────────────────────────────
    ax_a.plot(n_rooms, s_vals, "o-", color=WONG["blue"], ms=5, lw=1.5,
              zorder=3, label=r"median $|\hat{s}|$ at $N$ rooms")
    ax_a.axhline(s_full, ls="--", color=WONG["black"], lw=0.8, alpha=0.6,
                 zorder=2, label=r"full-sample $|\hat{s}|$")
    ax_a.axvline(50, ls=":", color=WONG["green"], lw=0.9, alpha=0.8,
                 zorder=1, label=r"$N{=}50$")

    ax_a.set_xlabel("$N$ (validation rooms)")
    ax_a.set_ylabel(r"Median $|\hat{s}|$")
    ax_a.set_title("(a) Prior decay slope")

    # ── Panel (b): DP vs N ───────────────────────────────────────────
    ax_b.plot(n_rooms, dP_T1, "o-", color=WONG["blue"], ms=5, lw=1.5,
              label=r"$\Delta P$ at $T{=}1$", zorder=3)
    ax_b.plot(n_rooms, dP_T50, "s--", color=WONG["red"], ms=4, lw=1.2,
              label=r"$\Delta P$ at $T{=}50$", zorder=3)
    ax_b.axvline(50, ls=":", color=WONG["green"], lw=0.9, alpha=0.8,
                 zorder=1, label=r"$N{=}50$")

    ax_b.set_xlabel("$N$ (validation rooms)")
    ax_b.set_ylabel(r"Median $\Delta P$")
    ax_b.set_title(r"(b) Cost of $|\hat{s}|$")

    fig.tight_layout(w_pad=1.5)
    legend_bottom(fig, ncol=5, fit_width=True)
    save_figure(fig, "fig23_size_ablation", output_dir)
    return fig


if __name__ == "__main__":
    cli_wrapper(generate)
