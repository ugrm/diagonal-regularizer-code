"""
Figure: Geometric Predictor Analysis — Room features vs optimal exponent.

Paper reference: Section 5, Layer 3 defense
Data: geometric_predictors.npz (from script 13)

Panels:
    (a) Scatter: p*(T=1000) vs K_total with linear fit
    (b) Bar chart: Spearman rho for all features, Tier 1 vs Tier 2

Reference numbers:
    RF R² = 0.14 (T=1000), pairwise rho=1.00 among size features
    Accessible features max |rho| = 0.15
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from src.visualization.style import (
    setup_style, save_figure, WONG, resolve_data_path, cli_wrapper,
    include_width, legend_bottom,
)

# Features accessible to the network at inference (Tier 1)
ACCESSIBLE = {"spectral_gap", "lambda_K", "weyl_exponent", "s_room"}
# Features NOT accessible (Tier 2 — require room geometry)
INACCESSIBLE = {"mean_spacing", "eigenvalue_density", "K_total", "room_area", "n_segments"}


def generate(data_dir=None, output_dir="figures/", config=None):
    """Generate geometric predictor figure (Section 5, Layer 3)."""
    setup_style()

    path = resolve_data_path("geometric_predictors.npz", data_dir)
    d = np.load(path, allow_pickle=True)

    feature_names = list(d["feature_names"])
    X = d["X"]               # (196, 9)
    p_star = d["p_star"]      # (196,) — p* at T_target
    rho = d["rho"]            # (9,) — Spearman rho
    rho_ci_lo = d["rho_ci_lo"]
    rho_ci_hi = d["rho_ci_hi"]
    T_target = int(d["T_target"])

    # Drawn at its include width (0.9\linewidth); legend on the right.
    fig, axes = plt.subplots(1, 2, figsize=(include_width(1.0), 2.9),
                             gridspec_kw={"width_ratios": [1, 1.3]})

    # ── Panel (a): Scatter p* vs K_total ──────────────────────────────
    ax = axes[0]
    k_idx = feature_names.index("K_total")
    K_total = X[:, k_idx]

    ax.scatter(K_total, p_star, s=8, alpha=0.4, color=WONG["blue"],
               edgecolors="none", zorder=2, label="one room")

    # Linear fit; its statistics are in the appendix tables, not the legend
    slope, intercept, r_val, p_val, se = stats.linregress(K_total, p_star)
    x_fit = np.linspace(K_total.min(), K_total.max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, color=WONG["red"],
            lw=1.5, ls="--", zorder=3, label="linear fit")

    ax.set_xlabel(r"$K_{\mathrm{total}}$ (total eigenvalues)")
    ax.set_ylabel(rf"$p^\star$ at $T{{=}}{T_target}$")
    ax.set_title(rf"(a) $p^\star$ vs $K_{{\mathrm{{total}}}}$")

    # ── Panel (b): Bar chart of Spearman rho ──────────────────────────
    ax = axes[1]

    # Sort by absolute rho
    order = np.argsort(np.abs(rho))[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_rho = rho[order]
    sorted_lo = rho_ci_lo[order]
    sorted_hi = rho_ci_hi[order]

    # Color by tier
    colors = []
    for name in sorted_names:
        if name in ACCESSIBLE:
            colors.append(WONG["cyan"])   # Tier 1: accessible
        else:
            colors.append(WONG["orange"])  # Tier 2: inaccessible

    y_pos = np.arange(len(sorted_names))
    xerr = np.array([sorted_rho - sorted_lo, sorted_hi - sorted_rho])
    ax.barh(y_pos, sorted_rho, color=colors, edgecolor="white",
            linewidth=0.5, height=0.7, zorder=2)
    ax.errorbar(sorted_rho, y_pos, xerr=xerr, fmt="none",
                ecolor=WONG["black"], elinewidth=0.6, capsize=2, zorder=3,
                label="95% bootstrap CI")

    # Format feature names for display. All-math labels:
    # paper body uses |Ω| for room area (Berry's conjecture), \bar{Δ} for
    # mean spacing (B.1 unfolding definition), λ_2 - λ_1 for spectral gap.
    # Weyl exponent and ρ(λ) lack body conventions; defaults used.
    LABEL_MAP = {
        "room_area":          r"$|\Omega|$",
        "K_total":            r"$K_{\mathrm{total}}$",
        "mean_spacing":       r"$\bar{\Delta}$",
        "eigenvalue_density": r"$\rho(\lambda)$",
        "spectral_gap":       r"$\lambda_2 - \lambda_1$",
        "n_segments":         r"$n_{\mathrm{seg}}$",
        "lambda_K":           r"$\lambda_K$",
        "weyl_exponent":      r"$\alpha_{\mathrm{Weyl}}$",
        "s_room":             r"$|s|_{\mathrm{room}}$",
    }
    display_names = [LABEL_MAP.get(name, name.replace("_", " ")) for name in sorted_names]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names)
    ax.set_xlabel(r"Spearman $\rho$")
    ax.set_title(rf"(b) Spearman $\rho$ with $p^\star$, $T{{=}}{T_target}$")
    ax.axvline(0, color=WONG["black"], lw=0.5, ls=":")
    ax.invert_yaxis()

    # Bar colours are named in the shared legend with the other handles.
    from matplotlib.patches import Patch
    tier_handles = [
        Patch(facecolor=WONG["cyan"], label="Tier 1: accessible at inference"),
        Patch(facecolor=WONG["orange"], label="Tier 2: needs room geometry"),
    ]
    h_a, l_a = axes[0].get_legend_handles_labels()
    h_b, l_b = ax.get_legend_handles_labels()
    handles = h_a + tier_handles + h_b
    labels = l_a + [h.get_label() for h in tier_handles] + l_b

    fig.tight_layout(w_pad=2.0)
    legend_bottom(fig, handles=handles, labels=labels)
    save_figure(fig, "fig_geometric_predictor", output_dir)
    return fig


if __name__ == "__main__":
    cli_wrapper(generate)
