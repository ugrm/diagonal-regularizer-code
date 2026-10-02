"""
Figure 1: Truncation Noise Profile — Is the noise isotropic?

Paper reference: Section 3, Figure 1 (main text)
Data: noise_profile_K50_M8.npz (n_rooms x K, shape-agnostic)

Panels:
    (a) Approach B (FEM-only): σ²_trunc,k vs λ_k — flat (q_B ≈ 0)
    (b) Approach A (total residual): σ²_total,k vs λ_k — also flat (q_A ≈ 0)
    (c) Histogram of per-room q_B and q_A slopes — centered near zero

Key annotations (computed from data, verified against KEY NUMBERS):
    q_B = -0.09, q_A = +0.12, R²=0.03, |s| = 1.13
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from src.visualization.style import (
    setup_style, save_figure, FULL_WIDTH, WONG,
    METHOD_COLORS, resolve_data_path, cli_wrapper, legend_bottom,
)

LOG_FLOOR = 1e-30


def _log10_safe(x):
    return np.log10(np.maximum(x, LOG_FLOOR))


def generate(data_dir=None, output_dir="figures/", config=None):
    """Generate Figure 1: Noise Profile."""
    setup_style()

    # Load data
    path = resolve_data_path("noise_profile_K50_M8.npz", data_dir)
    data = np.load(path, allow_pickle=True)

    eigenvalues = data["eigenvalues"]       # (n_rooms, K)
    sigma_B = data["sigma_B"]               # (n_rooms, K)
    sigma_A = data["sigma_A"]               # (n_rooms, K)
    sigma_sq_a = data["sigma_sq_a_all"]     # (n_rooms, K)
    q_B_full = data["q_B_full"]             # (n_rooms,)
    q_A_full = data["q_A_full"]             # (n_rooms,)
    s_per_room = data["s_per_room"]         # (n_rooms,)

    # Median eigenvalue curve as shared x-axis
    lam_median = np.median(eigenvalues, axis=0)
    log_lam = np.log10(lam_median)

    # Drawn at text width; one shared legend on the right names every colour
    # and style.
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.7))

    # ── Panel (a): Approach B (FEM-only truncation noise) ────────────
    ax = axes[0]

    log_trunc_median = _log10_safe(np.median(sigma_B, axis=0))
    log_trunc_q25 = _log10_safe(np.percentile(sigma_B, 25, axis=0))
    log_trunc_q75 = _log10_safe(np.percentile(sigma_B, 75, axis=0))

    ax.plot(log_lam, log_trunc_median, color=METHOD_COLORS["trunc_noise"],
            lw=1.8, label="truncation noise, median", zorder=4)
    ax.fill_between(log_lam, log_trunc_q25, log_trunc_q75,
                    color=METHOD_COLORS["trunc_noise"], alpha=0.15, zorder=2,
                    label="truncation noise, IQR")

    # Prior variance overlay (declining)
    log_prior = _log10_safe(np.median(sigma_sq_a, axis=0))
    ax.plot(log_lam, log_prior, color=METHOD_COLORS["prior"],
            lw=1.8, ls="--", label="prior variance, median", zorder=3)

    ax.set_xlabel(r"$\log_{10}(\lambda_k)$")
    ax.set_ylabel(r"$\log_{10}(\mathrm{variance})$")
    ax.set_title("(a) FEM-only")

    # ── Panel (b): Approach A (total residual) ───────────────────────
    ax = axes[1]

    log_A_median = _log10_safe(np.median(sigma_A, axis=0))
    log_A_q25 = _log10_safe(np.percentile(sigma_A, 25, axis=0))
    log_A_q75 = _log10_safe(np.percentile(sigma_A, 75, axis=0))

    ax.plot(log_lam, log_A_median, color=METHOD_COLORS["approach_a"],
            lw=1.8, label="total residual, median", zorder=4)
    ax.fill_between(log_lam, log_A_q25, log_A_q75,
                    color=METHOD_COLORS["approach_a"], alpha=0.15, zorder=2,
                    label="total residual, IQR")

    ax.plot(log_lam, log_prior, color=METHOD_COLORS["prior"],
            lw=1.8, ls="--", label="prior variance, median", zorder=3)

    ax.set_xlabel(r"$\log_{10}(\lambda_k)$")
    ax.set_title("(b) Total residual")

    # ── Panel (c): Histogram of per-room slopes ──────────────────────
    ax = axes[2]

    bins = np.linspace(
        min(q_B_full.min(), q_A_full.min()) - 0.05,
        max(q_B_full.max(), q_A_full.max()) + 0.05, 30)

    ax.hist(q_B_full, bins=bins, color=METHOD_COLORS["trunc_noise"], alpha=0.5,
            edgecolor="white", linewidth=0.5,
            label="slope per room, FEM-only", zorder=3)
    ax.hist(q_A_full, bins=bins, color=METHOD_COLORS["approach_a"], alpha=0.5,
            edgecolor="white", linewidth=0.5,
            label="slope per room, total", zorder=2)

    ax.axvline(0, color=WONG["black"], ls=":", lw=1.0, zorder=4,
               label="zero slope")
    ax.axvline(np.median(q_B_full), color=METHOD_COLORS["trunc_noise"],
               ls="--", lw=1.2, zorder=5, label="median slope, FEM-only")
    ax.axvline(np.median(q_A_full), color=METHOD_COLORS["approach_a"],
               ls="--", lw=1.2, zorder=5, label="median slope, total")

    ax.set_xlabel(r"Slope $q$ (log-log)")
    ax.set_ylabel("Rooms")
    ax.set_title(r"(c) Per-room slope")

    from matplotlib.ticker import MaxNLocator
    for ax in axes[:2]:
        ax.xaxis.set_major_locator(MaxNLocator(4))
    axes[2].set_xticks([-0.3, 0.0, 0.3])
    # Six entries: colour names the estimate in every panel, the band is
    # its IQR, dashed lines are the medians in (c).
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerTuple
    blue, orange = METHOD_COLORS["trunc_noise"], METHOD_COLORS["approach_a"]
    handles = [
        (Patch(facecolor=blue, alpha=0.25, edgecolor="none"),
         Line2D([], [], color=blue, lw=1.8)),
        (Patch(facecolor=orange, alpha=0.25, edgecolor="none"),
         Line2D([], [], color=orange, lw=1.8)),
        Line2D([], [], color=METHOD_COLORS["prior"], lw=1.8, ls="--"),
        Line2D([], [], color=blue, lw=1.2, ls="--"),
        Line2D([], [], color=orange, lw=1.2, ls="--"),
        Line2D([], [], color=WONG["black"], lw=1.0, ls=":"),
    ]
    labels = ["FEM-only noise", "total residual", "prior variance",
              "median, FEM-only", "median, total", "zero slope"]
    legend_bottom(fig, handles=handles, labels=labels, ncol=len(labels),
                  fit_width=True,
                 handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})
    save_figure(fig, "fig01_noise_profile", output_dir)
    return fig


if __name__ == "__main__":
    cli_wrapper(generate)
