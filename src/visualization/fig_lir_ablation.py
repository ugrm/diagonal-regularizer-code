"""
Figure: LIR L-Ablation — P vs L and ΔP vs L.

Paper reference: Appendix D.8 (Learned Iterative Ridge)
Data: data/experiments/lir/lir_summary.npz

Two panels:
    (a) P_modal vs L at each T, with Ridge(|s|) dashed and Oracle dotted references
    (b) ΔP = P_LIR - P_oracle vs L at each T, horizontal line at 0

Visual narrative: LIR stays above the oracle at every depth and window.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.visualization.style import (
    setup_style, save_figure, WONG,
    resolve_data_path, cli_wrapper, enable_all_spines,
    include_width, legend_bottom,
)


# T-indexed colors (consistent across panels)
T_COLORS = {1: WONG["blue"], 100: WONG["orange"], 1000: WONG["green"]}
T_VALUES = [1, 100, 1000]
L_VALUES = [1, 5, 10, 20]

def generate(data_dir=None, output_dir="figures/", config=None):
    """Generate LIR L-Ablation figure."""
    setup_style()

    # ── Load LIR summary (clean array-only npz) ─────────────────────
    summary_path = resolve_data_path("lir_summary.npz", data_dir)
    summary = np.load(summary_path)
    P_median = summary["P_median"]      # (4, 3, 5) — L × T × seeds
    L_values = summary["L_values"]      # (4,)
    T_values = summary["T_values"]      # (3,)

    # ── Baselines, read from the sweep evaluation rather than typed in ──
    # per-room power-law oracle and the closed form at oracle alpha, medians
    # over the validation rooms, at T = 1, 100, 1000
    sweep = np.load(resolve_data_path("sweep_P_eval.npz", data_dir),
                    allow_pickle=True)
    assert list(sweep["T_values"]) == T_VALUES
    P_ORACLE = sweep["P_oracle_baseline"]
    P_RIDGE_S = sweep["P_ridge_s"]

    # ── Aggregate: mean and std across seeds (convention: mean ± std) ─
    P_mean = np.mean(P_median, axis=2)  # (4, 3)
    P_std = np.std(P_median, axis=2)    # (4, 3)

    # ── Figure (full width: two panels plus a right legend need it) ───
    fig, axes = plt.subplots(1, 2, figsize=(include_width(1.0), 2.7))

    # ── Panel (a): P vs L ─────────────────────────────────────────────
    ax = axes[0]
    enable_all_spines(ax)

    for ti, T in enumerate(T_VALUES):
        color = T_COLORS[T]
        ax.errorbar(L_VALUES, P_mean[:, ti], yerr=P_std[:, ti], color=color,
                    fmt="o-", ms=4, capsize=2.5, capthick=1.0, lw=1.3,
                    label=f"LIR, $T={T}$", zorder=3)
        ax.axhline(P_RIDGE_S[ti], color=color, ls="--", lw=1.0, alpha=0.6)
        ax.axhline(P_ORACLE[ti], color=color, ls=":", lw=1.0, alpha=0.6)
    # neutral proxies for the two reference line styles
    ax.plot([], [], color="0.4", ls="--", lw=1.0, label="closed form")
    ax.plot([], [], color="0.4", ls=":", lw=1.0, label="per-room power-law oracle")

    ax.set_xlabel("Depth $L$")
    ax.set_ylabel(r"$P$ (mean over seeds)")
    ax.set_title("(a) Error vs depth")
    ax.set_xticks(L_VALUES)

    # ── Panel (b): ΔP vs L (relative to oracle) ──────────────────────
    ax = axes[1]
    enable_all_spines(ax)

    for ti, T in enumerate(T_VALUES):
        color = T_COLORS[T]
        delta_P = P_mean[:, ti] - P_ORACLE[ti]
        ax.errorbar(L_VALUES, delta_P, yerr=P_std[:, ti], color=color,
                    fmt="o-", ms=4, capsize=2.5, capthick=1.0, lw=1.3,
                    label=f"LIR, $T={T}$", zorder=3)

    ax.axhline(0, color="0.4", ls=":", lw=1.0, alpha=0.8, zorder=1,
               label="per-room power-law oracle")
    ax.set_xlabel("Depth $L$")
    ax.set_ylabel(r"$P_{\mathrm{LIR}} - P_{\mathrm{oracle}}$")
    ax.set_title("(b) Gap to the oracle")
    ax.set_xticks(L_VALUES)

    fig.tight_layout(w_pad=1.5)
    legend_bottom(fig, ncol=5, fit_width=True)
    save_figure(fig, "figS_lir_ablation", output_dir)
    return fig


if __name__ == "__main__":
    cli_wrapper(generate)
