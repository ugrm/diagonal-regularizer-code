"""
Figure 15: Berry Q-Q Stratified — Global Q-Q plot + K-stratified pass rates.

Paper reference: Supplement
Data: data/supplementary/berry_qq_data.npz

Panels:
    (a) Global Q-Q plot: observed |Ω|·|φ_k(x)|² vs χ²(1) quantiles. KS = 0.037.
    (b) Pass rate at α=0.05 by K_total bin. 88.5% at K=50-200.

Pre-computed by: scripts/appendix/A_B/run_berry_qq.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from src.visualization.style import (
    setup_style, save_figure, FULL_WIDTH, WONG, resolve_data_path, cli_wrapper,
    add_grid, enable_all_spines, legend_right,
)


def generate(data_dir=None, output_dir="figures/", config=None):
    """Generate Figure 15: Berry Q-Q Stratified."""
    setup_style()

    path = resolve_data_path("berry_qq_data.npz", data_dir)
    data = np.load(path, allow_pickle=True)

    all_Csq = data["all_Csq"]
    ks_pval = data["ks_pval"]
    ks_stat = data["ks_stat"]
    K_total = data["K_total"]
    alpha = 0.05

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.6),
                             gridspec_kw={"width_ratios": [1, 1.15]})

    # ── Panel (a): Global Q-Q ────────────────────────────────────────
    ax = axes[0]
    sorted_Csq = np.sort(all_Csq)
    n = len(sorted_Csq)
    theoretical = stats.chi2.ppf((np.arange(1, n + 1) - 0.5) / n, df=1)

    step = max(1, n // 2000)
    idx = np.arange(0, n, step)
    ax.scatter(theoretical[idx], sorted_Csq[idx], s=2, alpha=0.5,
               color=WONG["blue"], label="pooled quantiles")
    max_val = min(np.percentile(sorted_Csq, 99.5), np.percentile(theoretical, 99.5))
    ax.plot([0, max_val], [0, max_val], color=WONG["red"], lw=1.5,
            label=r"exact $\chi^2(1)$")

    ax.set_xlabel(r"$\chi^2(1)$ quantiles")
    ax.set_ylabel(r"Observed $|\Omega|\,|\varphi_k(x)|^2$")
    ax.set_title(r"(a) Pooled Q-Q")
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    ax.set_aspect("equal")
    enable_all_spines(ax)
    add_grid(ax)

    # ── Panel (b): Pass rate by K bin ────────────────────────────────
    ax = axes[1]
    # Last bin is open-ended: rooms with K_total >= 1000 (six in the validation
    # set) belong to "500+".
    bins = [(0, 50), (50, 200), (200, 500), (500, np.inf)]
    bin_labels = ["<50", "50\u2013200", "200\u2013500", "500+"]
    pass_rates, counts, med_ks = [], [], []
    for lo, hi in bins:
        mask = (K_total >= lo) & (K_total < hi)
        n_rooms = mask.sum()
        pr = np.mean(ks_pval[mask] > alpha) if n_rooms > 0 else 0
        pass_rates.append(pr)
        counts.append(n_rooms)
        med_ks.append(np.median(ks_stat[mask]) if n_rooms > 0 else 0)

    # Single neutral fill: ordinal K_total bins are encoded by x-position;
    # bar height already encodes the pass rate.
    ax.bar(range(len(bins)), [pr * 100 for pr in pass_rates],
           color="#888888", edgecolor="black", linewidth=0.8,
           label="KS pass rate")

    ax.axhline(95, color="gray", ls="--", alpha=0.6,
               label="expected under Berry")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax.set_xlabel(r"Total mode count $K_{\mathrm{total}}$")
    ax.set_ylabel(r"Pass rate at the $0.05$ level (%)")
    ax.set_title("(b) Pass rate vs mode count")
    ax.set_ylim(0, 105)

    legend_right(fig)
    save_figure(fig, "fig15_berry_qq_stratified", output_dir)
    return fig


if __name__ == "__main__":
    cli_wrapper(generate)
