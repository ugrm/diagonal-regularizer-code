"""
Figure S2: Architecture Capacity Verification.

Paper reference: Appendix, Figure S2
Data: data/supplementary/capacity_{results,1000ep_results}.npz (and m1_capacity_results.npz if present)

Panels:
    (a) M3 recovered Γ_k spectra at each p_target (0.5, 1.5, 2.5)
    (b) p_target vs p_recovered scatter for all methods
    (c) Sensitivity analysis: why p_target=0.5 is uninformative

Key numbers: M1=[0.18, 0.72, 0.47], M2=[0.59, 0.81, 0.86], M3=[0.35, 1.35, 2.50]
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.visualization.style import (
    setup_style, save_figure, WONG,
    METHOD_COLORS, METHOD_LABELS, METHOD_MARKERS,
    resolve_data_path, cli_wrapper, include_width, legend_bottom,
)

P_TARGET_COLORS = {0.5: WONG["blue"], 1.5: WONG["orange"], 2.5: WONG["purple"]}


def _load_capacity_data():
    """Load capacity check results for all available methods."""
    from src.utils.paths import SUPP
    cap_dir = str(SUPP)

    results = {"p_targets": [0.5, 1.5, 2.5]}

    conv_path = os.path.join(cap_dir, "capacity_1000ep_results.npz")
    cap_path = os.path.join(cap_dir, "capacity_results.npz")
    m1_path = os.path.join(cap_dir, "m1_capacity_results.npz")

    if os.path.isfile(conv_path):
        d = np.load(conv_path, allow_pickle=True)
        results["ev_median"] = d["ev_median"]
        results["converged"] = True
        for p in [0.5, 1.5, 2.5]:
            key = f"p{str(p).replace('.', '_')}"
            for m in ["m1", "m2"]:
                results[f"phat_{m}_{key}"] = float(d[f"{m}_phat_final_{key}"])
                results[f"gamma_{m}_{key}"] = d[f"{m}_gamma_{key}"]
        results["has_m1"] = results["has_m2"] = True
    else:
        results["converged"] = False
        if os.path.isfile(cap_path):
            d = np.load(cap_path, allow_pickle=True)
            results["ev_median"] = d["ev_median"]
            for p in [0.5, 1.5, 2.5]:
                key = f"p{str(p).replace('.', '_')}"
                results[f"gamma_m2_{key}"] = d[f"gamma_m2_{key}"]
                results[f"phat_m2_{key}"] = float(d[f"phat_m2_{key}"])
            results["has_m2"] = True
        else:
            results["has_m2"] = False

        if os.path.isfile(m1_path):
            d1 = np.load(m1_path, allow_pickle=True)
            if "ev_median" not in results:
                results["ev_median"] = d1["ev_median"]
            for p in [0.5, 1.5, 2.5]:
                key = f"p{str(p).replace('.', '_')}"
                results[f"gamma_m1_{key}"] = d1[f"gamma_m1_{key}"]
                results[f"phat_m1_{key}"] = float(d1[f"phat_m1_{key}"])
            results["has_m1"] = True
        else:
            results["has_m1"] = False

    # M3 always from capacity_results.npz (converged at 300 epochs)
    if os.path.isfile(cap_path):
        d_orig = np.load(cap_path, allow_pickle=True)
        for p in [0.5, 1.5, 2.5]:
            key = f"p{str(p).replace('.', '_')}"
            results[f"gamma_m3_{key}"] = d_orig[f"gamma_m3_{key}"]
            results[f"phat_m3_{key}"] = float(d_orig[f"phat_m3_{key}"])
        results["has_m3"] = True
        if "ev_median" not in results:
            results["ev_median"] = d_orig["ev_median"]
    else:
        results["has_m3"] = False

    return results


def generate(data_dir=None, output_dir="figures/", config=None):
    """Generate Figure S2: Capacity Check."""
    setup_style()

    data = _load_capacity_data()
    if not data.get("has_m3", False):
        print("No capacity data found. Figure S2 not generated.")
        return None

    ev = data["ev_median"]
    K = len(ev)
    k_idx = np.arange(1, K + 1)
    p_targets = data["p_targets"]

    fig, axes = plt.subplots(1, 3, figsize=(include_width(1.0), 2.8))

    # ── Panel (a): M3 Γ_k spectra ───────────────────────────────────
    ax = axes[0]
    for p_tgt in p_targets:
        color = P_TARGET_COLORS[p_tgt]
        key = f"p{str(p_tgt).replace('.', '_')}"
        target = ev ** p_tgt
        target /= target[0]
        # dashed planted spectrum in the colour of its M3 curve
        ax.plot(k_idx, target, ls="--", color=color, lw=1.5, alpha=0.5)
        gamma = data[f"gamma_m3_{key}"]
        gamma_norm = gamma / gamma[0]
        ax.plot(k_idx, gamma_norm, ls="-", color=color, lw=1.2,
                label=f"M3, planted $p={p_tgt:.1f}$")
    ax.plot([], [], ls="--", color="0.4", lw=1.5, alpha=0.7,
            label="planted spectrum")

    ax.set_xlabel("Mode index $k$")
    ax.set_ylabel(r"$\Gamma_k / \Gamma_1$")
    ax.set_title(r"(a) Recovered $\Gamma_k$")
    ax.set_yscale("log")

    # ── Panel (b): p_target vs p_recovered ───────────────────────────
    ax = axes[1]
    ax.plot([0, 3], [0, 3], ls="--", color="#AAAAAA", lw=1.0,
            label="recovered = planted")

    converged = data.get("converged", False)
    for method in ["M1", "M2", "M3"]:
        if not data.get(f"has_{method.lower()}", False):
            continue
        p_rec = [data[f"phat_{method.lower()}_p{str(p).replace('.', '_')}"]
                 for p in p_targets]
        lbl = METHOD_LABELS[method]
        if not converged and method in ("M1", "M2"):
            lbl += " (300ep)"
        ax.scatter(p_targets, p_rec, s=40, c=METHOD_COLORS[method],
                   marker=METHOD_MARKERS[method], zorder=3,
                   edgecolors="white", linewidths=0.5, label=lbl)

    ax.set_xlabel(r"Planted $p$")
    ax.set_ylabel(r"Recovered $\hat{p}$")
    ax.set_title(r"(b) Recovered $\hat{p}$")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])

    # ── Panel (c): Sensitivity analysis ──────────────────────────────
    ax = axes[2]
    T_rep = 100
    ATA_approx = T_rep * ev
    alpha = 1.0

    p_scan = np.linspace(0.0, 3.0, 100)
    mean_sens = np.zeros_like(p_scan)
    for i, p in enumerate(p_scan):
        gamma = ev ** p
        sens = alpha * gamma / (ATA_approx + alpha * gamma)
        mean_sens[i] = sens.mean()

    ax.plot(p_scan, mean_sens, color=WONG["blue"], lw=1.8,
            label="mean loss sensitivity")

    for p_tgt in p_targets:
        gamma = ev ** p_tgt
        sens = (alpha * gamma / (ATA_approx + alpha * gamma)).mean()
        ax.plot(p_tgt, sens, "o", color=P_TARGET_COLORS[p_tgt], ms=6,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
    ax.plot([], [], "o", color="0.4", ms=6, ls="none",
            label="at the planted exponents")

    ax.set_xlabel(r"Exponent $p$")
    ax.set_ylabel("Mean sensitivity")
    ax.set_title("(c) Sensitivity")
    ax.set_xlim(0, 3.1)
    ax.set_ylim(0, 0.8)

    legend_bottom(fig)
    save_figure(fig, "figS2_capacity_check", output_dir)
    return fig


if __name__ == "__main__":
    cli_wrapper(generate)
