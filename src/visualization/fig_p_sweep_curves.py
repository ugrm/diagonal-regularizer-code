"""
Figure 3: P_modal(p) Curves — Acoustic Short/Long T, Heat Short T, p*(T) Trajectory.

Paper reference: Section 3, Figure 3 (main text)
Data: p_sweep_K50_M8.npz (acoustic), exp2_p_sweep_heat_extended.npz (heat, p∈[0,6]),
      noise_profile_K50_M8.npz (for |s|)

Panels:
    (a) Acoustic T=1: p* ≈ 1.4, |s| = 1.13
    (b) Acoustic T=1000: p* ≈ 2.2 (temporal drift)
    (c) Heat T=1: p* ≈ 2.5, grid extends to 6 — "Within grid range" REMOVED (S6)
    (d) p*(T) trajectory: acoustic and heat
"""

import numpy as np
import matplotlib.pyplot as plt

from src.visualization.style import (
    setup_style, save_figure, FULL_WIDTH, WONG,
    PDE_COLORS, resolve_data_path, cli_wrapper, add_grid, legend_bottom,
)


def _find_T_idx(T_values, target):
    return int(np.argmin(np.abs(np.array(T_values) - target)))


def _plot_panel(ax, p_values, P_rooms, title, pde, show_ylabel=True,
                show_xlabel=True):
    """Plot one P(p) panel: median over rooms, IQR band, and the minimum.

    Reference lines for |s| are added by the caller so that every line style
    is named once in the shared legend.
    """
    color = PDE_COLORS[pde]
    median = np.nanmedian(P_rooms, axis=0)
    q25 = np.nanpercentile(P_rooms, 25, axis=0)
    q75 = np.nanpercentile(P_rooms, 75, axis=0)

    ax.plot(p_values, median, color=color, lw=1.6, zorder=3,
            label=f"median over rooms, {pde}")
    # IQR band not drawn.

    p_star_idx = np.nanargmin(median)
    ax.plot(p_values[p_star_idx], median[p_star_idx], "o", color=color,
            ms=5, zorder=4, label=f"minimum $p^\\star$, {pde}")

    ax.set_title(title)
    if show_xlabel:
        ax.set_xlabel(r"Exponent $p$")
    if show_ylabel:
        ax.set_ylabel(r"median $P$")
    ax.set_xlim(-0.1, p_values[-1] + 0.1)


def _p_star_trajectory(P_oracle, p_values, T_values, M_idx):
    """Compute p*(T) = argmin of median P across rooms, for each T."""
    n_T = len(T_values)
    result = np.zeros(n_T)
    for t in range(n_T):
        median_P = np.nanmedian(P_oracle[:, t, M_idx, :], axis=0)
        result[t] = p_values[np.nanargmin(median_P)]
    return result


def generate(data_dir=None, output_dir="figures/", config=None,
             T_short=1, T_long=2100, M_idx=1):
    """Generate Figure 3: P-Sweep Curves + Trajectory."""
    setup_style()

    # Load acoustic
    d_ac = np.load(resolve_data_path("p_sweep_K50_M8.npz", data_dir), allow_pickle=True)
    p_ac = d_ac["p_values"]
    T_ac = d_ac["T_values"]
    P_ac = d_ac["P_oracle"]

    # Load heat (prefer extended grid, fall back to the standard grid)
    d_ht = None
    for heat_file in ["exp2_p_sweep_heat_extended.npz", "exp2_p_sweep_heat.npz"]:
        try:
            d_ht = np.load(resolve_data_path(heat_file, data_dir), allow_pickle=True)
            break
        except FileNotFoundError:
            continue

    # Load |s|
    d_n = np.load(resolve_data_path("noise_profile_K50_M8.npz", data_dir), allow_pickle=True)
    s_abs = abs(float(np.nanmedian(d_n["s_per_room"])))

    # Reference-line styles, named once in the shared legend. The heat prior
    # is built with Var(a_k(0)) proportional to (1 + lambda_k)^-1, so its |s|
    # is 1 by construction (Section 7, Appendix E), not an estimate.
    S_HEAT = 1.0
    REF_AC = dict(ls="--", color=WONG["black"], lw=1.0, alpha=0.7, zorder=1)
    REF_HT = dict(ls=":", color=WONG["black"], lw=1.2, alpha=0.9, zorder=1)
    LAB_AC = r"acoustic $|\hat{s}|$ (fitted)"
    LAB_HT = r"heat $|s|$ (by construction)"

    # Five stacked rows: the two physics at a
    # short and a long window, then the optimal exponent against T. The
    # long window is the largest in the sweeps.
    fig, axes = plt.subplots(1, 5, figsize=(8.2, 2.5))

    # (a) Acoustic T=1
    t_idx = _find_T_idx(T_ac, T_short)
    _plot_panel(axes[0], p_ac, P_ac[:, t_idx, M_idx, :],
                title=f"(a) Acoustic, $T = {int(T_ac[t_idx])}$",
                pde="acoustic")
    axes[0].axvline(s_abs, label=LAB_AC, **REF_AC)

    # (b) Acoustic, long window
    t_idx = _find_T_idx(T_ac, T_long)
    _plot_panel(axes[1], p_ac, P_ac[:, t_idx, M_idx, :],
                title=f"(b) Acoustic, $T = {int(T_ac[t_idx])}$",
                pde="acoustic", show_ylabel=False)
    axes[1].axvline(s_abs, label=LAB_AC, **REF_AC)

    if d_ht is not None:
        p_ht = d_ht["p_values"]
        T_ht = d_ht["T_values"]
        P_ht = d_ht["P_oracle"]
        # (c) Heat T=1, with both reference exponents
        t_idx = _find_T_idx(T_ht, T_short)
        _plot_panel(axes[2], p_ht, P_ht[:, t_idx, M_idx, :],
                    title=f"(c) Heat, $T = {int(T_ht[t_idx])}$", pde="heat")
        axes[2].axvline(s_abs, label=LAB_AC, **REF_AC)
        axes[2].axvline(S_HEAT, label=LAB_HT, **REF_HT)
        # (d) Heat, long window
        t_idx = _find_T_idx(T_ht, T_long)
        _plot_panel(axes[3], p_ht, P_ht[:, t_idx, M_idx, :],
                    title=f"(d) Heat, $T = {int(T_ht[t_idx])}$", pde="heat",
                    show_ylabel=False)
        axes[3].axvline(s_abs, label=LAB_AC, **REF_AC)
        axes[3].axvline(S_HEAT, label=LAB_HT, **REF_HT)

    # (e) p*(T) trajectory
    ax = axes[4]
    p_star_ac = _p_star_trajectory(P_ac, p_ac, T_ac, M_idx)
    ax.plot(T_ac, p_star_ac, color=PDE_COLORS["acoustic"], lw=1.6,
            marker="o", ms=4, label=r"$p^\star(T)$, acoustic", zorder=3)

    if d_ht is not None:
        p_star_ht = _p_star_trajectory(P_ht, p_ht, T_ht, M_idx)
        ax.plot(T_ht, p_star_ht, color=PDE_COLORS["heat"], lw=1.6,
                marker="s", ms=4, ls="--", label=r"$p^\star(T)$, heat", zorder=3)

    ax.axhline(s_abs, label=LAB_AC, **REF_AC)
    ax.axhline(S_HEAT, label=LAB_HT, **REF_HT)

    ax.set_xscale("log")
    ax.set_xlabel("$T$ (snapshots)")
    ax.set_ylabel(r"$p^\star$")
    ax.set_title(r"(e) $p^\star$ vs $T$")
    ax.set_ylim(0, 4.0)

    legend_bottom(fig, ncol=5, fit_width=True)
    save_figure(fig, "fig03_p_sweep_curves", output_dir)
    return fig


if __name__ == "__main__":
    cli_wrapper(generate)
