"""
Figure 2: The Flat Landscape — P(p) basin widens with T.

Paper reference: Section 3-4, Figure 2 (main text)
Data: p_sweep/p_sweep_K50_M8.npz (197 rooms, 61 p-values [0,6])
      noise_profile_K50_M8.npz (for |s|)

Single panel: population-median P(p) curves at T in {1, 50, 200, 1000, 2100}.
Dashed vertical at |s| ~ 1.13, filled dots marking p*(T) on each curve.
Progressive flattening demonstrates posterior concentration at large T.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.visualization.style import (
    setup_style, save_figure, WONG, resolve_data_path, cli_wrapper,
    include_width, legend_right,
)

# Ordered T values and Wong-palette colors. T=2100 uses purple instead of red
# to break the warm-spectrum collision with T=1000 at the low-P plateau.
T_SHOW = [1, 50, 200, 1000, 2100]
T_COLORS = [WONG["blue"], WONG["cyan"], WONG["green"], WONG["orange"], WONG["purple"]]

XLIM = (0, 3.1)

# 10 boundary rooms with K_total <= K=50 (excluded from in-scope n=187).
BOUNDARY_ROOMS = {
    "scene_00800", "scene_00806", "scene_00810", "scene_00812", "scene_00829",
    "scene_00839", "scene_00848", "scene_00867", "scene_00981", "scene_00993",
}


def generate(data_dir=None, output_dir="figures/", config=None):
    """Generate Figure 2: Multi-T flat landscape."""
    setup_style()

    # ── Load ──────────────────────────────────────────────────────────
    psweep = np.load(
        resolve_data_path("p_sweep/p_sweep_K50_M8.npz", data_dir),
        allow_pickle=True,
    )
    noise = np.load(
        resolve_data_path("noise_profile_K50_M8.npz", data_dir),
        allow_pickle=True,
    )

    P_oracle = psweep["P_oracle"]   # (197, 10, n_M, 61) rooms x T x M x p
    p_values = psweep["p_values"]   # (61,)
    T_values = psweep["T_values"]   # (10,)
    M_values = psweep["M_values"]   # [4, 8, 16]
    room_ids = psweep["room_ids"]   # (197,)

    s_median = float(np.median(np.abs(noise["s_per_room"])))

    # M=8 index
    M_idx = int(np.where(M_values == 8)[0][0])

    # n=187 in-scope filter: drop rooms with K_total <= K=50
    keep = np.array([str(r) not in BOUNDARY_ROOMS for r in room_ids])
    n_inscope = int(keep.sum())
    assert n_inscope == 187, f"expected 187 in-scope rooms, got {n_inscope}"

    # Crop to display range
    p_mask = p_values <= XLIM[1]
    p_crop = p_values[p_mask]

    # ── Plot ──────────────────────────────────────────────────────────
    # Drawn at text width and included at \linewidth; the legend is measured
    # and the axes take the rest, so the basin is as wide as the page allows.
    fig, ax = plt.subplots(figsize=(include_width(1.0), 1.85))

    for T, color in zip(T_SHOW, T_COLORS):
        T_idx = int(np.where(T_values == T)[0][0])
        P_median = np.median(P_oracle[keep, T_idx, M_idx, :], axis=0)
        P_crop = P_median[p_mask]

        # Population oracle
        pstar_idx = int(np.argmin(P_crop))
        pstar = p_crop[pstar_idx]
        P_at_pstar = P_crop[pstar_idx]

        # Line-style differentiation for the long-T pair: T=1000 and T=2100
        # plateau at near-identical P values, so even high-contrast colors
        # press against each other at the basin minimum. Dashing T=2100
        # disambiguates the curves regardless of color.
        ls = "--" if T == 2100 else "-"
        ax.plot(p_crop, P_crop, color=color, lw=1.6, ls=ls,
                label=f"$T = {T}$", zorder=3)
        ax.plot(pstar, P_at_pstar, "o", color=color, ms=6,
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)

    # |s| vertical: a legend entry, not an in-plot label
    ax.axvline(s_median, color=WONG["black"], ls="--", lw=1.0,
               alpha=0.6, zorder=2, label=r"population $|\hat{s}|$")
    # proxy handle for the per-curve minimum markers
    ax.plot([], [], "o", color="0.45", ms=6, markeredgecolor="white",
            markeredgewidth=1.2, ls="none", label=r"minimum $p^\star(T)$")

    ax.set_xlabel(r"Exponent $p$")
    ax.set_ylabel(r"median $P$ over rooms")
    ax.set_xlim(*XLIM)

    legend_right(fig)
    save_figure(fig, "fig02_flat_landscape", output_dir)
    return fig


if __name__ == "__main__":
    cli_wrapper(generate)
