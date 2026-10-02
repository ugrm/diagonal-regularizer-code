"""
Figure: delta_rel vs ||E||_op scatter -- Appendix A companion.

Tests whether noise anisotropy (||E||_op, empirical Method B) predicts the
cost of using the population |s| instead of the per-room oracle
(delta_rel at T=1000). Points colored by room area to expose the
area confound (large rooms = low E_op, high delta).

Paper reference: Appendix A (anisotropy bound), Section 5 (median delta 5.83% at T=1000, 187 in-scope rooms;
the median over all 197 validation rooms is 5.62%)
Data:
    - E_op_empirical_187.npz: per-room ||E||_op (187 rooms, Method B)
    - cost_K50_M8.npz: per-room delta_rel (10 T x 197 rooms)
    - p_sweep_K50_M8.npz: room_ids for the 197-room cost arrays
    - modal_summary/room_geometries.npz: room vertices for area computation
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats

from src.utils.paths import DATA
from src.visualization.style import (
    setup_style, save_figure, WONG,
    resolve_data_path, cli_wrapper, add_grid, include_width, legend_right,
)


def _shoelace_area(vertices):
    """Signed area via shoelace formula, returned as absolute value."""
    v = vertices
    return abs(
        np.sum(v[:-1, 0] * v[1:, 1] - v[1:, 0] * v[:-1, 1])
        + v[-1, 0] * v[0, 1] - v[0, 0] * v[-1, 1]
    ) / 2


def generate(data_dir=None, output_dir="figures/", config=None):
    """Generate delta_rel vs ||E||_op scatter, colored by room area."""
    setup_style()

    # ── Load data ──────────────────────────────────────────────────────
    eop_path = resolve_data_path("E_op_empirical_187.npz", data_dir)
    cost_path = resolve_data_path("cost_K50_M8.npz", data_dir)
    psweep_path = resolve_data_path("p_sweep_K50_M8.npz", data_dir)

    eop_data = np.load(eop_path, allow_pickle=True)
    cost_data = np.load(cost_path, allow_pickle=True)
    psweep_data = np.load(psweep_path, allow_pickle=True)

    eop_rooms = list(eop_data["scene_ids"])
    eop_vals = eop_data["E_op_per_room"]

    cost_rooms = list(psweep_data["room_ids"])
    T_values = cost_data["T"]
    T_idx = int(np.where(T_values == 1000)[0][0])
    delta_rel_all = cost_data["per_room_delta_rel"][T_idx]  # (197,)

    # ── Room areas ────────────────────────────────────────────────────
    # written by scripts/build/room_geometries.py
    geom_path = resolve_data_path("room_geometries.npz",
                                  os.path.join(data_dir or DATA, "modal_summary"))
    geom = np.load(geom_path, allow_pickle=True)

    # ── Join on scene_id (187 of 197) ─────────────────────────────────
    cost_room_map = {sid: i for i, sid in enumerate(cost_rooms)}
    eop_list, delta_list, area_list = [], [], []
    for sid, eop_val in zip(eop_rooms, eop_vals):
        if sid in cost_room_map and sid in geom:
            eop_list.append(eop_val)
            delta_list.append(delta_rel_all[cost_room_map[sid]])
            area_list.append(_shoelace_area(geom[sid]))

    eop_arr = np.array(eop_list)
    delta_arr = np.array(delta_list) * 100  # convert to %
    area_arr = np.array(area_list)
    n_rooms = len(eop_arr)

    # ── Statistics ────────────────────────────────────────────────────
    rho, pval = stats.spearmanr(eop_arr, delta_arr)
    eop_median = float(np.median(eop_arr))
    # Both cross-hairs are medians of the plotted rooms, computed here rather
    # than typed in, so the figure cannot drift from its own data.
    delta_median = float(np.median(delta_arr))

    # ── Figure ────────────────────────────────────────────────────────
    # Drawn at text width and included at \linewidth; colourbar beside the
    # axes, legend on the right.
    fig, ax = plt.subplots(figsize=(include_width(1.0), 2.6))

    sc = ax.scatter(
        eop_arr, delta_arr, c=area_arr, s=14, alpha=0.7,
        cmap="viridis", norm=LogNorm(vmin=area_arr.min(), vmax=area_arr.max()),
        edgecolors="none", zorder=3,
    )
    # neutral proxy so the legend does not suggest one colour for all rooms
    ax.scatter([], [], s=14, color="0.5", label="one room")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.045, aspect=22)
    cbar.set_label(r"Room area (m$^2$)")
    cbar.ax.tick_params(labelsize=8)

    ax.axvline(eop_median, color=WONG["black"], ls="--", lw=0.8, alpha=0.6,
               label=r"median $\|E\|_{\mathrm{op}}$")
    ax.axhline(delta_median, color=WONG["red"], ls="--", lw=0.8, alpha=0.7,
               label=r"median $\delta_{\mathrm{rel}}$")

    ax.set_xscale("log")
    ax.set_xticks([0.3, 1, 3])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(r"$\|E\|_{\mathrm{op}}$  (empirical)")
    ax.set_ylabel(r"$\delta_{\mathrm{rel}}$ at $T{=}1000$ (%)")
    add_grid(ax, which="both", alpha=0.15)

    legend_right(fig, axes=[ax])
    save_figure(fig, "fig_delta_vs_Eop", output_dir)
    return fig


if __name__ == "__main__":
    cli_wrapper(generate)
