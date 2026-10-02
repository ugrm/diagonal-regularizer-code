"""
Figure 16: Cost Distribution — Per-room cost of using p=|s| vs oracle.

Paper reference: Section 4, supplement
Data: cost/cost_K50_M8.npz (pre-computed per-room costs from Script 06)

Panels:
    Top row: Per-T relative cost delta histograms at T in {1, 50, 500, 1000, 2100}
    Bottom: Absolute cost DP vs T (median, 95th percentile, max)

Key numbers: median DP < 0.011 at all T. 95th pct at T=1000 = 3.6pp.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.visualization.style import (
    setup_style, save_figure, FULL_WIDTH, WONG, resolve_data_path, cli_wrapper,
    _collect_handles,
)

T_DISPLAY = [1, 50, 500, 1000, 2100]


def generate(data_dir=None, output_dir="figures/", config=None):
    """Generate Figure 16: Cost Distribution."""
    setup_style()

    data = np.load(
        resolve_data_path("cost/cost_K50_M8.npz", data_dir),
        allow_pickle=True,
    )

    T = data["T"]                              # (10,)
    per_room_dP = data["per_room_delta_P"]     # (10, 197)
    per_room_rel = data["per_room_delta_rel"]  # (10, 197)

    T_idxs = [int(np.where(T == t)[0][0]) for t in T_DISPLAY]
    n_T = len(T_DISPLAY)

    # Histograms across the full width on top; the absolute-cost panel and
    # the legend share the bottom row.
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(FULL_WIDTH, 4.2))
    gs = GridSpec(2, n_T, figure=fig, height_ratios=[1.0, 1.15], wspace=0.08)
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(n_T)]
    # The absolute-cost threshold Section 5 reports the room fraction against.
    ABS_COST_THRESHOLD = 0.011

    all_median_dP = []
    all_p95_dP = []
    all_max_dP = []

    # Pre-compute shared histogram axes so cross-T comparison is meaningful.
    delta_max_pct = float(max(np.max(per_room_rel[t_idx]) for t_idx in T_idxs)) * 100
    shared_x_hi = np.ceil(delta_max_pct / 5) * 5  # round up to nearest 5%
    shared_bins = np.linspace(0, shared_x_hi, 26)
    # y-axis: take max histogram height across all T at the chosen bins
    shared_y_hi = 0
    for t_idx in T_idxs:
        h, _ = np.histogram(per_room_rel[t_idx] * 100, bins=shared_bins)
        shared_y_hi = max(shared_y_hi, h.max())
    shared_y_hi = int(np.ceil(shared_y_hi * 1.1))  # 10% headroom

    for col, (t_disp, t_idx) in enumerate(zip(T_DISPLAY, T_idxs)):
        dP = per_room_dP[t_idx]
        delta = per_room_rel[t_idx]

        all_median_dP.append(np.median(dP))
        all_p95_dP.append(np.percentile(dP, 95))
        all_max_dP.append(np.max(dP))

        # Top row: relative cost histogram (shared x and y axes).
        # First panel carries legend labels for shared top-row legend; rest
        # plot identically without labels.
        ax = axes_top[col]
        hist_label = r"rooms per bin of $\delta$" if col == 0 else None
        med_label = r"median $\delta$" if col == 0 else None
        ax.hist(delta * 100, bins=shared_bins, color=WONG["blue"], alpha=0.6,
                edgecolor="none", label=hist_label)
        med_pct = np.median(delta) * 100
        ax.axvline(med_pct, ls="--", color=WONG["red"], lw=1.2,
                   label=med_label)
        ax.set_title(f"$T={t_disp}$")
        ax.set_xlim(0, shared_x_hi)
        ax.set_ylim(0, shared_y_hi)
        if col == 0:
            ax.set_ylabel("Rooms")
        else:
            ax.tick_params(labelleft=False)
        if col == n_T // 2:
            ax.set_xlabel(r"Relative cost $\delta$ (%)")

    # Bottom row: DP vs T, with a one-column legend on its right
    ax_bot = fig.add_subplot(gs[1, :])
    ax_bot.plot(T_DISPLAY, all_median_dP, "o-", color=WONG["blue"],
                lw=1.8, ms=5, label=r"median $\Delta P$", zorder=3)
    ax_bot.plot(T_DISPLAY, all_p95_dP, "^--", color=WONG["orange"],
                lw=1.2, ms=4, alpha=0.7,
                label=r"95th percentile $\Delta P$", zorder=2)
    ax_bot.plot(T_DISPLAY, all_max_dP, "s:", color=WONG["red"],
                lw=1.0, ms=3, alpha=0.5,
                label=r"worst room $\Delta P$", zorder=1)
    ax_bot.axhline(ABS_COST_THRESHOLD, ls=":", color=WONG["black"], lw=1.0,
                   alpha=0.6, label="absolute-cost threshold (1.1 pp)")

    ax_bot.set_xlabel("$T$ (snapshots)")
    ax_bot.set_ylabel(r"Absolute cost $\Delta P$")
    ax_bot.set_xscale("log")

    # Top row keeps the full width. The legend is measured first, then the
    # bottom panel is shrunk from the right to make room for it.
    handles, labels = _collect_handles(fig, axes=[axes_top[0], ax_bot])
    leg = fig.legend(handles, labels, ncol=1, loc="center left", frameon=False,
                     borderaxespad=0.0, handlelength=2.2,
                     bbox_to_anchor=(0.5, 0.5), bbox_transform=fig.transFigure)
    fig.canvas.draw()
    leg_w = (leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width
             / fig.get_figwidth())
    LEFT, RIGHT, TOP, BOTTOM, HSPACE = 0.1, 0.99, 0.93, 0.11, 0.42
    gs.update(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM, hspace=HSPACE)
    pos = ax_bot.get_position()
    bot_right = RIGHT - leg_w - 0.03
    ax_bot.set_position([pos.x0, pos.y0, bot_right - pos.x0, pos.height])
    leg.set_bbox_to_anchor((bot_right + 0.03, pos.y0 + pos.height / 2),
                           transform=fig.transFigure)
    save_figure(fig, "fig16_cost_distribution", output_dir)
    return fig


if __name__ == "__main__":
    cli_wrapper(generate)
