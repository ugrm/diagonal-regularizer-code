"""
Figure 24: Locatello-Style ΔP Heatmap.

Paper reference: §5 (Learned Methods)
Data: sweep_P_eval.npz

Three panels (M1, M2, M3), each with:
    Main grid (5 rows × 3 cols): ΔP vs ridge(|ŝ|=1.1)
    Annotation row: ΔP vs per-room oracle ridge (best p* per room, oracle α)

    rows = n_train ∈ {50, 100, 200, 400, 800}
    cols = T ∈ {1, 100, 1000}
    color = ΔP = P_method − P_ridge, diverging RdBu_r, centered at 0.

Grey cells for missing configs (training did not converge).
Cells with <5 seeds annotated with count.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from src.visualization.style import (
    setup_style, save_figure, FULL_WIDTH, WONG,
    METHOD_COLORS, resolve_data_path, cli_wrapper, enable_all_spines,
    legend_right,
)


# Grid axes
_MODELS = ["M1", "M2", "M3"]
_N_LIST = [50, 100, 200, 400, 800]
_T_LIST = [1, 100, 1000]
_N_ROWS = len(_N_LIST)     # 5 data rows
_TOTAL_ROWS = _N_ROWS + 1  # +1 for oracle annotation row

# Map checkpoint directory names to display names
_MODEL_MAP = {
    "m1_condnet": "M1", "m2_fixedgamma": "M2", "m3_hypernet": "M3",
}


def _annotate_cell(ax, col, row, val, n_valid, vmax, is_oracle_row=False):
    """Write the cell's value. Cells without a converged run stay grey and
    empty; the legend names that state. Seed counts below five are in the
    per-seed appendix table rather than in the cells."""
    if np.isnan(val) or n_valid == 0:
        return

    txt_color = "white" if abs(val) > 0.6 * vmax else "black"
    label = f"{val:.3f}" if abs(val) < 0.1 else f"{val:.2f}"
    ax.text(col, row, label, ha="center", va="center",
            fontsize=6, color=txt_color, fontweight="medium")


def generate(data_dir=None, output_dir="figures/", config=None):
    """Generate Figure 24: Locatello-Style Sweep ΔP Heatmap."""
    setup_style()

    # ── Load eval results ───────────────────────────────────────────────
    eval_path = resolve_data_path("sweep_P_eval.npz", data_dir)
    ev = np.load(eval_path, allow_pickle=True)
    P_end2end = ev["P_end2end"]            # (3 models, 5 n, 3 T, 5 seeds)
    P_ridge_s = ev["P_ridge_s"]            # (3 T,)
    P_oracle_baseline = ev["P_oracle_baseline"]  # (3 T,)

    # ── Compute ΔP grids ────────────────────────────────────────────────
    # Main grid: vs ridge(|ŝ|=1.1)
    delta_P = np.full((3, _N_ROWS, 3), np.nan)
    # Oracle row: vs per-room oracle ridge
    delta_P_oracle = np.full((3, 3), np.nan)
    n_seeds = np.zeros((3, _N_ROWS, 3), dtype=int)
    n_seeds_oracle = np.zeros((3, 3), dtype=int)

    for mi in range(3):
        for ni in range(_N_ROWS):
            for ti in range(3):
                vals = P_end2end[mi, ni, ti, :]
                valid = vals[~np.isnan(vals)]
                n_seeds[mi, ni, ti] = len(valid)
                if len(valid) > 0:
                    med = np.median(valid)
                    delta_P[mi, ni, ti] = med - P_ridge_s[ti]
        # Oracle annotation row: n=800 (most data, best comparison)
        ni_800 = 4
        for ti in range(3):
            vals = P_end2end[mi, ni_800, ti, :]
            valid = vals[~np.isnan(vals)]
            n_seeds_oracle[mi, ti] = len(valid)
            if len(valid) > 0:
                delta_P_oracle[mi, ti] = np.median(valid) - P_oracle_baseline[ti]

    # ── Color scale (symmetric around 0, shared across both rows) ──────
    all_vals = np.concatenate([delta_P.ravel(), delta_P_oracle.ravel()])
    vmax = np.nanmax(np.abs(all_vals))
    vmax = max(vmax, 0.05) if not np.isnan(vmax) else 0.1
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu_r

    # ── Three-panel heatmap + colorbar + key ────────────────────────────
    # Drawn at text width and included at \linewidth so the cell values are
    # legible; the right strip holds the colourbar and the grey-cell key.
    fig = plt.figure(figsize=(FULL_WIDTH, 3.0))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.05], wspace=0.18,
                  left=0.16, right=0.70, bottom=0.14, top=0.9)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    for mi, (model, ax) in enumerate(zip(_MODELS, axes)):
        enable_all_spines(ax)

        # Build combined grid: 6 rows (5 data + 1 oracle), 3 cols
        combined = np.full((_TOTAL_ROWS, 3), np.nan)
        combined[:_N_ROWS, :] = delta_P[mi]
        combined[_N_ROWS, :] = delta_P_oracle[mi]

        ax.set_facecolor("#CCCCCC")

        masked = np.ma.masked_invalid(combined)
        ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto",
                  interpolation="nearest")

        # Separator line between main grid and oracle row
        ax.axhline(_N_ROWS - 0.5, color="black", linewidth=1.5)

        # Main grid annotations
        for ni in range(_N_ROWS):
            for ti in range(3):
                _annotate_cell(ax, ti, ni, delta_P[mi, ni, ti],
                               n_seeds[mi, ni, ti], vmax)

        # Oracle row annotations
        for ti in range(3):
            _annotate_cell(ax, ti, _N_ROWS, delta_P_oracle[mi, ti],
                           n_seeds_oracle[mi, ti], vmax, is_oracle_row=True)

        # Axes
        ax.set_xticks(range(3))
        ax.set_xticklabels([str(T) for T in _T_LIST])
        ax.set_xlabel("$T$")
        color = METHOD_COLORS.get(model, WONG["black"])
        ax.set_title(model, fontsize=10, color=color, fontweight="bold")

        if mi == 0:
            ax.set_yticks(range(_TOTAL_ROWS))
            labels = [str(n) for n in _N_LIST] + ["800,\nvs oracle"]
            ax.set_yticklabels(labels)
            ax.set_ylabel("$n$ (training rooms)")
        else:
            ax.set_yticks(range(_TOTAL_ROWS))
            ax.set_yticklabels([])

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cax.set_title(r"$\Delta P$")

    # Key for the one cell state the colourbar does not cover; the caption
    # says what the black line separates.
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#CCCCCC", edgecolor="none",
              label="no converged run"),
    ]
    legend_right(fig, handles=handles, labels=[h.get_label() for h in handles],
                 right=0.70, pack=False, loc="lower left",
                 bbox_to_anchor=(0.70, 0.05))

    save_figure(fig, "fig24_sweep_heatmap", output_dir)
    return fig


if __name__ == "__main__":
    cli_wrapper(generate)
