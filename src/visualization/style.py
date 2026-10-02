"""
Global figure style for NeurIPS 2026 paper.

All figure modules import this. Call setup_style() at the top of every
generate() function, and save_figure() at the end.

Decisions:
    - NeurIPS single-column: text width = 5.5"
    - Computer Modern math (matches LaTeX body)
    - Seaborn 'white' base (gridlines added per-axis)
    - Wong (2011) colorblind-safe palette
    - Top/right spines off globally; re-enable per-figure if needed
    - Method color assignment: M2 (star) = red for visual prominence (Option B)
"""

import os
import matplotlib
import matplotlib.pyplot as plt

# ── Output format (set by cli_wrapper, read by save_figure) ──────────
OUTPUT_FORMAT = "both"   # "pdf", "png", or "both"

# ── Figure dimensions (NeurIPS single-column) ──────────────────────────

FULL_WIDTH = 5.5        # \textwidth in NeurIPS template
HALF_WIDTH = 2.75       # 0.5 * \textwidth
SUPP_WIDTH = 6.5        # Slightly wider for supplement

FIGURE_SIZES = {
    "full":         (FULL_WIDTH, 2.5),
    "full_tall":    (FULL_WIDTH, 4.2),
    "full_2x2":    (FULL_WIDTH, 4.2),
    "half":         (HALF_WIDTH, 2.5),
    "half_square":  (HALF_WIDTH, HALF_WIDTH),
    "supp_tall":    (SUPP_WIDTH, 10.8),     # 6-row diagnostic
    "supp_wide":    (SUPP_WIDTH, 3.0),
}

# ── Wong (2011) colorblind-safe palette ────────────────────────────────

WONG = {
    "blue":    "#0072B2",
    "orange":  "#E69F00",
    "green":   "#009E73",
    "red":     "#D55E00",
    "purple":  "#CC79A7",
    "cyan":    "#56B4E9",
    "yellow":  "#F0E442",
    "black":   "#000000",
}

# ── Method colors (Option B: M2 = red, star of the paper) ─────────────

METHOD_COLORS = {
    "M1":           WONG["blue"],     # CondNet (geometry-aware)
    "M2":           WONG["red"],      # FixedGamma (geometry-blind) — THE STAR
    "M3":           WONG["orange"],   # HyperNet
    "M4":           WONG["purple"],   # CUT baseline (if available)
    "Ridge_I":      WONG["cyan"],     # Ridge(Γ=I), p=0
    "Ridge_s":      WONG["green"],    # Ridge(λ^|s|) — theory-predicted
    "Ridge_2":      WONG["purple"],   # Ridge(λ²)
    "Kalman":       WONG["purple"],   # Sequential (different marker from Ridge_2)
    "theory":       WONG["black"],    # Analytical prediction
    "prior":        WONG["red"],      # Prior variance (declining)
    "trunc_noise":  WONG["blue"],     # Truncation noise (flat)
    "approach_a":   WONG["orange"],   # Total residual noise
}

# Names follow the paper: M1 and M3 share the CondNet and differ in the solver,
# M2 learns one Gamma shared by all rooms. "Ridge" in the paper means p = 0
# only; lambda^|s| is "the closed form".
METHOD_LABELS = {
    "M1": "M1: CondNet, unrolled",
    "M2": r"M2: shared $\Gamma$",
    "M3": "M3: CondNet, closed-form",
    "M4": "M4: CUT",
}

RIDGE_LABELS = {
    "Ridge_I": r"ridge ($\Gamma{=}I$)",
    "Ridge_s": r"closed form $\lambda_k^{|s|}$",
    "Ridge_2": r"$\lambda_k^{2}$",
    "oracle":  "per-room power-law oracle",
}

METHOD_MARKERS = {
    "M1": "s",      # square
    "M2": "o",      # circle — prominent
    "M3": "^",      # triangle
    "M4": "D",      # diamond
    "Ridge_I":  "v",
    "Ridge_s":  "^",
    "Ridge_2":  "d",
    "Kalman":   "o",
}

# ── PDE-specific colors ───────────────────────────────────────────────

PDE_COLORS = {
    "acoustic": WONG["blue"],
    "heat":     WONG["red"],
}

# ── Geometry group colors (Berry check, diagnostics) ──────────────────

GEOM_COLORS = {
    "triangle":  WONG["red"],
    "mid":       WONG["orange"],
    "high":      WONG["blue"],
}

# ── RC params ─────────────────────────────────────────────────────────

# Figures are drawn at the width they are included at, so these sizes are
# the sizes on the page. Nothing below 8 pt; the paper's caption font is 9 pt.
_RC_PARAMS = {
    "font.size": 9,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "standard",   # the page is exactly figsize; see save_figure
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
}


def setup_style():
    """Call at the top of every figure's generate() function."""
    try:
        import seaborn as sns
        sns.set_theme(style="white", context="paper", font_scale=1.2)
    except ImportError:
        pass  # Seaborn optional; rcParams still applied below
    plt.rcParams.update(_RC_PARAMS)


def save_figure(fig, name, output_dir="figures/"):
    """
    Save figure as PDF and/or PNG, controlled by OUTPUT_FORMAT.

    The whole figure box is saved (no tight bounding box), so the page is
    exactly `figsize` and the figure prints at its design width when it is
    included at that width. Layout must therefore keep every artist inside
    the box: tight_layout for the axes, legend_right for the legend.

    Args:
        fig: matplotlib Figure
        name: filename stem (no extension)
        output_dir: target directory
    """
    fmt = OUTPUT_FORMAT
    os.makedirs(output_dir, exist_ok=True)
    if fmt in ("pdf", "both"):
        pdf_path = os.path.join(output_dir, f"{name}.pdf")
        fig.savefig(pdf_path, format="pdf")
        print(f"  Saved: {pdf_path}")
    if fmt in ("png", "both"):
        png_path = os.path.join(output_dir, f"{name}.png")
        fig.savefig(png_path, format="png", dpi=300)
        print(f"  Saved: {png_path}")
    plt.close(fig)


def enable_all_spines(ax):
    """Re-enable top/right spines for heatmaps, log-log plots, etc."""
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)


def add_grid(ax, which="both", alpha=0.2):
    """Gridlines are off in every figure."""
    ax.grid(False)


def include_width(fraction):
    """Figure width in inches for a figure included at `fraction` of \\linewidth.

    Drawing at the include width keeps every font at its rcParams size on the
    page instead of being scaled down by \\includegraphics.
    """
    return FULL_WIDTH * fraction


def legend_right(fig, axes=None, handles=None, labels=None, right=None,
                 gap=0.012, pack=True, **kw):
    """One legend for the whole figure, outside the axes on the right.

    Every colour and line style a figure uses is explained here and nowhere
    else: no in-plot text. Handles are collected from `axes` (all axes of the
    figure by default), de-duplicated by label in order of first appearance,
    unless explicit handles/labels are passed.

    The legend is drawn first and measured, then the axes are packed into
    the space to its left, so the legend always lies inside the figure box
    and the saved figure is as wide as `figsize`. Fonts therefore print at
    their rcParams size when the figure is included at its design width,
    and the axes get every inch the legend does not need.

    right: fraction of the figure width reserved for the axes. None (the
        default) measures the legend and uses everything it does not need.
    pack: run tight_layout over the axes region. Pass False when the module
        lays its axes out by hand (colourbars, fixed GridSpec extents).
    """
    if handles is None:
        handles, labels, seen = [], [], set()
        for ax in (axes if axes is not None else fig.axes):
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l and not l.startswith("_") and l not in seen:
                    seen.add(l)
                    handles.append(h)
                    labels.append(l)
    opts = dict(loc="center left", frameon=False, borderaxespad=0.0,
                handlelength=2.2)
    opts.update(kw)
    anchor = opts.pop("bbox_to_anchor", None)
    leg = fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.5),
                     bbox_transform=fig.transFigure, **opts)
    if right is None:
        fig.canvas.draw()
        bb = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        legend_frac = bb.width / fig.get_figwidth()
        right = 1.0 - legend_frac - gap - 0.004
    if pack:
        fig.tight_layout(rect=(0.004, 0.006, right, 0.994), pad=0.6)
    if anchor is None:
        anchor = (right + gap, 0.5)
    leg.set_bbox_to_anchor(anchor, transform=fig.transFigure)
    return leg


def _collect_handles(fig, axes=None, handles=None, labels=None):
    if handles is not None:
        return list(handles), list(labels)
    handles, labels, seen = [], [], set()
    for ax in (axes if axes is not None else fig.axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l and not l.startswith("_") and l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    return handles, labels


def legend_bottom(fig, axes=None, handles=None, labels=None, ncol=None,
                  gap=0.012, pack=True, fit_width=False, **kw):
    """One legend for the whole figure, in horizontal lines under the axes.

    Starts from one line (ncol = number of entries) and adds lines until the
    legend fits the figure width; then packs the axes above it. The saved
    page is exactly figsize, as with legend_right.

    fit_width=True keeps the requested ncol and instead widens the figure
    until the legend fits, so the panels widen with it. A figure wider than
    the text width prints scaled down when included at \\linewidth.
    """
    handles, labels = _collect_handles(fig, axes, handles, labels)
    opts = dict(loc="lower center", frameon=False, borderaxespad=0.0,
                handlelength=1.8, columnspacing=1.1, handletextpad=0.5)
    opts.update(kw)
    n = max(1, len(handles))
    ncol = n if ncol is None else ncol
    fig_w, fig_h = fig.get_figwidth(), fig.get_figheight()
    while fit_width:
        leg = fig.legend(handles, labels, ncol=ncol, bbox_to_anchor=(0.5, 0.012),
                         bbox_transform=fig.transFigure, **opts)
        fig.canvas.draw()
        bb = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        if bb.width > fig_w - 0.16:
            fig_w = bb.width + 0.16
            fig.set_size_inches(fig_w, fig_h)
            fig.canvas.draw()
            bb = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        break
    while not fit_width:
        leg = fig.legend(handles, labels, ncol=ncol, bbox_to_anchor=(0.5, 0.012),
                         bbox_transform=fig.transFigure, **opts)
        fig.canvas.draw()
        bb = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        if bb.width <= fig_w - 0.08 or ncol == 1:
            break
        leg.remove()
        ncol -= 1
    bottom = bb.height / fig_h + gap + 0.012
    if pack:
        fig.tight_layout(rect=(0.004, bottom, 0.996, 0.994), pad=0.6)
    return leg


def saved_width_check(fig, name):
    """Print the tight-bbox width of `fig` in inches next to its figsize."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    print(f"  {name}: figsize {fig.get_figwidth():.2f} in, "
          f"tight bbox {bb.width:.2f} x {bb.height:.2f} in")


# ── Data path resolution ──────────────────────────────────────────────

# Subdirectories searched under --data-root (release layout)
_DATA_SUBDIRS = [
    "experiments", "modal", "diagnostics", "eigenmodes", "results",
    "supplementary", "modal_summary",
    # Experiment outputs nested by type
    "experiments/noise_profile", "experiments/p_sweep",
    "experiments/berry", "experiments/cost", "experiments/size_ablation",
    "experiments/lir",
    "experiments/anisotropy",
    "experiments/sroom_vs_pop",
]

# Fallback search paths relative to the project root
_LEGACY_DIRS = [
    "data/experiments",
    "modal_data",
    "results",
]


def resolve_data_path(filename, data_dir=None, legacy_dirs=None):
    """
    Find a data file, checking data_dir (and its subdirectories) first,
    then legacy locations relative to project root.

    Args:
        filename: e.g. "noise_profile_K50_M8.npz"
        data_dir: root data directory (--data-root value, e.g. "data/")
        legacy_dirs: override legacy search paths (default: _LEGACY_DIRS)

    Returns:
        Absolute path to the file, or raises FileNotFoundError.
    """
    if legacy_dirs is None:
        legacy_dirs = _LEGACY_DIRS

    searched = []

    # 1. Check data_dir directly, then its subdirectories
    if data_dir is not None:
        candidate = os.path.join(data_dir, filename)
        searched.append(data_dir)
        if os.path.isfile(candidate):
            return candidate
        for subdir in _DATA_SUBDIRS:
            candidate = os.path.join(data_dir, subdir, filename)
            searched.append(os.path.join(data_dir, subdir))
            if os.path.isfile(candidate):
                return candidate

    # 2. Legacy directories relative to project root
    project_root = _find_project_root()
    for d in legacy_dirs:
        candidate = os.path.join(project_root, d, filename)
        searched.append(os.path.join(project_root, d))
        if os.path.isfile(candidate):
            return candidate

    # 3. Absolute path
    if os.path.isfile(filename):
        return filename

    raise FileNotFoundError(
        f"{filename} not found in: {searched}"
    )


def _find_project_root():
    """Walk up from this file to find the project root (contains src/ or modal_data/)."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(d, "modal_data")) or os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    # Fallback: assume CWD
    return os.getcwd()


def cli_wrapper(generate_fn):
    """Shared CLI entry point for all figure modules.

    Provides --data-root, --out-dir, and --format flags.
    Usage in each module's __main__:
        from src.visualization.style import cli_wrapper
        if __name__ == "__main__":
            cli_wrapper(generate)
    """
    import argparse
    parser = argparse.ArgumentParser(
        description=getattr(generate_fn, '__doc__', None) or "Generate figure",
    )
    parser.add_argument("--data-root", default="data/",
                        help="Root data directory (default: data/)")
    parser.add_argument("--out-dir", default="figures/",
                        help="Output directory for figures (default: figures/)")
    parser.add_argument("--format", choices=["pdf", "png", "both"],
                        default="both", help="Output format (default: both)")
    args = parser.parse_args()

    global OUTPUT_FORMAT
    OUTPUT_FORMAT = args.format

    generate_fn(data_dir=args.data_root, output_dir=args.out_dir)
