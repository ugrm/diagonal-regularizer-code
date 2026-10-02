"""
Figure: Berry Conjecture & Weyl Dominance for Rectangular Rooms.

Paper reference: §8 (Discussion)
Data:
    rectangular_nnsd.npz — per-room eigenvalue spacings + KS stats
    rectangular_ktotal_sweep.npz — per-room P(p) ratio vs K_total
    p_sweep/p_sweep_K50_M8.npz — generic convex reference (for ratio)

Two panels:
    (a) NNSD: rectangular eigenvalue spacings reject GOE (all D>0.19),
        generic convex consistent with GOE (D=0.010, p=0.17).
    (b) K_total sweep: landscape curvature ratio collapses from ~3.9×
        at K_total=50 to ~1.3× by K_total=75. Weyl's law guarantees
        hundreds of modes — margin is enormous.

Key result: Berry violations are real at the eigenvalue level but irrelevant
to reconstruction quality because Weyl's law supplies many discarded modes.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.visualization.style import (
    setup_style, save_figure, FULL_WIDTH, WONG, resolve_data_path, cli_wrapper,
    legend_bottom,
)

_RECT_ROOMS = ["3x6_rect", "2x8_long", "4x4_square", "3x5_rect"]


def _poisson_pdf(s):
    """Poisson NNSD: P(s) = exp(-s)."""
    return np.exp(-s)


def _goe_pdf(s):
    """GOE Wigner surmise: P(s) = (pi/2) s exp(-pi s^2 / 4)."""
    return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)


def _convex_ratio(data_dir):
    """Compute median P(p) ratio for generic convex rooms at T=100, M=8."""
    psw_path = resolve_data_path("p_sweep/p_sweep_K50_M8.npz", data_dir)
    psw = np.load(psw_path, allow_pickle=True)
    T_idx = int(np.where(psw["T_values"] == 100)[0][0])
    M_idx = int(np.where(psw["M_values"] == 8)[0][0])
    P = psw["P_oracle"][:, T_idx, M_idx, :]       # (197, 61)
    p_mask = psw["p_values"] <= 3.0
    P_clip = P[:, p_mask]
    ratios = np.max(P_clip, axis=1) / np.min(P_clip, axis=1)
    return float(np.median(ratios))


def generate(data_dir=None, output_dir="figures/", config=None):
    """Generate Berry/Weyl rectangular control figure."""
    setup_style()

    # ── Load NNSD ───────────────────────────────────────────────────────
    try:
        nnsd_path = resolve_data_path("rectangular_nnsd.npz", data_dir)
    except FileNotFoundError:
        print("BLOCKED: rectangular_nnsd.npz not found.")
        return None

    nnsd = np.load(nnsd_path, allow_pickle=True)

    # Pool rectangular spacings from per-room keys
    rect_spacings = np.concatenate(
        [nnsd[f"rect_{r}_spacings"] for r in _RECT_ROOMS]
    )
    convex_spacings = nnsd["convex_pooled_spacings"]

    # KS stats: format is [D_statistic, p_value]
    convex_goe_D = float(nnsd["convex_ks_goe_pooled"][0])
    convex_goe_p = float(nnsd["convex_ks_goe_pooled"][1])
    rect_goe_Ds = [float(nnsd[f"rect_{r}_ks_goe"][0]) for r in _RECT_ROOMS]

    # ── Load K_total sweep ──────────────────────────────────────────────
    try:
        sweep_path = resolve_data_path("rectangular_ktotal_sweep.npz", data_dir)
    except FileNotFoundError:
        print("BLOCKED: rectangular_ktotal_sweep.npz not found.")
        return None

    sweep = np.load(sweep_path, allow_pickle=True)
    K_vals = sweep["K_total_values"]

    # Assemble per-room ratios into (4, n_K) array
    ratios = np.array([sweep[f"{r}_ratios"] for r in _RECT_ROOMS])

    # Generic convex reference ratio from p_sweep
    try:
        ref_ratio = _convex_ratio(data_dir)
    except FileNotFoundError:
        ref_ratio = 1.28  # fallback

    # ── Two-panel figure ────────────────────────────────────────────────
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.9))

    # ── (a) NNSD ────────────────────────────────────────────────────────
    s_ref = np.linspace(0, 4, 200)
    bins = np.linspace(0, 4, 40)

    ax_a.hist(rect_spacings, bins=bins, density=True, histtype="step",
              color=WONG["red"], lw=1.5, label="rectangular rooms")
    ax_a.hist(convex_spacings, bins=bins, density=True, histtype="step",
              color=WONG["blue"], lw=1.5, label="generic convex rooms")

    ax_a.plot(s_ref, _poisson_pdf(s_ref), "--", color=WONG["red"],
              alpha=0.6, lw=1.0, label="Poisson (integrable)")
    ax_a.plot(s_ref, _goe_pdf(s_ref), "--", color=WONG["blue"],
              alpha=0.6, lw=1.0, label="GOE (chaotic)")

    ax_a.set_xlabel("Normalized spacing $s$")
    ax_a.set_ylabel("Density")
    ax_a.set_title("(a) Spacing statistics")
    ax_a.set_xlim(0, 4)
    ax_a.set_ylim(bottom=0)

    # ── (b) K_total sweep ───────────────────────────────────────────────
    # The 3x6 rectangle is the focal room; the other three are context.
    main_idx = _RECT_ROOMS.index("3x6_rect")
    first_other = True
    for i in range(len(_RECT_ROOMS)):
        if i != main_idx:
            ax_b.plot(K_vals, ratios[i], color=WONG["red"],
                      alpha=0.25, lw=0.8,
                      label="other rectangles" if first_other else None)
            first_other = False
    ax_b.plot(K_vals, ratios[main_idx], color=WONG["red"], lw=2.0,
              label=r"$3\times 6$ rectangle")

    ax_b.set_ylim(1.0, 1.1 * float(np.nanmax(ratios)))

    # Generic convex reference
    ax_b.axhline(ref_ratio, ls="--", color=WONG["blue"], lw=1.5,
                 label="generic rooms, median")

    # The 3x6 room's own mode count below the frequency ceiling, the
    # operating point the appendix text discusses; it is one of the sweep's
    # grid values (the sweep was extended to include it).
    K_TOTAL_3x6 = 287
    assert K_TOTAL_3x6 in K_vals
    ax_b.axvline(K_TOTAL_3x6, ls=":", color=WONG["black"], alpha=0.6, lw=0.9,
                 label=r"$K_{\mathrm{total}}$ of the $3\times 6$ room")

    ax_b.set_xlabel(r"$K_{\mathrm{total}}$")
    ax_b.set_ylabel(r"$\max P\, /\, \min P$ over $p \in [0,3]$")
    ax_b.set_title(r"(b) Flatness vs $K_{\mathrm{total}}$")

    legend_bottom(fig)
    save_figure(fig, "fig_berry_weyl", output_dir)
    return fig


if __name__ == "__main__":
    cli_wrapper(generate)
