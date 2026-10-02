"""
Figure 4: All Methods Converge to λ^s — Learned Gamma spectra.

Paper reference: Section 4, Figure 4 (main text)
Data: results/method{1,2,3,4}_*.npz, noise_profile_K50_M8.npz, p_sweep_K50_M8.npz

Panels:
    (a) Learned Γ_k vs k for all methods + the closed form λ^|s| (dashed)
    (b) Per-method P_modal vs T (closed form, per-room oracle, M1-M3, LIR)

M2 (geometry-blind) finds the same Γ as M1 (geometry-aware).
p̂ values are annotated in the legend/panel text.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.visualization.style import (
    setup_style, save_figure, WONG,
    METHOD_COLORS, METHOD_LABELS, METHOD_MARKERS, RIDGE_LABELS,
    resolve_data_path, cli_wrapper,
    include_width, legend_right,
)


# 10 boundary rooms with K_total <= K=50 (excluded from in-scope n=187).
BOUNDARY_ROOMS = {
    "scene_00800", "scene_00806", "scene_00810", "scene_00812", "scene_00829",
    "scene_00839", "scene_00848", "scene_00867", "scene_00981", "scene_00993",
}


def _inscope_mask(room_ids):
    """Boolean mask: True where the room is in the n=187 in-scope set."""
    return np.array([str(r) not in BOUNDARY_ROOMS for r in room_ids])


def _get_gamma_k(data, K=50, T_idx=0, inscope=True):
    """Extract gamma_k from various NPZ formats.

    For 3D arrays (rooms × modes × T_extract), select T_idx along the T axis,
    optionally filter rooms to the n=187 in-scope set, and return the
    per-mode median across rooms.
    """
    if "gamma_k" in data:
        g = np.array(data["gamma_k"])
    elif "learned_gamma" in data:
        g = np.array(data["learned_gamma"])
    else:
        return None
    if g.ndim == 3:
        if inscope and "room_ids" in data:
            keep = _inscope_mask(data["room_ids"])
            g = g[keep, :, T_idx]
        else:
            g = g[:, :, T_idx]
        g = np.nanmedian(g, axis=0)
    elif g.ndim == 2:
        if inscope and "room_ids" in data:
            keep = _inscope_mask(data["room_ids"])
            g = g[keep]
        g = np.nanmedian(g, axis=0)
    return g[:K]


def generate(data_dir=None, output_dir="figures/", config=None,
             T_idx_gamma=3, M_idx=1):
    """Generate Figure 4: Learned Methods."""
    setup_style()

    K = 50
    k_idx = np.arange(1, K + 1)

    # Load noise profile for |s| and eigenvalues
    noise = np.load(resolve_data_path("noise_profile_K50_M8.npz", data_dir), allow_pickle=True)
    eigenvalues = np.median(noise["eigenvalues"], axis=0)[:K]
    s_abs = abs(float(np.nanmedian(noise["s_per_room"])))
    gamma_analytical = eigenvalues ** s_abs
    gamma_analytical /= gamma_analytical[0]

    # Load methods via resolve_data_path
    name_map = {
        "M1": ["method1_condnet.npz", "exp4_learned_gamma.npz"],
        "M2": ["method2_fixedgamma.npz", "method2_results.npz"],
        "M3": ["method3_hypernet.npz", "method3_results.npz"],
        "M4": ["method4_cut.npz"],
    }
    method_results = {}
    for name, fnames in name_map.items():
        for fname in fnames:
            try:
                path = resolve_data_path(fname, data_dir)
                method_results[name] = dict(np.load(path, allow_pickle=True))
                break
            except FileNotFoundError:
                continue

    # Load sweep for Ridge baseline
    try:
        sweep = np.load(resolve_data_path("p_sweep_K50_M8.npz", data_dir), allow_pickle=True)
    except FileNotFoundError:
        sweep = None

    # Drawn at full text width (included at \linewidth) so fonts print at size.
    fig, axes = plt.subplots(1, 2, figsize=(include_width(1.0), 2.3),
                             gridspec_kw={"width_ratios": [1, 1.5]})
    # the same object in both panels: the closed form, black and dashed
    THEORY_LABEL = RIDGE_LABELS["Ridge_s"]

    # ── Panel (a): Learned Γ_k vs k (at T=1000 by default) ──────────
    ax = axes[0]
    ax.plot(k_idx, gamma_analytical, ls="--", color=WONG["black"],
            lw=1.8, label=THEORY_LABEL, zorder=5)

    for name in ["M1", "M2", "M3", "M4"]:
        if name not in method_results:
            continue
        gamma_k = _get_gamma_k(method_results[name], K, T_idx=T_idx_gamma)
        if gamma_k is None:
            continue
        gamma_k = gamma_k / gamma_k[0]
        # M2 and M3 carry the callout's claims; equal weight. M1/M4 stay thin.
        lw = 2.0 if name in ("M2", "M3") else 1.2
        alpha = 1.0 if name in ("M2", "M3") else 0.8
        zorder = 4 if name in ("M2", "M3") else 3
        ax.plot(k_idx, gamma_k, color=METHOD_COLORS[name], lw=lw,
                alpha=alpha, label=METHOD_LABELS[name], zorder=zorder)

    ax.set_xlabel("Mode index $k$")
    ax.set_ylabel(r"$\Gamma_k / \Gamma_1$")
    ax.set_title("(a) Learned spectrum")
    ax.set_yscale("log")

    # ── Panel (b): P_modal vs T ──────────────────────────────────────
    ax = axes[1]

    # Ridge(|ŝ|) baseline + per-room oracle ceiling, on all 197 validation
    # rooms, the cohort of the M1-M3 and LIR markers below.
    if sweep is not None and "P_oracle" in sweep:
        P_oracle = sweep["P_oracle"]
        T_grid = sweep["T_values"]
        p_values = sweep["p_values"]
        ps_idx = int(np.argmin(np.abs(p_values - s_abs)))
        sweep_keep = slice(None)
        P_ridge_s = np.nanmedian(P_oracle[sweep_keep, :, M_idx, ps_idx], axis=0)
        ax.plot(T_grid, P_ridge_s, color=WONG["black"], lw=1.8,
                ls="--", label=THEORY_LABEL, zorder=2)

        # Per-room oracle: best p* per room, then median across rooms.
        P_oracle_best = np.nanmin(P_oracle[sweep_keep, :, M_idx, :], axis=2)
        P_oracle_med = np.nanmedian(P_oracle_best, axis=0)
        ax.plot(T_grid, P_oracle_med, color=WONG["black"], lw=1.4,
                ls=":", label=RIDGE_LABELS["oracle"], zorder=1, alpha=0.7)

    # M1/M2/M3 from sweep eval (n=800), medians over the 197 validation rooms.
    try:
        eval_path = resolve_data_path("sweep_P_eval.npz", data_dir)
        ev = np.load(eval_path, allow_pickle=True)
        P_end2end = ev["P_end2end"]  # (3 models, 5 n, 3 T, 5 seeds)
        T_eval = ev["T_values"]      # [1, 100, 1000]
        model_names = ["M1", "M2", "M3"]
        ni_800 = 4  # n=800 index

        # The three medians coincide with the closed form, so each is a
        # translucent circle in its method's colour, nudged apart in T so
        # the pile reads as a cluster; seed IQRs are smaller than the marker.
        offsets = {"M1": 0.84, "M2": 1.0, "M3": 1.19}
        for mi, name in enumerate(model_names):
            seeds = P_end2end[mi, ni_800, :, :]  # (3 T, 5 seeds)
            med = np.nanmedian(seeds, axis=1)
            ax.plot(T_eval * offsets[name], med, "o", ms=8.5,
                    color=METHOD_COLORS[name], alpha=0.5,
                    markeredgecolor="none", ls="none",
                    label=METHOD_LABELS[name], zorder=3)
    except FileNotFoundError:
        pass

    # LIR (L=10) markers — purple diamond, mean ± std across 5 seeds of the
    # median over the 197 validation rooms.
    try:
        lir_path = resolve_data_path("lir_summary.npz", data_dir)
        lir = np.load(lir_path)
        L_values = lir["L_values"]    # (4,)
        T_lir = lir["T_values"]       # (3,) = [1, 100, 1000]
        li = int(np.where(L_values == 10)[0][0])
        lir_per_seed = lir["P_median"][li, :, :]  # (T, seeds)
        lir_mean = np.mean(lir_per_seed, axis=1)
        lir_std = np.std(lir_per_seed, axis=1)
        ax.errorbar(T_lir * 1.12, lir_mean, yerr=lir_std,
                    color=WONG["purple"], fmt="D", ms=7,
                    markeredgecolor="black", markeredgewidth=1.0,
                    capsize=2.5, capthick=1.0, lw=1.2,
                    label="LIR (non-diagonal)", zorder=5)
    except (FileNotFoundError, KeyError, IndexError):
        pass  # graceful fallback — data not yet available

    ax.set_xlabel("$T$ (snapshots)")
    ax.set_ylabel(r"$P$ (median over rooms)")
    ax.set_title("(b) Error vs $T$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.2, 0.5, 1.0])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.yaxis.set_minor_formatter(plt.NullFormatter())


    # One entry per method combining its line (a) and its marker (b).
    from matplotlib.legend_handler import HandlerTuple
    ha, la = axes[0].get_legend_handles_labels()
    hb, lb = axes[1].get_legend_handles_labels()
    order = ([THEORY_LABEL] + [METHOD_LABELS[m] for m in ("M1", "M2", "M3")]
             + [RIDGE_LABELS["oracle"], "LIR (non-diagonal)"])
    handles, labels = [], []
    for lab in order:
        hs = tuple(h for h, l in list(zip(ha, la)) + list(zip(hb, lb)) if l == lab)
        if hs:
            handles.append(hs if len(hs) > 1 else hs[0])
            labels.append(lab)
    legend_right(fig, handles=handles, labels=labels,
                 handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})
    save_figure(fig, "fig04_learned_spectra", output_dir)
    return fig


if __name__ == "__main__":
    cli_wrapper(generate)
