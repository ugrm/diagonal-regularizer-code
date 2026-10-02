"""
Figures 09-12: Heat Pipeline Diagnostics.

Paper reference: Appendix E (fig10, fig11; fig09 and fig12 are not in the paper)
Data: legacy modal dataset (src.utils.paths.MODAL), room scene_00840
      (eigenvalues, mic matrix, trajectories). The heat amplitudes are analytic
      (src.physics.temporal), so fig10/fig11 need no FDTD microphone signals;
      fig12 is drawn only when scene_00840/y_mics_eta1.npy is present.

Output:
    fig09_heat_basis_verification — Heat vs acoustic temporal basis columns
    fig10_heat_mode_survival      — Mode-by-mode signal survival in heat
    fig11_heat_amplitude_spectrum  — Actual heat amplitudes + power-law fits
    fig12_heat_per_mode_P          — Per-mode P(p) curves, heat vs acoustic
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset import load_room_modal
from src.data.features import pad_room_to_K, SKIP_FIRST, SKIP_LAST
from src.physics.temporal import (
    build_wave_temporal_matrix, build_heat_temporal_matrix, generate_heat_signals,
)
from src.utils.paths import MODAL, check_paper_dataset
from src.visualization.style import (
    setup_style, save_figure, FULL_WIDTH, SUPP_WIDTH, WONG,
    enable_all_spines, add_grid, PDE_COLORS, cli_wrapper, legend_right,
    legend_bottom,
)

ROOM_SID = "scene_00840"
K = 50  # K_TRUNC
SEED = 42
# Observation times (s) of figs 10, 11 and 17, fixed in physical time rather than
# by snapshot index so that the figures do not depend on the dataset's snapshot
# interval. The heat model is analytic, a_k(t) = a_k(0) exp(-lambda_k t) (kappa = 1).
T_OBS = (0.050, 0.250, 0.495)
T_ACOUSTIC = 0.500  # time of the acoustic damping envelope in fig10


def _get_modal_imports():
    """Repository implementations of the heat model used by the panels."""
    return {
        "build_wave_temporal_matrix": build_wave_temporal_matrix,
        "build_heat_temporal_matrix": build_heat_temporal_matrix,
        "generate_heat_signals": generate_heat_signals,
        "pad_room_to_K": pad_room_to_K,
        "SEED": SEED,
        "SKIP_FIRST": SKIP_FIRST,
        "SKIP_LAST": SKIP_LAST,
    }


def load_diag_room(modal_root=MODAL):
    """Legacy scene_00840 padded to K modes; microphone signals only if present."""
    modal_root = str(modal_root)
    check_paper_dataset(modal_root)
    rdir = os.path.join(modal_root, ROOM_SID)
    if os.path.exists(os.path.join(rdir, "y_mics_eta1.npy")):
        room = load_room_modal(ROOM_SID, modal_root, layout="legacy")
    else:
        eig = np.load(os.path.join(rdir, "eigenpairs.npz"))
        traj = np.load(os.path.join(rdir, "modal_trajectories.npz"))
        K_total = traj["a"].shape[0]
        K_use = min(K_total, K)
        dt_sim, dt_snap = float(traj["dt_sim"]), float(traj["dt_snap"])
        room = {
            "scene_id": ROOM_SID,
            "eigenvalues": eig["eigenvalues"].astype(np.float64)[:K_use],
            "frequencies": eig["frequencies"].astype(np.float64)[:K_use],
            "a": traj["a"].astype(np.float64)[:K_use],
            "Phi": np.load(os.path.join(rdir, "measurement_matrix.npy"))[:8, :K_use].astype(np.float64),
            "gamma_room": float(traj["gamma_room"]),
            "dt_sim": dt_sim, "dt_snap": dt_snap,
            "steps_per_snap": int(round(dt_snap / dt_sim)),
            "c": float(eig["c"]), "K": K_use, "K_total": K_total,
            "n_snaps": traj["a"].shape[1],
        }
    return pad_room_to_K(room, K)


def heat_amplitudes(room, seed, t):
    """Heat modal amplitudes a_k(t) of one realisation at physical time t (s)."""
    _, a_snaps, _ = generate_heat_signals(room, seed)
    return a_snaps[:, 0] * np.exp(-room["eigenvalues"] * t)


# ═══════════════════════════════════════════════════════════════════════
# FIG 09: Basis Verification
# ═══════════════════════════════════════════════════════════════════════
def _generate_fig09(room, funcs, output_dir):
    """Temporal basis functions: heat (exponential) vs acoustic (sinusoidal)."""
    ev = room["eigenvalues"]
    dt = room["dt_sim"]
    Phi = room["Phi"]
    T_raw = 100

    A_heat = funcs["build_heat_temporal_matrix"](Phi, ev, dt, T_raw)
    A_wave = funcs["build_wave_temporal_matrix"](
        Phi, ev, dt, T_raw, room["gamma_room"], room["c"])

    fig, axes = plt.subplots(2, 4, figsize=(SUPP_WIDTH, 4.0))
    modes_to_plot = [0, 4, 24, 49]
    tau_ms = (np.arange(T_raw) - (T_raw - 1)) * dt * 1000

    for i, k in enumerate(modes_to_plot):
        # Heat basis
        ax = axes[0, i]
        enable_all_spines(ax)
        ax.plot(tau_ms, A_heat[:T_raw, k], color=PDE_COLORS["heat"], lw=1.2)
        ax.set_title(f"$k={k}$, $\\lambda={ev[k]:.0f}$", fontsize=8)
        if i == 0:
            ax.set_ylabel("Heat basis", fontsize=8)
        ax.axhline(0, color="gray", lw=0.5)
        ax.tick_params(labelsize=6)

        # Acoustic basis
        ax = axes[1, i]
        enable_all_spines(ax)
        ax.plot(tau_ms, A_wave[:T_raw, 2*k], color=PDE_COLORS["acoustic"],
                lw=1, label="cos")
        ax.plot(tau_ms, A_wave[:T_raw, 2*k+1], color=PDE_COLORS["acoustic"],
                lw=1, ls="--", label="sin")
        if i == 0:
            ax.set_ylabel("Acoustic basis", fontsize=8)
            ax.legend(fontsize=5, loc="upper left")
        ax.set_xlabel(r"$\tau$ (ms)", fontsize=7)
        ax.axhline(0, color="gray", lw=0.5)
        ax.tick_params(labelsize=6)

    fig.suptitle(f"Temporal Basis Functions ({ROOM_SID}, $T=100$)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "fig09_heat_basis_verification", output_dir)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# FIG 10: Mode Survival
# ═══════════════════════════════════════════════════════════════════════
def _generate_fig10(room, funcs, output_dir):
    """Mode-by-mode signal survival: heat decays mode-dependently."""
    ev = room["eigenvalues"]
    gamma = room["gamma_room"]

    # Three well-separated physical times (T_OBS), shared with fig11 and fig17.
    snap_times = {rf"$t = {t * 1000:.0f}$ ms": t for t in T_OBS}

    fig, ax = plt.subplots(1, 1, figsize=(FULL_WIDTH, 2.5))
    enable_all_spines(ax)
    k_range = np.arange(K)

    colors = [WONG["blue"], WONG["orange"], WONG["red"]]
    markers = ["o", "s", "^"]
    for (label, t_phys), color, marker in zip(snap_times.items(), colors, markers):
        survival = np.exp(-2 * ev * t_phys)
        ax.plot(k_range, survival, color=color, lw=1.2,
                marker=marker, markersize=3.5, markevery=4,
                alpha=0.85, label=f"heat, {label}")

    # Acoustic envelope: one damping rate shared by all modes. The label
    # names the mechanism rather than a rate, because the code's gamma is
    # the energy rate and the paper's equations write the amplitude rate.
    t_mid = T_ACOUSTIC
    acoustic_surv = np.exp(-gamma * t_mid) * np.ones(K)
    ax.plot(k_range, acoustic_surv, "k--", lw=1.5,
            label=f"acoustic, uniform damping, $t = {t_mid*1000:.0f}$ ms")

    ax.set_xlabel("Mode index $k$")
    ax.set_ylabel(r"Energy survival $e^{-2\lambda_k t}$")
    ax.set_title(f"Room {ROOM_SID.split('_')[1]}")
    ax.set_yscale("log")
    ax.set_ylim(1e-25, 2)
    add_grid(ax)

    legend_right(fig, axes=[ax])
    save_figure(fig, "fig10_heat_mode_survival", output_dir)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# FIG 11: Amplitude Spectrum + Fits
# ═══════════════════════════════════════════════════════════════════════
def _generate_fig11(room, funcs, output_dir):
    """Heat amplitude spectrum: power-law vs exp x power-law fits."""
    ev = room["eigenvalues"]
    room_idx = int(ROOM_SID.split("_")[1])
    SEED = funcs["SEED"]
    a_acoustic = room["a"]

    n_seeds = 20

    fig, axes = plt.subplots(1, len(T_OBS), figsize=(FULL_WIDTH, 2.8))

    for idx, t_phys in enumerate(T_OBS):
        ax = axes[idx]
        enable_all_spines(ax)

        # Average over seeds
        a2_avg = np.zeros(K)
        for seed_off in range(n_seeds):
            a2_avg += heat_amplitudes(room, SEED + room_idx * 10 + 1 + seed_off, t_phys)**2
        a2_avg /= n_seeds

        valid = (a2_avg > 0) & (ev > 0)
        log_ev = np.log(ev[valid])
        log_a2 = np.log(a2_avg[valid])

        # Fit A: pure power law
        coeffs_A = np.polyfit(log_ev, log_a2, 1)
        s_A = -coeffs_A[0]
        pred_A = coeffs_A[0] * log_ev + coeffs_A[1]
        ss_res_A = np.sum((log_a2 - pred_A)**2)
        ss_tot = np.sum((log_a2 - log_a2.mean())**2)
        r2_A = 1 - ss_res_A / max(ss_tot, 1e-30)

        # Fit B: exp x power law
        X_B = np.column_stack([log_ev, ev[valid], np.ones(valid.sum())])
        coeffs_B = np.linalg.lstsq(X_B, log_a2, rcond=None)[0]
        s_B, c_B = -coeffs_B[0], -coeffs_B[1]
        pred_B = X_B @ coeffs_B
        ss_res_B = np.sum((log_a2 - pred_B)**2)
        r2_B = 1 - ss_res_B / max(ss_tot, 1e-30)

        # Plot data
        ax.scatter(ev[valid], a2_avg[valid], s=12, alpha=0.7,
                   color=WONG["blue"], label="heat data",
                   zorder=3)
        a2_acou_var = np.var(a_acoustic, axis=1)
        ax.scatter(ev[valid], a2_acou_var[valid], s=8, alpha=0.5,
                   color=WONG["orange"], marker="s",
                   label="acoustic data", zorder=2)

        # Fit curves
        ev_fit = np.linspace(ev[valid].min(), ev[valid].max(), 200)
        log_ev_fit = np.log(ev_fit)
        fit_A = np.exp(coeffs_A[0] * log_ev_fit + coeffs_A[1])
        fit_B = np.exp(coeffs_B[0] * log_ev_fit + coeffs_B[1] * ev_fit + coeffs_B[2])

        ax.plot(ev_fit, fit_A, "--", color=WONG["red"], lw=1.5,
                label="power-law fit")
        ax.plot(ev_fit, fit_B, "-", color=WONG["green"], lw=1.5,
                label=r"power law $\times$ exponential fit")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda_k$")
        if idx == 0:
            ax.set_ylabel(r"$E[a_k^2]$")
        ax.set_title(f"$t = {t_phys*1000:.0f}$ ms")
        ax.set_xticks([3, 10, 50])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_minor_formatter(plt.NullFormatter())

    legend_bottom(fig, axes=[axes[0]])
    save_figure(fig, "fig11_heat_amplitude_spectrum", output_dir)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# FIG 12: Per-Mode P(p) Curves
# ═══════════════════════════════════════════════════════════════════════
def _generate_fig12(room, funcs, output_dir):
    """Per-mode P(p) for heat vs acoustic: different modes benefit from different p."""
    ev = room["eigenvalues"]
    dt = room["dt_sim"]
    sps = room["steps_per_snap"]
    Phi = room["Phi"]
    room_idx = int(ROOM_SID.split("_")[1])
    SEED = funcs["SEED"]
    SKIP_FIRST = funcs["SKIP_FIRST"]
    SKIP_LAST = funcs["SKIP_LAST"]

    p_values = np.arange(0, 4.05, 0.2)
    lambda_grid = np.array([1e-6, 1e-4, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e6, 1e8, 1e10])
    T_raw = 10
    modes_to_plot = [0, 9, 24, 39, 49]
    n_snaps = room["n_snaps"]

    # Heat data
    Y_heat, a_tgt_heat, _ = funcs["generate_heat_signals"](
        room, SEED + room_idx * 10 + 1)
    T_sig_heat = Y_heat.shape[1]
    A_heat = funcs["build_heat_temporal_matrix"](Phi, ev, dt, T_raw)

    valid_heat = []
    for si in range(SKIP_FIRST, n_snaps - SKIP_LAST):
        sim_step = si * sps
        end = sim_step + 1
        start = end - T_raw
        if start >= 0 and end <= T_sig_heat:
            valid_heat.append(si)

    targets_heat = a_tgt_heat[:, valid_heat].T
    N_win = len(valid_heat)
    Y_flat_heat = np.empty((N_win, 8 * T_raw))
    for i, si in enumerate(valid_heat):
        sim_step = si * sps
        end = sim_step + 1
        start = end - T_raw
        Y_flat_heat[i] = Y_heat[:, start:end].reshape(-1)

    # Acoustic data
    a_acoustic = room["a"]
    y_click = room["y_click"]
    T_sig_wave = y_click.shape[1]
    A_wave = funcs["build_wave_temporal_matrix"](
        Phi, ev, dt, T_raw, room["gamma_room"], room["c"])

    valid_wave = []
    for si in range(SKIP_FIRST, n_snaps - SKIP_LAST):
        sim_step = si * sps
        end = sim_step + 1
        start = end - T_raw
        if start >= 0 and end <= T_sig_wave:
            valid_wave.append(si)

    targets_wave = a_acoustic[:, valid_wave].T
    N_win_w = len(valid_wave)
    Y_flat_wave = np.empty((N_win_w, 8 * T_raw))
    for i, si in enumerate(valid_wave):
        sim_step = si * sps
        end = sim_step + 1
        start = end - T_raw
        Y_flat_wave[i] = y_click[:, start:end].reshape(-1)

    def per_mode_P(A, Y_flat, targets, p_vals, lam_grid, is_heat=False):
        n_p = len(p_vals)
        K_est = A.shape[1] if is_heat else A.shape[1] // 2
        P_per_mode = np.full((n_p, K_est), np.nan)
        ATA = A.T @ A
        ATY = A.T @ Y_flat.T

        for p_idx, p in enumerate(p_vals):
            if p == 0:
                gamma_diag = np.ones(A.shape[1])
            elif is_heat:
                gamma_diag = np.power(np.maximum(ev[:K_est], 1e-10), p)
            else:
                gk = np.power(np.maximum(ev[:K_est], 1e-10), p)
                gamma_diag = np.empty(2 * K_est)
                gamma_diag[0::2] = gk
                gamma_diag[1::2] = gk

            best_P = np.full(K_est, np.inf)
            for lam in lam_grid:
                reg = ATA + lam * np.diag(gamma_diag)
                try:
                    X_r = np.linalg.solve(reg, ATY)
                except np.linalg.LinAlgError:
                    continue
                preds = X_r.T if is_heat else X_r[0::2, :].T
                for k in range(K_est):
                    var_k = np.var(targets[:, k])
                    if var_k < 1e-20:
                        continue
                    mse_k = np.mean((preds[:, k] - targets[:, k])**2)
                    P_k = mse_k / var_k
                    if P_k < best_P[k]:
                        best_P[k] = P_k
            P_per_mode[p_idx] = best_P
        return P_per_mode

    P_heat = per_mode_P(A_heat, Y_flat_heat, targets_heat, p_values,
                        lambda_grid, is_heat=True)
    P_wave = per_mode_P(A_wave, Y_flat_wave, targets_wave, p_values,
                        lambda_grid, is_heat=False)

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8))
    mode_colors = [WONG["blue"], WONG["orange"], WONG["green"],
                   WONG["red"], WONG["purple"]]

    for panel, (ax, P_arr, title) in enumerate(zip(
            axes, [P_heat, P_wave], ["Heat ($T=10$)", "Acoustic ($T=10$)"])):
        enable_all_spines(ax)
        for j, k in enumerate(modes_to_plot):
            ax.plot(p_values, P_arr[:, k], "-o", markersize=2, lw=1.2,
                    color=mode_colors[j],
                    label=f"$k={k}$ ($\\lambda={ev[k]:.0f}$)")
        ax.set_xlabel("$p$")
        ax.set_ylabel("$P_k$ (per-mode, oracle $\\lambda$)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=5.5)
        ax.set_ylim(-0.05, 2.5)
        add_grid(ax)

    fig.tight_layout(w_pad=1.5)
    save_figure(fig, "fig12_heat_per_mode_P", output_dir)
    return fig


# ═══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
def generate(data_dir=None, output_dir="figures/", config=None):
    """Generate Figures 09-12: Heat Pipeline Diagnostics."""
    setup_style()

    funcs = _get_modal_imports()
    room = load_diag_room()

    fig09 = _generate_fig09(room, funcs, output_dir)
    fig10 = _generate_fig10(room, funcs, output_dir)
    fig11 = _generate_fig11(room, funcs, output_dir)
    if "y_click" in room:
        _generate_fig12(room, funcs, output_dir)
    else:
        print(f"  fig12 skipped: {ROOM_SID}/y_mics_eta1.npy not in the modal dataset")

    return fig09


if __name__ == "__main__":
    cli_wrapper(generate)
