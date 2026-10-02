#!/usr/bin/env python
"""
Step 16: Rectangular control — NNSD + K_total sweep.

1. NNSD: Nearest Neighbor Spacing Distribution for rectangular (analytical
   Dirichlet) vs generic convex (FEM) eigenvalues. KS test against Poisson
   and GOE distributions.

2. K_total sweep: P(p) ratio as function of K_total for 4 rectangular rooms.

Output:
    data/experiments/rectangular_nnsd.npz
    data/experiments/rectangular_ktotal_sweep.npz

Usage:
    python scripts/rectangular_nnsd.py
    python scripts/rectangular_nnsd.py --config configs/default.yaml
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)
from rectangular_control import rect_eigenpairs, simulate_room, P_GRID, LAMBDA_GRID, GAMMA, C_SOUND, K_TRUNC, M_USE
from src.analysis.landscape import sweep_p_for_room
from src.utils.config import load_config
from src.utils.io import get_val_rooms, load_room_auto, get_split_params

ROOMS = [
    ("3x6_rect", 3.0, 6.0),
    ("2x8_long", 2.0, 8.0),
    ("4x4_square", 4.0, 4.0),
    ("3x5_rect", 3.0, 5.0),
]


def unfolded_spacings(eigenvalues):
    """Compute unfolded nearest-neighbor spacings via Weyl's law."""
    ev = np.sort(eigenvalues)
    ev = ev[ev > 0]
    N_cumul = np.arange(1, len(ev) + 1)
    coeffs = np.polyfit(ev, N_cumul, 3)
    N_smooth = np.polyval(coeffs, ev)
    spacings = np.diff(N_smooth)
    spacings = spacings / np.mean(spacings)
    return spacings


def poisson_cdf(s):
    return 1.0 - np.exp(-s)


def goe_cdf(s):
    return 1.0 - np.exp(-np.pi * s**2 / 4.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-convex", type=int, default=50)
    parser.add_argument("--sweep-only", action="store_true",
                        help="Skip Part 1 (NNSD) and rerun only the K_total sweep")
    args = parser.parse_args()

    cfg = load_config(args.config)
    modal_root = cfg["paths"].get("modal_root",
                                   os.path.join(os.path.dirname(os.path.dirname(
                                       os.path.abspath(__file__))), "data", "modal"))
    blacklist = set(cfg["blacklist"]["scene_ids"])

    out_dir = os.path.join("data", "experiments")
    os.makedirs(out_dir, exist_ok=True)

    # ─── Part 1: NNSD ─────────────────────────────────────────────────────
    if not args.sweep_only:
        _part1_nnsd(args, cfg, modal_root, blacklist, out_dir)
    _part2_sweep(out_dir)


def _part1_nnsd(args, cfg, modal_root, blacklist, out_dir):
    print("=== Part 1: NNSD computation ===")

    rect_spacings = {}
    rect_ks_poisson = {}
    rect_ks_goe = {}

    for name, Lx, Ly in ROOMS:
        eigenvalues, _, _ = rect_eigenpairs(Lx, Ly, K_total=500)
        spacings = unfolded_spacings(eigenvalues)
        rect_spacings[name] = spacings

        ks_p = stats.kstest(spacings, poisson_cdf)
        ks_g = stats.kstest(spacings, goe_cdf)
        rect_ks_poisson[name] = (ks_p.statistic, ks_p.pvalue)
        rect_ks_goe[name] = (ks_g.statistic, ks_g.pvalue)

        print(f"  {name}: n_spacings={len(spacings)}")
        print(f"    KS vs Poisson: D={ks_p.statistic:.4f}, p={ks_p.pvalue:.4f}")
        print(f"    KS vs GOE:     D={ks_g.statistic:.4f}, p={ks_g.pvalue:.4f}")

    # Generic convex rooms
    print("\n  Loading generic convex eigenvalues...")
    val_start, val_end = get_split_params(cfg)
    val_rooms = get_val_rooms(modal_root, blacklist,
                              val_start=val_start, val_end=val_end)
    n_convex = min(args.n_convex, len(val_rooms))

    convex_spacings_all = []
    convex_ks_poisson = []
    convex_ks_goe = []

    for sid in val_rooms[:n_convex]:
        room = load_room_auto(sid, modal_root=modal_root, k_trunc=500)
        ev = room["eigenvalues"]
        if len(ev) < 20:
            continue
        spacings = unfolded_spacings(ev)
        convex_spacings_all.append(spacings)

        ks_p = stats.kstest(spacings, poisson_cdf)
        ks_g = stats.kstest(spacings, goe_cdf)
        convex_ks_poisson.append((ks_p.statistic, ks_p.pvalue))
        convex_ks_goe.append((ks_g.statistic, ks_g.pvalue))

    convex_pooled = np.concatenate(convex_spacings_all)
    ks_p_pooled = stats.kstest(convex_pooled, poisson_cdf)
    ks_g_pooled = stats.kstest(convex_pooled, goe_cdf)

    print(f"\n  Generic convex ({n_convex} rooms):")
    print(f"    Pooled: n={len(convex_pooled)}")
    print(f"    KS vs Poisson: D={ks_p_pooled.statistic:.4f}, p={ks_p_pooled.pvalue:.6f}")
    print(f"    KS vs GOE:     D={ks_g_pooled.statistic:.4f}, p={ks_g_pooled.pvalue:.6f}")

    ks_p_stats = np.array(convex_ks_poisson)
    ks_g_stats = np.array(convex_ks_goe)
    print(f"    Per-room GOE: median D={np.median(ks_g_stats[:,0]):.4f}, "
          f"median p={np.median(ks_g_stats[:,1]):.4f}")

    # Save NNSD
    nnsd_path = os.path.join(out_dir, "rectangular_nnsd.npz")
    save_dict = {
        "convex_pooled_spacings": convex_pooled,
        "convex_ks_poisson_pooled": np.array([ks_p_pooled.statistic, ks_p_pooled.pvalue]),
        "convex_ks_goe_pooled": np.array([ks_g_pooled.statistic, ks_g_pooled.pvalue]),
        "convex_ks_poisson_per_room": ks_p_stats,
        "convex_ks_goe_per_room": ks_g_stats,
        "n_convex_rooms": n_convex,
    }
    for name in rect_spacings:
        save_dict[f"rect_{name}_spacings"] = rect_spacings[name]
        save_dict[f"rect_{name}_ks_poisson"] = np.array(rect_ks_poisson[name])
        save_dict[f"rect_{name}_ks_goe"] = np.array(rect_ks_goe[name])
    np.savez(nnsd_path, **save_dict)
    print(f"\n  Saved: {nnsd_path}")


def _part2_sweep(out_dir):
    # ─── Part 2: K_total sweep ────────────────────────────────────────────
    print("\n=== Part 2: K_total sweep ===")

    # 287 is the full K_total of the 3x6 room quoted in the paper.
    K_total_values = sorted(set(range(50, 301, 25)) | {287})
    T_RAW_SWEEP = 100

    sweep_results = {}
    for name, Lx, Ly in ROOMS:
        print(f"\n  {name}:")
        ratios = []
        ratios_full = []
        p_stars = []
        P_stars = []
        for k_total in K_total_values:
            rng = np.random.default_rng(42)
            room = simulate_room(Lx, Ly, rng, K_total=k_total)
            P_oracle = sweep_p_for_room(
                room, np.array([T_RAW_SWEEP]), P_GRID, LAMBDA_GRID,
                M_values=[M_USE], mic_subsets_4=None,
                is_wave=True, gamma=GAMMA, c=C_SOUND,
            )
            curve = P_oracle[0, 0, :]
            p_star = P_GRID[np.argmin(curve)]
            P_star = curve.min()
            # Flatness ratio over p in [0, 3], the paper's definition and the
            # range of the generic-convex reference in fig_berry_weyl.
            p_mask = P_GRID <= 3.0 + 1e-9
            ratio = curve[p_mask].max() / curve[p_mask].min()
            ratios.append(ratio)
            ratios_full.append(curve.max() / curve.min())
            p_stars.append(p_star)
            P_stars.append(P_star)
            print(f"    K_total={k_total:3d}: ratio={ratio:.2f}x")

        sweep_results[name] = {
            "ratios": np.array(ratios),
            "ratios_full_grid": np.array(ratios_full),
            "p_stars": np.array(p_stars),
            "P_stars": np.array(P_stars),
        }

    ktotal_path = os.path.join(out_dir, "rectangular_ktotal_sweep.npz")
    save_dict2 = {"K_total_values": np.array(K_total_values), "T_raw": T_RAW_SWEEP}
    for name in sweep_results:
        for key, val in sweep_results[name].items():
            save_dict2[f"{name}_{key}"] = val
    np.savez(ktotal_path, **save_dict2)
    print(f"\n  Saved: {ktotal_path}")


if __name__ == "__main__":
    main()
