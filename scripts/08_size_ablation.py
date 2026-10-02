#!/usr/bin/env python
"""
Step 5b: Dataset size ablation — validation-side.

Shows that the formula's verification converges quickly with sample size.
For n_val in {10, 20, 50, 100, 150, 197}: recompute key statistics.

Output: data/experiments/size_ablation/val_ablation.npz

Usage:
    python scripts/08_size_ablation.py --config configs/default.yaml
"""

import argparse
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.utils.paths import check_paper_dataset
from src.utils.io import get_val_rooms, get_split_params
from src.estimation.metrics import fit_power_law, method_B_cost

MODAL_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "modal")
LEGACY_MODAL = None
LEGACY_FDTD = None


def compute_stats_at_n(val_rooms, n_val, noise_data, p_sweep_data, berry_data):
    """Compute all key statistics using the first n_val rooms."""
    rooms = val_rooms[:n_val]

    # Find indices of these rooms in the full arrays
    all_ids = list(noise_data["room_ids"])
    indices = [all_ids.index(r) for r in rooms if r in all_ids]

    if len(indices) < 5:
        return None

    idx = np.array(indices)

    # |s| (median of per-room slopes)
    s_vals = noise_data["s_per_room"][idx]
    s_valid = s_vals[np.isfinite(s_vals)]
    s_median = float(-np.median(s_valid)) if len(s_valid) > 0 else np.nan

    # q_B, q_A — handle both legacy (q_B) and new (q_B_full) key names
    q_B_key = "q_B_full" if "q_B_full" in noise_data else "q_B"
    q_A_key = "q_A_full" if "q_A_full" in noise_data else "q_A"
    q_B = noise_data[q_B_key][idx]
    q_A = noise_data[q_A_key][idx]
    q_B_med = float(np.nanmedian(q_B))
    q_A_med = float(np.nanmedian(q_A))

    # delta_P from p-sweep (Method B at T=1 and T=1000)
    p_key = "p_values" if "p_values" in p_sweep_data else "p_grid"
    p_grid = p_sweep_data[p_key]
    P_oracle = p_sweep_data["P_oracle"]
    sweep_ids = list(p_sweep_data["room_ids"])
    sweep_idx = [sweep_ids.index(r) for r in rooms if r in sweep_ids]

    delta_P_T1 = np.nan
    delta_P_T50 = np.nan
    if len(sweep_idx) >= 5:
        si = np.array(sweep_idx)
        T_values = p_sweep_data["T_values"]
        for t_idx, T in enumerate(T_values):
            P_sub = P_oracle[si, t_idx, 1, :]  # M=8 = index 1
            if np.all(np.isnan(P_sub)):
                continue
            _, _, dp, _ = method_B_cost(P_sub, p_grid, s_median)
            if T == 1:
                delta_P_T1 = float(np.nanmedian(dp))
            elif T == 50:
                delta_P_T50 = float(np.nanmedian(dp))

    # Berry KS at this subset size
    berry_ks = np.nan
    if berry_data is not None:
        berry_ids = list(berry_data.get("room_ids", []))
        if len(berry_ids) == 0 and "K_total" in berry_data:
            # berry_qq_data.npz doesn't have per-room IDs for the pooled Csq,
            # but has ks_stat per room — use noise_data room_ids as proxy
            berry_idx = idx[idx < len(berry_data["ks_stat"])]
            if len(berry_idx) > 0:
                berry_ks = float(np.median(berry_data["ks_stat"][berry_idx]))
        else:
            bi = [berry_ids.index(r) for r in rooms if r in berry_ids]
            if len(bi) >= 5:
                berry_ks = float(np.median(berry_data["ks_stat"][np.array(bi)]))

    return {
        "n_val": n_val,
        "n_actual": len(indices),
        "median_s": s_median,
        "q_B_median": q_B_med,
        "q_A_median": q_A_med,
        "median_dP_T1": delta_P_T1,
        "median_dP_T50": delta_P_T50,
        "berry_ks": berry_ks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--modal-root", default=None, help="Override MODAL_ROOT (e.g. data/modal_v2)")
    parser.add_argument("--exp-dir", default=None, help="Override experiments dir (e.g. data/experiments_repro)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    check_paper_dataset(cfg["paths"]["modal_root"])

    global MODAL_ROOT, LEGACY_MODAL, LEGACY_FDTD
    MODAL_ROOT = args.modal_root or cfg["paths"].get("modal_root", MODAL_ROOT)
    LEGACY_MODAL = cfg["paths"].get("modal_root")
    LEGACY_FDTD = cfg["paths"].get("fdtd_dir")

    val_start, val_end = get_split_params(cfg)
    val_rooms = get_val_rooms(MODAL_ROOT, cfg["blacklist"]["scene_ids"],
                              val_start=val_start, val_end=val_end)

    exp_base = args.exp_dir or cfg["paths"]["experiments_dir"]
    out_dir = os.path.join(exp_base, "size_ablation")
    os.makedirs(out_dir, exist_ok=True)

    # Load existing experiment results (from same experiments dir)
    noise_path = os.path.join(exp_base, "noise_profile", "noise_profile_K50_M8.npz")
    p_sweep_path = os.path.join(exp_base, "p_sweep", "p_sweep_K50_M8.npz")

    if not os.path.exists(noise_path):
        print(f"ERROR: {noise_path} not found. Run 04_run_noise_profile.py first.")
        return
    if not os.path.exists(p_sweep_path):
        print(f"ERROR: {p_sweep_path} not found. Run 05_run_p_sweep.py first.")
        return

    noise_data = np.load(noise_path, allow_pickle=True)
    p_sweep_data = np.load(p_sweep_path, allow_pickle=True)

    # Load berry data for KS ablation
    berry_path = os.path.join(exp_base, "berry", "berry_qq_data.npz")
    if not os.path.exists(berry_path) and LEGACY_MODAL:
        berry_path = os.path.join(LEGACY_MODAL, "exp5_berry_check.npz")
    berry_data = None
    if os.path.exists(berry_path):
        berry_data = np.load(berry_path, allow_pickle=True)
        print(f"  Berry data loaded from {berry_path}")
    else:
        print("  WARNING: No berry data found. berry_ks will be NaN.")

    # Deterministic subsampling: first n_val in sorted order
    n_val_sizes = [10, 20, 50, 100, 150, 197]

    print("Dataset size ablation (validation-side):")
    print(f"  n_val sizes: {n_val_sizes}")
    print(f"  Total available: {len(val_rooms)} rooms\n")

    print(f"{'n_val':>5s}  {'|s|':>6s}  {'q_B':>7s}  {'q_A':>7s}  "
          f"{'ΔP T=1':>8s}  {'ΔP T=50':>8s}  {'KS':>6s}")

    results = []
    for n_val in n_val_sizes:
        row = compute_stats_at_n(val_rooms, n_val, noise_data, p_sweep_data,
                                 berry_data)
        if row is None:
            print(f"{n_val:5d}  --- insufficient data ---")
            continue
        results.append(row)
        print(f"{row['n_actual']:5d}  {row['median_s']:6.3f}  "
              f"{row['q_B_median']:7.4f}  {row['q_A_median']:7.4f}  "
              f"{row['median_dP_T1']:8.5f}  {row['median_dP_T50']:8.5f}  "
              f"{row['berry_ks']:6.4f}")

    # Save — key names match src/visualization/fig_size_ablation.py
    out_path = os.path.join(out_dir, "val_ablation.npz")
    np.savez(
        out_path,
        n_val_rooms=np.array([r["n_val"] for r in results]),
        median_s=np.array([r["median_s"] for r in results]),
        q_B_median=np.array([r["q_B_median"] for r in results]),
        q_A_median=np.array([r["q_A_median"] for r in results]),
        median_dP_T1=np.array([r["median_dP_T1"] for r in results]),
        median_dP_T50=np.array([r["median_dP_T50"] for r in results]),
        berry_ks=np.array([r["berry_ks"] for r in results]),
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
